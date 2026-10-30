"""Small auditable PPO self-play baseline; renderer and Torch training stay independent."""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from driversi.agents.interfaces.config import build_agent, load_inference
from driversi.contracts.learning import OBSERVATION_VERSION
from driversi.data.scenes.scene_store import source_inventory
from driversi.evaluation.search.teacher import evaluate as evaluate
from driversi.training.sampling.batch import learning_mask
from driversi.training.legacy.checkpoint import restore_checkpoint as restore_checkpoint
from driversi.training.legacy.checkpoint import save_checkpoint as save_checkpoint
from driversi.training.sampling.collector import Collector as Collector
from driversi.training.sampling.collector import rollout_nbytes
from driversi.training.legacy.config import add_training_arguments, resolve_training
from driversi.training.sampling.environment import validate_splits


def objective_terms(logits, value, actions, old_log_probs, advantages, returns):
    """Per-sample objective shared by single-process and distributed PPO."""
    dist = Categorical(logits=logits)
    log_ratio = dist.log_prob(actions) - old_log_probs
    ratio = log_ratio.exp()
    policy = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    value_loss = 0.5 * (value - returns).square()
    return policy, value_loss, dist.entropy(), ratio - 1 - log_ratio


def update(model, optimizer, rollout, *, epochs=4, minibatch=512, device="cpu"):
    valid = learning_mask(rollout)
    if valid.sum() < 2:
        raise ValueError("PPO needs at least two valid transitions")
    tensors = {key: torch.as_tensor(rollout[key][valid], device=device) for key in
               ("observations", "actions", "log_probs", "values", "advantages", "returns")}
    adv = tensors["advantages"]
    tensors["advantages"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    totals = []
    model.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(adv), device=device).split(minibatch):
            logits, value = model(tensors["observations"][indices])
            terms = objective_terms(logits, value, tensors["actions"][indices],
                                    tensors["log_probs"][indices], tensors["advantages"][indices],
                                    tensors["returns"][indices])
            policy_loss, value_loss, entropy, approx_kl = (term.mean() for term in terms)
            loss = policy_loss + .5 * value_loss - .01 * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            totals.append([loss.item(), policy_loss.item(), value_loss.item(), entropy.item(),
                           approx_kl.item()])
    return dict(zip(("loss", "policy_loss", "value_loss", "entropy", "approx_kl"),
                    np.mean(totals, axis=0).tolist()))


def train(args):
    from driversi.training.legacy.config import resolve_distribution
    if resolve_distribution(args)["enabled"]:
        from driversi.training.legacy.distributed import train_distributed
        return train_distributed(args)
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("torchrun requires explicit --distributed or distribution.enabled")
    args = resolve_training(args)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory, including for resumed experiments")
    if args.budget < 1 or args.rollout_steps < 1 or args.worlds < 1:
        raise ValueError("Positive training budgets required")
    required_bytes = rollout_nbytes(args.rollout_steps, args.worlds, opponents=args.opponent_pool is not None)
    if args.adaptive_opponents is not None:
        required_bytes += args.rollout_steps * args.worlds * 8
    if required_bytes > args.sampling["max_rollout_mib"] * 1024**2:
        raise MemoryError(f"Configured rollout requires {required_bytes} raw array bytes")
    suite = json.loads(args.suite.read_text())
    train_paths = [row["path"] for row in suite["train"]]
    eval_paths = [row["path"] for row in suite["evaluation"]]
    validate_splits(train_paths, eval_paths)
    opponents = None
    if args.opponent_pool is not None:
        from driversi.training.self_play.opponents import FrozenOpponentPool
        opponents = FrozenOpponentPool(args.opponent_pool)
        validate_splits(opponents.training_paths, eval_paths)
    curriculum_paths = []
    if args.adaptive_opponents is not None:
        from driversi.training.self_play.adaptive_opponents import refresh_for_training, validate_curriculum
        curriculum_paths = [row["path"] for row in suite.get("curriculum", [])]
        validate_curriculum(curriculum_paths, train_paths, eval_paths, opponents,
                            args.learner_count, args.adaptive_opponents)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = build_agent(args.agent).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optimization["learning_rate"], eps=1e-5)
    collector = Collector(train_paths, args.worlds, args.seed, opponent_pool=opponents,
                          learner_count=args.learner_count, **args.sampling)
    config = {"suite": suite, "seed": args.seed, "worlds": args.worlds,
              "rollout_steps": args.rollout_steps, "device": args.device,
              "observation_version": OBSERVATION_VERSION, "gamma": .99, "gae_lambda": .95,
              "ppo_clip": .2, "lr": args.optimization["learning_rate"],
              "epochs": args.optimization["epochs"], "minibatch_size": args.optimization["minibatch_size"],
              "agent": args.agent, "sampling": args.sampling, "recording": args.recording, "progress_weight": .02,
              "finite_horizon_no_bootstrap": True, "rendering": False}
    if opponents is not None:
        config["opponents"] = opponents.identity
        config["learner_count"] = args.learner_count
    if args.adaptive_opponents is not None:
        config["adaptive_opponents"] = args.adaptive_opponents
    refreshes = []
    start = time.perf_counter()
    total, index, optimizer_updates = 0, 0, 0
    try:
        output.mkdir(parents=True)
        (output / "rollouts").mkdir()
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        sources = source_inventory(collector.paths, collector.progress_weight)
        (output / "scene-inventory.json").write_text(json.dumps(sources), encoding="utf-8")
        # Initial always means seeded pre-training weights, including resumed runs.
        model.export(output / "initial.npz", f"ppo-seed{args.seed}-update{index}")
        baseline = {"coast": evaluate(None, eval_paths, training_paths=train_paths),
                    "initial": evaluate(load_inference(output / "initial.npz", args.agent), eval_paths, training_paths=train_paths)}
        if args.resume:
            index, total, optimizer_updates = restore_checkpoint(
                args.resume, model, optimizer, collector, config, return_counters=True)
            model.export(output / "resume-start.npz", f"ppo-seed{args.seed}-update{index}")
            baseline["resume_start"] = evaluate(load_inference(output / "resume-start.npz", args.agent), eval_paths, training_paths=train_paths)
        if opponents is not None:
            from driversi.evaluation.benchmarks.matchup import evaluate_matchups
            checkpoint = output / ("resume-start.npz" if args.resume else "initial.npz")
            baseline["matchups"] = evaluate_matchups(load_inference(checkpoint, args.agent),
                eval_paths, opponents, training_paths=train_paths, seed=args.seed,
                learner_count=args.learner_count)
        (output / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        if args.adaptive_opponents is not None:
            initial_refresh = refresh_for_training(model, args.agent, collector, curriculum_paths,
                eval_paths, output, args.adaptive_opponents, optimizer_updates,
                f"ppo-seed{args.seed}-update{index}")
            if initial_refresh is not None:
                refreshes.append(initial_refresh)
            # Persist even the initial revision if the first sampling batch fails.
            save_checkpoint(output / "checkpoint.pt", model, optimizer, collector, index, total, config,
                            optimizer_updates=optimizer_updates)
        while total < args.budget:
            begin = time.perf_counter()
            policy_version = f"ppo-seed{args.seed}-update{index}"
            rollout = collector.collect(model, args.rollout_steps, args.device, policy_version)
            count = int(rollout["valid"].sum())
            learner_valid = learning_mask(rollout)
            learner_count = int(learner_valid.sum())
            if count == 0:
                raise ValueError("No valid physical transitions")
            metadata = {"schema_version": 1, "kind": "privileged-ppo-rollout", "policy_version": policy_version,
                        "observation_version": OBSERVATION_VERSION, "agent": args.agent,
                        "joint_state_version": "puffer-state13-motion3-v1", "dt_seconds": .1,
                        "source_manifest": "scene-inventory.json",
                        "replay_motion_is_measured": False, "population_mode": "fixed",
                        "valid_transitions": count, "approved_teacher": False,
                        "eligibility": {"ppo_engineering": True, "expert_supervision": False,
                                        "photoreal_training": False}}
            if opponents is not None:
                metadata.update(behavior_policies=collector.behavior_policies(policy_version),
                                learner_transitions=learner_count, learner_count=args.learner_count,
                                sampling_mode=("adaptive_safety_pfsp" if collector.adaptive_sampling is not None
                                               else "frozen_history_weighted" if isinstance(opponents.identity["sampling"], dict)
                                               else "frozen_history_uniform"),
                                learning_mask="valid & learner_mask & on_policy_eligible")
            if collector.adaptive_sampling is not None:
                metadata["opponent_sampling_history"] = collector.adaptive_sampling["history"]
            # Preserve the exact frozen behavior weights with each on-policy batch.
            write_start = time.perf_counter()
            if args.recording == "full":
                model.export(output / "rollouts" / f"policy-{index:05d}.npz", policy_version)
                np.savez_compressed(output / "rollouts" / f"batch-{index:05d}.npz",
                                    metadata=json.dumps(metadata), **rollout)
            write_seconds = time.perf_counter() - write_start
            update_start = time.perf_counter()
            if learner_count >= 2:
                stats = update(model, optimizer, rollout, device=args.device,
                               epochs=args.optimization["epochs"], minibatch=args.optimization["minibatch_size"])
                optimizer_updates += 1
            else:
                stats = {"update_skipped": "fewer_than_two_learner_transitions"}
            update_seconds = time.perf_counter() - update_start
            total += count
            index += 1
            refresh = None
            if args.adaptive_opponents is not None:
                refresh = refresh_for_training(model, args.agent, collector, curriculum_paths,
                    eval_paths, output, args.adaptive_opponents, optimizer_updates,
                    f"ppo-seed{args.seed}-update{index}")
                if refresh is not None:
                    refreshes.append(refresh)
            checkpoint_start = time.perf_counter()
            save_checkpoint(output / "checkpoint.pt", model, optimizer, collector, index, total, config,
                            optimizer_updates=optimizer_updates)
            stats.update(update=optimizer_updates, batch=index, optimizer_updated=learner_count >= 2,
                         valid_transitions=total, batch_valid=count, batch_learner=learner_count,
                         learner_mean_reward=float(rollout["rewards"][learner_valid].mean()) if learner_count else None,
                         mean_reward=float(rollout["rewards"][rollout["valid"]].mean()),
                         sampling=collector.last_metrics, rollout_write_seconds=write_seconds,
                         update_seconds=update_seconds,
                         checkpoint_seconds=time.perf_counter() - checkpoint_start,
                         collect_record_update_checkpoint_seconds=time.perf_counter() - begin)
            if args.adaptive_opponents is not None:
                stats["curriculum_refresh"] = refresh
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(stats, allow_nan=False) + "\n")
            print(json.dumps(stats), flush=True)
            del rollout
        model.export(output / "candidate.npz", f"ppo-seed{args.seed}-update{index}")
        evaluation = evaluate(load_inference(output / "candidate.npz", args.agent), eval_paths, training_paths=train_paths)
        report = {"status": "complete", "valid_transitions": total, "updates": optimizer_updates, "batches": index,
                  "seconds_including_eval_recording": time.perf_counter() - start,
                  "approved_teacher": False, "official_test": False, "evaluation": evaluation,
                  "baseline": baseline, "limits": suite["limits"],
                  "resume_from": str(args.resume) if args.resume else None,
                  "recording": args.recording, "rollout_data_available": args.recording == "full"}
        if opponents is not None:
            report["matchups"] = evaluate_matchups(load_inference(output / "candidate.npz", args.agent),
                eval_paths, opponents, training_paths=train_paths, seed=args.seed,
                learner_count=args.learner_count)
        if args.adaptive_opponents is not None:
            report["adaptive_opponents"] = {"state": collector.adaptive_sampling,
                                            "refreshes_this_run": refreshes,
                                            "curriculum_is_training_feedback": True}
        report["seconds_including_eval_recording"] = time.perf_counter() - start
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        collector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
