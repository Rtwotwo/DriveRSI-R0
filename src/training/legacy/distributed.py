"""Single-host DDP PPO: shared model, independent worlds, complete valid batches."""
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from driversi.agents.interfaces.config import build_agent, load_inference
from driversi.contracts.learning import OBSERVATION_SIZE, OBSERVATION_VERSION
from driversi.data.scenes.scene_store import source_inventory
from driversi.evaluation.search.teacher import evaluate
from driversi.training.sampling.batch import learning_mask
from driversi.training.sampling.collector import Collector, rollout_nbytes
from driversi.training.legacy.config import resolve_training
from driversi.training.legacy.distributed_checkpoint import restore_distributed, save_distributed
from driversi.training.legacy.distributed_update import update_distributed
from driversi.training.legacy.distribution import initialize, partition, prepare_ddp
from driversi.training.sampling.environment import validate_splits


def train_distributed(raw_args):
    args = resolve_training(raw_args)
    if args.device != "cuda":
        raise ValueError("distributed production training requires device=cuda")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    context = initialize(args.distribution["timeout_seconds"])
    collector = None
    opponents = None
    pool_path = getattr(args, "opponent_pool", None)
    learner_roles = getattr(args, "learner_count", 1)
    start = time.perf_counter()
    try:
        def preflight():
            if args.worlds < context.size:
                raise ValueError("global worlds must be >= distributed ranks")
            required = rollout_nbytes(args.rollout_steps, args.worlds, opponents=pool_path is not None)
            if required > args.sampling["max_rollout_mib"] * 1024 ** 2:
                raise ValueError("global rollout exceeds max_rollout_mib budget")
            suite = json.loads(args.suite.read_text())
            training = [row["path"] for row in suite["train"]]
            evaluation = [row["path"] for row in suite["evaluation"]]
            validate_splits(training, evaluation)
            if not training or not evaluation:
                raise ValueError("nonempty training and held-out evaluation are required")
            return suite, training, evaluation
        suite, training, evaluation_paths = context.stage("preflight", preflight)
        if pool_path is not None:
            def opponent_preflight():
                from driversi.training.self_play.opponents import FrozenOpponentPool
                pool = FrozenOpponentPool(pool_path)
                validate_splits(pool.training_paths, evaluation_paths)
                return pool
            opponents = context.stage("opponent preflight", opponent_preflight)
        worlds = partition(args.worlds, context.size)
        workers = partition(args.sampling["workers"], context.size)
        # Global byte bound is checked above; local caps are rounded up by one MiB.
        local_sampling = dict(args.sampling, workers=workers[context.rank],
                              max_rollout_mib=max(1, (
                                  rollout_nbytes(args.rollout_steps, worlds[context.rank], opponents=pool_path is not None)
                                  + 1024**2-1)//1024**2))
        config = dict(suite=suite, seed=args.seed, worlds=args.worlds,
                      rollout_steps=args.rollout_steps, agent=args.agent,
                      sampling=args.sampling, optimization=args.optimization,
                      recording=args.recording, observation_version=OBSERVATION_VERSION,
                      world_size=context.size, rank_worlds=worlds, rank_workers=workers,
                      distribution=args.distribution, device="cuda",
                      minibatch_scope="per_rank", scene_sampling="shared_pool_with_replacement",
                      gamma=0.99, gae_lambda=0.95, ppo_clip=0.2,
                      finite_horizon_no_bootstrap=True, rendering=False,
                      progress_weight=0.02, ddp_bucket_warmup=3,
                      runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   nccl=list(torch.cuda.nccl.version())))
        if opponents is not None:
            config["opponents"] = opponents.identity
            config["learner_count"] = learner_roles
        configs = [None] * context.size
        invocation = dict(config=config, budget=args.budget,
                          output=str(args.output_dir.resolve()),
                          resume=str(args.resume.resolve()) if args.resume else None)
        dist.all_gather_object(configs, invocation)
        if any(value != invocation for value in configs):
            raise ValueError("rank configurations differ")
        output = args.output_dir.resolve()

        def create_output():
            output.mkdir(parents=True, exist_ok=False)
            (output/"rollouts").mkdir()
            (output/"config.json").write_text(json.dumps(config, indent=2))
        context.root("create output", create_output)
        rank_output = output/"rollouts"/f"rank-{context.rank:03d}"
        context.stage("rank output", lambda: rank_output.mkdir())
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        model = context.stage("model", lambda: build_agent(args.agent).to(context.device))
        # BatchNorm buffers would depend on dummy rows and violate empty-rank semantics.
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for module in model.modules()):
            raise ValueError("distributed PPO currently requires models without BatchNorm")
        wrapped = prepare_ddp(model, context.device, OBSERVATION_SIZE)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optimization["learning_rate"], eps=1e-5)
        rank_seed = (args.seed + 104729 * context.rank) % (2**63-1)
        torch.manual_seed(rank_seed)

        def create_collector():
            nonlocal collector
            collector = Collector(training, worlds[context.rank], rank_seed, opponent_pool=opponents,
                                  learner_count=learner_roles, **local_sampling)
        context.stage("collector", create_collector)
        index, total, optimizer_updates = 0, 0, 0
        if args.resume:
            index, total, optimizer_updates = restore_distributed(args.resume, context, model, optimizer,
                                              collector, config, return_counters=True)
        baseline = None

        def initial_evaluation(progress):
            nonlocal baseline
            sources = source_inventory(training, collector.progress_weight)
            (output/"scene-inventory.json").write_text(json.dumps(sources))
            model.export(output/"initial.npz", f"ddp-seed{args.seed}-update{index}")
            progress()
            baseline = dict(coast=evaluate(None, evaluation_paths, training_paths=training,
                                           on_episode=progress),
                            initial=evaluate(load_inference(output/"initial.npz", args.agent),
                                             evaluation_paths, training_paths=training,
                                             on_episode=progress))
            if opponents is not None:
                from driversi.evaluation.benchmarks.matchup import evaluate_matchups
                baseline["matchups"] = evaluate_matchups(load_inference(output/"initial.npz", args.agent),
                    evaluation_paths, opponents, training_paths=training, seed=args.seed,
                    learner_count=learner_roles, on_episode=progress)
            (output/"baseline.json").write_text(json.dumps(baseline, indent=2))
        context.root_with_progress("initial evaluation", initial_evaluation)
        while total < args.budget:
            begin = time.perf_counter()
            version = f"ddp-seed{args.seed}-update{index}"
            rollout = context.stage("collection", lambda: collector.collect(
                model, args.rollout_steps, str(context.device), version))
            count = int(rollout["valid"].sum())
            global_count = int(context.sum([count]).item())
            local_learner = int(learning_mask(rollout).sum())
            global_learner = int(context.sum([local_learner]).item())
            if global_count < 2:
                raise ValueError("global rollout has insufficient valid transitions")
            sample_seconds = time.perf_counter() - begin
            write_begin = time.perf_counter()

            def write_rollout():
                if args.recording == "full":
                    metadata = dict(schema_version=1, kind="privileged-ppo-rollout",
                                    rank=context.rank, world_size=context.size,
                                    policy_version=version, observation_version=OBSERVATION_VERSION,
                                    agent=args.agent, joint_state_version="puffer-state13-motion3-v1",
                                    dt_seconds=0.1, source_manifest="../../scene-inventory.json",
                                    replay_motion_is_measured=False, population_mode="fixed",
                                    valid_transitions=count, approved_teacher=False,
                                    eligibility=dict(ppo_engineering=True, expert_supervision=False,
                                                     photoreal_training=False))
                    if opponents is not None:
                        metadata.update(behavior_policies=collector.behavior_policies(version),
                                        learner_transitions=local_learner, learner_count=learner_roles,
                                        sampling_mode="frozen_history_uniform",
                                        learning_mask="valid & learner_mask & on_policy_eligible")
                    np.savez_compressed(rank_output/f"batch-{index:05d}.npz",
                                        metadata=json.dumps(metadata), **rollout)
            context.stage("rollout write", write_rollout)
            if args.recording == "full":
                context.root("behavior export", lambda: model.export(
                    output/"rollouts"/f"policy-{index:05d}.npz", version))
            write_seconds = time.perf_counter()-write_begin
            update_begin = time.perf_counter()
            if global_learner >= 2:
                stats = update_distributed(context, wrapped, optimizer, rollout,
                                           epochs=args.optimization["epochs"],
                                           minibatch=args.optimization["minibatch_size"])
                optimizer_updates += 1
            else:
                stats = {"update_skipped": "fewer_than_two_learner_transitions"}
            update_seconds = time.perf_counter()-update_begin
            total += global_count
            index += 1
            reward = context.sum([float(rollout["rewards"][rollout["valid"]].sum())]).item()
            checkpoint_begin = time.perf_counter()
            save_distributed(output/"checkpoint.pt", context, model, optimizer,
                             collector, index, total, config, optimizer_updates=optimizer_updates)
            checkpoint_seconds = time.perf_counter()-checkpoint_begin
            # Use maximum wall time across ranks, not a sum or rank-zero-only timing.
            timings = torch.tensor([sample_seconds, write_seconds, update_seconds,
                                    checkpoint_seconds, time.perf_counter()-begin],
                                   dtype=torch.float64, device=context.device)
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
            sampling = [None]*context.size
            dist.all_gather_object(sampling, collector.last_metrics)
            stats.update(update=optimizer_updates, batch=index, optimizer_updated=global_learner >= 2,
                         batch_physical=global_count, batch_learner=global_learner, valid_transitions=total,
                         mean_reward=reward/global_count, sampling_by_rank=sampling,
                         collect_seconds=timings[0].item(), rollout_write_seconds=timings[1].item(),
                         update_seconds=timings[2].item(), checkpoint_seconds=timings[3].item(),
                         collect_record_update_checkpoint_seconds=timings[4].item())

            def log():
                with (output/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(stats, allow_nan=False)+"\n")
                print(json.dumps({k:v for k,v in stats.items() if k!="sampling_by_rank"}), flush=True)
            context.root("metrics", log)
        context.stage("collector shutdown", collector.close)
        collector = None
        def finish(progress):
            model.export(output/"candidate.npz", f"ddp-seed{args.seed}-update{index}")
            progress()
            evaluation = evaluate(load_inference(output/"candidate.npz", args.agent),
                                  evaluation_paths, training_paths=training, on_episode=progress)
            report = dict(status="complete", valid_transitions=total, updates=optimizer_updates, batches=index,
                          seconds_including_eval_recording=time.perf_counter()-start,
                          world_size=context.size, approved_teacher=False, official_test=False,
                          evaluation=evaluation, baseline=baseline, limits=suite["limits"],
                          resume_from=str(args.resume) if args.resume else None,
                          recording=args.recording, rollout_data_available=args.recording=="full",
                          distributed=dict(global_budget=args.budget,
                                           budget_overshoot=total-args.budget,
                                           minibatch_scope="per_rank", dropped_valid=0))
            if opponents is not None:
                from driversi.evaluation.benchmarks.matchup import evaluate_matchups
                report["matchups"] = evaluate_matchups(load_inference(output/"candidate.npz", args.agent),
                    evaluation_paths, opponents, training_paths=training, seed=args.seed,
                    learner_count=learner_roles, on_episode=progress)
            report["seconds_including_eval_recording"] = time.perf_counter()-start
            (output/"report.json").write_text(json.dumps(report, indent=2))
        context.root_with_progress("final evaluation and export", finish)
    finally:
        try:
            if collector is not None:
                collector.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
