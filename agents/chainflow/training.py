"""Official ChainFlow stage1 math under the existing bounded training lifecycle.

Only process, sample order, budget and recovery plumbing live here. The released
model, feature/target builders, loss and get_optimizers are called unchanged.
"""

import gc
import math
import os
import sys
from pathlib import Path

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi.training.orchestration.state import deterministic, fingerprint, restore_rng, rng_state
from driversi_agents.chainflow.benchmark import evaluator_files
from driversi_agents.chainflow.data import (
    NativeScenes,
    scoring_cache_paths,
    validate_manifest,
    verify_records,
)

RECOVERY_SCHEMA_VERSION = 2


def _training_recipe(cfg, settings, batch_size):
    """Resolve explicit incremental schedule controls without changing native math.

    The declared scheduler dataset size is a horizon hyperparameter. It is not
    the measured number of samples in this run, and must be fixed across arms.
    No override keeps the released reference horizon and historical FP32 adapter.
    """
    overrides = settings.get("training_recipe", {})
    allowed = {"scheduler_dataset_size", "scheduler_num_epochs", "precision"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("Unsupported ChainFlow training recipe control")
    if overrides.get("precision", "32-true") != "32-true":
        raise ValueError("ChainFlow lifecycle currently supports explicit FP32 only")
    reference_size = int(cfg.scheduler_args.dataset_size)
    reference_epochs = int(cfg.scheduler_args.num_epochs)
    size = overrides.get("scheduler_dataset_size", reference_size)
    epochs = overrides.get("scheduler_num_epochs", reference_epochs)
    for name, value in (("scheduler_dataset_size", size), ("scheduler_num_epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError("Positive integer required for " + name)
    total_steps = math.ceil(size / batch_size) * epochs
    warmup_steps = int(total_steps * 0.1)
    if warmup_steps < 1:
        raise ValueError("Native schedule requires at least one warmup step")
    cfg.scheduler_args.dataset_size = size
    cfg.scheduler_args.num_epochs = epochs
    lr_args = dict(cfg.lr_args)
    base_lr = float(lr_args["base_lr"]) * math.sqrt(batch_size / int(lr_args["base_batch_size"]))
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "initialization": "parent_model_weights_new_native_optimizer",
        "optimizer": {"native_method": "get_optimizers", "arguments": lr_args, "scaled_base_lr": base_lr},
        "scheduler": {
            "interval": "step", "frequency": 1,
            "declared_dataset_size": size, "declared_num_epochs": epochs,
            "total_updates": total_steps, "warmup_updates": warmup_steps,
            "initial_lr_factor": 1e-6,
            "size_semantics": "fixed_schedule_hyperparameter_not_measured_sample_count",
        },
        "reference_schedule": {"dataset_size": reference_size, "num_epochs": reference_epochs},
        "incremental_schedule_override": bool(size != reference_size or epochs != reference_epochs),
        "precision": "32-true", "native_default_precision": "16-mixed",
        "precision_matches_native_default": False,
        "batch_size_per_device": batch_size, "world_size": 1,
        "gradient_accumulation": 1, "gradient_clip_val": 0.0,
        "sample_order": "deterministic_sequential_cycle",
        "epoch_boundary": "sequential_cycle_can_cross_manifest_boundary_not_native_epoch_loader",
        "native_full_recipe_reproduction": False,
    }


def _validate_recovery(state, context, recipe, rows):
    if state.get("recovery_schema_version") != RECOVERY_SCHEMA_VERSION:
        raise ValueError("Unsupported legacy ChainFlow recovery; restart from model weights")
    if state.get("context") != context or state.get("effective_training_recipe") != recipe:
        raise ValueError("Native recovery belongs to another data/recipe context")
    updates, cursor = state.get("total_updates"), state.get("cursor")
    batch_size = recipe["batch_size_per_device"]
    if type(updates) is not int or updates < 0 or type(cursor) is not int or cursor != updates * batch_size:
        raise ValueError("Native recovery update count and sample cursor differ")
    trace = state.get("sample_trace", [])
    if len(trace) != updates:
        raise ValueError("Native recovery sample exposure length differs")
    indices = _sample_order(rows, updates * batch_size, recipe)
    for step, item in enumerate(trace):
        expected_ids = [rows[indices[step * batch_size + j]]["sample_id"] for j in range(batch_size)]
        if item.get("optimizer_step") != step + 1 or item.get("sample_ids") != expected_ids or item.get("native_scheduler_step_before") != step:
            raise ValueError("Native recovery sample order or scheduler cursor differs")
    if state.get("scheduler", {}).get("last_epoch") != updates:
        raise ValueError("Native recovery scheduler and optimizer cursor differ")


def _sample_order(rows, count, recipe):
    if recipe["sample_order"] == "full_navtrain_seeded_mix":
        from driversi_agents.chainflow.benchmark import sample_indices
        return sample_indices(rows, count, recipe["sampling_seed"], recipe["synthetic_every"])
    return [i % len(rows) for i in range(count)]


def _parameter_fingerprint(agent):
    return fingerprint({name: p for name, p in agent.named_parameters() if p.requires_grad})


def _move(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_move(v, device) for v in value)
    return value


def _verify_inputs(rows, budget_file, limit):
    """Reserve deterministic reference replay before parsing native cache objects."""
    steps = 40 * sum(row.get("kind") == "synthetic_navtrain" for row in rows)
    if steps > limit:
        raise ValueError("Synthetic cache validation exceeds scoring budget")
    write_json(budget_file, {"reserved_agent_steps": steps, "cache_validation_agent_steps": steps, "proposal_count": 0})
    actual = verify_records(rows)
    if actual != steps:
        raise ValueError("Native cache verification accounting differs")
    return steps


def _score_function(agent, rows, budget_file, limit, *, validation_steps=0):
    import numpy as np
    from navsim.agents.chainflow_vla.utils.pdm_score import get_scores

    original, synthetic = scoring_cache_paths(rows)
    agent.train_metric_cache_paths = original
    agent.test_metric_cache_paths = original
    agent.train_metric_cache_paths_synthetic = synthetic
    agent.test_metric_cache_paths_synthetic = synthetic
    if not 0 <= validation_steps <= limit:
        raise ValueError("Invalid preflight scoring budget")
    spent = {"reserved_agent_steps": validation_steps, "cache_validation_agent_steps": validation_steps, "proposal_count": 0}
    write_json(budget_file, spent)

    def score(args):
        count = sum(len(a["poses"]) for a in args)
        reserve = count * 40
        if spent["reserved_agent_steps"] + reserve > limit:
            raise ValueError("Native scorer budget exhausted before scoring")
        spent["reserved_agent_steps"] += reserve
        spent["proposal_count"] += count
        write_json(budget_file, spent)
        result = get_scores(args)
        if len(result) != len(args):
            raise ValueError("Incomplete native scoring result")
        for row in result:
            if not np.isfinite(row[0]).all() or ((row[0] < 0) | (row[0] > 1)).any():
                raise ValueError("Native scoring returned invalid range; fallback rejected")
        return result

    agent.get_scores = score
    return spent


def _agent(plan, checkpoint):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    root = project_root()
    repo = root / plan["model_config"]["repository"]
    cfg = OmegaConf.load(
        repo / "navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml"
    )
    cfg.batch_size = plan["resolved"]["request"]["pilot"]["batch_size"]
    cfg.num_gpus = 1
    cfg.enable_ray = False
    cfg.progress_bar = False
    cfg.config.image_backbone.model_weights = str(
        path_at(root, plan["model_config"]["weights"]["backbone"])
    )
    cfg.config.lidar_backbone.model_weights = cfg.config.image_backbone.model_weights
    cfg.checkpoint_path = str(checkpoint)
    cfg.checkpoint_strict_load = True
    OmegaConf.save(cfg, Path(plan["output"]) / "native-config.yaml")
    agent = instantiate(cfg)
    agent.initialize()
    return agent.to("cuda"), cfg


def _evaluate(plan, previous, dataset, *, validation_steps=0):
    import lzma
    import pickle
    from dataclasses import asdict

    import numpy as np
    import torch
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from driversi_agents.chainflow.stage1 import trajectory_from_input

    out = Path(plan["output"])
    req = plan["resolved"]["request"]
    rows = dataset["splits"]["development"]
    evaluation_steps = len(rows) * 2 * 80
    if evaluation_steps + validation_steps > req["budget"]["max_scoring_agent_steps"]:
        raise ValueError("Independent evaluation scoring budget too small")
    context = plan.get("evaluation_context")
    if context:
        expected = context["value"]
        if expected["dataset_manifest_sha256"] != digest(plan["data"]["manifest"]):
            raise ValueError("Independent evaluator dataset differs")
        expected_files = (
            evaluator_files(dataset)
            if dataset["schema_version"] in (2, 3) else
            {r["metric_cache"]: dataset["files"][r["metric_cache"]] for r in rows}
        )
        if expected["files"] != expected_files:
            raise ValueError("Independent evaluator did not seal exact heldout caches")
    spent_steps = validation_steps
    metrics, per_sample = {}, {}
    names = {
        key: key
        for key in (
            "score",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
        )
    }
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    parent = path_at(project_root(), plan["model_config"]["weights"]["checkpoint"])
    for arm, checkpoint in (("parent", parent), ("candidate", previous["checkpoint"])):
        deterministic(req["pilot"]["seed"])
        agent, _ = _agent(plan, checkpoint)
        scenes = NativeScenes(rows, agent, native_original_windows=dataset["schema_version"] in (2, 3))
        per_sample[arm] = []
        for i, row in enumerate(rows):
            trajectory = trajectory_from_input(agent, scenes.scene(i).get_agent_input())
            with lzma.open(row["metric_cache"], "rb") as stream:
                cache = pickle.load(stream)
            spent_steps += 80
            write_json(out / "scoring-budget.json", {"reserved_agent_steps": spent_steps, "cache_validation_agent_steps": validation_steps, "evaluation_agent_steps": spent_steps - validation_steps})
            score = pdm_score(
                cache,
                Trajectory(trajectory.poses),
                sampling,
                PDMSimulator(sampling),
                PDMScorer(sampling),
            )
            values = {names[k]: float(v) for k, v in asdict(score).items()}
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()):
                raise ValueError("Nonfinite or invalid independent evaluation score")
            per_sample[arm].append(
                {
                    "sample_id": row["sample_id"],
                    "source_log": row["source_log"],
                    "metrics": values,
                    "poses": trajectory.poses.tolist(),
                }
            )
            with (out / (arm + "-samples.jsonl")).open("a") as stream:
                import json
                stream.write(json.dumps(per_sample[arm][-1], allow_nan=False) + "\n")
            if (i + 1) % 10 == 0:
                write_json(out / "evaluation-progress.json", {"arm": arm, "completed": i + 1, "expected": len(rows)})
                print("evaluation", arm, i + 1, "/", len(rows), flush=True)
        write_json(out / (arm + "-samples.json"), {"rows": per_sample[arm]})
        write_json(out / (arm + "-source-time-gaps.json"), scenes.original_time_gaps)
        metrics[arm] = {
            k: float(np.mean([r["metrics"][k] for r in per_sample[arm]])) for k in names.values()
        }
        del agent, scenes
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "status": "completed",
        "reason": None,
        "metrics": metrics,
        "directions": {k: 1 for k in names.values()},
        "primary_metric": "score",
        "safety_higher": [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "time_to_collision_within_bound",
            "comfort",
        ],
        "metric_scope": (
            "official_navtest_v1_locked_subset_adaptive_pilot_not_full_benchmark"
            if dataset["schema_version"] == 3 else
            "official_navtest_v1_full_coverage_adaptive_evaluation"
            if dataset["schema_version"] == 2
            else "native_navsim_pdm_current_run_log_heldout_not_formal_benchmark"
        ),
        "evaluated_sample_ids": [r["sample_id"] for r in rows],
        "paired_samples": [
            {
                "sample_id": a["sample_id"],
                "source_log": a["source_log"],
                "parent": a["metrics"],
                "candidate": b["metrics"],
            }
            for a, b in zip(per_sample["parent"], per_sample["candidate"])
        ],
        "evaluated_checkpoint": previous["checkpoint"],
        "consumed": {"environment_steps": 0, "scoring_agent_steps": spent_steps},
    }


def run(plan):
    if plan['resolved']['request']['schema_version']==4 and plan['phase']!='evaluate':
        from driversi_agents.chainflow.distributed import run as run_distributed
        return run_distributed(plan)
    import torch
    from torch.utils.data import default_collate

    root, out = project_root(), Path(plan["output"])
    repo = root / plan["model_config"]["repository"]
    sys.path[:0] = [str(repo), str(root / "src/vendor/driving/nuplan-devkit")]
    os.environ["CHAINFLOW_ENABLE_RAY"] = "0"
    os.environ["CHAINFLOW_USE_RAY_SCORE"] = "0"
    os.environ["NUPLAN_MAPS_ROOT"] = plan["model_config"]["paths"]["maps"]
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    req = plan["resolved"]["request"]
    fraction = plan["model_config"]["settings"].get("gpu_memory_fraction", 0.12)
    if not 0 < fraction <= 0.25:
        raise ValueError("Explicit bounded pilot memory fraction required")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    dataset = validate_manifest(plan["data"]["manifest"], groups=plan["source_groups"])
    preflight_rows = dataset["splits"]["train"] + dataset["splits"]["development"]
    if dataset["schema_version"] in (2, 3):
        # Original data were inventoried in full; NativeScenes verifies actual
        # inputs lazily. Do not decompress 103k caches before every bounded run.
        preflight_rows = [r for r in preflight_rows if r["kind"] == "synthetic_navtrain"]
    validation_steps = _verify_inputs(
        preflight_rows,
        out / "scoring-budget.json", req["budget"]["max_scoring_agent_steps"],
    )
    previous = read_mapping(plan["previous"]) if plan["previous"] else None
    deterministic(req["pilot"]["seed"])
    if plan["phase"] == "evaluate":
        return _evaluate(plan, previous, dataset, validation_steps=validation_steps)
    parent = path_at(root, plan["model_config"]["weights"]["checkpoint"])
    agent, cfg = _agent(plan, parent)
    agent.train()
    rows = dataset["splits"]["train"]
    scenes = NativeScenes(rows, agent, native_original_windows=dataset["schema_version"] in (2, 3))
    spent = _score_function(
        agent, rows, out / "scoring-budget.json", req["budget"]["max_scoring_agent_steps"],
        validation_steps=validation_steps,
    )
    recipe = _training_recipe(cfg, plan["model_config"]["settings"], req["pilot"]["batch_size"])
    if dataset["schema_version"] in (2, 3):
        recipe.update(sample_order="full_navtrain_seeded_mix", sampling_seed=req["pilot"]["seed"],
                      synthetic_every=8, training_population="complete_official_navtrain",
                      original_temporal_sampling="official_indexed_windows_raw_timestamps",
                      epoch_boundary="bounded_updates_from_shuffled_full_pool_not_full_epoch")
    # The native agent owns these same configuration values; bind explicitly so
    # future upstream constructors that copy their arguments remain correct.
    agent.scheduler_args = cfg.scheduler_args
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, out / "native-config.yaml")
    write_json(out / "effective-training-recipe.json", recipe)
    optimizers, schedules = agent.get_optimizers()
    if len(optimizers) != 1 or len(schedules) != 1 or schedules[0].get("interval", "epoch") != "step" or schedules[0].get("frequency", 1) != 1:
        raise ValueError("Native optimizer schedule protocol changed; explicit adaptation required")
    optimizer, scheduler = optimizers[0], schedules[0]["scheduler"]
    start, cursor, sample_trace = 0, 0, []
    if previous:
        state = torch.load(previous["recovery_checkpoint"], map_location="cpu", weights_only=False)
        _validate_recovery(state, plan["context"], recipe, rows)
        agent.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, cursor = state["total_updates"], state["cursor"]
        sample_trace = state["sample_trace"]
        restore_rng(state["rng"])

    def state_dict():
        return {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
            "cursor": cursor,
            "sample_trace": sample_trace,
            "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
            "effective_training_recipe": recipe,
        }

    restored = fingerprint(state_dict())
    parameters_before = _parameter_fingerprint(agent)
    if start + req["budget"]["max_optimizer_steps"] > recipe["scheduler"]["total_updates"]:
        raise ValueError("Requested updates exceed the sealed native schedule horizon")
    losses = []
    batch_size = req["pilot"]["batch_size"]
    indices = _sample_order(rows, (start + req["budget"]["max_optimizer_steps"]) * batch_size, recipe)
    for update in range(req["budget"]["max_optimizer_steps"]):
        # Deterministic sequential sample cursor is transport, not altered model
        # stochasticity. No loader workers consume model RNG before recovery.
        ids = [rows[indices[cursor + j]]["sample_id"] for j in range(batch_size)]
        batch = default_collate([scenes[indices[cursor + j]] for j in range(batch_size)])
        features, targets = _move(batch, "cuda")
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        scheduler_step = scheduler.last_epoch
        optimizer.zero_grad(set_to_none=True)
        predictions = agent.forward(features, targets)
        loss_dict = agent.compute_loss(features, targets, predictions)
        loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in agent.parameters()):
            raise ValueError("Nonfinite native gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in agent.parameters() if p.requires_grad):
            raise ValueError("Nonfinite native parameter after optimizer update")
        scheduler.step()
        cursor += batch_size
        sample_trace.append(
            {
                "optimizer_step": start + update + 1,
                "sample_ids": ids,
                "learning_rates_before": learning_rates,
                "native_scheduler_step_before": scheduler_step,
            }
        )
        losses.append(
            {
                k: float(v.detach().cpu())
                for k, v in loss_dict.items()
                if torch.is_tensor(v) and v.numel() == 1
            }
        )
        write_json(
            out / "training-progress.json",
            {"updates": update + 1, "losses": losses, "cursor": cursor},
        )
        print("optimizer_update", start + update + 1, "loss", losses[-1]["loss"], flush=True)
        del predictions, loss_dict, loss, features, targets, batch
    parameters_after = _parameter_fingerprint(agent)
    checkpoint, recovery = out / "checkpoints/model.ckpt", out / "checkpoints/recovery.pt"
    checkpoint.parent.mkdir()
    state = state_dict()
    state.update(context=plan["context"], total_updates=start + len(losses))
    torch.save(
        {"state_dict": {"agent." + k: v.detach().cpu() for k, v in agent.state_dict().items()}},
        checkpoint,
    )
    torch.save(state, recovery)
    return {
        "status": "completed",
        "reason": None,
        "checkpoint": str(checkpoint),
        "recovery_checkpoint": str(recovery),
        "checkpoint_files": {str(p): digest(p) for p in (checkpoint, recovery)},
        "updates": len(losses),
        "sample_trace": sample_trace,
        "total_updates": start + len(losses),
        "losses": losses,
        "state_fingerprint": fingerprint(
            {k: v for k, v in state.items() if k not in ("context", "total_updates")}
        ),
        "restored_fingerprint": restored,
        "resume_from": previous["recovery_checkpoint"] if previous else None,
        "checkpoint_state": ["model", "optimizer", "scheduler", "global_rng", "sample_cursor"],
        "native_loss_unchanged": True,
        "native_optimizer_scheduler_unchanged": not recipe["incremental_schedule_override"],
        "native_optimizer_scheduler_algorithm_unchanged": True,
        "effective_training_recipe": recipe,
        "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
        "training_population_size": len(rows),
        "actual_training_sample_count": len({s for item in sample_trace for s in item["sample_ids"]}),
        "native_original_source_time_gaps": scenes.original_time_gaps,
        "native_schedule_dataset_size": int(cfg.scheduler_args.dataset_size),
        "warmup_updates_executed": max(0, min(start + len(losses), recipe["scheduler"]["warmup_updates"]) - min(start, recipe["scheduler"]["warmup_updates"])),
        "parameter_update_evidence": {
            "before_sha256": parameters_before,
            "after_sha256": parameters_after,
            "trainable_parameters_changed": parameters_before != parameters_after,
            "scope": "all_trainable_parameter_bytes_not_driving_benefit",
        },
        "scope": "bounded_native_update_not_full_recipe_performance",
        "consumed": {"environment_steps": 0, "scoring_agent_steps": spent["reserved_agent_steps"]},
    }
