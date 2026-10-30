"""Bounded current-checkpoint confirmation of retrieved original navtrain scenes.

This adapter only selects, verifies, invokes unchanged native inference/scoring,
and records evidence. A threshold shortfall is not a causal failure diagnosis.
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Match the existing native training worker: old environments keep their released
# dependencies, while direct file invocation exposes only owned package paths.
if __package__ in (None, ""):
    import types

    _root = Path(__file__).resolve().parents[2]
    for _name, _folder in (("driversi", "src"), ("driversi_agents", "agents")):
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_root / _folder)]
        sys.modules[_name] = _module

from driversi.agents.backends.native import digest, path_at, project_root, signature
from driversi.runtime.budget import write_json
from driversi_agents.chainflow.data import NativeScenes, official_train_logs, source_group

SCORING_STEPS_PER_SAMPLE = 80
METRICS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
)


def official_train_tokens():
    import yaml

    path = project_root() / (
        "agents/chainflow/native/navsim/planning/script/config/common/"
        "train_test_split/scene_filter/navtrain.yaml"
    )
    tokens = yaml.safe_load(path.read_text()).get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Exact official navtrain token allowlist unavailable")
    return set(tokens)


def validate_official_tokens(rows, tokens):
    for row in rows:
        if row["token"] not in tokens:
            raise ValueError(
                "Selected token is outside exact official navtrain filter: " + row["token"]
            )


def validate_config(config):
    required = {
        "schema_version",
        "manifest",
        "selected_tokens",
        "allowed_source_logs",
        "stage1_config",
        "output_dir",
        "cuda_device",
        "max_samples",
        "max_scoring_steps",
    }
    optional = {
        "score_threshold",
        "seed",
        "gpu_memory_fraction",
        "maps_root",
        "protected_tokens",
        "protected_source_logs",
        "development_tokens",
        "development_source_logs",
        "retrieval_index",
    }
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError("Malformed bounded confirmation config")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported confirmation schema")
    tokens = config["selected_tokens"]
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(t, str) or not t for t in tokens)
        or len(tokens) != len(set(tokens))
    ):
        raise ValueError("Explicit unique nonempty selected_tokens required")
    maximum = config["max_samples"]
    if type(maximum) is not int or not 1 <= maximum <= 64 or len(tokens) > maximum:
        raise ValueError("Selected samples exceed bounded max_samples in [1,64]")
    budget = config["max_scoring_steps"]
    if (
        type(budget) is not int
        or budget < len(tokens) * SCORING_STEPS_PER_SAMPLE
        or budget > 64 * SCORING_STEPS_PER_SAMPLE
    ):
        raise ValueError("Native scoring budget must cover 80 steps per sample, at most 5120")
    if type(config["cuda_device"]) is not int or config["cuda_device"] < 0:
        raise ValueError("Explicit physical cuda_device required")
    if config.get("score_threshold", 0.95) != 0.95:
        raise ValueError("This confirmation protocol fixes score_threshold at 0.95")
    fraction = config.get("gpu_memory_fraction", 0.12)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (float, int))
        or not 0 < fraction <= 0.25
    ):
        raise ValueError("gpu_memory_fraction must be in (0,0.25]")
    if type(config.get("seed", 0)) is not int:
        raise ValueError("Integer seed required")
    for field in (
        "allowed_source_logs",
        "protected_tokens",
        "protected_source_logs",
        "development_tokens",
        "development_source_logs",
    ):
        values = config.get(field, [])
        if (
            not isinstance(values, list)
            or any(not isinstance(v, str) or not v for v in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("Invalid identity allow/deny list: " + field)
    if not config["allowed_source_logs"]:
        raise ValueError("Explicit source-log allowlist required")
    return config


def select_records(manifest, config, official_logs):
    """Fail closed before touching model/data files; reject heldout source groups."""
    if (
        manifest.get("schema_version") not in (1, 2)
        or manifest.get("kind") != "chainflow_native_scene_dataset"
        or not isinstance(manifest.get("splits"), dict)
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("Malformed native-scene manifest")
    splits = manifest["splits"]
    if not isinstance(splits.get("train"), list) or not splits["train"]:
        raise ValueError("Native train records required")
    denied_tokens = set(config.get("protected_tokens", []) + config.get("development_tokens", []))
    denied_groups = {
        source_group(s)
        for s in config.get("protected_source_logs", []) + config.get("development_source_logs", [])
    }
    seen, train = set(), {}
    for split, rows in splits.items():
        if not isinstance(rows, list):
            raise ValueError("Malformed native split")
        for row in rows:
            token, log = row.get("token"), row.get("source_log")
            if not token or not log or token in seen:
                raise ValueError("Duplicate or missing manifest identity")
            seen.add(token)
            if split == "train":
                train[token] = row
            else:
                denied_tokens.add(token)
                denied_groups.add(source_group(log))
    allow = set(config["allowed_source_logs"])
    selected = []
    for token in config["selected_tokens"]:
        if token not in train or token in denied_tokens:
            raise ValueError("Selected token is outside eligible train: " + token)
        row = train[token]
        if row["source_log"] not in allow or row["source_log"] not in official_logs:
            raise ValueError("Selected source is outside explicit/official navtrain allowlists")
        if source_group(row["source_log"]) in denied_groups:
            raise ValueError("Selected source overlaps development/protected source group")
        if row.get("kind") != "original_navtrain" or row.get("admission") is not None:
            raise ValueError("Only original navtrain records are eligible")
        if row.get("sample_id") != token:
            raise ValueError("Original sample_id must equal token")
        if type(row.get("anchor_index")) is not int or row["anchor_index"] < 3:
            raise ValueError("Complete native history anchor required")
        selected.append(row)
    return selected


def verify_seals(rows, files):
    """Hash selected train inputs only; heldout payloads are never opened."""
    checked = {}
    for row in rows:
        images = row.get("image_files", [])
        if len(images) != 4 or len(set(images)) != 4:
            raise ValueError("Four unique sealed native camera images required")
        required = [row[k] for k in ("log_path", "metric_cache", "training_metric_cache")]
        for name in required + images:
            if not Path(name).is_absolute() or name not in files:
                raise ValueError("Selected input not sealed: " + name)
            if name not in checked:
                checked[name] = digest(name)
            if checked[name] != files[name]:
                raise ValueError("Selected native input changed: " + name)
    return checked


def classify_metrics(metrics):
    if set(metrics) != set(METRICS) or any(
        isinstance(v, bool)
        or not isinstance(v, (float, int))
        or not math.isfinite(v)
        or not 0 <= v <= 1
        for v in metrics.values()
    ):
        raise ValueError("Invalid native PDM metrics; no fallback accepted")
    return {
        "status": "scored",
        "classification": "threshold_shortfall" if metrics["score"] < 0.95 else "threshold_met",
        "confirmation": "current_checkpoint_training_candidate",
        "causal_model_failure": None,
        "diagnostic_submetrics_below_one": [k for k in METRICS if k != "score" and metrics[k] < 1],
    }


def _runtime(root, cfg, stage):
    # Must execute before importing torch or native modules that initialize it.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_device"])
    os.environ["CHAINFLOW_ENABLE_RAY"] = "0"
    os.environ["CHAINFLOW_USE_RAY_SCORE"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["SWANLAB_MODE"] = "disabled"
    os.environ["NUPLAN_MAPS_ROOT"] = str(
        path_at(root, cfg.get("maps_root", "/data/xinyao/navsim_workspace/dataset/maps"))
    )
    sys.path[:0] = [
        str(path_at(root, stage["native_root"])),
        str(root / "src/vendor/driving/nuplan-devkit"),
    ]
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()


def _gpu_snapshot(device):
    # Device metadata only: no process command lines or unrelated workload changes.
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + str(device),
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).strip()
    fields = [s.strip() for s in output.split(",")]
    if len(fields) != 7:
        raise ValueError("Unable to inspect selected GPU")
    return dict(
        zip(
            (
                "index",
                "uuid",
                "name",
                "memory_total_mib",
                "memory_used_mib",
                "memory_free_mib",
                "utilization_percent",
            ),
            fields,
        )
    )


def _agent(root, stage, cfg, out):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from driversi.training.orchestration.state import deterministic

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one explicitly selected CUDA device required")
    torch.cuda.set_per_process_memory_fraction(cfg.get("gpu_memory_fraction", 0.12), 0)
    deterministic(cfg.get("seed", 0))
    native = OmegaConf.load(path_at(root, stage["native_root"]) / stage["native_config"])
    native.batch_size = 1
    native.num_gpus = 1
    native.enable_ray = False
    native.progress_bar = False
    native.config.image_backbone.model_weights = str(path_at(root, stage["backbone"]))
    native.config.lidar_backbone.model_weights = native.config.image_backbone.model_weights
    native.checkpoint_path = str(path_at(root, stage["checkpoint"]))
    native.checkpoint_strict_load = True
    OmegaConf.save(native, out / "native-config.yaml")
    agent = instantiate(native)
    agent.initialize()
    return agent.to("cuda").eval()


def _confirm_one(agent, scenes, index, row):
    import numpy as np
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from driversi_agents.chainflow.stage1 import trajectory_from_input

    trajectory = trajectory_from_input(agent, scenes.scene(index).get_agent_input())
    if np.asarray(trajectory.poses).shape != (8, 3) or not np.isfinite(trajectory.poses).all():
        raise ValueError("Invalid native selected trajectory")
    cache, cache_evidence = evaluation_cache(scenes, index, row)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    scored = pdm_score(cache, trajectory, sampling, PDMSimulator(sampling), PDMScorer(sampling))
    metrics = {k: float(v) for k, v in asdict(scored).items()}
    return dict(classify_metrics(metrics), metrics=metrics, poses=trajectory.poses.tolist(),
                evaluation_cache=cache_evidence)


def evaluation_cache(scenes, index, row):
    """Materialize native evaluation context when a training cache lacks it.

    The original training cache is immutable. A missing PDM reference is rebuilt
    by the native planner, never replaced by GT or a zero-progress shortcut.
    """
    import lzma
    import pickle
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    source = Path(row["metric_cache"])
    with lzma.open(source, "rb") as stream:
        cache = pickle.load(stream)
    evidence = dict(source=str(source), source_sha256=digest(source), rebuilt=False)
    if not hasattr(cache, "trajectory"):
        if row["kind"] != "original_navtrain":
            raise ValueError("Missing reference outside original training cache")
        started = time.monotonic()
        scenario = NavSimScenario(scenes.scene(index, sensors=False),
                                  map_root=os.environ["NUPLAN_MAPS_ROOT"],
                                  map_version="nuplan-maps-v1.0")
        processor = MetricCacheProcessor(str(scenes.evaluation_cache_dir), False)
        entry = processor.compute_metric_cache(scenario)
        if entry is None:
            raise ValueError("Native evaluation cache build failed")
        source = Path(entry.file_name)
        if not source.resolve().is_relative_to(scenes.evaluation_cache_dir.resolve()):
            raise ValueError("Generated cache escapes attempt directory")
        with lzma.open(source, "rb") as stream:
            cache = pickle.load(stream)
        evidence.update(rebuilt=True, preparation_seconds=time.monotonic() - started,
                        preparation_protocol="unchanged_native_MetricCacheProcessor")
    if (not hasattr(cache, "trajectory") or
            Path(cache.file_path).parent.name != row["token"] or
            Path(cache.file_path).parents[2].name != row["source_log"]):
        raise ValueError("Invalid native evaluation reference identity")
    evidence.update(path=str(source), sha256=digest(source))
    return cache, evidence


def run(config_path):
    root = project_root()
    config_path = path_at(root, str(config_path)).resolve(strict=True)
    cfg = validate_config(json.loads(config_path.read_text()))
    manifest_path = path_at(root, cfg["manifest"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    rows = select_records(manifest, cfg, official_train_logs())
    validate_official_tokens(rows, official_train_tokens())
    input_files = dict(manifest["files"])
    if manifest["schema_version"] == 2:
        from driversi_agents.chainflow.benchmark import validate_manifest
        validate_manifest(manifest)
        # The full index seals population/source identity; seal the actual small
        # confirmation payload here without hashing the entire image corpus.
        for row in rows:
            for name in [row[k] for k in ("log_path", "metric_cache", "training_metric_cache")] + row["image_files"]:
                if name not in input_files:
                    input_files[name] = digest(name)
    sealed = verify_seals(rows, input_files)
    stage_path = path_at(root, cfg["stage1_config"]).resolve(strict=True)
    stage = json.loads(stage_path.read_text())
    if stage.get("model") != "chainflow_stage1":
        raise ValueError("Current ChainFlow stage1 config required")
    if stage.get("inference_adapter") != "driversi_agents.chainflow.stage1:trajectory_from_input":
        raise ValueError("Unchanged native stage1 trajectory adapter required")
    checkpoint = path_at(root, stage["checkpoint"]).resolve(strict=True)
    native_root = path_at(root, stage["native_root"]).resolve(strict=True)
    protocol = {
        "name": "native_navsim_pdm_training_candidate_confirmation_v1",
        "sampling": {"num_poses": 40, "interval_length": 0.1},
        "admission": "explicit_train_log_and_exact_official_navtrain_token",
        "scoring_steps_per_sample": SCORING_STEPS_PER_SAMPLE,
        "score_threshold": 0.95,
        "cache_preparation": "at_most_one_native_rebuild_per_selected_original; separately timed",
        "native_sources": {
            str(p.relative_to(root)): digest(p)
            for p in (
                native_root / "navsim/evaluate/pdm_score.py",
                native_root
                / "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
                native_root
                / "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
                native_root / stage["native_config"],
                root / "agents/chainflow/stage1.py",
                root / "agents/chainflow/data.py",
                root / "agents/chainflow/mining.py",
                native_root / "navsim/planning/metric_caching/metric_cache_processor.py",
            )
        },
    }
    out = path_at(root, cfg["output_dir"]).resolve()
    if not out.is_relative_to((root / "exp").resolve()):
        raise ValueError("Confirmation output must remain under exp")
    if (out / "receipt.json").exists():
        raise ValueError("Confirmation receipt already exists; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "kind": "chainflow_training_candidate_confirmation",
        "status": "running",
        "scope": "selected_original_navtrain_only",
        "training_performed": False,
        "heldout_evaluation_performed": False,
        "config": str(config_path),
        "config_sha256": digest(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "stage1_config": str(stage_path),
        "stage1_config_sha256": digest(stage_path),
        "backbone_sha256": digest(path_at(root, stage["backbone"])),
        "protocol_id": signature(protocol),
        "protocol": protocol,
        "selected_tokens": cfg["selected_tokens"],
        "allowed_source_logs": cfg["allowed_source_logs"],
        "sealed_selected_inputs": sealed,
        "rows": [],
        "runtime_failures": [],
        "budget": {
            "max_samples": cfg["max_samples"],
            "max_scoring_steps": cfg["max_scoring_steps"],
            "reserved_scoring_steps": 0,
            "completed_scoring_steps": 0,
            "optimizer_steps": 0,
            "cache_preparation_count": 0,
            "cache_preparation_seconds": 0.0,
        },
        "interpretation": "Threshold shortfalls identify candidates; subscores are diagnostic, not causal labels.",
    }
    write_json(out / "receipt.json", receipt)
    agent = scenes = None
    started = time.monotonic()
    try:
        _runtime(root, cfg, stage)
        from driversi_agents.chainflow.data import verify_records

        if manifest["schema_version"] == 2:
            verify_records(rows, native_original_windows=True)
        else:
            verify_records(rows)
        receipt["actual_source_cache_alignment_verified"] = True
        receipt["gpu_before"] = _gpu_snapshot(cfg["cuda_device"])
        agent = _agent(root, stage, cfg, out)
        scenes = NativeScenes(rows, agent, native_original_windows=manifest["schema_version"] == 2)
        scenes.evaluation_cache_dir = out / "evaluation-cache"
        for index, row in enumerate(rows):
            identity = {k: row[k] for k in ("sample_id", "token", "source_log", "anchor_index")}
            budget = receipt["budget"]
            if (
                budget["reserved_scoring_steps"] + SCORING_STEPS_PER_SAMPLE
                > cfg["max_scoring_steps"]
            ):
                raise ValueError("Scoring budget exhausted before native invocation")
            budget["reserved_scoring_steps"] += SCORING_STEPS_PER_SAMPLE
            write_json(out / "receipt.json", receipt)
            try:
                result = dict(identity, **_confirm_one(agent, scenes, index, row))
                budget["completed_scoring_steps"] += SCORING_STEPS_PER_SAMPLE
                context = result.get("evaluation_cache", {})
                budget["cache_preparation_count"] += int(context.get("rebuilt", False))
                budget["cache_preparation_seconds"] += context.get("preparation_seconds", 0.0)
            except Exception as exc:
                result = dict(
                    identity,
                    status="runtime_failure",
                    classification=None,
                    causal_model_failure=None,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                receipt["runtime_failures"].append(result.copy())
            receipt["rows"].append(result)
            write_json(out / (row["token"] + ".json"), result)
            write_json(out / "receipt.json", receipt)
        receipt["status"] = (
            "completed_with_runtime_failures" if receipt["runtime_failures"] else "completed"
        )
    except Exception as exc:
        receipt["status"] = "runtime_failure"
        receipt["runtime_failures"].append({"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        try:
            verify_seals(rows, input_files)
            receipt["selected_inputs_unchanged"] = True
        except (OSError, ValueError) as exc:
            receipt["selected_inputs_unchanged"] = False
            receipt["status"] = "invalid_inputs_changed"
            receipt["runtime_failures"].append({"error_type": type(exc).__name__, "error": str(exc)})
        try:
            receipt["checkpoint_sha256_after"] = digest(checkpoint)
            receipt["checkpoint_unchanged"] = (
                receipt["checkpoint_sha256_after"] == receipt["checkpoint_sha256"]
            )
            if not receipt["checkpoint_unchanged"]:
                receipt["status"] = "invalid_checkpoint_changed"
                for result in receipt["rows"]:
                    result["confirmation"] = None
        except OSError as exc:
            receipt["status"] = "invalid_checkpoint_unavailable"
            receipt["runtime_failures"].append(
                {"error_type": type(exc).__name__, "error": str(exc)}
            )
        receipt["counts"] = {
            name: sum(r.get("classification") == name for r in receipt["rows"])
            for name in ("threshold_shortfall", "threshold_met")
        }
        receipt["counts"]["runtime_failure"] = len(receipt["runtime_failures"])
        receipt["consumed"] = {
            "scoring_agent_steps": receipt["budget"]["reserved_scoring_steps"],
            "completed_scoring_agent_steps": receipt["budget"]["completed_scoring_steps"],
            "environment_steps": 0,
            "optimizer_steps": 0,
        }
        write_json(out / "receipt.json", receipt)
        write_json(out / "report.json", receipt)
        del agent, scenes
        gc.collect()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(
        json.dumps({k: result[k] for k in ("status", "checkpoint_sha256", "protocol_id", "counts")})
    )
    return 0 if result["status"] in {"completed", "completed_with_runtime_failures"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
