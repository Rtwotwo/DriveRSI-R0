#!/usr/bin/env python3
"""Stream the training split into the existing InternVL hidden-state builder.

``mode=plan`` needs neither a model nor a GPU. ``mode=cache`` supports torchrun;
``mode=verify`` checks every selected cache without constructing InternVL.
The on-disk tensor retains the original FP32, untrimmed, no-mask semantics.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import timedelta
import hashlib
from itertools import groupby
import json
import logging
import os
from pathlib import Path
import tempfile
import uuid


logger = logging.getLogger(__name__)
MANIFEST_NAME = "internvl_cache_manifest.json"
_PDM_LOADERS = {}


def owns_sample(index: int, rank: int, world_size: int, max_samples: int = 0) -> bool:
    """A stable prefix and modulo partition, independent of cache completeness."""
    return (max_samples <= 0 or index < max_samples) and index % world_size == rank


def ensure_manifest(root: Path, identity: dict, *, create: bool) -> None:
    """Never adopt a nonempty legacy cache or mix extraction contracts."""
    manifest = root / MANIFEST_NAME
    if manifest.is_file():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded != identity:
            raise RuntimeError(f"InternVL extraction contract differs from {manifest}")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Nonempty cache root has no {MANIFEST_NAME}: {root}. "
            "Use a new output root; legacy caches may only be explicit fallback roots."
        )
    if not create:
        raise FileNotFoundError(f"Cache manifest does not exist: {manifest}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".manifest-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive publication also protects against another cache job starting.
        try:
            os.link(temporary, manifest)
        except FileExistsError:
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            if recorded != identity:
                raise RuntimeError(f"Another job published a different contract: {manifest}")
    finally:
        temporary.unlink(missing_ok=True)


def extraction_identity(cfg) -> dict:
    from omegaconf import OmegaConf
    from navsim.planning.training.vlm_cache_contract import fingerprint_checkpoint
    from navsim.planning.training.vlm_sources import resolve_sources

    agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
    checkpoint = str(agent_config.pop("checkpoint_path", ""))
    if not checkpoint:
        raise ValueError("agent.checkpoint_path must explicitly identify the original cache checkpoint")
    navsim_root = Path(__file__).resolve().parents[2]
    implementation_paths = [
        "agents/internvl_agent.py",
        "agents/chainflow_vla/encoders/vlm/utils/internvl_preprocess.py",
        "agents/chainflow_vla/encoders/vlm/utils/conversation.py",
        "planning/training/vlm_cache_reference.py",
        "planning/training/vlm_sources.py",
        "common/pdm_csv_scene_loader.py",
        "planning/script/run_dataset_caching_internvl_simscale.py",
    ]
    implementation = {
        path: hashlib.sha256((navsim_root / path).read_bytes()).hexdigest()
        for path in implementation_paths
    }
    sources = []
    for source in resolve_sources(cfg):
        source = dict(source)
        if OmegaConf.is_config(source["scene_filter"]):
            source["scene_filter"] = OmegaConf.to_container(source["scene_filter"], resolve=True)
        sources.append(source)
    return {
        "schema_version": 1,
        "pdm_shard": int(cfg.pdm_shard),
        "checkpoint": fingerprint_checkpoint(checkpoint),
        "agent": agent_config,
        "implementation_sha256": implementation,
        "training_sources": sources,
        "seed": int(cfg.seed),
        "reference_check": {"required": bool(cfg.require_reference_check), "samples": int(cfg.reference_samples),
                            "root": str(cfg.reference_cache_root), "rtol": float(cfg.reference_rtol), "atol": float(cfg.reference_atol)},
        "output": {
            "feature_name": "internvl_feature",
            "hidden_key": "last_hidden_state",
            "hidden_dim": int(cfg.hidden_dim),
            "dtype": "float32",
            "single_sample_forward": True,
            "hidden_state_layer": -1,
            "token_semantics": "original_untrimmed_hidden_no_saved_attention_mask",
        },
    }


def find_valid_cache(record: dict, roots: list[Path], validate, hidden_dim: int):
    """Match the reader's first-existing-root order; corrupt winners need repair."""
    errors = []
    for root in roots:
        path = root / record["log_name"] / record["token"] / "internvl_feature.gz"
        if not path.is_file():
            continue
        try:
            validate(path, hidden_key="last_hidden_state", hidden_dim=hidden_dim)
            return path, errors
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            # Training resolves the first existing file. A later valid fallback
            # must not conceal a corrupt file that training would actually use.
            break
    return None, errors


def write_sample_manifest(records, destination: Path) -> dict:
    """Publish identities only after full source/count/collision validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    counts = Counter()
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as handle:
            for record in records:
                line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
                handle.write(line)
                digest.update(line)
                counts[record["source"]] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(destination), "source_counts": dict(counts),
            "total_samples": sum(counts.values()), "sha256": digest.hexdigest()}


def read_sample_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _open_scene_loader(source, records, sensor_config):
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader

    if source.get("metric_cache_path"):
        from navsim.common.pdm_csv_scene_loader import PdmCsvSceneLoader

        # Each rank indexes its shard once, then reads only the requested pickle.
        key = (source["name"], source["data_path"], source["sensor_blobs_path"],
               source["metric_cache_path"], repr(sensor_config))
        if key not in _PDM_LOADERS:
            _PDM_LOADERS[key] = PdmCsvSceneLoader(
                data_path=Path(source["data_path"]),
                sensor_blobs_path=Path(source["sensor_blobs_path"]),
                scene_filter=instantiate(deepcopy(source["scene_filter"])),
                sensor_config=sensor_config,
                metric_cache_path=Path(source["metric_cache_path"]),
                load_image_path=True,
            )
        loader = _PDM_LOADERS[key]
        for record in records:
            if loader.token_to_log_file.get(record["token"]) != record["log_file"]:
                raise ValueError(f"PDM CSV identity differs from sample manifest: {record}")
        return loader

    scene_filter = instantiate(deepcopy(source["scene_filter"]))
    scene_filter.log_names = [records[0]["log_file"]]
    scene_filter.tokens = [record["token"] for record in records]
    # Original filtering already ran while enumerating the training population.
    scene_filter.max_scenes = None
    return SceneLoader(
        data_path=Path(source["data_path"]),
        sensor_blobs_path=Path(source["sensor_blobs_path"]),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )


def _validate_generated(data, hidden_dim):
    import torch
    from navsim.planning.training.vlm_stage2_compatibility import validate_legacy_payload

    # Certify every saved field, not only the hidden tensor consumed by Stage2.
    validate_legacy_payload(data)

    hidden = data.get("last_hidden_state")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
        raise ValueError("Original InternVL builder must return a rank-2 hidden tensor")
    if hidden.shape[0] <= 0 or hidden.shape[1] != hidden_dim:
        raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
    if hidden.dtype != torch.float32 or hidden.device.type != "cpu":
        raise ValueError("Original cache output must be CPU float32")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("InternVL output contains nonfinite hidden states")


def run(cfg) -> dict:
    from navsim.planning.training.vlm_sources import iter_samples, resolve_sources, select_sources

    mode = str(cfg.mode)
    if mode not in {"plan", "cache", "verify"}:
        raise ValueError("mode must be plan, cache, or verify")
    rank, world_size = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    max_samples = int(cfg.max_samples)
    verify_depth = str(cfg.verify_depth)
    if verify_depth not in {"paths", "full"}:
        raise ValueError("verify_depth must be paths or full")
    deep_sample_limit = int(cfg.verify_samples_per_source)
    if deep_sample_limit < 1:
        raise ValueError("verify_samples_per_source must be at least one")
    if max_samples < 0:
        raise ValueError("max_samples must be zero (all samples) or a positive prefix size")
    sources = select_sources(resolve_sources(cfg), cfg.source_names)
    shard = int(cfg.pdm_shard)
    expected_source = f"pdm_{shard}"
    expected_folder = f"train_VLM_features_synthetic_reaction_pdm_v1.0-{shard}"
    if shard not in range(5) or [source["name"] for source in sources] != [expected_source]:
        raise ValueError("Exactly one PDM source matching pdm_shard is required")
    if Path(str(cfg.cache_path)).name != expected_folder or list(cfg.fallback_cache_roots):
        raise ValueError("Use the matching train_VLM_features_synthetic_reaction_pdm_v1.0-N output with no fallback roots")
    source_by_name = {source["name"]: source for source in sources}
    if not source_by_name:
        raise ValueError("No data sources selected")
    if mode == "plan":
        counts = Counter(record["source"] for record in iter_samples(cfg, cfg.source_names))
        report = {
            "mode": mode, "source_counts": dict(counts), "total_samples": sum(counts.values()),
            "processing_limit": max_samples or None, "cache_path": str(cfg.cache_path),
            "fallback_cache_roots": list(cfg.fallback_cache_roots),
            "checkpoint_loaded": False,
        }
        if rank == 0:
            print(json.dumps(report, indent=2, sort_keys=True))
        return report

    from navsim.planning.training.vlm_cache_contract import atomic_dump_gzip, validate_hidden_file

    dist = None
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=int(cfg.distributed_timeout_seconds)))
    root = Path(str(cfg.cache_path))
    roots = [root, *(Path(str(path)) for path in cfg.fallback_cache_roots)]
    if len(set(path.resolve() for path in roots)) != len(roots):
        raise ValueError("Output and fallback roots must be distinct")
    run_id = uuid.uuid4().hex
    handshake = [None]
    if rank == 0:
        try:
            ensure_manifest(root, extraction_identity(cfg), create=mode == "cache")
            handshake[0] = {"run_id": run_id}
        except Exception as exc:
            handshake[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if dist is not None:
        dist.broadcast_object_list(handshake, src=0)
    if "error" in handshake[0]:
        if dist is not None:
            dist.destroy_process_group()
        raise RuntimeError(handshake[0]["error"])
    run_id = handshake[0]["run_id"]
    if cfg.fallback_cache_roots:
        logger.warning("Explicit fallback cache contents are validated, but their checkpoint provenance is not certified.")
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    # Only rank zero scans large raw scene metadata. Every rank subsequently
    # streams the same persisted lightweight identities; no model is loaded yet.
    population = [None]
    if rank == 0:
        try:
            population[0] = write_sample_manifest(
                iter_samples(cfg, cfg.source_names), report_dir / f"{run_id}.samples.jsonl"
            )
        except Exception as exc:
            population[0] = {"error": f"{type(exc).__name__}: {exc}"}
            (report_dir / f"{run_id}.population-error.json").write_text(
                json.dumps(population[0], indent=2), encoding="utf-8"
            )
    if dist is not None:
        dist.broadcast_object_list(population, src=0)
    if "error" in population[0]:
        if dist is not None:
            dist.destroy_process_group()
        raise RuntimeError(f"Source population validation failed before model initialization: {population[0]['error']}")
    logger.info("Validated shared sample manifest: %s", population[0])
    failures_path = report_dir / f"{run_id}.rank-{rank}.failures.jsonl"
    counts = Counter()
    source_counts = Counter()
    deep_counts = Counter()
    agent = builder = None
    fatal_error = None

    def failure(record, exc):
        counts["failed"] += 1
        entry = {**record, "error": f"{type(exc).__name__}: {exc}"}
        with failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        logger.error("Cache failure %s/%s: %s", record.get("log_name"), record.get("token"), exc)

    try:
        indexed = enumerate(read_sample_manifest(Path(population[0]["path"])))
        for (source_name, _), group in groupby(indexed, key=lambda item: (item[1]["source"], item[1]["log_file"])):
            assigned = []
            for index, record in group:
                source_counts[record["source"]] += 1
                if owns_sample(index, rank, world_size, max_samples):
                    assigned.append(record)
            pending = []
            for record in assigned:
                counts["assigned"] += 1
                check_content = mode != "verify" or verify_depth == "full" or deep_counts[source_name] < deep_sample_limit
                validator = validate_hidden_file if check_content else lambda *args, **kwargs: {}
                valid, corrupt = find_valid_cache(record, roots, validator, int(cfg.hidden_dim))
                if valid is not None:
                    counts["valid_existing"] += 1
                    if check_content:
                        deep_counts[source_name] += 1
                        counts["deep_checked"] += 1
                    else:
                        counts["path_checked_only"] += 1
                    if not valid.is_relative_to(root):
                        counts["fallback"] += 1
                    continue
                if mode == "verify":
                    failure(record, ValueError("; ".join(corrupt) if corrupt else "No feature cache in configured roots"))
                else:
                    if corrupt:
                        failure(record, ValueError("Existing cache is corrupt; refusing to overwrite. " + "; ".join(corrupt)))
                    else:
                        pending.append(record)
            if not pending:
                continue
            if builder is None:
                import random
                import numpy as np
                import torch
                from hydra.utils import instantiate
                if not torch.cuda.is_available():
                    raise RuntimeError("InternVL cache extraction requires CUDA")
                torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
                random.seed(int(cfg.seed))
                np.random.seed(int(cfg.seed))
                torch.manual_seed(int(cfg.seed))
                rank_runtime_cache = Path(tempfile.gettempdir()) / f"internvl-{run_id}-rank-{rank}"
                os.environ["TRANSFORMERS_CACHE"] = str(rank_runtime_cache / "models")
                os.environ["HF_MODULES_CACHE"] = str(rank_runtime_cache / "modules")
                if not bool(cfg.agent.cache_mode) or not bool(cfg.agent.cache_hidden_state):
                    raise ValueError("InternVL requires cache_mode=true and cache_hidden_state=true")
                agent = instantiate(cfg.agent)
                builders = agent.get_feature_builders()
                if len(builders) != 1 or builders[0].get_unique_name() != "internvl_feature":
                    raise ValueError("Expected the original single InternVL feature builder")
                builder = builders[0]
                if bool(getattr(cfg, "require_reference_check", False)):
                    from navsim.planning.training.vlm_cache_reference import validate_reference
                    reference_report = validate_reference(cfg, agent, builder, _open_scene_loader)
                    reference_report["cache_manifest_sha256"] = hashlib.sha256((root / MANIFEST_NAME).read_bytes()).hexdigest()
                    (report_dir / f"{run_id}.rank-{rank}.reference.json").write_text(
                        json.dumps(reference_report, indent=2), encoding="utf-8")
                    logger.info("Legacy real-cache numerical reference check PASSED on rank %d", rank)
            try:
                loader = _open_scene_loader(source_by_name[source_name], pending, agent.get_sensor_config())
            except Exception as exc:
                for record in pending:
                    failure(record, exc)
                continue
            for record in pending:
                try:
                    import torch
                    scene = loader.get_scene_from_token(record["token"])
                    metadata = scene.scene_metadata
                    if metadata.log_name != record["log_name"] or metadata.initial_token != record["token"]:
                        raise ValueError("Scene identity differs from the training-source manifest")
                    with torch.inference_mode():
                        data = builder.compute_features(scene.get_agent_input())
                    _validate_generated(data, int(cfg.hidden_dim))
                    output = root / record["log_name"] / record["token"] / "internvl_feature.gz"
                    atomic_dump_gzip(output, data)
                    counts["written"] += 1
                except Exception as exc:
                    failure(record, exc)
            del loader
            logger.info("rank=%d counts=%s", rank, dict(counts))
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        failure({"source": "<enumeration-or-model>"}, exc)

    report = {"rank": rank, "mode": mode, "verify_depth": verify_depth, "counts": dict(counts), "source_counts": dict(source_counts),
              "sample_manifest": population[0],
              "max_samples": max_samples, "failures_path": str(failures_path), "fatal_error": fatal_error}
    (report_dir / f"{run_id}.rank-{rank}.summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    reports = [report]
    if dist is not None:
        reports = [None] * world_size
        dist.all_gather_object(reports, report)
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"ranks": reports}, indent=2, sort_keys=True))
    if any(item["counts"].get("failed", 0) for item in reports):
        raise RuntimeError(f"Cache {mode} did not complete cleanly; inspect {report_dir}")
    return report


if __name__ == "__main__":
    import hydra

    hydra.main(config_path="config/training", config_name="cache_internvl_simscale", version_base=None)(run)()
