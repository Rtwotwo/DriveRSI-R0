"""Recompute legacy real samples before allowing new synthetic cache writes."""
from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path


def compare_features(reference, current, rtol=0.0, atol=0.0):
    import numpy as np

    if rtol < 0 or atol < 0:
        raise ValueError("Reference comparison tolerances must be non-negative")
    result = {}
    for key in ("history_trajectory", "high_command_one_hot", "status_feature", "last_hidden_state"):
        if key not in reference or key not in current:
            raise ValueError(f"Reference/current cache is missing {key}")
        values = []
        for data in (reference[key], current[key]):
            if hasattr(data, "detach"):
                data = data.detach().cpu().numpy()
            values.append(np.asarray(data))
        old, new = values
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError(f"{key}: shape/dtype differs: {old.shape}/{old.dtype} vs {new.shape}/{new.dtype}")
        if not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError(f"{key}: nonfinite values")
        error = float(np.abs(old - new).max()) if old.size else 0.0
        if not np.allclose(old, new, rtol=rtol, atol=atol):
            raise ValueError(f"{key}: legacy/current values differ (max_abs={error}, rtol={rtol}, atol={atol}). "
                             "Stop: confirm checkpoint, prompt, tokenizer, image preprocessing and runtime; do not mix caches.")
        result[key] = {"shape": list(old.shape), "dtype": str(old.dtype), "max_abs_error": error,
                       "exact_equal": bool(np.array_equal(old, new))}
    return result


def validate_reference(cfg, agent, builder, open_scene_loader):
    """Single-sample forwards, read-only legacy files; no generated reference is saved."""
    import torch
    from navsim.planning.training.vlm_sources import resolve_sources

    root = Path(str(cfg.reference_cache_root))
    if not root.is_dir():
        raise FileNotFoundError(f"Legacy reference cache missing: {root}")
    source = next(item for item in resolve_sources(cfg) if item["name"] == "real")
    scene_filter = source["scene_filter"]
    allowed_logs = scene_filter.get("log_names")
    allowed_logs = None if allowed_logs is None else set(allowed_logs)
    allowed_tokens = scene_filter.get("tokens")
    allowed_tokens = None if allowed_tokens is None else set(allowed_tokens)
    count = int(cfg.reference_samples)
    if count < 1:
        raise ValueError("At least one legacy reference sample is required")
    records = []
    # Choose different real logs without scanning or loading the full real dataset.
    for log_dir in sorted(root.iterdir()):
        if not log_dir.is_dir() or (allowed_logs is not None and log_dir.name not in allowed_logs):
            continue
        if not (Path(source["data_path"]) / (log_dir.name + ".pkl")).is_file():
            continue
        for token_dir in sorted(log_dir.iterdir()):
            if not token_dir.is_dir() or (allowed_tokens is not None and token_dir.name not in allowed_tokens):
                continue
            path = token_dir / "internvl_feature.gz"
            if path.is_file():
                records.append(({"source": "real", "log_file": log_dir.name, "log_name": log_dir.name,
                                 "token": token_dir.name}, path))
                break
        if len(records) == count:
            break
    if len(records) != count:
        raise RuntimeError(f"Need {count} legacy samples from distinct real logs; found {len(records)}")
    results = []
    for record, path in records:
        loader = open_scene_loader(source, [record], agent.get_sensor_config())
        scene = loader.get_scene_from_token(record["token"])
        if scene.scene_metadata.log_name != record["log_name"] or scene.scene_metadata.initial_token != record["token"]:
            raise RuntimeError(f"Legacy reference scene identity mismatch: {record}")
        with gzip.open(path, "rb") as handle:
            reference = pickle.load(handle)
            handle.read()
        with torch.inference_mode():
            current = builder.compute_features(scene.get_agent_input())
        result = compare_features(reference, current, float(cfg.reference_rtol), float(cfg.reference_atol))
        results.append({**record, "cache_file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "comparison": result})
        del loader, scene, reference, current
    return {"status": "passed", "samples": results, "rtol": float(cfg.reference_rtol), "atol": float(cfg.reference_atol),
            "scope": "Sampled numerical equivalence; not proof of provenance for every legacy file"}
