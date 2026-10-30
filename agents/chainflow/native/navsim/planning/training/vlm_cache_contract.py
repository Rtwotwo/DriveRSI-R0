"""Legacy InternVL cache contract, shared by the writer and training preflight."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


def cache_file(root, log_name, token, feature_name="internvl_feature"):
    for value in (log_name, token, feature_name):
        if not value or str(value) in (".", "..") or "/" in str(value) or "\\" in str(value):
            raise ValueError(f"Invalid cache path component: {value!r}")
    return Path(root) / str(log_name) / str(token) / f"{feature_name}.gz"


def resolve_cache_file(roots, log_name, token, feature_name="internvl_feature"):
    candidates = [cache_file(root, log_name, token, feature_name) for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing VLM cache for {log_name}/{token}; checked {candidates}")


def validate_hidden_data(data, hidden_key="last_hidden_state", hidden_dim=1536):
    import torch
    if not isinstance(data, dict) or hidden_key not in data:
        raise ValueError(f"Cache must contain {hidden_key!r}")
    value = data[hidden_key]
    value = value if torch.is_tensor(value) else torch.as_tensor(value)
    # Preserve the original single-sample hidden tensor, including tokenizer padding.
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != hidden_dim:
        raise ValueError(f"Expected [positive_seq_len, {hidden_dim}], got {tuple(value.shape)}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("VLM hidden state must be floating point and finite")
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def validate_hidden_file(path, hidden_key="last_hidden_state", hidden_dim=1536):
    with gzip.open(path, "rb") as stream:
        data = pickle.load(stream)
        # Read through gzip EOF to verify its CRC, not just the pickle STOP opcode.
        stream.read()
    return validate_hidden_data(data, hidden_key, hidden_dim)


def atomic_dump_gzip(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1) as stream:
                pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
            raw.flush()
            os.fsync(raw.fileno())
        # Publish exclusively: never overwrite even if another job won the race.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint_checkpoint(path):
    """Stream SHA256 for model weights and tokenizer/config/code files."""
    root = Path(path).resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config.json missing: {root}")
    files = []
    weights = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or ".git" in item.parts or "__pycache__" in item.parts:
            continue
        if item.suffix not in (".json", ".py", ".txt", ".model", ".safetensors", ".bin"):
            continue
        stat = item.stat()
        record = {"file": item.relative_to(root).as_posix(), "bytes": stat.st_size}
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
        if item.suffix in (".safetensors", ".bin"):
            weights.append(record)
        files.append(record)
    if not weights:
        raise FileNotFoundError(f"No checkpoint weight files found: {root}")
    return {"path": str(root), "files": files, "weight_identity": "full SHA256"}


def validate_dataset_vlm_coverage(dataset, spec, samples_per_source=1):
    """Check actual training IDs, not a cache directory's total file count.

    Every path is checked; a bounded set per source is unpickled. Full integrity
    verification is available in the standalone cache command's verify mode.
    """
    children = getattr(dataset, "datasets", [dataset])
    roots = [spec.cache_root, *getattr(spec, "fallback_roots", ())]
    seen = set()
    seen_tokens = set()
    counts = []
    missing = []
    for index, child in enumerate(children):
        loader = getattr(child, "_scene_loader", None)
        if loader is None:
            raise ValueError("VLM composite preflight requires SceneLoader-backed Dataset")
        checked = 0
        count = 0
        for token, frames in loader.scene_frames_dicts.items():
            log = frames[loader._scene_filter.num_history_frames - 1]["log_name"]
            pair = (log, token)
            if pair in seen or token in seen_tokens:
                raise ValueError(f"Duplicate training sample/token across sources: {pair}")
            seen.add(pair)
            seen_tokens.add(token)
            count += 1
            try:
                path = resolve_cache_file(roots, log, token, spec.feature_name)
            except FileNotFoundError:
                if len(missing) < 20:
                    missing.append(f"{log}/{token}")
                continue
            if checked < samples_per_source:
                validate_hidden_file(path, spec.hidden_key, 1536)
                checked += 1
        counts.append({"source_index": index, "samples": count, "deep_checked": checked})
    if missing:
        raise RuntimeError(f"Training VLM coverage incomplete; first missing samples: {missing}")
    if sum(row["samples"] for row in counts) != len(dataset):
        raise RuntimeError("VLM preflight sample count differs from training Dataset")
    return counts


def require_legacy_reference_report(root):
    """Do not publish a mixed view unless the new extractor passed its legacy gate."""
    root = Path(root)
    manifest_path = root / "internvl_cache_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    check = manifest.get("reference_check", {})
    if not check.get("required") or check.get("samples", 0) < 3 or check.get("rtol") != 0 or check.get("atol") != 0:
        raise ValueError("This mixed-cache release requires an enabled 3-sample exact legacy reference check")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    for report_path in sorted((root / "reports").glob("*.reference.json")):
        report = json.loads(report_path.read_text())
        if report.get("status") != "passed" or report.get("cache_manifest_sha256") != digest:
            continue
        if report.get("rtol") != 0 or report.get("atol") != 0 or len(report.get("samples", [])) < 3:
            continue
        fields = {"history_trajectory", "high_command_one_hot", "status_feature", "last_hidden_state"}
        if all(set(sample.get("comparison", {})) == fields and
               all(field.get("exact_equal") for field in sample["comparison"].values()) for sample in report["samples"]):
            return {"report": str(report_path), "cache_manifest_sha256": digest}
    raise RuntimeError("No successful exact legacy-reference report matches this new cache manifest")
