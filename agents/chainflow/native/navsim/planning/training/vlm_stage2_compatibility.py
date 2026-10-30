"""Add-only, source-aware acceptance of legacy InternVL caches for Stage2.

The original Dataset, feature builder, tensor padding and model are unchanged.
A full gzip/schema audit is reused only while every selected file's identity,
size and timestamps remain unchanged. Receipts are separate from feature roots.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import uuid

import torch


TRAIN_COUNTS = {"navtrain": 85109, "navtrainval": 103288, "navtrain_simscale": 308119}
PDM_COUNTS = [33000, 55471, 46261, 37897, 32202]
FIELDS = {"history_trajectory", "high_command_one_hot", "last_hidden_state", "status_feature"}


def validate_legacy_payload(data):
    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError(f"Expected exactly four legacy InternVL fields: {sorted(FIELDS)}")
    fixed = {"history_trajectory": (4, 3), "high_command_one_hot": (4,), "status_feature": (8,)}
    report = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError(f"{key} must be a CPU torch.float32 tensor")
        if key in fixed and tuple(value.shape) != fixed[key]:
            raise ValueError(f"{key}: expected {fixed[key]}, got {tuple(value.shape)}")
        if key == "last_hidden_state" and (value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != 1536):
            raise ValueError(f"last_hidden_state: expected [L>0,1536], got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} contains NaN or Inf")
        report[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return report


def load_legacy_payload(path):
    with gzip.open(path, "rb") as handle:
        data = pickle.load(handle)
        handle.read()  # gzip CRC, including bytes following the pickle STOP.
    validate_legacy_payload(data)
    return data


def source_cache_roots(split_name, cache_base, real_root):
    if split_name not in TRAIN_COUNTS:
        raise ValueError(f"Unsupported split: {split_name}")
    roots = {"real": Path(real_root).resolve()}
    if split_name == "navtrain_simscale":
        roots.update({f"pdm_{i}": (Path(cache_base) / f"train_VLM_features_synthetic_reaction_pdm_v1.0-{i}").resolve()
                      for i in range(5)})
    if len(set(roots.values())) != len(roots):
        raise ValueError("Different sources must have distinct VLM cache roots")
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{name}: VLM cache root is missing: {root}")
    return roots


def _cache_path(record, roots):
    from navsim.planning.training.vlm_cache_contract import cache_file
    return cache_file(roots[record["source"]], record["log_name"], record["token"])


def _stamp(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError(f"Cache is not a nonempty regular file: {path}")
    return [str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino]


@contextmanager
def _lock(path):
    # A single process performs the TB-scale full audit; DDP peers reuse it.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _exclusive_json(path, value):
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.part"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_records(records, roots, report_dir):
    """Check all selected paths, full schema/CRC once per unchanged population.

    Reused receipts rely on filesystem metadata, not a tamper-proof content
    hash. Never edit feature files while an audit or training job is running.
    """
    records = sorted(records, key=lambda row: (row["source"], row["log_name"], row["token"]))
    counts, keys, tokens, entries = Counter(), set(), set(), []
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for record in records:
        pair = (record["log_name"], record["token"])
        if pair in keys or record["token"] in tokens:
            raise ValueError(f"Duplicate training/cache identity: {pair}")
        keys.add(pair)
        tokens.add(record["token"])
        path = _cache_path(record, roots)
        stamp = _stamp(path)  # Missing even a single selected cache is fatal.
        entry = {"source": record["source"], "log_name": pair[0], "token": pair[1], "stamp": stamp}
        digest.update((json.dumps(entry, sort_keys=True) + "\n").encode())
        entries.append((path, stamp))
        counts[record["source"]] += 1
    if not entries:
        raise ValueError("Refusing to certify an empty cache population")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    receipt = report_dir / f"schema-{digest.hexdigest()}.json"
    with _lock(report_dir / "schema-audit.lock"):
        if receipt.is_file():
            report = json.loads(receipt.read_text())
            if report.get("status") != "full_schema_passed" or report.get("population_digest") != digest.hexdigest():
                raise ValueError(f"Invalid compatibility receipt: {receipt}")
            reused = True
        else:
            print(f"Full InternVL schema/CRC audit: {len(entries)} files. This reads all selected feature bytes.", flush=True)
            for index, (path, before) in enumerate(entries):
                load_legacy_payload(path)
                if _stamp(path) != before:
                    raise RuntimeError(f"Cache changed during audit: {path}")
                if (index + 1) % 1000 == 0:
                    print(f"InternVL schema audit {index + 1}/{len(entries)}", flush=True)
            report = {"status": "full_schema_passed", "population_digest": digest.hexdigest(),
                      "unique_samples": len(entries), "source_counts": dict(counts),
                      "schema": "four CPU float32 tensors; untrimmed hidden [L,1536]; no saved mask",
                      "roots": {name: str(root) for name, root in roots.items()}}
            _exclusive_json(receipt, report)
            reused = False
        # Detect files changed while this rank waited for another auditor.
        if any(_stamp(path) != before for path, before in entries):
            raise RuntimeError("Cache population changed while waiting for audit; retry with stable files")
    return {**report, "receipt": str(receipt), "reused": reused}


def _records(child, source):
    loader = child._scene_loader
    anchor = loader._scene_filter.num_history_frames - 1
    for token in loader.tokens:
        frame = loader.scene_frames_dicts[token][anchor]
        if str(frame["token"]) != str(token):
            raise ValueError("SceneLoader anchor token does not match its index")
        yield {"source": source, "log_name": str(frame["log_name"]), "token": str(token)}


def _smoke(children, roots):
    from navsim.planning.training.collate import chainflow_vla_collate_fn
    samples, tensors = [], []
    for name, child in children:
        record = next(_records(child, name))
        stored = load_legacy_payload(_cache_path(record, roots))["last_hidden_state"]
        sample = child[0]  # The actual Dataset/ReadThroughDataset and original builders.
        observed = sample[0].get("vlm_hidden_state")
        if not isinstance(observed, torch.Tensor) or not torch.equal(observed.cpu(), stored):
            raise ValueError(f"Dataset did not return the stored VLM tensor for {record}")
        samples.append(sample)
        tensors.append(stored)
    if any(set(sample[0]) != set(samples[0][0]) or set(sample[1]) != set(samples[0][1]) for sample in samples):
        raise ValueError("Mixed-source feature/target key sets differ")
    batch = chainflow_vla_collate_fn(samples)[0]
    hidden, mask = batch["vlm_hidden_state"], batch["vlm_attention_mask"]
    if hidden.shape[0] != len(samples) or hidden.shape[2] != 1536 or hidden.dtype != torch.float32:
        raise ValueError("Unexpected VLM mixed-batch shape/dtype")
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("Unexpected VLM batch mask")
    for index, tensor in enumerate(tensors):
        length = len(tensor)
        if not torch.equal(hidden[index, -length:].cpu(), tensor) or int(mask[index].sum()) != length:
            raise ValueError("Collation changed VLM values or sequence alignment")
        if bool(mask[index, :-length].any()) or bool(hidden[index, :-length].ne(0).any()):
            raise ValueError("VLM additional batch padding is not zero/left-aligned")
    return {"sources": [name for name, _ in children], "shape": list(hidden.shape),
            "mask_semantics": "original collate only; saved tokenizer padding remains untrimmed"}


def bind_and_validate_datasets(train_data, val_data, cfg, split_name, cache_base, real_root):
    from navsim.planning.training.dataset import VlmCacheSpec
    from navsim.planning.training.vlm_sources import resolve_sources
    from navsim.planning.training.vlm_cache_contract import require_legacy_reference_report

    roots = source_cache_roots(split_name, cache_base, real_root)
    sources = resolve_sources(cfg)
    children = list(getattr(train_data, "datasets", [train_data]))
    if [source["name"] for source in sources] != list(roots) or len(children) != len(sources):
        raise ValueError("Training children do not match the selected real/PDM source order")
    if len(train_data) != TRAIN_COUNTS[split_name] or len(val_data) != 18179:
        raise ValueError(f"Split count mismatch: train={len(train_data)}, val={len(val_data)}")
    expected = [85109 if split_name == "navtrain" else 103288]
    if split_name == "navtrain_simscale":
        expected += PDM_COUNTS
    paired, records = [], []
    for source, child, count in zip(sources, children, expected):
        name = source["name"]
        if len(child) != count:
            raise ValueError(f"{name}: selected {len(child)}, expected {count}")
        # Verify actual source membership rather than relying only on Concat order.
        loader = child._scene_loader
        actual_sensor = getattr(loader, "_sensor_blobs_path", None)
        if actual_sensor is None or Path(actual_sensor).resolve() != Path(source["sensor_blobs_path"]).resolve():
            raise ValueError(f"{name}: Dataset sensor root disagrees with configured source")
        spec = child._vlm_cache_spec
        if spec is None or spec.feature_name != "internvl_feature" or spec.hidden_key != "last_hidden_state":
            raise ValueError(f"{name}: incompatible Dataset VLM cache specification")
        child._vlm_cache_spec = VlmCacheSpec(roots[name], "internvl_feature", "last_hidden_state")
        if name != "real":
            require_legacy_reference_report(roots[name])
        paired.append((name, child))
        records.extend(_records(child, name))
    val_data._vlm_cache_spec = VlmCacheSpec(roots["real"], "internvl_feature", "last_hidden_state")
    # navtrain requires validation caches too; navtrainval already includes them.
    existing = {(record["log_name"], record["token"]): record["source"] for record in records}
    for record in _records(val_data, "real"):
        pair = (record["log_name"], record["token"])
        if pair not in existing:
            if split_name != "navtrain":
                raise ValueError("Validation population is not a subset of navtrainval")
            records.append(record)
        elif existing[pair] != "real" or split_name == "navtrain":
            raise ValueError("Unexpected training/validation identity overlap")
    report = audit_records(records, roots, Path(cache_base) / ".stage2_compatibility")
    report["train_batch"] = _smoke(paired, roots)
    report["val_batch"] = _smoke([("real", val_data)], roots)
    return report


def verify_configured_split(cfg, split_name, cache_base, real_root):
    """CPU-only full acceptance before creating the Stage2 model or Trainer."""
    from copy import deepcopy
    from navsim.planning.training.vlm_sources import resolve_sources, iter_source_samples, _copy_filter
    from navsim.planning.training.vlm_cache_contract import require_legacy_reference_report

    roots = source_cache_roots(split_name, cache_base, real_root)
    sources = resolve_sources(cfg)
    if [source["name"] for source in sources] != list(roots):
        raise ValueError("Selected Hydra sources do not match the requested split")
    for name, root in roots.items():
        if name != "real":
            require_legacy_reference_report(root)
    expected = [85109 if split_name == "navtrain" else 103288]
    if split_name == "navtrain_simscale":
        expected += PDM_COUNTS
    records = []
    for source, count in zip(sources, expected):
        source = dict(source, expected_count=count)
        records.extend(iter_source_samples(source))
    if len(records) != TRAIN_COUNTS[split_name]:
        raise ValueError("Training source count mismatch")
    if split_name == "navtrain":
        source = deepcopy(sources[0])
        # Match run_training_full.build_datasets from the original filter,
        # not from the already train-only resolved real source filter.
        scene_filter = _copy_filter(cfg.train_test_split.scene_filter)
        allowed = set(cfg.val_logs)
        original_logs = scene_filter.get("log_names")
        scene_filter["log_names"] = ([name for name in original_logs if name in allowed]
                                    if original_logs is not None else list(cfg.val_logs))
        source["scene_filter"] = scene_filter
        source["expected_count"] = 18179
        records.extend(iter_source_samples(source))
    return audit_records(records, roots, Path(cache_base) / ".stage2_compatibility")
