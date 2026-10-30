"""Shared training-source and sample identities for InternVL caching/checks.

Paths and real-log selection deliberately follow ``run_training_full``. In
particular, this module never rewrites ``navsim_logs`` to ``meta_datas`` and
never appends ``data_split`` to an already resolved real-data path.

SceneLoader is imported lazily and only one source's frame metadata is retained
at a time. Iterators must be exhausted for final total/collision checks to run.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Iterator


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _copy_filter(scene_filter: Any) -> Any:
    # Resolve interpolations while the original config still has its parent.
    # Plain dictionaries are also supported for small dependency-free tests.
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return deepcopy(scene_filter)
    if OmegaConf.is_config(scene_filter):
        return OmegaConf.create(OmegaConf.to_container(scene_filter, resolve=True))
    return deepcopy(scene_filter)


def _count(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError(f"Expected sample count must be non-negative: {value}")
    return result


def _source(
    name: str,
    data_path: Any,
    sensor_blobs_path: Any,
    scene_filter: Any,
    expected_count: Any,
    data_split: Any,
    require_invalid_tokens: bool = False,
    expected_hash: Any = None,
) -> dict[str, Any]:
    log_root, sensor_root = Path(str(data_path)), Path(str(sensor_blobs_path))
    for label, path in (("log", log_root), ("sensor", sensor_root)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: {label} directory is missing: {path}")
    invalid_path = sensor_root / "invalid_tokens.yaml"
    actual_hash = None
    if invalid_path.is_file():
        actual_hash = hashlib.sha256(invalid_path.read_bytes()).hexdigest()
    elif require_invalid_tokens or expected_hash is not None:
        raise FileNotFoundError(
            f"{name}: required invalid-token manifest is missing: {invalid_path}. "
            "Refusing to use the unfiltered shard."
        )
    if expected_hash is not None and actual_hash != str(expected_hash):
        raise RuntimeError(
            f"{name}: invalid-token manifest SHA256 changed: "
            f"{actual_hash} != {expected_hash}"
        )
    return {
        "name": name,
        "data_path": str(log_root),
        "sensor_blobs_path": str(sensor_root),
        "scene_filter": scene_filter,
        "expected_count": _count(expected_count),
        "data_split": None if data_split is None else str(data_split),
        "require_invalid_tokens": bool(require_invalid_tokens),
        "invalid_tokens_path": str(invalid_path) if invalid_path.is_file() else None,
        "invalid_tokens_sha256": actual_hash,
    }


def resolve_sources(cfg: Any) -> list[dict[str, Any]]:
    """Resolve real and synthetic sources without loading any scene pickles.

    ``scene_filter`` in each returned mapping can be passed to Hydra instantiate.
    The original config is not mutated. All declared roots and invalid-token
    hashes are checked, even when a caller later selects only some sources.
    """
    split = _get(cfg, "train_test_split")
    if split is None:
        raise ValueError("train_test_split is required")
    real_filter = _copy_filter(_get(split, "scene_filter"))
    if real_filter is None:
        raise ValueError("train_test_split.scene_filter is required")
    from navsim.planning.training.log_partitions import resolve_training_log_names

    train_logs = resolve_training_log_names(cfg)
    allowed_logs = set(train_logs)
    filter_logs = _get(real_filter, "log_names")
    selected_logs = (
        [str(log) for log in filter_logs if log in allowed_logs]
        if filter_logs is not None else train_logs
    )
    if isinstance(real_filter, Mapping):
        real_filter["log_names"] = selected_logs
    else:
        real_filter.log_names = selected_logs
    sources = [_source(
        "real", _get(cfg, "navsim_log_path"), _get(cfg, "sensor_blobs_path"),
        real_filter, _get(split, "expected_real_train_samples"),
        _get(split, "data_split"),
    )]
    for name, source_cfg in _get(split, "synthetic_sources", {}).items():
        if name in ("real", "all", "synthetic"):
            raise ValueError(f"Synthetic source name is reserved: {name}")
        source_split = _get(source_cfg, "data_split")
        if not source_split:
            raise ValueError(f"{name}: data_split is required")
        source_filter = _get(source_cfg, "scene_filter")
        if source_filter is None:
            raise ValueError(f"{name}: scene_filter is required")
        sources.append(_source(
            str(name),
            Path(str(_get(split, "synthetic_log_root"))) / str(source_split),
            Path(str(_get(split, "synthetic_sensor_root"))) / str(source_split),
            _copy_filter(source_filter), _get(source_cfg, "expected_num_samples"),
            source_split, bool(_get(source_cfg, "require_invalid_tokens", False)),
            _get(source_cfg, "invalid_tokens_sha256"),
        ))
    metric_root = _get(cfg, "synthetic_metric_cache_root")
    if metric_root is not None:
        for source in sources:
            if source["name"] != "real":
                source["metric_cache_path"] = str(
                    Path(str(metric_root)) / ("train_metric_" + source["data_split"])
                )
    expected_total = _count(_get(split, "expected_total_train_samples"))
    if expected_total is not None and all(s["expected_count"] is not None for s in sources):
        declared_total = sum(s["expected_count"] for s in sources)
        if declared_total != expected_total:
            raise RuntimeError(
                f"Declared source counts disagree with expected_total_train_samples: "
                f"{declared_total} != {expected_total}"
            )
    return sources


def select_sources(
    sources: list[dict[str, Any]], source_names: str | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select names or aliases (all, real, synthetic), preserving split order."""
    if source_names is None:
        names = ["all"]
    elif isinstance(source_names, str):
        names = [name.strip() for name in source_names.split(",") if name.strip()]
    else:
        names = list(source_names)
    if not names:
        raise ValueError("Source selection is empty")
    available = {source["name"] for source in sources}
    selected: set[str] = set()
    for name in names:
        if name == "all":
            selected.update(available)
        elif name == "synthetic":
            selected.update(available - {"real"})
        elif name in available:
            selected.add(name)
        else:
            raise ValueError(f"Unknown source {name!r}; available sources: {sorted(available)}")
    if not selected:
        raise ValueError(f"No sources match selection: {names}")
    return [source for source in sources if source["name"] in selected]


def iter_source_samples(source: Mapping[str, Any]) -> Iterator[dict[str, str]]:
    """Yield deterministic sample identities using a no-sensor SceneLoader.

    ``log_file`` is SceneLoader's log identifier (without ``.pkl``), suitable
    for ``scene_filter.log_names``. ``log_name`` matches SceneMetadata and the
    feature-cache directory. No cameras, lidar or VLM weights are loaded.
    """
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    scene_filter = instantiate(source["scene_filter"])
    if source.get("metric_cache_path"):
        from navsim.common.pdm_csv_scene_loader import PdmCsvSceneLoader

        loader = PdmCsvSceneLoader(
            data_path=Path(source["data_path"]),
            sensor_blobs_path=Path(source["sensor_blobs_path"]),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
            metric_cache_path=Path(source["metric_cache_path"]),
        )
        tokens = sorted(loader.tokens, key=lambda token: (loader.token_to_log_file[token], token))
        expected = source["expected_count"]
        if expected is not None and len(tokens) != int(expected):
            raise RuntimeError(f"{source['name']}: sample count changed: {len(tokens)} != {expected}")
        for token in tokens:
            log = str(loader.token_to_log_file[token])
            yield {"source": str(source["name"]), "log_name": log,
                   "token": token, "log_file": log}
        return
    loader = SceneLoader(
        data_path=Path(source["data_path"]),
        sensor_blobs_path=Path(source["sensor_blobs_path"]),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    try:
        tokens = sorted(loader.tokens, key=lambda token: (loader.token_to_log_file[token], token))
        expected_count = source["expected_count"]
        if expected_count is not None and len(tokens) != int(expected_count):
            raise RuntimeError(
                f"{source['name']}: sample count changed: {len(tokens)} != {expected_count}"
            )
        if len(tokens) != len(set(tokens)):
            raise RuntimeError(f"{source['name']}: duplicate token returned by SceneLoader")
        seen_keys: set[tuple[str, str]] = set()
        records: list[dict[str, str]] = []
        for token in tokens:
            frame = loader.scene_frames_dicts[token][scene_filter.num_history_frames - 1]
            initial_token, log_name = str(frame["token"]), str(frame["log_name"])
            if initial_token != str(token):
                raise RuntimeError(f"{source['name']}: token/initial-token mismatch: {token}/{initial_token}")
            key = (log_name, initial_token)
            if key in seen_keys:
                raise RuntimeError(f"{source['name']}: duplicate cache key: {key}")
            seen_keys.add(key)
            records.append({
                "source": str(source["name"]), "log_name": log_name,
                "token": initial_token, "log_file": str(loader.token_to_log_file[token]),
            })
        frame = None
    finally:
        # Drop large scene_frames_dicts before opening the next source.
        del loader
    # Consumers may construct a sensor-enabled loader while iterating. Retain
    # only lightweight identities during their model work, not all raw frames.
    yield from records


def iter_samples(
    cfg: Any, source_names: str | Iterable[str] | None = None,
) -> Iterator[dict[str, str]]:
    """Stream selected sources, rejecting token/cache-key collisions.

    A selection of synthetic sources checks their counts, without comparing
    against the full real+synthetic expected total. Exhaust this iterator to
    perform the final total check. SceneLoader itself is token-keyed, so a
    duplicate already overwritten inside a raw-log loader is outside this API's
    visibility; configured expected counts still guard against lost samples.
    """
    all_sources = resolve_sources(cfg)
    sources = select_sources(all_sources, source_names)
    seen_tokens: dict[str, str] = {}
    seen_keys: set[tuple[str, str]] = set()
    count = 0
    for source in sources:
        records = iter_source_samples(source)
        try:
            for record in records:
                token = record["token"]
                key = (record["log_name"], token)
                if token in seen_tokens or key in seen_keys:
                    raise RuntimeError(
                        f"Cross-source token/cache-key collision: {key}, "
                        f"sources={seen_tokens.get(token)}/{record['source']}"
                    )
                seen_tokens[token] = record["source"]
                seen_keys.add(key)
                count += 1
                yield record
        finally:
            records.close()
    if len(sources) == len(all_sources):
        expected = _count(_get(_get(cfg, "train_test_split"), "expected_total_train_samples"))
        if expected is not None and count != expected:
            raise RuntimeError(f"Mixed training sample count changed: {count} != {expected}")
