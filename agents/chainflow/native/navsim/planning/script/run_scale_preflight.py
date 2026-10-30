from __future__ import annotations

import logging
import lzma
import os
import pickle
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig
from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat
from navsim.common.pdm_csv_scene_loader import PdmCsvSceneLoader
from navsim.planning.training.log_partitions import resolve_training_log_names
from navsim.planning.training.scaling import build_scale_training_plan


logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_simscale"


install_numpy_2_pickle_compat()


def _read_metadata_index(cache_root: Path) -> Tuple[Dict[str, str], str]:
    """Read CSV text only; never stat every stale absolute path in the index."""

    metadata_files = sorted((cache_root / "metadata").glob("*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"Metric-cache metadata CSV is missing under {cache_root}")
    token_to_log: Dict[str, str] = {}
    first_path = ""
    for metadata_file in metadata_files:
        with metadata_file.open("r", encoding="utf-8") as handle:
            next(handle, None)
            for raw_line in handle:
                raw_path = raw_line.strip().split(",", 1)[0]
                if not raw_path:
                    continue
                if not first_path:
                    first_path = raw_path
                parts = PurePosixPath(raw_path).parts
                if cache_root.name not in parts:
                    raise RuntimeError(
                        f"Metadata path does not contain namespace {cache_root.name}: {raw_path}"
                    )
                namespace_index = len(parts) - 1 - parts[::-1].index(cache_root.name)
                relative = parts[namespace_index + 1 :]
                if len(relative) != 4 or relative[-1] != "metric_cache.pkl":
                    raise RuntimeError(f"Unexpected metric-cache metadata path: {raw_path}")
                log_name, _scenario_type, token, _filename = relative
                prior_log = token_to_log.setdefault(token, log_name)
                if prior_log != log_name:
                    raise RuntimeError(
                        f"Metric token {token} maps to both {prior_log} and {log_name}"
                    )
    if not first_path:
        raise RuntimeError(f"Metric-cache metadata is empty under {cache_root}")
    return token_to_log, first_path


def _relocate_indexed_path(cache_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    parts = list(PurePosixPath(raw_path).parts)
    if cache_root.name not in parts:
        raise FileNotFoundError(
            f"Cannot relocate stale metric-cache path without namespace {cache_root.name}: {raw_path}"
        )
    namespace_index = len(parts) - 1 - parts[::-1].index(cache_root.name)
    relocated = cache_root.joinpath(*parts[namespace_index + 1 :])
    if not relocated.is_file():
        raise FileNotFoundError(f"Indexed metric-cache sample is missing: {relocated}")
    return relocated


def _observation_horizon_s(metric_cache: Any) -> float:
    observation = metric_cache.observation
    samples = getattr(observation, "_observation_samples", None)
    interval = getattr(observation, "_sample_interval", None)
    if samples is None or interval is None:
        raise RuntimeError("Metric cache observation does not expose its temporal support")
    return float(samples) * float(interval)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Fast, data-read-only gate for a capability-aware scale run."""

    plan = build_scale_training_plan(cfg)
    if plan is None:
        raise RuntimeError("Scale preflight requires scale_training.enabled=true")
    cache_root_value = os.getenv("NAVSIM_METRIC_CACHE_ROOT")
    if not cache_root_value:
        raise RuntimeError("NAVSIM_METRIC_CACHE_ROOT must be set")
    cache_root = Path(cache_root_value)
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])

    actual_counts: Dict[str, int] = {}
    seen_tokens: Dict[str, str] = {}
    cross_namespace_collisions = 0
    for source_name, contract in plan.contracts.items():
        source_cache_root = cache_root / contract.metric_cache_name
        metadata_index, indexed_sample_path = _read_metadata_index(source_cache_root)
        if contract.is_synthetic:
            source_cfg = cfg.train_test_split.synthetic_sources[source_name]
            source_split = str(source_cfg.data_split)
            source_scene_filter = instantiate(source_cfg.scene_filter)
            source_scene_loader = PdmCsvSceneLoader(
                data_path=data_root / "navsim_logs" / source_split,
                sensor_blobs_path=data_root / "sensor_blobs" / source_split,
                scene_filter=source_scene_filter,
                sensor_config=SensorConfig.build_no_sensors(),
                metric_cache_path=source_cache_root,
                metadata_csv=source_cfg.get("metadata_csv"),
            )
            # This applies the same split/invalid-token policy as training and
            # verifies referenced raw-log filenames with one directory scan.
            source_tokens = set(source_scene_loader.tokens)
        else:
            source_scene_filter = instantiate(cfg.train_test_split.scene_filter)
            selected_real_logs = set(resolve_training_log_names(cfg))
            source_tokens = {
                token
                for token, log_name in metadata_index.items()
                if log_name in selected_real_logs
            }
        actual_counts[source_name] = len(source_tokens)
        sample_path = _relocate_indexed_path(source_cache_root, indexed_sample_path)
        with lzma.open(sample_path, "rb") as handle:
            metric_cache = pickle.load(handle)
        actual_horizon_s = _observation_horizon_s(metric_cache)
        tolerance_s = 1e-6
        if abs(actual_horizon_s - contract.observation_horizon_s) > tolerance_s:
            raise RuntimeError(
                f"{source_name}: metric-cache observation horizon {actual_horizon_s:.3f} s "
                f"differs from contract {contract.observation_horizon_s:.3f} s"
            )
        configured_future_frames = getattr(source_scene_filter, "num_future_frames", None)
        for requirement in plan.common_supervision.values():
            if (
                requirement.min_num_poses is not None
                and configured_future_frames is not None
                and int(configured_future_frames) < requirement.min_num_poses
            ):
                raise RuntimeError(
                    f"{source_name}: scene filter provides {configured_future_frames} future "
                    f"frames, below common supervision {requirement.name} requirement "
                    f"{requirement.min_num_poses}"
                )
        for token in source_tokens:
            prior_source = seen_tokens.setdefault(token, source_name)
            if prior_source != source_name:
                cross_namespace_collisions += 1
        logger.info(
            "Scale cache preflight: source=%s samples=%d horizon=%.1fs namespace=%s",
            source_name,
            len(source_tokens),
            actual_horizon_s,
            contract.metric_cache_name,
        )

    plan.validate_actual_counts(actual_counts)
    manifest_path = plan.write_manifest(Path(cfg.output_dir), actual_counts)
    manifest = plan.manifest(actual_counts)
    common_coverage = ", ".join(
        f"{name}={entry['satisfied_rate']:.6f}"
        for name, entry in manifest["common_supervision"].items()
    )
    logger.info(
        "Scale data preflight PASS: total=%d common_supervision=[%s] "
        "cross_namespace_token_collisions=%d",
        manifest["actual_total_samples"],
        common_coverage,
        cross_namespace_collisions,
    )
    logger.info("Scale manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
