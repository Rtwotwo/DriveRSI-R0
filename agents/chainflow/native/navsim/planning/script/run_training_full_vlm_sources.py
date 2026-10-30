#!/usr/bin/env python3
"""Source-routed, legacy-compatible VLM-cache preflight for V6 Stage2.

The ordinary training entrypoint still constructs every dataset and owns the
model, losses, optimizer, sampler and Trainer.  Only its dataset-builder return
is wrapped: each SceneLoader-backed child is bound to its own VLM cache root,
then checked before any training DataLoader can consume it.  There is no overlay
and no change to existing Dataset/InternVL/training source files.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
EXPECTED_TRAIN_SAMPLES = {
    "navtrain": 85_109,
    "navtrainval": 103_288,
    "navtrain_simscale": 308_119,
}
EXPECTED_PDM_SAMPLES = (33_000, 55_471, 46_261, 37_897, 32_202)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "on"}:
            return True
        if normalised in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
    return bool(value)


def _runtime_choice(group: str) -> str:
    if not HydraConfig.initialized():
        return ""
    value = HydraConfig.get().runtime.choices.get(group)
    return str(value) if value is not None else ""


def _configured_split(cfg: DictConfig) -> str:
    runtime_split = _runtime_choice("train_test_split")
    requested_split = os.environ.get("CACHE_TRAIN_SPLIT", "").strip()
    if runtime_split and requested_split and runtime_split != requested_split:
        raise ValueError(
            f"Hydra selected {runtime_split!r}, but CACHE_TRAIN_SPLIT="
            f"{requested_split!r}; refusing a split/routing mismatch"
        )
    split_name = runtime_split or requested_split
    if split_name not in EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            "Select train_test_split (or CACHE_TRAIN_SPLIT for programmatic use) "
            f"from {sorted(EXPECTED_TRAIN_SAMPLES)}, got {split_name!r}"
        )
    from navsim.planning.training.log_partitions import resolve_training_log_names

    source_cfg = cfg.train_test_split
    selected_real_logs = set(resolve_training_log_names(cfg))
    official_train_logs = set(cfg.train_logs)
    official_trainval_logs = official_train_logs | set(cfg.val_logs)
    synthetic = source_cfg.get("synthetic_sources", {}) or {}
    expected_sources = [f"pdm_{index}" for index in range(5)]
    if split_name == "navtrain":
        valid = selected_real_logs == official_train_logs and not synthetic
    elif split_name == "navtrainval":
        valid = selected_real_logs == official_trainval_logs and not synthetic
    else:
        valid = (
            selected_real_logs == official_trainval_logs
            and list(synthetic.keys()) == expected_sources
        )
    if not valid:
        raise ValueError(
            f"{split_name} disagrees with resolved split structure: "
            f"selected_real_logs={len(selected_real_logs)}, "
            f"synthetic_sources={list(synthetic.keys())}"
        )
    if str(source_cfg.get("data_split", "")) != "trainval":
        raise ValueError("These three training splits must use real data_split=trainval")
    if synthetic:
        for index, expected_count in enumerate(EXPECTED_PDM_SAMPLES):
            source = synthetic[f"pdm_{index}"]
            if str(source.data_split) != f"synthetic_reaction_pdm_v1.0-{index}":
                raise ValueError(f"pdm_{index} is bound to the wrong physical shard")
            if int(source.get("expected_num_samples", -1)) != expected_count:
                raise ValueError(f"pdm_{index} has an unexpected configured population")
    expected_total = source_cfg.get("expected_total_train_samples")
    if expected_total is not None and int(expected_total) != EXPECTED_TRAIN_SAMPLES[split_name]:
        raise ValueError("Split expected_total_train_samples disagrees with its selected name")
    return split_name


def _guard_config(cfg: DictConfig) -> tuple[str, Path, Path]:
    """Validate selectors only; this does not certify any sample's coverage."""
    split_name = _configured_split(cfg)
    selected_agent = _runtime_choice("agent")
    if selected_agent and selected_agent != "chainflow_mot_v6_stage2":
        raise ValueError("This entrypoint requires agent=chainflow_mot_v6_stage2")
    target = str(OmegaConf.select(cfg, "agent._target_", default=""))
    if target != "navsim.agents.chainflow_mot_v6.chainflow_vla_agent.ChainFlowVLAAgent":
        raise ValueError(f"Unexpected Stage2 agent implementation: {target!r}")
    if _as_bool(cfg.get("use_cache_without_dataset", False)):
        raise ValueError("use_cache_without_dataset=true bypasses source routing and is forbidden")
    if str(cfg.get("cache_mode", "")) != "read_through":
        raise ValueError("Set cache_mode=read_through to retain the existing lazy feature/target reader")
    if _as_bool(cfg.get("force_cache_computation", False)):
        raise ValueError("force_cache_computation must be false; existing caches must not be overwritten")
    if not _as_bool(OmegaConf.select(cfg, "agent.config.use_vlm_feature_cache", default=False)):
        raise ValueError("Stage2 source routing requires use_vlm_feature_cache=true")
    if _as_bool(OmegaConf.select(cfg, "agent.config.use_online_vlm", default=False)):
        raise ValueError("Online VLM inference is incompatible with this cache-only VLM contract")
    if not _as_bool(OmegaConf.select(cfg, "agent.config.use_diffusion_decoder", default=False)):
        raise ValueError("The Stage2 diffusion decoder must be enabled")
    exact_fields = {
        "agent.config.vlm_feature_cache_format": "gz",
        "agent.config.vlm_feature_cache_name": "internvl_feature",
        "agent.config.vlm_hidden_key": "last_hidden_state",
        "agent.config.diffusion_vlm_hidden_dim": 1536,
        "agent.scheduler_args.dataset_size": EXPECTED_TRAIN_SAMPLES[split_name],
    }
    for key, expected in exact_fields.items():
        observed = OmegaConf.select(cfg, key, default=None)
        if observed != expected:
            raise ValueError(f"{key} must be {expected!r}; got {observed!r}")
    fallbacks = OmegaConf.select(cfg, "agent.config.vlm_feature_cache_fallback_paths", default=[]) or []
    if list(fallbacks):
        raise ValueError("Fallback roots are forbidden; every source has one unambiguous cache root")
    if cfg.get("train_ckpt_path") is not None:
        raise ValueError("Use Stage1 agent.checkpoint_path; this launcher does not resume Lightning state")
    checkpoint = str(OmegaConf.select(cfg, "agent.checkpoint_path", default="") or "")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("Set agent.checkpoint_path / STAGE1_CKPT_PATH to an existing Stage1 checkpoint")
    requested_checkpoint = os.environ.get("STAGE1_CKPT_PATH", "").strip()
    if requested_checkpoint and Path(checkpoint).resolve() != Path(requested_checkpoint).resolve():
        raise ValueError("Resolved agent.checkpoint_path differs from STAGE1_CKPT_PATH")
    exp_root = os.environ.get("NAVSIM_EXP_ROOT", "").strip()
    cache_base_value = os.environ.get("VLM_CACHE_BASE_PATH", "").strip()
    if not cache_base_value and not exp_root:
        raise ValueError("Set NAVSIM_EXP_ROOT or VLM_CACHE_BASE_PATH explicitly")
    cache_base = Path(cache_base_value) if cache_base_value else Path(exp_root) / "vlm_feature_cache"
    real_root = Path(os.environ.get("REAL_VLM_FEATURE_CACHE_PATH", "").strip() or str(cache_base / "navtrain_vlm_feature_cache"))
    configured_root = str(OmegaConf.select(cfg, "agent.config.vlm_feature_cache_path", default="") or "")
    if not configured_root or Path(configured_root).resolve() != real_root.resolve():
        raise ValueError("agent.config.vlm_feature_cache_path must equal REAL_VLM_FEATURE_CACHE_PATH")
    if not cache_base.is_dir() or not real_root.is_dir():
        raise FileNotFoundError(f"Cache base or real reference root missing: {cache_base}, {real_root}")
    return split_name, cache_base, real_root


def _checked_builder(original_builder: Any, split_name: str, cache_base: Path, real_root: Path):
    def checked(cfg: DictConfig, agent: Any):
        # Keep the base SceneLoader population, sensor config and raw-cache reader.
        train_data, val_data = original_builder(cfg, agent)
        from navsim.planning.training.vlm_stage2_compatibility import bind_and_validate_datasets

        report = bind_and_validate_datasets(
            train_data, val_data, cfg, split_name, cache_base, real_root
        )
        logger.info("Source-routed Stage2 VLM preflight passed: %s", report)
        return train_data, val_data

    return checked


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    split_name, cache_base, real_root = _guard_config(cfg)
    if _as_bool(os.environ.get("VLM_STAGE2_CHECK_ONLY", "0")):
        print(json.dumps({
            "status": "configuration_only_passed",
            "split": split_name,
            "expected_train_samples": EXPECTED_TRAIN_SAMPLES[split_name],
            "cache_base": str(cache_base),
            "real_cache_root": str(real_root),
            "ordinary_cache_path": str(cfg.cache_path) if cfg.get("cache_path") is not None else None,
            "cache_mode": cfg.cache_mode,
            "sample_coverage": "NOT_CHECKED",
            "mixed_batch": "NOT_CHECKED",
            "model_constructed": False,
        }, indent=2))
        return
    # Fail on missing per-source cache roots before model/SceneLoader creation.
    from navsim.planning.training.vlm_stage2_compatibility import source_cache_roots

    source_cache_roots(split_name, cache_base, real_root)
    # Delayed import lets --check-only avoid model/Trainer setup entirely.
    from navsim.planning.script import run_training_full as base

    original_builder = base.build_datasets
    base.build_datasets = _checked_builder(original_builder, split_name, cache_base, real_root)
    try:
        base.main.__wrapped__(cfg)
    finally:
        base.build_datasets = original_builder


if __name__ == "__main__":
    main()
