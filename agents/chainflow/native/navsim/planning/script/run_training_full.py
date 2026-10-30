import hashlib
import os
import random
from typing import Optional, Tuple
from pathlib import Path
import logging
import pickle
import warnings
from datetime import datetime, timedelta

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from pytorch_lightning.strategies import DDPStrategy

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.collate import build_train_val_dataloaders
from navsim.planning.training.log_partitions import (
    resolve_cache_training_log_names,
    resolve_training_log_names,
)
from navsim.planning.training.scaling import (
    ScaleTrainingPlan,
    SourceAnnotatedDataset,
    build_scale_training_plan,
)

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in line_locate_point",
    category=RuntimeWarning,
    module=r"shapely\.linear",
)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def log_model_summary(cfg: DictConfig, agent: AbstractAgent) -> None:
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    total_param_size_mb = total_params * 4 / (1024 ** 2)

    logger.info("Model summary:")
    logger.info("  agent=%s", agent.__class__.__name__)
    logger.info("  total_params=%s", f"{total_params:,}")
    logger.info("  trainable_params=%s", f"{trainable_params:,}")
    logger.info("  approx_param_size_mb=%.2f", total_param_size_mb)
    logger.info(
        "  batch_size_per_gpu=%s | num_workers=%s | num_gpus=%s | max_epochs=%s",
        cfg.dataloader.params.batch_size,
        cfg.dataloader.params.num_workers,
        getattr(cfg.agent, "num_gpus", "unknown"),
        cfg.trainer.params.max_epochs,
    )

    agent_cfg = getattr(cfg.agent, "config", None)
    if agent_cfg is not None:
        key_fields = [
            "proposal_num",
            "num_poses",
            "ref_num",
            "tf_d_model",
            "tf_d_ffn",
            "trajectory_head_type",
            "one_token_per_traj",
            "ar_decoder_num_layers",
            "ar_use_bicycle_kinematics",
            "long_trajectory_additional_poses",
        ]
        summary_parts = [f"{key}={agent_cfg[key]}" for key in key_fields if key in agent_cfg]
        if summary_parts:
            logger.info("  config: %s", " | ".join(summary_parts))

def dist_ready():
    return dist.is_available() and dist.is_initialized()

def build_datasets(
    cfg: DictConfig,
    agent: AbstractAgent,
    scale_plan: Optional[ScaleTrainingPlan] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """

    cache_mode = cfg.get("cache_mode", "eager")
    synthetic_sources = cfg.train_test_split.get("synthetic_sources", {})
    synthetic_scene_loader = cfg.get("synthetic_scene_loader", "eager")
    synthetic_metric_cache_root = None
    # This opt-in affects synthetic sources only. Real training and validation
    # continue to use the original SceneLoader, including non-mixed runs.
    if synthetic_sources and synthetic_scene_loader != "eager":
        if synthetic_scene_loader != "metric_csv":
            raise ValueError(f"Unknown synthetic_scene_loader: {synthetic_scene_loader}")
        if (
            cache_mode != "eager"
            or cfg.cache_path is not None
            or cfg.get("synthetic_cache_root") is not None
            or cfg.force_cache_computation
            or any(source.get("cache_path") is not None for source in synthetic_sources.values())
        ):
            raise ValueError("metric_csv requires cache_mode=eager, all feature/target cache paths=null, and force_cache_computation=false")
        for flag in ("use_vlm_feature_cache", "use_cosmos_future_cache", "use_diffusion_decoder", "train_diffusion_decoder_only"):
            if OmegaConf.select(cfg, f"agent.config.{flag}", default=False):
                raise ValueError(f"metric_csv is Stage1-only and requires agent.config.{flag}=false")
        configured_root = cfg.get("synthetic_metric_cache_root", os.getenv("NAVSIM_METRIC_CACHE_ROOT"))
        if not configured_root:
            raise ValueError("metric_csv requires NAVSIM_METRIC_CACHE_ROOT or synthetic_metric_cache_root")
        synthetic_metric_cache_root = Path(configured_root)
        # Fail before loading real data if an index is missing. Do not silently
        # fall back to hours of eager PDM loading or rebuild any cache.
        for source_name, source_cfg in synthetic_sources.items():
            cache_name = f"train_metric_{source_cfg.data_split}"
            metadata_csv = source_cfg.get("metadata_csv")
            metadata_path = (
                Path(metadata_csv) if metadata_csv is not None
                else synthetic_metric_cache_root / cache_name / "metadata" / f"{cache_name}_metadata_node_0.csv"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(f"{source_name}: PDM metadata CSV is missing: {metadata_path}")
    dataset_type = Dataset
    if cache_mode == "read_through":
        from navsim.planning.training.read_through_dataset import ReadThroughDataset

        dataset_type = ReadThroughDataset
    elif cache_mode != "eager":
        raise ValueError(f"Unknown cache_mode: {cache_mode}")
    logger.info("Building datasets: cache_mode=%s real_cache=%s", cache_mode, cfg.cache_path)
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    train_log_names = resolve_training_log_names(cfg)
    allowed_train_logs = set(train_log_names)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names
            if log_name in allowed_train_logs
        ]
    else:
        train_scene_filter.log_names = train_log_names
    

    logger.info("Number of training logs: %s", len(train_scene_filter.log_names))

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = dataset_type(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = dataset_type(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    expected_real = cfg.train_test_split.get("expected_real_train_samples")
    if expected_real is not None and len(train_data) != int(expected_real):
        raise RuntimeError(
            "Real training sample count changed: "
            f"{len(train_data)} != {int(expected_real)}"
        )

    actual_scale_counts = {"real_train": len(train_data)}
    if scale_plan is not None:
        train_data = SourceAnnotatedDataset(
            train_data, scale_plan.source("real_train"), scale_plan
        )

    train_datasets = [train_data]
    for source_name, source_cfg in synthetic_sources.items():
        source_split = str(source_cfg.data_split)
        source_data_path = (
            Path(cfg.train_test_split.synthetic_log_root) / source_split
        )
        source_sensor_path = (
            Path(cfg.train_test_split.synthetic_sensor_root) / source_split
        )
        if not source_data_path.is_dir():
            raise FileNotFoundError(
                f"{source_name}: synthetic log directory is missing: "
                f"{source_data_path}"
            )
        if not source_sensor_path.is_dir():
            raise FileNotFoundError(
                f"{source_name}: synthetic sensor directory is missing: "
                f"{source_sensor_path}"
            )

        invalid_tokens_path = source_sensor_path / "invalid_tokens.yaml"
        if (
            bool(source_cfg.get("require_invalid_tokens", False))
            and not invalid_tokens_path.is_file()
        ):
            raise FileNotFoundError(
                f"{source_name}: required invalid-token manifest is missing: "
                f"{invalid_tokens_path}. Refusing to use the unfiltered shard."
            )

        expected_manifest_hash = source_cfg.get("invalid_tokens_sha256")
        if expected_manifest_hash is not None:
            actual_manifest_hash = hashlib.sha256(
                invalid_tokens_path.read_bytes()
            ).hexdigest()
            if actual_manifest_hash != str(expected_manifest_hash):
                raise RuntimeError(
                    f"{source_name}: invalid-token manifest SHA256 changed: "
                    f"{actual_manifest_hash} != {expected_manifest_hash}"
                )

        source_scene_filter: SceneFilter = instantiate(source_cfg.scene_filter)
        if synthetic_scene_loader == "metric_csv":
            from navsim.common.pdm_csv_scene_loader import PdmCsvSceneLoader

            source_scene_loader = PdmCsvSceneLoader(
                sensor_blobs_path=source_sensor_path,
                data_path=source_data_path,
                scene_filter=source_scene_filter,
                sensor_config=agent.get_sensor_config(),
                metric_cache_path=synthetic_metric_cache_root / f"train_metric_{source_split}",
                metadata_csv=source_cfg.get("metadata_csv"),
            )
        else:
            source_scene_loader = SceneLoader(
                sensor_blobs_path=source_sensor_path,
                data_path=source_data_path,
                scene_filter=source_scene_filter,
                sensor_config=agent.get_sensor_config(),
            )
        source_cache_path = source_cfg.get("cache_path", cfg.cache_path)
        synthetic_cache_root = cfg.get("synthetic_cache_root")
        if "cache_path" not in source_cfg and synthetic_cache_root is not None:
            source_cache_path = str(
                Path(synthetic_cache_root) / f"training_cache_{source_split}"
            )
        logger.info("Synthetic source %s: feature/target cache=%s", source_name, source_cache_path)
        source_dataset = dataset_type(
            scene_loader=source_scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=source_cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )

        expected_source = source_cfg.get("expected_num_samples")
        if expected_source is not None and len(source_dataset) != int(expected_source):
            raise RuntimeError(
                f"{source_name}: synthetic sample count changed: "
                f"{len(source_dataset)} != {int(expected_source)}"
            )
        logger.info(
            "Synthetic source %s: split=%s samples=%d",
            source_name,
            source_split,
            len(source_dataset),
        )
        actual_scale_counts[source_name] = len(source_dataset)
        if scale_plan is not None:
            source_dataset = SourceAnnotatedDataset(
                source_dataset, scale_plan.source(source_name), scale_plan
            )
        train_datasets.append(source_dataset)

    if len(train_datasets) > 1:
        train_data = ConcatDataset(train_datasets)

    expected_total = cfg.train_test_split.get("expected_total_train_samples")
    if expected_total is not None and len(train_data) != int(expected_total):
        raise RuntimeError(
            "Mixed training sample count changed: "
            f"{len(train_data)} != {int(expected_total)}"
        )

    if scale_plan is not None:
        scale_plan.validate_actual_counts(actual_scale_counts)
        manifest = scale_plan.manifest(actual_scale_counts)
        common_coverage = ", ".join(
            f"{name}={entry['satisfied_rate']:.6f}"
            for name, entry in manifest["common_supervision"].items()
        )
        logger.info(
            "Scale data contract passed: protocol=%s total=%d common_supervision=[%s]",
            scale_plan.protocol,
            len(train_data),
            common_coverage,
        )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig, *, cache_dataset_factory=None, auto_resume_search=True, trainer_logger=None) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    if os.getenv("DEBUG") == "True":
        import debugpy

        debugpy_host = os.getenv("DEBUGPY_HOST", "0.0.0.0")
        debugpy_port = int(os.getenv("DEBUGPY_PORT", "5678"))
        debugpy.listen((debugpy_host, debugpy_port))
        logger.info("Waiting for debugger attach on %s:%s...", debugpy_host, debugpy_port)
        debugpy.wait_for_client()
        logger.info("Debugger attached.")
    # Set NAVSIM_LOG_LEVEL=DEBUG to enable debug logs from modules such as chainflow_vla_model.
    _lvl_name = os.getenv("NAVSIM_LOG_LEVEL", "").strip().upper()
    if _lvl_name:
        _level = getattr(logging, _lvl_name, None)
        if _level is not None:
            logging.getLogger().setLevel(_level)
            logging.getLogger("navsim").setLevel(_level)
            for _h in logging.getLogger().handlers:
                _h.setLevel(_level)

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    if bool(cfg.train_test_split.get("require_exact_scheduler_dataset_size", False)):
        expected_size = cfg.train_test_split.get("expected_total_train_samples")
        scheduler_size = OmegaConf.select(cfg, "agent.scheduler_args.dataset_size", default=None)
        if expected_size is None or scheduler_size != expected_size:
            raise ValueError(
                "This split requires DATA_SIZE / agent.scheduler_args.dataset_size "
                f"to equal its full training population ({expected_size}); got {scheduler_size}. "
                "Use the real-plus-synthetic total, not the synthetic subset size."
            )

    # Missing cfg.scale_training means the legacy path. The data-only plan is
    # enabled only by the dedicated SimScale primary config.
    scale_plan = build_scale_training_plan(cfg)
    if scale_plan is not None:
        logger.info(
            "Capability-aware scale data enabled: protocol=%s sources=%s",
            scale_plan.protocol,
            ",".join(scale_plan.contracts),
        )

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    # Optional agent-specific input/config contract; training remains shared.
    configure_training_run = getattr(agent, "configure_training_run", None)
    if callable(configure_training_run):
        configure_training_run(cfg)
    logger.info("Initializing Agent (load pretrained checkpoint if configured)")
    agent.initialize()
    log_model_summary(cfg, agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    synthetic_sources = cfg.train_test_split.get("synthetic_sources", {})
    if cfg.use_cache_without_dataset and len(synthetic_sources) > 0:
        raise RuntimeError(
            "Composite synthetic splits require use_cache_without_dataset=false; "
            "the cache-only path cannot represent multiple physical data roots."
        )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        # Agent-local launchers may opt in; ordinary and synthetic runs keep
        # their existing dataset construction and split/validation semantics.
        cache_dataset = cache_dataset_factory or CacheOnlyDataset
        # New explicit partitions (currently navval) must also be honoured by
        # the cache-only path.  Splits without this field retain the historical
        # cfg.train_logs behavior exactly, including legacy navtrainval jobs.
        cache_train_log_names = resolve_cache_training_log_names(cfg)
        train_data = cache_dataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cache_train_log_names,
        )
        val_data = cache_dataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
        expected_real = cfg.train_test_split.get("expected_real_train_samples")
        if expected_real is not None and len(train_data) != int(expected_real):
            raise RuntimeError(
                "Cached real training sample count changed: "
                f"{len(train_data)} != {int(expected_real)}"
            )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent, scale_plan=scale_plan)

    if scale_plan is not None:
        # build_datasets already compared every actual length with the data contract.
        # Persist the resolved protocol under the experiment, never under data/cache.
        process_rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
        if process_rank == 0:
            actual_counts = {
                name: contract.expected_samples
                for name, contract in scale_plan.contracts.items()
            }
            manifest_path = scale_plan.write_manifest(Path(cfg.output_dir), actual_counts)
            logger.info("Scale manifest: %s", manifest_path)

    expected_total = cfg.train_test_split.get("expected_total_train_samples")
    if expected_total is not None:
        scheduler_size = OmegaConf.select(
            cfg, "agent.scheduler_args.dataset_size", default=None
        )
        logger.info(
            "Mixed training samples=%d; LR scheduler dataset_size=%s "
            "(schedule parameter only; all samples remain in the dataset)",
            len(train_data),
            scheduler_size,
        )

    scorer_online_validation = bool(
        OmegaConf.select(
            cfg,
            "agent.config.stage3_scorer_online_validation",
            default=False,
        )
    )
    if scorer_online_validation:
        train_logs = set(resolve_training_log_names(cfg))
        val_logs = set(cfg.val_logs)
        calibration_logs = set(cfg.get("calibration_logs", []))
        if not train_logs or not val_logs or train_logs & val_logs:
            raise RuntimeError(
                "Scorer Safe-CVPO train/validation log partitions are empty or overlap."
            )
        if train_logs & calibration_logs or val_logs & calibration_logs:
            raise RuntimeError(
                "Scorer Safe-CVPO calibration logs overlap train/validation logs."
            )
        expected_train = int(cfg.agent.scheduler_args.dataset_size)
        expected_val = int(cfg.agent.scheduler_args.validation_size)
        # The recording-disjoint validation partition contains 5,211 scenes.
        # Eight-card DistributedSampler would duplicate its first five scenes.
        # Withhold exactly three deterministic tail entries (not reassigned to
        # train/calibration) so the certified validation set is 5,208 and every
        # held-out token is evaluated exactly once globally.
        configured_world_size = int(cfg.agent.num_gpus)
        withheld_count = len(val_data) - expected_val
        if (
            withheld_count == len(val_data) % configured_world_size
            and withheld_count > 0
            and expected_val % configured_world_size == 0
        ):
            logger.info(
                "Scorer Safe-CVPO validation: deterministically withholding "
                "the final %d/%d scenes to prevent %d-card sampler padding.",
                withheld_count,
                len(val_data),
                configured_world_size,
            )
            val_data = torch.utils.data.Subset(
                val_data, range(expected_val)
            )
        if len(train_data) != expected_train or len(val_data) != expected_val:
            raise RuntimeError(
                "Scorer Safe-CVPO split cardinality changed; refusing an "
                "uncertified LR schedule or validation set. "
                f"train={len(train_data)}/{expected_train}, "
                f"val={len(val_data)}/{expected_val}."
            )

    configure_dataset_sizes = getattr(agent, "configure_dataset_sizes", None)
    if callable(configure_dataset_sizes):
        if bool(OmegaConf.select(cfg, "agent.config.teacher_development_validation", default=False)):
            development_logs = set(resolve_training_log_names(cfg))
            validation_logs = set(cfg.val_logs)
            if not development_logs or not validation_logs or development_logs & validation_logs:
                raise RuntimeError("Teacher development train/validation logs must be disjoint and nonempty")
        configure_dataset_sizes(len(train_data), len(val_data))

    logger.info("Building Datasets")
    train_dataloader, val_dataloader = build_train_val_dataloaders(train_data, val_data, cfg, agent)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    if cfg.train_ckpt_path is None and auto_resume_search and cfg.get("auto_resume_search", True):
        # Pretrain-only flow: load AR (etc.) via agent.initialize() from agent.checkpoint_path with
        # strict=False. Do not also auto-pick a Lightning ckpt — full trainer restore conflicts with
        # new diffusion submodule keys and breaks or masks partial init.
        _agent_ckpt = OmegaConf.select(cfg, "agent.checkpoint_path", default=None)
        if _agent_ckpt and str(_agent_ckpt).strip():
            logger.info(
                "agent.checkpoint_path is set (%s); skipping auto train_ckpt_path lookup. "
                "Use train_ckpt_path=... or TRAIN_RESUME_CKPT_PATH only for full Lightning resume.",
                _agent_ckpt,
            )
        else:
            # Pattern to match all .ckpt files in the base_path recursively
            search_pattern = (
                "/".join(str(cfg.output_dir).split("/")[:-1])
                + "/*/lightning_logs/version_*/checkpoints/"
                + "*.ckpt"
            )
            logger.info("Searching latest training checkpoint from pattern: %s", search_pattern)
            cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
            logger.info("Auto-resolved train_ckpt_path: %s", cfg.train_ckpt_path)

    if cfg.train_ckpt_path is not None:
        logger.info("Trainer will restore from ckpt_path: %s", cfg.train_ckpt_path)
        if os.path.exists(cfg.train_ckpt_path):
            try:
                checkpoint = torch.load(cfg.train_ckpt_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict", {})
                logger.info(
                    "Checkpoint readable: state_dict_keys=%d, epoch=%s, global_step=%s",
                    len(state_dict),
                    checkpoint.get("epoch", "unknown"),
                    checkpoint.get("global_step", "unknown"),
                )
            except Exception as exc:
                logger.warning("Failed to inspect checkpoint file before trainer restore: %s", exc)
        else:
            logger.warning("Configured ckpt_path does not exist: %s", cfg.train_ckpt_path)
    else:
        logger.info("Trainer will start without ckpt_path (fresh init unless agent loads its own checkpoint).")

    trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)
    if not isinstance(trainer_params, dict):
        raise TypeError("trainer.params must resolve to a mapping.")

    agent_cfg = getattr(cfg.agent, "config", None)
    use_static_graph_ddp = bool(
        getattr(agent_cfg, "stage3_ddp_static_graph", False)
        if agent_cfg is not None
        else False
    )
    if use_static_graph_ddp:
        configured_strategy = str(trainer_params.get("strategy", "ddp"))
        supported_ddp_aliases = {
            "ddp",
            "ddp_find_unused_parameters_false",
            "ddp_find_unused_parameters_true",
        }
        if configured_strategy not in supported_ddp_aliases:
            raise ValueError(
                "stage3_ddp_static_graph requires a DDP strategy, got "
                f"{configured_strategy!r}."
            )
        trainer_params["strategy"] = DDPStrategy(
            static_graph=True,
            find_unused_parameters=False,
        )
        logger.info(
            "Stage3 DDP strategy: static_graph=True, "
            "find_unused_parameters=False"
        )
    elif os.getenv("CHAINFLOW_DDP_TIMEOUT_SECONDS", "").strip():
        timeout_raw = os.environ["CHAINFLOW_DDP_TIMEOUT_SECONDS"].strip()
        try:
            timeout_seconds = int(timeout_raw)
        except ValueError as error:
            raise ValueError(
                "CHAINFLOW_DDP_TIMEOUT_SECONDS must be a positive integer; "
                f"got {timeout_raw!r}"
            ) from error
        if timeout_seconds <= 0:
            raise ValueError(
                "CHAINFLOW_DDP_TIMEOUT_SECONDS must be positive; "
                f"got {timeout_seconds}"
            )

        configured_strategy = str(trainer_params.get("strategy", "ddp"))
        ddp_strategy_kwargs = {
            "ddp": {},
            "ddp_find_unused_parameters_false": {
                "find_unused_parameters": False,
            },
            "ddp_find_unused_parameters_true": {
                "find_unused_parameters": True,
            },
        }
        if configured_strategy not in ddp_strategy_kwargs:
            raise ValueError(
                "CHAINFLOW_DDP_TIMEOUT_SECONDS requires a DDP strategy, got "
                f"{configured_strategy!r}."
            )
        trainer_params["strategy"] = DDPStrategy(
            timeout=timedelta(seconds=timeout_seconds),
            **ddp_strategy_kwargs[configured_strategy],
        )
        logger.info("DDP process-group timeout=%d seconds", timeout_seconds)

    if trainer_logger is not None:
        trainer_params["logger"] = trainer_logger
    trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
