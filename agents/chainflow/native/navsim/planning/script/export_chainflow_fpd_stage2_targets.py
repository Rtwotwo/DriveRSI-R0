"""Export fixed Stage 2 supervision, sharded without padding; no student training.

The teacher is instantiated from its saved configuration. This orchestration
script does not couple either Agent implementation to another Agent's imports.
"""
import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset
from navsim.planning.training.log_partitions import resolve_training_log_names
from navsim.agents.chainflow_fpd_native_student_stage2.scoring import NativePDMScoring
from navsim.agents.chainflow_fpd_native_student_stage2.target_cache import (
    SCHEMA_VERSION, HEAD_NAMES, METRIC_NAMES, cache_file, data_contract,
    file_sha256, json_hash, load_record, read_manifest,
)


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-scenes", type=int, default=103288)
    parser.add_argument("--precision", choices=("bf16-mixed", "32-true"), default="bf16-mixed")
    parser.add_argument("--disable-ray", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid loader configuration")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        # CPU synchronization; large exports may take hours. No GPU collective waits.
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    checkpoint = Path(args.checkpoint).resolve()
    teacher_hash = file_sha256(checkpoint)
    cfg_path = Path(args.training_config) if args.training_config else checkpoint.parents[3] / "resolved_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False).get("native_teacher_training", {})
    if metadata.get("agent") != "chainflow_fpd_native_teacher" or int(metadata.get("stage", -1)) != 2:
        raise ValueError("Only a frozen native Stage 2 teacher is supported")
    saved_config = metadata.get("contract", {}).get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Teacher checkpoint is missing its original config contract")
    supplied_config = OmegaConf.to_container(cfg.agent.config, resolve=True)
    # This budget does not affect prediction and was excluded by the teacher.
    supplied_config.pop("run_epoch_budget", None)
    changed = sorted(k for k in set(saved_config) | set(supplied_config)
                     if saved_config.get(k) != supplied_config.get(k))
    if changed:
        raise ValueError("Teacher snapshot differs from checkpoint config: " + ", ".join(changed))
    if int(cfg.agent.config.training_stage) != 2 or not cfg.agent.config.use_diffusion_decoder:
        raise ValueError("Teacher snapshot must describe Stage 2")
    if cfg.train_test_split.get("synthetic_sources"):
        raise ValueError("This export supports the real navtrainval protocol")
    # Only export/runtime overrides below; future inputs, sampler, score weights
    # and network settings must match the checkpoint contract checked above.
    cfg.agent.checkpoint_path = str(checkpoint)
    cfg.agent.stage1_checkpoint_path = ""
    cfg.agent.checkpoint_strict_load = True
    cfg.agent.enable_ray = False
    cfg.agent.config.teacher_development_validation = False
    teacher = instantiate(cfg.agent)
    teacher.initialize()
    if file_sha256(checkpoint) != teacher_hash:
        raise RuntimeError("Teacher checkpoint changed while loading; use an immutable epoch checkpoint")
    teacher.eval().requires_grad_(False).to(device)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    train_logs = set(resolve_training_log_names(cfg))
    scene_filter.log_names = sorted(train_logs if scene_filter.log_names is None else train_logs.intersection(scene_filter.log_names))
    loader = SceneLoader(data_path=Path(cfg.navsim_log_path), sensor_blobs_path=Path(cfg.sensor_blobs_path),
                         scene_filter=scene_filter, sensor_config=teacher.get_sensor_config())
    tokens = list(loader.tokens)
    if len(tokens) != args.expected_scenes or len(set(tokens)) != len(tokens):
        raise ValueError(f"Expected exactly {args.expected_scenes} unique training scenes, got {len(tokens)}")
    source = Path(__file__).resolve().parents[2] / "agents" / "chainflow_fpd_native_teacher"
    pdm_source = Path(__file__).resolve().parents[2] / "agents/chainflow_fpd_native_student_stage2/utils/pdm_score.py"
    def normalized_source(path):
        return "\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip()
    if normalized_source(source / "utils/pdm_score.py") != normalized_source(pdm_source):
        raise ValueError("Teacher/student true-PDM implementations differ; audit label semantics before export")
    source_hash = json_hash({str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob("*.py"))})
    identity = {
        "teacher_stage": 2, "trajectory_source": "diffusion",
        "schema_version": SCHEMA_VERSION, "metric_names": list(METRIC_NAMES),
        "teacher_sha256": teacher_hash, "teacher_source_sha256": source_hash,
        "teacher_checkpoint": str(checkpoint), "training_config_sha256": file_sha256(cfg_path),
        "data": data_contract(cfg), "tokens_sha256": json_hash(sorted(tokens)),
        "expected_scenes": len(tokens), "precision": args.precision,
        "proposal_num": int(cfg.agent.config.proposal_num), "num_poses": int(cfg.agent.config.num_poses),
        "teacher_config": OmegaConf.to_container(cfg.agent.config, resolve=True),
        "pdm_source_sha256": file_sha256(pdm_source),
    }
    root = Path(args.output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if (root / "manifest.json").exists():
            old = read_manifest(root, require_complete=False)
            if any(old.get(k) != v for k, v in identity.items()):
                raise ValueError("Output contains targets from a different checkpoint, source or data contract")
        else:
            write_json(root / "manifest.json", {**identity, "complete": False, "completed_scenes": 0})
            (root / "tokens.txt").write_text("\n".join(sorted(tokens)) + "\n")
    if world > 1:
        dist.barrier()
    dataset = Dataset(loader, teacher.get_feature_builders(), teacher.get_target_builders(), cache_path=None)
    # Sorted token ownership is stable across ranks; no DistributedSampler duplication.
    owned_tokens = sorted(tokens)[rank::world]
    index = {token: i for i, token in enumerate(tokens)}
    pending = []
    for token in owned_tokens:
        if cache_file(root, token).is_file():
            load_record(root, token, teacher_hash, identity["proposal_num"], identity["num_poses"])
        else:
            pending.append(index[token])
    dl = DataLoader(Subset(dataset, pending), batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, shuffle=False, drop_last=False)
    pdm = NativePDMScoring(cfg.agent.config.train_metric_cache_path, use_ray=not args.disable_ray)
    completed = len(owned_tokens) - len(pending)
    for batch_id, (features, targets) in enumerate(dl):
        current_tokens = list(targets["token"])
        features = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in features.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16-mixed"):
            prediction = teacher(features)
        if not prediction.get("score_uses_sampled", False):
            raise RuntimeError("Teacher export did not run the inference DiT sampler")
        poses = prediction["proposals"].detach().float()
        metrics = pdm(current_tokens, poses)
        arrays = {
            "proposals": poses.cpu().numpy(), "ar_proposals": prediction["ar_proposals"].detach().float().cpu().numpy(),
            "metrics": metrics.cpu().numpy(),
            "logits": torch.stack([prediction["pred_logit"][name].float() for name in HEAD_NAMES], dim=-1).cpu().numpy(),
            "rank_scores": prediction["pdm_score"].float().cpu().numpy(),
        }
        for row, token in enumerate(current_tokens):
            path = cache_file(root, token)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, token=np.asarray(token), teacher_sha256=np.asarray(teacher_hash),
                                    **{key: value[row] for key, value in arrays.items()})
            temporary.replace(path)
            load_record(root, token, teacher_hash, identity["proposal_num"], identity["num_poses"])
        completed += len(current_tokens)
        if batch_id % 10 == 0:
            logging.info("rank %d: %d/%d exported", rank, completed, len(owned_tokens))
    write_json(root / f"rank-{rank}-complete.json", {"teacher_sha256": teacher_hash, "count": completed, "world_size": world})
    if world > 1:
        dist.barrier()
    if rank == 0:
        if file_sha256(checkpoint) != teacher_hash:
            raise RuntimeError("Teacher checkpoint changed during export; cache remains incomplete")
        missing = [token for token in tokens if not cache_file(root, token).is_file()]
        if missing:
            raise RuntimeError(f"Export incomplete: {len(missing)} scenes missing")
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        write_json(root / "manifest.json", {**identity, "complete": True, "completed_scenes": len(tokens), "git_commit": git})
        logging.info("Completed frozen teacher targets: %s", root)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
