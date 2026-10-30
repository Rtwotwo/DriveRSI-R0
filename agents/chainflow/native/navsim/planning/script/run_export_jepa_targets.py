"""Export fixed JEPA visual supervision, without running the planning network."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Cameras, SensorConfig
from navsim.planning.training.log_partitions import resolve_training_log_names
from navsim.agents.chainflow_jepa_native_teacher.builders.feature_builder import ChainFlowVLAFeatureBuilder
from navsim.agents.chainflow_jepa_native_teacher.encoders.image.dinov2_lora import ImgEncoder
from navsim.agents.chainflow_jepa_native_teacher.model import ChainFlowJEPANativeTeacherModel
from navsim.agents.chainflow_jepa_native_teacher.target_cache import (
    CAMERA_ORDER, cache_config_metadata, target_encoder_fingerprint, token_list_digest,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, values):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_record(path, shape):
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["tokens"]:
            raise ValueError(f"Unexpected arrays in {path}")
        values = archive["tokens"]
    if values.dtype != np.float32 or list(values.shape) != shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid shape, dtype or non-finite values in {path}")


class FrozenTargets(torch.nn.Module):
    """Only the fixed image encoder and learned queries are instantiated."""
    def __init__(self, config, state):
        super().__init__()
        self._config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        self._config.jepa_target_cache_path = ""
        backbone_config = OmegaConf.create(OmegaConf.to_container(config.image_backbone, resolve=True))
        for key in ("image_size", "num_scene_tokens", "tf_d_model"):
            backbone_config[key] = config[key]
        self.target_image_backbone = ImgEncoder(backbone_config)
        prefix = "target_image_backbone."
        backbone_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not backbone_state or "target_scene_embeds" not in state:
            raise ValueError("Checkpoint lacks the complete frozen JEPA target encoder")
        self.target_image_backbone.load_state_dict(backbone_state, strict=True)
        queries = state["target_scene_embeds"].detach().clone()
        expected = (1, len(CAMERA_ORDER), int(config.num_scene_tokens), self.target_image_backbone.num_features)
        if tuple(queries.shape) != expected:
            raise ValueError(f"Unexpected target scene-query shape {tuple(queries.shape)} != {expected}")
        if "target_initialized" not in state or not bool(state["target_initialized"].item()):
            raise ValueError("Checkpoint target encoder was not initialized")
        self.register_buffer("target_scene_embeds", queries)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        queries = self.target_scene_embeds.expand(images.shape[0], -1, -1, -1)
        # Match training's separate current/future calls and actual autocast output.
        return torch.stack([self.target_image_backbone(images[:, slot], queries)
                            for slot in range(images.shape[1])], dim=1)


class TargetImages(Dataset):
    def __init__(self, loader, tokens, config):
        self.loader = loader
        self.tokens = tokens
        self.pixels = ChainFlowVLAFeatureBuilder(config)
        now = int(config.num_history_frames) - 1
        self.indices = [now] + [now + int(offset) for offset in config.future_offsets]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        frames = self.loader.scene_frames_dicts[token]
        # Supervision needs pixels only: avoid constructing maps, agents or PDM.
        images = []
        for frame_index in self.indices:
            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=self.loader._sensor_blobs_path,
                camera_dict=frames[frame_index]["cams"], sensor_names=list(CAMERA_ORDER),
                load_image_path=False,
            )
            images.append(self.pixels.camera_images(cameras))
        return token, torch.stack(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--expected-scenes", type=int, default=103288)
    parser.add_argument("--precision", choices=("bf16-mixed", "32-true", "16-mixed"), default="bf16-mixed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.num_workers < 0 or args.expected_scenes < 1:
        raise ValueError("Invalid export batch, workers or scene count")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("jepa_training", {})
    if metadata.get("agent") != "chainflow_jepa_native_teacher":
        raise ValueError("Export requires this JEPA agent's own checkpoint")
    contract = metadata.get("contract", {})
    config = OmegaConf.create(contract["config"])
    if contract.get("precision") != args.precision:
        raise ValueError("Export precision must match the source JEPA training precision")
    state = {}
    for key, value in checkpoint["state_dict"].items():
        key = key[len("agent."):] if key.startswith("agent.") else key
        prefix = "_chainflow_vla_model."
        if key.startswith(prefix):
            key = key[len(prefix):]
            if key.startswith("target_image_backbone.") or key in ("target_scene_embeds", "target_initialized"):
                state[key] = value
    model = FrozenTargets(config, state)
    target_hash = target_encoder_fingerprint(model)
    del state, checkpoint
    if file_hash(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Source checkpoint changed during loading; use an immutable checkpoint")

    config_path = Path(args.training_config) if args.training_config else checkpoint_path.parents[3] / "resolved_config.yaml"
    training = OmegaConf.load(config_path)
    if training.train_test_split.get("synthetic_sources"):
        raise ValueError("Export supports the real navtrainval training split only")
    if not training.train_test_split.get("include_val_logs_in_train", False):
        raise ValueError("Expected explicit navtrainval training, including validation logs")
    train_logs = set(resolve_training_log_names(training))
    val_logs = set(training.val_logs)
    development = bool(training.agent.config.teacher_development_validation)
    # Keep this contract identical to Agent.configure_training_run: identical
    # token names alone do not prove that pixels came from the same dataset.
    def log_hash(names):
        return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    training_data_contract = {
        "train_logs": log_hash(train_logs),
        "val_logs": log_hash(val_logs) if development else None,
        "split_hash": hashlib.sha256(json.dumps(
            OmegaConf.to_container(training.train_test_split, resolve=True), sort_keys=True
        ).encode()).hexdigest(),
        "navsim_log_path": str(training.navsim_log_path),
        "sensor_blobs_path": str(training.sensor_blobs_path),
    }
    if contract.get("data") != training_data_contract:
        raise ValueError("Export training-config data contract differs from the source checkpoint")
    scene_filter = instantiate(training.train_test_split.scene_filter)
    scene_filter.log_names = sorted(train_logs if scene_filter.log_names is None
                                   else train_logs.intersection(scene_filter.log_names))
    if int(scene_filter.num_history_frames) != int(config.num_history_frames):
        raise ValueError("Scene filter and frozen encoder history configurations differ")
    if int(scene_filter.num_future_frames) < max(config.future_offsets):
        raise ValueError("Scene filter does not contain all target future frames")
    now = int(config.num_history_frames) - 1
    indices = [now] + [now + int(offset) for offset in config.future_offsets]
    sensors = SensorConfig.build_no_sensors()
    for camera in CAMERA_ORDER:
        setattr(sensors, camera, indices)
    loader = SceneLoader(Path(training.navsim_log_path), Path(training.sensor_blobs_path),
                         scene_filter, sensor_config=sensors)
    tokens = sorted(loader.tokens)
    if len(tokens) != args.expected_scenes or len(set(tokens)) != len(tokens):
        raise ValueError(f"Expected {args.expected_scenes} unique scenes, found {len(tokens)}")
    if any(not token or not all(char.isalnum() or char in "_-" for char in token) for token in tokens):
        raise ValueError("Unsafe scene token in the training split")
    identity = {
        "schema_version": 1, "format": "npz", "array_key": "tokens", "dtype": "float32",
        "target_encoder_sha256": target_hash, "precision": args.precision,
        "token_count": len(tokens), "tokens_sha256": token_list_digest(tokens), "tokens_file": "tokens.txt",
        "data_split": "navtrainval", "navsim_log_path": str(training.navsim_log_path),
        "sensor_blobs_path": str(training.sensor_blobs_path), **cache_config_metadata(config),
        "training_data_contract": training_data_contract,
    }
    provenance = {"source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": checkpoint_hash}
    root = Path(args.output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed = [key for key, value in identity.items() if existing.get(key) != value]
            if changed:
                raise ValueError(f"Existing cache belongs to different encoder/data/settings: {changed}")
            if existing.get("status") not in ("exporting", "complete"):
                raise ValueError("Unexpected cache manifest status")
            recorded_tokens = (root / "tokens.txt").read_text(encoding="utf-8").splitlines()
            if recorded_tokens != tokens:
                raise ValueError("Existing cache token coverage does not match this dataset")
        else:
            if any(root.iterdir()):
                raise ValueError("Refusing a nonempty cache directory without a matching manifest")
            (root / "tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
        (root / "tokens").mkdir(exist_ok=True)
        # Invalidate completion while rechecking/resuming, so training cannot read a partial cache.
        write_json(manifest_path, {**identity, **provenance, "status": "exporting"})
    if world > 1:
        dist.barrier()

    owned = tokens[rank::world]  # Exact disjoint coverage; no sampler padding.
    pending = []
    for token in owned:
        path = root / "tokens" / (token + ".npz")
        if path.is_file():
            validate_record(path, identity["shape"])
        else:
            pending.append(token)
    dataset = TargetImages(loader, pending, config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, drop_last=False, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    model.to(device)
    completed = len(owned) - len(pending)
    readback_count = 0
    online_comparison = None
    logging.info("rank %d: %d cached, %d pending of %d owned", rank, completed, len(pending), len(owned))
    amp_dtype = torch.bfloat16 if args.precision == "bf16-mixed" else torch.float16
    for batch_id, (batch_tokens, images) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=args.precision != "32-true"):
            encoded = model(images)
            if batch_id == 0:
                # Invoke the actual online target method, without constructing
                # an AR/JEPA/Scorer model. Both calls see identical images,
                # weights, batch shape and autocast settings.
                current, future = ChainFlowJEPANativeTeacherModel._encode_world_targets(
                    model, images[:, 0], {"future_images": images[:, 1:]}
                )
                online = torch.cat((current[:, None], future), dim=1)
                max_error = float((encoded.float() - online.float()).abs().max().item())
                torch.testing.assert_close(encoded.float(), online.float(), rtol=1e-5, atol=1e-6)
                online_comparison = {
                    "passed": True, "records": len(batch_tokens), "maximum_absolute_error": max_error,
                    "exactly_equal": bool(torch.equal(encoded, online)), "rtol": 1e-5, "atol": 1e-6,
                    "reference": "ChainFlowJEPANativeTeacherModel._encode_world_targets (online branch)",
                    "same_batch_and_autocast": True,
                }
                logging.info("rank %d: online-target comparison passed, max error %.9g, exact=%s",
                             rank, max_error, online_comparison["exactly_equal"])
                del current, future, online
            values = encoded.float().cpu().numpy()
        for token, row in zip(batch_tokens, values):
            if list(row.shape) != identity["shape"] or not np.isfinite(row).all():
                raise FloatingPointError(f"Invalid frozen target output for {token}")
            path = root / "tokens" / (token + ".npz")
            temporary = path.with_name(path.name + f".rank{rank}.tmp")
            with temporary.open("wb") as stream:
                np.savez(stream, tokens=row)  # Uncompressed avoids CPU decompression in each training epoch.
            temporary.replace(path)
            validate_record(path, identity["shape"])
            if readback_count < 8:
                with np.load(path, allow_pickle=False) as archive:
                    reloaded = archive["tokens"]
                if not np.array_equal(row, reloaded):
                    raise RuntimeError(f"Cached output differs from the actual encoded output: {token}")
                readback_count += 1
        completed += len(batch_tokens)
        if batch_id == 0:
            write_json(root / f"rank-{rank}-verification.json", {
                "status": "exporting", "rank": rank, "verified_records": completed,
                "online_target_comparison": online_comparison,
                "readback_compared_records": readback_count,
                "storage_exactly_equal": True if readback_count else None,
                "maximum_storage_error": 0.0 if readback_count else None,
            })
        if batch_id % 20 == 0:
            logging.info("rank %d: %d/%d verified targets", rank, completed, len(owned))
    if completed != len(owned):
        raise RuntimeError("Local target export did not cover its exact token shard")
    write_json(root / f"rank-{rank}-verification.json", {
        "status": "complete", "rank": rank, "owned_records": len(owned), "verified_records": completed,
        "online_target_comparison": online_comparison,
        "readback_compared_records": readback_count,
        "storage_exactly_equal": True if readback_count else None,
        "maximum_storage_error": 0.0 if readback_count else None,
        "scope": "First batch also checked against the actual online target method under the same autocast. "
                 "Encoded output is compared with saved/reloaded npz. "
                 "This does not claim bitwise equality across different GPU batches or kernels.",
    })
    counts = torch.tensor(completed, dtype=torch.int64)
    if world > 1:
        dist.all_reduce(counts)
    if int(counts) != len(tokens):
        raise RuntimeError("Distributed export did not cover all training tokens")
    if rank == 0:
        if file_hash(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("Source checkpoint changed during export; cache remains incomplete")
        missing = [token for token in tokens if not (root / "tokens" / (token + ".npz")).is_file()]
        if missing:
            raise RuntimeError(f"Cache remains incomplete: {len(missing)} missing files")
        write_json(root / "manifest.json", {
            **identity, **provenance, "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        logging.info("Completed %d frozen target records: %s", len(tokens), root)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
