#!/usr/bin/env python3
"""Cache compact Cosmos3-Nano Wan-VAE future latents for Stage-2 distillation.

Encodes current+future NAVSIM clips with the frozen Wan VAE, spatially pools
the future latent onto a small grid via adaptive_avg_pool2d, and writes ONE
compact bf16 tensor per token as <output>/<log_name>/<token>/cosmos_future_latent.gz.

Supports split-source: token universe from one split (e.g. navtrain) while
future-frame images come from another (e.g. trainval_fut).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SensorConfig

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 2
CACHE_FILE_NAME = "cosmos_future_latent.gz"
NAVSIM_INTERVAL_SECONDS = 0.1
CAMERA_NAMES = (
    "cam_f0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2", "cam_b0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode current+future NAVSIM clips with the frozen Cosmos3-Nano "
            "Wan VAE, spatially pool to a compact grid, and cache as gz-pickle."
        )
    )
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--config-name", default="default_training")
    parser.add_argument(
        "--vae-path", type=Path, default=Path("weights/cosmos3_nano/vae")
    )
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Defaults to $NAVSIM_EXP_ROOT/cosmos_future_latent_v2/<split>.",
    )
    parser.add_argument(
        "--cameras", default="cam_f0",
        help="Comma-separated NAVSIM camera names.",
    )
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--num-future-frames", type=int, default=8,
        help="8 future frames correspond to the 4-second NAVSIM horizon.",
    )
    # Compact latent geometry
    parser.add_argument("--latent-height", type=int, default=4,
                        help="Spatial H after adaptive avg pool.")
    parser.add_argument("--latent-width", type=int, default=8,
                        help="Spatial W after adaptive avg pool.")
    # Split-source overrides
    parser.add_argument("--future-log-path", type=Path, default=None,
                        help="Override log path for future-frame scenes.")
    parser.add_argument("--future-sensor-path", type=Path, default=None,
                        help="Override sensor blob path for future-frame scenes.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--enable-tiling", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, camera_names: Sequence[str]) -> None:
    unknown = sorted(set(camera_names) - set(CAMERA_NAMES))
    if unknown:
        raise ValueError(f"Unknown cameras {unknown}; expected values from {CAMERA_NAMES}")
    if not camera_names:
        raise ValueError("At least one camera must be selected")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("Image dimensions must be positive and divisible by 16")
    if args.num_future_frames <= 0 or args.num_future_frames % 4:
        raise ValueError("num-future-frames must be a positive multiple of 4")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("Require world-size > 0 and 0 <= rank < world-size")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("Use --dtype float32 when running the Wan VAE on CPU")
    if args.latent_height <= 0 or args.latent_width <= 0:
        raise ValueError("latent-height and latent-width must be positive")


def make_sensor_config(camera_names: Sequence[str]) -> SensorConfig:
    values = {name: name in camera_names for name in CAMERA_NAMES}
    return SensorConfig(**values, lidar_pc=False)


def compose_navsim_config(config_name: str, split: str):
    config_dir = (
        Path(__file__).resolve().parents[3]
        / "navsim" / "planning" / "script" / "config" / "training"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=[f"train_test_split={split}"])


def load_tokens(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"Token file is empty: {path}")
    return tokens


def load_vae(path: Path, device: torch.device, dtype: torch.dtype, enable_tiling: bool):
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as error:
        raise RuntimeError(
            "AutoencoderKLWan is unavailable. Install diffusers>=0.37.1."
        ) from error
    vae = AutoencoderKLWan.from_pretrained(str(path), torch_dtype=dtype)
    vae.requires_grad_(False).eval().to(device)
    if enable_tiling:
        vae.enable_tiling()
    return vae


def camera_clip(
    scene: Scene,
    camera_name: str,
    num_future_frames: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Extract current + future frames as [C, T, H, W] float32 tensor in [-1, 1]."""
    current_index = scene.scene_metadata.num_history_frames - 1
    frame_indices = range(current_index, current_index + num_future_frames + 1)
    frames: list[torch.Tensor] = []
    for frame_index in frame_indices:
        camera = getattr(scene.frames[frame_index].cameras, camera_name)
        if camera.image is None:
            raise ValueError(f"Missing {camera_name} image at scene frame {frame_index}")
        image = Image.fromarray(camera.image).convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.0
        frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(frames, dim=1)


def normalize_latents(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    mean = torch.as_tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.as_tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    shape = (1, mean.numel(), 1, 1, 1)
    return (latents - mean.view(shape)) / std.view(shape)


def encode_compact_future_latent(
    scene: Scene,
    camera_names: Sequence[str],
    num_future_frames: int,
    width: int,
    height: int,
    latent_height: int,
    latent_width: int,
    vae: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode clip, normalize, drop current frame, spatially pool to compact grid.

    Returns: [C, T_future, H', W'] bf16 tensor.
    """
    assert len(camera_names) == 1, "Compact cache currently supports single-camera only"
    clips = torch.stack(
        [camera_clip(scene, name, num_future_frames, width, height) for name in camera_names]
    ).to(device=device, dtype=dtype)
    with torch.inference_mode():
        posterior = vae.encode(clips).latent_dist
        full_latent = normalize_latents(posterior.mode(), vae)
    # full_latent: [V, C, T_total, H_lat, W_lat]
    if full_latent.shape[2] <= 1:
        raise RuntimeError(f"Expected future latent frames, got {tuple(full_latent.shape)}")
    # Drop current frame (index 0), keep future frames only
    future_latent = full_latent[:, :, 1:]  # [V, C, T_future, H_lat, W_lat]
    # Spatially pool to compact grid: [V, C, T_future, H', W']
    V, C, T, H, W = future_latent.shape
    pooled = F.adaptive_avg_pool2d(
        future_latent.reshape(V * C * T, 1, H, W),
        (latent_height, latent_width),
    ).reshape(V, C, T, latent_height, latent_width)
    # Squeeze view dimension for single-camera: [C, T, H', W']
    return pooled.squeeze(0).cpu().to(torch.bfloat16)


def save_compact_cache(payload: dict[str, Any], path: Path) -> None:
    """Save as gzipped pickle, atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def cache_path(output_root: Path, scene: Scene) -> Path:
    return (
        output_root / scene.scene_metadata.log_name
        / scene.scene_metadata.initial_token / CACHE_FILE_NAME
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    camera_names = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    validate_args(args, camera_names)

    cfg = compose_navsim_config(args.config_name, args.split)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    requested_tokens = load_tokens(args.tokens_file)
    if requested_tokens is not None:
        scene_filter.tokens = requested_tokens
    if args.max_scenes is not None:
        scene_filter.max_scenes = args.max_scenes

    # Ensure the split provides enough future frames
    if args.num_future_frames > scene_filter.num_future_frames:
        raise ValueError(
            f"Requested {args.num_future_frames} future frames, but split only "
            f"provides {scene_filter.num_future_frames}"
        )

    output_root = args.output_path
    if output_root is None:
        output_root = (
            Path(os.environ["NAVSIM_EXP_ROOT"]) / "cosmos_future_latent_v2" / args.split
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve data paths: use split-source overrides if provided
    log_path = Path(args.future_log_path) if args.future_log_path else Path(cfg.navsim_log_path)
    sensor_path = Path(args.future_sensor_path) if args.future_sensor_path else Path(cfg.sensor_blobs_path)

    LOGGER.info("Log path:     %s", log_path)
    LOGGER.info("Sensor path:  %s", sensor_path)
    LOGGER.info("Output path:  %s", output_root)
    LOGGER.info("Latent grid:  %dx%d", args.latent_height, args.latent_width)

    scene_loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=sensor_path,
        scene_filter=scene_filter,
        sensor_config=make_sensor_config(camera_names),
    )
    tokens = scene_loader.tokens[args.start_index:]
    tokens = tokens[args.rank :: args.world_size]
    LOGGER.info("Rank %d/%d will cache %d scenes", args.rank, args.world_size, len(tokens))

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    vae = load_vae(args.vae_path, device, dtype, args.enable_tiling)

    if args.rank == 0:
        config_payload = {
            "cache_version": CACHE_VERSION,
            "split": args.split,
            "vae_path": str(args.vae_path.resolve()),
            "cameras": camera_names,
            "image_size": [args.width, args.height],
            "num_future_frames": args.num_future_frames,
            "latent_grid": [args.latent_height, args.latent_width],
            "dtype": args.dtype,
            "future_log_path": str(log_path),
            "future_sensor_path": str(sensor_path),
        }
        (output_root / "cache_config.json").write_text(
            json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
        )

    completed = skipped = failed = 0
    for index, token in enumerate(tokens, start=1):
        try:
            scene = scene_loader.get_scene_from_token(token)
            output_file = cache_path(output_root, scene)
            if output_file.is_file() and not args.overwrite:
                skipped += 1
                continue
            compact_latent = encode_compact_future_latent(
                scene, camera_names, args.num_future_frames,
                args.width, args.height,
                args.latent_height, args.latent_width,
                vae, device, dtype,
            )
            payload = {
                "cache_version": CACHE_VERSION,
                "token": scene.scene_metadata.initial_token,
                "log_name": scene.scene_metadata.log_name,
                "future_latent": compact_latent,  # [C, T, H', W'] bf16
            }
            save_compact_cache(payload, output_file)
            completed += 1
        except Exception:
            failed += 1
            LOGGER.exception("Failed token %s", token)
        if index % args.log_every == 0:
            LOGGER.info(
                "Progress %d/%d: completed=%d skipped=%d failed=%d",
                index, len(tokens), completed, skipped, failed,
            )
    LOGGER.info("Finished: completed=%d skipped=%d failed=%d", completed, skipped, failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
