#!/usr/bin/env python3
"""Cache Cosmos3-Nano *OmniTransformer* fused future latents for Stage-2 distillation.

Unlike the VAE-only variant (``run_dataset_caching_cosmos_future_latent.py``),
this entrypoint drives the FULL Cosmos3 world model to produce a much richer
teacher signal per token:

    current frame  -.
    GT trajectory  -+-> 9D AV pose-delta action -.
    text prompt    -'                            +-> Cosmos3OmniTransformer (36L MoE)
                                                 |        |
                                     (forward-dynamics)   v
                                              fused hidden state  [N_vis, 4096]

Design rationale (why forward-dynamics, not raw VAE encoding):
  The OmniTransformer is a diffusion *denoiser*, not an encoder. In
  ``forward_dynamics`` mode it rolls out the future conditioned on the current
  frame + a given action trajectory + a text instruction. We hook the
  transformer just before its vision output projection (``proj_out``) and
  capture the fused pre-projection hidden state. This is precisely "the world
  model's action-conditioned understanding of the future" -- a strictly stronger
  distillation target than a plain VAE encoding of the ground-truth future.

The captured hidden state ``[num_noisy_vision_tokens, 4096]`` is adaptively
pooled along the token dimension to a fixed ``[K, 4096]`` grid (geometry-agnostic
w.r.t. the resolution binning), stored bf16 as gz-pickle at
``<output>/<log_name>/<token>/cosmos_omni_latent.gz`` with key ``future_latent``.

Supports split-source: token universe from one split (e.g. navtrain) while
future-frame images come from another (e.g. trainval_fut).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneLoader

LOGGER = logging.getLogger("cache_cosmos_omni_latent")
CACHE_VERSION = 2
CACHE_FILE_NAME = "cosmos_omni_latent.gz"
CACHE_KEY = "future_latent"
PROMPT_STYLE = "internvl_context_v1"
AV_DOMAIN_NAME = "av"
AV_RAW_ACTION_DIM = 9
CAMERA_NAMES = (
    "cam_f0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2", "cam_b0",
)

_DRIVING_COMMAND_LABELS = {
    "left": "TURN LEFT",
    "straight": "GO STRAIGHT",
    "right": "TURN RIGHT",
    "unknown": "UNKNOWN",
}
_DRIVING_COMMAND_DESCRIPTIONS = {
    "left": "turn left while following the road geometry",
    "straight": "continue straight while following the current roadway",
    "right": "turn right while following the road geometry",
    "unknown": "continue along the roadway according to the supplied ego motion",
}


class _CaptureDone(Exception):
    """Sentinel raised inside the forward pre-hook to abort the (expensive) rest."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the frozen Cosmos3-Nano OmniTransformer in forward-dynamics "
            "mode and cache the fused future hidden state as a distillation target."
        )
    )
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--config-name", default="default_training")
    parser.add_argument(
        "--model-root", type=Path, default=Path("weights/cosmos3_nano"),
        help="Diffusers checkpoint root that from_pretrained loads the full pipeline from.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Defaults to $NAVSIM_EXP_ROOT/cosmos_omni_latent_v2/<split>.",
    )
    parser.add_argument(
        "--camera", default="cam_f0",
        help="Single NAVSIM camera name used as the current-frame conditioning image.",
    )
    parser.add_argument(
        "--num-future-frames", type=int, default=8,
        help="Number of future transitions (chunk_size). 8 == 4s NAVSIM horizon.",
    )
    parser.add_argument(
        "--resolution-tier", type=int, default=480, choices=(256, 480, 704, 720),
        help="Cosmos action resolution tier; 480 maps landscape AV frames to 480x832.",
    )
    parser.add_argument(
        "--latent-tokens", type=int, default=8,
        help="K: adaptive-pool the [N_vis, 4096] hidden state to [K, 4096].",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Conditioning frame rate. NAVSIM uses 0.5-second intervals (2 FPS).",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=1,
        help="Number of scheduler denoising steps to make available for capture.",
    )
    parser.add_argument(
        "--timestep-index", type=int, default=0,
        help="Zero-based denoising step whose pre-projection hidden state is cached.",
    )
    # Split-source overrides
    parser.add_argument("--future-log-path", type=Path, default=None)
    parser.add_argument("--future-sensor-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.camera not in CAMERA_NAMES:
        raise ValueError(f"Unknown camera {args.camera!r}; expected one of {CAMERA_NAMES}")
    if args.num_future_frames < 1:
        raise ValueError("num-future-frames must be >= 1")
    if args.latent_tokens < 1:
        raise ValueError("latent-tokens must be >= 1")
    if args.fps <= 0:
        raise ValueError("fps must be > 0")
    if args.num_inference_steps < 1:
        raise ValueError("num-inference-steps must be >= 1")
    if not 0 <= args.timestep_index < args.num_inference_steps:
        raise ValueError("timestep-index must satisfy 0 <= index < num-inference-steps")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("Require world-size > 0 and 0 <= rank < world-size")


def make_sensor_config(camera_name: str) -> SensorConfig:
    values = {name: name == camera_name for name in CAMERA_NAMES}
    return SensorConfig(**values, lidar_pc=False)


def compose_navsim_config(config_name: str, split: str):
    config_dir = (
        Path(__file__).resolve().parents[3]
        / "navsim" / "planning" / "script" / "config" / "training"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=[f"train_test_split={split}"])


def load_tokens(path):
    if path is None:
        return None
    tokens = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"Token file is empty: {path}")
    return tokens


def load_pipeline(model_root: Path, device: torch.device, dtype: torch.dtype):
    """Load the full Cosmos3 omni pipeline with the guardrail disabled (offline)."""
    try:
        from diffusers import Cosmos3OmniPipeline
    except ImportError as error:
        raise RuntimeError(
            "Cosmos3OmniPipeline unavailable. Install diffusers>=0.40 (dev) that ships it."
        ) from error
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(model_root), torch_dtype=dtype, enable_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    return pipe


def current_frame_image(scene: Scene, camera_name: str) -> Image.Image:
    """Return the current-frame RGB PIL image (pipeline resizes to the resolution tier)."""
    current_index = scene.scene_metadata.num_history_frames - 1
    camera = getattr(scene.frames[current_index].cameras, camera_name)
    if camera.image is None:
        raise ValueError(f"Missing {camera_name} image at current frame {current_index}")
    return Image.fromarray(camera.image).convert("RGB")


def _rotation_6d_from_heading(delta_heading: float):
    """First two columns of Rz(delta_heading), the Cosmos 6D continuous rotation."""
    c, s = math.cos(delta_heading), math.sin(delta_heading)
    # col0 = [c, s, 0], col1 = [-s, c, 0]
    return [c, s, 0.0, -s, c, 0.0]


def trajectory_to_av_actions(scene: Scene, num_future_frames: int) -> torch.Tensor:
    """Convert the NAVSIM future trajectory into [num_future_frames, 9] AV pose deltas.

    NAVSIM future poses are (x, y, heading) in the current ego frame. We build the
    frame chain [current(origin), fut_0, ..., fut_{N-1}] and emit, for each of the
    N transitions i->i+1, a 9D delta = [dx, dy, 0, <6D rotation of dheading>] where
    the translation delta is expressed in frame-i local coordinates (standard ego
    pose-delta convention used by the Cosmos AV embodiment).
    """
    poses = scene.get_future_trajectory(num_trajectory_frames=num_future_frames).poses
    poses = np.asarray(poses, dtype=np.float64)  # [N, 3] = (x, y, heading)
    if poses.shape[0] < num_future_frames:
        raise ValueError(
            f"Only {poses.shape[0]} future poses available, need {num_future_frames}"
        )
    # Frame chain in ego-0 coordinates: index 0 is the current pose at the origin.
    chain = np.concatenate([np.zeros((1, 3), dtype=np.float64), poses[:num_future_frames]], axis=0)
    actions = []
    for i in range(num_future_frames):
        t_i = chain[i, :2]
        t_next = chain[i + 1, :2]
        theta_i = chain[i, 2]
        theta_next = chain[i + 1, 2]
        # Translation delta expressed in frame-i local coordinates.
        c, s = math.cos(-theta_i), math.sin(-theta_i)
        rot_i_inv = np.array([[c, -s], [s, c]], dtype=np.float64)
        d_local = rot_i_inv @ (t_next - t_i)
        d_heading = math.atan2(math.sin(theta_next - theta_i), math.cos(theta_next - theta_i))
        rot6d = _rotation_6d_from_heading(d_heading)
        actions.append([float(d_local[0]), float(d_local[1]), 0.0, *rot6d])
    return torch.tensor(actions, dtype=torch.float32)  # [N, 9]


def _driving_command_key(command: Any) -> str:
    """Decode NAVSIM's [left, straight, right, unknown] one-hot command."""
    keys = ["left", "straight", "right", "unknown"]
    command = np.asarray(command, dtype=np.float32)
    if command.size >= 1 and float(command.max()) > 0:
        idx = int(command.argmax())
        return keys[idx] if 0 <= idx < len(keys) else "unknown"
    return "unknown"


def _format_motion_vector(values: Any) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    x = float(values[0]) if values.size > 0 else 0.0
    y = float(values[1]) if values.size > 1 else 0.0
    return f"({x:+.2f}, {y:+.2f})"


def driving_context_prompt(scene: Scene, num_future_frames: int) -> str:
    """Build a sample-specific action caption without future-image leakage."""
    ego_statuses = scene.get_agent_input().ego_statuses[-4:]
    current_status = ego_statuses[-1]
    command_key = _driving_command_key(current_status.driving_command)

    history_parts = []
    for index, status in enumerate(ego_statuses):
        pose = np.asarray(status.ego_pose, dtype=np.float64)
        relative_step = index - len(ego_statuses) + 1
        history_parts.append(
            f"t{relative_step:+d}=(x={pose[0]:+.2f} m, y={pose[1]:+.2f} m, "
            f"heading={pose[2]:+.3f} rad)"
        )

    future_poses = np.asarray(
        scene.get_future_trajectory(num_trajectory_frames=num_future_frames).poses,
        dtype=np.float64,
    )
    if future_poses.shape[0] < num_future_frames:
        raise ValueError(
            f"Only {future_poses.shape[0]} future poses available, need {num_future_frames}"
        )
    future_poses = future_poses[:num_future_frames]
    future_chain = np.concatenate([np.zeros((1, 3)), future_poses], axis=0)
    path_length = float(np.linalg.norm(np.diff(future_chain[:, :2], axis=0), axis=1).sum())
    final_pose = future_poses[-1]

    return (
        "An ego-view front-camera driving sequence. "
        f"The active navigation command is {_DRIVING_COMMAND_LABELS[command_key]}: the ego vehicle should "
        f"{_DRIVING_COMMAND_DESCRIPTIONS[command_key]}. "
        "Recent ego poses in the current ego frame, ordered from oldest to current, are "
        f"{'; '.join(history_parts)}. "
        f"The current planar velocity components are {_format_motion_vector(current_status.ego_velocity)} m/s, "
        f"and the current planar acceleration components are {_format_motion_vector(current_status.ego_acceleration)} m/s^2. "
        f"Over the next {num_future_frames} action steps, the ego motion covers approximately "
        f"{path_length:.2f} m and ends at (x={final_pose[0]:+.2f} m, y={final_pose[1]:+.2f} m, "
        f"heading={final_pose[2]:+.3f} rad) in the current ego frame. "
        "The generated evolution should follow that ego motion smoothly, remain consistent with "
        "the visible lane and road geometry, respect traffic controls, preserve the identities and "
        "appearance of surrounding road users, and maintain temporal continuity of scene layout, "
        "lighting, and weather."
    )


def build_action_condition(scene: Scene, args: argparse.Namespace, image: Image.Image):
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

    raw_actions = trajectory_to_av_actions(scene, args.num_future_frames)
    return CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=args.num_future_frames,
        domain_name=AV_DOMAIN_NAME,
        resolution_tier=args.resolution_tier,
        raw_actions=raw_actions,
        image=image,
        view_point="ego_view",
    )


def extract_fused_latent(pipe, scene, args, device, dtype) -> tuple[torch.Tensor, str]:
    """Run one OmniTransformer forward and capture the fused vision hidden state.

    Returns: [K, 4096] bf16 pooled along the token dimension.
    """
    image = current_frame_image(scene, args.camera)
    action = build_action_condition(scene, args, image)
    prompt = driving_context_prompt(scene, args.num_future_frames)

    captured = {}
    forward_index = 0

    def _pre_hook(_module, inputs):
        nonlocal forward_index
        if forward_index != args.timestep_index:
            forward_index += 1
            return
        # proj_out input == last_hidden_state[vision_mse_loss_indexes] == [N_vis_noisy, 4096]
        captured["hidden"] = inputs[0].detach().float()
        captured["step_index"] = forward_index
        raise _CaptureDone

    handle = pipe.transformer.proj_out.register_forward_pre_hook(_pre_hook)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with torch.inference_mode():
            pipe(
                prompt=prompt,
                action=action,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,
                enable_safety_check=False,
                output_type="latent",
                generator=generator,
            )
    except _CaptureDone:
        pass
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Forward pre-hook did not fire; transformer.proj_out was not called")
    hidden = captured["hidden"]  # [N_vis, 4096]
    if hidden.ndim != 2:
        raise RuntimeError(f"Unexpected hidden shape {tuple(hidden.shape)}; expected [N, C]")
    # Adaptive-pool the token dimension to a fixed K (geometry-agnostic).
    pooled = F.adaptive_avg_pool1d(
        hidden.transpose(0, 1).unsqueeze(0), args.latent_tokens
    ).squeeze(0).transpose(0, 1)  # [K, 4096]
    return pooled.contiguous().to("cpu", torch.bfloat16), prompt


def save_compact_cache(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def cache_path(output_root: Path, scene: Scene) -> Path:
    return (
        output_root / scene.scene_metadata.log_name
        / scene.scene_metadata.initial_token / CACHE_FILE_NAME
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_args(args)

    cfg = compose_navsim_config(args.config_name, args.split)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    requested_tokens = load_tokens(args.tokens_file)
    if requested_tokens is not None:
        scene_filter.tokens = requested_tokens
    if args.max_scenes is not None:
        scene_filter.max_scenes = args.max_scenes
    if args.num_future_frames > scene_filter.num_future_frames:
        raise ValueError(
            f"Requested {args.num_future_frames} future frames, but split only "
            f"provides {scene_filter.num_future_frames}"
        )

    output_root = args.output_path
    if output_root is None:
        output_root = (
            Path(os.environ["NAVSIM_EXP_ROOT"]) / "cosmos_omni_latent_v2" / args.split
        )
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.future_log_path) if args.future_log_path else Path(cfg.navsim_log_path)
    sensor_path = Path(args.future_sensor_path) if args.future_sensor_path else Path(cfg.sensor_blobs_path)
    if not log_path.exists():
        raise SystemExit(f"[preflight] Log path does not exist: {log_path}")
    if not sensor_path.exists():
        raise SystemExit(f"[preflight] Sensor path does not exist: {sensor_path}")

    LOGGER.info("Model root:   %s", args.model_root)
    LOGGER.info("Log path:     %s", log_path)
    LOGGER.info("Sensor path:  %s", sensor_path)
    LOGGER.info("Output path:  %s", output_root)
    LOGGER.info("Latent shape: [K=%d, 4096]", args.latent_tokens)

    scene_loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=sensor_path,
        scene_filter=scene_filter,
        sensor_config=make_sensor_config(args.camera),
    )
    tokens = scene_loader.tokens[args.start_index:]
    tokens = tokens[args.rank :: args.world_size]
    LOGGER.info("Rank %d/%d will cache %d scenes", args.rank, args.world_size, len(tokens))

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    pipe = load_pipeline(args.model_root, device, dtype)

    if args.rank == 0:
        config_payload = {
            "cache_version": CACHE_VERSION,
            "split": args.split,
            "model_root": str(args.model_root.resolve()),
            "camera": args.camera,
            "num_future_frames": args.num_future_frames,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "timestep_index": args.timestep_index,
            "resolution_tier": args.resolution_tier,
            "latent_tokens": args.latent_tokens,
            "latent_channels": 4096,
            "dtype": args.dtype,
            "future_log_path": str(log_path),
            "future_sensor_path": str(sensor_path),
            "teacher": "cosmos3_omni_forward_dynamics",
            "prompt_style": PROMPT_STYLE,
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
            latent, prompt = extract_fused_latent(pipe, scene, args, device, dtype)
            payload = {
                CACHE_KEY: latent,  # [K, 4096] bf16
                "cache_version": CACHE_VERSION,
                "token": scene.scene_metadata.initial_token,
                "log_name": scene.scene_metadata.log_name,
                "camera": args.camera,
                "latent_shape": list(latent.shape),
                "num_future_frames": args.num_future_frames,
                "fps": args.fps,
                "num_inference_steps": args.num_inference_steps,
                "timestep_index": args.timestep_index,
                "teacher": "cosmos3_omni_forward_dynamics",
                "prompt_style": PROMPT_STYLE,
                "prompt": prompt,
                "pooling": "adaptive_avg_pool1d(token_dim)",
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
