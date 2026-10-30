#!/usr/bin/env python3
"""Phase-1 sanity check: can Cosmos3-Nano act as a NAVSIM World Teacher?

For each NAVSIM scene we build a set of candidate future trajectories:
  * gt          -- the human GT trajectory
  * bad kinds   -- lateral_shift, wrong_turn, hard_brake_stop, collision
and feed each (current front frame I, trajectory as 9D AV pose-delta actions)
to Cosmos3 in forward-dynamics mode to generate a future video V_tau.

We then score the "world consistency / physical plausibility" S_C(V_tau):

  方法 A (image fidelity vs the real future):
      PSNR / SSIM between the Cosmos-generated future frames and the REAL
      GT future frames. A good world model that follows the given trajectory
      should yield S_C(gt) > S_C(bad) because the real future IS the gt
      trajectory rendered by the world.

  方法 D (ego-motion adherence / no-collision):
      Estimate the ego motion implied by the generated video (optical flow
      between consecutive generated frames) and check that it (a) follows the
      commanded trajectory rather than diverging, and (b) does not stop
      abruptly (hard brake) or slide sideways (lateral) or turn wrongly.

Goal: report, per scene and overall, whether S_C(gt) > S_C(bad) consistently.

Usage:
  PYTHON_BIN=.../cosmos3/bin/python run_validate_cosmos_world_teacher.py \
      --split navmini --config-name default_training \
      --model-root weights/cosmos3_nano --max-scenes 4 \
      --num-future-frames 8 --resolution-tier 480 --num-inference-steps 10 \
      --output-path <out> --camera cam_f0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneLoader

LOGGER = logging.getLogger("validate_cosmos_world_teacher")
AV_DOMAIN_NAME = "av"
AV_RAW_ACTION_DIM = 9
CAMERA_NAMES = (
    "cam_f0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2", "cam_b0",
)
_DRIVING_COMMAND_LABELS = {
    "left": "TURN LEFT", "straight": "GO STRAIGHT",
    "right": "TURN RIGHT", "unknown": "UNKNOWN",
}
_DRIVING_COMMAND_DESCRIPTIONS = {
    "left": "turn left while following the road geometry",
    "straight": "continue straight while following the current roadway",
    "right": "turn right while following the road geometry",
    "unknown": "continue along the roadway according to the supplied ego motion",
}
BAD_KINDS = ("lateral_shift", "wrong_turn", "hard_brake_stop", "collision")


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


def load_pipeline(model_root: Path, device: torch.device, dtype: torch.dtype):
    try:
        from diffusers import Cosmos3OmniPipeline
    except ImportError as error:
        raise RuntimeError("Cosmos3OmniPipeline unavailable (diffusers>=0.40 dev)") from error
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(model_root), torch_dtype=dtype, enable_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    return pipe


def current_frame_image(scene: Scene, camera_name: str) -> Image.Image:
    current_index = scene.scene_metadata.num_history_frames - 1
    camera = getattr(scene.frames[current_index].cameras, camera_name)
    if camera.image is None:
        raise ValueError(f"Missing {camera_name} image at current frame {current_index}")
    return Image.fromarray(camera.image).convert("RGB")


def future_frames_images(scene: Scene, camera_name: str, num_frames: int):
    """Return PIL images of the REAL GT future frames [num_frames] (after current frame)."""
    h = scene.scene_metadata.num_history_frames - 1
    frames = []
    for i in range(1, num_frames + 1):
        camera = getattr(scene.frames[h + i].cameras, camera_name)
        if camera.image is None:
            return None
        frames.append(Image.fromarray(camera.image).convert("RGB"))
    return frames


def _rotation_6d_from_heading(delta_heading: float):
    c, s = math.cos(delta_heading), math.sin(delta_heading)
    return [c, s, 0.0, -s, c, 0.0]


def poses_to_av_actions(poses: np.ndarray, num_future_frames: int) -> torch.Tensor:
    """Convert [N,3]=(x,y,heading) future poses (in current ego frame) to [N,9] actions."""
    chain = np.concatenate([np.zeros((1, 3), dtype=np.float64), poses[:num_future_frames]], axis=0)
    actions = []
    for i in range(num_future_frames):
        t_i, t_next = chain[i, :2], chain[i + 1, :2]
        theta_i, theta_next = chain[i, 2], chain[i + 1, 2]
        c, s = math.cos(-theta_i), math.sin(-theta_i)
        rot_i_inv = np.array([[c, -s], [s, c]], dtype=np.float64)
        d_local = rot_i_inv @ (t_next - t_i)
        d_heading = math.atan2(math.sin(theta_next - theta_i), math.cos(theta_next - theta_i))
        rot6d = _rotation_6d_from_heading(d_heading)
        actions.append([float(d_local[0]), float(d_local[1]), 0.0, *rot6d])
    return torch.tensor(actions, dtype=torch.float32)


def trajectory_actions(scene: Scene, num_future_frames: int) -> torch.Tensor:
    poses = np.asarray(
        scene.get_future_trajectory(num_trajectory_frames=num_future_frames).poses,
        dtype=np.float64,
    )
    return poses_to_av_actions(poses, num_future_frames)


def make_bad_actions(gt_actions: torch.Tensor, kind: str) -> torch.Tensor:
    """Perturb the GT 9D action sequence into a physically-bad trajectory.

    Bad trajectories are made *kinematically* distinct from GT so that a
    motion-following world model must produce visibly different futures.
    """
    a = gt_actions.clone()
    n = a.shape[0]
    if kind == "lateral_shift":
        # Driving command says straight, but the ego drifts laterally (lane
        # departure). Add a growing +y offset every step.
        for i in range(n):
            a[i, 1] = a[i, 1] + 1.0 + 0.25 * i
    elif kind == "wrong_turn":
        # Force a pronounced turn opposite to the travel direction: override the
        # heading delta to a large fixed value (e.g. -0.6 rad/step) so the world
        # model must swerve. Rz(dh) rot6d = [c, s, 0, -s, c, 0].
        dh = -0.6
        c, s = math.cos(dh), math.sin(dh)
        for i in range(n):
            a[i, 3], a[i, 4], a[i, 5] = c, s, -s
    elif kind == "hard_brake_stop":
        # Abrupt stop: scale translation down to a near standstill.
        for i in range(n):
            a[i, 0] *= 0.05
            a[i, 1] *= 0.05
    elif kind == "collision":
        # Violent lateral jerk into a barrier (fixed large +y, tiny forward).
        for i in range(n):
            a[i, 0] *= 0.2
            a[i, 1] = 2.0 + 0.4 * i
    else:
        raise ValueError(f"Unknown bad kind: {kind}")
    return a


def build_action(raw_actions: torch.Tensor, image: Image.Image, chunk_size: int, tier: str):
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
    return CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=chunk_size,
        domain_name=AV_DOMAIN_NAME,
        resolution_tier=tier,
        raw_actions=raw_actions,
        image=image,
        view_point="ego_view",
    )


def gen_video(pipe, prompt: str, action, num_steps: int, fps: float, device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            action=action,
            fps=fps,
            num_inference_steps=num_steps,
            guidance_scale=1.0,
            enable_safety_check=False,
            output_type="np",
            generator=generator,
        )
    vid = out.video  # [T, H, W, C] float32 [0,1] numpy (batch dim removed)
    if torch.is_tensor(vid):
        vid = vid.detach().cpu().numpy()
    vid = np.asarray(vid, dtype=np.float32)
    if vid.max() <= 1.0 + 1e-3:
        vid = vid * 255.0
    return np.clip(np.round(vid), 0, 255).astype(np.uint8)


def driving_context_prompt(scene: Scene, num_future_frames: int) -> str:
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
    )[:num_future_frames]
    future_chain = np.concatenate([np.zeros((1, 3)), future_poses], axis=0)
    path_length = float(np.linalg.norm(np.diff(future_chain[:, :2], axis=0), axis=1).sum())
    final_pose = future_poses[-1]
    return (
        "An ego-view front-camera driving sequence. "
        f"The active navigation command is {_DRIVING_COMMAND_LABELS[command_key]}: "
        f"the ego vehicle should {_DRIVING_COMMAND_DESCRIPTIONS[command_key]}. "
        "Recent ego poses in the current ego frame are "
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


def _driving_command_key(command: Any) -> str:
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


# ---------------- metrics ----------------

def _to_gray_batch(vid: np.ndarray) -> np.ndarray:
    """vid: [T,H,W,C] uint8 -> [T,H,W] float64 gray."""
    return vid.mean(axis=-1).astype(np.float64)


def psnr_ssim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """a,b: [H,W,C] uint8. Returns (psnr, ssim)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    psnr = peak_signal_noise_ratio(a, b, data_range=255.0)
    ssim = structural_similarity(a, b, channel_axis=-1, data_range=255.0)
    return float(psnr), float(ssim)


def _dense_optical_flows(vid: np.ndarray) -> np.ndarray:
    """vid: [T,H,W,C] uint8 -> dense flow fields [T-1,H,W,2] (float64)."""
    import cv2
    gray = _to_gray_batch(vid).astype(np.uint8)
    flows = []
    for i in range(len(gray) - 1):
        flows.append(cv2.calcOpticalFlowFarneback(
            gray[i], gray[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        ))
    return np.stack(flows) if flows else np.zeros((0, 0, 0, 2), dtype=np.float64)


def optical_flow_magnitude(vid: np.ndarray) -> list[float]:
    """vid: [T,H,W,C] uint8. Mean optical-flow magnitude per transition (method D)."""
    flows = _dense_optical_flows(vid)
    if flows.size == 0:
        return []
    return [float(np.abs(flows[i]).mean()) for i in range(len(flows))]


def estimate_ego_motion(vid: np.ndarray) -> dict:
    """Rough ego-motion stats from optical flow, incl. directional asymmetry (method D)."""
    mags = np.asarray(optical_flow_magnitude(vid))
    d = {
        "mean_flow": float(mags.mean()) if mags.size else 0.0,
        "flow_std": float(mags.std()) if mags.size else 0.0,
        "flow_max": float(mags.max()) if mags.size else 0.0,
        "flow_drop_ratio": float(mags[0] / mags[-1]) if mags.size > 1 and mags[-1] > 0 else 1.0,
    }
    flows = _dense_optical_flows(vid)
    if flows.size > 0:
        fx, fy = flows[..., 0], flows[..., 1]
        H, W = fx.shape[1], fx.shape[2]
        mf = float(np.sqrt(fx**2 + fy**2).mean()) + 1e-6
        left = fx[..., :W // 2].mean()
        right = fx[..., W // 2:].mean()
        top = fy[..., :H // 2].mean()
        bot = fy[..., H // 2:].mean()
        q1 = fx[..., :H // 2, W // 2:].mean()   # top-right
        q2 = fx[..., :H // 2, :W // 2].mean()   # top-left
        q3 = fx[..., H // 2:, :W // 2].mean()   # bottom-left
        q4 = fx[..., H // 2:, W // 2:].mean()   # bottom-right
        d["lat_asym"] = float(left - right)
        d["vert_asym"] = float(top - bot)
        d["rot"] = float(q1 - q2 - q3 + q4)
        d["lat_asym_rel"] = float((left - right) / mf)
        d["rot_rel"] = float((q1 - q2 - q3 + q4) / mf)
    return d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 Cosmos World Teacher sanity check")
    parser.add_argument("--split", default="navmini")
    parser.add_argument("--config-name", default="default_training")
    parser.add_argument("--model-root", type=Path, default=Path("weights/cosmos3_nano"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--camera", default="cam_f0")
    parser.add_argument("--num-future-frames", type=int, default=8)
    parser.add_argument("--resolution-tier", type=int, default=480, choices=(256, 480, 704, 720))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--future-log-path", type=Path, default=None)
    parser.add_argument("--future-sensor-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--bad-kinds", default=",".join(BAD_KINDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true", help="Save generated frames as png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = compose_navsim_config(args.config_name, args.split)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    if args.max_scenes is not None:
        scene_filter.max_scenes = args.max_scenes
    if args.num_future_frames > scene_filter.num_future_frames:
        raise ValueError(f"Requested {args.num_future_frames} future frames, split provides {scene_filter.num_future_frames}")

    out_root = args.output_path or (Path(os.environ["NAVSIM_EXP_ROOT"]) / "cosmos_world_teacher_validate" / args.split)
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.future_log_path) if args.future_log_path else Path(cfg.navsim_log_path)
    sensor_path = Path(args.future_sensor_path) if args.future_sensor_path else Path(cfg.sensor_blobs_path)
    if not log_path.exists():
        raise SystemExit(f"[preflight] Log path does not exist: {log_path}")
    if not sensor_path.exists():
        raise SystemExit(f"[preflight] Sensor path does not exist: {sensor_path}")

    bad_kinds = [k.strip() for k in args.bad_kinds.split(",") if k.strip()]

    scene_loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=sensor_path,
        scene_filter=scene_filter,
        sensor_config=make_sensor_config(args.camera),
    )
    tokens = scene_loader.tokens[args.start_index:]
    LOGGER.info("Validating %d scenes; bad kinds=%s", len(tokens), bad_kinds)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    LOGGER.info("Loading Cosmos3 pipeline from %s ...", args.model_root)
    t0 = time.time()
    pipe = load_pipeline(args.model_root, device, dtype)
    LOGGER.info("Pipeline loaded in %.1fs", time.time() - t0)

    results = []          # per kind per scene
    per_scene_active = []  # active-trajectory set (only gt + perturbed versions)
    for si, token in enumerate(tokens, start=1):
        try:
            scene = scene_loader.get_scene_from_token(token)
        except Exception:
            LOGGER.exception("Load scene failed %s", token)
            continue
        LOGGER.info("[%d/%d] scene %s", si, len(tokens), token)
        img = current_frame_image(scene, args.camera)
        gt_actions = trajectory_actions(scene, args.num_future_frames)
        real_fut = future_frames_images(scene, args.camera, args.num_future_frames)
        prompt = driving_context_prompt(scene, args.num_future_frames)

        trajs = {"gt": gt_actions}
        for k in bad_kinds:
            trajs[k] = make_bad_actions(gt_actions, k)

        scene_rec = {}
        vids = {}
        for name, acts in trajs.items():
            action = build_action(acts, img, args.num_future_frames, str(args.resolution_tier))
            t0 = time.time()
            try:
                vid = gen_video(pipe, prompt, action, args.num_inference_steps, args.fps, device, args.seed)
            except Exception:
                LOGGER.exception("  %s generation failed", name)
                continue
            dv = time.time() - t0
            vid_np = vid  # [T,H,W,C] uint8
            vids[name] = vid_np
            LOGGER.info("  %-14s gen %.1fs shape=%s", name, dv, list(vid_np.shape))

            # Method A: fidelity vs real future frames
            rec = {"name": name, "gen_time": dv}
            if real_fut is not None and vid_np.shape[0] >= 1 + args.num_future_frames:
                psnrs, ssims = [], []
                for i in range(args.num_future_frames):
                    gen_frame = vid_np[1 + i]       # generated frame i+1
                    real_frame = np.asarray(real_fut[i].resize((gen_frame.shape[1], gen_frame.shape[0])))
                    p, s = psnr_ssim(gen_frame, real_frame)
                    psnrs.append(p)
                    ssims.append(s)
                rec["psnr_mean"] = float(np.mean(psnrs))
                rec["ssim_mean"] = float(np.mean(ssims))

            # Method D: ego-motion stats from optical flow
            rec["flow"] = estimate_ego_motion(vid_np)
            scene_rec[name] = rec

            if args.save_videos:
                vdir = out_root / f"scene{si}_{token[:8]}_{name}"
                vdir.mkdir(parents=True, exist_ok=True)
                for fi, fr in enumerate(vid_np):
                    Image.fromarray(fr).save(str(vdir / f"f{fi}.png"))
                img.save(str(vdir / "cond_frame.png"))

        results.append({"scene": token, "records": scene_rec})
        if "gt" in scene_rec:
            gt_a = scene_rec["gt"]
            active = {"scene": token, "S_gt": gt_a}
            for k in bad_kinds:
                if k in scene_rec:
                    active[f"S_{k}"] = scene_rec[k]
            per_scene_active.append(active)

        if si % 5 == 0:
            _write_report(out_root, results, per_scene_active, bad_kinds)

    _write_report(out_root, results, per_scene_active, bad_kinds)
    LOGGER.info("Done. Report at %s", out_root / "report.json")


def _write_report(out_root: Path, results, per_scene_active, bad_kinds) -> None:
    # Method A: S(gt) vs S(bad) via psnr/ssim
    summary = {"n_scene": len(results), "methodA_psnr": {}, "methodA_ssim": {}, "methodD": {}}
    for metric in ("psnr_mean", "ssim_mean"):
        gt_vals = [s["S_gt"][metric] for s in per_scene_active if "S_gt" in s and metric in s["S_gt"]]
        if not gt_vals:
            continue
        for k in bad_kinds:
            bad_vals = [
                s.get(f"S_{k}", {}).get(metric)
                for s in per_scene_active if f"S_{k}" in s and metric in s[f"S_{k}"]
            ]
            bad_vals = [v for v in bad_vals if v is not None]
            if not bad_vals:
                continue
            gt_a = np.asarray(gt_vals)
            bad_a = np.asarray(bad_vals)
            wins = int((gt_a > bad_a).sum())
            key = "psnr" if metric == "psnr_mean" else "ssim"
            summary[f"methodA_{key}"][k] = {
                "S_gt_mean": float(gt_a.mean()),
                "S_bad_mean": float(bad_a.mean()),
                "S_gt_gt_bad_ratio": float(gt_a.mean() / bad_a.mean()) if bad_a.mean() > 0 else None,
                "n_pairs": int(len(bad_vals)),
                "n_gt_wins": wins,
                "acc": float(wins / len(bad_vals)),
            }
    # Method D: does hard-brake reduce flow (ego should slow)?
    gt_flows = [s["S_gt"]["flow"]["mean_flow"] for s in per_scene_active if "S_gt" in s]
    brake_flows = [
        s["S_hard_brake_stop"]["flow"]["mean_flow"]
        for s in per_scene_active if "S_hard_brake_stop" in s
    ]
    lateral_flows = [
        s["S_lateral_shift"]["flow"]["mean_flow"]
        for s in per_scene_active if "S_lateral_shift" in s
    ]
    if gt_flows and brake_flows:
        gt_a, br_a = np.asarray(gt_flows), np.asarray(brake_flows)
        summary["methodD"]["hard_brake_slows"] = {
            "gt_flow_mean": float(gt_a.mean()),
            "brake_flow_mean": float(br_a.mean()),
            "n_brake_lt_gt": int((br_a < gt_a).sum()),
            "n": int(len(br_a)),
            "acc": float((br_a < gt_a).sum() / len(br_a)),
        }
    if gt_flows and lateral_flows:
        gt_a, la_a = np.asarray(gt_flows), np.asarray(lateral_flows)
        summary["methodD"]["lateral_flow"] = {
            "gt_flow_mean": float(gt_a.mean()),
            "lateral_flow_mean": float(la_a.mean()),
        }

    # Method D (enhanced): directional / motion-command discrimination.
    # For each motion metric, report how often the bad trajectory's generated
    # video deviates "more" than GT in the direction the perturbed *command*
    # implies (action-following / self-consistency check).
    DIR_DEFS = {
        # (metric, sign) : sign>0 -> bad should have HIGHER |metric| than gt
        "wrong_turn":    ("rot", 1),
        "lateral_shift": ("lat_asym", 1),
        "hard_brake_stop": ("mean_flow", -1),  # bad should have LOWER mean_flow
        "collision":     ("rot", 1),
    }
    summary["methodD"]["action_consistency"] = {}
    for kind, (metric, sign) in DIR_DEFS.items():
        pairs = []
        for s in per_scene_active:
            if "S_gt" not in s or f"S_{kind}" not in s:
                continue
            gt = s["S_gt"]["flow"].get(metric)
            bad = s[f"S_{kind}"]["flow"].get(metric)
            if gt is None or bad is None:
                continue
            # "follows command": bad deviates from gt in the commanded direction
            deviation = abs(bad) - abs(gt) if sign > 0 else gt - bad
            pairs.append((gt, bad, deviation))
        if pairs:
            devs = np.asarray([p[2] for p in pairs])
            summary["methodD"]["action_consistency"][kind] = {
                "metric": metric,
                "sign": sign,
                "n": int(len(pairs)),
                "mean_deviation": float(devs.mean()),
                "n_follows": int((devs > 0).sum()),
                "acc": float((devs > 0).sum() / len(pairs)),
            }

    payload = {"args": _args_dict(), "summary": summary, "scenes": results}
    (out_root / "report.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    LOGGER.info("Report written: %s", out_root / "report.json")


def _args_dict() -> dict:
    import sys
    argv = sys.argv
    return {"argv": " ".join(argv)}


if __name__ == "__main__":
    main()
