"""Actual four-frame contexts and executed ego-relative labels, never future inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def four_frame_ticks(anchor_tick: int, stride: int) -> tuple[int, ...]:
    if (
        type(anchor_tick) is not int
        or type(stride) is not int
        or stride < 1
        or anchor_tick < 3 * stride
    ):
        raise ValueError("Four distinct past/current captures require anchor >= 3 * stride")
    return tuple(anchor_tick - i * stride for i in (3, 2, 1, 0))


def executed_label(states, actor_index: int, anchor_tick: int, target_ticks, dt: float):
    states = np.asarray(states, dtype=float)
    ticks = np.asarray(target_ticks)
    if (
        states.ndim != 3
        or states.shape[-1] < 4
        or not np.isfinite(states).all()
        or type(actor_index) is not int
        or not 0 <= actor_index < states.shape[1]
        or type(anchor_tick) is not int
        or not 0 <= anchor_tick < len(states)
        or ticks.ndim != 1
        or not len(ticks)
        or ticks.dtype.kind not in "iu"
        or ticks[0] <= anchor_tick
        or ticks[-1] >= len(states)
        or np.any(np.diff(ticks) <= 0)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid executed trajectory indexing/time")
    pose = states[anchor_tick, actor_index, [0, 1, 3]]
    xy = states[ticks, actor_index, :2] - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    local = xy @ np.asarray([[c, -s], [s, c]])
    yaw = states[ticks, actor_index, 3] - pose[2]
    return {
        "times": ((ticks - anchor_tick) * dt).tolist(),
        "poses": np.column_stack((local, np.arctan2(np.sin(yaw), np.cos(yaw)))).tolist(),
        "coordinate_frame": "vehicle_center_at_observation",
        "kind": "executed",
    }


def source_history(frames, scene, sensor_root, *, camera_ids):
    """Four real frames ending at initialization, preserving actual timestamps.

    This is the source context for candidate generation; it does not assert that
    a rule generator consumed RGB, nor that these images describe a later branch.
    """
    from driversi.data.assets.sensors import extract_sensor_manifest

    start = scene.metadata["source_start_frame"]
    if type(start) is not int or not 3 <= start < len(frames):
        raise ValueError("Four source history frames are unavailable")
    selected = frames[start - 3 : start + 1]
    stamps = [int(f["timestamp"]) for f in selected]
    if (
        stamps[-1] != scene.metadata["source_start_timestamp_us"]
        or selected[-1]["token"] != scene.metadata["scenario_id"]
        or len({f["token"] for f in selected}) != 4
        or any(
            f["log_name"] != scene.metadata["source_log"]
            or f["map_location"] != scene.metadata["map_name"]
            for f in selected
        )
        or any(not 0 < b - a <= 750000 for a, b in zip(stamps, stamps[1:]))
    ):
        raise ValueError("Source history identity or timestamp continuity mismatch")
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("Declare unique history cameras")
    rows = []
    for frame in selected:
        manifest = extract_sensor_manifest(
            frame, Path(sensor_root), origin_xyz=scene.metadata["origin_xyz"]
        )
        cameras = {c["camera_id"]: c for c in manifest["cameras"]}
        if not set(camera_ids) <= cameras.keys():
            raise ValueError("Required source camera is absent")
        views = []
        for name in camera_ids:
            camera = cameras[name]
            asset = camera["asset"]
            if not asset["file_exists"] or asset["dimensions_status"] != "header_read_only":
                raise ValueError("Source image missing/unreadable")
            path = (Path(sensor_root) / asset["relative_path"]).resolve()
            views.append(
                {
                    "camera_id": name,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "calibration": camera,
                }
            )
        rows.append(
            {
                "source_token": frame["token"],
                "timestamp_us": int(frame["timestamp"]),
                "relative_seconds": (int(frame["timestamp"]) - stamps[-1]) / 1e6,
                "T_scene_from_ego": manifest["T_scene_from_ego"],
                "cameras": views,
            }
        )
    return {
        "kind": "four_real_source_frames",
        "source_log": scene.metadata["source_log"],
        "anchor_token": scene.metadata["scenario_id"],
        "frames": rows,
        "span_seconds": (stamps[-1] - stamps[0]) / 1e6,
        "rgb_used_by_generator": False,
        "camera_timestamp_semantics": "source_lidar_frame_time",
        "permission": "initial_logged_context_only_not_later_counterfactual_input",
    }


def export_four_frame_sample(
    manifest_path,
    output,
    *,
    actor_id,
    anchor_tick,
    history_stride=5,
    future_steps=40,
    target_stride=5,
    include_future_images=False,
):
    """Verify the full render episode then separate four-frame inputs and labels.

    All images are real captures from this simulated branch, not source-image
    references. The existing strict episode validator owns physics/calibration
    validation. Quality gates remain unchanged; this produces inspection samples.
    """
    from driversi.data.assets.mtgs_episode import export_mtgs_episode

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    history = four_frame_ticks(anchor_tick, history_stride)
    if (
        type(future_steps) is not int
        or future_steps < 1
        or type(target_stride) is not int
        or target_stride < 1
        or future_steps % target_stride
        or type(include_future_images) is not bool
    ):
        raise ValueError("Future horizon must be positive and divisible by target stride")
    if output.exists():
        raise FileExistsError(output)
    # Validate before publishing any sample. The episode is durable validation
    # evidence even when a requested window is unavailable.
    episode = export_mtgs_episode(manifest_path, output / "episode")
    if len(episode["generations"]) != 1:
        raise ValueError("Export one branch generation at a time; never join resets")

    def rows(kind):
        return [
            json.loads(line)
            for line in (output / "episode" / f"generation-000-{kind}.jsonl")
            .read_text()
            .splitlines()
        ]

    physics, sensors = rows("physics"), rows("sensors")
    frames = {f["tick"]: f for f in sensors}
    if actor_id not in episode["controlled_ids"]:
        raise ValueError("Observation owner must be a controlled actor")
    owner_cameras = [c for c in episode["camera_ids"] if episode["camera_owners"][c] == actor_id]
    if not owner_cameras or any(t not in frames for t in history):
        raise ValueError("Four requested owner captures are unavailable; no interpolation/padding")
    end = anchor_tick + future_steps
    if end > len(physics):
        raise ValueError("Requested future is unavailable; no automatic horizon shortening")
    target_ticks = list(range(anchor_tick + target_stride, end + 1, target_stride))
    if include_future_images and any(t not in frames for t in target_ticks):
        raise ValueError("Future camera capture missing")
    slot, actor_index = (
        episode["controlled_ids"].index(actor_id),
        episode["actor_ids"].index(actor_id),
    )
    states = [physics[0]["privileged"]["world_before"]["state"]] + [
        r["privileged"]["world_after"]["state"] for r in physics
    ]
    dt = episode["dt_seconds"]
    root = Path(episode["source"]["artifact_root"])

    def capture(t):
        frame = frames[t]
        views = []
        for c in frame["cameras"]:
            if c["camera_id"] not in owner_cameras:
                continue
            artifacts = {
                k: {**c["artifacts"][k], "path": str(root / c["artifacts"][k]["path"])}
                for k in ("image", "valid")
            }
            views.append(
                {
                    "camera_id": c["camera_id"],
                    "owner_actor_id": actor_id,
                    **artifacts,
                    "calibration": {
                        key: c["capture_metadata"][key]
                        for key in (
                            "width",
                            "height",
                            "K",
                            "distortion",
                            "camera_model",
                            "camera_to_body",
                            "camera_to_body_semantics",
                            "camera_pose_axes",
                        )
                        if key in c["capture_metadata"]
                    },
                }
            )
        return {
            "tick": t,
            "timestamp_us": frame["simulation_timestamp_us"],
            "relative_seconds": (t - anchor_tick) * dt,
            "source": "rendered_world",
            "cameras": views,
        }

    target = executed_label(states, actor_index, anchor_tick, target_ticks, dt)
    label = output / "trajectory.json"
    label.write_text(json.dumps(target, allow_nan=False, indent=2))
    span = physics[history[0] : end]
    safety = {
        event: any(row["events"][event][slot] for row in span) for event in ("collision", "offroad")
    }
    owner_valid = all(row["masks"]["valid"][slot] for row in span) and bool(
        physics[end - 1]["privileged"]["world_after"]["alive"][slot]
    )
    reasons = [
        "independent_teacher_score_missing",
        "generator_not_approved_teacher",
        "source_rendering_not_training_qualified",
    ]
    if not owner_valid:
        reasons.append("owner_inactive_within_sample")
    reasons.extend(event for event, failed in safety.items() if failed)
    sample = {
        "schema_version": 1,
        "kind": "four_frame_counterfactual_sample",
        "trainable": False,
        "actor_id": actor_id,
        "anchor_tick": anchor_tick,
        "history_ticks": list(history),
        "history_span_seconds": 3 * history_stride * dt,
        "future_seconds": future_steps * dt,
        "input": {"frames": [capture(t) for t in history], "future_images_allowed": False},
        "target": {
            "trajectory": {
                "path": str(label),
                "sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
            },
            "future_frames": [capture(t) for t in target_ticks] if include_future_images else [],
        },
        "permissions": {
            "input": "past_and_current_observations",
            "target": "supervision_only",
            "episode": "privileged_world_and_joint_actions_not_visual_actor_input",
        },
        "safety": {**safety, "owner_valid_throughout": owner_valid},
        "rejection_reasons": reasons,
        "source": episode["source"],
        "dt_seconds": dt,
        "official_score": None,
        "rgb_used_by_behavior_policy": sensors[0]["used_for_policy"],
        "quality_audit": {str(t): frames[t]["render_quality"] for t in history},
    }
    (output / "sample.json").write_text(json.dumps(sample, allow_nan=False, indent=2))
    return sample


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor", default="ego")
    parser.add_argument("--anchor-tick", type=int, required=True)
    parser.add_argument("--history-stride", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=40)
    parser.add_argument("--target-stride", type=int, default=5)
    parser.add_argument("--future-images", action="store_true")
    args = parser.parse_args(argv)
    result = export_four_frame_sample(
        args.session,
        args.output,
        actor_id=args.actor,
        anchor_tick=args.anchor_tick,
        history_stride=args.history_stride,
        future_steps=args.future_steps,
        target_stride=args.target_stride,
        include_future_images=args.future_images,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("history_ticks", "history_span_seconds", "future_seconds", "trainable")
            }
        )
    )


if __name__ == "__main__":
    main()
