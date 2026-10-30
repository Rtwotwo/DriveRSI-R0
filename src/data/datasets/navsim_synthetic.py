"""OpenScene/NAVSIM logs and SimScale-style history/future sensor packages.

This is an inspection exporter, not a training-admission bypass. No original
future pixels/poses are reused. Only ego has a calibrated rear-axle convention.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from driversi.data.assets.sensors import SOURCE_EGO_TO_CENTER_METRES
from driversi.data.datasets.history_windows import export_four_frame_sample

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
VERSION = "navsim-simscale-synthetic-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rear_axle_states(states, ego_index, offset=SOURCE_EGO_TO_CENTER_METRES):
    """Convert centre poses, including rotating offset; never translate by global x."""
    state = np.asarray(states, dtype=float)
    if state.ndim != 3 or state.shape[2] < 7 or not np.isfinite(state).all():
        raise ValueError("Malformed world states")
    pose = state[:, ego_index, :4].copy()
    pose[:, 0] -= offset * np.cos(pose[:, 3])
    pose[:, 1] -= offset * np.sin(pose[:, 3])
    return pose


def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def frame_geometry(scene, world, rear_pose):
    """All present world objects, in the current simulated rear-axle frame."""
    state = np.asarray(world["state"], dtype=float)
    present = np.asarray(world["presence"], dtype=bool)
    indices = [i for i, token in enumerate(scene.actor_ids) if token != "ego" and present[i]]
    rot = _rotation(rear_pose[3])
    xyz = (state[indices, :3] - rear_pose[:3]) @ rot
    yaw = state[indices, 3] - rear_pose[3]
    boxes = np.column_stack((xyz, scene.dimensions[indices], np.arctan2(np.sin(yaw), np.cos(yaw))))
    names = {1: "vehicle", 2: "pedestrian", 3: "bicycle", 10: "generic_object"}
    return {
        "gt_boxes": boxes.astype(np.float32),
        "gt_names": [names.get(int(scene.kinds[i]), "generic_object") for i in indices],
        "gt_velocity_3d": (state[indices, 4:7] @ rot).astype(np.float32),
        "track_tokens": [scene.actor_ids[i] for i in indices],
    }


def _copy_checked(item, destination):
    path = Path(item["path"])
    if _sha(path) != item["sha256"]:
        raise ValueError("Camera artifact changed after validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    if _sha(destination) != item["sha256"]:
        raise ValueError("Copied camera artifact failed integrity check")


def causal_signal_records(timeline, timestamp_us):
    """Preserve source signal observations; no interpolation or future lookahead."""
    if timeline is None:
        return [], False
    if not isinstance(timeline, list):
        raise ValueError("Signal timeline must be a list")
    previous = -1
    current = None
    available = False
    for row in timeline:
        if (
            not isinstance(row, dict)
            or set(row) != {"timestamp_us", "records", "available"}
            or type(row["available"]) is not bool
            or type(row["timestamp_us"]) is not int
            or row["timestamp_us"] <= previous
            or not isinstance(row["records"], list)
        ):
            raise ValueError("Ordered explicit source signal records required")
        previous = row["timestamp_us"]
        records = []
        for item in row["records"]:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or type(item[1]) is not bool
            ):
                raise ValueError("Source signals require connector ID and red boolean")
            records.append([item[0], item[1]])
        if row["timestamp_us"] <= timestamp_us:
            current = records
            available = row["available"]
    return (current, True) if current is not None and available else ([], False)


def export_navsim_sample(
    manifest_path,
    scene_path,
    output,
    *,
    anchor_tick,
    cameras=CAMERAS,
    navigation=None,
    include_future_images=True,
    signal_timeline=None,
):
    """Export four observed histories and eight structured future states at 2 Hz.

    Future RGB is optional; omitting it never shortens the executed target.

    navigation is an optional explicit task declaration (roadblock_ids and
    driving_command). Missing navigation is UNKNOWN and blocks training; it is
    never inferred from the future target. All eight camera keys exist in logs;
    unavailable cameras are None and must remain disabled in SensorConfig.
    """
    if type(include_future_images) is not bool:
        raise ValueError("include_future_images must be boolean")
    output, scene_path = Path(output).resolve(), Path(scene_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    cameras = tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= set(CAMERAS):
        raise ValueError("Select unique NAVSIM ego camera names")
    if "CAM_R0" not in cameras:
        raise ValueError("CAM_R0 is required by the ChainFlow synthetic-log filter")
    from driversi.contracts.scene import Scene

    scene = Scene.load(scene_path)
    if scene.metadata.get("pose_reference") != "vehicle_center":
        raise ValueError("Only declared vehicle-centre input is supported")
    if scene.metadata.get("coordinate_frame") != "map_axes_translated_from_initial_ego":
        raise ValueError("Unknown map coordinate convention")
    if scene.metadata.get("split") not in ("train", "development", "dev"):
        raise ValueError("Synthetic augmentation is restricted to development sources")
    navigation_known = navigation is not None
    navigation = navigation or {"roadblock_ids": [], "driving_command": [0, 0, 0, 0]}
    command = np.asarray(navigation["driving_command"])
    roads = navigation["roadblock_ids"]
    if (
        command.shape != (4,)
        or not np.isin(command, [0, 1]).all()
        or (navigation_known and (command.sum() != 1 or not roads))
        or not isinstance(roads, list)
        or any(not isinstance(x, str) or not x for x in roads)
    ):
        raise ValueError("Navigation must declare roadblock IDs and a four-class one-hot command")
    sample = export_four_frame_sample(
        manifest_path,
        output / "audit",
        actor_id="ego",
        anchor_tick=anchor_tick,
        history_stride=5,
        future_steps=40,
        target_stride=5,
        include_future_images=include_future_images,
    )
    ep = json.loads((output / "audit/episode/manifest.json").read_text())
    if not np.isclose(ep["dt_seconds"], 0.1, rtol=0, atol=1e-10):
        raise ValueError("NAVSIM profile requires 0.5-second captures from 0.1-second ticks")
    if (
        ep["inputs"]["scene"]["sha256"] != _sha(scene_path)
        or tuple(ep["actor_ids"]) != scene.actor_ids
    ):
        raise ValueError("World scene differs from the rendered episode")
    physics = [
        json.loads(line)
        for line in (output / "audit/episode/generation-000-physics.jsonl").read_text().splitlines()
    ]
    worlds = [physics[0]["privileged"]["world_before"]] + [
        p["privileged"]["world_after"] for p in physics
    ]
    states = np.asarray([w["state"] for w in worlds])
    rear = rear_axle_states(states, scene.actor_ids.index("ego"))
    future = sample["target"]["future_frames"]
    if not include_future_images:
        anchor_us = sample["input"]["frames"][-1]["timestamp_us"]
        future = [
            {"tick": tick, "timestamp_us": anchor_us + (tick - anchor_tick) * 100000, "cameras": []}
            for tick in range(anchor_tick + 5, anchor_tick + 41, 5)
        ]
    frames = sample["input"]["frames"] + future
    if len(frames) != 12 or frames[0]["tick"] < 2:
        raise ValueError("Need four history/eight future frames and causal telemetry warmup")
    for frame in frames if include_future_images else frames[:4]:
        available = {c["camera_id"] for c in frame["cameras"]}
        if not set(cameras) <= available:
            raise ValueError("Requested camera was not rendered; never substitute source images")
    signals = [causal_signal_records(signal_timeline, f["timestamp_us"]) for f in frames]
    signals_known = all(known for _, known in signals)
    identity = {
        "version": VERSION,
        "source": sample["source"]["manifest_sha256"],
        "anchor_tick": anchor_tick,
        "cameras": cameras,
        "navigation": navigation,
        "signal_timeline": signal_timeline,
    }
    if not include_future_images:
        identity["future_images"] = False
    clip_id = (
        "driversi-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    )
    log_dir = output / "navsim_logs/synthetic_driversi"
    log_dir.mkdir(parents=True)
    records, checksums, masks = [], {}, []
    origin = np.asarray(scene.metadata["origin_xyz"])
    for idx, capture in enumerate(frames):
        tick, pose = capture["tick"], rear[capture["tick"]]
        token = hashlib.sha256(f"{clip_id}:{tick}".encode()).hexdigest()[:32]
        rotation = _rotation(pose[3])
        velocity = (rear[tick, :3] - rear[tick - 1, :3]) / 0.1
        previous_velocity = (rear[tick - 1, :3] - rear[tick - 2, :3]) / 0.1
        acceleration = (velocity - previous_velocity) / 0.1
        annotations = frame_geometry(scene, worlds[tick], pose)
        annotations["instance_tokens"] = [
            hashlib.sha256(f"{token}:{track}".encode()).hexdigest()[:32]
            for track in annotations["track_tokens"]
        ]
        # Complete camera key set is a NAVSIM dataclass requirement. Missing
        # cameras deliberately fail if a caller enables them.
        camera_dict = {
            name: {
                "data_path": None,
                "sensor2lidar_rotation": None,
                "sensor2lidar_translation": None,
                "cam_intrinsic": None,
                "distortion": None,
            }
            for name in CAMERAS
        }
        sensor_root = "sensor_blobs_hist" if idx < 4 else "sensor_blobs_fut"
        for view in capture["cameras"]:
            name = view["camera_id"]
            if name not in cameras:
                continue
            cal = view["calibration"]
            if cal.get("camera_model") != "opencv_distorted_source_lens":
                raise ValueError("NAVSIM export requires the declared OpenCV source-lens model")
            transform = np.asarray(cal["camera_to_body"], dtype=float).copy()
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Invalid body camera transform")
            transform[0, 3] += SOURCE_EGO_TO_CENTER_METRES
            relative = Path(clip_id) / name / f"{token}.png"
            _copy_checked(view["image"], output / sensor_root / relative)
            mask_path = Path("masks") / relative
            _copy_checked(view["valid"], output / mask_path)
            checksums[str(Path(sensor_root) / relative)] = view["image"]["sha256"]
            checksums[str(mask_path)] = view["valid"]["sha256"]
            masks.append({"frame_idx": idx, "camera": name, "path": str(mask_path)})
            camera_dict[name] = {
                "data_path": relative.as_posix(),
                "sensor2lidar_rotation": transform[:3, :3],
                "sensor2lidar_translation": transform[:3, 3],
                "cam_intrinsic": np.asarray(cal["K"]),
                "distortion": np.asarray(cal["distortion"]),
            }
        annotations["gt_names"] = list(annotations["gt_names"])
        records.append(
            {
                "token": token,
                "frame_idx": idx,
                "timestamp": capture["timestamp_us"],
                "log_name": clip_id,
                "log_token": clip_id,
                "scene_token": clip_id,
                "scene_name": clip_id,
                "map_location": scene.metadata["map_name"],
                "roadblock_ids": list(roads),
                "ego2global_translation": pose[:3] + origin,
                "ego2global_rotation": np.array(
                    [np.cos(pose[3] / 2), 0.0, 0.0, np.sin(pose[3] / 2)]
                ),
                "ego_dynamic_state": np.concatenate(
                    ((velocity @ rotation)[:2], (acceleration @ rotation)[:2])
                ),
                "driving_command": command.copy(),
                "anns": annotations,
                "cams": camera_dict,
                "traffic_lights": signals[idx][0],
                "lidar_path": "",
                "lidar2ego_translation": np.zeros(3),
                "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            }
        )
    log_path = log_dir / f"{clip_id}.pkl"
    # Output contains only locally constructed dictionaries/NumPy arrays, not
    # executable external objects. Reading imported logs is outside this API.
    log_path.write_bytes(pickle.dumps(records, protocol=4))
    checksums[str(log_path.relative_to(output))] = _sha(log_path)
    config = {name.lower(): [0, 1, 2, 3] if name in cameras else False for name in CAMERAS}
    config["lidar_pc"] = False
    blockers = sorted(
        set(
            sample["rejection_reasons"]
            + [
                "render_physics_registration_unverified",
                "lidar_not_synthesized",
                "world_population_not_certified",
            ]
            + ([] if navigation_known else ["navigation_not_provided"])
            + ([] if signals_known else ["signal_states_not_exported"])
        )
    )
    report = {
        "version": VERSION,
        "status": "completed",
        "trainable": False,
        "purpose": "loader_compatibility_inspection_not_training_admission",
        "clip_id": clip_id,
        "log_file": str(log_path.relative_to(output)),
        "token": records[3]["token"],
        "num_history_frames": 4,
        "num_future_frames": 8,
        "interval_length": 0.5,
        "history_span_seconds": 1.5,
        "future_horizon_seconds": 4.0,
        "total_timestamp_span_seconds": 5.5,
        "sensor_config": config,
        "scene_filter": {
            "num_history_frames": 4,
            "num_future_frames": 8,
            "frame_interval": 12,
            "has_route": navigation_known,
            "tokens": [records[3]["token"]],
        },
        "cameras": list(cameras),
        "missing_cameras": sorted(set(CAMERAS) - set(cameras)),
        "image_counts": {
            "history": 4 * len(cameras),
            "future": 8 * len(cameras) if include_future_images else 0,
        },
        "future_images_included": include_future_images,
        "source_log": scene.metadata["source_log"],
        "source_token": scene.metadata["scenario_id"],
        "source_split": scene.metadata["split"],
        "split_verification": scene.metadata.get("split_verification", "unknown"),
        "source_manifest_sha256": sample["source"]["manifest_sha256"],
        "rejection_reasons": blockers,
        "official_score": None,
        "mask_paths": masks,
        "sha256": checksums,
        "semantics": {
            "ego_pose": "map_global_rear_axle_planar",
            "annotation_frame": "current_rear_axle",
            "camera_fields": "optical_camera_to_display_rear_axle_not_verified_physics_registration",
            "telemetry": "causal_backward_difference_of_executed_rear_axle_pose",
            "navigation": "explicit_task"
            if navigation_known
            else "unknown_all_zero_command_empty_route",
            "signals": "latest_causal_source_observation_empty_is_not_verified_green"
            if signals_known
            else "unknown_empty_not_verified_green",
            "lidar": "unavailable",
            "sensor_paths": "same_relative_layout_in_separate_history_and_future_roots",
            "future_pixels": "targets_only_do_not_feed_to_current_actor",
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, allow_nan=False, indent=2))
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-tick", type=int, required=True)
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    parser.add_argument("--navigation", type=Path)
    parser.add_argument("--future-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result = export_navsim_sample(
        args.session,
        args.scene,
        args.output,
        anchor_tick=args.anchor_tick,
        cameras=args.cameras,
        include_future_images=args.future_images,
        navigation=json.loads(args.navigation.read_text()) if args.navigation else None,
    )
    print(json.dumps({k: result[k] for k in ("status", "clip_id", "image_counts", "trainable")}))


if __name__ == "__main__":
    main()
