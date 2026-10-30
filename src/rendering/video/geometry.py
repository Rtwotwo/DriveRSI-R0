"""Convert a sealed executed world to calibrated per-frame video conditions."""

import json
import pickle
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.rendering.gaussian.mtgs_registration import motion_in_source, move_camera
from driversi.rendering.gaussian.worldengine_replay import physical_delta
from driversi.rendering.video.bundle import CAMERAS, FUTURE, HISTORY, load_physics, validate_bundle

# mmdet3d LiDARInstance3DBoxes corner order, centred, x forward/y left/z up.
SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [1, 1, -1],
    ],
    float,
)


def rigid(pose):
    result = np.eye(4)
    result[:3] = pose
    if (
        result.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6)
    ):
        raise ValueError("Finite calibrated rigid transform required")
    return result


def box_matrix(raw, token):
    annotation = raw["anns"]
    i = list(annotation["track_tokens"]).index(token)
    box = np.asarray(annotation["gt_boxes"][i], float)
    c, s = np.cos(box[6]), np.sin(box[6])
    local = np.eye(4)
    local[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    local[:3, 3] = box[:3]
    return np.asarray(raw["ego2global"], float) @ local


def build_geometry(bundle_path, output, *, source_control=False, dense=False):
    bundle = validate_bundle(bundle_path)
    branch, _, scene, states, presence = load_physics(
        bundle["branch_report"], bundle["trajectory_window"]
    )
    raw_path = Path(scene.metadata["navsim_source"]["log_path"])
    with raw_path.open("rb") as f:
        raw_frames = pickle.load(f)
    root = Path(scene.metadata["navsim_source"]["sensor_root"])
    start = scene.metadata["source_start_frame"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {str(Path(bundle_path).resolve()): digest(bundle_path), **bundle["files"]}
    rows = []
    anchors = {}
    origin = np.asarray(scene.metadata["origin_xyz"])
    ref = raw_frames[start]
    ref_pose = np.asarray(ref["ego2global"], float)
    refs = {}
    for camera in CAMERAS:
        image = (root / ref["cams"][camera]["data_path"]).resolve(strict=True)
        files[str(image)] = digest(image)
        refs[camera] = str(image)
    if source_control and dense:
        raise ValueError("Source image controls require actual capture times")
    for tick in range(0, 71, 1 if dense else 5):
        raw = raw_frames[start + tick // 5]
        if (
            tick % 5 == 0
            and abs(
                int(raw["timestamp"])
                - (scene.metadata["source_start_timestamp_us"] + tick * 100000)
            )
            > 50000
        ):
            raise ValueError("Capture/physics time mismatch")
        source = np.asarray(raw["ego2global"], float)
        if tick % 5:
            from scipy.spatial.transform import Rotation, Slerp

            following = np.asarray(raw_frames[start + tick // 5 + 1]["ego2global"], float)
            f = (
                scene.metadata["source_start_timestamp_us"] + tick * 100000 - int(raw["timestamp"])
            ) / float(int(raw_frames[start + tick // 5 + 1]["timestamp"]) - int(raw["timestamp"]))
            if not 0 <= f <= 1:
                raise ValueError("Interpolation outside source support")
            source = source.copy()
            source[:3, :3] = Slerp(
                [0, 1], Rotation.from_matrix([source[:3, :3], following[:3, :3]])
            )([f]).as_matrix()[0]
            source[:3, 3] = (1 - f) * source[:3, 3] + f * following[:3, 3]
        frame = {"ego2global": source, "ego2global_original": source}
        delta = (
            np.zeros(3)
            if source_control
            else np.asarray(physical_delta(states[tick, 0], scene.poses[tick, 0]))
        )
        pivot = source[:3, 3] + source[:3, 0] * 1.461
        target = rigid(move_camera(frame, source[:3], pivot, delta[:2], delta[2]))
        if not source_control:
            yaw_delta = states[tick, 0, 3] - np.arctan2(target[1, 0], target[0, 0])
            c, s = np.cos(yaw_delta), np.sin(yaw_delta)
            target[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ target[:3, :3]
            target[:2, 3] = origin[:2] + states[tick, 0, :2] - 1.461 * target[:2, 0]
        actors = []
        for i, token in enumerate(scene.actor_ids[1:], 1):
            if token in raw["anns"]["track_tokens"] and (
                token not in anchors
                or np.linalg.norm(states[tick, i, :2] - states[0, i, :2]) > 1e-4
            ):
                anchors[token] = (raw, (tick // 5) * 5)
            if not presence[tick, i]:
                continue
            if token not in anchors:
                raise ValueError("Present actor lacks a past appearance anchor: " + token)
            observation, at = anchors[token]
            pose = box_matrix(observation, token)
            ref_scene_pose = scene.poses[at, i]
            actor_delta = (
                np.zeros(3)
                if source_control
                else np.asarray(physical_delta(states[tick, i], ref_scene_pose))
            )
            ref_source = np.asarray(observation["ego2global"], float)
            translation, rotation, _ = motion_in_source(
                {"ego2global": ref_source, "ego2global_original": ref_source},
                actor_delta[:2],
                actor_delta[2],
            )
            moved = pose.copy()
            moved[:3, :3] = rotation @ pose[:3, :3]
            moved[:3, 3] += translation
            if not source_control:
                yaw_delta = states[tick, i, 3] - np.arctan2(moved[1, 0], moved[0, 0])
                c, s = np.cos(yaw_delta), np.sin(yaw_delta)
                moved[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ moved[:3, :3]
                moved[:2, 3] = origin[:2] + states[tick, i, :2]
            corners = SIGNS * scene.dimensions[i] / 2
            world = corners @ moved[:3, :3].T + moved[:3, 3]
            local = (world - target[:3, 3]) @ target[:3, :3]
            actors.append(
                dict(
                    actor_id=token,
                    kind=scene.metadata["source_actor_kinds"][token],
                    physical_pose=states[tick, i, :4].tolist(),
                    source_anchor_tick=at,
                    source_anchor_token=observation["token"],
                    corners_world=world.tolist(),
                    corners_ego=local.tolist(),
                    elevation_status="unverified_source_observation_transport",
                    dimensions=scene.dimensions[i].tolist(),
                )
            )
        if source_control:
            # A source reconstruction control must use every actual source
            # annotation, including source-only actors, without retained physics.
            actors = []
            for token, box, kind in zip(
                raw["anns"]["track_tokens"], raw["anns"]["gt_boxes"], raw["anns"]["gt_names"]
            ):
                box = np.asarray(box, float)
                pose = box_matrix(raw, token)
                world = (SIGNS * box[3:6] / 2) @ pose[:3, :3].T + pose[:3, 3]
                local = (world - target[:3, 3]) @ target[:3, :3]
                actors.append(
                    dict(
                        actor_id=token,
                        kind=str(kind),
                        physical_pose=None,
                        source_anchor_tick=tick,
                        source_anchor_token=raw["token"],
                        corners_world=world.tolist(),
                        corners_ego=local.tolist(),
                        elevation_status="canonical_source_3d_annotation",
                        dimensions=box[3:6].tolist(),
                    )
                )
        cams = {}
        for name in CAMERAS:
            cam = raw["cams"][name]
            extr = np.eye(4)
            extr[:3, :3] = cam["sensor2lidar_rotation"]
            extr[:3, 3] = cam["sensor2lidar_translation"]
            to_world = target @ extr
            image = (root / cam["data_path"]).resolve(strict=True)
            files[str(image)] = digest(image)
            cams[name] = dict(
                intrinsic=np.asarray(cam["cam_intrinsic"]).tolist(),
                distortion=np.asarray(cam["distortion"]).tolist(),
                camera_to_ego=extr.tolist(),
                camera_to_world=to_world.tolist(),
                source_image=str(image),
                source_image_role="reference_only_at_initial_tick_other_frames_evaluation_only",
            )
        rows.append(
            dict(
                tick=tick,
                timestamp_us=scene.metadata["source_start_timestamp_us"] + tick * 100000,
                source_timestamp_us=int(raw["timestamp"]),
                source_token=raw["token"],
                ego_to_world=target.tolist(),
                source_ego_to_world=source.tolist(),
                delta_xy_yaw=delta.tolist(),
                actors=actors,
                cameras=cams,
                ground_alignment_certified=False,
            )
        )
    report = dict(
        schema_version=1,
        kind="calibrated_video_conditions",
        branch_identity=bundle["branch_identity"],
        source_log=bundle["source_log"],
        source_control=source_control,
        dense=dense,
        camera_order=list(CAMERAS),
        history_ticks=list(HISTORY),
        future_ticks=list(FUTURE),
        frames=rows,
        initial_reference_images=refs,
        initial_reference_ego_to_world=ref_pose.tolist(),
        map_file=scene.metadata["map_file"],
        map_location=scene.metadata["map_name"],
        bundle=str(Path(bundle_path).resolve()),
        files=files,
        trainable=False,
        geometry_limitations=[
            "Native nuPlan planar-map height assumption; independent terrain registration still required",
            "Full observed actor orientation retained; XY/yaw transported from exact executed physics",
            "Raw calibrated camera rig, not reconstruction-optimized rig; no mixing of intrinsics/extrinsics",
        ],
    )
    report["files"][scene.metadata["map_file"]] = digest(scene.metadata["map_file"])
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
