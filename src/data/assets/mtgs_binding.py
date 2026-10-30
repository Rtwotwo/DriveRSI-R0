"""Exact NAVSIM/MTGS provenance matching; never infer a global registration.

MTGS retains both original and refined poses. Matching tokens and original
poses establishes shared source data, not equality of refined geometry.
This module stays independent of torch and the optional renderer runtime.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from driversi.contracts.scene import Scene


def audit_source_files(
    *,
    log_path: Path,
    metadata_path: Path,
    scene_path: Path,
    travel_id: int,
    camera_name: str = "CAM_F0",
    trusted_pickle: bool = False,
    render_end_tick: int | None = None,
) -> dict:
    """Audit a reproducible source binding without loading torch or changing assets."""
    if not trusted_pickle:
        raise ValueError("Source logs and MTGS metadata require --trusted-pickle")
    if type(travel_id) is not int or travel_id < 0:
        raise ValueError("Traversal ID must be a nonnegative integer")
    paths = [Path(p).resolve(strict=True) for p in (log_path, metadata_path, scene_path)]
    with paths[0].open("rb") as stream:
        frames = pickle.load(stream)
    with paths[1].open("rb") as stream:
        videos = pickle.load(stream)
    matches = [
        video for key, video in videos.items() if int(str(key).rsplit("-", 1)[1]) == travel_id
    ]
    if len(matches) != 1:
        raise ValueError("Traversal must identify exactly one reconstruction video")
    report = match_navsim_traversal(
        frames,
        matches[0],
        camera_name=camera_name,
        scene=Scene.load(paths[2]),
        render_end_tick=render_end_tick,
    )
    report["inputs"] = {}
    for label, path in zip(("navsim_log", "mtgs_metadata", "scene"), paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        report["inputs"][label] = {"path": str(path), "sha256": digest.hexdigest()}
    return report


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError(f"{label} must be a finite rigid 4x4 transform")
    return matrix


def _index(frames: list[dict], label: str) -> dict[str, tuple[int, dict]]:
    index: dict[str, tuple[int, dict]] = {}
    timestamps = []
    for i, frame in enumerate(frames):
        token, timestamp = str(frame["token"]), frame["timestamp"]
        if token in index:
            raise ValueError(f"Duplicate {label} frame token: {token}")
        if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, (int, np.integer)):
            raise ValueError(f"{label} timestamps must be integer microseconds")
        index[token] = (i, frame)
        timestamps.append(int(timestamp))
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{label} timestamps must strictly increase")
    return index


def _objects(frame: dict, label: str) -> dict[str, tuple[str, np.ndarray]]:
    tokens, names, boxes = frame["track_tokens"], frame["gt_names"], frame["gt_boxes"]
    boxes = np.asarray(boxes, dtype=np.float64)
    if (
        len(tokens) != len(names)
        or boxes.shape != (len(tokens), 7)
        or not np.isfinite(boxes).all()
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError(f"Invalid {label} object annotations")
    return {
        str(token): (str(name), box) for token, name, box in zip(tokens, names, boxes, strict=True)
    }


def match_navsim_traversal(
    frames: list[dict],
    video: dict,
    *,
    camera_name: str = "CAM_F0",
    scene: Scene | None = None,
    render_end_tick: int | None = None,
) -> dict:
    """Validate exact source identity and expose reconstruction pose residuals.

    ``scene`` restricts the audit to its compiled source window, including the
    bracketing source endpoint. All raw annotations remain in the inventory;
    a source object missing from the simulation is explicitly marked unbound.
    No world/render transform or training eligibility is granted by this audit.
    """
    if render_end_tick is not None and scene is None:
        raise ValueError("A bounded render interval requires a compiled scene")
    nav_index = _index(frames, "NAVSIM")
    mt_index = _index(video["frame_infos"], "MTGS")
    offset = np.asarray(video["global2world_translation"], dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("MTGS reconstruction offset must be a finite XYZ vector")
    log_name, map_name = str(video["log_name"]), str(video["map_location"])
    lower, upper = None, None
    world_ids: set[str] = set()
    if scene is not None:
        scene.validate()
        if scene.metadata["source_log"] != log_name or scene.metadata["map_name"] != map_name:
            raise ValueError("Scene and reconstruction source identity differ")
        lower = int(scene.metadata["source_start_frame"])
        upper = lower + int(scene.metadata["source_frames"]) - 1
        if render_end_tick is not None:
            if type(render_end_tick) is not int or not 0 < render_end_tick < len(scene.times):
                raise ValueError("Render interval must be within physical source support")
            stamp = int(scene.metadata["source_start_timestamp_us"]) + round(
                float(scene.times[render_end_tick]) * 1e6
            )
            upper = int(np.searchsorted([f["timestamp"] for f in frames], stamp))
        if not 0 <= lower < upper < len(frames):
            raise ValueError("Compiled scene source window is outside NAVSIM log")
        if (
            frames[lower]["token"] != scene.metadata["scenario_id"]
            or int(frames[lower]["timestamp"]) != scene.metadata["source_start_timestamp_us"]
        ):
            raise ValueError("Compiled scene start identity differs from NAVSIM source")
        world_ids = set(scene.actor_ids)
    matches, residual_xyz, residual_degrees = [], [], []
    for token, (mt_i, mt) in mt_index.items():
        if token not in nav_index or mt.get("skipped"):
            continue
        nav_i, nav = nav_index[token]
        if lower is not None and not lower <= nav_i <= upper:
            continue
        if (
            nav["log_name"] != log_name
            or nav["map_location"] != map_name
            or mt["log_name"] != log_name
        ):
            raise ValueError("Matched frame belongs to a different source log or map")
        if int(nav["timestamp"]) != int(mt["timestamp"]):
            raise ValueError("Matched frame tokens have different timestamps")
        nav_camera, mt_camera = nav["cams"][camera_name], mt["cams"][camera_name]
        image = str(nav_camera["data_path"])
        if image != str(mt_camera["data_path"]):
            raise ValueError("Matched source camera image paths differ")
        camera_token = PurePosixPath(image).stem
        if str(mt_camera["token"]) != camera_token:
            raise ValueError("MTGS camera token differs from source image filename")
        original = _matrix(mt["ego2global_original"], "MTGS original pose")
        refined = _matrix(mt["ego2global"], "MTGS refined pose")
        nav_pose = _matrix(nav["ego2global"], "NAVSIM pose").copy()
        nav_pose[:3, 3] -= offset
        if not np.allclose(nav_pose, original, rtol=0, atol=1e-7):
            raise ValueError("NAVSIM pose and translated MTGS original pose disagree")
        correction = refined @ np.linalg.inv(original)
        relative_rotation = refined[:3, :3] @ original[:3, :3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
        displacement = refined[:3, 3] - original[:3, 3]
        residual_xyz.append(displacement)
        residual_degrees.append(angle)
        nav_objects, mt_objects = _objects(nav["anns"], "NAVSIM"), _objects(mt, "MTGS")
        common = sorted(nav_objects.keys() & mt_objects.keys())
        for actor in common:
            nav_kind, nav_box = nav_objects[actor]
            mt_kind, mt_box = mt_objects[actor]
            if nav_kind != mt_kind or not np.allclose(nav_box, mt_box, rtol=0, atol=1e-7):
                raise ValueError(f"Matched source annotations differ for track {actor}")
        inventory = [
            {
                "track_token": actor,
                "kind": nav_objects[actor][0],
                "present_in_mtgs_annotations": actor in mt_objects,
                "physics_bound": actor in world_ids if scene is not None else None,
            }
            for actor in sorted(nav_objects)
        ]
        matches.append(
            {
                "navsim_index": nav_i,
                "mtgs_frame_index": mt_i,
                "token": token,
                "timestamp_us": int(nav["timestamp"]),
                "camera_token": camera_token,
                "camera_timestamp_us": int(mt_camera["timestamp"]),
                "camera_image_path": image,
                "original_pose_local": original.tolist(),
                "refined_pose_local": refined.tolist(),
                "T_refined_from_original_at_source_frame": correction.tolist(),
                "refined_minus_original_translation_m": displacement.tolist(),
                "refined_original_rotation_degrees": angle,
                "source_objects": inventory,
                "mtgs_only_annotation_tracks": sorted(mt_objects.keys() - nav_objects.keys()),
            }
        )
    if not matches:
        raise ValueError("No exact usable NAVSIM/MTGS source-frame matches")
    if scene is not None and [m["navsim_index"] for m in matches] != list(range(lower, upper + 1)):
        raise ValueError("Reconstruction does not cover every compiled scene source frame")
    xyz = np.asarray(residual_xyz)
    return {
        "render_end_tick": render_end_tick,
        "schema_version": 1,
        "kind": "navsim_mtgs_exact_source_binding_audit",
        "log_name": log_name,
        "map_name": map_name,
        "video_token": video["video_token"],
        "travel_id": int(str(video["video_token"]).rsplit("-", 1)[1]),
        "camera_name": camera_name,
        "global2world_translation": offset.tolist(),
        "frames": matches,
        "match_count": len(matches),
        "source_start_timestamp_us": matches[0]["timestamp_us"],
        "source_end_timestamp_us": matches[-1]["timestamp_us"],
        "matched_span_seconds": (matches[-1]["timestamp_us"] - matches[0]["timestamp_us"]) / 1e6,
        "refinement_residual": {
            "translation_axis_min_m": xyz.min(axis=0).tolist(),
            "translation_axis_max_m": xyz.max(axis=0).tolist(),
            "max_translation_norm_m": float(np.linalg.norm(xyz, axis=1).max()),
            "max_rotation_degrees": max(residual_degrees),
        },
        "coordinate_mapping": "original_pose_translation_verified_refined_registration_not_assumed",
        "render_binding_ready": False,
        "trainable": False,
        "limitations": [
            "source_identity_is_not_reconstruction_quality",
            "camera_and_lidar_timestamps_differ",
            "per_frame_refinement_is_not_a_global_rigid_transform",
            "renderer_instance_assets_not_checked",
        ],
    }
