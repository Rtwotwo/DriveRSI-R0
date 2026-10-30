"""Publish original NAVSIM photographs and source context without simulation."""

import base64
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

import numpy as np

from driversi.data.scenes.context import read_map_context
from driversi.data.sources.navsim import load_log, transform_box, yaw_from_quaternion
from driversi.data.assets.sensors import _asset_path, extract_sensor_manifest
from driversi.rendering.visual.source_projection import project_source_box

_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")


def select_source_window(frames: list[dict], start: int, count: int) -> list[dict]:
    if (type(start) is not int or type(count) is not int or start < 0 or count < 2
            or count > 121 or start + count > len(frames)):
        raise ValueError("Require 2–121 existing consecutive source frames; no padding")
    window = frames[start:start + count]
    identity = window[0]["log_name"], window[0]["map_location"]
    if any((f["log_name"], f["map_location"]) != identity for f in window):
        raise ValueError("Source preview cannot cross log or map boundaries")
    intervals = np.diff([f["timestamp"] for f in window]) / 1e6
    if np.any(intervals <= 0) or np.any(intervals > .75):
        raise ValueError("Source preview cannot cross a time gap larger than 0.75 s")
    return window


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _scene(selection: dict, dataset: Path, output: Path, *, trusted: bool) -> dict:
    source_id = selection.get("id")
    if not isinstance(source_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", source_id):
        raise ValueError("Scene id must be a safe short identifier")
    split = selection.get("split", "mini")
    if split not in {"mini", "trainval"}:
        raise ValueError("Source gallery supports explicit mini or trainval source partitions")
    _, log_file = _asset_path(dataset / "navsim_logs" / split, selection["log"], "log")
    frames = load_log(log_file, trusted=trusted)
    start, count = selection["start_frame"], selection["frames"]
    window = select_source_window(frames, start, count)
    sensor_root = (dataset / "sensor_blobs" / split).resolve()
    origin = np.asarray(window[0]["ego2global_translation"], dtype=float)
    t0 = int(window[0]["timestamp"])
    result, assets, positions = [], [], []
    for offset, frame in enumerate(window):
        timestamp = int(frame["timestamp"])
        manifest = extract_sensor_manifest(frame, sensor_root, origin_xyz=origin)
        calibration_path = Path("calibration") / source_id / f"{start+offset:06d}.json"
        _json(output / calibration_path, manifest)
        translation = np.asarray(frame["ego2global_translation"], dtype=float)
        yaw = yaw_from_quaternion(frame["ego2global_rotation"])
        objects = []
        for token, kind, box in zip(frame["anns"]["track_tokens"], frame["anns"]["gt_names"],
                                    frame["anns"]["gt_boxes"], strict=True):
            pose = transform_box(box, translation, yaw)
            pose[:3] -= origin
            objects.append({"id": str(token), "kind": str(kind), "pose": pose.tolist(),
                            "dimensions": np.asarray(box[3:6]).tolist()})
            positions.append(pose[:2])
        ego_pose = [float(translation[0]-origin[0]), float(translation[1]-origin[1]), yaw]
        positions.append(np.asarray(ego_pose[:2]))
        cameras = []
        by_name = {c["camera_id"]: c for c in manifest["cameras"]}
        for name in dict.fromkeys([*_CAMERAS, *by_name]):
            camera = by_name.get(name)
            if camera is None:
                cameras.append({"id": name, "available": False, "image": None,
                                "width": None, "height": None, "projection": [],
                                "note": "相机字段未提供"})
                continue
            asset = camera["asset"]
            _, image_path = _asset_path(sensor_root, asset["relative_path"], "camera asset")
            available = asset["file_exists"] and asset["dimensions_status"] == "header_read_only"
            projections, relative_image = [], None
            if available:
                # Use our safe name, preserving bytes and extension; source paths stay in manifest.
                extension = image_path.suffix.lower()
                if extension not in {".jpg", ".jpeg", ".png"}:
                    raise ValueError("Only original PNG/JPEG source assets are supported")
                relative_image = Path("images") / source_id / f"{start+offset:06d}_{name}{extension}"
                if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                    raise ValueError("Unsafe source camera id")
                destination = output / relative_image
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, destination)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                assets.append({"image": relative_image.as_posix(), "sha256": digest,
                               "source_relative_path": asset["relative_path"],
                               "source_index": start+offset, "camera_id": name})
                for obj, box in zip(objects, frame["anns"]["gt_boxes"], strict=True):
                    segments = project_source_box(box, camera["intrinsics"],
                                                   camera["T_ego_from_camera"],
                                                   asset["width"], asset["height"])
                    if segments:
                        projections.append({"id": obj["id"], "kind": obj["kind"], "segments": segments})
            cameras.append({"id": name, "available": bool(available),
                            "image": relative_image.as_posix() if relative_image else None,
                            "width": asset["width"], "height": asset["height"],
                            "projection": projections,
                            "note": "源照片；pinhole 投影未应用畸变" if available
                            else f"未提供/无法读取: {asset['dimensions_status']}"})
        result.append({"source_index": start+offset, "timestamp_us": timestamp,
                       "time_seconds": (timestamp-t0)/1e6, "ego_pose": ego_pose,
                       "objects": objects, "cameras": cameras,
                       "calibration": calibration_path.as_posix(),
                       "signals": {"available": "traffic_lights" in frame,
                                   "records": [[str(i), bool(red)] for i, red in frame.get("traffic_lights", [])]}})
    points = np.stack(positions)
    bounds = np.r_[points.min(axis=0)-20., points.max(axis=0)+20.]
    features, layers, map_provenance = read_map_context(
        window[0]["map_location"], dataset / "maps", origin, bounds)
    intervals = np.diff([f["timestamp"] for f in window]) / 1e6
    return {"id": source_id, "label": str(selection.get("label", source_id)),
            "log_name": window[0]["log_name"], "map_name": window[0]["map_location"],
            "source_file": str(log_file), "source_sha256": hashlib.sha256(log_file.read_bytes()).hexdigest(),
            "source_rate_hz": float(1/np.median(intervals)), "frames": result,
            "source_interval_seconds": {"min": float(intervals.min()), "max": float(intervals.max())},
            "assets": assets, "origin_xyz": origin.tolist(),
            "map": {"features": features, "layers": layers, "bounds": bounds.tolist(),
                    "provenance": map_provenance},
            "timing_scope": "frame/LiDAR timestamp; individual camera exposure timestamps unavailable",
            "annotation_scope": "source ego boxes; planar yaw BEV; pinhole wireframes not visible contours"}


def build_source_gallery(config_path: Path, dataset_root: Path, output_dir: Path, *,
                         trusted: bool = False, diagnostic_record: Path | None = None) -> dict:
    """Reproducible source-photo gallery. No dynamics, fake views, or image alteration."""
    from driversi.rendering.visual.source_gallery import export_source_gallery

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    selections = config.get("scenes")
    if (config.get("schema_version") != 1 or not isinstance(selections, list) or not selections
            or len(selections) > 16 or len({s.get("id") for s in selections}) != len(selections)):
        raise ValueError("Invalid source gallery selection")
    if not trusted:
        raise ValueError("Only load trusted NAVSIM pickle logs with --trusted-pickle")
    if output_dir.exists():
        raise FileExistsError("Source gallery output directory must be new")
    dataset = dataset_root.expanduser().resolve()
    output_dir.mkdir(parents=True)
    manifest = {"schema_version": 1, "kind": "source_camera_gallery_not_simulation",
                "status": "building", "configuration": config, "scenes": []}
    _json(output_dir / "manifest.json", manifest)
    bundle = {"schema_version": 1, "kind": manifest["kind"], "scenes": []}
    try:
        for selection in selections:
            scene = _scene(selection, dataset, output_dir, trusted=trusted)
            bundle["scenes"].append(scene)
            manifest["scenes"].append({
                "id": scene["id"], "map_name": scene["map_name"],
                "source_file": scene["source_file"], "source_sha256": scene["source_sha256"],
                "frames": len(scene["frames"]), "duration_seconds": scene["frames"][-1]["time_seconds"],
                "images_copied": len(scene["assets"]), "assets": scene["assets"],
                "missing_cameras": sum(not c["available"] for f in scene["frames"] for c in f["cameras"]),
            })
            _json(output_dir / "manifest.json", manifest)
        if diagnostic_record is not None:
            from driversi.rendering.visual.visual_replay import _png
            with np.load(diagnostic_record, allow_pickle=False) as record:
                meta = json.loads(str(record["metadata"]))
                if meta.get("modality") != "primitive_rgb":
                    raise ValueError("Comparison illustration must be labeled primitive_rgb")
                rgb = record["rgb"][0, 0, -1, 0]
            image_name = "primitive-reference.png"
            (output_dir / image_name).write_bytes(base64.b64decode(_png(rgb).split(",", 1)[1]))
            bundle["diagnostic_image"] = image_name
            manifest["diagnostic_reference"] = {"path": str(diagnostic_record),
                                                "scope": "unpaired modality illustration only"}
        _json(output_dir / "source-gallery.json", bundle)
        export_source_gallery(bundle, output_dir / "index.html")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(output_dir / "manifest.json", manifest)
        raise
    _json(output_dir / "manifest.json.partial", manifest)
    (output_dir / "manifest.json.partial").replace(output_dir / "manifest.json")
    return manifest
