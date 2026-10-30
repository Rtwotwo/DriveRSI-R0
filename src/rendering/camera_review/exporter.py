"""Bind existing RGB to calibrated semantic proxies and complete NAVSIM annotations.

Does not rerender GS, certify novel views, or use logged futures as observations.
"""

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.data.scenes.context import compile_context


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {shape}")
    return result


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def corners(center, size, yaw):
    center, size = array(center, (3,)), array(size, (3,))
    if np.any(size <= 0):
        raise ValueError("Nonpositive object size")
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )
    return (signs * size / 2) @ rotation(yaw).T + center


def transform(points, matrix):
    matrix = array(matrix, (4, 4))
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def validate_camera(camera, actors, tick):
    owner = camera["owner_actor_id"]
    if owner not in {a["id"] for a in actors if a["present"]}:
        raise ValueError("Camera owner is not present")
    if camera.get("camera_pose_axes") != "nerfstudio_right_up_back":
        raise ValueError("Unsupported camera axes")
    if camera.get("camera_model") != "opencv_distorted_source_lens":
        raise ValueError("Unsupported camera lens")
    if camera.get("camera_coordinate_frame") != "mtgs_reconstruction_metres":
        raise ValueError("Camera is not in reconstruction coordinates")
    pose = array(camera["camera_to_world"], (3, 4))
    if not np.allclose(pose[:, :3].T @ pose[:, :3], np.eye(3), atol=2e-5):
        raise ValueError("Non-rigid camera")
    if not np.isclose(np.linalg.det(pose[:, :3]), 1, atol=2e-5):
        raise ValueError("Reflected camera")
    k = array(camera["K"], (3, 3))
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid focal length")
    array(camera["distortion"], (5,))
    if camera.get("virtual") and camera.get("source"):
        raise ValueError("Virtual camera cannot claim real source image")
    if not camera.get("image") or tick < 0:
        raise ValueError("Image and tick required")


def previous_source(frames, timestamp):
    indices = [i for i, f in enumerate(frames) if f["timestamp_us"] <= timestamp]
    if not indices:
        raise ValueError("No causal source annotation")
    result = frames[indices[-1]]
    if timestamp - result["timestamp_us"] > 750000:
        raise ValueError("Source annotation too old")
    return result


def safe_media(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or escaping media path")
    return path


def reference_at(scene, timestamp):
    seconds = (timestamp - scene.metadata["source_start_timestamp_us"]) / 1e6
    if seconds < scene.times[0] - 0.001 or seconds > scene.times[-1] + 0.001:
        raise ValueError("Seed outside reference support")
    seconds = float(np.clip(seconds, scene.times[0], scene.times[-1]))
    left = int(
        np.clip(
            np.searchsorted(scene.times, seconds + 1e-10, side="right") - 1, 0, len(scene.times) - 1
        )
    )
    right = min(left + 1, len(scene.times) - 1)
    ratio = (
        0
        if left == right
        else (seconds - scene.times[left]) / (scene.times[right] - scene.times[left])
    )
    before, after = scene.poses[left], scene.poses[right]
    present = scene.presence_at(left)
    interpolate = present & scene.presence_at(right)
    result = before.copy()
    result[interpolate] += ratio * (after[interpolate] - before[interpolate])
    yaw = np.arctan2(np.sin(after[:, 3] - before[:, 3]), np.cos(after[:, 3] - before[:, 3]))
    result[interpolate, 3] = before[interpolate, 3] + ratio * yaw[interpolate]
    return result, present


def verify_png(path, width, height):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or struct.unpack(">II", header[16:24]) != (width, height)
    ):
        raise ValueError("Camera image format/dimensions mismatch")


def semantic_boxes(scene, frame, camera, seed, mode):
    """Annotation geometry transformed with the recorded branch motion convention.

    These are NOT optimized GS node extents. Unbound objects stay visible as proxies.
    """
    refined, original = np.array(seed["refined"]), np.array(seed["original"])
    correction = refined @ np.linalg.inv(original)
    offset = np.array(scene.metadata["origin_xyz"]) - np.array(seed["global_offset"])
    annotations = dict(zip(seed["ids"], seed["boxes"], strict=True))
    height_rows = next(
        (
            x["actors"]
            for x in frame["height_binding"]["per_camera"]
            if x["camera_id"] == camera["camera_id"]
        ),
        [],
    )
    heights = {x["actor_id"]: x["correction_m"] for x in height_rows}
    references, seed_present = reference_at(scene, seed["timestamp_us"])
    output = []
    for actor in frame["actors"]:
        if not actor["present"]:
            continue
        index = scene.actor_ids.index(actor["id"])
        ref = references[index] if seed_present[index] else scene.poses[frame["tick"], index]
        box = annotations.get(actor["id"])
        if box is not None:
            points = transform(corners(box[:3], box[3:6], box[6]), refined)
            origin = points.mean(axis=0)
            geometry = "source_3d_annotation_not_optimized_gs_extent"
        else:
            center = ref[:3].copy()
            if actor["id"] == "ego":
                center[2] += actor["dimensions"][2] / 2
            points = transform(corners(center + offset, actor["dimensions"], ref[3]), correction)
            origin = points.mean(axis=0)
            geometry = "compiled_reference_proxy_source_box_unavailable"
        delta = np.r_[np.array(actor["position"]) - ref[:2], 0.0]
        yaw = actor["yaw"] - ref[3]
        if mode == "source-frame":
            basis = refined[:3, :3] @ original[:3, :3].T
            rot, delta = basis @ rotation(yaw) @ basis.T, basis @ delta
        else:
            rot = rotation(yaw)
        points = (points - origin) @ rot.T + origin + delta
        points[:, 2] += heights.get(actor["id"], 0.0)
        output.append(
            dict(
                id=actor["id"],
                kind=scene.metadata["source_actor_kinds"].get(actor["id"], "other"),
                corners=points.tolist(),
                controlled=actor["controlled"],
                reconstructed=actor["reconstructed"],
                geometry=geometry,
                height_correction_m=heights.get(actor["id"]),
                hidden_owner=actor["id"] == camera["owner_actor_id"],
            )
        )
    return output, correction, offset


def source_whitebox(sensor, raw_frame, camera_id, width, height):
    """Original NAVSIM geometry with its own raw lens, never branch calibration."""
    record = next(c for c in sensor["cameras"] if c["camera_id"] == camera_id)
    asset = record["asset"]
    if not asset["file_exists"] or asset["dimensions_status"] != "header_read_only":
        raise ValueError("Original camera asset dimensions unavailable")
    pose = np.array(record["T_scene_from_camera"]) @ np.diag([1, -1, -1, 1])
    k = np.array(record["intrinsics"], dtype=float)
    k[0] *= width / asset["width"]
    k[1] *= height / asset["height"]
    distortion = array(record["distortion_coefficients"], (5,))
    annotation = raw_frame["anns"]
    objects = []
    for token, kind, box in zip(
        annotation["track_tokens"], annotation["gt_names"], annotation["gt_boxes"], strict=True
    ):
        objects.append(
            dict(
                id=str(token),
                kind=str(kind),
                corners=transform(
                    corners(box[:3], box[3:6], box[6]), sensor["T_scene_from_ego"]
                ).tolist(),
                hidden_owner=False,
                reconstructed=True,
                geometry="raw_navsim_annotation_not_asset_coverage",
            )
        )
    return dict(
        camera_id=camera_id,
        owner_actor_id="ego",
        camera_to_world=pose[:3].tolist(),
        K=k.tolist(),
        distortion=distortion.tolist(),
        width=width,
        height=height,
        objects=objects,
        map_to_reconstruction=np.eye(4).tolist(),
        map_offset=[0, 0, 0],
        map_ground_z=sensor["T_scene_from_ego"][2][3],
        source_timestamp_us=int(raw_frame["timestamp"]),
        source_frame_token=raw_frame["token"],
        camera_token=Path(asset["relative_path"]).stem,
        semantics="original_raw_NAVSIM_camera_and_annotations; five_coefficient_OpenCV_diagnostic_not_pixel_truth",
    )


def export(
    manifest_path,
    output,
    maps_root,
    metadata_python,
    *,
    trusted=False,
    source_log=None,
    sensor_root=None,
):
    if not trusted:
        raise ValueError("Trusted NAVSIM and MTGS pickle assets require explicit opt-in")
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    scene_entry, metadata_entry = manifest["inputs"]["scene"], manifest["assets"]["source_metadata"]
    for entry in (scene_entry, metadata_entry):
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Source asset hash mismatch")
    scene = Scene.load(scene_entry["path"])
    if manifest["source_log"] != scene.metadata["source_log"]:
        raise ValueError("Scene/log mismatch")
    mode = manifest.get("registration_mode", "legacy-map-axes")
    if mode not in ("legacy-map-axes", "source-frame"):
        raise ValueError("Unknown registration convention")
    frames = manifest["frames"]
    identities = [(f["generation"], f["tick"]) for f in frames]
    if len(set(identities)) != len(frames) or len({x[0] for x in identities}) != 1:
        raise ValueError("Export exactly one generation, no duplicate ticks")
    if any(b[1] <= a[1] for a, b in zip(identities, identities[1:])):
        raise ValueError("Frame time order")
    if any(not f.get("cameras") or f["tick"] != f["render_tick"] for f in frames):
        raise ValueError("Every demo frame requires a bound camera at its world tick")
    tokens = sorted({c["source_seed"]["frame_token"] for f in frames for c in f["cameras"]})
    request = dict(path=metadata_entry["path"], tokens=tokens)
    proc = subprocess.run(
        [str(metadata_python), "-B", str(Path(__file__).with_name("metadata_reader.py"))],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=180,
        check=True,
    )
    seeds = json.loads(proc.stdout)
    log_path = source_log or scene.metadata.get("navsim_source", {}).get("log_path")
    if not log_path:
        raise ValueError("Original source log path is required")
    context = compile_context(scene, Path(log_path), Path(maps_root), trusted=True)
    source = context.frames
    # Include actual ego annotation for source/branch BEV comparison.
    from driversi.data.sources.navsim import load_log, yaw_from_quaternion

    raw = load_log(Path(log_path), trusted=True)
    raw_by_time = {f["timestamp"]: f for f in raw}
    sensors = {}
    for f in source:
        item = raw_by_time[f["timestamp_us"]]
        yaw = yaw_from_quaternion(item["ego2global_rotation"])
        xyz = np.array(item["ego2global_translation"]) - np.array(scene.metadata["origin_xyz"])
        xyz[:2] += 1.461 * np.array([np.cos(yaw), np.sin(yaw)])
        f["objects"].insert(
            0,
            dict(
                id="ego",
                kind="vehicle",
                pose=[*xyz, yaw],
                dimensions=scene.dimensions[scene.actor_ids.index("ego")].tolist(),
            ),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".camera-review-", dir=output.parent) as temporary:
        root = Path(temporary)
        (root / "media").mkdir()
        media, rows, camera_owners = {}, [], {}
        source_photos = {}
        if sensor_root is not None:
            from driversi.data.assets.sensors import _asset_path

            photos = set()
            for f in frames:
                original_frame = raw_by_time[
                    previous_source(source, f["simulation_timestamp_us"])["timestamp_us"]
                ]
                for c in f["cameras"]:
                    if not c["virtual"]:
                        _, raw_path = _asset_path(
                            Path(sensor_root).resolve(),
                            original_frame["cams"][c["camera_id"]]["data_path"],
                            "source_photo",
                        )
                        photos.add((str(raw_path), c["width"], c["height"]))
            proc = subprocess.run(
                [str(metadata_python), "-B", str(Path(__file__).with_name("photo_reader.py"))],
                input=json.dumps(dict(output=str(root), photos=sorted(photos))),
                text=True,
                capture_output=True,
                timeout=180,
                check=True,
            )
            for photo in json.loads(proc.stdout):
                source_photos[tuple(photo["key"])] = (photo["image"], photo["raw_sha256"])
                media[photo["image"]] = photo["sha256"]
        for frame in frames:
            tick = frame["tick"]
            if (
                not 0 <= tick < len(scene.times)
                or abs(frame["time_seconds"] - scene.times[tick]) > 1e-6
            ):
                raise ValueError("World tick/time mismatch")
            source_frame = previous_source(source, frame["simulation_timestamp_us"])
            row = {
                k: frame[k]
                for k in ("generation", "tick", "time_seconds", "simulation_timestamp_us", "actors")
            }
            row["source_index"] = source.index(source_frame)
            row["cameras"] = []
            for camera in frame["cameras"]:
                validate_camera(camera, frame["actors"], tick)
                cid, owner = camera["camera_id"], camera["owner_actor_id"]
                if camera_owners.setdefault(cid, owner) != owner:
                    raise ValueError("Camera owner changes within episode")
                seed = seeds[camera["source_seed"]["frame_token"]]
                if (
                    seed["log_name"] != manifest["source_log"]
                    or seed["timestamp_us"] != camera["source_timestamp_us"]
                ):
                    raise ValueError("Camera reconstruction source mismatch")
                if abs(seed["timestamp_us"] - frame["simulation_timestamp_us"]) > 60000:
                    raise ValueError("Camera source seed too far from world time")
                cam = {
                    k: camera[k]
                    for k in (
                        "camera_id",
                        "owner_actor_id",
                        "virtual",
                        "width",
                        "height",
                        "camera_to_world",
                        "K",
                        "distortion",
                        "camera_timestamp_us",
                        "source_timestamp_us",
                    )
                }
                if not camera["virtual"] and sensor_root is not None:
                    from driversi.data.assets.sensors import extract_sensor_manifest

                    # NAVSIM sampled images are 2 Hz; MTGS can contain extra intermediate
                    # photos. Show actual previous NAVSIM sample and its own camera,
                    # never borrow an interpolated pose for a different source photo.
                    original_frame = raw_by_time[source_frame["timestamp_us"]]
                    identity = original_frame["token"]
                    if identity not in sensors:
                        sensors[identity] = extract_sensor_manifest(
                            original_frame,
                            Path(sensor_root),
                            origin_xyz=scene.metadata["origin_xyz"],
                        )
                    cam["source_whitebox"] = source_whitebox(
                        sensors[identity], original_frame, cid, cam["width"], cam["height"]
                    )
                cam["quality"] = camera.get("quality", {})
                cam["surface_diagnostics"] = {
                    k: camera.get("surface_diagnostics", {}).get(k)
                    for k in (
                        "low_opacity_fraction",
                        "black_opaque_fraction",
                        "opaque_near_surface_fraction",
                    )
                }
                for key in ("image", "source", "valid"):
                    if not camera.get(key):
                        continue
                    if key == "source" and cam.get("source_whitebox"):
                        from driversi.data.assets.sensors import _asset_path

                        _, raw_path = _asset_path(
                            Path(sensor_root).resolve(),
                            original_frame["cams"][cid]["data_path"],
                            "source_photo",
                        )
                        photo_key = (str(raw_path), cam["width"], cam["height"])
                        cam[key], raw_hash = source_photos[photo_key]
                        cam["source_photo"] = dict(
                            raw_file_sha256=raw_hash,
                            frame_timestamp_us=original_frame["timestamp"],
                            camera_token=cam["source_whitebox"]["camera_token"],
                            timing="previous_NAVSIM_sample; camera_exposure_timestamp_not_provided",
                        )
                        continue
                    path = safe_media(manifest_path.parent, camera[key])
                    verify_png(path, camera["width"], camera["height"])
                    checksum = digest(path)
                    target = f"media/{checksum}.png"
                    if target not in media:
                        shutil.copyfile(path, root / target)
                        media[target] = checksum
                    cam[key] = target
                boxes, matrix, offset = semantic_boxes(scene, frame, camera, seed, mode)
                cam["objects"] = boxes
                cam["map_to_reconstruction"] = matrix.tolist()
                cam["map_offset"] = offset.tolist()
                cam["map_ground_z"] = float(scene.poses[tick, scene.actor_ids.index("ego"), 2])
                row["cameras"].append(cam)
            if len({c["camera_id"] for c in row["cameras"]}) != len(row["cameras"]):
                raise ValueError("Duplicate camera identity")
            rows.append(row)
        data = dict(
            schema_version=1,
            kind="camera_bound_scene_inspection",
            trainable=False,
            source_log=manifest["source_log"],
            scenario_id=manifest["scenario_id"],
            session_id=manifest["session_id"],
            source_manifest_sha256=digest(manifest_path),
            world_source=manifest.get("world_source", "recorded_simulation"),
            source_metadata_sha256=metadata_entry["sha256"],
            registration_mode=mode,
            camera_owners=camera_owners,
            source_frames=source,
            frames=rows,
            map_features=context.map_features,
            map_layers=context.map_layers,
            semantics=dict(
                whitebox="annotation proxies, not optimized GS extents or pixel truth",
                road="planar map projection; no certified terrain or mesh occlusion",
                source="previous real log sample, no invented interpolation",
                rgb="unchanged recorded GS pixels; original photo is context, not counterfactual truth",
            ),
            media_sha256=media,
        )
        (root / "data.json").write_text(
            json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            encoding="utf-8",
        )
        for name in ("index.html", "viewer.js", "projection.js", "semantic.js", "downloads.js"):
            shutil.copyfile(Path(__file__).with_name(name), root / name)
        shutil.copytree(root, output)
    return dict(
        frames=len(rows),
        cameras=len(camera_owners),
        camera_frames=sum(len(f["cameras"]) for f in rows),
        source_frames=len(source),
        media_files=len(media),
        trainable=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maps-root", required=True)
    parser.add_argument("--metadata-python", required=True)
    parser.add_argument("--trusted-assets", action="store_true")
    parser.add_argument("--source-log")
    parser.add_argument("--sensor-root")
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.manifest,
                args.output,
                args.maps_root,
                args.metadata_python,
                trusted=args.trusted_assets,
                source_log=args.source_log,
                sensor_root=args.sensor_root,
            )
        )
    )
