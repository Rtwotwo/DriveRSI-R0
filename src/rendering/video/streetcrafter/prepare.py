"""Coloured point-cloud conditions from actual executed states, never admission."""

import argparse
import json
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    import types

    root = Path(__file__).resolve().parents[4]
    for name, folder in (("driversi", "src"),):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(root / folder)]
        sys.modules[name] = module

import numpy as np


def project_cloud(points, colors, camera):
    """Measured pinhole/distortion projection with nearest-depth visibility."""
    import cv2

    transform = np.asarray(camera["camera_to_world"])
    xyz = (points - transform[:3, 3]) @ transform[:3, :3]
    valid = (xyz[:, 2] > 0.1) & (xyz[:, 2] < 100)
    xyz, colors = xyz[valid], colors[valid]
    k = np.diag([1024 / 1920, 576 / 1080, 1]) @ np.asarray(camera["intrinsic"])
    uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, np.asarray(camera["distortion"]))[
        0
    ].reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1) & (np.abs(uv).max(axis=1) < 1000000)
    uv, xyz, colors = uv[finite], xyz[finite], colors[finite]
    uv = np.rint(uv).astype(np.int64)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < 1024) & (uv[:, 1] >= 0) & (uv[:, 1] < 576)
    uv, xyz, colors = uv[valid], xyz[valid], colors[valid]
    flat = uv[:, 1] * 1024 + uv[:, 0]
    order = np.lexsort((xyz[:, 2], flat))
    flat = flat[order]
    _, first = np.unique(flat, return_index=True)
    image = np.zeros((576 * 1024, 3), np.uint8)
    image[flat[first]] = np.rint(colors[order[first]] * 255).clip(0, 255).astype(np.uint8)
    return image.reshape(576, 1024, 3)


def prepare(geometry, asset_root, output):
    import open3d as o3d
    from PIL import Image

    from driversi.agents.backends.native import digest
    from driversi.rendering.video.bundle import validate_bundle
    from driversi.rsi.plugins.synthesis import check_files

    g = json.loads(Path(geometry).read_text())
    check_files(g["files"])
    bundle = validate_bundle(g["bundle"])
    if [f["tick"] for f in g["frames"]] != list(range(71)) or g["source_control"]:
        raise ValueError("Executed 10Hz conditions including matching initial reference required")
    if not np.allclose(g["frames"][0]["delta_xy_yaw"], 0, atol=1e-5, rtol=0):
        raise ValueError("First appearance reference does not match initial physical pose")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    asset_root = Path(asset_root)
    metadata = asset_root / "video_scene_dict.pkl"
    with metadata.open("rb") as stream:
        traversals = pickle.load(stream)
    candidates = [
        (key, travel, f)
        for key, travel in traversals.items()
        for f in travel["frame_infos"]
        if f["token"] == g["frames"][0]["source_token"]
    ]
    if len(candidates) != 1:
        raise ValueError("Ambiguous/missing reconstruction source anchor")
    key, travel, anchor = candidates[0]
    if travel["log_name"] != g["source_log"]:
        raise ValueError("Point clouds from another source log")
    background = asset_root / "rgb_point_cloud" / (key + ".pcd")
    cloud = o3d.io.read_point_cloud(str(background)).voxel_down_sample(0.08)
    if not cloud.has_colors():
        raise ValueError("Coloured background missing")
    raw = np.asarray(g["initial_reference_ego_to_world"])
    align = raw @ np.linalg.inv(np.asarray(anchor["ego2global"]))
    points = np.asarray(cloud.points) @ align[:3, :3].T + align[:3, 3]
    colors = np.asarray(cloud.colors)
    files = {
        str(Path(geometry).resolve()): digest(geometry),
        str(metadata): digest(metadata),
        str(background): digest(background),
    }
    instances = {}
    for token in sorted({a["actor_id"] for f in g["frames"] for a in f["actors"]}):
        p = asset_root / "instance_point_cloud" / key / (token + ".pcd")
        if not p.is_file():
            raise ValueError("Visible physical actor has no colour asset: " + token)
        c = o3d.io.read_point_cloud(str(p)).voxel_down_sample(0.04)
        if not c.has_colors():
            raise ValueError("Uncoloured actor cloud: " + token)
        instances[token] = (np.asarray(c.points), np.asarray(c.colors))
        files[str(p)] = digest(p)
    rows = []
    for frame in g["frames"]:
        pp = [points]
        cc = [colors]
        for actor in frame["actors"]:
            corners = np.asarray(actor["corners_world"])
            dims = np.asarray(actor["dimensions"])
            rotation = np.stack(
                [
                    (corners[4] - corners[0]) / dims[0],
                    (corners[3] - corners[0]) / dims[1],
                    (corners[1] - corners[0]) / dims[2],
                ],
                axis=1,
            )
            xyz, rgb = instances[actor["actor_id"]]
            pp.append(xyz @ rotation.T + corners.mean(axis=0))
            cc.append(rgb)
        xyz, rgb = np.concatenate(pp), np.concatenate(cc)
        row = dict(tick=frame["tick"], timestamp_us=frame["timestamp_us"], cameras={})
        for cam in g["camera_order"]:
            path = output / f"{frame['tick']:04d}-{cam}-guide.png"
            image = project_cloud(xyz, rgb, frame["cameras"][cam])
            Image.fromarray(image).save(path)
            files[str(path)] = digest(path)
            row["cameras"][cam] = dict(
                guide=str(path), nonzero_fraction=float(np.any(image, axis=2).mean())
            )
        rows.append(row)
        print("point-cloud conditions", frame["tick"], flush=True)
    for path in g["initial_reference_images"].values():
        files[path] = digest(path)
    report = dict(
        kind="streetcrafter_native_conditions",
        schema_version=1,
        branch_identity=bundle["branch_identity"],
        bundle=g["bundle"],
        geometry=str(Path(geometry).resolve()),
        camera_order=g["camera_order"],
        frames=rows,
        initial_reference_images=g["initial_reference_images"],
        files=files,
        trainable=False,
        geometry_registration_certified=False,
        appearance_asset_scope="offline_full_source_traversal_including_future_observations",
        causal_history_verified=False,
        point_cloud_recipe=dict(
            background_voxel_m=0.08,
            actor_voxel_m=0.04,
            renderer="sparse_nearest_depth_opencv",
            resolution=[576, 1024],
        ),
        limitations=[
            "Aggregated MTGS clouds replace original StreetCrafter coloured LiDAR input; condition distribution differs and requires independent verification",
            "Alignment uses initial matched source pose; terrain and global registration unverified",
            "Native per-camera generation is not joint six-view generation",
            "Full-traversal appearance assets include future source observations; not history-only",
            "Native25-frame temporal conditioning can affect historical pixels; candidate-future history invariance unverified",
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("geometry", "asset-root", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    prepare(args.geometry, args.asset_root, args.output)
