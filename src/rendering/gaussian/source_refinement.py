"""Bounded source-view color/opacity repair of an immutable portable GS asset.

This stage produces a child asset, never an admission certificate. Physical
geometry, calibrated cameras and object models remain unchanged. Object regions
are excluded from reconstruction updates, but never from PSNR evaluation.
"""
import argparse
import itertools
import json
import pickle
import shutil
import time
import types
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def object_rectangles(frame, camera_to_world, intrinsic, width, height):
    """Conservative bounding rectangles, clipping OBB edges at the near plane."""
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    edges = [(i, j) for i in range(8) for j in range(i+1, 8)
             if np.sum(signs[i] != signs[j]) == 1]
    ego = np.asarray(frame["ego2global"], dtype=float)
    camera = np.asarray(camera_to_world, dtype=float)
    result = []
    for box in frame["gt_boxes"]:
        box = np.asarray(box, dtype=float)
        if box.shape[0] < 7 or not np.isfinite(box[:7]).all() or min(box[3:6]) <= 0:
            raise ValueError("Finite source annotation boxes required")
        yaw = box[6]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        xyz = (signs * box[3:6] / 2) @ rotation.T + box[:3]
        xyz = xyz @ ego[:3, :3].T + ego[:3, 3]
        xyz = (xyz - camera[:3, 3]) @ camera[:3, :3]
        near = .1
        points = [p for p in xyz if p[2] >= near]
        for i, j in edges:
            a, b = xyz[i], xyz[j]
            if (a[2] < near) != (b[2] < near):
                points.append(a + (b-a) * ((near-a[2]) / (b[2]-a[2])))
        if not points:
            continue
        uvw = np.asarray(points) @ np.asarray(intrinsic).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        low = np.maximum(np.floor(uv.min(0)-8), [0, 0]).astype(int)
        high = np.minimum(np.ceil(uv.max(0)+8), [width, height]).astype(int)
        if np.all(high > low):
            result.append([int(low[0]), int(low[1]), int(high[0]), int(high[1])])
    return result


def run(source_report, output, *, steps=1200, max_wall_seconds=7200, optimize_geometry=False):
    import cv2
    import torch
    from PIL import Image
    from gsplat.rendering import rasterization
    from driversi.agents.backends.sources import activate_packages
    from driversi.vendor import path as dependency_path
    from driversi.rendering.gaussian.reconstruction_support import reconstruction_loss_support
    from driversi.rendering.gaussian.worldengine_portable import (
        PortableRenderer, camera_calibration, camera_pose, supported_log_pose,
    )

    if type(optimize_geometry) is not bool:
        raise ValueError("Explicit geometry strategy required")
    if type(steps) is not int or not 1 <= steps <= (3600 if optimize_geometry else 1200):
        raise ValueError("Refinement exceeds its declared strategy step budget")
    if type(max_wall_seconds) is not int or not 60 <= max_wall_seconds <= 7200:
        raise ValueError("Bounded wall time required")
    out = Path(output).resolve()
    if not out.is_relative_to(project_root() / "exp") or out.exists():
        raise ValueError("A new owned experiment output is required")
    source_path = Path(source_report).resolve(strict=True)
    report = read_mapping(source_path)
    if report.get("source_mode") != "canonical_source_pose_without_motion_v1":
        raise ValueError("Canonical source report required")
    for name, sha in {**report["inputs"], **report["files"]}.items():
        if digest(name) != sha:
            raise ValueError("Source evidence changed: " + name)
    checkpoints = [Path(p) for p in report["inputs"] if Path(p).suffix == ".ckpt"]
    metadata = [Path(p) for p in report["inputs"] if Path(p).name == "video_scene_dict.pkl"]
    if len(checkpoints) != 1 or len(metadata) != 1:
        raise ValueError("One portable asset checkpoint and metadata required")
    checkpoint, metadata_path = checkpoints[0], metadata[0]
    owned = dependency_path("mtgs")
    activate_packages({"mtgs": owned / "mtgs", "nuplan_scripts": owned / "nuplan_scripts"})
    from mtgs.scene_model.mtgs_scene_graph import MTGSSceneModel
    from mtgs.utils.ssim import MaskedSSIM

    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.80)
    out.mkdir(parents=True)
    renderer = PortableRenderer(checkpoint, metadata=metadata_path)
    with metadata_path.open("rb") as stream:
        video = next(iter(pickle.load(stream).values()))
    frames = {f["token"]: f for f in video["frame_infos"]}
    rows = []
    for item in report["frames"]:
        frame = frames[item["source_token"]]
        if frame.get("skipped"):
            raise ValueError("Source frame skipped in reconstruction")
        camera = frame["cams"][item["camera"]]
        k, distortion = camera_calibration(camera, 960)
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, distortion, (960, 540), 1)
        pose = np.eye(4); pose[:3] = camera_pose(frame, camera)
        rows.append(dict(tick=item["tick"], token=item["source_token"],
                         camera=item["camera"], timestamp=int(frame["timestamp"]),
                         image=item["real_source_image"], exposure=item["exposure_factor"],
                         image_sha256=digest(item["real_source_image"]),
                         intrinsic=k.tolist(), distortion=distortion.tolist(),
                         rectified_intrinsic=new_k.tolist(), camera_to_world=pose.tolist(),
                         split="holdout" if item["tick"] in (5, 35, 65) else "train",
                         excluded_rectangles=object_rectangles(frame, pose, new_k, 960, 540)))
    if len(rows) != 60 or len({(r["tick"], r["camera"]) for r in rows}) != 60:
        raise ValueError("Complete15-time four-camera source window required")
    manifest = dict(source_report=str(source_path), source_report_sha256=digest(source_path),
                    checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    metadata=str(metadata_path), metadata_sha256=digest(metadata_path),
                    producer_sha256=digest(__file__), steps=steps,
                    max_wall_seconds=max_wall_seconds, rows=rows,
                    mutable="bounded_background_geometry_color_opacity" if optimize_geometry else "background_sky_color_opacity_only",
                    geometry_bounds=dict(max_mean_delta_m=.2, scale_factor_range=[.5, 2.]) if optimize_geometry else None,
                    objective="native_MTGS_0.8_L1_0.2_MaskedSSIM",
                    held_out_from="this_refinement_only_not_original_reconstruction",
                    evaluation_mask="none_full_source_canvas", trainable=False)
    frozen_json(out / "manifest.json", manifest)
    groups, geometry_parents = [], []
    mutable = [("features_dc", .0025), ("features_rest", .0025/16), ("opacities", .01)]
    for name, model in renderer.models.items():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if name in ("background", "skybox"):
            for key, lr in mutable:
                parameter = model.gauss_params[key]
                parameter.requires_grad_(True)
                groups.append(dict(params=[parameter], lr=lr))
            if optimize_geometry and name == "background":
                for key, lr in (("means", .00016), ("scales", .005), ("quats", .001)):
                    parameter = model.gauss_params[key]
                    parameter.requires_grad_(True)
                    groups.append(dict(params=[parameter], lr=lr))
                    geometry_parents.append((key, parameter, parameter.detach().clone()))
    if not groups:
        raise ValueError("No mutable background or sky parameters")
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    cfg = types.SimpleNamespace(ssim_lambda=.2, use_ssim_on_raw_rgb=False,
        output_depth_during_training=False, use_normal_loss=False, two_d_gaussians=False,
        adapter_lambda=0., oob_lambda=0., sharp_shape_reg_lambda=0.)
    objective = types.SimpleNamespace(config=cfg, device=torch.device("cuda:0"),
        ssim=MaskedSSIM(data_range=1., size_average=True, channel=3).cuda(), training=False)
    for name in ("get_gt_img", "composite_with_background", "combine_visible_mask"):
        setattr(objective, name, types.MethodType(getattr(MTGSSceneModel, name), objective))
    objective.get_loss = types.MethodType(MTGSSceneModel.get_loss_dict, objective)
    cache = {}

    def inputs(row):
        key = (row["tick"], row["camera"])
        if key not in cache:
            image = np.asarray(Image.open(row["image"]).convert("RGB"))
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
            hsv[..., 2] = np.clip(hsv[..., 2]*row["exposure"], 0, 255)
            real = cv2.resize(cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB),
                              (960, 540), interpolation=cv2.INTER_AREA)
            k, dist, nk = map(np.asarray, (row["intrinsic"], row["distortion"], row["rectified_intrinsic"]))
            mx, my = cv2.initUndistortRectifyMap(k, dist, None, nk, (960, 540), cv2.CV_32FC1)
            target = cv2.remap(real, mx, my, cv2.INTER_LINEAR)
            mask = reconstruction_loss_support(mx, my, source_size=(960, 540),
                                               excluded_rectangles=row["excluded_rectangles"])
            if mask.mean() < .1:
                raise ValueError("Too little genuine source support for reconstruction")
            cache[key] = (torch.as_tensor(target.copy(), device="cuda", dtype=torch.float32)/255,
                torch.as_tensor(nk, device="cuda", dtype=torch.float32),
                torch.as_tensor(row["camera_to_world"], device="cuda", dtype=torch.float32),
                torch.as_tensor(mask, device="cuda"))
        return cache[key]

    def render(row):
        target, k, pose, mask = inputs(row)
        chunks = {key: [] for key in ("means", "quats", "scales", "opacities", "colors")}
        for model in renderer.models.values():
            if hasattr(model, "static_in_log"):
                state = supported_log_pose(model, row["timestamp"])
                if state is None:
                    continue
                q, t = state
                gs = dict(means=model.get_means(q, t), quats=model.get_quats(q, t),
                          scales=model.get_scales(), opacities=model.get_opacity())
            else:
                gs = model.get_global_gaussians()
            for key, value in gs.items():
                chunks[key].append(value)
            chunks["colors"].append(model.get_gaussian_rgbs(pose[None], timestamp=row["timestamp"],
                                                          device=torch.device("cuda:0")))
        pixels, _, _ = rasterization(**{k: torch.cat(v, 0) for k, v in chunks.items()},
            viewmats=torch.linalg.inv(pose)[None], Ks=k[None], width=960, height=540,
            packed=False, near_plane=.01, far_plane=1e10, render_mode="RGB",
            rasterize_mode="antialiased", radius_clip=0.)
        prediction = pixels[0].clamp(0, 1)
        return objective.get_loss(dict(rgb=prediction, rgb_appearance=prediction,
            background=prediction.new_zeros(3), camera=None), dict(image=target, mask=mask))

    train = [row for row in rows if row["split"] == "train"]
    start = time.monotonic()
    history = []
    with (out / "updates.jsonl").open("w") as log:
        for step in range(steps):
            if time.monotonic()-start >= max_wall_seconds:
                raise TimeoutError("Refinement budget exhausted")
            row = train[step % len(train)]
            optimizer.zero_grad(set_to_none=True)
            terms = render(row); loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            loss.backward(); optimizer.step()
            with torch.no_grad():
                for key, value, original in geometry_parents:
                    if key == "means":
                        delta = value-original
                        delta *= (.2 / delta.norm(dim=-1, keepdim=True).clamp(min=.2))
                        value.copy_(original+delta)
                    elif key == "scales":
                        value.copy_(torch.maximum(torch.minimum(value, original+np.log(2.)), original+np.log(.5)))
            record = dict(step=step+1, tick=row["tick"], camera=row["camera"],
                          loss=float(loss.detach()), wall_seconds=time.monotonic()-start)
            log.write(json.dumps(record)+"\n")
            if (step+1) % 20 == 0:
                log.flush(); write_json(out / "progress.json", record)
                print(json.dumps(record), flush=True)
            history.append(record)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    changes = []
    for name, model in renderer.models.items():
        if name not in ("background", "skybox"):
            continue
        keys = [k for k, _ in mutable] + (["means", "scales", "quats"] if optimize_geometry and name == "background" else [])
        for key in keys:
            value = model.gauss_params[key].detach().cpu()
            original = raw[name]["state_dict"]["gauss_params."+key]
            changes.append(dict(model=name, key=key, max_abs_delta=float((value-original).abs().max())))
            raw[name]["state_dict"]["gauss_params."+key] = value
    child = out / checkpoint.stem
    (child / "background").mkdir(parents=True)
    child_checkpoint = child / "background" / checkpoint.name
    torch.save(raw, child_checkpoint)
    shutil.copy2(metadata_path, child / metadata_path.name)
    terrain = checkpoint.parent.parent / "road_height_map"
    if terrain.exists():
        shutil.copytree(terrain, child / "road_height_map")
    result = dict(status="refined_requires_independent_rerender", child_asset=str(child),
        checkpoint_sha256=digest(child_checkpoint), parent_checkpoint_sha256=manifest["checkpoint_sha256"],
        manifest_sha256=digest(out / "manifest.json"), steps=steps,
        wall_seconds=time.monotonic()-start, changes=changes,
        geometry_calibration_and_actor_models_unchanged=not optimize_geometry,
        calibration_and_actor_models_unchanged=True, optimize_geometry=optimize_geometry, trainable=False)
    write_json(out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args))), flush=True)


if __name__ == "__main__":
    main()
