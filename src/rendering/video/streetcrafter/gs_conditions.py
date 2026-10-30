"""Actual native GS and same-camera measured RGB-cloud conditions; diagnostic only."""

import argparse
import itertools
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


def match_sources(frames, start, count=25, tolerance_us=60000):
    """One actual unique capture for each exact physical tick; never interpolate."""
    frames = [f for f in frames if not f.get("skipped")]
    result = [min(frames, key=lambda f: abs(int(f["timestamp"]) - (start + i * 100000)))
              for i in range(count)]
    if len({f["token"] for f in result}) != count:
        raise ValueError("Repeated actual source captures")
    if any(abs(int(f["timestamp"]) - (start + i * 100000)) > tolerance_us
           for i, f in enumerate(result)):
        raise ValueError("Native source capture outside tolerance")
    if any(int(round((int(f["timestamp"]) - start) / 100000)) != i
           for i, f in enumerate(result)):
        raise ValueError("Native source capture rounds to another physical tick")
    return result


def ego_reference(source, reconstruction_origin, scene_origin):
    """Actual raw capture rear axle -> compiler's planar vehicle center."""
    from pyquaternion import Quaternion
    pose = np.asarray(source["ego2global_original"], dtype=float)
    yaw = Quaternion(matrix=pose[:3, :3]).yaw_pitch_roll[0]
    xyz = pose[:3, 3] + reconstruction_origin - scene_origin
    xyz[:2] += 1.461 * np.array([np.cos(yaw), np.sin(yaw)])
    return np.r_[xyz, yaw]


def project_points(points, calibration):
    import cv2
    pose = np.asarray(calibration["camera_to_world"], dtype=float)
    xyz = (np.asarray(points, dtype=float) - pose[:3, 3]) @ pose[:3, :3]
    uv = np.full((len(xyz), 2), np.nan)
    valid = (xyz[:, 2] > .1) & (xyz[:, 2] < 100)
    if valid.any():
        uv[valid] = cv2.projectPoints(xyz[valid], np.zeros(3), np.zeros(3),
                                     np.asarray(calibration["distorted_intrinsic"], dtype=float),
                                     np.asarray(calibration["distortion"], dtype=float))[0].reshape(-1, 2)
    return uv, xyz[:, 2]


def project_cloud(points, colors, calibration):
    """Sparse measured-color projection, with occupancy independent of RGB black."""
    width, height = calibration["output_calibration_size"]
    uv, depth = project_points(points, calibration)
    valid = np.isfinite(uv).all(axis=1) & (np.nan_to_num(np.abs(uv), nan=1e9).max(1) < 1e6)
    uv, depth, colors = uv[valid], depth[valid], colors[valid]
    uv = np.rint(uv).astype(np.int64)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, depth, colors = uv[valid], depth[valid], colors[valid]
    flat = uv[:, 1] * width + uv[:, 0]
    order = np.lexsort((depth, flat))
    flat = flat[order]
    _, first = np.unique(flat, return_index=True)
    rgb = np.zeros((height * width, 3), np.uint8)
    mask = np.zeros(height * width, np.uint8)
    rgb[flat[first]] = np.rint(colors[order[first]] * 255).clip(0, 255).astype(np.uint8)
    mask[flat[first]] = 255
    return rgb.reshape(height, width, 3), mask.reshape(height, width)


def transport(points, pivot, rotation, translation):
    return (np.asarray(points) - pivot) @ rotation.T + pivot + translation


def json_projection(points, calibration):
    uv, z = project_points(points, calibration)
    return {"pixels_960x540": [p.tolist() if np.isfinite(p).all() else None for p in uv],
            "camera_depth_m": z.tolist()}


def run(args):
    import yaml
    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi.data.scenes.pufferdrive import native_observed_pose
    from driversi.rendering.video.bundle import validate_bundle
    from driversi.rsi.plugins.synthesis import check_files
    from driversi.sim.backends.pufferdrive import physical_scene_key

    source_conditions = Path(args.conditions).resolve(strict=True)
    prepared = json.loads(source_conditions.read_text())
    check_files(prepared["files"])
    bundle_path = Path(prepared["bundle"])
    bundle = validate_bundle(bundle_path)
    geometry_path = Path(prepared["geometry"])
    geometry = json.loads(geometry_path.read_text())
    check_files(geometry["files"])
    if not (prepared["branch_identity"] == geometry["branch_identity"] == bundle["branch_identity"]):
        raise ValueError("Different physical branch")
    scene_path, rollout_path = Path(args.scene).resolve(), Path(args.rollout).resolve()
    scene = Scene.load(scene_path)
    with np.load(rollout_path, allow_pickle=False) as data:
        states, presence = data["states"], data["presence"]
        if tuple(data["actor_ids"].tolist()) != scene.actor_ids:
            raise ValueError("Rollout actors differ")
        if json.loads(str(data["metadata"]))["scene_key"] != physical_scene_key(scene):
            raise ValueError("Rollout scene differs")
    # The branch reports' sealed hashes, not filenames, bind this exact execution.
    branch = json.loads(Path(bundle["branch_report"]).read_text())
    def contains_hash(value, expected):
        if isinstance(value, dict):
            return any(contains_hash(v, expected) for v in value.values())
        if isinstance(value, list):
            return any(contains_hash(v, expected) for v in value)
        return value == expected
    for path in (scene_path, rollout_path):
        if not contains_hash(branch, digest(path)):
            raise ValueError("Execution file not bound by selected branch: " + str(path))
    asset = Path(args.asset_root).resolve()
    road = yaml.load((asset / "road_block.yml").read_text(), Loader=yaml.BaseLoader)
    data_root = Path(road["data_root"]) / road["road_block_name"]
    metadata_path = data_root / "video_scene_dict.pkl"
    with metadata_path.open("rb") as stream:
        video_id = road["road_block_name"] + "-2"
        video = pickle.load(stream)[video_id]
    if video["log_name"] != scene.metadata["source_log"]:
        raise ValueError("Different source traversal log")
    start = scene.metadata["source_start_timestamp_us"]
    sources = match_sources(video["frame_infos"], start)
    references = [ego_reference(f, np.asarray(video["global2world_translation"]),
                                np.asarray(scene.metadata["origin_xyz"])) for f in sources]
    if not np.allclose(references[0], scene.poses[0, 0], atol=1e-8, rtol=0):
        raise ValueError("Dense ego reference fails initial compiler-coordinate identity")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preflight = {"branch_identity": bundle["branch_identity"], "actual_frames": 25,
                 "actual_source_tokens": [f["token"] for f in sources],
                 "source_offset_us": [int(f["timestamp"]) - (start+i*100000) for i,f in enumerate(sources)],
                 "ego_reference_initial_max_abs_error": float(np.max(np.abs(references[0]-scene.poses[0,0]))),
                 "diagnostic_only": True}
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps(preflight), flush=True)
        return preflight

    import torch
    import open3d as o3d
    from PIL import Image, ImageDraw
    from driversi.rendering.gaussian.mtgs import _array
    from driversi.rendering.gaussian.mtgs_training_window import NativeTrainingRenderer
    from driversi.rendering.gaussian.mtgs_registration import motion_in_source
    from driversi.rendering.gaussian.mtgs_runtime import project_source_lens
    from driversi.rendering.gaussian.mtgs_bindings import observed_3d_box_corners
    from driversi.rendering.gaussian.worldengine_replay import physical_delta
    torch.cuda.set_per_process_memory_fraction(.08, 0)
    renderer = NativeTrainingRenderer(asset / "config.yml", asset / "step-000030000.ckpt",
                                      asset / "road_block.yml", 2)
    # NativeTrainingRenderer activates its owned pinned package before imports.
    from mtgs.scene_model.gaussian_model.utils import quat_to_rotmat
    renderer.bind_physics(scene, states, presence)
    files = {str(p): digest(p) for p in (source_conditions, bundle_path, geometry_path, scene_path,
                                        rollout_path, Path(__file__).resolve())}
    files.update(renderer.asset_files)
    bg_path = data_root / "rgb_point_cloud" / (video_id + ".pcd")
    background = o3d.io.read_point_cloud(str(bg_path)).voxel_down_sample(.08)
    if not background.has_colors():
        raise ValueError("Measured colored background missing")
    bg_xyz, bg_rgb = np.asarray(background.points), np.asarray(background.colors)
    instances = {}
    for name, model in renderer.models.items():
        if not hasattr(model, "static_in_log"):
            continue
        token = name.removeprefix("rigid_object_")
        path = data_root / "instance_point_cloud" / video_id / (token + ".pcd")
        if not path.is_file():
            raise ValueError("Native actor lacks measured color cloud " + token)
        cloud = o3d.io.read_point_cloud(str(path)).voxel_down_sample(.04)
        if not cloud.has_colors():
            raise ValueError("Instance lacks measured color " + token)
        instances[token] = np.asarray(cloud.points), np.asarray(cloud.colors)
        files[str(path)] = digest(path)

    rows = []
    with torch.no_grad():
        for tick, source in enumerate(sources):
            native_reference = native_observed_pose(scene, references[tick])
            # Actual dense capture is the reference even at ticks whose compiled
            # scene observation is a small timestamp-resampled approximation.
            motions = {"ego": physical_delta(states[tick, 0], native_reference)}
            absent = [token for i,token in enumerate(scene.actor_ids) if i and not presence[tick,i]]
            for i, token in enumerate(scene.actor_ids):
                if i and presence[tick, i]:
                    model = renderer.models.get("rigid_object_" + token)
                    if model is None or renderer.physical_binding(token, model, source) is None:
                        raise ValueError("Present physical actor has no exact observed asset binding " + token)
                    motions[token] = [0.,0.,0.]  # Renderer replaces with its sealed physical binding.
            capture = {}
            original_render = renderer.runtime.render_camera
            def capture_render(*a, **kw):
                result = original_render(*a, **kw)
                capture["surface"] = project_source_lens(result, renderer.runtime.source_lens_projection(a[0]))
                return result
            renderer.runtime.render_camera = capture_render
            try:
                pixels, evidence = renderer.render(source, "CAM_F0", motions=motions, absent_physical=absent)
            finally:
                renderer.runtime.render_camera = original_render
            for token, binding in evidence["object_bindings"].items():
                i = scene.actor_ids.index(token)
                if binding["target_tick"] != tick or not np.array_equal(
                    np.asarray(binding["target_physical_pose"]), states[tick,i,:4]
                ):
                    raise ValueError("Native object binding targets another physical state")
            calibration = {key:evidence[key] for key in ("camera_to_world", "distorted_intrinsic", "distortion",
                                                          "output_calibration_size", "calibration_source")}
            pp, cc, actors = [bg_xyz], [bg_rgb], []
            for token in evidence["active_object_tokens"]:
                model = renderer.models["rigid_object_" + token]
                anchor = renderer.asset_anchor(model, source["timestamp"])
                q, t = renderer.supported_log_pose(model, source["timestamp"])
                center = _array(t)
                original_rotation = _array(quat_to_rotmat(q))
                local_points, color = instances[token]
                source_points = local_points @ original_rotation.T + center
                binding = evidence["object_bindings"].get(token)
                rotation, translation = np.eye(3), np.zeros(3)
                if binding:
                    delta = binding["planar_delta"]
                    terrain = evidence["terrain_evidence"][token]
                    if terrain["status"] != "source_identity":
                        translation, rotation, _ = motion_in_source(anchor[1], delta[:2], delta[2])
                        if terrain["status"] == "supported":
                            translation[2] = terrain["vertical_displacement_m"]
                points = transport(source_points, center, rotation, translation)
                pp.append(points); cc.append(color)
                # The learned asset bounds are independent from the measured PCD.
                learned = _array(model.node.get_means(q, t))
                lo, hi = learned.min(0), learned.max(0)
                asset_box = np.array(list(itertools.product(*zip(lo, hi))))
                asset_box = transport(asset_box, center, rotation, translation)
                row = {"actor_id":token, "present_in_authoritative_physics":token in scene.actor_ids,
                       "asset_center_reconstruction":(center+translation).tolist(),
                       "asset_center_projection":json_projection([center+translation], calibration),
                       "reconstructed_asset_bounds":json_projection(asset_box, calibration),
                       "source_token":anchor[1]["token"],
                       "measured_cloud_transform":{"native_source_rotation":original_rotation.tolist(),
                           "native_source_center":center.tolist(), "transport_rotation":rotation.tolist(),
                           "transport_translation":translation.tolist()},
                       "label":"reconstructed asset, not authoritative physical geometry"}
                if binding:
                    i = scene.actor_ids.index(token)
                    corners, annotation = observed_3d_box_corners(anchor[1], token, scene.dimensions[i])
                    # Physical source OBB rotates about its own observed center.
                    corners = transport(corners, corners.mean(0), rotation, translation)
                    row.update(authoritative_physical_pose=states[tick,i,:4].tolist(),
                               authoritative_dimensions=scene.dimensions[i].tolist(),
                               physical_source_bound_box=json_projection(corners, calibration),
                               physical_source_bound_center=json_projection([corners.mean(0)], calibration),
                               physical_source_bound_corners_reconstruction=corners.tolist(),
                               source_annotation=annotation, binding=binding,
                               physical_box_label="authoritative dimensions/pose; source-bound 3D embedding assumption")
                actors.append(row)
            guide, guide_mask = project_cloud(np.concatenate(pp), np.concatenate(cc), calibration)
            surface = capture["surface"]
            if not np.array_equal(surface["rgb"], pixels):
                raise ValueError("Captured mask and rendered RGB use different projections")
            render_mask = np.rint(np.clip(surface["accumulation"],0,1)*255).astype(np.uint8)
            overlay = Image.fromarray(pixels)
            draw = ImageDraw.Draw(overlay)
            draw.rectangle((0,0,959,37),fill=(0,0,0))
            draw.text((5,4),"GREEN: physical source-bound embedding (assumption); MAGENTA: learned asset",fill="white")
            draw.text((5,19),"Projection is diagnostic; matching outlines do not certify pixels or geometry",fill="white")
            for actor in actors:
                for key, color in (("physical_source_bound_box",(0,255,0)),("reconstructed_asset_bounds",(255,0,255))):
                    coords = actor.get(key,{}).get("pixels_960x540",[])
                    for i,j in itertools.combinations(range(len(coords)),2):
                        if (i ^ j) in (1,2,4) and coords[i] is not None and coords[j] is not None:
                            if max(abs(v) for p in (coords[i],coords[j]) for v in p)<10000:
                                draw.line([tuple(coords[i]),tuple(coords[j])],fill=color,width=2)
                coord=actor["asset_center_projection"]["pixels_960x540"][0]
                if coord and 0<=coord[0]<960 and 0<=coord[1]<540:
                    draw.text(tuple(coord),actor["actor_id"][:8],fill="yellow")
            camera = {"calibration":calibration}
            for label, array, resample in (("gs",pixels,Image.Resampling.LANCZOS),
                                          ("guide",guide,Image.Resampling.NEAREST),
                                          ("render_mask",render_mask,Image.Resampling.BILINEAR),
                                          ("guide_mask",guide_mask,Image.Resampling.NEAREST)):
                path=output/f"{tick:04d}-CAM_F0-{label}.png"
                Image.fromarray(array).resize((1024,576),resample).save(path)
                camera[label]=str(path); files[str(path)]=digest(path)
            overlay_path=output/f"{tick:04d}-CAM_F0-overlay.png"
            overlay.save(overlay_path);files[str(overlay_path)]=digest(overlay_path)
            projection_path=output/f"{tick:04d}-CAM_F0-projection.json"
            projection_path.write_text(json.dumps({"tick":tick,"calibration":calibration,"actors":actors,
                "absent_physical_actor_ids":absent,"physical_3d_embedding_certified":False},indent=2))
            files[str(projection_path)]=digest(projection_path)
            camera.update(projection_overlay=str(overlay_path),projection_report=str(projection_path),
                          guide_nonzero_fraction=float((guide_mask>0).mean()),
                          render_mask_semantics="native alpha accumulation remapped with identical source-lens projection",
                          guide_mask_semantics="nearest-depth measured PCD projection occupancy including black points")
            row={"tick":tick,"timestamp_us":start+tick*100000,"source_timestamp_us":int(source["timestamp"]),
                 "source_token":source["token"],"source_offset_us":int(source["timestamp"])-(start+tick*100000),
                 "geometry_tick":tick,"geometry_frame_source_timestamp_us":geometry["frames"][tick]["source_timestamp_us"],
                 "ego_reference_physical_pose":native_reference.tolist(),"ego_planar_delta":motions["ego"],
                 "cameras":{"CAM_F0":camera},"native_evidence":evidence}
            rows.append(row)
            print("GS matched conditions",tick,"guide coverage",camera["guide_nonzero_fraction"],flush=True)
    receipt={"schema_version":1,"kind":"streetcrafter_gs_matched_conditions",
             "branch_identity":bundle["branch_identity"],"bundle":str(bundle_path),"geometry":str(geometry_path),
             "source_conditions":str(source_conditions),"camera_order":["CAM_F0"],"frames":rows,"files":files,
             "initial_reference_images":{"CAM_F0":rows[0]["cameras"]["CAM_F0"]["gs"]},
             "raw_reference_images":{"CAM_F0":prepared["initial_reference_images"]["CAM_F0"]},
             "diagnostic_only":True,"trainable":False,"geometry_registration_certified":False,
             "native_mask_blending_active":False,"native_render_size":[960,540],"saved_image_size":[1024,576],
             "resize_contract":"paired 16:9 resize; calibration records original native 960x540 sampling",
             "point_cloud_recipe":{"background_voxel_m":.08,"actor_voxel_m":.04,"projection":"sparse_nearest_depth"},
             "max_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
             "limitations":["Measured aggregated PCD differs from StreetCrafter training LiDAR and learned GS shape",
                 "Physical source-bound 3D embedding and reconstruction quality are not certified",
                 "Raw photo is unpaired appearance reference, not metric ground truth",
                 "Native diffusion masks do not enforce geometry; output requires independent checks",
                 "25 actual physics ticks use nearest unique 10Hz source captures, no image interpolation"]}
    (output/"report.json").write_text(json.dumps(receipt,indent=2))
    print("RECEIPT",output/"report.json",flush=True)
    return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for key in ("conditions","asset-root","scene","rollout","output"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--preflight-only",action="store_true")
    run(parser.parse_args())

