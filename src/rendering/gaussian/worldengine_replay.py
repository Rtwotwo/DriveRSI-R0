"""Render a sealed physical branch with released WorldEngine assets (diagnostic).

Missing appearance is reported, never inferred from token counts or low alpha.
This exporter cannot grant visual admission.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def physical_delta(state, source_pose):
    """Both contracts begin x, y, z, yaw; elevation is not planar yaw."""
    state, source_pose = np.asarray(state), np.asarray(source_pose)
    if (
        state.ndim != 1
        or len(state) < 4
        or source_pose.shape != (4,)
        or not np.isfinite(state[:4]).all()
        or not np.isfinite(source_pose).all()
    ):
        raise ValueError("Finite x/y/z/yaw poses required")
    xy = state[:2] - source_pose[:2]
    yaw = float((state[3] - source_pose[3] + np.pi) % (2 * np.pi) - np.pi)
    return [float(xy[0]), float(xy[1]), yaw]



def bind_output_size(pixels, evidence, width):
    """Normalize actual native/portable pixels without reinterpreting calibration."""
    expected = [width, round(width * 1080 / 1920)]
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.shape != (expected[1], expected[0], 3):
        raise ValueError("Actual renderer pixels differ from requested output size")
    for key in ("output_image_size", "output_calibration_size"):
        if key in evidence and list(evidence[key]) != expected:
            raise ValueError("Renderer calibration output size differs from pixels")
    return dict(evidence, output_image_size=expected)

def run(
    scene_path,
    rollout_path,
    bundle,
    raw_log,
    output,
    *,
    future_points=8,
    width=960,
    renderer=None,
    asset_files=None,
    image_writer=None,
    image_contract="history_future_rgb_v1",
):
    from PIL import Image

    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi.rendering.gaussian.worldengine_portable import (
        PortableRenderer,
        supported_log_pose,
    )
    from driversi.sim.backends.pufferdrive import physical_scene_key

    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Expected eight primary or ten native future points")
    if type(width) is not int or width not in (960, 1920):
        raise ValueError("Explicit half or native reconstruction width required")
    end_tick = 20 + future_points * 5
    if image_contract not in {"history_future_rgb_v1", "history_rgb_future_states_v1", "anchor_rgb_history_future_states_v1"}:
        raise ValueError("Unknown image contract")
    render_ticks = ([20] if image_contract == "anchor_rgb_history_future_states_v1" else [5, 10, 15, 20] if image_contract == "history_rgb_future_states_v1" else range(0, end_tick + 1, 5))
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = Scene.load(Path(scene_path))
    with np.load(rollout_path, allow_pickle=False) as z:
        states, present = z["states"], z["presence"]
        ids = tuple(z["actor_ids"].tolist())
        metadata = json.loads(str(z["metadata"]))
    expected_key = physical_scene_key(scene)
    if ids != scene.actor_ids or len(states) <= end_tick or metadata["scene_key"] != expected_key:
        raise ValueError("Physical branch and source scene do not match")
    if scene.metadata.get("pose_reference") != "vehicle_center":
        raise ValueError("Vehicle-center physical poses are required")
    with Path(raw_log).open("rb") as stream:
        frames = pickle.load(stream)
    if renderer is None:
        renderer = PortableRenderer(
            bundle / "background" / (bundle.name + ".ckpt"),
            metadata=bundle / "video_scene_dict.pkl",
            terrain=bundle / "road_height_map",
        )
    if hasattr(renderer, "bind_physics"):
        renderer.bind_physics(scene, states, present)
    pose_query = getattr(renderer, "supported_log_pose", supported_log_pose)
    if renderer.binding.source_log != scene.metadata["source_log"]:
        raise ValueError("Physical branch and reconstruction source differ")
    start = scene.metadata["source_start_frame"]
    from driversi.rendering.gaussian.worldengine_geometry import (
        background_box_support,
        background_physical_box_support,
        box_frustum_support,
        physical_box_corners,
    )

    background = renderer.models["background"]
    background_means = background.get_global_gaussians()["means"].detach().cpu().numpy()
    background_opacity = background.get_opacity().detach().cpu().numpy()
    static_kinds = {"barrier", "traffic_barrier", "traffic_cone", "czone_sign", "generic_object"}
    object_models = {
        key.removeprefix("rigid_object_"): model
        for key, model in renderer.models.items()
        if hasattr(model, "static_in_log")
    }
    from driversi.rendering.gaussian.worldengine_static_bindings import (
        build_static_background_binding,
        static_background_box_at_tick,
    )

    background_bindings, background_binding_errors = {}, {}
    for token in ids:
        if token == "ego" or token in object_models:
            continue
        kind = scene.metadata.get("source_actor_kinds", {}).get(token)
        if kind not in static_kinds | {"vehicle", "bicycle"}:
            continue
        try:
            background_bindings[token] = build_static_background_binding(
                scene, states, present, frames, renderer.binding.resolve, token
            )
        except ValueError as error:
            # Keep rendering diagnostics, but never turn a rejected actor into
            # an eligible background binding or suppress its physical box.
            background_binding_errors[token] = str(error)
    result = dict(
        schema_version=1,
        kind="worldengine_physical_branch_render",
        status="running",
        asset_id=renderer.binding.asset_id,
        renderer_backend=getattr(renderer, "backend", "worldengine_portable"),
        backend_config=getattr(renderer, "backend_config", {}),
        travel_id=getattr(renderer, "travel_id", 0),
        source_log=scene.metadata["source_log"],
        trainable=False,
        quality_admitted=False,
        static_background_bindings=background_bindings,
        static_background_binding_errors=background_binding_errors,
        producer_code={
            str((Path(__file__).parents[1] / "diagnostics/dynamic_binding.py").resolve()): digest(Path(__file__).parents[1] / "diagnostics/dynamic_binding.py"),
            **{str((Path(__file__).parent / name).resolve()): digest(Path(__file__).parent / name)
            for name in ("worldengine_replay.py", "worldengine_geometry.py", "worldengine_static_bindings.py", "worldengine_portable.py")},
        },
        history_ticks=[5, 10, 15, 20],
        future_ticks=list(range(25, end_tick + 1, 5)),
        image_contract=image_contract,
        primary_future_ticks=list(range(25, 61, 5)),
        native_future_points=future_points,
        output_image_size=[width, round(width * 1080 / 1920)],
        inputs={
            str(Path(p).resolve()): digest(p)
            for p in [
                scene_path,
                rollout_path,
                raw_log,
                *(
                    asset_files
                    or [
                        bundle / "video_scene_dict.pkl",
                        bundle / "background" / (bundle.name + ".ckpt"),
                        bundle / "road_height_map" / "road_height_map.npy",
                        bundle / "road_height_map" / "sim2.json",
                    ]
                ),
            ]
        },
        frames=[],
        limitations=[
            "Missing physical object appearances remain unqualified",
            "Frame timestamp rather than individual camera shutter timestamp",
            "Finite terrain lifting where supported; source pitch/roll retained and surface orientation not certified",
            "Image quality and physical alignment require independent admission",
        ],
    )

    def save():
        (output / "report.json").write_text(json.dumps(result, indent=2))

    save()
    try:
        for tick in render_ticks:
            raw = frames[start + tick // 5]
            source = renderer.binding.resolve(raw["token"])
            expected = scene.metadata["source_start_timestamp_us"] + tick * 100000
            residual = int(source["timestamp"]) - expected
            if abs(residual) > 50000:
                raise ValueError("Source capture is too far from physical timeline")
            motions = {}
            missing = []
            static_candidates = []
            absent_assets = []
            for j, token in enumerate(ids):
                if not present[tick, j]:
                    if (
                        token in object_models
                        and pose_query(object_models[token], source["timestamp"]) is not None
                    ):
                        absent_assets.append(token)
                    continue
                delta = physical_delta(states[tick, j], scene.poses[tick, j])
                if token == "ego" or (
                    token in object_models
                    and pose_query(object_models[token], source["timestamp"]) is not None
                ):
                    motions[token] = delta
                else:
                    missing.append(token)
                    kind = scene.metadata.get("source_actor_kinds", {}).get(token)
                    if kind in static_kinds or token in background_bindings:
                        # MTGS foreground extraction excludes these classes;
                        # absence of a rigid node is not absence of pixels.
                        annotation = raw.get("anns", raw)
                        annotation_tokens = list(annotation["track_tokens"])
                        candidate = dict(
                            actor_id=token,
                            kind=kind,
                            binding="static_background_candidate",
                            independent_visual_evidence_required=True,
                            source_annotation_present=token in annotation_tokens,
                        )
                        if token in annotation_tokens:
                            k = annotation_tokens.index(token)
                            box_key = "boxes" if "boxes" in annotation else "gt_boxes"
                            candidate["background_geometry"] = background_box_support(
                                background_means,
                                background_opacity,
                                annotation[box_key][k],
                                source["ego2global"],
                            )
                            candidate["physical_source_xy_difference_m"] = float(
                                np.linalg.norm(delta[:2])
                            )
                            candidate["physical_source_yaw_difference_rad"] = float(delta[2])
                            candidate["physical_source_z_difference_m"] = float(
                                states[tick, j, 2] - scene.poses[tick, j, 2]
                            )
                        else:
                            candidate["reason"] = "physical_retained_current_annotation_unavailable"
                        static_candidates.append(candidate)
            physical_boxes = {
                token: physical_box_corners(
                    states[tick, ids.index(token)],
                    scene.dimensions[ids.index(token)],
                    scene.metadata["origin_xyz"],
                    renderer.origin,
                    source,
                ).tolist()
                for token in missing
            }
            for token in set(missing) & set(background_bindings):
                corners, anchor_evidence = static_background_box_at_tick(
                    background_bindings[token], states[tick, ids.index(token)], tick, expected
                )
                physical_boxes[token] = corners.tolist()
                next(c for c in static_candidates if c["actor_id"] == token)[
                    "immutable_source_binding"
                ] = anchor_evidence
            for candidate in static_candidates:
                token = candidate["actor_id"]
                index = ids.index(token)
                observed = states[: tick + 1, index, :4][present[: tick + 1, index]]
                candidate["physical_immobile"] = bool(
                    len(observed) and np.max(np.abs(observed - observed[0])) <= 1e-5
                )
                candidate["physical_background_geometry"] = background_physical_box_support(
                    background_means, background_opacity, physical_boxes[token]
                )
            extra = sorted(set(object_models) - set(ids))
            source_only_boxes = {}
            annotation = raw.get("anns", raw)
            box_key = "boxes" if "boxes" in annotation else "gt_boxes"
            for token, box in zip(annotation["track_tokens"], annotation[box_key]):
                if token not in extra:
                    continue
                from itertools import product

                box = np.asarray(box, dtype=float)
                corners = np.asarray(list(product((-1, 1), repeat=3))) * box[3:6] / 2
                c, s = np.cos(box[6]), np.sin(box[6])
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                corners = corners @ rotation.T + box[:3]
                refined = np.asarray(source["ego2global"])
                source_only_boxes[token] = (corners @ refined[:3, :3].T + refined[:3, 3]).tolist()
            row = dict(
                tick=tick,
                source_token=raw["token"],
                capture_residual_us=residual,
                physical_missing_appearance=missing,
                appearance_field_semantics="unresolved_binding_not_proven_missing_pixels",
                static_background_candidates=static_candidates,
                absent_physical_with_asset=absent_assets,
                assets_not_in_physical_population=extra,
                source_only_object_boxes=source_only_boxes,
                motions=motions,
                unresolved_physical_boxes=physical_boxes,
                cameras={},
            )
            for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
                pixels, evidence = renderer.render(
                    source, cam, motions=motions, absent_physical=absent_assets, width=width
                )
                evidence = bind_output_size(pixels, evidence, width)
                evidence["unresolved_object_frusta"] = {
                    token: box_frustum_support(
                        corners, evidence["camera_to_world"], evidence["output_ray_bounds"]
                    )
                    for token, corners in physical_boxes.items()
                }
                name = f"{tick:04d}-{cam}.png"
                if image_writer is None:
                    Image.fromarray(pixels).save(output / name)
                else:
                    image_writer(pixels, output / name)
                row["cameras"][cam] = dict(file=name, **evidence)
            result["frames"].append(row)
            save()
            print("rendered", tick, flush=True)
        result["status"] = "rendered_diagnostic"
        result["missing_appearance_tokens"] = sorted(
            {t for row in result["frames"] for t in row["physical_missing_appearance"]}
        )
        result["terrain_fully_supported"] = all(
            camera["terrain_fully_supported"]
            for row in result["frames"]
            for camera in row["cameras"].values()
        )
        result["static_background_candidate_tokens"] = sorted(
            {
                item["actor_id"]
                for row in result["frames"]
                for item in row["static_background_candidates"]
            }
        )
        result["image_count"] = sum(len(row["cameras"]) for row in result["frames"])
    except Exception as ex:
        result.update(status="failed", reason=type(ex).__name__ + ": " + str(ex))
        raise
    finally:
        save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("scene", "rollout", "bundle", "raw-log", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--future-points", type=int, choices=(8, 10), default=8)
    parser.add_argument("--width", type=int, choices=(960, 1920), default=960)
    parser.add_argument("--image-contract", choices=("history_future_rgb_v1", "history_rgb_future_states_v1", "anchor_rgb_history_future_states_v1"), default="history_future_rgb_v1")
    args = parser.parse_args()
    import torch

    torch.cuda.set_per_process_memory_fraction(0.10, 0)
    run(
        args.scene,
        args.rollout,
        args.bundle,
        args.raw_log,
        args.output,
        future_points=args.future_points,
        width=args.width,
        image_contract=args.image_contract,
    )
