"""Replay the physical object transport; this is not novel-view pixel truth."""

from pathlib import Path

import numpy as np


def source_registration(source, token, cloud):
    """Use the actual decoded point cloud and observed 3D box, never node count alone."""
    tokens = list(source["track_tokens"])
    if tokens.count(token) != 1:
        raise ValueError("Dynamic node lacks unique source object observation")
    box = np.asarray(source["gt_boxes"][tokens.index(token)], dtype=float)
    pose = np.asarray(source["ego2global"], dtype=float)
    if box.shape != (7,) or pose.shape != (4, 4) or not np.isfinite(box).all() or not np.isfinite(pose).all() or np.any(box[3:6] <= 0):
        raise ValueError("Finite observed 3D box and source transform required")
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError("Actual finite nonempty object Gaussian cloud required")
    center = cloud.mean(0)
    local = (center - pose[:3, 3]) @ pose[:3, :3]
    c, s = np.cos(box[6]), np.sin(box[6])
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    normalized = ((local - box[:3]) @ yaw) / (box[3:6] / 2)
    return dict(actor_id=token, source_token=source["token"],
                observed_box=box.tolist(), gaussian_count=len(cloud),
                decoded_source_center=center.tolist(),
                center_inside_observed_box=bool(np.all(np.abs(normalized) <= 1.0)),
                normalized_center=normalized.tolist())


def validate_history_binding(report, scene, states, presence, canonical):
    from driversi.rendering.diagnostics.worldengine_admission import (
        _validate_background_frame, _validate_physical_frame,
    )
    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support
    from driversi.rendering.gaussian.mtgs_registration import motion_in_source
    from driversi.rendering.gaussian.worldengine_portable import camera_pose, ego_center_pivot
    from driversi.rendering.gaussian.mtgs_registration import move_camera

    expected = {"history_rgb_future_states_v1": [5, 10, 15, 20], "anchor_rgb_history_future_states_v1": [20]}.get(report.get("image_contract"))
    if expected is None or [f["tick"] for f in report["frames"]] != expected:
        raise ValueError("Dynamic training image ticks differ from declared native input contract")
    reasons, evidence = [], []
    terrain_paths = [Path(p) for p in report.get("inputs", {}) if Path(p).name == "road_height_map.npy"]
    terrain = None
    if terrain_paths:
        from driversi.rendering.gaussian.worldengine_geometry import RoadHeightMap
        if len(terrain_paths) != 1:
            raise ValueError("Unique sealed terrain required")
        terrain = RoadHeightMap.load(terrain_paths[0].parent)
    static_kinds = {"barrier", "traffic_barrier", "traffic_cone", "czone_sign", "generic_object"}
    for frame in report["frames"]:
        source = canonical[frame["source_token"]]
        _validate_background_frame(frame, report.get("static_background_bindings", {}), scene, states)
        for camera, item in frame["cameras"].items():
            # Verify the pose actually consumed by rasterization, not only its declared delta.
            original_camera = camera_pose(source, source["cams"][camera])
            delta = frame["motions"]["ego"]
            pivot = ego_center_pivot(source)
            translation, _, _ = motion_in_source(source, delta[:2], delta[2])
            corrected = translation
            ego_terrain = item.get("terrain_evidence", {}).get("ego", {})
            if ego_terrain.get("status") == "supported":
                if terrain is None:
                    raise ValueError("Camera vertical transport lacks sealed terrain")
                corrected, _ = terrain.lift(pivot, translation)
            elif ego_terrain.get("status") not in {"unsupported", "unverified"}:
                raise ValueError("Unknown camera vertical transport")
            expected_camera = move_camera(source, original_camera, pivot, delta[:2], delta[2])
            expected_camera[:, 3] += corrected - translation
            source_camera = np.eye(4); source_camera[:3] = original_camera
            extrinsic = np.linalg.solve(np.asarray(source["ego2global"]), source_camera)
            if (not np.allclose(item["camera_to_world"], expected_camera, atol=1e-8, rtol=0)
                or not np.allclose(item["camera_to_rear_axle"], extrinsic, atol=1e-8, rtol=0)):
                raise ValueError("Rasterization camera does not follow same-branch ego")
            if _validate_physical_frame(frame, item, scene, states, presence):
                reasons.append("visible_asset_without_physical_actor")
            expected = set(frame["motions"]) - {"ego"}
            bindings = item.get("physical_object_transport", {})
            if set(bindings) != expected:
                raise ValueError("Actual renderer object transport evidence incomplete")
            for token, binding in bindings.items():
                if binding["source_token"] != frame["source_token"] or binding["actor_id"] != token:
                    raise ValueError("Object source identity differs")
                delta = frame["motions"][token]
                if not np.allclose(binding["applied_planar_delta"], delta, rtol=0, atol=1e-8):
                    raise ValueError("Applied object delta differs from branch")
                translation, rotation, _ = motion_in_source(source, delta[:2], delta[2])
                # Replay source-height fallback or the actual sealed terrain lift.
                applied = np.asarray(binding["applied_translation"])
                center, pivot = np.asarray(binding["decoded_source_center"]), np.asarray(binding["source_pivot"])
                terrain_row = item.get("terrain_evidence", {}).get(token, {})
                if terrain_row.get("status") == "supported":
                    if terrain is None:
                        raise ValueError("Supported vertical transport lacks sealed height map")
                    translation, _ = terrain.lift(pivot, translation)
                elif terrain_row.get("status") not in {"unsupported", "unverified"}:
                    raise ValueError("Unknown object vertical transport")
                if not np.allclose(applied, translation, atol=1e-6, rtol=0):
                    raise ValueError("Object translation differs from physical and vertical transport")
                observed = source_registration(source, token, center[None])
                if (observed["observed_box"] != binding["observed_box"]
                    or observed["center_inside_observed_box"] != binding["center_inside_observed_box"]):
                    raise ValueError("Recorded object registration differs from source observation")
                target = (center - pivot) @ rotation.T + pivot + applied
                if not np.allclose(target, binding["decoded_target_center"], atol=1e-4, rtol=0):
                    raise ValueError("Actual decoded object cloud did not follow branch transform")
                if binding["gaussian_count"] <= 0 or not binding["center_inside_observed_box"]:
                    reasons.append("source_object_cloud_not_registered_to_observed_box")
            static = {x["actor_id"]: x for x in frame.get("static_background_candidates", [])}
            unresolved = []
            for token in frame["physical_missing_appearance"]:
                corners = frame.get("unresolved_physical_boxes", {}).get(token)
                if corners is not None and box_frustum_support(corners, item["camera_to_world"], item["output_ray_bounds"])["outside_frustum"]:
                    continue
                candidate = static.get(token, {})
                # Only physically immobile scene fixtures may remain in the background.
                # Vehicles and people need independently controlled appearance assets.
                if (scene.metadata.get("source_actor_kinds", {}).get(token) in static_kinds
                    and token in report.get("static_background_bindings", {})
                    and candidate.get("physical_immobile") is True
                    and candidate.get("physical_background_geometry", {}).get("gaussian_count", 0) > 0):
                    continue
                unresolved.append(token)
            if unresolved:
                reasons.append("visible_physical_actor_without_bound_appearance")
            evidence.append(dict(tick=frame["tick"], camera=camera, controlled_tokens=sorted(expected), unresolved_tokens=unresolved))
    return dict(method="actual_decoded_cloud_physical_transport_v1", rejection_reasons=sorted(set(reasons)),
                frames=evidence, independent_pixel_review=False,
                scope="object_identity_pose_transport_not_novel_view_ground_truth")
