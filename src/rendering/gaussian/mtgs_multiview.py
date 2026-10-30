"""Explicit virtual roof-camera geometry; no claim of real sensor calibration."""

from __future__ import annotations

import re

import numpy as np

OPTICAL_TO_NERFSTUDIO = np.diag([1., -1., -1., 1.])


def rigid_matrix(value, label="camera pose") -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0., 0., 0., 1.]))
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f"Invalid rigid {label}")
    return matrix


def virtual_camera_owners(owners, actor_ids, controlled, node_tokens) -> tuple[str, ...]:
    owners = tuple(owners)
    if (len(set(owners)) != len(owners)
            or any(not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", owner)
                   for owner in owners)):
        raise ValueError("Virtual camera owners require unique safe actor IDs")
    controlled_ids = {actor for actor, active in zip(actor_ids, controlled, strict=True) if active}
    if any(owner == "ego" or owner not in controlled_ids or owner not in node_tokens for owner in owners):
        raise ValueError("Virtual camera owners must be controlled non-ego actors with reconstructed nodes")
    return owners


def roof_front_camera(xyz, wxyz, dimensions) -> tuple[np.ndarray, dict]:
    """Return NF C2W and optical->box-centre rig for a virtual vehicle roof mount.

    Body axes are forward/left/up; reconstruction node origin is 3D box centre.
    Ego's rear-axle-height reference is deliberately never reused here.
    """
    xyz, q, size = (np.asarray(value, dtype=float) for value in (xyz, wxyz, dimensions))
    if (xyz.shape != (3,) or q.shape != (4,) or size.shape != (3,)
            or not np.isfinite(np.concatenate((xyz, q, size))).all()
            or np.any(size <= 0) or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)):
        raise ValueError("Virtual camera requires finite box dimensions and a unit rigid-node quaternion")
    w, x, y, z = q
    body = np.eye(4)
    body[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    body[:3, 3] = xyz
    rig = np.eye(4)
    rig[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    rig[:3, 3] = [.35*size[0], 0, size[2]/2+.2]
    target = rigid_matrix(body @ rig @ OPTICAL_TO_NERFSTUDIO)
    return target[:3], {
        "version": "virtual-roof-front-v1", "sensor_id": "virtual_front",
        "body_reference": "reconstructed_3d_box_center", "body_axes": "forward_left_up",
        "optical_axes": "right_down_forward", "pose_axes": "nerfstudio_right_up_back",
        "camera_to_body_optical": rig.tolist(), "owner_dimensions_lwh": size.tolist(),
        "mount_semantics": "synthetic_roof_mount_not_original_vehicle_calibration",
    }


def camera_input_for_target(source_pose, optimized_source_pose, target_pose) -> np.ndarray:
    """Invert Nerfstudio world-axis SO3xR3 adjustment, not SE(3) composition.

    The official optimizer applies R_out=A@R_in, t_out=t_in+b. Preserve its
    source index/appearance parameters but make its single render-time result
    equal to the independently assigned virtual camera pose.
    """
    source = rigid_matrix(source_pose)
    optimized = rigid_matrix(optimized_source_pose)
    target = rigid_matrix(target_pose)
    rotation = optimized[:3, :3] @ source[:3, :3].T
    translation = optimized[:3, 3]-source[:3, 3]
    result = target.copy()
    result[:3, :3] = rotation.T @ target[:3, :3]
    result[:3, 3] -= translation
    return rigid_matrix(result)[:3]


def novel_view_support(target_pose, source_pose) -> dict:
    target, source = rigid_matrix(target_pose), rigid_matrix(source_pose)
    cosine = (np.trace(target[:3, :3] @ source[:3, :3].T)-1)/2
    return {
        "novel_view_quality": "unverified", "trainable": False,
        "distance_from_seed_camera_m": float(np.linalg.norm(target[:3, 3]-source[:3, 3])),
        "rotation_from_seed_camera_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "distance_scope": "current_source_seed_not_nearest_training_camera",
        "ego_without_reconstruction_asset": True,
        "pedestrian_cyclist_appearance": "unverified",
        "lens_mask_semantics": "sampling_support_only_not_reconstruction_quality",
        "height_support": "source_time_node_altitude_not_terrain_following",
        "counterfactual_road_clearance": "unverified_world_uses_planar_motion",
        "limitations": ["virtual_pose_not_real_sensor_calibration", "novel_view_quality_unverified",
                       "ego_vehicle_has_no_reconstruction_asset", "pedestrian_cyclist_assets_unverified"],
    }


def validate_ego_camera_owner(actor_ids, controlled):
    """Source ego cameras need an actual ego policy owner, not two learner slots.

    Other vehicles may use traffic feedback controllers without learner slots.
    This does not relax per-instance asset binding or novel-view quality checks.
    """
    flags = np.asarray(controlled)
    if (flags.dtype != np.bool_ or flags.shape != (len(actor_ids),)
            or len(set(actor_ids)) != len(actor_ids) or "ego" not in actor_ids
            or not flags[list(actor_ids).index("ego")]):
        raise ValueError("Matched source cameras require a controlled ego")
