"""Display-only geometry; never mutate measured states or certify GS elevation."""
import copy

import numpy as np

PLANAR = "planar_consistent_v1"
MEASURED = "measured_world_v2"
LEGACY = "legacy_mixed_v1"


def display_geometry(frame, camera, profile):
    if profile not in (PLANAR, MEASURED, LEGACY):
        raise ValueError("Unknown whitebox projection profile")
    result, objects = copy.deepcopy(camera), copy.deepcopy(frame["objects"])
    if profile == PLANAR:
        ego = np.asarray(frame["ego_rear_axle_to_local"], float)
        measured_camera = np.eye(4)
        measured_camera[:3] = camera["camera_to_world"]
        if "camera_mount_to_ego" in camera:
            mount = np.asarray(camera["camera_mount_to_ego"], float)
            if (mount.shape != (4, 4) or not np.isfinite(mount).all()
                    or not np.allclose(mount[3], [0, 0, 0, 1])
                    or not np.allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3), atol=2e-5)
                    or not np.isclose(np.linalg.det(mount[:3, :3]), 1, atol=2e-5)):
                raise ValueError("Invalid camera mount")
        elif frame.get("sampling") == "executed_physics_state":
            raise ValueError("Executed branch requires explicit native camera mount")
        else:
            mount = np.linalg.solve(ego, measured_camera)
        yaw = np.arctan2(ego[1, 0], ego[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        planar_ego = np.array([[c,-s,0,ego[0,3]], [s,c,0,ego[1,3]],
                              [0,0,1,0], [0,0,0,1.]])
        result["camera_to_world"] = (planar_ego @ mount)[:3].tolist()
        for obj in objects:
            # Upright display boxes rest on the same plane as the road. Their
            # measured height/grade remains in source.json, not in this view.
            obj["pose_xyz_yaw"][2] = obj["dimensions_lwh"][2] / 2
    result["geometry_profile"] = profile
    return result, objects


def profile_description(profile, state_role="source_log"):
    if state_role not in ("source_log", "executed_branch"):
        raise ValueError("Unknown whitebox state role")
    basis = "executed physics" if state_role == "executed_branch" else "measured/interpolated source"
    if profile == PLANAR:
        return {"profile":profile, "map_ground_z":0,
                "state_role":state_role,
                "position_basis":basis + " map XY and heading; object bottoms projected to ground plane",
                "camera_basis":basis + " XY and yaw, original camera mount; ego Z/pitch/roll removed only for display",
                "measured_3d_modified":False, "metric_3d_ground_truth":False,
                "limitations":["Planar proxy does not represent road grade, curbs, stacked roads or true pixel occlusion.",
                               "Do not use this planar projection to certify GS registration or as photometric ground truth."]}
    if profile in (MEASURED, LEGACY):
        return {"profile":profile, "state_role":state_role, "position_basis":basis,
                "camera_basis":("source calibration transported by executed ego with unverified source height/tilt"
                    if state_role == "executed_branch" else "source camera calibration and measured/interpolated ego SE3"),
                "map_ground_z":0, "measured_3d_modified":False,
                "metric_3d_ground_truth":False,
                "limitations":["Stored camera/actor states and unmeasured flat map are diagnostic only; apparent floating is not road truth."]}
    raise ValueError("Unknown whitebox projection profile")
