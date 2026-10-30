"""Explicit nuPlan rear-axle-local motion to map-axis vehicle-center motion."""

import numpy as np

from driversi.data.sources.navsim import yaw_from_quaternion

REAR_AXLE_TO_CENTER = 1.461  # nuPlan Pacifica, shared compiler convention


def ego_center_motion(record, previous=None):
    yaw = yaw_from_quaternion(record["ego_quaternion_wxyz"])
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
    position = np.asarray(record["ego_rear_axle_global"][:2], float) + REAR_AXLE_TO_CENTER * forward
    local = np.asarray(record["ego_velocity_db"][:2], float)
    valid = False
    omega = 0.0
    if previous is not None:
        dt = (record["ego_pose_timestamp_us"] - previous["ego_pose_timestamp_us"]) / 1e6
        if not 0 < dt <= 0.15:
            raise ValueError("Ego motion requires increasing, continuous pose timestamps")
        old_yaw = yaw_from_quaternion(previous["ego_quaternion_wxyz"])
        omega = ((yaw - old_yaw + np.pi) % (2 * np.pi) - np.pi) / dt
        valid = True
    velocity = local[0] * forward + (local[1] + REAR_AXLE_TO_CENTER * omega) * lateral
    if not np.isfinite(np.r_[position, yaw, velocity]).all():
        raise ValueError("Nonfinite nuPlan ego center motion")
    return position, yaw, velocity, valid
