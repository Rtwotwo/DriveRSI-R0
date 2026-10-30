"""Inspection contract shared by measured logs and recorded physical branches."""

import numpy as np


def validate_sequence(value):
    if value.get("schema_version") != 1 or value.get("role") not in {
        "source_log",
        "executed_branch",
    }:
        raise ValueError("Unknown whitebox schema or state role")
    if value.get("trainable") is not False:
        raise ValueError("Whitebox export does not establish training admission")
    origin = np.asarray(value["origin_xyz"], dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid coordinate origin")
    frames = value["frames"]
    if len(frames) < 2:
        raise ValueError("At least two recorded frames required")
    last_time, last_stamp = -1.0, -1
    anchor = value["provenance"]["anchor_timestamp_us"]
    for frame in frames:
        time, stamp = frame["time_seconds"], frame["timestamp_us"]
        if (
            not np.isfinite(time)
            or time <= last_time
            or stamp <= last_stamp
            or abs((stamp - anchor) / 1e6 - time) > 1e-6
        ):
            raise ValueError("Nonmonotonic or inconsistent recording clock")
        last_time, last_stamp = time, stamp
        identities = set()
        for obj in frame["objects"]:
            actor = obj["actor_id"]
            pose, size = (
                np.asarray(obj["pose_xyz_yaw"], float),
                np.asarray(obj["dimensions_lwh"], float),
            )
            if (
                not isinstance(actor, str)
                or not actor
                or actor in identities
                or pose.shape != (4,)
                or size.shape != (3,)
                or not np.isfinite(pose).all()
                or not np.isfinite(size).all()
                or np.any(size <= 0)
            ):
                raise ValueError("Duplicate identity or invalid object geometry")
            identities.add(actor)
    if abs(frames[0]["time_seconds"]) > 1e-9:
        raise ValueError("Recording must begin at the shared anchor")


def validate_pair(source, branch):
    validate_sequence(source)
    validate_sequence(branch)
    if source["role"] != "source_log" or branch["role"] != "executed_branch":
        raise ValueError("Role ordering mismatch")
    for key in ("source_log", "source_split", "scenario_id", "origin_xyz", "coordinate_frame"):
        if source[key] != branch[key]:
            raise ValueError(f"Streams do not share {key}")
    if source["provenance"]["anchor_timestamp_us"] != branch["provenance"]["anchor_timestamp_us"]:
        raise ValueError("Streams do not share time origin")
