"""Auditable treatment of sparse vehicle-box orientation ambiguity."""

import numpy as np


def source_vehicle_headings(
    source_poses: np.ndarray,
    source_times: np.ndarray,
    actor_ids: tuple[str, ...],
    kinds: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return unwrapped headings without interpolating low-speed half turns.

    An annotated rectangular vehicle box has the same footprint at yaw and
    yaw + pi. Near-half-turn jumps at low centre displacement are treated as
    axis ambiguity, not an observed physical turn. This does not recover the
    vehicle's true front direction. Ego localization is not a box annotation.
    Inputs, including all positions and the first heading, remain unchanged.
    """
    poses = np.asarray(source_poses, dtype=np.float64)
    times = np.asarray(source_times, dtype=np.float64)
    actor_kinds = np.asarray(kinds)
    if (poses.ndim != 3 or poses.shape[2] != 4 or poses.shape[0] < 2
            or times.shape != (poses.shape[0],) or not np.isfinite(poses).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Heading audit requires finite poses and increasing source times")
    count = poses.shape[1]
    if (len(actor_ids) != count or len(set(actor_ids)) != count
            or any(not isinstance(actor, str) or not actor for actor in actor_ids)
            or actor_kinds.shape != (count,) or actor_kinds.dtype.kind not in "iu"):
        raise ValueError("Heading audit actor IDs/kinds must match the pose axis")

    raw = poses[..., 3]
    raw_delta = np.arctan2(np.sin(np.diff(raw, axis=0)), np.cos(np.diff(raw, axis=0)))
    intervals = np.diff(times)
    centre_speed = np.linalg.norm(np.diff(poses[..., :2], axis=0), axis=2) / intervals[:, None]
    yaw = np.unwrap(raw, axis=0)
    corrections = []
    threshold = np.deg2rad(150.)
    for actor, (identifier, kind) in enumerate(zip(actor_ids, actor_kinds)):
        if kind != 1 or identifier == "ego":
            continue
        offset = 0.
        for sample in range(1, len(times)):
            delta = float(raw_delta[sample - 1, actor])
            speed = float(centre_speed[sample - 1, actor])
            if abs(delta) >= threshold and speed <= 1.:
                adjustment = -float(np.copysign(np.pi, delta))
                offset += adjustment
                corrections.append({
                    "actor_id": identifier,
                    "source_start_index": sample - 1,
                    "source_end_index": sample,
                    "start_time_seconds": float(times[sample - 1]),
                    "end_time_seconds": float(times[sample]),
                    "raw_start_yaw_radians": float(raw[sample - 1, actor]),
                    "raw_end_yaw_radians": float(raw[sample, actor]),
                    "raw_wrapped_delta_radians": delta,
                    "centre_speed_mps": speed,
                    "pi_adjustment_radians": adjustment,
                    "cumulative_pi_offset_radians": offset,
                    "corrected_delta_radians": delta + adjustment,
                })
            yaw[sample, actor] += offset
    return yaw, {
        "schema_version": 1,
        "policy": "slow_vehicle_box_axis_continuity_v1",
        "scope": "vehicle_annotations_only_excluding_ego_localization",
        "maximum_centre_speed_mps": 1.,
        "minimum_wrapped_jump_degrees": 150.,
        "corrections": corrections,
        "orientation_caveat": "Axis continuity is not verified vehicle front direction; original source annotations remain in Context.",
    }
