"""Causal NAVSIM past resampling for a native DriveRL decision at rollout t=0.

Only observed records at or before t=0 are allowed. Interpolation is declared
adapter input, not additional measurements or executed interaction history.
"""

import copy
import math

import numpy as np


def interpolate_objects(left, right, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("No history extrapolation")
    if fraction == 0:
        return copy.deepcopy(left)
    if fraction == 1:
        return copy.deepcopy(right)
    a = {x["track_token"]: x for x in left}
    b = {x["track_token"]: x for x in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("Duplicate history identity")
    result = []
    for token in sorted(a.keys() & b.keys()):
        x, y = a[token], b[token]
        if x["kind"] != y["kind"]:
            raise ValueError("History object class changes")
        z = copy.deepcopy(x)
        for field in ("pose", "size", "velocity"):
            z[field] = (
                (1 - fraction) * np.asarray(x[field], float)
                + fraction * np.asarray(y[field], float)
            ).tolist()
        delta = (y["pose"][2] - x["pose"][2] + math.pi) % (2 * math.pi) - math.pi
        z["pose"][2] = (x["pose"][2] + fraction * delta + math.pi) % (2 * math.pi) - math.pi
        # Unknown angular rates remain unknown; never manufacture a measurement.
        va, vb = x.get("angular_velocity"), y.get("angular_velocity")
        if (
            type(va) in (int, float)
            and type(vb) in (int, float)
            and math.isfinite(va)
            and math.isfinite(vb)
        ):
            z["angular_velocity"] = (1 - fraction) * va + fraction * vb
        else:
            z["angular_velocity"] = None
        result.append(z)
    return result


def causal_grid(times, anchor_us):
    times = np.asarray(times, dtype=np.int64)
    if len(times) < 3 or np.any(np.diff(times) <= 0) or times[-1] != anchor_us:
        raise ValueError("Ordered real source past ending exactly at initialization required")
    targets = np.arange(anchor_us - 1000000, anchor_us, 100000, dtype=np.int64)
    if targets[0] < times[0] or np.any(np.diff(times) > 550000):
        raise ValueError("Insufficient continuous source history")
    pairs = []
    for stamp in targets:
        right = int(np.searchsorted(times, stamp, side="left"))
        if times[right] == stamp:
            left = right
            fraction = 0.0
        else:
            left = right - 1
            fraction = float((stamp - times[left]) / (times[right] - times[left]))
        pairs.append((int(stamp), left, right, fraction))
    return pairs


def bootstrap(native):
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

    from driversi_agents.driverl.closed_loop import pack_objects, unpack_objects

    indices = (-3, -2, -1, 0)
    states = [native.get_ego_state_at_iteration(i) for i in indices]
    observations = [pack_objects(native.get_tracked_objects_at_iteration(i)) for i in indices]
    times = [int(e.time_us) for e in states]
    anchor = times[-1]
    grid = causal_grid(times, anchor)
    trajectory = InterpolatedTrajectory(states)
    egos = []
    objects = []
    support = []
    for stamp, left, right, fraction in grid:
        egos.append(trajectory.get_state_at_time(TimePoint(stamp)))
        rows = interpolate_objects(observations[left], observations[right], fraction)
        objects.append(unpack_objects(rows, stamp))
        support.append(
            dict(
                timestamp_us=stamp,
                source_left_us=times[left],
                source_right_us=times[right],
                fraction=fraction,
                extrapolated=False,
            )
        )
    return (
        egos,
        objects,
        dict(
            mode="causal_resampled_10hz_v1",
            source_timestamps_us=times,
            anchor_timestamp_us=anchor,
            source_frequency_hz=2,
            output_frequency_hz=10,
            measured_10hz=False,
            future_source_read=False,
            object_presence="identity_intersection_between_observed_frames",
            support=support,
        ),
    )
