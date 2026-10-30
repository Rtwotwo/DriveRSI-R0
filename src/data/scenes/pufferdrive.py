"""Write the pinned PufferDrive binary ABI from a validated canonical scene."""

import struct
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene

TRAJECTORY_LENGTH = 91  # Preserve byte-identical short-scene encoding.
MAX_TRAJECTORY_LENGTH = 6001


def write_binary(scene: Scene, target: Path) -> None:
    scene.validate()
    required = set(scene.metadata.get("unbound_controlled_tasks", []))
    bound = set(scene.metadata.get("driving_tasks", {}).get("actors", {}))
    if required - bound:
        raise ValueError(
            "Stationary source window requires explicit driving tasks before simulation"
        )
    if not scene.metadata["map_complete"] or len(scene.road_types) == 0:
        raise ValueError("PufferDrive requires an attached map")
    extent = np.ptp(scene.road_points[:, :2], axis=0)
    if np.any(extent <= 0) or np.any(extent > 2000):
        raise ValueError("PufferDrive requires a nondegenerate map extent of at most 2 km")
    if len(scene.times) > MAX_TRAJECTORY_LENGTH or not np.allclose(np.diff(scene.times), 0.1):
        raise ValueError("The backend supports at most 6001 samples at 0.1 seconds")
    scenario_id = scene.metadata["scenario_id"].encode("ascii")
    if len(scenario_id) > 16:
        raise ValueError("PufferDrive scenario IDs must be at most 16 ASCII bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:

        def ints(*values: int) -> None:
            stream.write(struct.pack(f"<{len(values)}i", *values))

        def floats(values) -> None:
            stream.write(np.asarray(values, dtype="<f4").tobytes())

        stream.write(struct.pack("16s", scenario_id))
        # No privileged SDC selection: the explicit control mask applies to every actor.
        ints(-1, 0, len(scene.actor_ids), len(scene.road_types))
        trajectory_length = max(TRAJECTORY_LENGTH, len(scene.times))
        padding = trajectory_length - len(scene.times)
        for i in range(len(scene.actor_ids)):
            ints(0, int(scene.kinds[i]) if scene.kinds[i] <= 3 else 1, i, trajectory_length)
            positions = np.pad(scene.poses[:, i, :3], ((0, padding), (0, 0)), mode="edge")
            velocities = np.pad(scene.velocities[:, i], ((0, padding), (0, 0)), mode="edge")
            for values in (positions, velocities):
                for column in range(3):
                    floats(values[:, column])
            floats(np.pad(scene.poses[:, i, 3], (0, padding), mode="edge"))
            ints(
                *(
                    (
                        scene.presence[:, i].astype(int).tolist()
                        if scene.presence is not None
                        else [1] * len(scene.times)
                    )
                    + [0] * padding
                )
            )
            length, width, height = scene.dimensions[i]
            goal = scene.metadata.get("task_goals_xy", {}).get(
                scene.actor_ids[i], scene.poses[-1, i, :2]
            )
            floats([width, length, height, *goal, scene.poses[-1, i, 2]])
            ints(int(not scene.controlled[i]))
        for i, kind in enumerate(scene.road_types):
            points = scene.road_points[scene.road_offsets[i] : scene.road_offsets[i + 1]]
            ints(0, int(kind), i, len(points))
            for column in range(3):
                floats(points[:, column])
            floats(np.zeros(6))
            ints(0)


def native_initial_center(scene: Scene):
    """Reproduce pinned set_means float operations, not an approximate mean.

    Native initialization uses present actors at t=0 followed by every road
    vertex in binary order. Future annotations do not affect this origin.
    """
    mean = np.zeros(2, dtype=np.float32)
    count = 0
    positions = scene.poses[0, scene.presence_at(0), :2]
    for points in (positions, scene.road_points[:, :2]):
        for point in np.asarray(points, dtype=np.float32):
            count += 1
            mean += (point - mean) / np.float32(count)
    if not count or not np.isfinite(mean).all():
        raise ValueError("Finite native initialization center required")
    return mean


def native_observed_pose(scene: Scene, pose):
    """Exact binary float32 -> centered state -> world readback representation."""
    value = np.asarray(pose, dtype=np.float32).copy()
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Finite source pose required")
    center = native_initial_center(scene)
    value[:2] = (value[:2] - center) + center
    return value.astype(float)
