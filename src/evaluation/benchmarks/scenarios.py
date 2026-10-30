"""Declared canonical scenarios, not replacements for held-out real traffic."""

import numpy as np

from driversi.contracts.scene import LANE, ROAD_EDGE, Scene


def following_scene(speed: float, gap: float, *, replay: bool = False, duration: float = 9) -> Scene:
    if not np.isfinite([speed, gap, duration]).all() or speed <= 0 or gap <= 0 or not 1 < duration <= 9:
        raise ValueError("Following case requires positive speed/gap and duration in (1,9]")
    if not np.isclose(duration*10, round(duration*10)):
        raise ValueError("Duration must align with 0.1 s")
    times = np.arange(round(duration*10)+1)*.1
    poses = np.zeros((len(times), 3, 4))
    poses[:, :, 0] = times[:, None]*speed + [0, gap+4, 0]
    poses[:, :, 1] = [0, 0, 80]
    velocities = np.zeros((len(times), 3, 3))
    velocities[:, :, 0] = speed
    controlled = np.asarray([not replay, True, True])
    actors = ("follower", "lead", "far")
    return Scene(
        {"schema_version": 1, "scenario_id": "following-v1", "source_log": "synthetic-following-v1",
         "split": "canonical", "map_name": "straight", "origin_xyz": [0, 0, 0], "map_complete": True,
         "heading_qualification": dict.fromkeys(actors, "declared_synthetic"),
         "task_goals_xy": {actor: [220, 80 if actor == "far" else 0]
                           for actor, active in zip(actors, controlled) if active}},
        actors, np.ones(3, np.int32), np.tile([4, 2, 1.5], (3, 1)), times, poses, velocities, controlled,
        np.asarray([[-30, 0, 0], [250, 0, 0], [-30, 80, 0], [250, 80, 0],
                    [-30, -15, 0], [250, -15, 0], [250, 100, 0], [-30, 100, 0], [-30, -15, 0]]),
        np.asarray([0, 2, 4, 9]), np.asarray([LANE, LANE, ROAD_EDGE]))
