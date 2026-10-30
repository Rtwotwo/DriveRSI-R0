import os

import numpy as np
import pytest

from driversi.contracts.scene import LANE, ROAD_EDGE, Scene

# Load an explicitly selected host build before importorskip collection. A bad
# override must fail the run, never turn native coverage into skipped tests.
if "DRIVERSI_NATIVE_PATH" in os.environ:
    from driversi.sim.backends.native_loader import load_native

    load_native()


@pytest.fixture
def straight_scene():
    times = np.arange(31) * 0.1
    poses = np.zeros((31, 3, 4))
    poses[:, :, 0] = times[:, None] * 4
    poses[:, :, 1] = [0, 8, -8]
    velocities = np.zeros((31, 3, 3))
    velocities[:, :, 0] = 4
    return Scene(
        metadata={"schema_version": 1, "scenario_id": "fixture-straight",
                  "source_log": "synthetic", "split": "unit", "map_name": "straight",
                  "origin_xyz": [0, 0, 0], "map_complete": True,
                  "heading_qualification": dict.fromkeys(("a", "b", "replay"), "declared_synthetic")},
        actor_ids=("a", "b", "replay"), kinds=np.ones(3, dtype=np.int32),
        dimensions=np.tile([4, 2, 1.5], (3, 1)), times=times, poses=poses,
        velocities=velocities, controlled=np.asarray([True, True, False]),
        road_points=np.asarray([[-40, 0, 0], [60, 0, 0], [-40, -20, 0],
                                [60, -20, 0], [60, 20, 0], [-40, 20, 0], [-40, -20, 0]]),
        road_offsets=np.asarray([0, 2, 7]), road_types=np.asarray([LANE, ROAD_EDGE]),
    )
