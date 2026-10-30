"""Terminal vehicle geometry remains fixed under later joint steering actions."""

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


def test_goal_stopped_vehicle_does_not_rotate_under_later_steering(straight_scene):
    straight_scene.metadata["task_goals_xy"] = {"a": [4., 0.], "b": [50., 8.]}
    with PufferDriveWorld(straight_scene) as world:
        for _ in range(10):
            world.step({"a": 7, "b": 7})
            if world.state()[0, 9]:
                break
        stopped = world.state()[0].copy()
        assert stopped[9] == 1
        assert stopped[12] == 1
        np.testing.assert_array_equal(stopped[4:6], [0., 0.])
        for _ in range(10):
            world.step({"a": 11, "b": 7})
            np.testing.assert_array_equal(world.state()[0, :7], stopped[:7])
