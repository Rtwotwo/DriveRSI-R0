"""Analytical motion and independent geometry checks on canonical cases."""

import numpy as np
import pytest

from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap

pytest.importorskip("driversi.sim._native")
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


@pytest.mark.parametrize("action,jerk,steps", [(7, 0, 10), (10, 4, 5), (1, -15, 3)])
def test_unsaturated_motion_against_closed_form(straight_scene, action, jerk, steps):
    straight_scene.poses[-1, :2, 0] = 100
    with PufferDriveWorld(straight_scene) as world:
        for _ in range(steps):
            world.step({"a": action, "b": 7})
        t, dt = steps * .1, .1
        # Exact sum of trapezoidal integration of quadratic sampled velocity.
        distance = 4 * t + jerk * dt**3 * (steps**3 / 6 + steps / 12)
        assert world.state()[0, 0] == pytest.approx(distance, abs=5e-4)
        assert np.linalg.norm(world.state()[0, 4:6]) == pytest.approx(4 + jerk * t*t/2, abs=1e-4)
        assert world.motion()[0, 0] == pytest.approx(jerk*t, abs=1e-5)


def test_acceleration_saturation_and_lateral_symmetry(straight_scene):
    straight_scene.poses[-1, :2, 0] = 100
    with PufferDriveWorld(straight_scene) as left, PufferDriveWorld(straight_scene) as right:
        for _ in range(10):
            left.step({"a": 11, "b": 7})
            right.step({"a": 9, "b": 7})
        assert left.motion()[0, 0] == pytest.approx(2.5)
        assert left.state()[0, 0] == pytest.approx(right.state()[0, 0], abs=1e-4)
        assert left.state()[0, 1] == pytest.approx(-right.state()[0, 1], abs=1e-4)
        assert 0 < left.motion()[0, 2] <= .55 + 1e-6
        assert abs(left.motion()[0, 1]) <= 4 + 1e-5


@pytest.mark.parametrize("y,expected", [(0, True), (3, False)])
def test_collision_against_independent_rectangles(straight_scene, y, expected):
    straight_scene.poses[:, 1, 0] += 5.0
    straight_scene.poses[:, 1, 1] = y
    straight_scene.velocities[:, 0, 0] = 15
    straight_scene.poses[-1, :2, 0] = 100
    with PufferDriveWorld(straight_scene) as world:
        transition = world.step({"a": 7, "b": 7})
        state = world.state()
        boxes = [rectangle_corners(*state[i, :2], state[i, 3], *straight_scene.dimensions[i, :2])
                 for i in (0, 1)]
        assert rectangles_overlap(*boxes) == expected
        assert bool(transition.events["collision"].any()) == expected


def test_metric_observation_visibility_and_owned_arrays(straight_scene):
    straight_scene.poses[:, 2, 1] = -100
    with PufferDriveWorld(straight_scene) as world:
        metric = world.observe().metric
        assert metric.ego.shape == (2, 10)
        assert metric.visible.tolist() == [[False, True, False], [True, False, False]]
        assert metric.neighbor_ids[0] == ("", "b", "")
        assert not metric.neighbors[:, 2].any()
        metric.ego[:] = 123
        metric.neighbors[:] = 456
        assert not (world.observe().metric.ego == 123).any()


def test_can_start_either_turn_after_straight_motion(straight_scene):
    straight_scene.poses[-1, :2, 0] = 100
    with PufferDriveWorld(straight_scene) as left, PufferDriveWorld(straight_scene) as right:
        for _ in range(5):
            left.step({"a": 7, "b": 7})
            right.step({"a": 7, "b": 7})
        assert left.motion()[0, 1] == 0
        for _ in range(4):
            left.step({"a": 8, "b": 7})
            right.step({"a": 6, "b": 7})
        assert left.state()[0, 3] > .01
        assert right.state()[0, 3] < -.01
        assert left.state()[0, 1] == pytest.approx(-right.state()[0, 1], abs=1e-4)
