import copy

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")
from driversi.training.sampling.environment import TeacherEnvironment


def test_teacher_joint_transition_and_restore(straight_scene):
    env = TeacherEnvironment(straight_scene)
    try:
        snapshot = env.world.snapshot()
        obs, rewards, transition = env.step({"a": 7, "b": 1})
        assert obs.shape == (2, 1125)
        assert transition.valid.all()
        assert rewards[0] > 0
        env.world.restore(snapshot)
        again = env.step({"a": 7, "b": 1})
        np.testing.assert_array_equal(obs, again[0])
        np.testing.assert_array_equal(rewards, again[1])
    finally:
        env.close()


def test_future_reference_is_not_a_teacher_observation(straight_scene):
    other = copy.deepcopy(straight_scene)
    # Preserve initial state, assigned endpoint and uncontrolled replay.
    other.poses[1:-1, :2, :2] += 100
    a, b = TeacherEnvironment(straight_scene), TeacherEnvironment(other)
    try:
        np.testing.assert_array_equal(a.observe(), b.observe())
        for _ in range(3):
            np.testing.assert_array_equal(a.step(np.array([7, 7]))[0],
                                          b.step(np.array([7, 7]))[0])
    finally:
        a.close()
        b.close()
