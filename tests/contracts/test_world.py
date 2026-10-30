from dataclasses import replace

import numpy as np
import pytest

from driversi.contracts.transition import Snapshot
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytest.importorskip("driversi.sim._native")
pytestmark = pytest.mark.native


def test_snapshot_restores_dynamics_and_fork_isolation(straight_scene, tmp_path):
    with PufferDriveWorld(straight_scene) as world:
        for _ in range(3):
            world.step({"b": 7, "a": 10})
        saved = world.snapshot()
        saved.save(tmp_path / "snapshot.npz")
        restored = Snapshot.load(tmp_path / "snapshot.npz")
        with world.fork() as child:
            child.step({"a": 1, "b": 7})
            np.testing.assert_array_equal(world.snapshot().native_state, saved.native_state)
            original_next = world.step({"b": 7, "a": 7})
            expected_state = world.state()
            assert not np.array_equal(child.state(), expected_state)
            world.restore(restored)
            next_again = world.step({"a": 7, "b": 7})
            np.testing.assert_array_equal(world.state(), expected_state)
            np.testing.assert_array_equal(next_again.observation.features, original_next.observation.features)
            np.testing.assert_array_equal(next_again.rewards, original_next.rewards)


def test_joint_mapping_order_and_reset_are_reproducible(straight_scene):
    with PufferDriveWorld(straight_scene) as world:
        initial = world.state()
        world.step({"b": 1, "a": 10})
        expected = world.state()
        world.reset()
        np.testing.assert_array_equal(world.state(), initial)
        world.step({"a": 10, "b": 1})
        np.testing.assert_array_equal(world.state(), expected)


def test_time_limit_preserves_final_state_and_requires_reset(straight_scene):
    with PufferDriveWorld(straight_scene) as world:
        for _ in range(world.max_steps):
            transition = world.step(np.asarray([1, 1]))
        assert transition.episode_done
        assert transition.truncated.all()
        assert not transition.terminated.any()
        assert transition.observation.step == world.max_steps
        assert np.any(world.state()[:, 0] != straight_scene.poses[0, :, 0])
        with pytest.raises(RuntimeError, match="reset"):
            world.step(np.asarray([7, 7]))


def test_goal_terminates_only_once_and_masks_later_steps(straight_scene):
    with PufferDriveWorld(straight_scene) as world:
        transitions = []
        while not world.episode_done:
            transitions.append(world.step({"a": 10, "b": 1}))
        assert sum(t.terminated[0] for t in transitions) == 1
        assert sum(t.events["goal"][0] for t in transitions) == 1
        after = next(i for i, t in enumerate(transitions) if t.terminated[0]) + 1
        assert after < len(transitions)
        assert all(not t.valid[0] and t.rewards[0] == 0 for t in transitions[after:])


def test_collision_is_joint_and_snapshot_mismatch_rejected(straight_scene):
    straight_scene.poses[:, 1, 1] = 0
    straight_scene.poses[:, 1, 0] = 8 - straight_scene.times * 4
    straight_scene.poses[:, 1, 3] = np.pi
    straight_scene.velocities[:, 1, 0] = -4
    with PufferDriveWorld(straight_scene) as world:
        while not world.episode_done:
            transition = world.step({"a": 7, "b": 7})
        assert transition.events["collision"].all()
        assert transition.terminated.all()
        np.testing.assert_array_equal(transition.rewards, [-1, -1])
        with pytest.raises(ValueError, match="mismatch"):
            world.restore(replace(world.snapshot(), build_id="other-build"))


def test_invalid_actions_do_not_advance_world(straight_scene):
    with PufferDriveWorld(straight_scene) as world:
        for actions in ({"a": 7}, np.asarray([7.5, 7]), np.asarray([7, 12])):
            with pytest.raises(ValueError):
                world.step(actions)
        assert world.observe().step == 0


def test_road_edge_contact_is_an_actor_termination(straight_scene):
    straight_scene.road_points = np.concatenate((straight_scene.road_points, [[4, -3, 0], [4, 3, 0]]))
    straight_scene.road_offsets = np.append(straight_scene.road_offsets, len(straight_scene.road_points))
    straight_scene.road_types = np.append(straight_scene.road_types, 6)
    with PufferDriveWorld(straight_scene) as world:
        for _ in range(10):
            transition = world.step({"a": 7, "b": 7})
            if transition.terminated[0]:
                break
        assert transition.events["offroad"][0]
        assert transition.rewards[0] == -1
        assert not transition.terminated[1]


def test_degenerate_map_is_rejected_before_native_code(straight_scene):
    straight_scene.road_points[:, 1] = 0
    with pytest.raises(ValueError, match="nondegenerate"):
        PufferDriveWorld(straight_scene)
