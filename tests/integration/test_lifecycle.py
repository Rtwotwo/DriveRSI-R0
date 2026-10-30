import copy

import numpy as np
import pytest

from driversi.contracts.scene import MAX_WORLD_ACTORS
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


def populated(scene, count=3):
    scene = copy.deepcopy(scene)
    if count > 3:
        extra = count - 3
        scene.actor_ids += tuple(f"object-{i}" for i in range(extra))
        scene.poses = np.concatenate((scene.poses, np.zeros((31, extra, 4))), axis=1)
        scene.poses[:, 3:, 0] = 100 + (np.arange(extra) % 80) * 2
        scene.poses[:, 3:, 1] = 30 + (np.arange(extra) // 80) * 2
        scene.velocities = np.concatenate((scene.velocities, np.zeros((31, extra, 3))), axis=1)
        scene.dimensions = np.concatenate((scene.dimensions, np.tile([0.6, 0.6, 1.7], (extra, 1))))
        scene.kinds = np.concatenate((scene.kinds, np.full(extra, 2)))
        scene.controlled = np.concatenate((scene.controlled, np.zeros(extra, bool)))
    scene.presence = np.ones(scene.poses.shape[:2], bool)
    return scene


@pytest.mark.parametrize("count", [40, 256, 576, MAX_WORLD_ACTORS])
@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_late_actor_collision_beyond_policy_capacity(straight_scene, kind, count):
    scene = populated(straight_scene, count)
    scene.kinds[-1] = kind
    scene.poses[:, -1, :] = [0.4, 0, 0, 0]
    scene.velocities[:, -1] = 0
    scene.presence[:1, -1] = False
    scene.presence[2:, -1] = False
    with PufferDriveWorld(scene) as world:
        assert not world.presence()[-1]
        assert scene.actor_ids[-1] not in world.observe().metric.neighbor_ids[0]
        result = world.step({a: 7 for a in world.actor_ids})
        assert result.events["collision"][0]
        assert ["a", scene.actor_ids[-1]] in world.lifecycle()["contacts"]
        assert world.lifecycle()["entered"] == [scene.actor_ids[-1]]
        world.step({a: 7 for a in world.actor_ids})
        assert world.lifecycle()["exited"] == [scene.actor_ids[-1]]
        assert not world.contacts()


def test_disappearance_reentry_and_restore(straight_scene):
    scene = populated(straight_scene)
    scene.presence[1:3, 2] = False
    with PufferDriveWorld(scene) as world:
        world.step({a: 7 for a in world.actor_ids})
        snapshot = world.snapshot()
        with world.fork() as branch:
            for tick in [2, 3, 4]:
                world.step({a: 7 for a in world.actor_ids})
                branch.step({a: 7 for a in branch.actor_ids})
                np.testing.assert_array_equal(world.state(), branch.state())
                assert world.lifecycle() == branch.lifecycle()
                assert world.presence()[2] == (tick >= 3)
            final = world.state().copy()
        world.restore(snapshot)
        for _ in range(3):
            world.step({a: 7 for a in world.actor_ids})
        np.testing.assert_array_equal(final, world.state())
        world.reset()
        assert world.presence().all()


@pytest.mark.parametrize("count", [40, 576, MAX_WORLD_ACTORS])
def test_late_nearest_actor_enters_native_observation(straight_scene, count):
    scene = populated(straight_scene, count)
    scene.poses[:, -1, :] = [15, 0, 0, 0]
    scene.presence[0, -1] = False
    baseline = copy.deepcopy(scene)
    baseline.presence[:, -1] = False
    with PufferDriveWorld(scene) as world, PufferDriveWorld(baseline) as control:
        np.testing.assert_array_equal(world.observe().features, control.observe().features)
        world.step({a: 7 for a in world.actor_ids})
        control.step({a: 7 for a in control.actor_ids})
        assert not np.array_equal(world.observe().features[0], control.observe().features[0])
        assert scene.actor_ids[-1] in world.observe().metric.neighbor_ids[0]


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_entering_passive_actor_changes_vehicle_decision_and_exit_releases_it(straight_scene, kind):
    from driversi.agents.traffic_control.following import RouteFollower
    from driversi.contracts.road import Route

    scene = populated(straight_scene)
    scene.kinds[2] = kind
    scene.dimensions[2] = [.8, .6, 1.7]
    scene.poses[:, 2, :] = [10, 0, 0, 0]
    scene.velocities[:, 2] = 0
    scene.presence[:, 2] = False
    scene.presence[1:3, 2] = True
    baseline = copy.deepcopy(scene)
    baseline.presence[:, 2] = False
    follower = RouteFollower({"a": Route([[0, 0], [100, 0]]),
                              "b": Route([[0, 8], [100, 8]])})
    with PufferDriveWorld(scene) as world, PufferDriveWorld(baseline) as control:
        assert follower.commands(world.observe()) == follower.commands(control.observe())
        for tick in range(1, 4):
            world.step({a: 7 for a in world.actor_ids})
            control.step({a: 7 for a in control.actor_ids})
            command = follower.commands(world.observe())[0]
            negative = follower.commands(control.observe())[0]
            if tick < 3:
                assert command["lead_id"] == "replay"
                assert command["desired_acceleration"] < negative["desired_acceleration"]
                assert command["action"] != negative["action"]
            else:
                assert command == negative


def test_capacity_boundary_and_old_backend_guard(straight_scene, monkeypatch):
    from types import SimpleNamespace
    too_large = populated(straight_scene, MAX_WORLD_ACTORS+1)
    with pytest.raises(ValueError, match="unique actor"):
        too_large.validate()
    old = SimpleNamespace(max_world_actors=256)
    monkeypatch.setattr("driversi.sim.backends.pufferdrive.load_native", lambda:old)
    with pytest.raises(RuntimeError, match="native backend capacity"):
        PufferDriveWorld(populated(straight_scene, 576))


def test_dense_high_index_presence_and_snapshot(straight_scene):
    scene = populated(straight_scene, MAX_WORLD_ACTORS)
    scene.presence[1:3,-1] = False
    scene.poses[:,-1,:] = [15,4,0,0]
    with PufferDriveWorld(scene) as world:
        world.step({a:7 for a in world.actor_ids})
        saved = world.snapshot()
        suffix = []
        for _ in range(3):
            step = world.step({a:7 for a in world.actor_ids})
            suffix.append((world.state(),world.presence(),step.observation.features.copy(),world.contacts()))
        world.restore(saved)
        for states, present, features, contacts in suffix:
            step = world.step({a:7 for a in world.actor_ids})
            np.testing.assert_array_equal(states,world.state())
            np.testing.assert_array_equal(present,world.presence())
            np.testing.assert_array_equal(features,step.observation.features)
            assert contacts == world.contacts()
        assert world.presence()[-1]


def test_dense_reactive_override_batch_is_atomic_and_restorable(straight_scene):
    scene = populated(straight_scene, MAX_WORLD_ACTORS)
    with PufferDriveWorld(scene) as world:
        before = world.snapshot()
        initial = world.state().copy()
        rows = np.column_stack((
            np.arange(2, MAX_WORLD_ACTORS), scene.poses[1, 2:], scene.velocities[1, 2:]
        )).astype(np.float64)
        rows[:, 1] += .125
        actions = np.full(len(world.actor_ids), 7, np.int32)
        invalid = rows.copy()
        invalid[-1, 0] = MAX_WORLD_ACTORS
        with pytest.raises((ValueError, RuntimeError)):
            world._native.step(world._handle, actions, invalid)
        np.testing.assert_array_equal(world.state(), initial)
        world._native.step(world._handle, actions, rows)
        expected = world.state().copy()
        assert abs(expected[-1, 0] - rows[-1, 1]) < 1e-4
        world.restore(before)
        world._native.step(world._handle, actions, rows)
        np.testing.assert_array_equal(world.state(), expected)
