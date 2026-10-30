import copy

import numpy as np
import pytest

from driversi.agents.traffic_control.background import ReactiveBackground
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord

pytestmark = pytest.mark.native


def test_all_vehicle_roles_no_replay_and_parked_does_not_launch(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.poses[:, 2] = [20, -8, 0, 1.2]
    source.velocities[:, 2] = 0
    scene = configure_vehicle_fleet(source)
    assert set(scene.metadata["vehicle_fleet"]["roles"]) == set(scene.actor_ids)
    assert scene.metadata["vehicle_fleet"]["roles"]["replay"]["mode"] == "stationary_hold"
    np.testing.assert_array_equal(scene.controlled, source.controlled)
    assert "vehicle_fleet" not in source.metadata
    with PufferDriveWorld(scene) as w:
        initial = w.state()[2].copy()
        record = RolloutRecord.start(w)
        for _ in range(10):
            tr = w.step([7, 7])
            record.append(w, dict(zip(w.actor_ids, [7, 7])), tr)
            np.testing.assert_array_equal(w.state()[2, :7], initial[:7])
            assert tr.traffic_controls[0]["observation_tick"] == w.observe().step - 1
        assert len(record.lifecycle[0]["executed_traffic_controls"]) == 1
        assert "executed_traffic_controls" not in record.lifecycle[-1]


def test_late_spawn_source_gap_and_snapshot_exact(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[:, 2] = False
    source.presence[5:10, 2] = True
    source.poses[:, 2] = 0
    source.poses[5:10, 2] = [0, -8, 0, 0]
    source.velocities[:, 2] = 0
    source.metadata["first_observations"] = {"replay": {"velocity_world": [4.0, 0.0, 0.0]}}
    scene = configure_vehicle_fleet(source)
    assert scene.metadata["vehicle_fleet"]["roles"]["replay"]["mode"] == "uncertain_brake"
    assert not scene.presence[:5, 2].any() and scene.presence[5:, 2].all()
    with PufferDriveWorld(scene) as w:
        for _ in range(5):
            w.step([7, 7])
        assert w.presence()[2]
        assert w.state()[2, 4] == pytest.approx(4.0)
        saved = w.snapshot()
        with w.fork() as b:
            for _ in range(15):
                ta, tb = w.step([7, 7]), b.step([7, 7])
                np.testing.assert_array_equal(w.state(), b.state())
                assert ta.traffic_controls == tb.traffic_controls
                assert "replay" not in w.lifecycle()["exited"]
            final = w.state()
        w.restore(saved)
        for _ in range(15):
            w.step([7, 7])
        np.testing.assert_array_equal(final, w.state())


def test_unknown_motion_not_marked_parked_and_axis_not_rotated(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.poses[:, 2, 3] = np.pi
    source.metadata["first_observations"] = {"replay": {"velocity_world": [4.0, 0.0, 0.0]}}
    scene = configure_vehicle_fleet(source)
    assert scene.metadata["vehicle_fleet"]["roles"]["replay"]["mode"] == "uncertain_brake"
    with PufferDriveWorld(scene) as w:
        old = w.state()[2].copy()
        w.step([7, 7])
        new = w.state()[2]
        assert new[0] > old[0] and 0 < new[4] < old[4]
        assert new[3] == old[3]
    source.presence = np.ones((31, 3), bool)
    source.presence[1:, 2] = False
    source.metadata["first_observations"]["replay"]["velocity_world"] = None
    source.velocities[:, 2] = 0
    scene = configure_vehicle_fleet(source)
    role = scene.metadata["vehicle_fleet"]["roles"]["replay"]
    assert role["mode"] == "uncertain_brake" and role["velocity_evidence"] == "unknown_velocity"


def test_feedback_changes_actions_not_just_ownership(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.poses[:, 2, 1] = 0
    source.poses[:, 2, 0] = -8 + source.times * 4
    scene = configure_vehicle_fleet(source)
    with PufferDriveWorld(scene) as w:
        state = w.state()
        present = w.presence()
        runtime = ReactiveBackground(scene)
        # Lead car is stopped in one frozen world and moving away in another.
        state[0, :2] = [1, 0]
        state[0, 4:6] = [0, 0]
        state[1, :2] = [100, 100]
        blocked, bd = runtime.propose(state, present, present)
        free_state = state.copy()
        free_state[0, 4] = 15
        free, fd = runtime.propose(free_state, present, present)
        assert bd[0]["conflict_id"] == "a"
        assert blocked[0, 5] < free[0, 5]
        # Changing unrelated actor state and all logged future positions cannot change decision.
        unrelated = state.copy()
        unrelated[1, :2] += 100
        np.testing.assert_array_equal(blocked, runtime.propose(unrelated, present, present)[0])
        scene.poses[1:, :2, :2] += 1000
        np.testing.assert_array_equal(blocked, runtime.propose(state, present, present)[0])
        assert fd[0]["conflict_id"] is None


def test_more_than_32_traffic_vehicles_do_not_enter_ppo_mask(straight_scene):
    source = copy.deepcopy(straight_scene)
    n = 40
    source.actor_ids += tuple(f"park-{i}" for i in range(n))
    source.kinds = np.r_[source.kinds, np.ones(n, int)]
    source.dimensions = np.vstack([source.dimensions, np.tile([4, 2, 1.5], (n, 1))])
    poses = np.zeros((31, n, 4))
    poses[:, :, 0] = np.arange(n) * 6
    poses[:, :, 1] = 30
    source.poses = np.concatenate([source.poses, poses], axis=1)
    source.velocities = np.concatenate([source.velocities, np.zeros((31, n, 3))], axis=1)
    source.controlled = np.r_[source.controlled, np.zeros(n, bool)]
    scene = configure_vehicle_fleet(source)
    with PufferDriveWorld(scene) as w:
        tr = w.step([7, 7])
        assert len(tr.traffic_controls) == 41
        assert len(w.actor_ids) == 2 and tr.valid.shape == (2,)
        assert w.presence().sum() == 43


def test_roles_identity_validation_and_nonvehicle_preservation(straight_scene):
    scene = configure_vehicle_fleet(straight_scene)
    assert configure_vehicle_fleet(scene).metadata == scene.metadata
    modified = copy.deepcopy(scene)
    modified.metadata["vehicle_fleet"]["roles"]["replay"]["mode"] = "stationary_hold"
    with pytest.raises(ValueError, match="role disagrees"):
        modified.validate()
    modified = copy.deepcopy(scene)
    modified.metadata.pop("vehicle_fleet")
    with PufferDriveWorld(scene) as a, PufferDriveWorld(modified) as b:
        with pytest.raises(ValueError, match="mismatch"):
            b.restore(a.snapshot())
    source = copy.deepcopy(straight_scene)
    source.kinds[2] = 3
    no_vehicle_background = configure_vehicle_fleet(source)
    assert set(no_vehicle_background.metadata["vehicle_fleet"]["roles"]) == {"a", "b"}
    assert no_vehicle_background.kinds[2] == 3


def test_window_external_identity_is_not_spawned(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[:, 2] = False
    source.poses[:, 2] = 0
    source.velocities[:, 2] = 0
    scene = configure_vehicle_fleet(source)
    assert scene.metadata["vehicle_fleet"]["roles"]["replay"]["mode"] == "outside_window"
    with PufferDriveWorld(scene) as world:
        for _ in range(10):
            world.step([7, 7])
            assert not world.presence()[2]
    scene.presence[0, 2] = True
    with pytest.raises(ValueError, match="Outside-window"):
        scene.validate()


def test_replay_verifies_traffic_execution_and_cli(straight_scene, tmp_path):
    import json

    from driversi.cli import main
    from driversi.contracts.scene import Scene
    from driversi.rendering.bev.replay import export_replay
    from driversi.sim.persistence.recording import write_rollout

    source = tmp_path / "source.npz"
    target = tmp_path / "fleet.npz"
    straight_scene.save(source)
    assert main(["configure-vehicle-fleet", "--scene", str(source), "--output", str(target)]) == 0
    scene = Scene.load(target)
    run_path = tmp_path / "run.npz"
    with PufferDriveWorld(scene) as world:
        snap = world.snapshot()
        record = RolloutRecord.start(world)
        for _ in range(5):
            tr = world.step([7, 7])
            record.append(world, dict(zip(world.actor_ids, [7, 7])), tr)
        write_rollout(
            scene,
            run_path,
            record=record,
            snapshot=snap,
            controlled_ids=world.actor_ids,
            policy="coast",
            controller_configuration={},
        )
    export_replay(scene, run_path, tmp_path / "viewer.html")
    with np.load(run_path, allow_pickle=False) as bundle:
        fields = {k: bundle[k].copy() for k in bundle.files}
    lifecycle = json.loads(str(fields["lifecycle"]))
    lifecycle[0]["executed_traffic_controls"][0]["next_state"][0] += 1
    fields["lifecycle"] = json.dumps(lifecycle)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **fields)
    with pytest.raises(ValueError, match="differs from actual"):
        export_replay(scene, bad, tmp_path / "bad.html")


def test_two_traffic_vehicles_coordinate_crossing_without_order_dependence():
    from driversi.data.participants.background import configure_background
    from driversi.evaluation.search.traffic import crossing_scene

    source, routes = crossing_scene()
    source.controlled = np.array([False, False, True])
    source.metadata["task_goals_xy"] = {"far": routes["far"].points[-1].tolist()}
    source = configure_background(
        source,
        {
            a: {
                "points": [source.poses[0, i, :2].tolist(), routes[a].points[-1].tolist()],
                "desired_speed": 6.0,
                "navigation_source": "explicit_task",
            }
            for i, a in enumerate(("a", "b"))
        },
    )
    scene = configure_vehicle_fleet(source)
    with PufferDriveWorld(scene) as world:
        runtime = ReactiveBackground(scene)
        state, present = world.state(), world.presence()
        rows, _ = runtime.propose(state, present, present)
        runtime.tasks = dict(reversed(list(runtime.tasks.items())))
        reordered, _ = runtime.propose(state, present, present)
        np.testing.assert_array_equal(rows, reordered[np.argsort(reordered[:, 0])])
        contacts = []
        for _ in range(90):
            world.step([7])
            contacts.extend(world.contacts())
        assert not contacts
        assert world.state()[0, 0] > 5 and world.state()[1, 1] > 5


def test_fleet_outlives_learner_and_does_not_enable_new_pedestrians(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.poses[:, 1, :2] = source.poses[:, 0, :2]  # learners collide immediately
    scene = configure_vehicle_fleet(source)
    with PufferDriveWorld(scene) as world:
        world.step([7, 7])
        assert not world.observe().alive.any()
        assert not world.episode_done
        for _ in range(29):
            transition = world.step([7, 7])
        assert world.episode_done
        assert transition.traffic_controls and not transition.valid.any()
    source = copy.deepcopy(straight_scene)
    source.kinds[2] = 2
    source.velocities[:, 2, 0] = 1
    source.poses[:, 2, 0] = source.times
    scene = configure_vehicle_fleet(source)
    assert "reactive_background" not in scene.metadata


def test_actual_inertia_is_checked_when_desired_motion_is_slow(straight_scene):
    scene = configure_vehicle_fleet(straight_scene)
    with PufferDriveWorld(scene) as world:
        state, present = world.state(), world.presence()
    state[2, :6] = [0, 0, 0, 0, 10, 0]
    state[0, :6] = [20, 0, 0, 0, 0, 0]
    state[1, :2] = [100, 100]
    runtime = ReactiveBackground(scene)
    hazard, reason, limit = runtime.vehicle_conflict(2, state, present, np.array([0.1, 0.0]))
    assert hazard is not None and hazard[1] == 0 and limit == 0
    assert reason == "swept_box_conflict"
    old = copy.deepcopy(scene)
    old.metadata["vehicle_fleet"].update(version=1, controller="traffic-fleet-v1")
    assert (
        ReactiveBackground(old).vehicle_conflict(2, state, present, np.array([0.1, 0.0]))[0] is None
    )
    with PufferDriveWorld(scene) as new_world, PufferDriveWorld(old) as old_world:
        with pytest.raises(ValueError, match="mismatch"):
            new_world.restore(old_world.snapshot())
    assert configure_vehicle_fleet(old).metadata["vehicle_fleet"]["version"] == 2


def test_route_support_end_brakes_without_turning_back(straight_scene):
    scene = configure_vehicle_fleet(straight_scene)
    task = scene.metadata["reactive_background"]["assignments"]["replay"]
    with PufferDriveWorld(scene) as world:
        state, present = world.state(), world.presence()
    endpoint = np.asarray(task["points"][-1])
    state[2, :2] = endpoint + [1, 0.4]
    state[2, 3:6] = [0, 5, 0]
    present[:2] = False
    rows, diag = ReactiveBackground(scene).propose(state, present, present)
    assert diag[0]["route_support_exhausted"]
    assert rows[0, 4] == 0 and rows[0, 1] > state[2, 0]
    assert rows[0, 5] == pytest.approx(4.5)
    assert rows[0, 2] == state[2, 1]  # no unphysical position clamp


@pytest.mark.parametrize(
    "case,held",
    [
        ("jitter", True),
        ("departure", False),
        ("missing", False),
        ("inconsistent_position", False),
        ("explicit", False),
    ],
)
def test_source_velocity_evidence_does_not_turn_jitter_into_route(straight_scene, case, held):
    from driversi.data.participants.background import configure_background

    source = copy.deepcopy(straight_scene)
    source.poses[:, 2, 0] = 20 + 0.25 * np.sin(source.times * 3)
    source.poses[:, 2, 1] = -8 + 0.3 * source.times
    source.poses[:, 2, 3] = 0
    source.velocities[:, 2, :2] = [0.6, 0]
    source.metadata["first_observations"] = {"replay": {"velocity_world": [0, 0, 0]}}
    motion = {
        "samples": 31,
        "finite_velocity_samples": 31,
        "max_reported_speed_mps": 0.0,
        "max_position_excursion_m": 1.0,
    }
    if case == "departure":
        motion["max_reported_speed_mps"] = 3.0
    if case == "missing":
        motion["finite_velocity_samples"] = 30
    if case == "inconsistent_position":
        motion["max_position_excursion_m"] = 4.0
    source.metadata["source_motion_evidence"] = {
        "version": 1,
        "scope": "first_contiguous_run_privileged_navigation",
        "actors": {"replay": motion},
    }
    if case == "explicit":
        source = configure_background(
            source,
            {
                "replay": {
                    "points": [[20, -8], [40, -8]],
                    "desired_speed": 4.0,
                    "navigation_source": "explicit_task",
                }
            },
        )
    scene = configure_vehicle_fleet(source)
    role = scene.metadata["vehicle_fleet"]["roles"]["replay"]
    assert role["stationary_evidence"]["accepted"] is held
    if not held:
        return
    assert role["mode"] == "stationary_hold"
    with PufferDriveWorld(scene) as world:
        initial = world.state()[2, :6].copy()
        for _ in range(25):
            world.step([7, 7])
            np.testing.assert_array_equal(world.state()[2, :6], initial)
