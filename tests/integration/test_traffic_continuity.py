import copy

import numpy as np
import pytest

from driversi.agents.traffic_control.following import _actuator_choice
from driversi.contracts.checkpoint import decode_world, encode_world
from driversi.contracts.transition import Snapshot
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.execution.traffic_lifecycle import TrafficLifecycle

pytestmark = pytest.mark.native


def configured(source):
    scene = configure_vehicle_fleet(source)
    scene.metadata["traffic_continuity"] = {
        "version": 1,
        "forward_only": True,
        "domain": [-30.0, -15.0, 50.0, 15.0],
        "admission_horizon_seconds": 2.0,
        "admission_margin_m": 0.5,
        "navigation": {
            "replay": {
                "departure_allowed": True,
                "exit_sides": [2],
                "exit_route_max_distance_m": 2.0,
            }
        },
    }
    return scene


def test_forward_braking_stops_and_holds_without_reverse(straight_scene):
    scene = configured(straight_scene)
    scene.velocities[:, :2, 0] = 0.17
    with PufferDriveWorld(scene) as world:
        positions = []
        for _ in range(25):
            world.step([1, 1])
            assert (world.state()[:2, 4] >= 0).all()
            positions.append(world.state()[:2, :2])
        np.testing.assert_array_equal(positions[-1], positions[4])
        assert (world.state()[:2, 4:6] == 0).all()


def test_forward_controller_can_select_actual_stop_instead_of_residual_crawl():
    action, _ = _actuator_choice(-5, 0, np.array([0.0, 0.0]), 0.17, forward_only=True)
    assert action // 3 == 0


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_every_participant_kind_waits_for_overlap_then_enters_without_deletion(
    straight_scene, kind
):
    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[:5, 2] = False
    source.poses[:, 2, :2] = [2, 0]
    scene = configured(source)
    scene.kinds[2] = kind
    manager = TrafficLifecycle(scene)
    state = np.zeros((3, 13))
    state[:, :4] = scene.poses[0]
    state[0, :2] = [2, 0]
    state[1, :2] = [2, 8]
    manager.state["tick"] = 4
    next_state, rows = manager.propose(4, state, np.empty((0, 8)))
    assert next_state["status"][2] == "pending" and not len(rows)
    assert next_state["pending"][0]["actor"] == "replay"
    manager.restore(next_state)
    state[0, 0] = 20
    next_state, rows = manager.propose(5, state, np.empty((0, 8)))
    assert next_state["status"][2] == "active" and rows.shape == (1, 8)
    assert rows[0, 1] == 2  # original pose, not a later source teleport


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_pending_snapshot_roundtrip_and_fork_are_exact(straight_scene, tmp_path, kind):
    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[:5, 2] = False
    source.poses[:, 2, :2] = [2, 0]
    source.poses[:, :2, 0] = source.times[:, None] * 8
    source.velocities[:, :2, 0] = 8
    scene = configured(source)
    scene.kinds[2] = kind
    scene.metadata["reactive_background"]["assignments"]["replay"]["desired_speed"] = 0
    scene.metadata["reactive_background"]["assignments"]["replay"]["navigation_source"] = (
        "uncertain_motion_braking"
    )
    scene.metadata["vehicle_fleet"]["roles"].pop("replay")
    if kind == 1:
        scene.metadata["vehicle_fleet"]["roles"]["replay"] = {"mode": "uncertain_brake"}
    with PufferDriveWorld(scene) as world:
        for _ in range(5):
            world.step([7, 7])
        assert not world.presence()[2]
        before = world.snapshot()
        path = tmp_path / "checkpoint.npz"
        before.save(path)
        assert Snapshot.load(path) == before
        encoded = encode_world(before, tick=5, configuration={}, actor_ids=world.actor_ids)
        assert (
            decode_world(encoded, initial=before, configuration={}, actor_ids=world.actor_ids)
            == before
        )
        with world.fork() as branch:
            for _ in range(20):
                a, b = world.step([7, 7]), branch.step([7, 7])
                np.testing.assert_array_equal(world.state(), branch.state())
                assert a.traffic_controls == b.traffic_controls
            assert world.snapshot() == branch.snapshot()
        assert world.presence()[2]
        assert not world.contacts()
        final = world.snapshot()
        world.restore(before)
        for _ in range(20):
            world.step([7, 7])
        assert world.snapshot() == final


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_exit_requires_entire_footprint_and_outward_motion(straight_scene, kind):
    scene = configured(straight_scene)
    scene.kinds[2] = kind
    scene.metadata["reactive_background"]["assignments"]["replay"]["points"] = [[0, -8], [100, -8]]
    manager = TrafficLifecycle(scene)
    state = np.zeros((3, 13))
    state[:, :4] = scene.poses[0]
    state[2, :2] = [51, -8]
    state[2, 4] = 4
    manager.state["tick"] = 1
    proposal, _ = manager.propose(1, state, np.empty((0, 8)))
    assert proposal["status"][2] == "active"  # rear still inside
    state[2, 0] = 53
    state[2, 4] = -4
    manager.state["tick"] = 1
    proposal, _ = manager.propose(1, state, np.empty((0, 8)))
    assert proposal["status"][2] == "active"  # returning inward
    state[2, 4] = 4
    manager.state["tick"] = 1
    proposal, _ = manager.propose(1, state, np.empty((0, 8)))
    assert proposal["status"][2] == "departed"
    assert proposal["events"][0]["kind"] == "supported_domain_exit"
    manager.restore(proposal)
    proposal, _ = manager.propose(2, state, np.empty((0, 8)))
    assert proposal["status"][2] == "departed"  # cannot re-enter from source


def test_annotation_gap_is_not_a_physical_departure(straight_scene):
    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[5:, 2] = False
    scene = configured(source)
    with PufferDriveWorld(scene) as world:
        for _ in range(15):
            world.step([7, 7])
        assert world.presence()[2]
        assert not world.lifecycle()["exited"]


def test_initial_overlap_rejected_not_silently_merged(straight_scene):
    scene = configured(straight_scene)
    scene.poses[0, 2] = scene.poses[0, 0]
    with pytest.raises(ValueError, match="initial overlapping"):
        TrafficLifecycle(scene)


def test_parallel_lanes_can_move_without_disabling_real_collision_checks():
    from driversi.agents.traffic_control.traffic import parallel_clearance_margin, swept_contact

    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0])
    b = np.array([0.0, 2.2, 0.0, 0.0, 0.0, 4.0, 2.0])
    assert swept_contact(a, b, np.zeros(2), 3, 0.4) == 0
    margin = parallel_clearance_margin(a, b, 0.4)
    assert swept_contact(a, b, np.zeros(2), 3, margin) is None
    # A new command moving into the other lane is still caught.
    assert swept_contact(a, b, np.array([0.0, 1.0]), 3, margin) is not None
    b[1] = 1.9
    assert swept_contact(a, b, np.zeros(2), 3, parallel_clearance_margin(a, b, 0.4)) == 0


def test_corrupt_pre_spawn_departure_checkpoint_is_rejected_transactionally(straight_scene):
    from dataclasses import replace

    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[:5, 2] = False
    scene = configured(source)
    with PufferDriveWorld(scene) as world:
        original = world.snapshot()
        runtime = copy.deepcopy(original.runtime)
        runtime["status"][2] = "departed"
        with pytest.raises(ValueError, match="admission evidence"):
            world.restore(replace(original, runtime=runtime))
        assert original == world.snapshot()


def test_wrong_boundary_and_off_route_are_not_departures(straight_scene):
    scene = configured(straight_scene)
    scene.metadata["reactive_background"]["assignments"]["replay"]["points"] = [[0, -8], [100, -8]]
    manager = TrafficLifecycle(scene)
    state = np.zeros((3, 13))
    state[:, :4] = scene.poses[0]
    state[2, :2] = [0, 18]
    state[2, 5] = 4
    value, _ = manager.propose(0, state, np.empty((0, 8)))
    assert value["status"][2] == "active"
    state[2, :2] = [53, 8]
    state[2, 4:6] = [4, 0]
    value, _ = manager.propose(0, state, np.empty((0, 8)))
    assert value["status"][2] == "active"


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_native_exit_recording_replay_and_restore_for_every_kind(straight_scene, tmp_path, kind):
    from driversi.rendering.bev.replay import export_replay
    from driversi.sim.persistence.recording import RolloutRecord, write_rollout

    source = copy.deepcopy(straight_scene)
    source.poses[:, 2, 0] = source.times * 2
    source.velocities[:, 2, 0] = 2
    scene = configured(source)
    scene.kinds[2] = kind
    scene.metadata["traffic_continuity"]["domain"][2] = 2
    task = scene.metadata["reactive_background"]["assignments"]["replay"]
    task.update(points=[[0, -8], [100, -8]], desired_speed=2)
    scene.metadata["vehicle_fleet"]["roles"].pop("replay")
    if kind == 1:
        scene.metadata["vehicle_fleet"]["roles"]["replay"] = {"mode": "route_feedback"}
    with PufferDriveWorld(scene) as world:
        initial = world.snapshot()
        record = RolloutRecord.start(world)
        for _ in range(30):
            actions = dict.fromkeys(world.actor_ids, 7)
            transition = world.step(actions)
            record.append(world, actions, transition)
            assert not world.contacts()
        assert not world.presence()[2]
        assert world.lifecycle()["continuity"]["status"][2] == "departed"
        final = world.snapshot()
        with world.fork() as restored:
            assert restored.snapshot() == final
        path = tmp_path / "rollout.npz"
        write_rollout(
            scene,
            path,
            record=record,
            snapshot=initial,
            controlled_ids=world.actor_ids,
            policy="fixture",
            controller_configuration={},
        )
        export_replay(scene, path, tmp_path / "viewer.html")
        # Source identity and contact evidence survive; only a supported exit
        # changes native membership. A falsified early disappearance is rejected.
        bad = copy.deepcopy(record)
        bad.presence[1][2] = False
        with pytest.raises(ValueError, match="presence differs"):
            write_rollout(
                scene,
                tmp_path / "bad.npz",
                record=bad,
                snapshot=initial,
                controlled_ids=world.actor_ids,
                policy="fixture",
                controller_configuration={},
            )


def test_recorded_admission_cannot_teleport_the_retained_spawn(straight_scene, tmp_path):
    from driversi.rendering.bev.replay import export_replay
    from driversi.sim.persistence.recording import RolloutRecord, write_rollout

    source = copy.deepcopy(straight_scene)
    source.presence = np.ones((31, 3), bool)
    source.presence[:5, 2] = False
    scene = configured(source)
    with PufferDriveWorld(scene) as world:
        initial = world.snapshot()
        record = RolloutRecord.start(world)
        for _ in range(10):
            actions = dict.fromkeys(world.actor_ids, 7)
            transition = world.step(actions)
            record.append(world, actions, transition)
        assert record.presence[5][2]
        path = tmp_path / "rollout.npz"
        write_rollout(
            scene,
            path,
            record=record,
            snapshot=initial,
            controlled_ids=world.actor_ids,
            policy="fixture",
            controller_configuration={},
        )
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["states"][5, 2, 0] += 1
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="source entry"):
            export_replay(scene, path, tmp_path / "bad.html")


@pytest.mark.parametrize("destination", [False, True])
def test_forward_task_reaches_centre_endpoint_without_permanent_early_hold(
    straight_scene, destination
):
    from driversi.agents.traffic_control.following import RouteFollower
    from driversi.contracts.driving_task import bind_driving_tasks
    from driversi.contracts.road import Route

    scene = configured(straight_scene)
    scene.velocities[:, :2, 0] = 0.1
    routes = {a: Route([[0, y], [4.5, y]]) for a, y in zip(("a", "b"), (0, 8))}
    scene = bind_driving_tasks(scene, routes, destinations=("a", "b") if destination else ())
    controller = RouteFollower(routes, forward_only=True)
    seen = np.zeros(2, bool)
    with PufferDriveWorld(scene) as world:
        for _ in range(30):
            transition = world.step(controller.act(world.observe()))
            seen |= transition.events["goal" if destination else "route_support_end"]
            assert not world.contacts()
            assert (world.state()[:2, 4] >= 0).all()
        assert seen.all()
        assert not world.observe().alive.any()


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_initial_source_contact_retained_counted_and_restorable(straight_scene, kind):
    source = copy.deepcopy(straight_scene)
    source.poses[:, 2] = source.poses[:, 0]
    source.velocities[:, 2] = 0
    scene = configured(source)
    scene.metadata["traffic_continuity"]["initial_overlap_policy"] = "retain_and_report"
    scene.kinds[2] = kind
    if kind != 1:
        scene.metadata["vehicle_fleet"]["roles"].pop("replay")
    manager = TrafficLifecycle(scene)
    assert manager.state["present"] == [True, True, True]
    event = manager.state["events"][0]
    assert event["kind"] == "source_initial_contact"
    assert event["actors"] == ["a", "replay"]
    assert not event["collision_exempt"] and not event["trainable"]
    manager.restore(copy.deepcopy(manager.state))
    with PufferDriveWorld(scene) as world:
        before = world.snapshot()
        assert world.presence().all()
        assert world.contacts()
        world.step([1, 1])
        after = world.snapshot()
        world.restore(before)
        world.step([1, 1])
        assert world.snapshot() == after
        assert world.presence().all()
