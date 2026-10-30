import copy

import numpy as np
import pytest

from driversi.data.participants.background import audit_presence, configure_background, propose_logged_tasks
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


def task(points, speed=4.0):
    return {"points": points, "desired_speed": speed, "navigation_source": "explicit_task"}


def crossing(scene):
    scene = copy.deepcopy(scene)
    scene.poses[:, 0, 0] = 10 + scene.times * 4
    scene.poses[:, 2] = np.column_stack(
        (np.full(31, 18.0), -4 + 1.5 * scene.times, np.zeros(31), np.full(31, np.pi / 2))
    )
    scene.velocities[:, 2] = [0, 1.5, 0]
    scene.kinds[2] = 2
    scene.dimensions[2] = [0.6, 0.6, 1.7]
    return scene


def test_pedestrian_reacts_to_current_vehicle_not_its_future(straight_scene):
    base = crossing(straight_scene)
    reactive = configure_background(base, {"replay": task([[18, -4], [18, 4]], 1.5)})
    far = copy.deepcopy(reactive)
    far.poses[:, 0, 1] = 15
    with PufferDriveWorld(reactive) as world, PufferDriveWorld(far) as control:
        d = world.background_diagnostics()["background_decisions"][0]
        assert d["conflict_id"] == "a"
        assert control.background_diagnostics()["background_decisions"][0]["conflict_id"] is None
        world.step([7, 7])
        control.step([7, 7])
        assert world.state()[2, 5] < control.state()[2, 5]
    future = copy.deepcopy(reactive)
    future.poses[1:, 0, :2] += 123
    with PufferDriveWorld(reactive) as world, PufferDriveWorld(future) as other:
        # Future goal affects native stop rules; background current decision must not see it.
        assert world.background_diagnostics() == other.background_diagnostics()


def test_vehicle_yields_to_crossing_pedestrian(straight_scene):
    base = crossing(straight_scene)
    # Make the first car background, second car the learner, pedestrian replay.
    base.controlled = np.array([False, True, False])
    scene = configure_background(base, {"a": task([[10, 0], [45, 0]])})
    far = copy.deepcopy(scene)
    far.poses[:, 2, 0] = 50
    with PufferDriveWorld(scene) as w, PufferDriveWorld(far) as c:
        assert w.background_diagnostics()["background_decisions"][0]["conflict_id"] == "replay"
        w.step([7])
        c.step([7])
        assert w.state()[0, 4] < c.state()[0, 4]


def test_owned_actor_survives_source_gap_without_teleport_and_restores(straight_scene):
    base = copy.deepcopy(straight_scene)
    base.presence = np.ones((31, 3), bool)
    base.presence[:3, 2] = False
    base.presence[6:10, 2] = False
    base.poses[~base.presence] = 0
    base.velocities[~base.presence] = 0
    scene = configure_background(base, {"replay": task([[1.2, -8], [50, -8]])})
    assert scene.presence[3:, 2].all()
    assert not scene.metadata["reactive_background"]["source_presence"][7][2]
    with PufferDriveWorld(scene) as w:
        for _ in range(5):
            w.step([7, 7])
        saved = w.snapshot()
        with w.fork() as b:
            for _ in range(7):
                w.step([7, 7])
                b.step([7, 7])
                np.testing.assert_array_equal(w.state(), b.state())
                assert w.lifecycle() == b.lifecycle()
            final = w.state()
        w.restore(saved)
        for _ in range(7):
            w.step([7, 7])
        np.testing.assert_array_equal(final, w.state())
        assert abs(w.state()[2, 0] - 4.8) < 0.3
        w.reset()
        assert not w.presence()[2]
    assert audit_presence(scene)["actors"][2]["internal_gaps"] == 1


def test_background_collision_is_native_contact(straight_scene):
    base = copy.deepcopy(straight_scene)
    base.poses[:, 2, :] = [4.1, 0, 0, 0]
    base.velocities[:, 2] = 0
    scene = configure_background(base, {"replay": task([[4.1, 0], [20, 0]], 1.0)})
    with PufferDriveWorld(scene) as w:
        result = w.step([7, 7])
        assert result.events["collision"][0]
        assert ("a", "replay") in w.contacts()


def test_invalid_native_background_batch_does_not_mutate(straight_scene):
    with PufferDriveWorld(straight_scene) as w:
        saved = w.snapshot()
        for rows in (
            [[0, 0, 0, 0, 0, 0, 0, 0]],
            [[2, 0, 0, 0, 0, 0, 0, float("nan")]],
            [[2, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0]],
        ):
            with pytest.raises(ValueError):
                w._native.step(w._handle, np.array([7, 7], np.int32), np.array(rows, float))
            assert w.snapshot() == saved


def test_configuration_identity_and_validation(straight_scene):
    scene = configure_background(straight_scene, {"replay": task([[0, -8], [50, -8]])})
    other = copy.deepcopy(scene)
    other.metadata["reactive_background"]["assignments"]["replay"]["desired_speed"] = 3.0
    with PufferDriveWorld(scene) as w, PufferDriveWorld(other) as b:
        with pytest.raises(ValueError):
            b.restore(w.snapshot())
    with pytest.raises(ValueError):
        configure_background(straight_scene, {"a": task([[0, 0], [50, 0]])})
    with pytest.raises(ValueError):
        configure_background(straight_scene, {"replay": task([[0, -8], [50, -8]], 21.0)})
    assignments, rejected = propose_logged_tasks(straight_scene)
    assert set(assignments) == {"replay"} and not rejected


def test_stopped_reactive_actor_does_not_rotate(straight_scene):
    base = copy.deepcopy(straight_scene)
    base.velocities[:, 2] = 0
    base.poses[:, 2] = [0, -8, 0, 0]
    scene = configure_background(base, {"replay": task([[0, -8], [0.1, -8]], 1.0)})
    with PufferDriveWorld(scene) as w:
        initial = w.state()[2].copy()
        for _ in range(5):
            w.step([7, 7])
        np.testing.assert_array_equal(w.state()[2, :4], initial[:4])


def test_direction_ambiguity_stays_replay_and_six_ticks_are_short(straight_scene):
    base = copy.deepcopy(straight_scene)
    base.poses[:, 2, 3] = np.pi
    assignments, rejected = propose_logged_tasks(base)
    assert "replay" not in assignments
    assert rejected["replay"] == "unverified_forward_direction"
    base.presence = np.ones((31, 3), bool)
    base.presence[:, 2] = False
    base.presence[5:11, 2] = True
    assert audit_presence(base)["actors"][2]["short_run"]


def test_crossing_audit_requires_decision_and_position_change(straight_scene, tmp_path):
    from driversi.evaluation.benchmarks.crossing import evaluate_crossing
    scene = crossing(straight_scene)
    scene.actor_ids = ("car", "ego", "ped")
    scene.metadata["heading_qualification"] = dict.fromkeys(scene.actor_ids, "declared_synthetic")
    scene.save(tmp_path / "scene.npz")
    report = evaluate_crossing(tmp_path / "scene.npz", tmp_path / "audit",
                               vehicle_id="car", pedestrian_id="ped")
    pair = report["paired"]["vehicle_yield"]
    assert pair["paired_controller_response"]
    assert pair["target_conflict_change_ticks"] and pair["max_position_difference_m"] > 0.01
    assert all(row["restore_fork_exact"] for row in report["runs"].values())
    assert not report["trainable"]


def test_single_observation_velocity_keeps_actor_after_source_loss(straight_scene):
    from driversi.data.participants.background import configure_observed_background
    base = copy.deepcopy(straight_scene)
    base.presence = np.ones((31, 3), bool)
    base.presence[:, 2] = False
    base.presence[5:10, 2] = True
    base.poses[:, 2] = 0
    base.poses[5:10, 2] = [0, -8, 0, 0]
    base.velocities[:, 2] = 0
    base.metadata["first_observations"] = {"replay": {
        "velocity_world": [4., 0., 0.], "source_index": 1,
        "source_timestamp_us": 500000, "velocity_source": "gt_velocity_3d_rotated_by_ego_yaw"}}
    scene, _ = configure_observed_background(base)
    assert not scene.presence[:5, 2].any() and scene.presence[5:, 2].all()
    assert not scene.metadata["reactive_background"]["source_presence"][10][2]
    with PufferDriveWorld(scene) as world:
        for _ in range(10):
            world.step([7, 7])
        assert world.presence()[2] and world.state()[2, 0] > 1
        checkpoint = world.snapshot()
        with world.fork() as branch:
            for _ in range(10):
                world.step([7, 7])
                branch.step([7, 7])
                np.testing.assert_array_equal(world.state(), branch.state())
                assert "replay" not in world.lifecycle()["exited"]
        final = world.state()
        world.restore(checkpoint)
        for _ in range(10):
            world.step([7, 7])
        np.testing.assert_array_equal(world.state(), final)
    # An absent velocity is unknown, not evidence of a parked car.
    base.metadata["first_observations"]["replay"]["velocity_world"] = None
    with pytest.raises(ValueError, match="No supported"):
        configure_observed_background(base)


def test_explicit_measured_zero_velocity_holds_short_vehicle(straight_scene):
    from driversi.data.participants.background import configure_observed_background
    scene = copy.deepcopy(straight_scene)
    scene.presence = np.ones((31, 3), bool)
    scene.presence[:, 2] = False
    scene.presence[5:10, 2] = True
    scene.poses[:, 2] = 0
    scene.poses[5:10, 2] = [10, -8, 0, 0]
    scene.velocities[:, 2] = 0
    scene.metadata["first_observations"] = {"replay": {"velocity_world": [0., 0., 0.]}}
    active, _ = configure_observed_background(scene)
    with PufferDriveWorld(active) as world:
        for _ in range(20):
            world.step([7, 7])
        assert world.presence()[2]
        np.testing.assert_allclose(world.state()[2, :2], [10, -8], atol=1e-5)
    assert active.metadata["reactive_background"]["assignments"]["replay"]["navigation_source"] == "observed_stationary_hold"
