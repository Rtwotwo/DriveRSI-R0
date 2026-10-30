"""Visual actors, clocks, interventions and full episode recovery contracts."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.agents.traffic_control.following import quantize_jerk
from driversi.agents.interfaces.visual import PixelBrakePolicy
from driversi.contracts.visual import CameraCalibration, CameraRig, VisualPolicyResult
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.sim.sensors.visual import VisualSession, VisualSnapshot

pytestmark = pytest.mark.native


@dataclass
class ExplicitStatePolicy:
    invalid: bool = False

    def describe(self):
        return {"version": "test-explicit-state-v1", "invalid": self.invalid}

    def initial_memories(self, count):
        return np.zeros((count, 2))

    def infer(self, observation, memories, rngs):
        assert not any(hasattr(observation, name) for name in
                       ("depth", "instances", "metric", "features", "neighbors", "world"))
        memories[:, 0] += 1
        memories[:, 1] = [rng.random() for rng in rngs]
        return VisualPolicyResult(np.full(len(rngs), 12 if self.invalid else 7), memories)


@pytest.fixture
def rig():
    return CameraRig((CameraCalibration.front(80, 48),))


def assert_snapshot_equal(first, second):
    assert first.world == second.world
    assert first.metadata == second.metadata
    assert first.arrays.keys() == second.arrays.keys()
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], second.arrays[key])


def test_actor_only_batch_and_multirate_history(straight_scene, rig):
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig, sensor_stride=2) as session:
        initial = session.observe()
        np.testing.assert_array_equal(initial.history_ticks, [[-1, 0], [-1, 0]])
        assert not initial.history_valid[:, 0].any()
        assert initial.history_valid[:, 1].all()
        first = session.step()
        second = session.step()
        assert first.newly_decided and not second.newly_decided
        assert first.observation_tick == second.observation_tick == 0
        np.testing.assert_array_equal(first.policy_actions, second.policy_actions)
        np.testing.assert_array_equal(session.memories[:, 0], [1, 1])
        assert session.memories[0, 1] != session.memories[1, 1]
        third = session.step()
        assert third.newly_decided and third.observation_tick == 2
        np.testing.assert_array_equal(third.observation.history_ticks, [[0, 2], [0, 2]])
        np.testing.assert_array_equal(session.memories[:, 0], [2, 2])
        assert session.tick == 3


def test_snapshot_disk_restores_held_actions_images_rng_and_memories(straight_scene, rig, tmp_path):
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig, sensor_stride=3, seed=83) as session:
        session.step()
        snapshot = session.snapshot()
        path = tmp_path / "visual-snapshot.npz"
        snapshot.save(path)
        loaded = VisualSnapshot.load(path)
        expected = [session.step() for _ in range(5)]
        expected_snapshot = session.snapshot()
        session.restore(loaded)
        for wanted in expected:
            actual = session.step()
            np.testing.assert_array_equal(actual.observation.rgb, wanted.observation.rgb)
            np.testing.assert_array_equal(actual.executed_actions, wanted.executed_actions)
            np.testing.assert_array_equal(actual.transition.observation.features,
                                          wanted.transition.observation.features)
        assert_snapshot_equal(session.snapshot(), expected_snapshot)
        with np.load(path, allow_pickle=False) as data:
            assert all(data[key].dtype.kind != "O" for key in data.files)


def test_fork_identity_no_parent_or_constructor_buffer_alias(straight_scene, rig):
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig) as session:
        session.step()
        straight_scene.poses[:] += 1000
        observed = session.observe()
        observed.rgb.setflags(write=True)
        observed.rgb[:] = 0
        assert session.observe().rgb.any()
        before = session.snapshot()
        with session.fork("intervention") as child:
            assert child.branch_id != session.branch_id
            assert child.last_sensor.branch_id == "intervention"
            assert child.last_sensor.episode_id == session.last_sensor.episode_id
            step = child.step({"a": 1})
            assert step.overrides.tolist() == [True, False]
            assert step.policy_actions[0] == 7 and step.executed_actions[0] == 1
            assert_snapshot_equal(session.snapshot(), before)
            child.restore(before)
            first, second = child.step(), session.step()
            np.testing.assert_array_equal(first.observation.rgb, second.observation.rgb)
            np.testing.assert_array_equal(child.memories, session.memories)


def test_invalid_policy_and_override_are_atomic(straight_scene, rig):
    with VisualSession(straight_scene, ExplicitStatePolicy(invalid=True), rig) as session:
        before = session.snapshot()
        with pytest.raises(ValueError, match="policy actions"):
            session.step()
        assert_snapshot_equal(session.snapshot(), before)
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig) as session:
        before = session.snapshot()
        for overrides in ({"missing": 1}, {"a": True}, {"a": 12}):
            with pytest.raises(ValueError):
                session.step(overrides)
            assert_snapshot_equal(session.snapshot(), before)


def test_invalid_sensor_rejects_without_advancing_or_consuming_memory(straight_scene, rig):
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig) as session:
        render = session.renderer.render_batch
        session.renderer.render_batch = lambda frame, ids: replace(
            render(frame, ids), valid=np.zeros((len(ids), 1), bool))
        before = session.snapshot()
        with pytest.raises(ValueError, match="invalid actor cameras"):
            session.step()
        assert_snapshot_equal(session.snapshot(), before)


def test_restore_rejects_clock_and_configuration_then_reset_owns_new_generation(straight_scene, rig):
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig) as session:
        session.step()
        original = session.snapshot()
        metadata = dict(original.metadata, tick=2)
        with pytest.raises(ValueError, match="clock"):
            session.restore(VisualSnapshot(original.world, metadata, original.arrays))
        assert_snapshot_equal(session.snapshot(), original)
        with VisualSession(straight_scene, ExplicitStatePolicy(), rig, sensor_stride=2) as other:
            with pytest.raises(ValueError, match="configuration"):
                other.restore(original)
        with VisualSession(straight_scene, ExplicitStatePolicy(), rig, seed=1) as differently_seeded:
            with pytest.raises(ValueError, match="configuration"):
                differently_seeded.restore(original)
        observation = session.reset()
        assert session.tick == 0 and session.generation == 1
        assert session.episode_id != original.metadata["episode_id"]
        assert session.last_sensor.generation == 1
        np.testing.assert_array_equal(observation.history_ticks[:, 0], -1)
        np.testing.assert_array_equal(session.memories, 0)


def test_rgb_intervention_causes_later_response_with_replay_and_far_controls():
    scene = following_scene(4, 14, duration=4)
    rig = CameraRig((CameraCalibration.front(160, 96),))
    with VisualSession(scene, PixelBrakePolicy(), rig) as base:
        for _ in range(8):
            base.step()
        with base.fork("brake") as brake, base.fork("far") as far, base.fork("replay") as replay:
            follower = base.actor_ids.index("follower")
            differences, visual_differences = [], []
            for offset in range(25):
                def braking(session, actor):
                    row = session.observe().telemetry[session.actor_ids.index(actor)]
                    return quantize_jerk(-3 if row[0] > 1 else 0, 0, row[1:3], row[0])
                baseline = base.step()
                changed = brake.step({"lead": braking(brake, "lead")})
                irrelevant = far.step({"far": braking(far, "far")})
                replayed = replay.step({"lead": braking(replay, "lead"),
                                        "follower": int(baseline.executed_actions[follower])})
                if offset == 0:
                    np.testing.assert_array_equal(changed.observation.rgb, baseline.observation.rgb)
                    np.testing.assert_array_equal(changed.policy_actions, baseline.policy_actions)
                differences.append(changed.policy_actions[follower] != baseline.policy_actions[follower])
                visual_differences.append(not np.array_equal(changed.observation.rgb[follower],
                                                            baseline.observation.rgb[follower]))
                np.testing.assert_array_equal(irrelevant.observation.rgb[follower], baseline.observation.rgb[follower])
                assert irrelevant.policy_actions[follower] == baseline.policy_actions[follower]
                assert replayed.executed_actions[follower] == baseline.executed_actions[follower]
                np.testing.assert_array_equal(replay.world.state()[0], base.world.state()[0])
            assert any(visual_differences[1:])
            assert any(differences[1:])
            assert np.flatnonzero(visual_differences)[0] <= np.flatnonzero(differences)[0]


def rear_camera(front):
    turn = np.diag([-1., -1., 1., 1.])
    return replace(front, camera_id="rear", camera_to_body=turn @ front.camera_to_body)


def test_pixel_policy_selects_front_by_id_after_camera_reordering(rig):
    scene = following_scene(4, 14, duration=2)
    rear = rear_camera(rig.cameras[0])
    reordered = CameraRig((rear, rig.cameras[0]))
    with VisualSession(scene, PixelBrakePolicy(), rig) as front_only:
        with VisualSession(scene, PixelBrakePolicy(), reordered) as multi:
            expected, actual = front_only.step(), multi.step()
            assert actual.observation.camera_ids == ("rear", "front")
            np.testing.assert_array_equal(actual.observation.camera_to_body[1],
                                          rig.cameras[0].camera_to_body)
            np.testing.assert_array_equal(actual.observation.rgb[:, :, 1], expected.observation.rgb[:, :, 0])
            assert not np.array_equal(actual.observation.rgb[0, -1, 0], actual.observation.rgb[0, -1, 1])
            np.testing.assert_array_equal(actual.policy_actions, expected.policy_actions)
            np.testing.assert_array_equal(multi.memories, front_only.memories)
            assert actual.diagnostics == expected.diagnostics


@pytest.mark.parametrize("camera_case", ["rear_only", "rotated_front", "shifted_front"])
def test_pixel_policy_rejects_unsupported_camera_before_advancing(straight_scene, rig, camera_case):
    front = rig.cameras[0]
    if camera_case == "rear_only":
        camera = rear_camera(front)
    elif camera_case == "rotated_front":
        camera = replace(rear_camera(front), camera_id="front")
    else:
        transform = front.camera_to_body.copy()
        transform[1, 3] += .4
        camera = replace(front, camera_to_body=transform)
    selected = CameraRig((camera,))
    with VisualSession(straight_scene, PixelBrakePolicy(), selected) as session:
        before = session.snapshot()
        with pytest.raises(ValueError, match="front"):
            session.step()
        assert_snapshot_equal(session.snapshot(), before)
    # The rig remains valid for a general policy that handles that calibration.
    with VisualSession(straight_scene, ExplicitStatePolicy(), selected) as generic:
        step = generic.step()
        assert generic.tick == 1
        assert step.observation.camera_ids == (camera.camera_id,)
        np.testing.assert_array_equal(step.observation.camera_to_body[0], camera.camera_to_body)


@pytest.mark.parametrize("changed", ["extrinsic", "camera_id"])
def test_snapshot_rejects_actor_calibration_misalignment(straight_scene, rig, changed):
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig) as session:
        session.step()
        before = session.snapshot()
        arrays, metadata = dict(before.arrays), dict(before.metadata)
        if changed == "extrinsic":
            arrays["observation_camera_to_body"] = arrays["observation_camera_to_body"].copy()
            arrays["observation_camera_to_body"][0, 1, 3] += .1
        else:
            metadata["observation"] = dict(metadata["observation"], camera_ids=("rear",))
        with pytest.raises(ValueError, match="camera geometry"):
            session.restore(VisualSnapshot(before.world, metadata, arrays))
        assert_snapshot_equal(session.snapshot(), before)


def test_unverified_heading_requires_opt_in_and_is_bound_to_fork_and_restore(straight_scene, rig):
    straight_scene.metadata.pop("heading_qualification")
    with pytest.raises(ValueError, match="Unverified camera heading"):
        VisualSession(straight_scene, ExplicitStatePolicy(), rig)
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig, allow_unverified_heading=True) as session:
        session.step()
        assert all(item["heading_qualification"] == "unverified_box_axis"
                   for item in session.describe()["bindings"])
        with session.fork("virtual-axis") as child:
            assert child.allow_unverified_heading
            assert child.describe()["configuration_hash"] == session.describe()["configuration_hash"]
    straight_scene.metadata["heading_qualification"] = dict.fromkeys(straight_scene.actor_ids, "declared_synthetic")
    with VisualSession(straight_scene, ExplicitStatePolicy(), rig, allow_unverified_heading=True) as opted_in:
        with VisualSession(straight_scene, ExplicitStatePolicy(), rig) as strict:
            with pytest.raises(ValueError, match="configuration"):
                strict.restore(opted_in.snapshot())
