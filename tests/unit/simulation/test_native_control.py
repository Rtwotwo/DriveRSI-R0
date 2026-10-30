import numpy as np
import pytest

from driversi.rendering.bev.replay import export_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout


def target(w, actor, dx=0.0, y=None):
    i = w._scene.actor_ids.index(actor)
    s = w.state()[i]
    m = w.motion()[i]
    return [s[0] + dx, s[1] if y is None else y, s[2], s[3], *s[4:7], *m]


def test_native_targets_joint_collision_and_restore(straight_scene):
    with PufferDriveWorld(straight_scene) as w:
        before = w.snapshot()
        t = w.step_native(
            {"a": target(w, "a", 0.4), "b": target(w, "b", 0.4, y=0.0)}, observation_tick=0
        )
        assert t.events["collision"].all()
        assert ("a", "b") in w.contacts()
        state = w.state()
        w.restore(before)
        w.step_native(
            {"a": target(w, "a", 0.4), "b": target(w, "b", 0.4, y=0.0)}, observation_tick=0
        )
        np.testing.assert_array_equal(w.state(), state)


@pytest.mark.parametrize("case", ["missing", "duplicate_owner", "nonfinite", "stale", "teleport"])
def test_native_rejection_atomic(straight_scene, case):
    with PufferDriveWorld(straight_scene) as w:
        before = w.snapshot()
        values = {a: target(w, a, 0.4) for a in w.actor_ids}
        tick = 0
        if case == "missing":
            values.pop("a")
        if case == "duplicate_owner":
            values["replay"] = target(w, "replay")
        if case == "nonfinite":
            values["a"][0] = float("nan")
        if case == "stale":
            tick = 1
        if case == "teleport":
            values["a"][0] = 100.0
        with pytest.raises(ValueError):
            w.step_native(values, observation_tick=tick)
        assert before == w.snapshot()


def test_native_no_leak_to_discrete(straight_scene):
    with PufferDriveWorld(straight_scene) as w, PufferDriveWorld(straight_scene) as ref:
        original = w.snapshot()
        w.step_native({a: target(w, a, 0.4) for a in w.actor_ids}, observation_tick=0)
        w.restore(original)
        for _ in range(10):
            a = {x: 7 for x in w.actor_ids}
            w.step(a)
            ref.step(a)
            np.testing.assert_array_equal(w.state(), ref.state())
            np.testing.assert_array_equal(w.motion(), ref.motion())


def test_native_continuous_record_and_viewer(straight_scene, tmp_path):
    straight_scene.metadata["native_initial_motion"] = {"a": [0.5, 0.0, 0.1], "b": [0.0, 0.0, 0.0]}
    with PufferDriveWorld(straight_scene) as w:
        initial = w.snapshot()
        r = RolloutRecord.start(w)
        for tick in range(3):
            actions = {a: [0.2, 0.1] for a in w.actor_ids}
            tr = w.step_native({a: target(w, a, 0.4) for a in w.actor_ids}, observation_tick=tick)
            r.append(
                w,
                actions,
                tr,
                [dict(actor=a, action=v, controller="test-native") for a, v in actions.items()],
            )
        write_rollout(
            straight_scene,
            tmp_path / "r.npz",
            record=r,
            snapshot=initial,
            controlled_ids=w.actor_ids,
            policy="test-native",
            controller_configuration={"action_space": "driverl_native_jerk_steering_rate"},
        )
        export_replay(straight_scene, tmp_path / "r.npz", tmp_path / "viewer.html")
        with np.load(tmp_path / "r.npz") as f:
            assert f["actions"].shape == (3, 2, 2)


def test_native_initial_motion_is_snapshot_identity(straight_scene):
    import copy

    revised = copy.deepcopy(straight_scene)
    revised.metadata["native_initial_motion"] = {"a": [0.5, 0.2, 0.1], "b": [-0.3, 0.0, -0.1]}
    with PufferDriveWorld(revised) as w, PufferDriveWorld(straight_scene) as old:
        np.testing.assert_allclose(w.motion()[:2], [[0.5, 0.2, 0.1], [-0.3, 0.0, -0.1]])
        with pytest.raises(ValueError):
            w.restore(old.snapshot())
        w.step({"a": 7, "b": 7})
        before = w.snapshot()
        with pytest.raises(ValueError):
            w._native.initialize_motion(w._handle, np.zeros((2, 3)))
        assert w.snapshot() == before
