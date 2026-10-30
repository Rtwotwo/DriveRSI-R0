import copy
import json
import sqlite3

import numpy as np
import pytest

from driversi.data.sources.nuplan_db import compile_window, read_window
from driversi.data.participants.nuplan_motion import REAR_AXLE_TO_CENTER, ego_center_motion
from driversi.sim.execution.continuous import ContinuousSession
from driversi.sim.execution.structured_history import StructuredHistory
from tests.unit.evaluation.test_continuous_world import MemoryPolicy, long_scene
from tests.unit.data.test_nuplan_db import make_db, token


class InitialWorld:
    def __init__(self, scene):
        self.scene = scene

    def state(self):
        s = self.scene
        out = np.zeros((len(s.actor_ids), 13))
        out[:, :4] = s.poses[0]
        out[:, 4:7] = s.velocities[0]
        return out

    def presence(self):
        return self.scene.presence[0]

    def observe(self):
        class Clock:
            step = 0

        return Clock()


def source_history(tmp_path, **db_options):
    path = make_db(tmp_path, **db_options)
    window = read_window(path, token=token(140).hex(), duration=2)
    scene = compile_window(window, duration=2)
    return window, scene, StructuredHistory(scene, InitialWorld(scene), 20)


def test_real_timestamp_sampling_no_future_ids_and_no_missing_object_padding(tmp_path):
    _, _, history = source_history(tmp_path)
    sample = history.sample()
    assert sample["frame_valid"].all()
    assert tuple(sample["origins"]) == ("source_history",) * 3 + ("physical_world",)
    assert sample["requested_timestamp_us"].tolist() == [1500000, 2000000, 2500000, 3000000]
    ped = sample["actor_ids"].index(token(21).hex())
    assert sample["present"][ped].tolist() == [False, False, False, True]
    assert not sample["derivative_valid"][ped].any()
    np.testing.assert_array_equal(sample["states"][ped, :3], 0)
    assert np.max(sample["sampled_timestamp_us"]) == sample["anchor_timestamp_us"]
    # A future-only identity in the simulation registry is not an observation.
    history.scene.actor_ids += ("future-only",)
    assert "future-only" not in history.sample()["actor_ids"]


def test_missing_frame_is_invalid_not_repeated_and_velocity_mask_survives(tmp_path):
    _, _, history = source_history(tmp_path)
    history.records = type(history.records)(
        r for r in history.records if not 1750000 < r["timestamp_us"] <= 2000000
    )
    sample = history.sample()
    assert sample["frame_valid"].tolist() == [True, False, True, True]
    assert sample["sampled_timestamp_us"][1] == -1
    assert not sample["present"][:, 1].any()
    assert not sample["derivative_valid"][:, 1:3].any()


def test_missing_velocity_is_not_a_stopped_actor(tmp_path):
    _, _, history = source_history(tmp_path)
    for r in history.records:
        if r["origin"] == "source_history":
            r["actors"][token(20).hex()]["velocity_valid"] = False
    sample = history.sample()
    i = sample["actor_ids"].index(token(20).hex())
    assert sample["present"][i].all()
    assert not sample["velocity_valid"][i, :3].any()
    assert not sample["derivative_valid"][i].any()
    assert sample["velocity_valid"][i, -1]


def test_initial_motion_uses_local_reported_velocity_not_global_or_pose_jitter(tmp_path):
    path = make_db(tmp_path, yaw=np.pi / 2)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE ego_pose SET vx=2,vy=1")
        c.execute("UPDATE ego_pose SET x=x+0.4 WHERE timestamp=2950123")
    window = read_window(path, token=token(140).hex(), duration=2)
    scene = compile_window(window, duration=2)
    np.testing.assert_allclose(scene.velocities[0, 0, :2], [-1, 2], atol=1e-6)
    assert "reported_local" in scene.metadata["nuplan_source"]["initial_velocity_source"]
    a, b = copy.deepcopy(window["history"][-2:])
    a["ego_quaternion_wxyz"] = [1, 0, 0, 0]
    b["ego_quaternion_wxyz"] = [np.cos(0.01 / 2), 0, 0, np.sin(0.01 / 2)]
    _, yaw, velocity, valid = ego_center_motion(b, a)
    expected = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [
        2,
        1 + REAR_AXLE_TO_CENTER * 0.2,
    ]
    np.testing.assert_allclose(velocity, expected, atol=1e-8)
    assert valid
    b["ego_pose_timestamp_us"] = a["ego_pose_timestamp_us"]
    with pytest.raises(ValueError, match="pose timestamps"):
        ego_center_motion(b, a)


def test_history_rotation_uses_centers_and_vectors(tmp_path):
    _, _, history = source_history(tmp_path, yaw=np.pi / 2)
    value = history.sample()
    np.testing.assert_allclose(value["states"][0, -1, :3], 0, atol=1e-6)
    np.testing.assert_allclose(value["states"][0, -1, 3:5], [4, 0], atol=1e-6)
    assert "center" in value["reference"]
    assert value["ego_rear_axle_to_center_m"] == REAR_AXLE_TO_CENTER


def test_future_source_history_rejected(tmp_path):
    _, scene, _ = source_history(tmp_path)
    scene.metadata["nuplan_source"]["history"][-1]["timestamp_us"] += 1
    with pytest.raises(ValueError, match="future leakage"):
        StructuredHistory(scene, InitialWorld(scene), 20)


def test_gap_between_selected_observations_does_not_create_derivative(tmp_path):
    _, _, history = source_history(tmp_path)
    for r in history.records:
        if r["timestamp_us"] == 2200000:
            r["actors"].pop(token(20).hex())
    sample = history.sample()
    i = sample["actor_ids"].index(token(20).hex())
    assert sample["present"][i].all()
    assert not sample["derivative_valid"][i, 2]
    # Source -> physical world is a provenance boundary, not measured acceleration.
    assert not sample["derivative_valid"][:, -1].any()


class StructuredPolicy(MemoryPolicy):
    def describe(self):
        return {
            "version": "structured-test",
            "structured_history": {"actor": "lead", "frames": 4, "interval": 0.5},
        }

    def infer_structured(self, obs, context, memories, rngs):
        assert context["anchor_timestamp_us"] == obs.step * 100000
        assert context["actor_ids"][0] == "lead"
        assert context["origins"][-1] == "physical_world"
        assert context["frame_valid"][-1]
        for a in obs.actor_ids:
            memories[a]["calls"] += 1
            memories[a]["draws"] += int(rngs[a].integers(0, 100))
        return dict.fromkeys(obs.actor_ids, 7), memories, []


def assert_samples_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key]


def test_structured_policy_across_windows_and_json_checkpoint():
    s = ContinuousSession(long_scene(), StructuredPolicy(), seed=12)
    try:
        assert s.model_input(actor="lead")["frame_valid"].tolist() == [False, False, False, True]
        s.advance(95)
        saved = json.loads(json.dumps(s.snapshot()))
        expected = s.model_input(actor="lead")
        s.advance(50)
        end = s.snapshot()
        s.restore(saved)
        assert_samples_equal(s.model_input(actor="lead"), expected)
        s.advance(50)
        assert s.snapshot() == end
        assert s.memories["lead"]["calls"] == 145
    finally:
        s.close()


def test_bad_structured_restore_is_atomic_and_does_not_accept_future_or_terminal_tampering():
    s = ContinuousSession(long_scene(), StructuredPolicy())
    try:
        s.advance(95)
        saved = s.snapshot()
        for mutate in [
            lambda x: x["records"][-1].update(timestamp_us=99999999),
            lambda x: x["records"][-1]["actors"]["lead"]["state"].__setitem__(0, 999),
            lambda x: x["records"][-2]["actors"]["lead"].update(velocity_valid=1),
        ]:
            bad = copy.deepcopy(saved)
            mutate(bad["structured_history"])
            with pytest.raises(ValueError):
                s.restore(bad)
            assert s.snapshot() == saved
    finally:
        s.close()


def test_failed_history_append_rolls_back_joint_step(monkeypatch):
    s = ContinuousSession(long_scene(), StructuredPolicy())
    try:
        before = s.snapshot()

        def fail(world):
            s.structured_history.records.clear()
            raise RuntimeError("history error")

        monkeypatch.setattr(s.structured_history, "append_world", fail)
        with pytest.raises(RuntimeError, match="history error"):
            s.step()
        assert s.snapshot() == before
    finally:
        s.close()


@pytest.mark.parametrize(
    "query",
    [
        {"frames": 5, "interval": 0.5},
        {"frames": 4, "interval": float("nan")},
        {"max_age": 0.5},
        {"actor": "unknown"},
    ],
)
def test_unsupported_requests_are_rejected(query):
    s = ContinuousSession(long_scene(), StructuredPolicy())
    try:
        with pytest.raises(ValueError):
            s.model_input(actor="lead", **query) if "actor" not in query else s.model_input(**query)
    finally:
        s.close()


def test_static_nuplan_window_preserved_without_fabricated_goal(tmp_path):
    from driversi.contracts.driving_task import bind_driving_tasks
    from driversi.contracts.road import Route
    from driversi.data.scenes.pufferdrive import write_binary

    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE ego_pose SET x=100,vx=0,vy=0")
    window = read_window(path, token=token(140).hex(), duration=2)
    scene = compile_window(window, duration=2, max_actors=1)
    assert scene.controlled[0]
    assert scene.metadata["unbound_controlled_tasks"] == ["ego"]
    assert "task_goals_xy" not in scene.metadata
    np.testing.assert_allclose(scene.poses[:, 0, :2], np.tile(scene.poses[0, 0, :2], (21, 1)))
    assert StructuredHistory(scene, InitialWorld(scene), 20).sample()["frame_valid"].all()
    with pytest.raises(ValueError, match="explicit driving tasks"):
        write_binary(scene, tmp_path / "rejected.bin")
    # A separately provided route can enable execution; no synthetic extension
    # is inferred merely from stationary expert poses.
    road = long_scene()
    scene.road_points, scene.road_offsets, scene.road_types = (
        road.road_points,
        road.road_offsets,
        road.road_types,
    )
    scene.metadata["map_complete"] = True
    route = Route(
        np.array([scene.poses[0, 0, :2], scene.poses[0, 0, :2] + [20, 0]]),
        end_reason="source_support_end",
    )
    bound = bind_driving_tasks(scene, {"ego": route})
    write_binary(bound, tmp_path / "bound.bin")
    assert (tmp_path / "bound.bin").is_file()


@pytest.mark.parametrize("offset", [-50000, -3743, 123, 50000])
def test_associated_time_offsets_are_exposed_and_survive_checkpoint(tmp_path, offset):
    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE ego_pose SET timestamp=timestamp-123+?", (offset,))
    window = read_window(path, token=token(140).hex(), duration=2)
    assert window["timestamp_alignment"]["exact_sensor_synchrony"] is False
    scene = compile_window(window, duration=2)
    history = StructuredHistory(scene, InitialWorld(scene), 20)
    sample = history.sample()
    assert sample["ego_pose_measurement_offset_us"].tolist() == [offset, offset, offset, 0]
    assert sample["initial_pose_measurement_offset_us"] == offset
    saved = json.loads(json.dumps(history.snapshot()))
    history.restore(saved, InitialWorld(scene))
    assert_samples_equal(sample, history.sample())


def test_excess_pose_lidar_skew_is_rejected(tmp_path):
    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE ego_pose SET timestamp=timestamp+50000")
    with pytest.raises(ValueError, match="alignment tolerance"):
        read_window(path, token=token(140).hex(), duration=2)
