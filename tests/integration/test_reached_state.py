import numpy as np
import pytest

from driversi.agents.traffic_control.following import RouteFollower
from driversi.evaluation.search.augmentation import generate_pair
from driversi.rsi.correction.reached_state import verify_reached_state
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import write_rollout
from tests.integration.test_augmentation import following_case


def case(scene, tmp_path):
    scene, routes = following_case(scene)
    scene.metadata["source_log"] = "training-log"
    report, records, initial, boundaries = generate_pair(
        scene, routes, actor="a", explore_steps=15, explore_speed=1.0
    )
    path = tmp_path / "rollout.npz"
    write_rollout(
        scene,
        path,
        record=records[1],
        snapshot=initial,
        controlled_ids=tuple(routes),
        policy="test-reactive",
        controller_configuration=RouteFollower(routes).describe(),
    )
    return scene, path, boundaries[1]


def test_actual_history_and_training_purposes(straight_scene, tmp_path):
    scene, path, anchor = case(straight_scene, tmp_path)
    with PufferDriveWorld(scene) as world:
        world.restore(anchor)
        task = verify_reached_state(scene, path, world, 15, actor="a")
    assert task["history_ticks"] == [0, 5, 10, 15]
    assert task["required_teacher_future_ticks"] == 40
    assert not task["original_log_score_applicable"]
    assert not task["trainable"] and not task["teacher_qualified"]
    assert "not_admitted" in task["training_uses"]["on_policy_rl"]


def test_wrong_snapshot_and_missing_history_rejected(straight_scene, tmp_path):
    scene, path, anchor = case(straight_scene, tmp_path)
    with PufferDriveWorld(scene) as world:
        with pytest.raises(ValueError, match="live and at"):
            verify_reached_state(scene, path, world, 15, actor="a")
        world.restore(anchor)
        with pytest.raises(ValueError, match="15 ticks"):
            verify_reached_state(scene, path, world, 14, actor="a")
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k].copy() for k in data.files}
        payload["states"][15, 1, 0] += 1
        np.savez_compressed(path, **payload)
        with pytest.raises(ValueError, match="state mismatch"):
            verify_reached_state(scene, path, world, 15, actor="a")


def test_absent_owner_not_filled_from_source(straight_scene, tmp_path):
    scene, path, anchor = case(straight_scene, tmp_path)
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k].copy() for k in data.files}
    payload["presence"][5, 0] = False
    np.savez_compressed(path, **payload)
    with PufferDriveWorld(scene) as world:
        world.restore(anchor)
        with pytest.raises(ValueError, match="missing from arrival"):
            verify_reached_state(scene, path, world, 15, actor="a")


def test_pending_task_does_not_require_future_supervision(straight_scene, tmp_path):
    scene, routes = following_case(straight_scene)
    scene.metadata["source_log"] = "training-log"
    _, records, initial, anchors = generate_pair(scene, routes, actor="a", explore_steps=15)
    from driversi.sim.persistence.recording import RolloutRecord

    a = records[1]
    prefix = RolloutRecord(
        states=a.states[:16],
        motion=a.motion[:16],
        transitions=a.transitions[:15],
        actions=a.actions[:15],
        diagnostics=a.diagnostics[:15],
        presence=a.presence[:16],
        lifecycle=a.lifecycle[:16],
    )
    path = tmp_path / "prefix.npz"
    write_rollout(
        scene,
        path,
        record=prefix,
        snapshot=initial,
        controlled_ids=tuple(routes),
        policy="test",
        controller_configuration=RouteFollower(routes).describe(),
    )
    with PufferDriveWorld(scene) as world:
        world.restore(anchors[1])
        task = verify_reached_state(scene, path, world, 15, actor="a")
    assert task["arrival_verified"] and task["available_future_ticks"] == 0
    assert task["required_teacher_future_ticks"] == 40 and not task["trainable"]
