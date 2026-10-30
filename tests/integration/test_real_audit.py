"""Audits must reject corrupted events and distinguish unexercised worlds."""

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.evaluation.benchmarks.real_scenes import summarize_rollout, verify_replay
from driversi.sim.execution.rollout import rollout

pytestmark = pytest.mark.native


def test_replay_audit_checks_recorded_events(straight_scene, tmp_path):
    path = tmp_path / "run.npz"
    rollout(straight_scene, path)
    assert all(verify_replay(straight_scene, path).values())
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    arrays["collision"][0, 0] = ~arrays["collision"][0, 0]
    np.savez_compressed(path, **arrays)
    assert verify_replay(straight_scene, path)["joint_action_replay_exact"] is False


def test_initially_offroad_is_not_successful_interaction(straight_scene, tmp_path):
    straight_scene.poses[:, 0, 1] = -20
    straight_scene.poses[:, 1, 1] = 20
    path = tmp_path / "stopped.npz"
    rollout(straight_scene, path)
    behavior = summarize_rollout(straight_scene, path)
    assert behavior["steps"] == 0
    assert behavior["valid_agent_steps"] == 0
    assert behavior["initial_inactive_actors"] == 2
    assert all(actor["initial_inactive"] for actor in behavior["actors"].values())
    assert verify_replay(straight_scene, path) == {
        "joint_action_replay_exact": None, "mid_episode_restore_exact": None}


def test_initial_absent_origin_is_not_a_contact(straight_scene, tmp_path):
    straight_scene.presence = np.ones(straight_scene.poses.shape[:2], bool)
    straight_scene.presence[:, 2] = False
    path = tmp_path / "absent.npz"
    rollout(straight_scene, path)
    report = summarize_rollout(straight_scene, path)
    assert report["initial_controlled_2d_contacts"] == 0
    assert not report["actors"]["a"]["initial_2d_contact"]
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "presence"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="recorded presence"):
        summarize_rollout(straight_scene, path)
