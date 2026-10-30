import json
from pathlib import Path

import numpy as np
import pytest

from driversi.evaluation.benchmarks.visual import evaluate_visual
from driversi.sim.sensors.visual_rollout import rollout_visual

pytestmark = pytest.mark.native
pytest.importorskip("driversi.sim._native")


def test_record_keeps_input_clock_final_frame_and_nonlearning_mask(straight_scene, tmp_path):
    straight_scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
    directory = tmp_path / "visual"
    report = rollout_visual(straight_scene, directory, width=80, height=48, sensor_stride=2)
    assert report["joint_steps"] == 30
    assert report["decisions"] == 15
    with np.load(directory / "visual.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["final_observation_tick"] == 30
        assert data["states"].shape[0] == 31
        assert data["rgb"].shape[:3] == (30, 2, 2)
        assert not data["trainable"].any()
        assert data["physical_actor_ids"].tolist() == ["a", "b", "replay"]
        assert data["observation_tick"].tolist() == [i//2*2 for i in range(30)]
        np.testing.assert_array_equal(data["rgb"][0], data["rgb"][1])
        np.testing.assert_array_equal(data["policy_actions"], data["executed_actions"])
        assert data["final_history_ticks"][0].tolist() == [28, 30]
    assert (directory / "manifest.json").is_file()
    assert (directory / "viewer.html").is_file()
    with pytest.raises(ValueError, match="new"):
        rollout_visual(straight_scene, directory)


def test_full_paired_visual_protocol_preserves_collision_evidence(tmp_path):
    config = Path(__file__).resolve().parents[2] / "src/config/evaluation/visual.toml"
    report = evaluate_visual(config, tmp_path / "protocol")
    assert report["passed"], report["checks"]
    assert report["first_changed_image_seconds"] > 1.
    assert report["first_changed_action_seconds"] >= report["first_changed_image_seconds"]
    assert report["branches"]["action_replay"]["actual_seconds"] == 6.
    assert report["replay_precontact_states"] <= 61
    with np.load(tmp_path / "protocol/action_replay/visual.npz", allow_pickle=False) as data:
        expected = int(np.any(data["collision"], axis=0).sum())
        assert report["branches"]["action_replay"]["events"]["collision"] == expected
        assert data["states"].shape[0] == 61  # keep contact and terminal successor


def test_held_observation_alive_preserves_input_after_physical_termination(straight_scene, tmp_path):
    # Actor a enters the 2 m goal radius on the first physical step. Actor b
    # stays active so a second step can consume the held tick-0 observation.
    straight_scene.metadata["task_goals_xy"] = {"a": [2.2, 0], "b": [100, 8]}
    directory = tmp_path / "held-termination"
    rollout_visual(straight_scene, directory, width=80, height=48, sensor_stride=2)
    with np.load(directory / "visual.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["schema_version"] == 1  # additive field; legacy readers remain usable
        assert "observation_tick" in metadata["observation_alive_semantics"]
        assert "physical transition" in metadata["physical_valid_semantics"]
        assert data["observation_alive"].dtype == np.bool_
        assert data["observation_alive"].shape == data["valid"].shape == (30, 2)
        assert data["observation_tick"][:3].tolist() == [0, 0, 2]
        assert data["newly_decided"][:3].tolist() == [True, False, True]
        assert data["terminated"][0].tolist() == [True, False]
        assert data["goal"][0].tolist() == [True, False]
        assert data["observation_alive"][:3, 0].tolist() == [True, True, False]
        assert data["valid"][:3, 0].tolist() == [True, False, False]
        np.testing.assert_array_equal(data["rgb"][0], data["rgb"][1])
        np.testing.assert_array_equal(data["policy_actions"][0], data["policy_actions"][1])
        assert data["observation_alive"][:, 1].all()
        assert data["valid"][:, 1].all()
        assert data["final_alive"].tolist() == [False, True]
    # Existing visual and physical exporters still consume the additive NPZ.
    assert (directory / "viewer.html").is_file()
    assert (directory / "bev.html").is_file()
