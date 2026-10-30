import json

import numpy as np
import pytest

from driversi.rsi.plugins.feedback import prepare_evaluation, run_feedback
from driversi.rsi.plugins.student import prepare_student, run_student


def test_feedback_rejects_foreign_dataset(tmp_path):
    dataset = {"summary": {"artifact_reference": {"identity": "data"}}}
    student = {"summary": {"artifact_reference": {"identity": "student"}}}
    evaluation = {"summary": {"dataset_identity": "different", "student_identity": "student"}}
    with pytest.raises(ValueError, match="lineage"):
        run_feedback({}, dict(dataset=dataset, student=student, evaluation=evaluation), tmp_path / "out", tmp_path / "db")


def test_evaluation_reserves_both_arms(straight_scene, tmp_path):
    path = tmp_path / "scene.npz"
    straight_scene.save(path)
    with pytest.raises(ValueError, match="Reserve both"):
        prepare_evaluation(dict(protocol="structured-paired-diagnostic-v1", scenes=[str(path)],
                                max_sim_agent_steps=1), tmp_path, tmp_path / "db")


def test_student_consumes_labels_and_changes_weights(tmp_path):
    pytest.importorskip("torch")
    from tests.support.diagnostic_config import DEFAULT_AGENT
    from driversi.data.assets.mtgs_assets import file_sha256
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps({**DEFAULT_AGENT, "parameters": {"hidden_sizes": []}}))
    config, _ = prepare_student(dict(purpose="structured_imitation_diagnostic", agent_config=str(agent_path),
        parent=None, seed=7, epochs=3, batch_size=4, learning_rate=.001), tmp_path, tmp_path / "registry")
    samples = tmp_path / "samples.npz"
    np.savez(samples, observations=np.ones((8, 1125), np.float32), actions=np.ones(8, np.int64))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(allowed_uses=["structured_imitation_diagnostic"], samples=8,
        samples_sha256=file_sha256(samples), source_groups=["log-a"], recipe="test")))
    dataset = {"artifacts": {"samples": str(samples), "report": str(manifest)},
               "summary": {"artifact_reference": {"identity": "dataset-1"}}}
    result = run_student(config, {"dataset": dataset}, tmp_path / "student", tmp_path / "registry")
    assert result["summary"]["samples_consumed"] == 8
    assert result["summary"]["sample_exposures"] == 24
    assert result["summary"]["weights_changed"]
    assert result["summary"]["final_loss"] < result["summary"]["initial_loss"]
    assert not result["admission"]["trainable"]


def test_unknown_parent_provenance_rejected_before_training(tmp_path):
    pytest.importorskip("torch")
    from unittest.mock import patch

    from tests.support.diagnostic_config import DEFAULT_AGENT
    from driversi.rsi.plugins import student as harness_student
    agent = tmp_path / "agent.json"
    agent.write_text(json.dumps(DEFAULT_AGENT))
    parent = tmp_path / "parent.npz"
    np.savez(parent, metadata=json.dumps({"training_method": "legacy"}))
    with patch.object(harness_student, "load_inference", return_value=None):
        with pytest.raises(ValueError, match="provenance"):
            prepare_student(dict(purpose="structured_imitation_diagnostic", agent_config=str(agent),
                parent=str(parent), seed=7, epochs=1, batch_size=4, learning_rate=.001), tmp_path, tmp_path / "db")


def test_evaluation_rejects_ancestral_training_overlap(straight_scene, tmp_path):
    from driversi.rsi.plugins.feedback import run_evaluation
    from driversi.training.sampling.environment import source_group
    path = tmp_path / "eval.npz"
    straight_scene.save(path)
    config, _ = prepare_evaluation(dict(protocol="structured-paired-diagnostic-v1", scenes=[str(path)],
        max_sim_agent_steps=10000), tmp_path, tmp_path / "db")
    inputs = {"dataset": {"summary": {"artifact_reference": {"identity": "data"}}},
              "student": {"summary": {"dataset_identity": "data", "implementation_files": {},
                           "source_groups": [source_group(straight_scene)]}}}
    with pytest.raises(ValueError, match="overlaps"):
        run_evaluation(config, inputs, tmp_path / "out", tmp_path / "db")
    assert not (tmp_path / "out").exists()
    config["rules"]["progress_weight"] = 20
    with pytest.raises(ValueError, match="protocol changed"):
        run_evaluation(config, inputs, tmp_path / "out", tmp_path / "db")


@pytest.mark.parametrize("name,package", [("mlp", "privileged_mlp"), ("linear", "linear_actor_critic")])
def test_model_versioning_includes_implementation_behind_facade(tmp_path, name, package):
    pytest.importorskip("torch")
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    config, _ = prepare_student(dict(purpose="structured_imitation_diagnostic",
        agent_config=str(root / "tests/fixtures/legacy/agents" / (name + ".yaml")), parent=None,
        seed=7, epochs=1, batch_size=4, learning_rate=.001), tmp_path, tmp_path / "db")
    expected = [root / "tests/support/diagnostic_agents.py",
                root / "tests/support/diagnostic_agents.py",
                root / "src/training/legacy/frozen.py"]
    # Linear inherits the MLP implementation and shares its inference class.
    for path in expected:
        implementation = str(path.resolve())
        assert implementation in config["implementation_files"]
    config["implementation_files"][implementation] = "changed"
    with pytest.raises(ValueError, match="implementation changed"):
        run_student(config, {}, tmp_path / "out", tmp_path / "db")
    assert not (tmp_path / "out").exists()
