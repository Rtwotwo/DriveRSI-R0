# ruff: noqa: E402
import argparse
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from driversi.data.datasets.world_model import export_sequences, iter_joint_sequences, load_sequence
from driversi.training.ppo import train


@pytest.fixture
def recorded_run(straight_scene, tmp_path):
    a, b = copy.deepcopy(straight_scene), copy.deepcopy(straight_scene)
    a.metadata["source_log"], b.metadata["source_log"] = "train-drive", "eval-drive"
    a.metadata["task_goals_xy"] = b.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
    a.save(tmp_path / "train.npz")
    b.save(tmp_path / "eval.npz")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"train": [{"path": str(tmp_path / "train.npz")}],
                                "evaluation": [{"path": str(tmp_path / "eval.npz")}], "limits": []}))
    output = tmp_path / "run"
    train(argparse.Namespace(suite=suite, output_dir=output, budget=300, rollout_steps=4,
                             worlds=2, seed=7, device="cpu", resume=None))
    return output


def test_real_joint_sequences_cross_batches_but_not_resets(recorded_run, tmp_path):
    stats = {}
    sequences = list(iter_joint_sequences(recorded_run, context_steps=2, prediction_steps=4,
                                           stride=2, statistics=stats))
    assert sequences and stats["reset_boundaries"] > 0
    assert any(len(set(s.metadata["policy_versions"])) > 1 for s in sequences)
    for sequence in sequences:
        assert sequence.states.shape == (7, 3, 16)
        assert sequence.metadata["end_tick"] - sequence.metadata["start_tick"] == 6
        assert sequence.metadata["end_tick"] <= 30
        assert (sequence.actions[:, 2] == -1).all()
        assert not sequence.action_applied[:, 2].any()
        assert not sequence.state_field_valid[2, [7, 8, 13, 14, 15]].any()
        assert (sequence.states[:, 2, [7, 8, 13, 14, 15]] == 0).all()
    output = tmp_path / "dataset"
    result = export_sequences(recorded_run, output, context_steps=2, prediction_steps=4, stride=2)
    assert result["status"] == "complete"
    assert result["statistics"] == stats
    restored = load_sequence(output, 0)
    np.testing.assert_array_equal(restored.states, sequences[0].states)
    np.testing.assert_array_equal(restored.actions, sequences[0].actions)
    assert "poses" not in restored.model_inputs()["scene"]


def test_future_labels_cannot_change_model_context(recorded_run):
    sequence = next(iter_joint_sequences(recorded_run, context_steps=2, prediction_steps=4))
    changed = sequence.states.copy()
    changed[3:, :, 0] += 50
    alternate = replace(sequence, states=changed)
    inputs, others = sequence.model_inputs(), alternate.model_inputs()
    for key in ("state_history", "action_history", "history_action_applied", "future_actions"):
        np.testing.assert_array_equal(inputs[key], others[key])
    assert not np.array_equal(sequence.training_targets()["future_states"],
                              alternate.training_targets()["future_states"])
    assert not {"terminated", "truncated", "events", "rewards", "action_applied"} & set(inputs)
    assert inputs["remaining_seconds"] == pytest.approx(2.8)
    with pytest.raises(ValueError):
        replace(sequence, scene_context={**sequence.scene_context, "poses": changed})


@pytest.mark.parametrize("corruption", ["missing_motion", "time", "state"])
def test_corrupt_input_is_rejected_and_export_is_not_published(recorded_run, tmp_path, corruption):
    path = sorted((recorded_run / "rollouts").glob("batch-*.npz"))[0]
    with np.load(path, allow_pickle=False) as file:
        arrays = {k: file[k].copy() for k in file.files}
    if corruption == "missing_motion":
        del arrays["world_motion_before"]
    elif corruption == "time":
        arrays["tick"][1, :2] += 2
    else:
        arrays["world_before"][1, 0, 0, 0] += 1
    np.savez_compressed(path, **arrays)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        export_sequences(recorded_run, output, context_steps=2, prediction_steps=4)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="complete"):
        load_sequence(output, 0)
