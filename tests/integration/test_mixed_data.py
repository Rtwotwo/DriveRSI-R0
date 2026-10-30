# ruff: noqa: E402
"""Frozen opponents remain physical data but never become learner PPO samples."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from tests.support.diagnostic_agents import ActorCritic
from tests.support.diagnostic_config import DEFAULT_AGENT
from driversi.data.datasets.world_model import export_sequences, iter_joint_sequences, load_sequence
from driversi.training.sampling.audit import audit_run
from driversi.training.sampling.collector import Collector


@pytest.fixture
def mixed_run(straight_scene, tmp_path):
    torch.set_num_threads(1)
    scene = copy.deepcopy(straight_scene)
    scene.metadata["source_log"] = "mixed-data-training"
    scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
    path = tmp_path / "scene.npz"
    scene.save(path)
    config = copy.deepcopy(DEFAULT_AGENT)
    config["parameters"] = {"hidden_sizes": [8]}
    learner, opponent = ActorCritic(hidden_sizes=(8,)), ActorCritic(hidden_sizes=(8,))
    with torch.no_grad():
        for model, action in ((learner, 4), (opponent, 7)):
            for parameter in model.parameters():
                parameter.zero_()
            model.actor.bias[action] = 20.
            model.critic.bias.fill_(float(action))
    checkpoint = tmp_path / "opponent.npz"
    opponent.export(checkpoint, "historical-v1")
    pool = {"schema_version": 1, "policies": [{"id": "history", "checkpoint": str(checkpoint),
            "agent_config": config, "training_paths": [str(path)]}]}
    root = tmp_path / "run"
    (root / "rollouts").mkdir(parents=True)
    collector = Collector([path], num_worlds=2, seed=5, opponent_pool=pool)
    count = 0
    try:
        for index in range(2):
            version = f"learner-v{index}"
            data = collector.collect(learner, 20, policy_version=version)
            n = int(data["valid"].sum())
            count += n
            meta = {"policy_version": version, "agent": config, "valid_transitions": n,
                    "joint_state_version": "puffer-state13-motion3-v1", "dt_seconds": .1,
                    "scene_paths": [str(path)], "actor_ids": [list(scene.actor_ids)],
                    "controlled_ids": [["a", "b"]], "sampling_mode": "frozen_history_uniform",
                    "behavior_policies": collector.behavior_policies(version),
                    "learner_transitions": int((data["valid"] & data["learner_mask"]
                                                 & data["on_policy_eligible"]).sum())}
            learner.export(root / "rollouts" / f"policy-{index:05d}.npz", version)
            np.savez_compressed(root / "rollouts" / f"batch-{index:05d}.npz",
                                metadata=json.dumps(meta), **data)
    finally:
        collector.close()
    (root / "report.json").write_text(json.dumps({"resume_from": None, "valid_transitions": count}))
    return root


def corrupt_first(root, mutation):
    path = sorted((root / "rollouts").glob("batch-*.npz"))[0]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(data["metadata"]))
    mutation(data, meta)
    data["metadata"] = json.dumps(meta)
    np.savez_compressed(path, **data)


def test_audit_uses_frozen_opponent_probabilities_values_and_learner_gae(mixed_run):
    report = audit_run(mixed_run)
    assert report["mixed_behavior_batches"] == 2
    assert report["frozen_opponent_identities_verified"]
    assert report["learner_transitions_in_this_run"] > 0
    assert report["opponent_transitions_in_this_run"] > 0
    assert report["valid_transitions_in_this_run"] == (
        report["learner_transitions_in_this_run"] + report["opponent_transitions_in_this_run"])


def test_sequences_keep_opponent_actions_versions_and_learning_mask(mixed_run, tmp_path):
    sequences = list(iter_joint_sequences(mixed_run, context_steps=2, prediction_steps=4, stride=2))
    assert sequences
    assert any(len(set(s.metadata["policy_versions"])) > 1 for s in sequences)
    for seq in sequences:
        assert seq.metadata["schema_version"] == 2
        np.testing.assert_array_equal(seq.behavior_policy_index[:, 0], 0)
        np.testing.assert_array_equal(seq.behavior_policy_index[:, 1], 1)
        np.testing.assert_array_equal(seq.behavior_policy_index[:, 2], -1)
        assert (seq.actions[:, 1] == 7).all()
        assert not seq.learning_eligible[:, 1:].any()
        assert seq.action_applied[:, 1].any()
        assert all(row[1] == "historical-v1" and row[2] is None for row in seq.metadata["actor_policy_versions"])
        assert "learning_eligible" not in seq.model_inputs()
        assert "policy_learning_eligible" in seq.training_targets()
    target = tmp_path / "dataset"
    export_sequences(mixed_run, target, context_steps=2, prediction_steps=4, stride=2)
    restored = load_sequence(target, 0)
    np.testing.assert_array_equal(restored.behavior_policy_index, sequences[0].behavior_policy_index)
    np.testing.assert_array_equal(restored.learning_eligible, sequences[0].learning_eligible)
    assert not restored.learning_eligible.flags.writeable


@pytest.mark.parametrize("corruption", ["missing_table", "missing_index", "index", "learner", "eligible", "version"])
def test_mixed_record_corruption_is_rejected_by_audit_and_export(mixed_run, corruption):
    def change(data, meta):
        if corruption == "missing_table":
            del meta["behavior_policies"]
        elif corruption == "missing_index":
            del data["behavior_policy_index"]
        elif corruption == "index":
            data["behavior_policy_index"][0, 1] = 9
        elif corruption == "learner":
            data["learner_mask"][0, 1] = True
        elif corruption == "eligible":
            data["on_policy_eligible"][0, 31] = True
        else:
            meta["behavior_policies"][0]["policy_version"] = "wrong-version"
    corrupt_first(mixed_run, change)
    with pytest.raises(ValueError):
        audit_run(mixed_run)
    with pytest.raises(ValueError):
        list(iter_joint_sequences(mixed_run, context_steps=2, prediction_steps=4))


@pytest.mark.parametrize("corruption", ["hash", "logprob", "value", "next_value", "gae", "count"])
def test_mixed_audit_rejects_false_behavior_evidence(mixed_run, corruption):
    def change(data, meta):
        if corruption == "hash":
            meta["behavior_policies"][1]["checkpoint_sha256"] = "0" * 64
        elif corruption == "count":
            meta["learner_transitions"] += 1
        else:
            key = {"logprob": "log_probs", "value": "values", "next_value": "next_values", "gae": "advantages"}[corruption]
            data[key][0, 1] += 1.
    corrupt_first(mixed_run, change)
    with pytest.raises(ValueError):
        audit_run(mixed_run)


def test_sequence_cannot_claim_opponent_as_learning_eligible(mixed_run):
    seq = next(iter_joint_sequences(mixed_run, context_steps=2, prediction_steps=4))
    bad = seq.learning_eligible.copy()
    bad[0, 1] = True
    with pytest.raises(ValueError, match="mixed behavior"):
        replace(seq, learning_eligible=bad)
    metadata = copy.deepcopy(seq.metadata)
    metadata["actor_policy_versions"][0][1] = "learner-v0"
    with pytest.raises(ValueError, match="versions"):
        replace(seq, metadata=metadata)


def test_frozen_identity_cannot_change_across_batches(mixed_run):
    path = sorted((mixed_run / "rollouts").glob("batch-*.npz"))[1]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(data["metadata"]))
    meta["behavior_policies"][1]["id"] = "other-opponent"
    data["metadata"] = json.dumps(meta)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="identities changed"):
        list(iter_joint_sequences(mixed_run, context_steps=2, prediction_steps=4))
    with pytest.raises(ValueError, match="identities changed"):
        audit_run(mixed_run)


def test_loaded_sequence_rejects_mid_episode_role_or_identity_change(mixed_run):
    seq = next(iter_joint_sequences(mixed_run, context_steps=2, prediction_steps=4))
    index = seq.behavior_policy_index.copy()
    index[1, :2] = index[1, :2][::-1]
    with pytest.raises(ValueError, match="mixed behavior"):
        replace(seq, behavior_policy_index=index)
    meta = copy.deepcopy(seq.metadata)
    meta["behavior_policies"][1][1]["id"] = "replacement"
    with pytest.raises(ValueError, match="identities changed"):
        replace(seq, metadata=meta)
