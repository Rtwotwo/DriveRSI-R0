# ruff: noqa: E402
"""End-to-end mixed-policy training, continuation and physical cost accounting."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")

from tests.support.diagnostic_config import normalize_agent
from driversi.cli import main
from driversi.training.sampling.audit import audit_run

ROOT = Path(__file__).resolve().parents[2]


def _fixture(scene, root):
    paths = {}
    for name in ("training-source", "evaluation-source"):
        item = copy.deepcopy(scene)
        item.metadata["source_log"] = name
        path = root / (name + ".npz")
        item.save(path)
        paths[name] = path
    suite = root / "suite.json"
    suite.write_text(json.dumps({"train": [{"path": str(paths["training-source"])}],
                                 "evaluation": [{"path": str(paths["evaluation-source"])}],
                                 "limits": ["Synthetic native integration fixture"]}))
    agent = normalize_agent()
    agent["parameters"] = {"hidden_sizes": []}
    rows = []
    for name, action in (("history-coast", 7), ("history-control1", 1)):
        path = root / (name + ".npz")
        meta = {k: agent[k] for k in ("observation_version", "action_space", "input_mode")}
        meta.update(policy_version=name,
                    architecture={"version": 1, "hidden_sizes": [], "activation": "tanh"})
        bias = np.full(12, -60., np.float32)
        bias[action] = 60.
        np.savez(path, metadata=json.dumps(meta), **{
            "actor.weight": np.zeros((12, 1125), np.float32), "actor.bias": bias,
            "critic.weight": np.zeros((1, 1125), np.float32), "critic.bias": np.zeros(1, np.float32)})
        rows.append({"id": name, "checkpoint": str(path), "agent_config": agent,
                     "training_paths": [str(paths["training-source"])]})
    pool = root / "pool.json"
    pool.write_text(json.dumps({"schema_version": 1, "policies": rows}))
    config = root / "run.json"
    config.write_text(json.dumps({"schema_version": 1, "trainer": "ppo",
                                  "agent": str(ROOT / "tests/fixtures/legacy/agents/linear.yaml"),
                                  "optimization": {"epochs": 1, "minibatch_size": 8,
                                                   "learning_rate": .001}}))
    return ["legacy-train", "--config", str(config), "--suite", str(suite),
            "--opponent-pool", str(pool), "--learner-count", "1",
            "--rollout-steps", "4", "--seed", "47", "--device", "cpu"]


def _npz_equal(a, b):
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        assert set(left.files) == set(right.files)
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def _metrics(path):
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("agent_name", ["linear", "mlp"])
def test_mixed_cli_updates_resume_and_metrics_recording_are_consistent(straight_scene, tmp_path, agent_name):
    args = _fixture(straight_scene, tmp_path) + ["--agent", agent_name, "--worlds", "2"]
    full, partial, resumed, metrics = [tmp_path / name for name in ("full", "partial", "resumed", "metrics")]
    for output, budget, extra in ((full, "32", []), (partial, "16", []),
                                  (resumed, "32", ["--resume", str(partial / "checkpoint.pt")]),
                                  (metrics, "32", ["--recording", "metrics"])):
        assert main([*args, "--output-dir", str(output), "--budget", budget, *extra]) == 0
    _npz_equal(full / "candidate.npz", resumed / "candidate.npz")
    _npz_equal(full / "candidate.npz", metrics / "candidate.npz")
    _npz_equal(full / "rollouts/batch-00000.npz", partial / "rollouts/batch-00000.npz")
    _npz_equal(full / "rollouts/batch-00001.npz", resumed / "rollouts/batch-00001.npz")
    with np.load(full / "initial.npz") as initial, np.load(full / "candidate.npz") as final:
        assert any(not np.array_equal(initial[k], final[k]) for k in initial.files if k != "metadata")
    report = json.loads((full / "report.json").read_text())
    assert report["valid_transitions"] == 32 and report["updates"] == report["batches"] == 2
    assert len(report["matchups"]["matchups"]) == len(report["baseline"]["matchups"]["matchups"]) == 2
    assert report["matchups"]["source_separation_checked"]
    rows = _metrics(full)
    assert sum(row["batch_valid"] for row in rows) == 32
    assert sum(row["batch_learner"] for row in rows) == 16
    assert sum(row["sampling"]["opponent_transitions"] for row in rows) == 16
    assert all(row["optimizer_updated"] for row in rows)
    for output, physical, learner in ((full, 32, 16), (partial, 16, 8), (resumed, 16, 8)):
        audit = audit_run(output)
        assert audit["frozen_policy_logprob_reproduced"] and audit["gae_reproduced"]
        assert audit["frozen_opponent_identities_verified"]
        assert audit["valid_transitions_in_this_run"] == physical
        assert audit["learner_transitions_in_this_run"] == learner
        assert audit["opponent_transitions_in_this_run"] == physical - learner
    assert not list((metrics / "rollouts").glob("*.npz"))
    assert (metrics / "checkpoint.pt").is_file()
    with pytest.raises(ValueError, match="No rollout batches"):
        audit_run(metrics)
    # Checkpoints carry the exact counter, optimizer and RNG continuation; no opponent is updated.
    uninterrupted = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=False)
    continuation = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert uninterrupted["optimizer_updates"] == continuation["optimizer_updates"] == 2
    assert uninterrupted["collector"]["opponents"] == continuation["collector"]["opponents"]
    for parameter, state in uninterrupted["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, continuation["optimizer"]["state"][parameter][key], rtol=0, atol=0)
    assert torch.equal(uninterrupted["torch_rng"], continuation["torch_rng"])


def test_terminated_learner_does_not_stop_opponent_cost_or_force_invalid_update(straight_scene, tmp_path):
    scene = copy.deepcopy(straight_scene)
    # Valid at reset (>2m away), but the first 0.1s advance reaches a's endpoint.
    # b remains physically active for both collected batches without a learner.
    scene.metadata["task_goals_xy"] = {"a": [2.05, 0.], "b": [50., 8.]}
    args = _fixture(scene, tmp_path) + ["--agent", "linear", "--worlds", "1"]
    out = tmp_path / "early-end"
    assert main([*args, "--output-dir", str(out), "--budget", "9"]) == 0
    report = json.loads((out / "report.json").read_text())
    rows = _metrics(out)
    assert report["updates"] == 0 and report["batches"] == 2 and report["valid_transitions"] == 9
    assert [row["batch_valid"] for row in rows] == [5, 4]
    assert [row["batch_learner"] for row in rows] == [1, 0]
    assert all(row["update_skipped"] == "fewer_than_two_learner_transitions" for row in rows)
    assert rows[1]["learner_mean_reward"] is None
    with np.load(out / "initial.npz") as before, np.load(out / "candidate.npz") as after:
        for key in before.files:
            if key != "metadata":
                np.testing.assert_array_equal(before[key], after[key])
    audit = audit_run(out)
    assert audit["learner_transitions_in_this_run"] == 1
    assert audit["opponent_transitions_in_this_run"] == 8
    assert audit["gae_reproduced"]
