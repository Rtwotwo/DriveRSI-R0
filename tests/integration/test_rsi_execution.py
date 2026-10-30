# ruff: noqa: E402
"""Actual subprocess supervision tests, optional Torch/native runtime required."""
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from driversi.rsi.execution.executor import execute, recover
from driversi.rsi.execution.jobs import JobStore, process_identity

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def execution_case(straight_scene, tmp_path):
    paths = []
    for name in ("training", "evaluation"):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = name + "-log"
        path = tmp_path / (name + ".npz")
        scene.save(path)
        paths.append({"path": str(path)})
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"official_test": False, "train": paths[:1], "evaluation": paths[1:]}))
    arm = {"version": "initial", "initial_model": "seed0", "information_set": "synthetic", "tools": "v1"}
    plan = {"schema_version": 1, "id": "p", "target": "model", "hypothesis": "TEST ONLY",
            "arms": {"baseline": arm, "candidate": {**arm, "version": "updated"}},
            "protocol": {"id": "e", "evaluator_version": "driversi.teacher.v1",
                         "role": "development", "task_set": "tasks", "metric": "goal_per_actor",
                         "direction": "maximize"},
            "data": {"development_groups": ["training-log"], "evaluation_groups": ["evaluation-log"]},
            "budget_per_arm": {"gpu_seconds": 0, "cpu_seconds": 240, "llm_tokens": 0,
                               "sim_agent_steps": 20000, "rendered_frames": 0, "trials": 3},
            "scene_ids": []}
    registry = tmp_path / "ledger.sqlite"
    store = JobStore(registry)
    store.registry.register(plan)
    job = {"schema_version": 1, "id": "j", "plan_id": "p", "arm": "candidate",
           "tool": "structured_sample", "suite": str(suite),
           "agent_config": str(ROOT / "tests/fixtures/legacy/agents/linear.yaml"),
           "target_transitions": 16, "max_train_agent_steps": 64, "worlds": 1,
           "rollout_steps": 4, "epochs": 1, "minibatch_size": 8, "wall_seconds": 90}
    return store, job, tmp_path


def run_case(case, **overrides):
    store, job, root = case
    spec = root / "job.json"
    spec.write_text(json.dumps({**job, **overrides}))
    return execute(root / "ledger.sqlite", spec, root / "runs")


def numeric_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files if data[k].dtype.kind in "biufc"}


def test_sample_and_portable_model_initialization(execution_case):
    first = run_case(execution_case)
    assert first["status"] == "completed", first["failure_reason"]
    directory = Path(first["directory"])
    a, b = numeric_arrays(directory / "initial.npz"), numeric_arrays(directory / "candidate.npz")
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert first["evaluation"]["source_separation_checked"]
    assert first["evaluation"]["independent_process"]
    assert first["progress"]["updates"] == 0
    # A separate initial-model reference intentionally carries the candidate weights forward.
    store, job, root = execution_case
    plan = store.plan("p")
    plan["id"] = "continuation"
    for arm in plan["arms"].values():
        arm["initial_model"] = "portable-parent"
    store.registry.register(plan)
    second = run_case(execution_case, id="continued", plan_id="continuation",
                      initial_weights=str(directory / "model-weights.pt"))
    assert second["status"] == "completed", second["failure_reason"]
    assert first["job"]["candidate_version"] != second["job"]["candidate_version"]
    initial = numeric_arrays(Path(second["directory"]) / "initial.npz")
    for k in b:
        np.testing.assert_array_equal(initial[k], b[k])


@pytest.mark.parametrize("kind", ["wall", "steps", "cpu", "gpu", "invalid_model"])
def test_failure_is_settled_without_candidate(execution_case, kind):
    store, job, root = execution_case
    if kind == "wall":
        change = {"wall_seconds": 0.05}
    elif kind == "steps":
        change = {"target_transitions": 1, "max_train_agent_steps": 1}
    elif kind in ("cpu", "gpu"):
        plan = store.plan("p")
        plan["id"] = "limited-" + kind
        plan["budget_per_arm"][kind + "_seconds"] = .01
        store.registry.register(plan)
        change = {"plan_id": plan["id"], "device": "cuda" if kind == "gpu" else "cpu"}
    else:
        import yaml
        config = yaml.safe_load(Path(job["agent_config"]).read_text())
        config["_target_"] = "driversi.agents.this_model_does_not_exist.Model"
        path = root / "bad.yaml"
        path.write_text(yaml.safe_dump(config))
        change = {"agent_config": str(path)}
    result = run_case(execution_case, **change)
    assert result["status"] == ("failed" if kind == "invalid_model" else "budget_exhausted")
    assert result["cost"]["trials"] == 1 and result["cost"]["cpu_seconds"] > 0
    assert result["evaluation"] is None
    if kind == "steps":
        assert result["cost"]["sim_agent_steps"] == 0
    attempts = [a for e in store.registry.public_summary()["experiments"] for a in e["attempts"]]
    assert len(attempts) == 1 and attempts[0]["candidate_model"] is None
    child = result.get("child")
    if child:
        assert process_identity(child["pid"]) != child["start"]


@pytest.mark.parametrize("interruption", ["cancel", "kill_supervisor"])
def test_interruption_leaves_no_tool_and_preserves_accounting(execution_case, interruption):
    store, job, root = execution_case
    job.update(target_transitions=10000, max_train_agent_steps=10000)
    spec = root / "job.json"
    spec.write_text(json.dumps(job))
    with (root / "cli.log").open("wb") as log:
        proc = subprocess.Popen([sys.executable, "-B", "-m", "driversi.rsi", "--registry",
                                 str(root / "ledger.sqlite"), "execute", str(spec),
                                 "--output-root", str(root / "runs")], stdout=log, stderr=log,
                                start_new_session=True)
        child = None
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    state = store.get("j")
                except (KeyError, ValueError):
                    state = {}
                child = state.get("child")
                if child and (root / "runs/j/progress.json").exists():
                    break
                if proc.poll() is not None:
                    pytest.fail((root / "cli.log").read_text())
                time.sleep(.05)
            assert child and (root / "runs/j/progress.json").exists(), "Tool did not start"
            if interruption == "cancel":
                store.cancel("j")
            else:
                os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            deadline = time.monotonic() + 5
            while process_identity(child["pid"]) == child["start"] and time.monotonic() < deadline:
                time.sleep(.05)
            assert process_identity(child["pid"]) != child["start"]
            if interruption == "kill_supervisor":
                assert store.get("j")["status"] == "running"
                result = recover(store, "j")
                assert result["cost_basis"] == "reserved_caps_unknown_usage"
                assert result["cost"]["sim_agent_steps"] == 20000
            else:
                result = store.get("j")
                assert result["status"] == "cancelled"
            assert result["cost"]["trials"] == 1
            attempts = store.registry.public_summary()["experiments"][0]["attempts"]
            assert len(attempts) == 1 and attempts[0]["candidate_model"] is None
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            if child and process_identity(child["pid"]) == child["start"]:
                os.killpg(child["pid"], signal.SIGKILL)
