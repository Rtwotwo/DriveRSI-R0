import copy

import pytest

from driversi.rsi.contracts import validate_plan
from driversi.rsi.execution.registry import ExperimentRegistry


@pytest.fixture
def plan():
    return {"schema_version": 1, "id": "test-plan", "target": "improver",
            "hypothesis": "TEST ONLY: equal-start meta comparison",
            "arms": {"baseline": {"version": "r0", "initial_model": "p0",
                                  "information_set": "history0", "tools": "tools0"},
                     "candidate": {"version": "r1", "initial_model": "p0",
                                   "information_set": "history0", "tools": "tools0"}},
            "protocol": {"id": "protocol0", "evaluator_version": "e0", "role": "development",
                         "task_set": "task0", "metric": "gain", "direction": "maximize"},
            "data": {"development_groups": ["log-a"], "evaluation_groups": ["log-b"]},
            "budget_per_arm": {"gpu_seconds": 10, "cpu_seconds": 100, "llm_tokens": 1000,
                               "sim_agent_steps": 100,
                               "rendered_frames": 5, "trials": 2}, "scene_ids": ["scene-1"]}


def attempt(status="completed", name="trial1"):
    return {"id": name, "plan_id": "test-plan", "arm": "candidate", "status": status,
            "cost": {"gpu_seconds": 12, "cpu_seconds": 5, "llm_tokens": 600,
                     "sim_agent_steps": 20, "rendered_frames": 0, "trials": 1},
            "metrics": {"gain": 0.1} if status == "completed" else {},
            "artifacts": {}, "candidate_model": "p1" if status == "completed" else None,
            "failure_reason": "test failure" if status != "completed" else None}


@pytest.mark.parametrize("field", ["initial_model", "information_set", "tools"])
def test_meta_comparison_requires_equal_resources(plan, field):
    plan["arms"]["candidate"][field] = "different"
    with pytest.raises(ValueError, match="equal"):
        validate_plan(plan)


def test_lineage_overlap(plan):
    plan["data"]["evaluation_groups"] = ["log-a"]
    with pytest.raises(ValueError, match="lineage"):
        validate_plan(plan)


def test_immutable_plan_failure_cost_and_no_promotion(tmp_path, plan):
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite")
    registry.register(plan)
    plan["hypothesis"] = "mutated outside registry"
    with pytest.raises(ValueError, match="already exists"):
        registry.register(plan)
    registry.record(attempt())
    registry.record(attempt("failed", "trial2"))
    with pytest.raises(ValueError, match="already recorded"):
        registry.record(attempt("failed", "trial2"))
    summary = registry.public_summary()["experiments"][0]
    assert summary["cost_by_arm"]["candidate"]["gpu_seconds"] == 24
    assert summary["over_budget"]["candidate"] == ["gpu_seconds", "llm_tokens"]
    assert summary["decision"] == "not_adjudicated"
    assert summary["plan"]["hypothesis"] != plan["hypothesis"]
    assert len(summary["attempts"]) == 2


@pytest.mark.parametrize("role", ["promotion", "meta_holdout"])
def test_protected_evidence_is_not_browsed(tmp_path, plan, role):
    plan["protocol"]["role"] = role
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite")
    registry.register(plan)
    registry.record(attempt())
    assert registry.public_summary()["experiments"] == []


def test_read_only_does_not_initialize_or_write(tmp_path, plan):
    p = tmp_path / "x.sqlite"
    with pytest.raises(ValueError, match="does not exist"):
        ExperimentRegistry(p, read_only=True)
    ExperimentRegistry(p).register(plan)
    readonly = ExperimentRegistry(p, read_only=True)
    with pytest.raises(ValueError, match="Read-only"):
        readonly.record(attempt())


@pytest.mark.parametrize("value", [-1, float("nan"), True])
def test_invalid_costs_rejected(tmp_path, plan, value):
    registry = ExperimentRegistry(tmp_path / "x.sqlite")
    registry.register(plan)
    row = attempt()
    row["cost"]["gpu_seconds"] = value
    with pytest.raises(ValueError):
        registry.record(row)


def test_missing_metric_and_unknown_plan_fail(tmp_path, plan):
    registry = ExperimentRegistry(tmp_path / "x.sqlite")
    row = attempt()
    with pytest.raises(ValueError, match="Unknown"):
        registry.record(row)
    registry.register(plan)
    row["metrics"] = {}
    with pytest.raises(ValueError, match="metric"):
        registry.record(row)
    changed = copy.deepcopy(plan)
    changed["id"] = "new-plan"
    changed["budget_per_arm"]["sim_agent_steps"] = 1.5
    with pytest.raises(ValueError, match="integer"):
        registry.register(changed)
