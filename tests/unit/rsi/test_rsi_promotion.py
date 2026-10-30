import copy

import pytest

from driversi.rsi.decisions.improver import propose, validate_proposal
from driversi.rsi.decisions.promotion import decide, validate_policy


@pytest.fixture
def gate_case():
    policy = {
        "id": "TEST-ONLY",
        "metric": "return_sum_per_actor",
        "direction": "maximize",
        "min_gain": 0.1,
        "min_scenes": 1,
        "min_groups": 1,
        "min_actors": 2,
        "max_collision": 0.1,
        "max_offroad": 0.1,
        "collision_tolerance": 0.0,
        "offroad_tolerance": 0.0,
    }
    row = {
        "scene": "/synthetic.npz",
        "actors": 2,
        "steps": 20,
        "collision": 0,
        "offroad": 0,
        "goal": 1,
        "return_sum": 1.0,
        "travel_m": 20.0,
        "endpoint_progress_m": 10.0,
    }
    ref = {"episodes": [{**row, "return_sum": 0.0}], "actor_episodes": 2}
    pair = {
        "purpose": "promotion_pair",
        "independent_process": True,
        "source_separation_checked": True,
        "reference_model": "parent",
        "episodes": [row],
        "actor_episodes": 2,
        "reference": ref,
    }
    return policy, pair


def test_gain_and_safety_allow_research_promotion(gate_case):
    policy, pair = gate_case
    result = decide(policy, pair, ["/synthetic.npz"], ["log"], "parent")
    assert result["outcome"] == "promoted"
    assert result["approved_teacher"] is False


@pytest.mark.parametrize("failure", ["collision", "offroad", "utility", "support"])
def test_gain_cannot_compensate_failed_guards(gate_case, failure):
    policy, pair = gate_case
    if failure in ("collision", "offroad"):
        pair["episodes"][0][failure] = 1
        pair["episodes"][0]["return_sum"] = 100
    elif failure == "utility":
        pair["episodes"][0]["return_sum"] = 0
    else:
        policy["min_groups"] = 2
    result = decide(policy, pair, ["/synthetic.npz"], ["log"], "parent")
    assert result["outcome"] == ("inconclusive" if failure == "support" else "rejected")
    assert result["reasons"]


@pytest.mark.parametrize("mutation", ["identity", "coverage", "duplicate", "nan", "actors"])
def test_bad_pair_rejected(gate_case, mutation):
    policy, pair = gate_case
    if mutation == "identity":
        pair["reference_model"] = "other"
    if mutation == "coverage":
        pair["reference"]["episodes"][0]["scene"] = "/else.npz"
    if mutation == "duplicate":
        pair["episodes"] *= 2
    if mutation == "nan":
        pair["episodes"][0]["return_sum"] = float("nan")
    if mutation == "actors":
        pair["actor_episodes"] = 3
    with pytest.raises(ValueError):
        decide(policy, pair, ["/synthetic.npz"], ["log"], "parent")


def test_policy_is_explicit_and_no_zero_gain_promotion(gate_case):
    policy, _ = gate_case
    with pytest.raises(ValueError):
        validate_policy({**policy, "min_gain": 0})
    with pytest.raises(ValueError):
        validate_policy({**policy, "new_rule": True})


def test_improver_only_selects_from_development_evidence(gate_case):
    _, pair = gate_case
    rows = [dict(pair["episodes"][0], scene=f"/s{i}.npz", collision=int(i == 0)) for i in range(4)]
    evidence = {"purpose": "development_mining", "episodes": rows}
    before = copy.deepcopy(evidence)
    config = {
        "name": "hardcase_curriculum_v1",
        "max_scenes": 3,
        "replay_fraction": 0.34,
        "recipe": "short",
    }
    proposal = propose(evidence, config, {"short": {}}, 12)
    assert proposal["selected_paths"][0] == "/s0.npz"
    assert len(proposal["replay_paths"]) == 2
    assert evidence == before
    validate_proposal(proposal, [r["scene"] for r in rows], {"short": {}})
    with pytest.raises(ValueError):
        propose(pair, config, {}, 12)
    with pytest.raises(ValueError):
        validate_proposal({**proposal, "gate": {}}, [r["scene"] for r in rows], {"short": {}})
    with pytest.raises(ValueError):
        validate_proposal(
            {**proposal, "selected_paths": ["/gate.npz"]}, [r["scene"] for r in rows], {"short": {}}
        )


@pytest.mark.parametrize("boundary", ["promote", "cancel", "budget", "lineage"])
def test_atomic_incumbent_update(gate_case, tmp_path, monkeypatch, boundary):
    from driversi.rsi.experiments.campaign_store import CampaignStore
    from driversi.rsi.contracts import BUDGET_FIELDS

    policy, pair = gate_case
    store = CampaignStore(tmp_path / "ledger.sqlite")
    cap = dict.fromkeys(BUDGET_FIELDS, 100)
    spec = {
        "id": "TEST_ONLY_ATOMIC",
        "initial_model": "parent",
        "initial_weights": "/parent.pt",
        "initial_training_groups": ["train"],
        "train_groups": ["train"],
        "development_groups": ["dev"],
        "promotion_groups": [["gate"]],
        "promotion_slots": [["/synthetic.npz"]],
        "agent": {"name": "fixture"},
        "budget": cap,
        "wall_seconds": 100,
        "gate": policy,
    }
    store.register(spec, tmp_path / "runs")
    state = store.acquire(spec["id"])
    candidate = {"version": "candidate", "weights": "/training/model-weights.pt"}
    state["current"] = {
        "index": 0,
        "parent": dict(state["incumbent"]),
        "candidate": candidate,
        "training_job": "training",
        "gate_job": "gate",
    }
    jobs = {
        "training": {
            "status": "completed",
            "directory": "/training",
            "job": {
                "tool": "structured_ppo",
                "agent": spec["agent"],
                "initial_weights": "/parent.pt",
                "candidate_version": "candidate",
            },
        },
        "gate": {
            "status": "completed",
            "evaluation": pair,
            "job": {
                "tool": "structured_gate",
                "agent": spec["agent"],
                "initial_weights": candidate["weights"],
                "reference_weights": "/parent.pt",
            },
        },
    }
    if boundary == "budget":
        state["cost"]["cpu_seconds"] = 101
    if boundary == "lineage":
        jobs["gate"]["job"]["reference_weights"] = "/wrong-parent.pt"
    store.update(state)
    monkeypatch.setattr(store.jobs, "get", lambda name: jobs[name])
    if boundary == "cancel":
        store.cancel(spec["id"])
    if boundary == "promote":
        updated = store.close_round(spec["id"])
        assert updated["incumbent"] == candidate
        assert updated["history"][0]["decision"]["outcome"] == "promoted"
        assert store.get(spec["id"])["incumbent"] == candidate
        with pytest.raises(ValueError):
            store.close_round(spec["id"])
    else:
        with pytest.raises(InterruptedError if boundary == "cancel" else ValueError):
            store.close_round(spec["id"])
        assert store.get(spec["id"])["incumbent"]["version"] == "parent"
        assert store.get(spec["id"])["history"] == []


def test_time_and_route_support_endings_are_not_hard_failures(gate_case):
    _, pair = gate_case
    base = pair["episodes"][0]
    rows = [dict(base, scene="/time.npz", goal=0, endings={"time_budget": 2}),
            dict(base, scene="/route.npz", goal=0, endings={"route_support_end": 2}),
            dict(base, scene="/unknown.npz", goal=0)]
    config = {"name": "hardcase_curriculum_v1", "max_scenes": 3,
              "replay_fraction": 0.1, "recipe": "short"}
    value = propose({"purpose": "development_mining", "episodes": rows}, config, {"short": {}}, 0)
    assert value["hard_cases"] == [] and len(value["replay_paths"]) == 3
    assert value["improver_version"] == "hardcase_safety_evidence_v2"
