import copy
import json
import sqlite3

import pytest

from driversi.data.catalog.store import identity
from driversi.rsi.decisions.capability_skills import SkillMemory, make_plan, training_outcome, validate_plan


def candidates():
    rows = []
    for i, cap in enumerate(("direction_compliance", "interaction", "route_progress")):
        for j in range(3):
            rows.append({"token": f"train-{i}-{j}", "source_log": f"log-{i}-{j}",
                "source_group": f"group-{i}-{j}", "capability": cap,
                "shared_conditions": ["history_turning"], "ranks": {"visual": 1},
                "status": "condition_match_requires_diagnosis"})
    return {"snapshot_id": "snapshot", "candidates": rows,
        "capability_feedback": {"schema_version": 1, "benchmark": "navtest",
            "checkpoint": "/private/parent.ckpt", "evaluator_version": "native-v1",
            "slices": [{"capability": c, "count": 100, "failures": 5}
                       for c in ("direction_compliance", "interaction", "route_progress")]}}


def plan(memory):
    return make_plan(candidates(), model="chainflow_stage1", memory=memory,
                     budget={"max_tasks": 3, "max_wall_seconds_per_task": 100})


def feedback(p, *, gain=.1, unsafe=False):
    return {"kind": "chainflow_paired_feedback", "status": "completed",
        "effective_recipe_verified": True, "exposed_added_sample_ids": ["synthetic"],
        "unconsumed_added_sample_ids": [], "total_experiment_wall_seconds": 100,
        "treatment_minus_continued_training": {"score": gain},
        "regressions_against_control": [{"metric": "collision"}] if unsafe else [],
        "capability_skill_binding": {"plan_id": p["plan_id"], "assignment_ids": [p["assignments"][0]["id"]],
            "evaluation_split": "official_navtest_locked_subset_pilot", "model": "chainflow_stage1",
            "parent_checkpoint": p["context"]["parent_checkpoint"],
            "verified_benchmark": p["context"]["benchmark"],
            "verified_evaluator_version": p["context"]["evaluator_version"]}}


def test_real_memory_persistence_duplicate_and_next_round(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m)
    assert len({a["capability"] for a in p["assignments"]}) == 3
    assert all(a["skill"] == "reactive_new_history" for a in p["assignments"])
    evidence=training_outcome(p,feedback(p),"real-fixture-evidence")
    assert evidence["credited"] is True
    assert m.record(evidence)["recorded"] is True
    assert m.record(evidence)["recorded"] is False
    with pytest.raises(ValueError,match="Conflicting"):
        m.record({**evidence,"gain_vs_control":.2})
    m.close()
    m=SkillMemory(tmp_path/"m.sqlite");p2=plan(m)
    assert p2["assignments"][0]["skill"] == "bounded_perturbation_history"  # untried arm
    assert p2["assignments"][1]["skill"] == "reactive_new_history"
    assert p2["promoted"] is False
    c=candidates();c["capability_feedback"]["checkpoint"]="new-parent"
    p3=make_plan(c,model="chainflow_stage1",memory=m,budget=p["budget"])
    assert p3["assignments"][0]["skill"] == "reactive_new_history"
    assert p3["assignments"][0]["options"][0]["statistics"]["cross_parent_prior_trials"] == 1
    m.close()


@pytest.mark.parametrize("mutation,reason", [
    (lambda f:f.pop("capability_skill_binding"),"unlinked_intervention"),
    (lambda f:f["capability_skill_binding"].update(evaluation_split="navtrain_engineering_regression"),"evaluation_protocol_not_navtest"),
    (lambda f:f["capability_skill_binding"].update(parent_checkpoint="foreign"),"model_or_parent_mismatch"),
    (lambda f:f.update(unconsumed_added_sample_ids=["synthetic"]),"paired_intervention_not_verified"),
    (lambda f:f["capability_skill_binding"].update(assignment_ids=["assignment-0","assignment-1"]),"mixed_or_unknown_skill_attribution"),
])
def test_invalid_or_historical_feedback_never_earns_credit(tmp_path,mutation,reason):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);f=feedback(p);mutation(f)
    outcome=training_outcome(p,f,"e")
    assert not outcome["credited"] and reason in outcome["reasons"]
    m.record(outcome);assert plan(m)["assignments"][0]["skill"] == "reactive_new_history"
    m.close()


def test_unsafe_and_incomplete_cost_are_not_positive_efficiency(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);f=feedback(p,unsafe=True)
    f["total_experiment_wall_seconds"]=None
    m.record(training_outcome(p,f,"e"))
    stats=m.statistics(p["context"],"direction_compliance","reactive_new_history")
    assert stats["safety_failures"]==1 and stats["mean_gain_vs_control"]<0
    assert stats["net_gain_per_wall_hour"] is None and not stats["complete_cost_evidence"]
    m.close()


def test_tampered_plan_rejected(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);p["assignments"][0]["skill"]="made-up"
    with pytest.raises(ValueError,match="changed"):validate_plan(p)
    m.close()


def test_llm_payload_minimized_and_only_allowlisted_actions(tmp_path,monkeypatch):
    from driversi.rsi.decisions import llm_advisor
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);seen={}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return json.dumps({"choices":[{"message":{"content":json.dumps({"choices":[{"case_id":"assignment-0","skill":"reactive_new_history"}]})}}]}).encode()
    class Client:
        def open(self,request,timeout):seen["body"]=request.data.decode();return Response()
    monkeypatch.setenv("TEST_SKILL_KEY","test-key")
    monkeypatch.setattr(llm_advisor.urllib.request,"build_opener",lambda *a:Client())
    choice,receipt=llm_advisor.advise_skills({"endpoint":"https://example.com/v1/chat/completions", "model":"test",
        "api_key_env":"TEST_SKILL_KEY","timeout_seconds":1,"max_tokens":100},p["assignments"],p["capability_feedback"])
    assert choice["assignment-0"]=="reactive_new_history" and receipt["requests"]==1
    assert all(x not in seen["body"] for x in ["train-0-0","log-0-0","parent.ckpt","test-key"])
    m.close()


def test_execution_failure_records_cost_and_changes_next_token(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);a=p["assignments"][0]
    m.record({"evidence_id":"attempt-1","outcome_type":"execution","context":p["context"],
        "capability":a["capability"],"skill":a["skill"],"skill_version":a["skill_version"],
        "token":a["token"],"execution_status":"admission_rejected","cost_wall_seconds":12,
        "credited":False})
    p2=plan(m)
    assert a["token"] not in [x["token"] for x in p2["assignments"]]
    stats=m.statistics(p["context"],a["capability"],a["skill"])
    assert stats["execution_failures"]==1 and stats["execution_wall_seconds"]==12
    assert stats["credited_trials"]==0
    m.close()


def test_evaluator_identity_not_trusted_from_plan(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);f=feedback(p)
    f["capability_skill_binding"]["verified_benchmark"]="navhard"
    assert "executed_evaluator_mismatch" in training_outcome(p,f,"e")["reasons"]
    f=feedback(p);f["capability_skill_binding"]["verified_evaluator_version"]="different"
    assert not training_outcome(p,f,"e2")["credited"]
    m.close()


def test_platform_failure_does_not_penalize_teacher_or_block_retry(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);a=p["assignments"][0]
    m.record({"evidence_id":"broken-path","outcome_type":"execution","context":p["context"],
        "capability":a["capability"],"skill":a["skill"],"skill_version":a["skill_version"],
        "token":a["token"],"execution_status":"preparation_or_execution_failed", "failure_domain":"platform",
        "cost_wall_seconds":19, "credited":False})
    stats=m.statistics(p["context"],a["capability"],a["skill"])
    assert stats["platform_failures"]==1 and stats["execution_failures"]==0
    assert stats["execution_wall_seconds"]==19
    assert plan(m)["assignments"][0]["token"]==a["token"]
    assert plan(m)["assignments"][0]["skill"]==a["skill"]
    m.close()


def test_unsafe_cross_parent_experience_stays_negative(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m)
    m.record(training_outcome(p,feedback(p,gain=5,unsafe=True),"unsafe"))
    context={**p["context"],"parent_checkpoint":"new-parent"}
    stats=m.statistics(context,"direction_compliance","reactive_new_history")
    assert stats["cross_parent_prior_mean_gain"] < 0
    assert stats["cross_parent_prior_safety_failures"] == 1
    m.close()


def test_llm_rejects_length_finish_even_if_json_parseable():
    from driversi.rsi.decisions.llm_advisor import _proposal
    with pytest.raises(ValueError,match="incomplete"):
        _proposal({"choices":[{"finish_reason":"length","message":{"content":"{}"}}],"usage":{"total_tokens":4096}})
    with pytest.raises(ValueError,match="not complete JSON"):
        _proposal({"choices":[{"finish_reason":"stop","message":{"content":"SECRET incomplete"}}]})


def test_high_teacher_score_with_contacts_is_unresolved_world_evidence():
    from driversi.rsi.experiments.skill_execution import execution_evidence
    evidence=execution_evidence({"arrival_contacts":[{}],"history_changed":True,"restore_exact":True,
        "branches":{"teacher":{"object_contacts":[{}],"completed_steps":50,"failure":None,"score":{"score_pass":True}}}})
    assert evidence["failure_domain"]=="world_validity_pending"
    assert evidence["failed_checks"]==["arrival_contacts_absent","future_contacts_absent"]
    assert evidence["causal_attribution"]=="unresolved"


def test_execution_diagnosis_review_preserves_cost_and_original(tmp_path):
    m=SkillMemory(tmp_path/"m.sqlite");p=plan(m);a=p["assignments"][0]
    original={"evidence_id":"first","outcome_type":"execution","context":p["context"],
        "capability":a["capability"],"skill":a["skill"],"skill_version":a["skill_version"],
        "token":a["token"],"execution_status":"execution_or_admission_rejected",
        "cost_wall_seconds":12,"credited":False}
    m.record(original)
    reviewed={**original,"evidence_id":"reviewed","supersedes":"first","failure_domain":"world_validity_pending"}
    m.record(reviewed);assert not m.record(reviewed)["recorded"]
    stats=m.statistics(p["context"],a["capability"],a["skill"])
    assert stats["execution_wall_seconds"]==12 and stats["world_validity_pending"]==1
    assert stats["execution_failures"]==0 and a["token"] in m.attempted_tokens(p["context"])
    assert m.db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]==2
    with pytest.raises(ValueError,match="cannot rewrite"):
        m.record({**reviewed,"evidence_id":"bad","supersedes":"reviewed","cost_wall_seconds":1})
    m.close()
