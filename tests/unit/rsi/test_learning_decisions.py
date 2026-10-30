from copy import deepcopy
import pytest
from driversi.rsi.decisions.learning_decisions import allocate, route, CHECKS, COSTS

def case(**kw):
    c=dict(id="a",node="planning",failure_class="blocked",source_log="train-log",split="train",
           parent_version="p1",evaluator_version="official-v1",input_kind="original",
           checks={x:"pass" for x in CHECKS},evidence=["evidence.json"])
    c["checks"]["task"]="fail";c.update(kw);return c

EXECUTION = dict(mode="supervised",model_config_sha256="a"*64,max_optimizer_steps=2)
def recipe(name="sft",action="train",cost=10):return dict(id=name,action=action,node="planning",cost={k:cost for k in COSTS},execution=EXECUTION if action=="train" else None)
def mem(**kw):
    m=dict(id="m1",case_id="other",node="planning",recipe="sft",parent_version="p1",evaluator_version="official-v1",source_logs=["other-log"],training_source_logs=["fit-log"],evaluation_role="development",failure_class="blocked",parent_improved=True,outcome="completed",paired=True,matched_budget=True,safety_passed=True,delta=.1,cost={k:10 for k in COSTS},execution=EXECUTION);m.update(kw);return m

def test_unknown_and_platform_failure_never_route_to_training():
    c=case();c["checks"]["world_validity"]="unknown";assert route(c,[])[0]=="probe"
    c["checks"]["world_validity"]="fail";assert route(c,[])[0]=="repair_world"
    c["checks"]["runtime_parity"]="fail";assert route(c,[])[0]=="repair_adapter"

@pytest.mark.parametrize("split",["test","development"])
def test_holdout_never_training(split):assert route(case(split=split),[])[0]=="evaluation_only"

def test_original_label_and_new_state_are_different_operations():
    c=case();c["checks"]["trajectory_admission"]="unknown";assert route(c,[])[0]=="relabel"
    c["input_kind"]="counterfactual";assert route(c,[])[0]=="recover"
    c["checks"]["trajectory_admission"]="pass";c["checks"]["history_validity"]="unknown";assert route(c,[])[0]=="render_history"

def test_exact_multiresource_reservation_without_mutating_budget():
    budget={k:15 for k in COSTS};out=allocate([case(),case(id="b")],[recipe()],[],budget,protected_logs=[])
    assert out["decisions"][0]["recipe"]=="sft" and out["decisions"][1]["recipe"] is None
    assert budget["gpu_seconds"]==15 and out["remaining_budget"]["gpu_seconds"]==5

@pytest.mark.parametrize("change",[{"parent_version":"old"},{"source_logs":["train-log"]},{"source_logs":["test-log"]},{"matched_budget":False},{"paired":False},{"evaluator_version":"other"}])
def test_inapplicable_learning_memory_cannot_rank_candidate(change):
    out=allocate([case()],[recipe()], [mem(**change)],{k:20 for k in COSTS},protected_logs=["test-log"])
    assert out["decisions"][0]["learning_evidence"]==[]

def test_safety_regression_does_not_earn_positive_credit():
    out=allocate([case()],[recipe(),recipe("alternative")],[mem(safety_passed=False,delta=10.)],{k:20 for k in COSTS},protected_logs=[])
    assert out["decisions"][0]["recipe"]=="alternative"

def test_repeated_failures_backoff_but_do_not_block_alternative():
    memory=[mem(id=str(i),case_id="a",outcome="failed",delta=None) for i in range(3)]
    out=allocate([case()],[recipe(),recipe("alternative")],memory,{k:20 for k in COSTS},protected_logs=[])
    assert out["decisions"][0]["recipe"]=="alternative"
    assert out["decisions"][0]["excluded"][0]["reason"]=="unsuccessful_attempt_cap"

def test_duplicate_and_nonfinite_evidence_rejected():
    with pytest.raises(ValueError,match="Duplicate memory"):allocate([case()],[recipe()],[mem(),mem()],{k:20 for k in COSTS},protected_logs=[])
    with pytest.raises(ValueError,match="finite resource"):allocate([case()],[recipe()],[],{k:float("nan") for k in COSTS},protected_logs=[])

@pytest.mark.parametrize("change",[{"evaluation_role":"train"},{"training_source_logs":["other-log"]},{"failure_class":"unrelated"},{"training_source_logs":[]}])
def test_nonindependent_or_unrelated_feedback_cannot_rank(change):
    out=allocate([case()],[recipe()],[mem(**change)],{k:20 for k in COSTS},protected_logs=[])
    assert not out["decisions"][0]["learning_evidence"]


def test_other_training_cases_are_also_excluded_from_learning_evidence():
    c=case(id="b",source_log="other-log")
    out=allocate([case(),c],[recipe()],[mem()],{k:40 for k in COSTS},protected_logs=[])
    assert all(not x["learning_evidence"] for x in out["decisions"])


def test_same_recipe_name_with_changed_native_configuration_gets_no_credit():
    out=allocate([case()],[recipe()],[mem(execution={**EXECUTION,"max_optimizer_steps":3})],{k:20 for k in COSTS},protected_logs=[])
    assert not out["decisions"][0]["learning_evidence"]
