import copy
import json

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.correction.curriculum import SAFETY, admit, select_anchor, validate_curriculum
from driversi.rsi.workflow.providers import builtin_providers


def evidence():
    return dict(same_anchor=True,complete=True,valid=True,teacher_no_fallback=True,contact_states=0), dict(status="scored",official_benchmark=False,score=.97,metric_scores=dict.fromkeys(SAFETY,1.0))


def test_proximity_selection_cannot_use_dead_anchor():
    rows=[dict(tick=20,eligible=True,clearance_m=9,closing_ttc_s=None),
          dict(tick=25,eligible=False,clearance_m=0,closing_ttc_s=0),
          dict(tick=30,eligible=True,clearance_m=15,closing_ttc_s=1.5)]
    assert select_anchor(rows)["tick"]==30
    with pytest.raises(ValueError):select_anchor([rows[1]])


@pytest.mark.parametrize("field,value",[("same_anchor",False),("complete",False),("valid",False),("teacher_no_fallback",False),("contact_states",1)])
def test_high_score_does_not_override_execution_failure(field,value):
    execution,score=evidence();execution[field]=value
    assert not admit(execution,score)["accepted"]


@pytest.mark.parametrize("metric",SAFETY)
def test_each_safety_metric_is_required(metric):
    execution,score=evidence();score["metric_scores"].pop(metric)
    assert not admit(execution,score)["accepted"]
    score["metric_scores"][metric]=.5
    assert not admit(execution,score)["accepted"]


@pytest.mark.parametrize("value",[None,float("nan"),float("inf"),.94999])
def test_bad_score_rejected(value):
    execution,score=evidence();score["score"]=value
    assert not admit(execution,score)["accepted"]


def test_curriculum_is_not_a_visual_or_onpolicy_teacher_batch():
    gate=admit(*evidence())
    assert gate["accepted"] and not gate["visual_trainable"] and not gate["teacher_actions_are_on_policy"]
    assert not gate["promoted"] and not gate["demonstrated_correction_benefit"]


def test_all_stages_are_typed():
    ps=builtin_providers()
    select=ps.get("automatic_recovery_selection")
    correct=ps.get("same_anchor_counterfactual_correction")
    dataset=ps.get("online_recovery_curriculum")
    train=ps.get("curriculum_training")
    assert dict(correct.inputs)["selection"]==select.output
    assert dict(dataset.inputs)["correction"]==correct.output
    assert dict(train.inputs)["dataset"]==dataset.output
    assert dict(ps.get("curriculum_resume").inputs)["student"]==train.output


def test_curriculum_revalidates_source_parent_and_teacher(tmp_path):
    scene=tmp_path/"scene";scene.write_bytes(b"scene")
    weight=tmp_path/"weight";weight.write_bytes(b"parent")
    snap=tmp_path/"snapshot";snap.write_bytes(b"native-snapshot")
    correction=tmp_path/"correction.json"
    execution,score=evidence();correction.write_text(json.dumps(dict(execution=execution,score=score)))
    manifest=dict(schema_version=1,purpose="feasible_online_reset_curriculum",admission=admit(execution,score),
        source_log="source",scene_sha256=digest(scene),parent_checkpoint_sha256=digest(weight),
        files={str(correction):digest(correction),str(snap):digest(snap)},correction_report=str(correction),
        starts={"treatment":dict(snapshot=str(snap),tick=30)})
    path=tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    args=dict(scene=scene,source_log="source",checkpoint=weight,arm="treatment")
    assert validate_curriculum(path,**args)[1]["tick"]==30
    with pytest.raises(ValueError,match="source"):
        validate_curriculum(path,**{**args,"source_log":"heldout"})
    weight.write_bytes(b"other")
    with pytest.raises(ValueError,match="parent"):validate_curriculum(path,**args)
    weight.write_bytes(b"parent");snap.write_bytes(b"mutated")
    with pytest.raises(ValueError,match="changed"):validate_curriculum(path,**args)


def test_feedback_refuses_reversed_arms_and_different_dev_scene():
    from driversi.rsi.plugins.recovery import validate_arms
    def arm(name):
        return {"plan":{"data":{"train":{"curriculum_arm":name},"development":{"scene":"heldout-a"}}}}
    a,b=arm("control"),arm("treatment")
    validate_arms(a,b)
    with pytest.raises(ValueError,match="input port"):validate_arms(b,a)
    b["plan"]["data"]["development"]["scene"]="heldout-b"
    with pytest.raises(ValueError,match="development"):validate_arms(a,b)


def test_evaluation_cost_is_not_silently_zero(tmp_path):
    from driversi.rsi.plugins.recovery import _learning_result
    value=_learning_result(tmp_path,dict(status="completed",consumed={"environment_steps":40}))
    assert value["sim_agent_steps"]==40
    value=_learning_result(tmp_path,dict(status="completed",consumed={"environment_steps":40,"sim_agent_steps":1800}))
    assert value["sim_agent_steps"]==1800


@pytest.mark.parametrize("status", ["no_anchor", "control_anchor_unavailable"])
def test_known_selection_rejection_is_published_not_unknown(tmp_path, status):
    from driversi.rsi.plugins.recovery import selection_result
    report = dict(status=status, reason="no_eligible_reached_state", physics_steps=80,
                  all_world_slots=144, selection=None)
    (tmp_path / "selection.json").write_text(json.dumps(report))
    (tmp_path / "worker-plan.json").write_text("{}")
    value = selection_result(tmp_path, report)
    assert value["status"] == "rejected"
    assert value["sim_agent_steps"] == 11520
    assert "anchor" not in value["artifacts"]
    builtin_providers().get("automatic_recovery_selection").validate_output(value)


def test_failure_gallery_keeps_null_selection_receipts_and_blocked_stages(tmp_path):
    import runpy
    from driversi.agents.backends.native import project_root
    module = runpy.run_path(str(project_root() / "scripts/legacy/reports/export_recovery_curriculum.py"))
    task = tmp_path / "tasks" / "failed-task"
    (task / "select/scout").mkdir(parents=True)
    (task / "select/scout/viewer.html").write_text("retained failure")
    (task / "select/selection.json").write_text(json.dumps(dict(status="no_anchor", selection=None)))
    (task / "select.receipt.json").write_text(json.dumps(dict(phase="failed", reason="no anchor")))
    (task / "report.json").write_text(json.dumps(dict(stages=[dict(id="correct", status="blocked", reason="dependency_unavailable")])))
    report = module["export"](tmp_path, "failed-task")
    rows = {x["stage"]: x for x in report["stages"]}
    assert rows["select"]["status"] == "failed"
    assert rows["correct"]["status"] == "blocked"
    page = (tmp_path / "index.html").read_text()
    assert "select/scout/viewer.html" in page and "correct/student/viewer.html" not in page
