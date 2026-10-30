"""Exercise actual feedback provider, not pre-filled attribution flags."""
import json
from types import SimpleNamespace

import pytest

from driversi.agents.backends.native import digest
from driversi.data.catalog.store import identity
from driversi.rsi.plugins import chainflow


def fixture(tmp_path, monkeypatch, *, foreign_branch=False):
    def save(name, value):
        path=tmp_path/name;path.write_text(json.dumps(value));return str(path)
    assignment={"id":"a","token":"training-token","source_log":"training-log",
                "skill":"reactive_new_history","skill_version":1}
    plan={"kind":"capability_skill_plan","context":{"model":"chainflow_stage1"},"assignments":[assignment]}
    plan["plan_id"]=identity(plan)
    plan_path=save("skill.json",plan)
    lineage={"plan_id":"old-plan" if foreign_branch else plan["plan_id"],"assignment_id":"a",
             "skill":assignment["skill"],"skill_version":1}
    scene_path=save("scene.json",{"capability_skill":lineage})
    monkeypatch.setattr("driversi.contracts.scene.Scene.load",lambda p:SimpleNamespace(metadata=json.loads(p.read_text())))
    proof={"parent":{"source_log":"training-log","token":"training-token"},
           "branch":{"scene":scene_path},"files":{scene_path:digest(scene_path)}}
    admission=save("admission.json",proof)
    files={admission:digest(admission),scene_path:digest(scene_path)}
    row={"sample_id":"new","kind":"synthetic_navtrain","admission":{"path":admission,"sha256":digest(admission)}}
    dataset={"splits":{"train":[row]},"files":files}
    manifest=save("manifest.json",dataset)
    checkpoint=str(tmp_path/"weights.ckpt")
    execution_plan={"data":{"manifest":manifest},"source_groups":{},"files":files,
                    "model_config":{"weights":{"checkpoint":checkpoint}}}
    training_plan=save("training-plan.json",execution_plan)
    student=save("student.json",{})
    # This is the actual lifecycle output name; evaluator_context is NOT used.
    evaluation=save("evaluation.json",{
        "metric_scope":"official_navtest_v1_locked_subset_adaptive_pilot_not_full_benchmark",
        "evaluation_context":{"value":{"protocol":"chainflow_native_navsim_pdm_v1"}},
        "consumed":{"wall_seconds":2}})
    monkeypatch.setattr(chainflow,"validate_manifest",lambda p,**kw:json.loads(open(p).read()))
    monkeypatch.setattr(chainflow,"compare",lambda a,b:{"added_sample_ids":["new"]})
    monkeypatch.setattr(chainflow.lifecycle,"training_lineage",lambda p:{"receipts":[]})
    inputs={}
    for side in ("control","treatment"):
        inputs[side+"_student"]={"artifacts":{"plan":training_plan,"report":student}}
        inputs[side+"_evaluation"]={"artifacts":{"report":evaluation}}
    binding={"plan":plan_path,"plan_sha256":digest(plan_path),"plan_id":plan["plan_id"],
             "assignment_ids":["a"],"added_sample_ids":["new"],"parent_checkpoint":checkpoint,
             "evaluation_split":"official_navtest_locked_subset_pilot"}
    return {"skill_binding":binding},inputs


def test_real_feedback_entry_accepts_lifecycle_context_name(tmp_path,monkeypatch):
    config,inputs=fixture(tmp_path,monkeypatch)
    chainflow.run_feedback(config,inputs,tmp_path/"out",None)
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert report["capability_skill_binding"]["verified_evaluator_version"]=="native_navsim_v1_pdm_40x0.1_locked_subset"


def test_direct_provider_cannot_reattribute_old_branch(tmp_path,monkeypatch):
    config,inputs=fixture(tmp_path,monkeypatch,foreign_branch=True)
    # Caller knows the correct added IDs, checkpoint and protocol; still reject.
    with pytest.raises(ValueError,match="executed skill lineage"):
        chainflow.run_feedback(config,inputs,tmp_path/"out",None)


def test_direct_provider_cannot_replace_assignment_ids(tmp_path,monkeypatch):
    config,inputs=fixture(tmp_path,monkeypatch)
    config["skill_binding"]["assignment_ids"]=["invented"]
    with pytest.raises(ValueError,match="assignments differ"):
        chainflow.run_feedback(config,inputs,tmp_path/"out",None)
