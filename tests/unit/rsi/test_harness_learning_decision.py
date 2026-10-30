import json
from pathlib import Path
import pytest
import yaml
from driversi.rsi.plugins import learning as h
from driversi.rsi.workflow.providers import builtin_providers
from driversi.agents.backends.native import signature, digest
from driversi_agents.augustus.adapter import build
from tests.unit.rsi.test_learning_decisions import case,recipe,COSTS

ROOT=Path(__file__).resolve().parents[3]

def test_native_step_cap_keeps_schedule_model_and_loss(tmp_path):
    cfg=yaml.safe_load((ROOT/"src/config/agents/augustus/native.yaml").read_text())
    parent=tmp_path/"parent";parent.mkdir()
    raw=yaml.safe_load((ROOT/cfg["repository"]/"AUGUstus_SFT/configs/navsim_qwen3p5.yaml").read_text())
    (parent/"config.yaml").write_text(yaml.safe_dump(raw))
    paths={k:str(tmp_path/k) for k in cfg["paths"]};weights={"parent":str(parent),"backbone":str(tmp_path/"backbone")}
    args=(ROOT,ROOT/cfg["repository"],"/missing/python",weights,paths,ROOT/"outputs/fixture","train")
    a=build(cfg,*args)
    cfg["settings"]["max_optimizer_steps"]=100
    b=build(cfg,*args)
    original=yaml.safe_load(a["generated_files"]["native.yaml"])
    bounded=yaml.safe_load(b["generated_files"]["native.yaml"])
    assert bounded["training"].pop("max_steps")==100
    original["training"].pop("max_steps",None)
    assert bounded==original
    cfg["settings"]["max_optimizer_steps"]=True
    with pytest.raises(ValueError,match="optimizer-step"):build(cfg,*args)

def test_bound_evidence_rechecked_before_decision(tmp_path):
    evidence=tmp_path/"raw.json";evidence.write_text("{}")
    c=case();c["evidence"]=[str(evidence)]
    doc=tmp_path/"input.json";doc.write_text(json.dumps(dict(schema_version=1,cases=[c],recipes=[recipe()],memory=[],budget={k:100 for k in COSTS},protected_logs=[],max_failed_attempts=3)))
    cfg,_=h.prepare_decision({"evidence":str(doc)},tmp_path,tmp_path/"registry")
    result=h.run_decision(cfg,{},tmp_path/"out",tmp_path/"registry")
    assert result["summary"]["decisions"][0]["action"]=="train"
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="evidence changed"):h.run_decision(cfg,{},tmp_path/"second",tmp_path/"registry")

def test_unselected_train_does_not_start_native_runtime(tmp_path,monkeypatch):
    p=tmp_path/"decision.json";p.write_text(json.dumps({"decisions":[dict(case_id="c",action="probe",status="reserved",recipe="p")]}))
    monkeypatch.setattr(h.aug,"run_training",lambda *a:pytest.fail("should not train"))
    out=h.run_selected({"case_id":"c","recipe_id":"p"},{"decision":{"artifacts":{"report":str(p)}}},tmp_path/"out",tmp_path/"registry")
    assert out["status"]=="rejected" and not out["summary"]["training_started"]

def test_selected_native_dispatch_preserves_budget_and_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(h.aug,"bundle",lambda _: {"parent-file":"digest"})
    cfg=tmp_path/"model.yaml";cfg.write_text(yaml.safe_dump({"weights":{"parent":"weights/parent"},"settings":{"gpus":[0]}}))
    manifest=tmp_path/"dataset.json";manifest.write_text(json.dumps({"splits":{"train":{"records":[{"log_name":"train-log"}]}}}))
    decision=dict(execution={"mode":"supervised","model_config_sha256":digest(cfg),"max_optimizer_steps":2},case_id="c",action="train",status="reserved",recipe="sft",parent_version=signature({"parent-file":"digest"}),source_log="train-log",reserved_cost={"wall_seconds":10,"gpu_seconds":10,"score_queries":1})
    path=tmp_path/"decision.json";path.write_text(json.dumps({"decisions":[decision]}))
    inputs={"decision":{"artifacts":{"report":str(path)}},"dataset":{"artifacts":{"manifest":str(manifest)}}}
    config=dict(case_id="c",recipe_id="sft",model_config=str(cfg),max_wall_seconds=10,max_sim_agent_steps=80,max_optimizer_steps=2)
    def native(c,i,o,r):
        o.mkdir()
        assert c["max_optimizer_steps"]==2 and set(i)=={"dataset"}
        return {"summary":{},"status":"completed"}
    monkeypatch.setattr(h.aug,"run_training",native)
    assert h.run_selected(config,inputs,tmp_path/"out",tmp_path/"registry")["status"]=="completed"
    config["max_sim_agent_steps"]=81
    with pytest.raises(ValueError,match="reservation"):h.run_selected(config,inputs,tmp_path/"out",tmp_path/"registry")

def test_new_providers_use_existing_student_evaluation_chain():
    p=builtin_providers()
    assert p.get("native_model_selected_training").output==p.get("native_model_rsi_training").output
    assert dict(p.get("native_model_selected_training").inputs)["decision"]==p.get("learning_decision").output


def test_registered_decision_cannot_change_when_memory_finishes(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text("{}")
    document = {"cases": []}
    report = {"revision": "incomplete"}
    monkeypatch.setattr(h, "_document", lambda *a: (document, dict(report)))
    config, _ = h.prepare_decision({"evidence": str(path)}, tmp_path, tmp_path / "registry")
    report["revision"] = "completed"
    with pytest.raises(ValueError, match="context changed"):
        h.run_decision(config, {}, tmp_path / "out", tmp_path / "registry")
