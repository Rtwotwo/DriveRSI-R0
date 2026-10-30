"""Study-level fixtures test attribution gates, never stand in for model results."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from driversi.rsi.experiments import native_study as study
from driversi.rsi.workflow.artifacts import seal
from driversi.rsi.workflow.store import HarnessStore

ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def compiled(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "project_root", lambda: tmp_path)
    template = json.loads((ROOT / "src/config/rsi/examples/augustus-original-study.json").read_text())
    for node in template["steps"]:
        for key in ("bank", "cases", "protected_logs", "manifest", "python", "model_config"):
            if key in node["config"]:
                # Preserve lexical interpreter path, matching production.
                node["config"][key] = str(study.path_at(ROOT / "src/config/rsi/examples", node["config"][key]))
    model = yaml.safe_load((ROOT / "src/config/agents/augustus/native.yaml").read_text())
    model_path = write(tmp_path / "model.json", model)
    for node in template["steps"]:
        if "model_config" in node["config"]:
            node["config"]["model_config"] = str(model_path)
    template_path = write(tmp_path / "template.json", template)
    spec = write(tmp_path / "spec.json", dict(schema_version=1, id="TEST-ONLY",
        template=str(template_path), minimum_mean_pdms_gain=0.001, evaluation_role="development"))
    value = study.prepare(spec, tmp_path / "ledger.sqlite", tmp_path / "outputs/study")
    return value, tmp_path / "outputs/study/study.json"


def test_study_registers_same_parent_pool_budget_and_only_changes_labels(compiled):
    value, path = compiled
    a, b = (copy.deepcopy(value["tasks"][arm]) for arm in study.ARMS)
    for task in (a, b):
        task.pop("id")
        task.pop("label")
        dataset = next(x for x in task["steps"] if x["id"] == "dataset")
        assert dataset["config"].pop("label_source") in ("original", "drivetto")
    assert a == b
    report = study.collect(path)
    assert report["status"] == "incomplete" and not any(report["claims"].values())
    assert report["core_requirements"]["improver_gain_from_matched_start_and_budget"] == "not_evaluated"


def complete_fixture(value, *, regression=False, altered_recipe=False, altered_parent=False, altered_cache=False, native_cap=2):
    """Explicit fake stage outputs for collector validation; no model is trained."""
    store = HarnessStore(Path(value["registry"]))
    parent = Path(value["parent"])
    parent.mkdir(parents=True)
    (parent / "weights.fixture").write_text("parent")
    (parent / "norm.fixture").write_text("normalization")
    records = study.manifest(value["source_manifest"])["splits"]["development"]["records"]
    def scores(pdms, nc=1):
        return {"samples": [{"frame_id": r["frame_id"], "PDMS": pdms, "NC": nc,
                            **{k: 1.0 for k in study.METRICS if k not in ("PDMS", "NC")}}
                           for r in records]}
    for index, arm in enumerate(study.ARMS):
        state = store.claim(value["tasks"][arm]["id"])
        directory = Path(state["directory"])
        checkpoint = parent.parent / arm
        checkpoint.mkdir()
        (checkpoint / "weights.fixture").write_text(arm)
        recipe = {"training": {"output_dir": str(checkpoint), "seed": 1, "max_steps": native_cap},
                  "data": {"train_data_path": str(directory / "dataset")},
                  "normalization": {"stats_file": str(parent / "norm.fixture")}}
        if altered_recipe and index == 1:
            recipe["training"]["seed"] = 2
        plan = {"generated_files": {"native.yaml": yaml.safe_dump(recipe)},
                "source": {"fixture": True}, "runtime": {"fixture": True}, "env": {},
                "files": {}, "protocol": "fixture"}
        plan["plan_sha256"] = study.signature(plan)
        write(directory / "train/native/plan.json", plan)
        for role in ("parent", "candidate"):
            write(directory / f"evaluation/{role}/native/plan.json", plan)
            score = (0.75 if altered_parent and index == 1 else 0.7) if role == "parent" else (0.8 + index * 0.1)
            nc = 0.5 if regression and index == 1 and role == "candidate" else 1
            write(directory / f"evaluation/{role}/native/eval_detailed.json", scores(score, nc))
        card = {"checkpoint": str(checkpoint), "files": study.harness_augustus.bundle(checkpoint),
                "parent": {"checkpoint": str(parent), "files": study.harness_augustus.bundle(parent)},
                "updates": 2, "native_plan_sha256": plan["plan_sha256"], "reserved_gpu_seconds": 1,
                "training_sources": [r["log_name"] for r in study.manifest(value["source_manifest"])["splits"]["train"]["records"]]}
        data = {"source_manifest_sha256": value["source_manifest_sha256"], "original_rows": 32,
                "source_cache_files": {"source.arrow": "changed" if altered_cache and index == 1 else "same"},
                "label_source": "original" if index == 0 else "drivetto", "replaced_rows": index}
        reports = {"proposals": {"fixture": True}, "dataset": data, "train": card,
                   "evaluation": {"protocols": [{"plan": plan["plan_sha256"]}] * 2, "reserved_gpu_seconds": 1}, "feedback": {"fixture": True}}
        for declaration in state["task"]["steps"]:
            name = declaration["id"]
            folder = directory / name
            report_path = write(folder / "report.json", reports[name])
            result = {"status": "completed", "reason": None, "sim_agent_steps": 0,
                      "summary": reports[name], "admission": {}, "artifacts": {"report": str(report_path)}}
            if name == "dataset":
                result["artifacts"]["manifest"] = str(write(folder / "dataset.json", {}))
            if name == "train":
                result["artifacts"]["candidate"] = str(write(folder / "candidate.json", card))
                result["artifacts"]["plan"] = str(folder / "native/plan.json")
            if name == "evaluation":
                result["artifacts"].update({f"{arm}_{kind}": str(folder / arm / "native" / filename)
                    for arm in ("parent", "candidate")
                    for kind, filename in (("plan", "plan.json"), ("scores", "eval_detailed.json"))})
            result["summary"] = dict(result["summary"], artifact_reference=seal(
                result, folder, state["id"], declaration))
            stage = {**declaration, "status": "completed", "result": result}
            state["stages"].append(stage)
            store.save(state, "stage/end", {"id": name, "status": "completed"})
        state.update(status="completed", elapsed_seconds=1)
        store.save(state, "task/end", {"status": "completed"})


@pytest.mark.parametrize("regression", [False, True])
def test_label_benefit_does_not_imply_recursive_improvement(compiled, regression):
    value, path = compiled
    complete_fixture(value, regression=regression)
    report = study.collect(path)
    assert report["status"] == "completed"
    assert report["observed_label_benefit"] != regression
    assert not report["claims"]["recursive_improvement"]
    assert not report["next_round"]["policy_activated"]
    assert not report["promoted"]
    assert not report["cost_scope"]["total_improvement_budget_matched"]
    assert report["core_requirements"]["model_gain_over_equal_total_improvement_budget"] == "not_evaluated"


@pytest.mark.parametrize("mismatch", ["recipe", "reference_inference"])
def test_unmatched_comparison_rejected(compiled, mismatch):
    value, path = compiled
    complete_fixture(value, altered_recipe=mismatch == "recipe",
                     altered_parent=mismatch == "reference_inference")
    with pytest.raises(ValueError, match="Unmatched|Reference inference"):
        study.collect(path)


def test_changed_model_declaration_rejected(compiled):
    value, path = compiled
    complete_fixture(value)
    model = Path(value["model_config"])
    model.write_text(model.read_text() + " ")
    with pytest.raises(ValueError, match="declarations changed"):
        study.collect(path)


def test_study_gate_cannot_change_after_registration(compiled):
    value, path = compiled
    value["minimum_mean_pdms_gain"] = 0.5
    write(path, value)
    with pytest.raises(ValueError, match="Frozen study protocol"):
        study.collect(path)


@pytest.mark.parametrize("evidence", ["evaluation/candidate/native/eval_detailed.json", "train/native/plan.json"])
def test_native_evidence_tampering_invalidates_study(compiled, evidence):
    value, path = compiled
    complete_fixture(value)
    state = HarnessStore(Path(value["registry"])).get(value["tasks"][study.ARMS[1]]["id"])
    p = Path(state["directory"]) / evidence
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="Stage artifact changed"):
        study.collect(path)


def test_same_manifest_but_different_source_cache_rejected(compiled):
    value, path = compiled
    complete_fixture(value, altered_cache=True)
    with pytest.raises(ValueError, match="Unmatched"):
        study.collect(path)


@pytest.mark.parametrize("regression",[False,True])
def test_measured_study_becomes_memory_without_promoting(compiled,regression):
    from driversi.rsi.plugins.learning import memory_from_study
    from driversi.rsi.decisions.learning_decisions import validate_memory
    value,path=compiled
    context={"case_id":"failure-1","node":"planning","recipe":"bounded-sft","failure_class":"blocked"}
    pending=memory_from_study(path,context)
    assert pending["record"] is None and not pending["positive_credit"]
    complete_fixture(value,regression=regression)
    memory=memory_from_study(path,context)
    validate_memory([memory["record"]])
    assert memory["positive_credit"] is (not regression) and not memory["promoted"]
    assert memory["record"]["execution"]["max_optimizer_steps"]==2
    assert memory["record"]["evaluation_role"]=="development"
    assert set(memory["record"]["source_logs"]).isdisjoint(memory["record"]["training_source_logs"])
    assert "not_total_historical" in memory["budget_scope"]


@pytest.mark.parametrize("cap", [100, None])
def test_memory_declared_cap_is_not_observed_steps(compiled, cap):
    from driversi.rsi.plugins.learning import memory_from_study
    value, path = compiled
    complete_fixture(value, native_cap=cap)
    context = {"case_id": "c", "node": "planning", "recipe": "sft", "failure_class": "blocked"}
    memory = memory_from_study(path, context)
    assert memory["observed_native_global_steps"] == 2
    execution = memory["record"]["execution"]
    if cap is None:
        assert execution is None
    else:
        assert execution["max_optimizer_steps"] == 100
