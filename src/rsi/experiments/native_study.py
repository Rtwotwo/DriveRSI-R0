"""Matched native-learning studies, executed by the existing DriveHarness.

This first study isolates original-anchor label improvement. It cannot establish
reactive driving or recursive improvement; those remain required RSI evidence.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

from driversi.agents.backends.native import digest, path_at, project_root, signature
from driversi.rsi.workflow import engine as harness
from driversi.rsi.plugins import augustus as harness_augustus
from driversi.runtime.budget import write_json
from driversi.rsi.contracts import _identity, _keys
from driversi.rsi.workflow.artifacts import verify
from driversi.rsi.workflow.store import TERMINAL, HarnessStore
from driversi_agents.augustus.data import manifest

ARMS = ("continued_training", "fixed_tto")
PROVIDERS = {
    "proposals": "drivetto_candidates",
    "dataset": "native_model_dataset",
    "train": "native_model_rsi_training",
    "evaluation": "native_model_rsi_evaluation",
    "feedback": "native_model_learning_feedback",
}
METRICS = ("PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC")


def prepare(spec, registry, output):
    spec = Path(spec).resolve()
    raw = json.loads(spec.read_text())
    _keys(raw, ("schema_version", "id", "template", "minimum_mean_pdms_gain",
                "evaluation_role"), "native study")
    _identity(raw["id"], "study id")
    gain = raw["minimum_mean_pdms_gain"]
    if (raw["schema_version"] != 1 or raw["evaluation_role"] != "development"
            or type(gain) not in (int, float) or not 0 < gain <= 1):
        raise ValueError("Explicit development-only study and positive raw PDMS gain required")
    template = (spec.parent / raw["template"]).resolve(strict=True)
    task = json.loads(template.read_text())
    if (len(task["steps"]) != len(PROVIDERS)
            or {x["id"]: x["provider"] for x in task["steps"]} != PROVIDERS):
        raise ValueError("Study requires the native AUGUstus five-stage template")
    for step in task["steps"]:
        for key in ("bank", "cases", "protected_logs", "manifest", "python", "model_config"):
            if key in step["config"]:
                step["config"][key] = str(path_at(template.parent, step["config"][key]))
    nodes = {x["id"]: x for x in task["steps"]}
    model = nodes["train"]["config"]["model_config"]
    if nodes["evaluation"]["config"]["model_config"] != model:
        raise ValueError("Training and evaluation must use the same model declaration")
    data_path = nodes["dataset"]["config"]["manifest"]
    data = manifest(data_path)
    cfg = yaml.safe_load(Path(model).read_text())
    if cfg["model"] != "augustus":
        raise ValueError("First study supports native AUGUstus only")
    output = Path(output).resolve()
    if not output.is_relative_to(project_root() / "outputs") or output.exists():
        raise ValueError("New study directory under project outputs required")
    prepared = {}
    for arm in ARMS:
        value = copy.deepcopy(task)
        value["id"] = raw["id"] + "-" + arm
        value["label"] = "Matched label study: " + arm
        for step in value["steps"]:
            if step["id"] == "dataset":
                step["config"]["label_source"] = "original" if arm == ARMS[0] else "drivetto"
        prepared[arm] = harness.prepare(value, template.parent, registry)
    store = HarnessStore(Path(registry))
    with store.registry._connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS native_studies (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        if db.execute("SELECT 1 FROM native_studies WHERE id=?", (raw["id"],)).fetchone():
            raise ValueError("Study is already frozen")
        if any(db.execute("SELECT 1 FROM harness_tasks WHERE id=?", (t["id"],)).fetchone()
               for t in prepared.values()):
            raise ValueError("Study task identity already registered")
    output.mkdir(parents=True)
    for arm, task in prepared.items():
        store.register(task, output / "tasks" / task["id"])
    result = {
        **raw, "template": str(template), "directory": str(output),
        "registry": str(Path(registry).resolve()), "tasks": prepared,
        "source_manifest": data_path, "source_manifest_sha256": digest(data_path),
        "model_config": model, "model_config_sha256": digest(model),
        "parent": str(path_at(project_root(), cfg["weights"]["parent"])),
        "initial_pool_records": {k: len(b["records"]) for k, b in data["splits"].items()},
        "controls": ["same parent", "same input pool", "same native recipe and seed",
                     "same per-arm training resource caps", "same evaluation source and code"],
        "scope": "original_anchor_offline_label_effect",
        "recursive_improvement_is_core_requirement": True,
        "automatic_activation": False,
        "cost_scope": {"comparison": "matched_training_budget_given_existing_labels",
            "historical_tto_generation_and_scoring": "excluded_unmeasured",
            "total_improvement_budget_matched": False},
    }
    with store.registry._connect() as db:
        db.execute("INSERT INTO native_studies VALUES (?,?)",
                   (result["id"], json.dumps(result, allow_nan=False)))
    write_json(output / "study.json", result)
    return result


def _requirements():
    return {
        "model_gain_over_equal_training_budget_continued_training": "not_evaluated",
        "model_gain_over_equal_total_improvement_budget": "not_evaluated",
        "new_policy_used_in_next_reactive_rollout": "missing_runtime_evidence",
        "improver_updated_and_used_on_new_tasks": "not_implemented_in_this_study",
        "improver_gain_from_matched_start_and_budget": "not_evaluated",
    }


def _read(path):
    return json.loads(Path(path).read_text())


def _stage(state, name):
    stage = next(s for s in state["stages"] if s["id"] == name)
    declaration = next(s for s in state["task"]["steps"] if s["id"] == name)
    verify(stage["result"], state["id"], declaration)
    return _read(stage["result"]["artifacts"]["report"])


def _recipe(plan):
    value = yaml.safe_load(plan["generated_files"]["native.yaml"])
    # These locations differ by arm; their data/parent content is checked separately.
    value["training"].pop("output_dir")
    value["data"].pop("train_data_path")
    stats = value["normalization"].pop("stats_file")
    return {"recipe": value, "normalization": digest(stats)}


def _execution(plan):
    root = project_root()
    code = {p: h for p, h in plan["files"].items()
            if any(Path(p).is_relative_to(root / d) for d in ("src", "agents"))}
    return {"recipe": _recipe(plan), "source": plan["source"], "runtime": plan["runtime"],
            "env": plan["env"], "code": code, "protocol": plan["protocol"]}


def _evidence_plan(path, expected):
    plan = _read(path)
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("plan_sha256") != expected or signature(unsigned) != expected:
        raise ValueError("Native plan identity differs from sealed execution evidence")
    return _execution(plan)


def _scores(report):
    return {r["frame_id"]: {k: r[k] for k in METRICS} for r in report["samples"]}


def _registered(study_path):
    study = _read(study_path)
    store = HarnessStore(Path(study["registry"]))
    with store.registry._connect() as db:
        row = db.execute("SELECT payload FROM native_studies WHERE id=?", (study["id"],)).fetchone()
    if not row or json.loads(row[0]) != study:
        raise ValueError("Frozen study protocol changed or not registered")
    return study, store


def collect(study_path):
    study, store = _registered(study_path)
    states = {arm: store.get(study["tasks"][arm]["id"]) for arm in ARMS}
    for arm, state in states.items():
        if state["task"] != study["tasks"][arm]:
            raise ValueError("Registered study task differs from frozen comparison")
    report = {
        "study_id": study["id"], "scope": study["scope"],
        "status": "incomplete", "claims": {"label_effect": False,
            "reactive_driving_improvement": False, "recursive_improvement": False},
        "core_requirements": _requirements(), "promoted": False,
        "arms": {k: {"task_id": s["id"], "status": s["status"],
                     "failure_reason": s.get("failure_reason"),
                     "wall_seconds": s.get("elapsed_seconds", 0),
                     "sim_agent_steps": s.get("sim_agent_steps", 0),
                     "stages": [{"id": x["id"], "status": x["status"],
                                 "reason": x.get("reason")} for x in s["stages"]]}
                 for k, s in states.items()},
        "cost_scope": study["cost_scope"],
        "parent_pretraining_disjointness": "declared_provenance_not_independently_verified",
        "next_round": {"policy_activated": False, "improver_activated": False,
                       "reason": "explicit qualification and runtime-use evidence required"},
    }
    if any(s["status"] != "completed" for s in states.values()):
        write_json(Path(study["directory"]) / "report.json", report)
        return report
    if (digest(study["model_config"]) != study["model_config_sha256"]
            or digest(study["source_manifest"]) != study["source_manifest_sha256"]):
        raise ValueError("Study parent/data declarations changed")
    records = manifest(study["source_manifest"])["splits"]["development"]["records"]
    data, students, parent_scores, candidate_scores, training_plans, eval_plans = {}, {}, {}, {}, {}, {}
    for arm, state in states.items():
        for name in PROVIDERS:
            _stage(state, name)
        data[arm], students[arm] = _stage(state, "dataset"), _stage(state, "train")
        card = students[arm]
        harness_augustus.verify_bundle(card)
        harness_augustus.verify_bundle(card["parent"])
        if card["parent"]["checkpoint"] != study["parent"] or card["updates"] <= 0:
            raise ValueError("Wrong starting model or zero optimizer updates")
        if data[arm]["source_manifest_sha256"] != study["source_manifest_sha256"]:
            raise ValueError("Training arms did not consume the same source pool")
        artifacts = {s["id"]: s["result"]["artifacts"] for s in state["stages"]}
        evaluation = _stage(state, "evaluation")
        training_plans[arm] = _evidence_plan(artifacts["train"]["plan"], card["native_plan_sha256"])
        if len(evaluation["protocols"]) != 2:
            raise ValueError("Both native evaluation protocols required")
        eval_plans[arm] = [_evidence_plan(artifacts["evaluation"][f"{x}_plan"], protocol["plan"])
                            for x, protocol in zip(("parent", "candidate"), evaluation["protocols"])]
        parent_scores[arm] = _read(artifacts["evaluation"]["parent_scores"])
        candidate_scores[arm] = _read(artifacts["evaluation"]["candidate_scores"])
        harness_augustus.compare(parent_scores[arm], candidate_scores[arm], records)
        if state["elapsed_seconds"] > state["task"]["budget"]["max_wall_seconds"]:
            raise ValueError("Completed study arm exceeded its wall budget")
    a, b = ARMS
    if (training_plans[a] != training_plans[b] or eval_plans[a] != eval_plans[b]
            or students[a]["parent"] != students[b]["parent"]
            or students[a]["updates"] != students[b]["updates"]
            or not data[a].get("source_cache_files")
            or data[a]["source_cache_files"] != data[b].get("source_cache_files")
            or data[a]["original_rows"] != data[b]["original_rows"]
            or data[a]["label_source"] != "original" or data[a]["replaced_rows"] != 0
            or data[b]["label_source"] != "drivetto" or data[b]["replaced_rows"] <= 0):
        raise ValueError("Unmatched starting point, source contents, recipe, updates or label intervention")
    if _scores(parent_scores[a]) != _scores(parent_scores[b]):
        raise ValueError("Reference inference differs across arms; parity must be established first")
    comparison = harness_augustus.compare(candidate_scores[a], candidate_scores[b], records)
    from_parent = harness_augustus.compare(parent_scores[b], candidate_scores[b], records)
    benefit = (comparison["safety_nonregression"] and from_parent["safety_nonregression"]
               and from_parent["mean_pdms_delta"] > 1e-8
               and comparison["mean_pdms_delta"] >= study["minimum_mean_pdms_gain"])
    report.update(status="completed", comparison=comparison, candidate_vs_parent=from_parent,
                  observed_label_benefit=benefit,
                  interpretation="paired_development_label_effect_requires_replication")
    report["claims"]["label_effect"] = benefit
    report["core_requirements"]["model_gain_over_equal_training_budget_continued_training"] = (
        "observed_development_gain" if benefit else "not_demonstrated")
    write_json(Path(study["directory"]) / "report.json", report)
    return report


def run(study_path):
    study, store = _registered(study_path)
    for arm in ARMS:
        state = store.get(study["tasks"][arm]["id"])
        if state["status"] == "registered":
            harness.run(Path(study["registry"]), state["id"])
        elif state["status"] not in TERMINAL:
            raise ValueError("Active/interrupted task requires existing Harness recovery")
    return collect(study_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("spec", type=Path)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("run", "report"):
        sub.add_parser(name).add_argument("study", type=Path)
    args = parser.parse_args()
    value = prepare(args.spec, args.registry, args.output) if args.command == "prepare" else (
        run(args.study) if args.command == "run" else collect(args.study))
    print(json.dumps(value, allow_nan=False))
    return 2 if value.get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
