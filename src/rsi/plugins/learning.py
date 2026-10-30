"""Bounded decision and selected native learning stages in DriveHarness."""
import json
import math
from pathlib import Path
import yaml

from driversi.agents.backends.native import digest, project_root, path_at, signature
from driversi.configuration import read_mapping
from driversi.rsi.plugins import augustus as aug
from driversi.rsi.contracts import _keys
from driversi.rsi.decisions.learning_decisions import allocate
from driversi.training.sampling.environment import source_log_group
from driversi.training.contracts.protocols import require_native_mode


def _document(path, registry):
    from driversi.rsi.execution.registry import ExperimentRegistry
    value = {"studies": [], **read_mapping(path)}
    _keys(value, ("schema_version", "cases", "recipes", "memory", "budget", "protected_logs", "max_failed_attempts", "studies"), "learning decision input")
    if value["schema_version"] != 1:
        raise ValueError("Unknown decision input schema")
    protected = list(value["protected_logs"]) + list(ExperimentRegistry(registry).protected_groups())
    official = project_root() / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    protected += yaml.safe_load(official.read_text())["log_names"]
    imported = []
    for item in value["studies"]:
        _keys(item, ("path", "case_id", "node", "recipe", "failure_class"), "study memory source")
        imported.append(memory_from_study((Path(path).parent / item["path"]).resolve(), item))
    records = value["memory"] + [x["record"] for x in imported if x["record"] is not None]
    report = allocate(value["cases"], value["recipes"], records, value["budget"],
                      protected_logs=protected, max_failed_attempts=value["max_failed_attempts"])
    report["imported_studies"] = imported
    return value, report


def prepare_decision(config, base, registry):
    _keys(config, ("evidence",), "decision provider")
    path = (base / config["evidence"]).resolve(strict=True)
    value, report = _document(path, registry)
    sources = {}
    for case in value["cases"]:
        for ref in case["evidence"]:
            file = (path.parent / ref).resolve(strict=True)
            sources[str(file)] = digest(file)
    return {"evidence": str(path), "sha256": digest(path), "files": sources, "decision_sha256": signature(report),
            "source_groups": sorted({source_log_group(x["source_log"]) for x in value["cases"]})}, 0


def run_decision(config, inputs, output, registry):
    if digest(config["evidence"]) != config["sha256"] or any(digest(p) != sha for p, sha in config["files"].items()):
        raise ValueError("Diagnostic evidence changed after task registration")
    _, report = _document(Path(config["evidence"]), registry)
    if signature(report) != config["decision_sha256"]:
        raise ValueError("Decision context changed after task registration")
    report["evidence_scope"] = "declared_checks_with_bound_source_files_not_automatic_causal_inference"
    output.mkdir(parents=True, exist_ok=False)
    return aug.result(output, report)


def prepare_selected(config, base, registry):
    _keys(config, ("model_config", "max_wall_seconds", "max_sim_agent_steps", "max_optimizer_steps", "case_id", "recipe_id"), "selected native learner")
    for name in ("case_id", "recipe_id"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError("Explicit decision case and recipe required")
    if type(config["max_optimizer_steps"]) is not int or config["max_optimizer_steps"] <= 0:
        raise ValueError("Positive native optimizer-step cap required")
    job, cap = aug.prepare_job({k: config[k] for k in ("model_config", "max_wall_seconds", "max_sim_agent_steps")}, base, registry)
    require_native_mode(read_mapping(job["model_config"])["model"], "supervised")
    return {**job, **{k: config[k] for k in ("max_optimizer_steps", "case_id", "recipe_id")}}, cap


def run_selected(config, inputs, output, registry):
    report = json.loads(Path(inputs["decision"]["artifacts"]["report"]).read_text())
    choices = [r for r in report["decisions"] if r["case_id"] == config["case_id"]]
    choice = choices[0] if len(choices) == 1 else None
    if not choice or choice["action"] != "train" or choice["status"] != "reserved" or choice["recipe"] != config["recipe_id"]:
        output.mkdir(parents=True, exist_ok=False)
        return aug.result(output, {"decision": choice, "training_started": False}, status="rejected", reason="training_not_selected")
    cfg = read_mapping(config["model_config"])
    expected = {"mode": "supervised", "model_config_sha256": digest(config["model_config"]),
                "max_optimizer_steps": config["max_optimizer_steps"]}
    if choice["execution"] != expected:
        raise ValueError("Selected recipe execution differs from native consumer")
    parent = aug.bundle(path_at(project_root(), cfg["weights"]["parent"]))
    if not parent or signature(parent) != choice["parent_version"]:
        raise ValueError("Decision belongs to another parent checkpoint")
    data = read_mapping(inputs["dataset"]["artifacts"]["manifest"])
    groups = {source_log_group(r["log_name"]) for r in data["splits"]["train"]["records"]}
    if source_log_group(choice["source_log"]) not in groups:
        raise ValueError("Selected case is not in the training dataset")
    cap = choice["reserved_cost"]
    gpus = len(cfg["settings"]["gpus"])
    if (config["max_wall_seconds"] > cap["wall_seconds"] or
            config["max_wall_seconds"] * gpus > cap["gpu_seconds"] or
            config["max_sim_agent_steps"] > cap["score_queries"] * 80):
        # Current native scoring reserves reference + proposal: 80 agent steps/query.
        raise ValueError("Native job exceeds selected recipe reservation")
    result = aug.run_training(config, {"dataset": inputs["dataset"]}, output, registry)
    result["summary"]["decision_recipe"] = choice["recipe"]
    result["summary"]["training_mode"] = "supervised"
    result["summary"]["max_native_global_steps"] = config["max_optimizer_steps"]
    aug.write_json(output / "report.json", result["summary"])
    return result


def memory_from_study(path, context):
    """Recompute the existing sealed paired study, not a self-reported score."""
    from driversi.rsi.experiments import native_study as study_api
    study, store = study_api._registered(path)
    report = study_api.collect(path)
    if report["status"] != "completed":
        return {"study": str(path), "record": None, "status": report["status"],
                "arms": report["arms"], "positive_credit": False}
    state = store.get(study["tasks"]["fixed_tto"]["id"])
    student = study_api._stage(state, "train")
    evaluation = study_api._stage(state, "evaluation")
    artifacts = next(s for s in state["stages"] if s["id"] == "evaluation")["result"]["artifacts"]
    plan = read_mapping(artifacts["parent_plan"])
    evaluator_version = signature(study_api._execution(plan))
    data = aug.manifest(study["source_manifest"])
    total_wall = sum(a["wall_seconds"] for a in report["arms"].values())
    total_steps = sum(a["sim_agent_steps"] for a in report["arms"].values())
    # GPU allocation accounting is separately recorded by both native train/eval stages.
    gpu_seconds = 0.
    for arm in study["tasks"].values():
        arm_state = store.get(arm["id"])
        for name in ("train", "evaluation"):
            stage = study_api._stage(arm_state, name)
            value = stage.get("reserved_gpu_seconds")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Completed native study lacks GPU allocation accounting")
            gpu_seconds += value
    training_artifacts = next(s for s in state["stages"] if s["id"] == "train")["result"]["artifacts"]
    training_plan = read_mapping(training_artifacts["plan"])
    native_cap = yaml.safe_load(training_plan["generated_files"]["native.yaml"])["training"].get("max_steps")
    execution = ({"mode": "supervised", "model_config_sha256": study["model_config_sha256"],
                  "max_optimizer_steps": native_cap} if type(native_cap) is int and native_cap > 0 else None)
    record = {"id": study["id"], "case_id": context["case_id"], "node": context["node"],
              "recipe": context["recipe"], "parent_version": signature(student["parent"]["files"]),
              "evaluator_version": evaluator_version,
              "source_logs": sorted({r["log_name"] for r in data["splits"]["development"]["records"]}),
              "training_source_logs": student["training_sources"], "evaluation_role": "development",
              "failure_class": context["failure_class"],
              "parent_improved": report["candidate_vs_parent"]["mean_pdms_delta"] > 0,
              "outcome": "completed", "paired": True, "matched_budget": True,
              "safety_passed": bool(report["comparison"]["safety_nonregression"] and report["candidate_vs_parent"]["safety_nonregression"]),
              "delta": report["comparison"]["mean_pdms_delta"],
              "cost": {"wall_seconds": total_wall, "gpu_seconds": gpu_seconds, "score_queries": total_steps / 80},
              "execution": execution}
    return {"study": str(path), "record": record, "status": "completed",
            "observed_native_global_steps": student["updates"],
            "budget_scope": "equal_training_updates_and_recipe_not_total_historical_generation_cost",
            "positive_credit": record["safety_passed"] and record["parent_improved"] and record["delta"] > 0,
            "promoted": False}
