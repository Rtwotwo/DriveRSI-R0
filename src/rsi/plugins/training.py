"""Unified native training jobs under existing Harness process supervision."""

from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.training.contracts.job import resolve_request
from driversi.training.orchestration import engine


def prepare_job(config, base, registry):
    _keys(config, ("request",), "unified training provider")
    path = (base / config["request"]).resolve(strict=True)
    resolved = resolve_request(path)
    budget = resolved["request"]["budget"]
    if resolved["request"]["schema_version"] in (2, 3, 4):
        from driversi.training.orchestration.lifecycle import sources

        groups = sources(read_mapping((path.parent / resolved["request"]["data"]).resolve()))
    else:
        groups = None
    return {
        "request": str(path),
        "request_sha256": digest(path),
        "model_config": resolved["model_config"],
        "model_config_sha256": resolved["model_config_sha256"],
        "source_groups": sorted(set(groups["train"] + groups["development"]))
        if groups is not None
        else (
            engine._sources(read_mapping(resolved["model_config"]), registry)
            if resolved["backend"]
            else []
        ),
    }, budget["max_scoring_agent_steps"] + max(
        20 + 2 * budget["max_environment_steps"]
        if resolved["request"]["schema_version"] == 2 and resolved["model"] == "driverl"
        else budget["max_environment_steps"],
        2 * resolved["request"]["pilot"]["evaluation_steps"]
        if resolved["request"]["schema_version"] == 2 and resolved["model"] == "driverl"
        else 0,
    )


def run_job(config, inputs, output, registry):
    if (
        digest(config["request"]) != config["request_sha256"]
        or digest(config["model_config"]) != config["model_config_sha256"]
    ):
        raise ValueError("Unified training request changed since registration")
    engine.prepare(config["request"], output, registry=registry)
    report = engine.run(output / "training-plan.json", own_process_group=False)
    status = "budget_exhausted" if report["status"] == "timeout" else report["status"]
    return {
        "status": status,
        "reason": report["reason"],
        "sim_agent_steps": report["consumed"].get(
            "sim_agent_steps",
            report["consumed"].get("scoring_agent_steps", 0)
            + report["consumed"].get("environment_steps", 0),
        ),
        "artifacts": {
            "report": str(output / "training-result.json"),
            "plan": str(output / "training-plan.json"),
        },
        "summary": report,
        "admission": {"trainable": False, "promoted": False},
    }


def _phase(config, inputs, output, registry, phase):
    from driversi.training.orchestration import lifecycle

    if (
        digest(config["request"]) != config["request_sha256"]
        or digest(config["model_config"]) != config["model_config_sha256"]
    ):
        raise ValueError("Learning declaration changed")
    resolved = resolve_request(config["request"])
    plan = lifecycle.prepare(
        resolved,
        output,
        phase=phase,
        previous=inputs["student"]["artifacts"]["report"],
        evaluator=config.get("evaluator"),
    )
    report = lifecycle.execute(plan, own_process_group=False)
    return {
        "status": report["status"],
        "reason": report.get("reason"),
        "sim_agent_steps": report["consumed"].get(
            "sim_agent_steps",
            report["consumed"].get("environment_steps", 0)
            + report["consumed"].get("scoring_agent_steps", 0),
        ),
        "artifacts": {
            "report": str(output / "training-result.json"),
            "plan": str(output / "training-plan.json"),
        },
        "summary": report,
        "admission": {"trainable": False, "promoted": False},
    }


def run_resume(config, inputs, output, registry):
    return _phase(config, inputs, output, registry, "resume")


def prepare_evaluation_job(config, base, registry):
    if set(config) not in ({"request"}, {"request", "evaluator"}):
        raise ValueError("Malformed independent evaluation job")
    prepared, budget = prepare_job({"request": config["request"]}, base, registry)
    if "evaluator" in config:
        from driversi.training.contracts.evaluation import resolve_context
        from driversi.training.orchestration.lifecycle import sources

        resolved = resolve_request(prepared["request"])
        data = read_mapping(
            (Path(prepared["request"]).parent / resolved["request"]["data"]).resolve()
        )
        context = resolve_context(
            (base / config["evaluator"]).resolve(), resolved["model"], sources(data)["development"]
        )
        prepared["evaluator"] = context["manifest"]
        prepared["evaluator_sha256"] = context["manifest_sha256"]
    return prepared, budget


def run_evaluation(config, inputs, output, registry):
    if config.get("evaluator") and digest(config["evaluator"]) != config["evaluator_sha256"]:
        raise ValueError("Evaluator context changed after registration")
    return _phase(config, inputs, output, registry, "evaluate")


def prepare_feedback(config, base, registry):
    if config:
        raise ValueError("Learning feedback has no tunable scoring thresholds")
    return {}, 0


def run_feedback(config, inputs, output, registry):
    from driversi.runtime.budget import write_json
    from driversi.training.orchestration.lifecycle import feedback, training_lineage

    reports = {k: read_mapping(v["artifacts"]["report"]) for k, v in inputs.items()}
    result = feedback(reports["student"], reports["evaluation"])
    lineage = training_lineage(inputs["student"]["artifacts"]["report"])
    result["training_cost"] = lineage["cost"]
    result["training_cost_evidence"] = lineage
    repeat = reports["replay"]
    student = reports["student"]
    if repeat["context"] != student["context"] or repeat["resume_from"] != student["resume_from"]:
        raise ValueError("Resume comparison used a different recovery boundary")
    result["cross_process_resume"] = {
        "different_pids": repeat["worker_pid"] != student["worker_pid"],
        "same_compared_state": repeat["state_fingerprint"] == student["state_fingerprint"],
        "student_pid": student["worker_pid"],
        "replay_pid": repeat["worker_pid"],
    }
    result["cross_process_resume"]["comparison_scope"] = (
        "model_optimizer_scheduler_rng_world_observation"
        if student["model"] == "driverl"
        else "model_optimizer_scheduler_global_rng_sample_cursor"
        if student["model"] == "chainflow_stage1"
        else "model_optimizer_scheduler_global_rng"
    )
    result["cross_process_resume"]["uninterrupted_equivalence_tested"] = False
    result["replay_cost"] = repeat["consumed"]
    from driversi.rsi.decisions.weaknesses import review

    result["weakness_review"] = review(student, reports["evaluation"], result)
    if result["weakness_review"]["local_safety_regressions"]:
        result["pilot_metric_improved"] = False
        result["reason"] = "paired_sample_safety_regression_requires_diagnosis"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "feedback.json", result)
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": {"report": str(output / "feedback.json")},
        "summary": result,
        "admission": {"promoted": False, "trainable": False},
    }


def _training_reference(config, registry):
    from driversi.rsi.workflow.artifacts import verify
    from driversi.rsi.workflow.store import HarnessStore

    store = HarnessStore(registry)
    state = store.get(config["source_task"])
    stage = next((x for x in state["stages"] if x["id"] == config["source_stage"]), None)
    declaration = next(
        (x for x in state["task"]["steps"] if x["id"] == config["source_stage"]), None
    )
    if (
        stage is None
        or stage["status"] != "completed"
        or declaration["output_kind"] != "unified_training_result"
    ):
        raise ValueError("Only a completed typed training stage can be reused")
    if set(state["task"]["source_groups"]) & store.registry.protected_groups():
        raise ValueError("Reused training source became protected")
    ref = verify(stage["result"], state["id"], declaration)
    report = read_mapping(stage["result"]["artifacts"]["report"])
    if report["status"] != "completed" or report["phase"] not in ("train", "resume"):
        raise ValueError("Not a completed learning receipt")
    return state, stage["result"], ref


def prepare_reference(config, base, registry):
    _keys(config, ("source_task", "source_stage"), "completed training reference")
    state, result, ref = _training_reference(config, registry)
    return {
        **config,
        "identity": ref["identity"],
        "source_groups": state["task"]["source_groups"],
    }, 0


def run_reference(config, inputs, output, registry):
    import shutil

    from driversi.runtime.budget import write_json

    _, result, ref = _training_reference(config, registry)
    if ref["identity"] != config["identity"]:
        raise ValueError("Referenced training stage changed after registration")
    if not output.resolve().is_relative_to(engine.native.project_root() / "exp"):
        raise ValueError("Training references must live under exp")
    # Reuse sealed metadata; never repeat optimizer steps or duplicate large weights.
    # The evaluator/resumer revalidates every referenced checkpoint content hash.
    output.mkdir(parents=True, exist_ok=False)
    for name, filename in (("report", "training-result.json"), ("plan", "training-plan.json")):
        shutil.copy2(result["artifacts"][name], output / filename)
    write_json(output / "origin.json", {"reused": True, "training_executed": False, "source": ref})
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": {
            "report": str(output / "training-result.json"),
            "plan": str(output / "training-plan.json"),
            "origin": str(output / "origin.json"),
        },
        "summary": {"reused_training": True, "source_reference": ref["identity"]},
        "admission": {"promoted": False, "trainable": False},
    }
