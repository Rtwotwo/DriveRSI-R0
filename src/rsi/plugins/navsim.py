"""NAVSIM-centered augmentation decisions and artifact-bound synthesis dispatch."""

from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.decisions.augmentation import choose
from driversi.rsi.plugins import reactive
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files, protect
from driversi.runtime.budget import write_json
from driversi.training.sampling.environment import source_log_group


def _protected(registry):
    from driversi.rsi.execution.registry import ExperimentRegistry

    official = (
        project_root()
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    return (
        list(ExperimentRegistry(registry).protected_groups()) + read_mapping(official)["log_names"]
    )


def _load(path, registry):
    doc = read_mapping(path)
    feedback = doc.pop("capability_feedback", None)
    _keys(
        doc,
        (
            "schema_version",
            "train_logs",
            "development_logs",
            "protected_logs",
            "cases",
            "budget",
            "costs",
            "student_checkpoint",
            "score_contract",
        ),
        "NAVSIM augmentation evidence",
    )
    if (
        doc["schema_version"] != 1
        or not isinstance(doc["student_checkpoint"], str)
        or not doc["student_checkpoint"]
    ):
        raise ValueError("Versioned student evidence required")
    scoring = doc["score_contract"]
    _keys(scoring, ("protocol", "metric", "threshold", "context"), "fixed scoring contract")
    # This selector has no scoring authority and cannot lower the existing 95-point requirement.
    if (
        scoring["metric"] not in ("PDMS", "EPDMS")
        or scoring["threshold"] != 0.95
        or scoring["context"] != "executed_counterfactual"
    ):
        raise ValueError(
            "Require 0.95 independent counterfactual scoring, distinct from logged caches"
        )
    if not isinstance(scoring["protocol"], str) or not scoring["protocol"]:
        raise ValueError("Explicit independent scoring protocol required")
    if any(c["parent_version"] != doc["student_checkpoint"] for c in doc["cases"]):
        raise ValueError("Case evidence belongs to another student checkpoint")
    protected = doc["protected_logs"] + _protected(registry)
    plan = choose(
        doc["cases"],
        train_logs=doc["train_logs"],
        development_logs=doc["development_logs"],
        protected_logs=protected,
        budget=doc["budget"],
        costs=doc["costs"],
        capability_feedback=feedback,
    )
    doc["capability_feedback"] = feedback
    return doc, protected, plan


def prepare_decision(config, base, registry):
    _keys(config, ("evidence", "llm"), "NAVSIM selection config")
    path = (base / config["evidence"]).resolve(strict=True)
    doc, _, _ = _load(path, registry)
    files = {str(path): digest(path)}
    for case in doc["cases"]:
        for ref in case["evidence"]:
            p = (path.parent / ref).resolve(strict=True)
            files[str(p)] = digest(p)
    llm = None
    if config["llm"] is not None:
        from driversi.rsi.decisions.llm_advisor import validate_config

        p = (base / config["llm"]).resolve(strict=True)
        llm = validate_config(read_mapping(p))
        files[str(p)] = digest(p)
    return {
        "evidence": str(path),
        "files": files,
        "llm": llm,
        "source_groups": sorted(
            {source_log_group(x) for x in doc["train_logs"] + doc["development_logs"]}
        ),
    }, 0


def run_decision(config, inputs, output, registry):
    check_files(config["files"])
    doc, protected, report = _load(Path(config["evidence"]), registry)
    feedback_ref = None
    if inputs:
        feedback = read_mapping(inputs["feedback"]["artifacts"]["report"])
        weakness = feedback.get("weakness_review", {})
        if weakness.get("checkpoint") != doc["student_checkpoint"]:
            raise ValueError("Synthesis selection belongs to a different evaluated student")
        feedback_ref = inputs["feedback"]["summary"]["artifact_reference"]["identity"]
        from driversi.rsi.decisions.evaluation_aggregate import from_weakness_review

        if weakness.get("support", {}).get("paired_samples"):
            measured = from_weakness_review(
                weakness,
                evaluator_version=str(
                    weakness.get("evaluator_context")
                    or weakness.get("metric_scope")
                    or "native-development"
                ),
            )
            doc["capability_feedback"] = measured
            report = choose(
                doc["cases"],
                train_logs=doc["train_logs"],
                development_logs=doc["development_logs"],
                protected_logs=protected,
                budget=doc["budget"],
                costs=doc["costs"],
                capability_feedback=measured,
            )
            report["feedback_use"] = "measured_development_capabilities_applied_to_train_selection"
        else:
            report["feedback_use"] = "insufficient_measured_support_no_new_priority"
        # Only the aggregate crosses this stage. Detailed development example IDs
        # remain in the independent evaluator's receipt.
        report["development_weaknesses"] = report.get("capability_feedback")
    llm_receipt = {"status": "disabled", "requests": 0}
    if config["llm"] is not None:
        from driversi.rsi.decisions.llm_advisor import advise

        try:
            proposal, usage = advise(
                config["llm"], doc["cases"], capability_feedback=doc.get("capability_feedback")
            )
            new = choose(
                doc["cases"],
                train_logs=doc["train_logs"],
                development_logs=doc["development_logs"],
                protected_logs=protected,
                budget=doc["budget"],
                costs=doc["costs"],
                advice=proposal,
                capability_feedback=doc.get("capability_feedback"),
            )
            report.update({k: new[k] for k in ("decisions", "remaining_budget", "advice")})
            llm_receipt = {"status": "received_and_validated", **usage}
        except Exception as exc:
            # Credentials, HTTP bodies and local paths never enter an exception report.
            llm_receipt = {
                "status": "unavailable_or_invalid_using_deterministic_plan",
                "exception_type": type(exc).__name__,
                "requests_at_most": 1,
            }
    check_files(config["files"])
    report.update(
        student_checkpoint=doc["student_checkpoint"],
        score_contract=doc["score_contract"],
        source_files=config["files"],
        feedback_reference=feedback_ref,
        llm=llm_receipt,
        train_logs=doc["train_logs"],
        development_logs=doc["development_logs"],
    )
    output.mkdir(parents=True)
    return result(output, report)


def prepare_world(config, base, registry):
    _keys(config, ("case_id", "world"), "selected NAVSIM synthesis")
    world, cap = reactive.prepare_world(config["world"], base, registry)
    scene = Scene.load(Path(world["scene"]))
    if "nuplan_source" in scene.metadata or scene.metadata["split"] != "train":
        raise ValueError("Visual synthesis mainline requires a NAVSIM training source")
    if scene.metadata["scenario_id"] != config["case_id"]:
        raise ValueError("Selected token differs from scene anchor")
    from driversi.data.datasets.history_windows import source_history
    from driversi.data.sources.navsim import load_log

    history = source_history(
        load_log(Path(world["log"]), trusted=True),
        scene,
        world["sensor_root"],
        camera_ids=("CAM_F0", "CAM_L0", "CAM_R0"),
    )
    for frame in history["frames"]:
        for camera in frame["cameras"]:
            world["files"][camera["path"]] = camera["sha256"]
    return {
        "case_id": config["case_id"],
        "world": world,
        "source_groups": sorted({source_log_group(x) for x in world["source_groups"]}),
    }, cap


def run_world(config, inputs, output, registry):
    decision = read_mapping(inputs["decision"]["artifacts"]["report"])
    check_files(decision["source_files"])
    rows = [x for x in decision["decisions"] if x["case_id"] == config["case_id"]]
    row = rows[0] if len(rows) == 1 else None
    scene = Scene.load(Path(config["world"]["scene"]))
    protect([scene.metadata["source_log"]], registry, decision["development_logs"])
    if (
        not row
        or row["status"] != "reserved"
        or row["mode"] != "new_state"
        or row["generator"] != "simscale_rule"
        or row["source_log"] != scene.metadata["source_log"]
        or source_log_group(row["source_log"])
        not in {source_log_group(x) for x in decision["train_logs"]}
    ):
        output.mkdir(parents=True)
        return result(
            output,
            {"selected": row, "execution_started": False},
            status="rejected",
            reason="no_matching_navsim_synthesis_reservation",
        )
    if row["reserved_cost"]["wall_seconds"] < 120:
        raise ValueError("Selected budget cannot cover the bounded 120-second generator")
    reservation = claim_reservation(
        registry,
        inputs["decision"]["summary"]["artifact_reference"]["identity"],
        row["case_id"],
        output,
        row["reserved_cost"],
    )
    out = reactive.run_world(config["world"], {}, output, registry)
    # Keep the atomic registry marker canonical, but publish a stage-local receipt:
    # Harness sealing disallows artifacts outside this stage's directory.
    local_reservation = output / "reservation.json"
    local_reservation.write_bytes(reservation.read_bytes())
    out["artifacts"]["reservation"] = str(local_reservation)
    out["summary"].update(
        decision_reference=inputs["decision"]["summary"]["artifact_reference"]["identity"],
        case_id=row["case_id"],
        generator="simscale_rule",
        learned_self_play=False,
    )
    receipt = read_mapping(out["artifacts"]["report"])
    receipt.update(
        selection=out["summary"],
        score_contract=decision["score_contract"],
        reservation=str(reservation),
    )
    write_json(Path(out["artifacts"]["report"]), receipt)
    return out


def claim_reservation(registry, decision_identity, case_id, output, cost):
    """One immutable reservation per sealed decision/case, across tasks/processes.

    Consumed before execution; crashes/failures never implicitly release it.
    A new experiment requires a newly sealed decision with its own budget.
    """
    import json
    import os
    import time

    from driversi.agents.backends.native import signature

    key = signature({"decision": decision_identity, "case_id": case_id})
    ledger = Path(registry).resolve()
    root = ledger.parent / (ledger.name + ".augmentation-reservations")
    root.mkdir(parents=True, exist_ok=True)
    path = root / (key + ".json")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Synthesis reservation already consumed; require a new decision") from exc
    with os.fdopen(fd, "w") as stream:
        json.dump(
            {
                "decision_identity": decision_identity,
                "case_id": case_id,
                "execution_directory": str(output.resolve()),
                "reserved_cost": cost,
                "claimed_at": time.time(),
                "pid": os.getpid(),
                "status": "consumed_before_execution_including_failures",
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return path


def prepare_export(config, base, registry):
    _keys(config, ("cameras",), "NAVSIM mainline export")
    return reactive.prepare_export({**config, "future_images": True}, base, registry)


def run_export(config, inputs, output, registry):
    scene = Scene.load(Path(inputs["rendered"]["artifacts"]["scene"]))
    if "nuplan_source" in scene.metadata or scene.metadata["split"] != "train":
        raise ValueError("NAVSIM mainline export requires training-source images")
    protect([scene.metadata["source_log"]], registry)
    out = reactive.run_export(config, inputs, output, registry)
    report = read_mapping(Path(out["artifacts"]["report"]))
    if not report["future_images_included"] or report["image_counts"] != {
        "history": 4 * len(config["cameras"]),
        "future": 8 * len(config["cameras"]),
    }:
        raise ValueError("Mainline requires four history and eight future RGB captures per camera")
    report["rsi_usage"] = {
        "history": "student_input",
        "future_rgb": "supervision_only",
        "future_trajectory": "teacher_label_requires_independent_admission",
        "source_group": source_log_group(report["source_log"]),
        "return_pool": "navtrain_after_trajectory_and_visual_admission",
        "nuplan_role": "analysis_only",
        "training_started": False,
    }
    write_json(Path(out["artifacts"]["report"]), report)
    out["summary"]["rsi_usage"] = report["rsi_usage"]
    return out


def register(registry, Provider):
    for p in (
        Provider(
            "navsim_augmentation_decision",
            "1",
            (),
            "navsim_augmentation_plan",
            prepare_decision,
            run_decision,
            ("report",),
        ),
        Provider(
            "navsim_feedback_augmentation",
            "1",
            (("feedback", "unified_learning_feedback"),),
            "navsim_augmentation_plan",
            prepare_decision,
            run_decision,
            ("report",),
        ),
        Provider(
            "navsim_selected_synthesis",
            "1",
            (("decision", "navsim_augmentation_plan"),),
            "reactive_execution",
            prepare_world,
            run_world,
            ("report", "bundle"),
        ),
        Provider(
            "navsim_history_future_export",
            "1",
            (("rendered", "rendered_reactive_branch"),),
            "navsim_inspection_dataset",
            prepare_export,
            run_export,
            ("report",),
        ),
    ):
        registry.register(p)
