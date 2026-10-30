"""Bounded admitted-data rounds on the existing Harness training lifecycle.

Preparation is not a training result. Visual rejection cannot be overridden
here; the existing synthetic validators and independent evaluation own gates.
"""

from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.plugins.chainflow_synthetic import compile_paired_task
from driversi.rsi.workflow import engine
from driversi.runtime.budget import write_json


def run(config_path, *, execute=False):
    config_path = Path(config_path).resolve(strict=True)
    config = read_mapping(config_path)
    _keys(
        config,
        ("base_manifest", "sample_reports", "model_config", "output", "updates_per_arm",
         *(k for k in ("capability_plan", "skill_memory", "epoch_training", "batch_size") if k in config)),
        "ChainFlow incremental round",
    )
    root = project_root().resolve()
    output = (root / config["output"]).resolve()
    if not output.is_relative_to(root / "exp") or output.exists():
        raise ValueError("Incremental round requires a new exp directory")
    if not isinstance(config["sample_reports"], list) or not config["sample_reports"]:
        raise ValueError("Explicit admitted data receipt list required")

    def path(value):
        return (root / value).resolve(strict=True)

    output.mkdir(parents=True)
    report = dict(
        schema_version=1,
        kind="chainflow_incremental_round",
        config_path=str(config_path),
        config_sha256=digest(config_path),
        status="preparing",
        execution_requested=execute,
        training_executed=False,
        promoted=False,
        stable_improvement_demonstrated=False,
    )
    execution_started = False
    try:
        task = compile_paired_task(
            path(config["base_manifest"]),
            [path(p) for p in config["sample_reports"]],
            path(config["model_config"]),
            output / (output.name + "-pair"),
            updates_per_arm=config["updates_per_arm"],
            epoch_training=config.get("epoch_training", False),
            batch_size=config.get("batch_size", 1),
            capability_plan=path(config["capability_plan"]) if config.get("capability_plan") else None,
            skill_memory=(root / config["skill_memory"]).resolve() if config.get("skill_memory") else None,
        )
        report.update(status="prepared", task=str(task), task_sha256=digest(task))
        if execute:
            registry = output / "harness.sqlite"
            engine.register(registry, task, output / "tasks")
            task_id = read_mapping(task)["id"]
            execution_started = True
            report["training_executed"] = None
            outcome = engine.run(registry, task_id)
            report.update(
                status="executed",
                harness_outcome={k: outcome.get(k) for k in ("status", "failure_reason")},
                execution_record=str(output / "tasks" / task_id),
            )
            feedback = output / "tasks" / task_id / "paired_feedback" / "report.json"
            if outcome.get("status") == "completed" and feedback.exists():
                value = read_mapping(feedback)
                verification = feedback.parent.parent / "treatment_feedback" / "feedback.json"
                resume = read_mapping(verification)["cross_process_resume"]
                verified = all(
                    resume.get(k) is True for k in ("different_pids", "same_compared_state")
                )
                report.update(
                    status="feedback_available" if verified else "resume_verification_failed",
                    training_executed=True,
                    paired_feedback=str(feedback),
                    paired_feedback_sha256=digest(feedback),
                    resume_verification=str(verification),
                    resume_verification_sha256=digest(verification),
                    cross_process_resume=resume,
                    observed_pilot_benefit=verified and value["observed_pilot_benefit"],
                    next_action=value["next_action"] if verified else "diagnose_resume_mismatch",
                )
            else:
                report["status"] = "incomplete_execution"
                report["training_executed"] = None  # stage records carry any partial updates
    except (ValueError, FileNotFoundError) as error:
        report.update(
            status="execution_error" if execution_started else "blocked",
            reason=type(error).__name__ + ": " + str(error),
            training_executed=None if execution_started else False,
        )
    except BaseException as error:
        report.update(
            status="execution_error" if execution_started else "preparation_error",
            reason=type(error).__name__ + ": " + str(error),
            training_executed=None,
        )
        raise
    finally:
        write_json(output / "report.json", report)
    return report
