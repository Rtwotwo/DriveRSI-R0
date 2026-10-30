"""Durable worker phases: distinguish known preconditions from uncertain execution."""
import json
from pathlib import Path

from driversi.runtime.budget import write_json


def receipt(directory, task_id, stage_id, phase, error=None, failed_phase=None):
    value = {"schema_version": 1, "task_id": task_id, "stage_id": stage_id,
             "phase": phase, "failed_phase": failed_phase,
             "error_type": type(error).__name__ if error else None,
             "reason": str(error) if error else None}
    write_json(Path(directory) / (stage_id + ".receipt.json"), value)


def failure(directory, task_id, stage_id):
    try:
        value = json.loads((Path(directory) / (stage_id + ".receipt.json")).read_text())
    except (OSError, ValueError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    if (value.get("schema_version"), value.get("task_id"), value.get("stage_id")) != (1, task_id, stage_id):
        value = {}
    phase = value.get("failed_phase") if value.get("phase") == "failed" else value.get("phase", "unknown")
    known_precondition = value.get("phase") == "failed" and phase == "validating"
    category = {"validating": "precondition_failed", "executing": "provider_execution",
                "publishing": "output_validation"}.get(phase, "execution_outcome_unknown")
    return {"category": category, "phase": phase, "error_type": value.get("error_type"),
            "reason": value.get("reason"), "execution_not_entered": known_precondition,
            "automatic_retry": False}
