"""Native model inventory and official load check for DriveHarness.

Evaluation is not dispatched here until per-scenario simulator cost and source
isolation have a native accounting adapter. No zero-cost evaluation loophole.
"""

import json
from pathlib import Path

from driversi.agents.backends.native import execute, prepare
from driversi.rsi.contracts import _keys


def prepare_inventory(config, base, registry):
    _keys(config, ("model_config", "operation"), "native inventory")
    path = (base / config["model_config"]).resolve(strict=True)
    from driversi.agents.backends.native import project_root

    plan = prepare(
        path, config["operation"], project_root() / "outputs/native-inventory-placeholder"
    )
    return {"plan": plan}, 0


def run_inventory(config, inputs, output, registry):
    output.mkdir(parents=True, exist_ok=False)
    plan = config["plan"]
    current = prepare(plan["config_path"], plan["operation"], plan["output"])
    if current != plan:
        raise ValueError("Native model changed since Harness preparation")
    report = output / "report.json"
    report.write_text(json.dumps(plan, indent=2))
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": {"report": str(report)},
        "summary": {"ready_to_attempt": plan["ready_to_attempt"], "missing": plan["missing"]},
        "admission": {"trainable": False, "model": "inventory_only"},
    }


def prepare_preflight(config, base, registry):
    _keys(config, ("max_wall_seconds",), "native preflight")
    import math

    value = config["max_wall_seconds"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 600:
        raise ValueError("Bounded official preflight time required")
    return config, 0


def run_preflight(config, inputs, output, registry):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(inputs["inventory"]["artifacts"]["report"]).read_text())
    if plan["operation"] != "preflight":
        raise ValueError("Only official checkpoint-load preflight has zero simulator cost")
    current = prepare(plan["config_path"], "preflight", plan["output"])
    if current != plan:
        raise ValueError("Native evidence changed since inventory")
    if not plan["ready_to_attempt"]:
        report = output / "report.json"
        report.write_text(json.dumps(plan, indent=2))
        return {
            "status": "failed",
            "reason": "native_prerequisites_missing",
            "sim_agent_steps": 0,
            "artifacts": {"report": str(report)},
            "summary": {"missing": plan["missing"]},
            "admission": {"trainable": False, "promoted": False},
        }
    actual = prepare(plan["config_path"], "preflight", output / "native")
    result = execute(actual, timeout_seconds=config["max_wall_seconds"], own_process_group=False)
    return {
        "status": "budget_exhausted" if result["status"] == "timeout" else result["status"],
        "reason": "native_preflight_wall_timeout"
        if result["status"] == "timeout"
        else result.get("reason"),
        "sim_agent_steps": 0,
        "artifacts": {"report": str(output / "native/report.json")},
        "summary": result,
        "admission": {"trainable": False, "promoted": False},
    }
