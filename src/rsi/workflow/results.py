"""Model-independent Harness result envelope; no model/runtime import."""
from driversi.runtime.budget import write_json


def result(output, report, *, status="completed", reason=None, steps=0, artifacts=None):
    write_json(output / "report.json", report)
    return {
        "status": status,
        "reason": reason,
        "sim_agent_steps": steps,
        "artifacts": {"report": str(output / "report.json"), **(artifacts or {})},
        "summary": report,
        "admission": {"trainable": False, "promoted": False},
    }
