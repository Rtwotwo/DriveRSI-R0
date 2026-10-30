"""Model-specific supervision validators behind one generated-data interface."""

import json
from pathlib import Path


def validate_sample(path):
    value = json.loads(Path(path).read_text())
    if value.get("kind") == "augustus_admitted_synthetic_sample":
        from driversi_agents.augustus.branch_supervision import validate_sample as validate

        return validate(path)
    if (
        value.get("kind") != "chainflow_admitted_synthetic_sample"
        or value.get("trainable") is not True
    ):
        raise ValueError("Known admitted native sample kind required")
    from driversi_agents.chainflow.synthetic import validate_admission

    validate_admission(value["row"], value["files"])
    return dict(value, sample_id=value["row"]["sample_id"], source_log=value["row"]["source_log"])
