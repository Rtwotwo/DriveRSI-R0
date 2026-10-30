"""Aggregate evaluation -> training-type hypotheses, never test scene synthesis."""

from copy import deepcopy

CAPABILITIES = {
    "interaction",
    "recovery",
    "distribution_shift",
    "occlusion",
    "collision",
    "road_compliance",
    "direction_compliance",
    "route_progress",
    "comfort",
}


def validate_feedback(value, checkpoint=None):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "benchmark",
        "checkpoint",
        "evaluator_version",
        "slices",
    }:
        raise ValueError("Aggregate-only capability feedback required")
    if value["schema_version"] != 1 or value["benchmark"] not in {
        "navtest",
        "navhard",
        "development",
    }:
        raise ValueError("Unknown evaluation aggregate protocol")
    if any(
        not isinstance(value[k], str) or not value[k] for k in ("checkpoint", "evaluator_version")
    ):
        raise ValueError("Evaluation identity required")
    if checkpoint is not None and value["checkpoint"] != checkpoint:
        raise ValueError("Capability feedback belongs to another student")
    if not isinstance(value["slices"], list) or len(value["slices"]) > len(CAPABILITIES):
        raise ValueError("Bounded capability slices required")
    seen = set()
    for row in value["slices"]:
        if not isinstance(row, dict) or set(row) != {"capability", "count", "failures"}:
            raise ValueError(
                "No tokens, scenes, coordinates or labels allowed in aggregate feedback"
            )
        if row["capability"] not in CAPABILITIES or row["capability"] in seen:
            raise ValueError("Unknown/duplicate capability")
        if (
            type(row["count"]) is not int
            or type(row["failures"]) is not int
            or not 0 <= row["failures"] <= row["count"]
            or row["count"] == 0
        ):
            raise ValueError("Valid observed counts required")
        seen.add(row["capability"])
    return deepcopy(value)


def priorities(value, checkpoint=None):
    value = validate_feedback(value, checkpoint)
    return {r["capability"]: r["failures"] / r["count"] for r in value["slices"] if r["failures"]}


def public_summary(value):
    value = validate_feedback(value)
    return {
        "benchmark": value["benchmark"],
        "slices": value["slices"],
        "purpose": "training_type_hypotheses_only",
        "evaluation_role": "adaptive_evaluation"
        if value["benchmark"] != "development"
        else "development",
    }
