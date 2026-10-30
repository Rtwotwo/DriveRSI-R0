"""Declared comparison contracts. Validation is not independent experiment verification."""
import copy
import json
import math
import re

BUDGET_FIELDS = ("gpu_seconds", "cpu_seconds", "llm_tokens",
                 "sim_agent_steps", "rendered_frames", "trials")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _identity(value, name):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f"{name} requires a bounded version/id")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} requires nonempty text")
    return value


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} requires exactly {sorted(keys)}")


def _groups(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} requires raw-log lineage groups")
    for group in value:
        _text(group, name)
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate {name}")
    return set(value)


def _cost(value, name, positive=False):
    _keys(value, BUDGET_FIELDS, name)
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (float, int)):
            raise ValueError(f"Invalid {name}.{key}")
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"Invalid {name}.{key}")
        if key not in ("gpu_seconds", "cpu_seconds") and int(number) != number:
            raise ValueError(f"{name}.{key} must be an integer")
    if positive and value["trials"] < 1:
        raise ValueError("At least one trial must be budgeted")


def validate_plan(value):
    plan = copy.deepcopy(value)
    _keys(plan, ("schema_version", "id", "target", "hypothesis", "arms", "protocol",
                 "data", "budget_per_arm", "scene_ids"), "plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Unsupported plan schema")
    _identity(plan["id"], "plan.id")
    if plan["target"] not in ("model", "improver"):
        raise ValueError("target must distinguish model from improver")
    _text(plan["hypothesis"], "hypothesis")
    _keys(plan["arms"], ("baseline", "candidate"), "arms")
    for arm in plan["arms"].values():
        _keys(arm, ("version", "initial_model", "information_set", "tools"), "arm")
        for key, val in arm.items():
            _identity(val, key)
    a, b = plan["arms"]["baseline"], plan["arms"]["candidate"]
    if a["version"] == b["version"]:
        raise ValueError("Comparison needs distinct version references")
    if plan["target"] == "improver":
        for key in ("initial_model", "information_set", "tools"):
            if a[key] != b[key]:
                raise ValueError(f"Improver comparison requires equal {key}")
    protocol = plan["protocol"]
    _keys(protocol, ("id", "evaluator_version", "role", "task_set", "metric", "direction"),
          "protocol")
    for key in ("id", "evaluator_version", "task_set"):
        _identity(protocol[key], key)
    _text(protocol["metric"], "metric")
    if protocol["role"] not in ("development", "promotion", "meta_holdout"):
        raise ValueError("Unknown protocol role")
    if protocol["direction"] not in ("maximize", "minimize"):
        raise ValueError("Unknown metric direction")
    _keys(plan["data"], ("development_groups", "evaluation_groups"), "data")
    develop = _groups(plan["data"]["development_groups"], "development_groups")
    evaluate = _groups(plan["data"]["evaluation_groups"], "evaluation_groups")
    if develop & evaluate:
        raise ValueError("Declared source-log lineage overlap")
    _cost(plan["budget_per_arm"], "budget_per_arm", positive=True)
    if not isinstance(plan["scene_ids"], list):
        raise ValueError("scene_ids must be a list")
    for sid in plan["scene_ids"]:
        _identity(sid, "scene_id")
    json.dumps(plan, allow_nan=False)
    return plan


def validate_attempt(value):
    attempt = copy.deepcopy(value)
    _keys(attempt, ("id", "plan_id", "arm", "status", "cost", "metrics", "artifacts",
                    "candidate_model", "failure_reason"), "attempt")
    for key in ("id", "plan_id"):
        _identity(attempt[key], key)
    if attempt["arm"] not in ("baseline", "candidate"):
        raise ValueError("Unknown arm")
    if attempt["status"] not in ("completed", "failed", "cancelled"):
        raise ValueError("Record terminal attempts, including failures and cancellations")
    _cost(attempt["cost"], "cost")
    if attempt["cost"]["trials"] != 1:
        raise ValueError("Each recorded attempt consumes exactly one trial")
    if not isinstance(attempt["metrics"], dict):
        raise ValueError("metrics must be a mapping")
    for key, val in attempt["metrics"].items():
        _text(key, "metric")
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError("Metrics must be finite numeric observations")
    if not isinstance(attempt["artifacts"], dict):
        raise ValueError("artifacts must be a mapping")
    for key, val in attempt["artifacts"].items():
        _text(key, "artifact name")
        _text(val, "artifact reference")
    if attempt["candidate_model"] is not None:
        _identity(attempt["candidate_model"], "candidate_model")
    if attempt["status"] == "completed":
        if attempt["candidate_model"] is None:
            raise ValueError("Completed attempts require resulting model version")
        if attempt["failure_reason"] is not None:
            raise ValueError("Completed attempt cannot have failure reason")
    else:
        _text(attempt["failure_reason"], "failure_reason")
    json.dumps(attempt, allow_nan=False)
    return attempt
