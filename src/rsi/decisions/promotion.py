"""Frozen paired engineering gate; never a statistical or driving qualification."""

import math
from pathlib import Path

from driversi.rsi.execution.execution_spec import METRICS


def validate_policy(policy):
    fields = {
        "id",
        "metric",
        "direction",
        "min_gain",
        "min_scenes",
        "min_groups",
        "min_actors",
        "max_collision",
        "max_offroad",
        "collision_tolerance",
        "offroad_tolerance",
    }
    if not isinstance(policy, dict) or set(policy) != fields:
        raise ValueError("Gate policy fields must be explicit")
    from driversi.rsi.contracts import _identity

    _identity(policy["id"], "gate policy")
    if policy["metric"] not in METRICS or policy["direction"] not in ("maximize", "minimize"):
        raise ValueError("Unknown gate metric/direction")
    for name in ("min_scenes", "min_groups", "min_actors"):
        if type(policy[name]) is not int or policy[name] < 1:
            raise ValueError("Positive gate evidence minimum required")
    for name in (
        "min_gain",
        "max_collision",
        "max_offroad",
        "collision_tolerance",
        "offroad_tolerance",
    ):
        x = policy[name]
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError("Finite nonnegative gate thresholds required")
        if name != "min_gain" and x > 1:
            raise ValueError("Safety rates/tolerances cannot exceed one")
    if policy["min_gain"] <= 0:
        raise ValueError("Promotion requires a positive minimum gain")
    return policy


def _metrics(result, expected_paths):
    rows = result["episodes"]
    if {r["scene"] for r in rows} != set(expected_paths) or len(rows) != len(expected_paths):
        raise ValueError("Paired evaluation lacks exact frozen scene coverage")
    for row in rows:
        if type(row["actors"]) is not int or row["actors"] < 1:
            raise ValueError("Invalid actor denominator")
        for key in ("collision", "offroad", "goal"):
            if type(row[key]) is not int or not 0 <= row[key] <= row["actors"]:
                raise ValueError("Invalid event count")
        if any(
            not math.isfinite(row[key]) for key in ("return_sum", "travel_m", "endpoint_progress_m")
        ):
            raise ValueError("Non-finite gate evidence")
    actors = sum(r["actors"] for r in rows)
    metrics = {k: sum(r[k.removesuffix("_per_actor")] for r in rows) / actors for k in METRICS}
    if result["actor_episodes"] != actors:
        raise ValueError("Actor denominator mismatch")
    return metrics, actors


def decide(policy, paired, paths, groups, reference_model):
    validate_policy(policy)
    paths = [str(Path(p).resolve()) for p in paths]
    if (
        paired.get("purpose") != "promotion_pair"
        or not paired.get("independent_process")
        or not paired.get("source_separation_checked")
        or paired.get("reference_model") != reference_model
    ):
        raise ValueError("Independent paired evaluation identity mismatch")
    candidate, actors = _metrics(paired, paths)
    baseline, old_actors = _metrics(paired["reference"], paths)
    if actors != old_actors or {r["scene"]: r["actors"] for r in paired["episodes"]} != {
        r["scene"]: r["actors"] for r in paired["reference"]["episodes"]
    }:
        raise ValueError("Paired actor support mismatch")
    evidence = (
        len(paths) >= policy["min_scenes"]
        and len(groups) >= policy["min_groups"]
        and actors >= policy["min_actors"]
    )
    gain = (candidate[policy["metric"]] - baseline[policy["metric"]]) * (
        1 if policy["direction"] == "maximize" else -1
    )
    checks = {"sufficient_evidence": evidence, "minimum_gain": gain >= policy["min_gain"]}
    for kind in ("collision", "offroad"):
        key = kind + "_per_actor"
        checks[kind + "_absolute"] = candidate[key] <= policy["max_" + kind]
        checks[kind + "_non_regression"] = (
            candidate[key] <= baseline[key] + policy[kind + "_tolerance"]
        )
    outcome = "promoted" if all(checks.values()) else "inconclusive" if not evidence else "rejected"
    return {
        "outcome": outcome,
        "policy_id": policy["id"],
        "checks": checks,
        "reasons": [k for k, v in checks.items() if not v],
        "gain": gain,
        "candidate_metrics": candidate,
        "reference_metrics": baseline,
        "scenes": len(paths),
        "groups": len(groups),
        "actor_episodes": actors,
        "approved_teacher": False,
        "official_score": False,
        "statistical_significance": "not_claimed",
    }
