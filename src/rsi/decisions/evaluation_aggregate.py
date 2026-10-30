"""Evaluator-side aggregation. Only counts and fixed history tags leave this boundary."""

import math
from collections import Counter, defaultdict

from driversi.rsi.decisions.capability_feedback import validate_feedback
from driversi.training.sampling.environment import source_log_group

METRICS = {
    "no_at_fault_collisions": "collision",
    "drivable_area_compliance": "road_compliance",
    "driving_direction_compliance": "direction_compliance",
    "ego_progress": "route_progress",
    "time_to_collision_within_bound": "interaction",
    "comfort": "comfort",
}
ALIASES = {"collision_avoidance": "collision", "anticipatory_interaction": "interaction"}


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Finite native metrics in [0,1] required")
    return value


def aggregate(manifest, rows, *, allow_partial=False):
    """Manifest declares the exact evaluation cohort, not only its size.

    Formal NAVSIM callers also verify this against the official scene filter.
    Missing/runtime-failed samples never become zero-score driving failures.
    """
    from driversi.rsi.decisions.scene_index import CONDITION_TAGS

    required = {
        "schema_version",
        "benchmark",
        "checkpoint",
        "evaluator_version",
        "expected_tokens",
        "allowed_logs",
        "scope",
    }
    if set(manifest) != required or manifest["schema_version"] != 1:
        raise ValueError("Versioned evaluation manifest required")
    if manifest["scope"] not in ("formal_benchmark", "bounded_probe"):
        raise ValueError("Explicit evaluation scope required")
    expected = manifest["expected_tokens"]
    if not isinstance(expected, list) or not expected or len(set(expected)) != len(expected):
        raise ValueError("Unique nonempty expected tokens required")
    if any(not isinstance(x, str) or not x for x in expected):
        raise ValueError("Invalid expected token")
    allowed = set(manifest["allowed_logs"])
    if not allowed or any(not isinstance(x, str) or not x for x in allowed):
        raise ValueError("Explicit evaluation log allowlist required")
    expected = set(expected)
    seen, failed, groups = set(), 0, set()
    counts, deficits, sums = Counter(), Counter(), Counter()
    cohorts = defaultdict(lambda: [0, 0])
    for row in rows:
        required = {"token", "source_log", "status", "metrics", "conditions"}
        if set(row) != required:
            raise ValueError("Canonical evaluation row required")
        token = row["token"]
        if token not in expected or token in seen or row["source_log"] not in allowed:
            raise ValueError("Unexpected/duplicate token or source outside evaluation manifest")
        seen.add(token)
        tags = row["conditions"]
        if (
            not isinstance(tags, list)
            or len(set(tags)) != len(tags)
            or not set(tags) <= set(CONDITION_TAGS)
        ):
            raise ValueError("Only fixed history condition tags may be aggregated")
        if row["status"] in ("failed", "outcome_unknown"):
            if row["metrics"] is not None:
                raise ValueError("Runtime failures must not masquerade as driving scores")
            failed += 1
            continue
        if row["status"] != "scored" or set(row["metrics"]) != {"score", *METRICS}:
            raise ValueError("Complete native score and submetrics required")
        values = {k: number(v) for k, v in row["metrics"].items()}
        groups.add(source_log_group(row["source_log"]))
        for metric, capability in METRICS.items():
            shortfall = int(values[metric] < 1.0)
            counts[capability] += 1
            deficits[capability] += shortfall
            sums[metric] += values[metric]
            # Single observable conditions keep slices bounded and interpretable.
            for tag in tags:
                cohorts[(capability, tag)][0] += 1
                cohorts[(capability, tag)][1] += shortfall
        sums["score"] += values["score"]
    missing = len(expected - seen)
    if missing and not allow_partial:
        raise ValueError("Evaluation coverage incomplete; explicit partial mode required")
    scored = len(seen) - failed
    feedback = validate_feedback(
        {
            "schema_version": 1,
            "benchmark": manifest["benchmark"],
            "checkpoint": manifest["checkpoint"],
            "evaluator_version": manifest["evaluator_version"],
            "slices": [
                {"capability": k, "count": counts[k], "failures": deficits[k]}
                for k in sorted(counts)
            ],
        }
    )
    return {
        "schema_version": 1,
        "kind": "evaluation_capability_aggregate",
        "capability_feedback": feedback,
        "cohorts": [
            {"capability": cap, "conditions": [tag], "count": count, "failures": failure}
            for (cap, tag), (count, failure) in sorted(cohorts.items())
        ],
        "coverage": {
            "expected": len(expected),
            "received": len(seen),
            "scored": scored,
            "runtime_failed_or_unknown": failed,
            "missing": missing,
            "complete": missing == 0,
            "all_scored": missing == 0 and failed == 0,
            "observed_source_groups": len(groups),
        },
        "mean_native_metrics": {k: v / scored for k, v in sums.items()} if scored else {},
        "scope": manifest["scope"],
        "evaluation_role": "adaptive_evaluation"
        if manifest["benchmark"] != "development"
        else "development",
        "shortfall_rule": "native_submetric_below_one_not_causal_failure",
        "counts_are_not_independent_trials": True,
        "trainable": False,
        "promoted": False,
    }


def from_weakness_review(weakness, *, evaluator_version):
    """Explicit aliases; ignore sub-resolution deltas and unmeasured hypotheses."""
    from driversi.rsi.decisions.capability_feedback import CAPABILITIES

    support = weakness.get("support", {}).get("paired_samples")
    if type(support) is not int or support <= 0:
        raise ValueError("Measured paired-sample support required")
    affected = defaultdict(set)
    for signal in weakness.get("signals", []):
        raw = signal.get("capability_hypothesis")
        cap = ALIASES.get(raw, raw)
        if cap not in CAPABILITIES:
            continue
        ids = set()
        if signal.get("candidate_compliance_shortfall") is True:
            ids.update(signal.get("deficient_sample_ids", []))
        if signal.get("actionable_from_this_change") is True:
            ids.update(signal.get("regressed_sample_ids", []))
        affected[cap].update(ids)
    return validate_feedback(
        {
            "schema_version": 1,
            "benchmark": "development",
            "checkpoint": weakness["checkpoint"],
            "evaluator_version": evaluator_version,
            "slices": [
                {"capability": cap, "count": support, "failures": len(ids)}
                for cap, ids in sorted(affected.items())
                if ids
            ],
        }
    )
