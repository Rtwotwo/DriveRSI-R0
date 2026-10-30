"""Log-level paired evidence for bounded experiments, never an auto-promotion rule.

The unit is the original recording, not adjacent scenes. Bootstrap intervals are
exploratory summaries: repeated adaptive testing and one training seed cannot
establish a general improvement claim. Native evaluator thresholds are untouched.
"""

import math
import random
from collections import defaultdict

from driversi.training.sampling.environment import source_log_group

POLICY = {
    "version": "paired-log-evidence-v1",
    "minimum_original_logs": 8,
    "bootstrap_draws": 4096,
    "bootstrap_seed": 42,
    "interval_mass": 0.95,
    "score_effect_floor": 1e-7,
}


def assess(rows, *, primary_metric, safety_metrics):
    """Assess already paired native scores without modifying their values."""
    if not rows or not primary_metric or primary_metric in safety_metrics:
        raise ValueError("Nonempty paired evidence and distinct primary metric required")
    seen, groups, regressions = set(), defaultdict(list), []
    required = {primary_metric, *safety_metrics}
    for row in rows:
        if set(row) != {"sample_id", "source_log", "control", "treatment"}:
            raise ValueError("Malformed paired evidence row")
        identity = row["sample_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Unique paired evidence identities required")
        seen.add(identity)
        if not isinstance(row["source_log"], str) or not row["source_log"]:
            raise ValueError("Original source log required")
        for side in ("control", "treatment"):
            if not required <= set(row[side]) or any(
                type(row[side][key]) not in (float, int) or not math.isfinite(row[side][key])
                for key in required
            ):
                raise ValueError("Complete finite paired native metrics required")
        delta = row["treatment"][primary_metric] - row["control"][primary_metric]
        groups[source_log_group(row["source_log"])].append(delta)
        regressions.extend(
            {"sample_id": identity, "metric": key}
            for key in safety_metrics
            if row["treatment"][key] < row["control"][key]
        )
    per_log = [
        {"source_group": key, "samples": len(values), "mean_delta": math.fsum(values) / len(values)}
        for key, values in sorted(groups.items())
    ]
    deltas = [row["mean_delta"] for row in per_log]
    n = len(deltas)
    enough = n >= POLICY["minimum_original_logs"]
    interval = None
    if enough:
        rng = random.Random(POLICY["bootstrap_seed"])
        draws = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(POLICY["bootstrap_draws"]))
        tail = (1 - POLICY["interval_mass"]) / 2
        interval = [draws[int(tail * (len(draws) - 1))], draws[int((1 - tail) * (len(draws) - 1))]]
    detectable = interval is not None and interval[0] > POLICY["score_effect_floor"]
    status = (
        "safety_regression"
        if regressions
        else "insufficient_independent_logs"
        if not enough
        else "positive_exploratory_evidence"
        if detectable
        else "gain_not_established"
    )
    return {
        "policy": dict(POLICY),
        "primary_metric": primary_metric,
        "samples": len(rows),
        "original_logs": n,
        "per_log": per_log,
        "sample_mean_delta": math.fsum(math.fsum(v) for v in groups.values()) / len(rows),
        "equal_log_mean_delta": math.fsum(deltas) / n,
        "log_bootstrap_interval": interval,
        "safety_regressions": regressions,
        "status": status,
        "ready_for_independent_replication": status == "positive_exploratory_evidence",
        "scope": "exploratory_single_training_seed_not_repeated_testing_validated",
        "promotion_authorized": False,
        "remaining_evidence": [
            "independent_training_seeds",
            "prespecified_target_and_guardrail_slices",
            "locked_promotion_holdout",
        ],
    }
