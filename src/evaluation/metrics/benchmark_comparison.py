"""Require a complete, unique fixed population before reporting benchmark scores."""

import math
from collections import Counter


def compare_official_results(
    expected_tokens, runner_rows, score_rows, paper_score, *, required_metrics
):
    expected = list(expected_tokens)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected benchmark population must be nonempty and unique")
    expected = set(expected)
    required_metrics = set(required_metrics)
    if not required_metrics:
        raise ValueError("Official required score components must be declared")
    problems = []
    names = [str(row["scenario_name"]) for row in runner_rows]
    if set(names) != expected or any(n != 1 for n in Counter(names).values()):
        problems.append("runner_token_set_incomplete_or_duplicate")
    # Do not accept truthy strings or missing statuses as successful execution.
    if any(row.get("succeeded") is not True for row in runner_rows):
        problems.append("runner_failures_or_invalid_status")
    per_scene = [row for row in score_rows if row.get("num_scenarios") is None]
    names = [str(row["scenario"]) for row in per_scene]
    if set(names) != expected or any(n != 1 for n in Counter(names).values()):
        problems.append("scored_token_set_incomplete_or_duplicate")

    def score_ok(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        )

    if any(not score_ok(row.get("score")) for row in per_scene):
        problems.append("nonfinite_or_out_of_range_scenario_scores")
    if any(not score_ok(row.get(metric)) for row in per_scene for metric in required_metrics):
        problems.append("missing_or_invalid_required_metric_components")
    final = [row for row in score_rows if row.get("scenario") == "final_score"]
    if len(final) != 1 or final[0].get("num_scenarios") != len(expected):
        problems.append("invalid_aggregate_denominator")
    if len(final) == 1 and not score_ok(final[0].get("score")):
        problems.append("nonfinite_or_out_of_range_score")
    value = final[0]["score"] * 100 if not problems else None
    return dict(
        problems=problems,
        score=value,
        delta=None if value is None else value - paper_score,
        status="complete" if not problems else "incomplete",
    )


def component_comparison(expected, runner_rows, score_rows, paper_metrics, paper_score):
    """Compare official aggregate components only after the complete population passes."""
    check = compare_official_results(expected, runner_rows, score_rows, paper_score,
                                     required_metrics=paper_metrics)
    if check["status"] != "complete":
        return None
    final = next(row for row in score_rows if row.get("scenario") == "final_score")
    result = {}
    for key, reference in paper_metrics.items():
        value = final.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid official aggregate component: " + key)
        result[key] = {"paper": reference, "actual": 100*value, "delta": 100*value-reference}
    return result
