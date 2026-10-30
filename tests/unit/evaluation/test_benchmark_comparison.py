import copy

import pytest

from driversi.evaluation.metrics.benchmark_comparison import compare_official_results


def data():
    return (
        [dict(scenario_name=x, succeeded=True) for x in ["a", "b"]],
        [dict(scenario=x, num_scenarios=None, score=0.9, safety=1.0) for x in ["a", "b"]]
        + [dict(scenario="final_score", num_scenarios=2, score=0.9)],
    )


def test_complete_official_population_comparison():
    runner, scores = data()
    result = compare_official_results(["a", "b"], runner, scores, 95.0, required_metrics={"safety"})
    assert result == dict(problems=[], score=90.0, delta=-5.0, status="complete")


@pytest.mark.parametrize(
    "defect",
    [
        "missing_runner",
        "duplicate_runner",
        "failure",
        "string_status",
        "missing_score",
        "duplicate_score",
        "unknown_token",
        "nonfinite",
        "denominator",
        "final_nan",
    ],
)
def test_no_success_only_or_mismatched_comparison(defect):
    runner, scores = copy.deepcopy(data())
    if defect == "missing_runner":
        runner.pop()
    if defect == "duplicate_runner":
        runner.append(runner[0])
    if defect == "failure":
        runner[0]["succeeded"] = False
    if defect == "string_status":
        runner[0]["succeeded"] = "True"
    if defect == "missing_score":
        scores.pop(0)
    if defect == "duplicate_score":
        scores.insert(0, scores[0])
    if defect == "unknown_token":
        scores[0]["scenario"] = "other"
    if defect == "nonfinite":
        scores[0]["score"] = float("nan")
    if defect == "denominator":
        scores[-1]["num_scenarios"] = 1
    if defect == "final_nan":
        scores[-1]["score"] = float("nan")
    result = compare_official_results(["a", "b"], runner, scores, 95.0, required_metrics={"safety"})
    assert result["status"] == "incomplete"
    assert result["score"] is None and result["delta"] is None


def test_population_must_be_fixed_and_unique():
    runner, scores = data()
    with pytest.raises(ValueError):
        compare_official_results(["a", "a"], runner, scores, 95.0, required_metrics={"safety"})


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_finite_aggregate_cannot_hide_missing_safety_component(missing):
    runner, scores = data()
    scores[0]["safety"] = missing
    result = compare_official_results(["a", "b"], runner, scores, 95.0, required_metrics={"safety"})
    assert result["score"] is None
    assert "missing_or_invalid_required_metric_components" in result["problems"]


def test_component_comparison_requires_complete_population():
    from driversi.evaluation.metrics.benchmark_comparison import component_comparison
    rows = [{"scenario": "a", "num_scenarios": None, "score": .8, "metric": .7},
            {"scenario": "final_score", "num_scenarios": 1, "score": .8, "metric": .7}]
    runner = [{"scenario_name": "a", "succeeded": True}]
    assert component_comparison(["a"], runner, rows, {"metric": 72.}, 81.) == {
        "metric": {"paper": 72., "actual": 70., "delta": -2.}}
    assert component_comparison(["a", "b"], runner, rows, {"metric": 72.}, 81.) is None
