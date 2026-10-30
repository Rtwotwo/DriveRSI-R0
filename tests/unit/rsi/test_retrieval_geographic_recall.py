import math

import pytest

from driversi.rsi.experiments.retrieval_geographic_recall import geographic_truth, metrics


def test_complete_ground_truth_filters_city_and_heading():
    poses = {
        "q": dict(token="q", split="navtest", map="a", xy=[0, 0], yaw=0),
        "near": dict(token="near", split="navtrain", map="a", xy=[3, 4], yaw=0),
        "opposite": dict(token="opposite", split="navtrain", map="a", xy=[0, 0], yaw=math.pi),
        "far": dict(token="far", split="navtrain", map="a", xy=[10, 0], yaw=0),
        "other_city": dict(token="other_city", split="navtrain", map="b", xy=[0, 0], yaw=0),
    }
    assert geographic_truth(["q"], poses, 5, 30) == {"q": {"near"}}
    assert geographic_truth(["q"], poses, 5, None) == {"q": {"near", "opposite"}}


def test_macro_recall_and_hit_are_distinct():
    result, _ = metrics({"a": {"x"}, "b": {"z"}}, {"a": {"x", "y", "w"}, "b": {"z"}})
    assert result["hit_fraction_all_queries"] == 1
    assert result["macro_recall"] == pytest.approx(2 / 3)
    assert result["micro_recall"] == .5


def test_intersection_empty_is_not_perfect_precision():
    result, _ = metrics({"a": set(), "b": {"z"}}, {"a": {"x"}, "b": {"z"}})
    assert result["macro_precision_empty_is_zero"] == .5
    assert result["micro_precision"] == 1
    assert result["macro_recall"] == .5
    assert result["empty_retrieval_queries"] == 1


def test_empty_truth_explicit_not_counted_as_perfect_recall():
    result, _ = metrics({"a": {"x"}, "b": {"z"}}, {"a": set(), "b": {"z"}})
    assert result["queries_without_ground_truth"] == 1
    assert result["macro_recall"] == 1
    assert result["hit_fraction_all_queries"] == .5
    assert result["hit_fraction_eligible_queries"] == 1
