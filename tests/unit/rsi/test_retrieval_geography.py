import math

import pytest

from driversi.rsi.experiments.retrieval_geography import pose_distance, summarize


def test_same_map_metres_and_wrapped_heading():
    a = dict(map="boston", xy=[0, 0], yaw=math.radians(179))
    b = dict(map="boston", xy=[3, 4], yaw=math.radians(-179))
    result = pose_distance(a, b)
    assert result["distance_m"] == 5
    assert result["heading_difference_deg"] == pytest.approx(2)


def test_cross_map_is_not_near_even_with_identical_xy():
    a = dict(map="boston", xy=[0, 0], yaw=0)
    b = dict(map="singapore", xy=[0, 0], yaw=0)
    assert pose_distance(a, b) == dict(same_map=False, distance_m=None, heading_difference_deg=None)


def test_invalid_pose_rejected():
    with pytest.raises(ValueError):
        pose_distance(dict(map="x", xy=[0, float("nan")], yaw=0), dict(map="x", xy=[0, 0], yaw=0))


def test_query_denominators_include_cross_map_and_heading():
    pairs = [dict(query="a", same_map=True, distance_m=10, heading_difference_deg=180),
             dict(query="b", same_map=False, distance_m=None, heading_difference_deg=None)]
    result = summarize(pairs, 2)
    assert result["same_map_pair_fraction"] == .5
    assert result["same_map_nearest_median_m"] == 10
    assert result["within_radius"]["50"] == dict(pairs=1, queries=1, queries_heading_30deg=0)
    with pytest.raises(ValueError):
        summarize(pairs, 3)
