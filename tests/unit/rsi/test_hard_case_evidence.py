import math
import pytest

from driversi.rsi.experiments.hard_case_evidence import eligible_members, check_replay, relative_xy


def member(token, role="development", command="GO STRAIGHT", node="following_speed_check"):
    return dict(token=token, case_id=token, role=role,
                skill=dict(id=node, evidence=dict(navigation_command=command)))


def score(token, ttc, arm="incumbent"):
    return dict(token=token, arm=arm, seed=42, metrics={"TTC": ttc})


def test_targets_only_incumbent_failures_development_and_correct_node():
    prepared = dict(members=[member("a"), member("b", role="validation"), member("c"),
        member("d", command="STOP"), member("e", node="turning_check"), member("f")])
    rows = [score("a", 0), score("b", 0), score("c", 1), score("c", 0, "native"),
            score("d", 0), score("e", 0), score("f", 0, "native")]
    assert [m["token"] for m in eligible_members(prepared, rows)] == ["a"]


def test_selection_order_does_not_rank_by_validation_or_score_gain():
    prepared = dict(members=[member("b"), member("a")])
    assert [m["token"] for m in eligible_members(prepared, [score("b", 0), score("a", 0)])] == ["a", "b"]


def test_original_native_metrics_required_including_small_changes():
    baseline = {k: 1.0 for k in ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C")}
    check_replay(baseline, dict(baseline))
    for metric in baseline:
        with pytest.raises(ValueError, match="Native replay differs"):
            check_replay(baseline, {**baseline, metric: baseline[metric] - 1e-6})
    with pytest.raises(ValueError):
        check_replay(baseline, {**baseline, "PDMS": float("nan")})


def test_relative_coordinates_are_anchor_frame_not_world_location():
    x, y = relative_xy(11, 22, [10, 20, math.pi / 2])
    assert x == pytest.approx(2)
    assert y == pytest.approx(-1)
