import json

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_height import ReferenceTrackHeight


def test_analytic_slope_spatial_query_and_zero_displacement():
    profile = ReferenceTrackHeight([[0, 0], [10, 0], [20, 0]], [7, 8, 9])
    result = profile.correction([4, 1.5], [16, 0])
    assert result["supported"]
    assert result["correction_m"] == pytest.approx(-1.2)
    assert result["target_projection"]["height_m"] == pytest.approx(7.4)
    assert result["target_projection"]["distance_m"] == 1.5
    assert result["source_projection"]["segment_index"] == 1
    assert profile.correction([12, .3], [12, .3])["correction_m"] == 0.
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_height_offset_cancels_and_constructor_owns_samples():
    xy = np.array([[0., 0], [10, 0]])
    z = np.array([5., 6])
    first = ReferenceTrackHeight(xy, z)
    shifted = ReferenceTrackHeight(xy, z+100)
    xy[:] = 999
    z[:] = 999
    assert first.correction([2, 0], [8, 0])["correction_m"] == pytest.approx(-.6)
    assert shifted.correction([2, 0], [8, 0])["correction_m"] == pytest.approx(-.6)


@pytest.mark.parametrize(("query", "reason"), [
    ([5, 2.01], "target_outside_reference_corridor"),
    ([-.01, 0], "target_outside_reference_extent"),
    ([10.01, 0], "target_outside_reference_extent"),
])
def test_unsupported_query_returns_explicit_zero_without_extrapolation(query, reason):
    result = ReferenceTrackHeight([[0, 0], [10, 0]], [0, 1]).correction(query, [5, 0])
    assert not result["supported"]
    assert result["reason"] == reason
    assert result["correction_m"] == 0
    assert result["fallback"] == "retain_source_time_height"
    json.dumps(result, allow_nan=False)


def test_source_query_must_also_have_support():
    result = ReferenceTrackHeight([[0, 0], [10, 0]], [0, 1]).correction([5, 0], [-1, 0])
    assert result["reason"] == "source_outside_reference_extent"
    assert not result["supported"]


def test_exact_endpoints_and_numerical_pose_roundoff_are_supported():
    profile = ReferenceTrackHeight([[0, 0], [10, 0]], [0, 1])
    assert profile.correction([10, 0], [0, 0])["correction_m"] == 1
    assert profile.correction([-1e-6, 0], [0, 0])["supported"]


def test_stationary_track_cannot_invent_spatial_height():
    for xy, z in [([[2, 2]], [3]), ([[2, 2], [2, 2]], [3, 4])]:
        result = ReferenceTrackHeight(xy, z).correction([2, 2], [2, 2])
        assert not result["supported"]
        assert result["reason"] == "source_no_moving_reference_segment"


def test_identical_stationary_samples_do_not_break_valid_motion():
    profile = ReferenceTrackHeight([[0, 0], [0, 0], [10, 0], [10, 0]], [5, 5, 6, 6])
    assert profile.correction([2, 0], [8, 0])["correction_m"] == pytest.approx(-.6)


def test_stationary_height_disagreement_is_exposed_not_averaged():
    profile = ReferenceTrackHeight([[0, 0], [0, 0], [10, 0]], [0, .3, 1])
    result = profile.correction([0, 0], [5, 0])
    assert result["reason"] == "target_ambiguous_reference_height"
    assert not result["supported"]


def test_crossing_reference_segments_with_different_heights_are_ambiguous():
    profile = ReferenceTrackHeight([[-5, -5], [5, 5], [-5, 5], [5, -5]], [0, 0, 1, 1])
    result = profile.correction([0, 0], [-4, -4])
    assert result["reason"] == "target_ambiguous_reference_height"
    assert result["correction_m"] == 0


def test_same_height_crossing_is_safe_for_height_only():
    profile = ReferenceTrackHeight([[-5, -5], [5, 5], [-5, 5], [5, -5]], [2, 2, 2, 2])
    assert profile.correction([0, 0], [-4, -4])["supported"]


def test_large_sample_gap_does_not_interpolate_or_choose_farther_segment():
    profile = ReferenceTrackHeight([[0, 0], [100, 0]], [0, 1])
    result = profile.correction([50, 0], [10, 0])
    assert not result["supported"]
    assert result["reason"] == "source_reference_segment_gap"


def test_excessive_grade_and_correction_are_explicit_fallbacks():
    steep = ReferenceTrackHeight([[0, 0], [10, 0]], [0, 3])
    assert steep.correction([1, 0], [2, 0])["reason"] == "source_reference_grade_exceeded"
    long = ReferenceTrackHeight([[0, 0], [10, 0], [20, 0]], [0, 2, 4])
    result = long.correction([0, 0], [20, 0])
    assert result["reason"] == "correction_limit_exceeded"
    assert result["correction_m"] == 0 and not result["supported"]


@pytest.mark.parametrize(("xy", "z"), [
    ([], []), ([[1, 2, 3]], [1]), ([[1, 2]], [1, 2]),
    ([[1, np.nan]], [1]), ([[1, 2]], [np.inf]),
])
def test_malformed_profile_is_rejected(xy, z):
    with pytest.raises(ValueError):
        ReferenceTrackHeight(xy, z)


@pytest.mark.parametrize("name", ["max_lateral_m", "max_segment_m", "max_grade", "max_correction_m"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True])
def test_malformed_limits_are_rejected(name, value):
    with pytest.raises(ValueError):
        ReferenceTrackHeight([[0, 0], [1, 0]], [0, 0], **{name: value})


@pytest.mark.parametrize("query", [[1], [1, 2, 3], [1, np.inf]])
def test_malformed_query_is_rejected(query):
    with pytest.raises(ValueError):
        ReferenceTrackHeight([[0, 0], [1, 0]], [0, 0]).correction(query, [0, 0])


def test_configuration_is_json_safe_detached_and_complete():
    profile = ReferenceTrackHeight([[0, 0], [10, 0]], [0, 1], max_lateral_m=1.5)
    config = profile.configuration
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    assert config["ambiguity_distance_m"] == .05
    assert config["ambiguity_height_m"] == .1
    assert config["max_lateral_m"] == 1.5
    config["max_lateral_m"] = 100
    assert profile.configuration["max_lateral_m"] == 1.5
