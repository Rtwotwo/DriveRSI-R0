import copy

import numpy as np
import pytest

from driversi.rsi.decisions.trajectory_coverage import family_coverage, pair_metrics
from driversi.rsi.decisions.trajectory_diversity import compare


def row(name, xy=None):
    if xy is None:
        xy = np.c_[np.arange(71) * 0.5, np.zeros(71)]
    return dict(
        candidate_id=name,
        comparison_key={"source": "same"},
        xy=xy.tolist(),
        initial_heading_rad=0.0,
        behavior_types=["unclassified"],
    )


def test_stop_then_catchup_is_visible_despite_identical_endpoint():
    a = row("a")
    bxy = np.array(a["xy"])
    bxy[21:61, 0] -= 0.4 * np.sin(np.linspace(0, np.pi, 40))
    b = row("b", bxy)
    metrics = pair_metrics(a["xy"], b["xy"], 0)
    assert metrics["fde_m"] == 0
    assert metrics["max_future_1s_mean_separation_m"] > metrics["ade_future_5s_m"]
    assert metrics["max_future_1s_mean_speed_difference_mps"] > 0
    assert not compare(b, [a])["distinct"]
    audit = family_coverage([a, b])
    assert audit["admission_effect"] == "none"
    assert not audit["distinct_samples_certified"]
    assert audit["v1_retained_in_input_order"] == ["a"]


def test_world_rotation_does_not_change_physical_metrics_or_grid():
    a = row("a")
    b = row("b", np.array(a["xy"]) + np.c_[np.zeros(71), np.linspace(0, 2, 71)])
    before = family_coverage([a, b])
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    changed = copy.deepcopy([a, b])
    for x in changed:
        x["xy"] = (np.array(x["xy"]) @ rotation.T + [42, -17]).tolist()
        x["initial_heading_rad"] = angle
    after = family_coverage(changed)
    for key, value in before["pairs"][0].items():
        if isinstance(value, float):
            assert after["pairs"][0][key] == pytest.approx(value, abs=1e-10)
    # Grid boundaries can be numerically sensitive; compare the half-cell origin here.
    assert before["grid_origin_sensitivity"][1] == after["grid_origin_sensitivity"][1]


def test_grid_counts_are_origin_sensitive_and_are_not_quota_certificates():
    a = row("a")
    b = row("b", np.array(a["xy"]) + np.c_[np.zeros(71), np.linspace(0, -0.01, 71)])
    audit = family_coverage([a, b])
    assert [g["anchor_2s_occupied_cells"] for g in audit["grid_origin_sensitivity"]] == [2, 1]
    assert audit["v1_retained_in_input_order"] == ["a"]
    assert not audit["causal_behavior_verified"]


def test_reference_provenance_and_corrupt_inputs():
    a, b = row("a"), row("b")
    del a["initial_heading_rad"]
    assert family_coverage([a, b])["status"] == "unavailable_missing_initial_heading"
    assert family_coverage([])["status"] == "empty"
    for mutate in [
        lambda x: x.update(comparison_key={"source": "other"}),
        lambda x: x.update(initial_heading_rad=float("nan")),
        lambda x: x.update(initial_heading_rad=0.2),
        lambda x: x["xy"][0].__setitem__(0, 20),
        lambda x: x["xy"][0].__setitem__(0, float("nan")),
    ]:
        a, b = row("a"), row("b")
        mutate(b)
        with pytest.raises(ValueError):
            family_coverage([a, b])
    with pytest.raises(ValueError):
        family_coverage([row("same"), row("same")])
