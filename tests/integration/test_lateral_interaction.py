import copy

import numpy as np
import pytest

from driversi.contracts.road import Route
from driversi.evaluation.benchmarks.lateral import blend_lane_change, run_lateral_pair


def case(scene):
    scene = copy.deepcopy(scene)
    times = np.arange(91) * 0.1
    scene.times = times
    scene.poses = np.zeros((91, 3, 4))
    scene.velocities = np.zeros((91, 3, 3))
    scene.poses[:, :, 0] = times[:, None] * 6 + np.array([0, 18, -20])
    scene.poses[:, :, 1] = [0, 3.5, -8]
    scene.velocities[:, :, 0] = 6
    scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 3.5]}
    routes = {"a": Route([[-40, 0], [140, 0]]), "b": Route([[-40, 3.5], [140, 3.5]])}
    return scene, routes


def test_lane_change_is_action_executed_and_response_is_causal(straight_scene):
    scene, routes = case(straight_scene)
    candidate = blend_lane_change(routes["b"], routes["a"], [24, 3.5], length_m=30)
    report, records, _ = run_lateral_pair(scene, routes, candidate, challenger="b", responder="a")
    assert report["common_prefix_equal"] and report["causal_action_order"]
    assert "route_tracking_bound" not in report["baseline"]["bounds_failures"]
    assert report["first_challenger_action_difference_tick"] >= 10
    assert (
        report["first_responder_action_difference_tick"]
        > report["first_challenger_action_difference_tick"]
    )
    states = np.asarray(records[1].states)
    assert np.max(np.linalg.norm(np.diff(states[:, 1, :2], axis=0), axis=1)) < 1.5
    assert abs(states[-1, 1, 1] - 3.5) > 1
    assert not report["trainable"]
    assert report["task_semantics"]["authoritative_tasks"] == "frozen_identically_across_branches"
    assert report["maneuver_validation_pass"] == bool(
        report["interaction_evidence"]
        and not report["baseline"]["bounds_failures"]
        and report["baseline"]["ticks"] == report["requested_ticks"]
        and report["challenge"]["ticks"] == report["requested_ticks"]
        and report["challenge"]["maneuver_completed"]
    )


def test_same_route_override_is_negative_control(straight_scene):
    scene, routes = case(straight_scene)
    report, records, _ = run_lateral_pair(
        scene, routes, routes["b"], challenger="b", responder="a", steps=30
    )
    assert report["first_challenger_action_difference_tick"] is None
    assert report["first_responder_action_difference_tick"] is None
    assert not report["interaction_evidence"]
    assert not report["maneuver_validation_pass"]
    np.testing.assert_array_equal(records[0].states, records[1].states)


def test_invalid_geometry_and_heldout_rejected(straight_scene):
    scene, routes = case(straight_scene)
    with pytest.raises(ValueError, match="aligned"):
        blend_lane_change(routes["a"], Route([[-40, 20], [140, 20]]), [0, 0])
    with pytest.raises(ValueError, match="support"):
        blend_lane_change(routes["b"], routes["a"], [130, 3.5])
    scene.metadata["split"] = "navtest"
    with pytest.raises(ValueError, match="Official"):
        run_lateral_pair(scene, routes, routes["b"], challenger="b", responder="a")


def test_merge_requires_shared_downstream_and_runs_action_dynamics(straight_scene):
    scene, routes = case(straight_scene)
    routes["a"] = Route(routes["a"].points, ("lane:main", "lane:shared"))
    candidate = blend_lane_change(routes["b"], routes["a"], [24, 3.5], length_m=30)
    report, records, _ = run_lateral_pair(
        scene, routes, candidate, challenger="b", responder="a", maneuver="merge"
    )
    assert report["maneuver"] == "merge" and report["causal_action_order"]
    assert report["first_responder_action_difference_tick"] is not None
    assert len(records[1].actions) > 10
    with pytest.raises(ValueError, match="downstream"):
        run_lateral_pair(
            scene, routes, Route(candidate.points), challenger="b", responder="a", maneuver="merge"
        )


@pytest.mark.parametrize("maneuver", ["cut_in", "merge"])
def test_complete_maneuver_requires_safe_baseline_and_supported_map(straight_scene, maneuver):
    scene, routes = case(straight_scene)
    routes["a"] = Route(routes["a"].points, ("lane:main", "lane:shared"))
    candidate = blend_lane_change(routes["b"], routes["a"], [24, 3.5], length_m=30)
    clipped, _, _ = run_lateral_pair(scene, routes, candidate, challenger="b", responder="a", maneuver=maneuver)
    assert clipped["challenge"]["maneuver_completed"]
    assert not clipped["maneuver_validation_pass"]
    assert "offroad_in_world" in clipped["baseline"]["bounds_failures"]
    # Synthetic map coverage must extend beyond the full 9-second maneuver.
    scene.road_points[scene.road_points[:, 0] == 60, 0] = 160
    supported, _, _ = run_lateral_pair(scene, routes, candidate, challenger="b", responder="a", maneuver=maneuver)
    assert supported["maneuver_validation_pass"]
    assert supported["baseline"]["ticks"] == supported["challenge"]["ticks"] == 90
    assert supported["challenge"]["pair_valid_seconds"] == 9
