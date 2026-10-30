import numpy as np
import pytest

from driversi.contracts.road import LaneMatchError, RoadGraph, Route


def graph():
    return RoadGraph({"schema_version": 1, "map_name": "straight", "map_version": "test",
                      "origin_xyz": [0, 0, 0], "bounds_local": [-10, -10, 100, 100]}, {
        "lane:1": {"kind": "lane", "points": [[0, 0], [10, 0]], "outgoing": ["connector:2"], "speed_limit_mps": 8},
        "connector:2": {"kind": "connector", "points": [[10, 0], [20, 0]], "outgoing": ["lane:3", "lane:4"], "speed_limit_mps": None},
    })


def test_route_stops_before_branch_and_serializes(tmp_path):
    value = graph()
    path = tmp_path / "lanes.json"
    value.save(path)
    value = RoadGraph.load(path)
    route = value.route_from_pose(2, .5, 0)
    assert route.node_ids == ("lane:1", "connector:2")
    assert route.end_reason == "branch"
    assert route.project([3, 1]) == pytest.approx((3, 1, 0))
    np.testing.assert_allclose(route.at(15), [15, 0])
    assert not route.points.flags.writeable
    with pytest.raises(ValueError, match="direction"):
        value.route_from_pose(2, 0, np.pi)


def test_missing_exit_and_map_binding(straight_scene):
    value = graph()
    del value.nodes["connector:2"]
    assert value.route_from_pose(2, 0, 0).end_reason == "unloaded_exit"
    with pytest.raises(ValueError, match="version"):
        value.validate_scene(straight_scene)
    straight_scene.metadata["map_version"] = "test"
    value.validate_scene(straight_scene)
    with pytest.raises(ValueError, match="duplicate"):
        Route([[0, 0], [0, 0]])


def test_unique_topology_seam_selects_downstream_without_guessing_turn():
    value = graph()
    match = value.match_pose(10, .5, 0)
    assert match.status == "matched"
    assert match.reason == "continuous_topology_seam"
    assert match.selected_node_id == "connector:2"
    assert value.route_from_pose(10, .5, 0).end_reason == "branch"
    # The very same geometry is genuinely ambiguous if the upstream node branches.
    value.nodes["lane:1"]["outgoing"].append("connector:unloaded")
    assert value.match_pose(10, .5, 0).status == "ambiguous"


def test_shared_branch_prefix_requires_explicit_task_and_reports_candidates():
    value = graph()
    value.nodes = {
        "connector:straight": {"kind": "connector", "points": [[0, 0], [5, 0], [15, 0]],
                               "outgoing": ["lane:straight"], "speed_limit_mps": 8},
        "connector:left": {"kind": "connector", "points": [[0, 0], [5, 0], [10, 5]],
                           "outgoing": ["lane:left"], "speed_limit_mps": 8},
    }
    for nodes in (value.nodes, dict(reversed(list(value.nodes.items())))):
        value.nodes = nodes
        diagnostic = value.match_pose(2, .5, 0)
        assert diagnostic.status == "ambiguous"
        assert diagnostic.selected_node_id is None
        assert set(diagnostic.ambiguous_node_ids) == {"connector:straight", "connector:left"}
        assert diagnostic.to_dict()["candidates"][0]["distance_m"] == pytest.approx(.5)
        with pytest.raises(LaneMatchError) as rejected:
            value.route_from_pose(2, .5, 0)
        assert rejected.value.diagnostic == diagnostic
    chosen = value.route_from_pose(2, .5, 0, initial_node_id="connector:left")
    assert chosen.node_ids == ("connector:left",)
    assert chosen.end_reason == "unloaded_exit"
    assert value.match_pose(2, .5, 0, initial_node_id="connector:left").reason == "explicit_initial_node"
    with pytest.raises(LaneMatchError, match="Explicit"):
        value.route_from_pose(2, .5, 0, initial_node_id="lane:unknown")
    # An explicit task is not permission to assign a lane facing the other way.
    assert value.match_pose(2, .5, np.pi, initial_node_id="connector:left").status == "unmatched"


def test_seam_match_requires_endpoint_geometry_and_pose_proximity():
    value = graph()
    value.nodes["connector:2"]["points"] = [[10, .4], [20, .4]]
    assert value.match_pose(10, .2, 0).status == "ambiguous"
    # Coincident interiors do not establish a seam even with a graph edge.
    value.nodes["connector:2"]["points"] = [[0, 0], [10, 0]]
    assert value.match_pose(5, .5, 0).status == "ambiguous"


def test_all_near_candidates_are_checked_for_ambiguity():
    value = graph()
    # A closer, poorly aligned alternative must not hide a third equally plausible lane.
    value.nodes = {
        "lane:a": {"kind": "lane", "points": [[0, 0], [10, 0]], "outgoing": [], "speed_limit_mps": None},
        "lane:b": {"kind": "lane", "points": [[0, .1], [10, 3.1]], "outgoing": [], "speed_limit_mps": None},
        "lane:c": {"kind": "lane", "points": [[0, .2], [10, .2]], "outgoing": [], "speed_limit_mps": None},
    }
    diagnostic = value.match_pose(0, 0, 0)
    assert diagnostic.status == "ambiguous"
    assert set(diagnostic.ambiguous_node_ids) == {"lane:a", "lane:c"}


def test_unmatched_pose_has_machine_readable_diagnostics():
    value = graph()
    assert value.match_pose(101, 0, 0).reason == "outside_loaded_region"
    assert value.match_pose(2, 0, np.pi).reason == "no_compatible_lane"
    with pytest.raises(ValueError, match="node budget"):
        value.route_from_pose(2, 0, 0, max_nodes=1.5)


def test_pose_score_rejects_closer_but_misaligned_crossing_lane():
    value = graph()

    def baseline(distance, heading):
        direction = np.array([np.cos(heading), np.sin(heading)])
        closest = distance * np.array([-direction[1], direction[0]])
        return [closest-10*direction, closest+10*direction]

    value.nodes = {
        "connector:crossing": {"kind": "connector", "points": baseline(1.36, .90),
                               "outgoing": [], "speed_limit_mps": None},
        "connector:aligned": {"kind": "connector", "points": baseline(2.33, .116),
                              "outgoing": [], "speed_limit_mps": None},
    }
    diagnostic = value.match_pose(0, 0, 0)
    assert diagnostic.selected_node_id == "connector:aligned"
    assert diagnostic.candidates[0].score_m == pytest.approx(2.562)
    assert diagnostic.to_dict()["matching_policy"]["heading_weight_m_per_rad"] == 2
    # With equal heading, nearest parallel lane remains preferred.
    for key, distance in (("connector:crossing", 1.36), ("connector:aligned", 2.33)):
        value.nodes[key]["points"] = baseline(distance, 0)
    assert value.match_pose(0, 0, 0).selected_node_id == "connector:crossing"


def test_similar_scores_with_different_headings_remain_ambiguous():
    value = graph()
    value.nodes = {
        "lane:near": {"kind": "lane", "points": [[0, 0], [10, 0]],
                      "outgoing": [], "speed_limit_mps": None},
        "connector:far": {"kind": "connector", "points": [[-1, 1], [11, 1]],
                          "outgoing": [], "speed_limit_mps": None},
    }
    # A near lane rotated by 0.5 rad and a one-metre-offset aligned lane tie.
    value.nodes["lane:near"]["points"] = [[-10*np.cos(.5), -10*np.sin(.5)], [10*np.cos(.5), 10*np.sin(.5)]]
    assert value.match_pose(0, 0, 0).status == "ambiguous"


def branching_graph():
    value = graph()
    value.nodes.update({
        "lane:3": {"kind": "lane", "points": [[20, 0], [40, 0]], "outgoing": [], "speed_limit_mps": 8},
        "lane:4": {"kind": "lane", "points": [[20, 0], [30, 10]], "outgoing": ["lane:5"], "speed_limit_mps": 8},
        "lane:5": {"kind": "lane", "points": [[30, 10], [40, 20]], "outgoing": [], "speed_limit_mps": 8},
    })
    return value


def test_explicit_successor_consumes_true_branch_and_preserves_automatic_edges():
    value = branching_graph()
    route = value.route_from_pose(2, 0, 0, successors={"connector:2": "lane:4"})
    assert route.node_ids == ("lane:1", "connector:2", "lane:4", "lane:5")
    assert route.end_reason == "dead_end"
    assert value.route_from_pose(2, 0, 0).end_reason == "branch"


@pytest.mark.parametrize("choices, message", [
    ({"lane:1": "lane:3"}, "not adjacent"),
    ({"connector:2": "lane:missing"}, "missing node"),
    ({"lane:4": "lane:5"}, "Unused"),
])
def test_invalid_or_unused_successor_tasks_are_rejected(choices, message):
    with pytest.raises(ValueError, match=message):
        branching_graph().route_from_pose(2, 0, 0, successors=choices)


def test_successor_tasks_cannot_hide_geometry_budget_or_cycle():
    value = branching_graph()
    choice = {"connector:2": "lane:3"}
    with pytest.raises(ValueError, match="Unused"):
        value.route_from_pose(2, 0, 0, max_nodes=2, successors=choice)
    value.nodes["lane:3"]["points"] = [[23, 0], [40, 0]]
    with pytest.raises(ValueError, match="Unused"):
        value.route_from_pose(2, 0, 0, successors=choice)
    value.nodes["lane:3"].update(points=[[20, 0], [10, 0]], outgoing=["connector:2"])
    with pytest.raises(ValueError, match="cycle"):
        value.route_from_pose(2, 0, 0, successors=choice)
