import copy

import numpy as np

from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import RoadGraph, Route
from driversi.data.road_network.route_continuation import extend_ego_route


def fixture(scene):
    scene = copy.deepcopy(scene)
    scene.actor_ids = tuple("ego" if a == "a" else a for a in scene.actor_ids)
    scene.controlled[:] = False
    scene.controlled[0] = True
    scene.metadata["map_version"] = "fixture"
    scene = bind_driving_tasks(
        scene, {"ego": Route([[2, 0], [10, 0]], end_reason="source_support_end")}
    )
    nodes = {
        "lane:1": dict(
            kind="lane",
            points=[[0, 0], [20, 0]],
            outgoing=["connector:2"],
            speed_limit_mps=None,
            roadblock_id="1",
        ),
        "connector:2": dict(
            kind="connector",
            points=[[20, 0], [40, 0]],
            outgoing=[],
            speed_limit_mps=None,
            roadblock_id="2",
        ),
    }
    graph = RoadGraph(
        dict(
            schema_version=1,
            map_name=scene.metadata["map_name"],
            map_version="fixture",
            origin_xyz=scene.metadata["origin_xyz"],
            bounds_local=[-10, -10, 100, 100],
        ),
        nodes,
    )
    return scene, graph


def test_extension_preserves_source_and_reaches_declared_goal(straight_scene):
    scene, graph = fixture(straight_scene)
    before = copy.deepcopy(scene.metadata)
    result = extend_ego_route(scene, graph, ["1", "2"], [35, 0])
    assert scene.metadata == before
    np.testing.assert_array_equal(result.poses, scene.poses)
    task = result.metadata["driving_tasks"]["actors"]["ego"]
    assert task["endpoint"] == "destination"
    assert task["route"]["points"][:2] == [[2.0, 0.0], [10.0, 0.0]]
    assert task["route"]["points"][-1] == [35.0, 0.0]
    assert result.metadata["ego_route_continuation"]["status"] == "extended"


def test_extension_never_extrapolates_past_loaded_map(straight_scene):
    scene, graph = fixture(straight_scene)
    del graph.nodes["connector:2"]
    result = extend_ego_route(scene, graph, ["1", "2"], [35, 0])
    assert result.metadata["ego_route_continuation"]["end_reason"] == "map_boundary"
    assert result.metadata["driving_tasks"]["actors"]["ego"]["endpoint"] == "support_boundary"


def test_extension_cannot_select_undeclared_turn(straight_scene):
    scene, graph = fixture(straight_scene)
    result = extend_ego_route(scene, graph, ["1"], [35, 0])
    assert result.metadata["driving_tasks"]["actors"]["ego"]["route"]["points"][-1] == [20.0, 0.0]
    assert result.metadata["ego_route_continuation"]["end_reason"] == "navigation_end"


def test_ambiguous_branch_stays_supported_boundary(straight_scene):
    scene, graph = fixture(straight_scene)
    graph.nodes["connector:3"] = dict(
        kind="connector",
        points=[[20, 0], [40, 0.1]],
        outgoing=[],
        speed_limit_mps=None,
        roadblock_id="2",
    )
    graph.nodes["lane:1"]["outgoing"].append("connector:3")
    result = extend_ego_route(scene, graph, ["1", "2", "3"], None)
    assert result.metadata["ego_route_continuation"]["end_reason"] == "ambiguous_navigation_branch"
    assert not result.metadata["ego_route_continuation"]["destination"]


def test_repeated_navigation_block_uses_next_occurrence(straight_scene):
    scene, graph = fixture(straight_scene)
    graph.nodes["connector:2"]["roadblock_id"] = "revisited"
    result = extend_ego_route(scene, graph, ["revisited", "before", "1", "revisited"], None)
    assert result.metadata["ego_route_continuation"]["nodes"] == ["lane:1", "connector:2"]
    assert result.metadata["driving_tasks"]["actors"]["ego"]["route"]["points"][-1] == [40.0, 0.0]


def test_route_cannot_skip_intervening_navigation_block(straight_scene):
    scene, graph = fixture(straight_scene)
    result = extend_ego_route(scene, graph, ["1", "missing", "2"], None)
    assert result.metadata["ego_route_continuation"]["nodes"] == ["lane:1"]
    assert result.metadata["driving_tasks"]["actors"]["ego"]["endpoint"] == "support_boundary"


def test_navigation_resolves_only_near_tied_candidates_not_distant_lanes(straight_scene):
    scene, graph = fixture(straight_scene)
    graph.nodes["lane:wrong"] = dict(
        kind="lane",
        points=[[0, 0.05], [20, 0.05]],
        outgoing=[],
        speed_limit_mps=None,
        roadblock_id="wrong",
    )
    graph.nodes["lane:distant"] = dict(
        kind="lane", points=[[0, 2], [20, 2]], outgoing=[], speed_limit_mps=None, roadblock_id="2"
    )
    result = extend_ego_route(scene, graph, ["1", "2"], None)
    assert result.metadata["ego_route_continuation"]["matching"]["selected_node_id"] == "lane:1"
    assert result.metadata["ego_route_continuation"]["extended_length_m"] > 30


def goal_fork(scene):
    scene, graph = fixture(scene)
    graph.nodes["connector:dead"] = dict(
        kind="connector",
        points=[[20, 0], [40, 0.1]],
        outgoing=[],
        speed_limit_mps=None,
        roadblock_id="2",
    )
    graph.nodes["lane:1"]["outgoing"].append("connector:dead")
    graph.nodes["connector:2"]["outgoing"] = ["lane:goal"]
    graph.nodes["lane:goal"] = dict(
        kind="lane",
        points=[[40, 0], [60, 0]],
        outgoing=[],
        speed_limit_mps=None,
        roadblock_id="3",
    )
    return scene, graph


def test_ambiguous_heading_uses_unique_ordered_goal_support(straight_scene):
    scene, graph = goal_fork(straight_scene)
    result = extend_ego_route(scene, graph, ["1", "2", "3"], [55, 0])
    evidence = result.metadata["ego_route_continuation"]
    assert evidence["destination"]
    assert evidence["nodes"] == ["lane:1", "connector:2", "lane:goal"]
    assert evidence["decisions"][0]["goal_support"] == {
        "connector:2": True,
        "connector:dead": False,
    }
    np.testing.assert_array_equal(result.poses, scene.poses)


def test_goal_support_does_not_guess_when_alternative_map_missing(straight_scene):
    scene, graph = goal_fork(straight_scene)
    graph.nodes["connector:dead"]["outgoing"] = ["lane:missing"]
    result = extend_ego_route(scene, graph, ["1", "2", "3"], [55, 0])
    assert result.metadata["ego_route_continuation"]["end_reason"] == "ambiguous_navigation_branch"


def test_goal_support_selects_parallel_proven_lanes(straight_scene):
    scene, graph = goal_fork(straight_scene)
    graph.nodes["connector:dead"]["outgoing"] = ["lane:goal"]
    result = extend_ego_route(scene, graph, ["1", "2", "3"], [55, 0])
    assert (
        result.metadata["ego_route_continuation"]["decisions"][0]["selection_rule"]
        == "parallel_navigation_minimum_turn"
    )


def test_goal_support_rejects_unordered_shortcut(straight_scene):
    scene, graph = goal_fork(straight_scene)
    result = extend_ego_route(scene, graph, ["1", "2", "intervening", "3"], [55, 0])
    assert result.metadata["ego_route_continuation"]["end_reason"] == "ambiguous_navigation_branch"


def test_no_goal_uses_unique_ordered_navigation_support(straight_scene):
    scene, graph = goal_fork(straight_scene)
    result = extend_ego_route(scene, graph, ["1", "2", "3"], None)
    evidence = result.metadata["ego_route_continuation"]
    assert evidence["nodes"] == ["lane:1", "connector:2", "lane:goal"]
    assert evidence["decisions"][0]["navigation_support"] == {
        "connector:2": True,
        "connector:dead": False,
    }
    assert not evidence["destination"]
    np.testing.assert_array_equal(result.poses, scene.poses)


def test_no_goal_keeps_missing_routes_unknown_and_resolves_parallel_lanes(straight_scene):
    for target in ("lane:missing", "lane:goal"):
        scene, graph = goal_fork(straight_scene)
        graph.nodes["connector:dead"]["outgoing"] = [target]
        result = extend_ego_route(scene, graph, ["1", "2", "3"], None)
        evidence = result.metadata["ego_route_continuation"]
        if target == "lane:missing":
            assert evidence["end_reason"] == "ambiguous_navigation_branch"
        else:
            assert evidence["decisions"][0]["selection_rule"] == "parallel_navigation_minimum_turn"


def test_parallel_choice_does_not_merge_different_road_maneuvers(straight_scene):
    from driversi.data.road_network.route_continuation import _parallel_lane_choice

    scene, g = goal_fork(straight_scene)
    g.nodes["connector:dead"]["outgoing"] = ["lane:other"]
    g.nodes["lane:other"] = dict(g.nodes["lane:goal"], roadblock_id="different")
    candidates = [(0.0, "connector:2"), (0.001, "connector:dead")]
    supported = {x: True for _, x in candidates}
    assert _parallel_lane_choice(g, candidates, supported)[0] is None
    g.nodes["lane:other"]["roadblock_id"] = "3"
    g.nodes["connector:dead"]["points"] = [[20, 0], [30, 0], [30, 10]]
    assert _parallel_lane_choice(g, candidates, supported)[0] is None


def test_parallel_choice_is_invariant_to_map_dictionary_order(straight_scene):
    scene, g = goal_fork(straight_scene)
    g.nodes["connector:dead"]["outgoing"] = ["lane:goal"]
    a = extend_ego_route(scene, g, ["1", "2", "3"], None)
    g.nodes["lane:1"]["outgoing"].reverse()
    b = extend_ego_route(scene, g, ["1", "2", "3"], None)
    assert a.metadata["driving_tasks"] == b.metadata["driving_tasks"]


def test_known_last_route_node_resolves_lane_connector_seam(straight_scene):
    scene, graph = fixture(straight_scene)
    scene = bind_driving_tasks(
        scene,
        {"ego": Route([[2, 0], [19.9, 0]], node_ids=("lane:1",), end_reason="source_support_end")},
    )
    graph.nodes["lane:near"] = {**graph.nodes["lane:1"], "points": [[0, 0.05], [20, 0.05]]}
    result = extend_ego_route(scene, graph, ["1", "2"], None)
    ev = result.metadata["ego_route_continuation"]
    assert ev["matching"]["selected_node_id"] == "lane:1"
    assert result.metadata["driving_tasks"]["actors"]["ego"]["route"]["points"][-1] == [40.0, 0.0]


def test_loaded_parallel_maneuver_continues_until_actual_map_boundary(straight_scene):
    scene, graph = goal_fork(straight_scene)
    graph.nodes["connector:dead"]["outgoing"] = ["lane:parallel"]
    graph.nodes["lane:parallel"] = dict(
        graph.nodes["lane:goal"], points=[[40, 0.1], [60, 0.1]], outgoing=["connector:missing"]
    )
    graph.nodes["lane:goal"]["outgoing"] = ["connector:missing"]
    result = extend_ego_route(scene, graph, ["1", "2", "3", "4", "5"], None)
    evidence = result.metadata["ego_route_continuation"]
    assert evidence["end_reason"] == "map_boundary"
    assert evidence["decisions"][0]["navigation_support"] == {
        "connector:2": None,
        "connector:dead": None,
    }
    assert evidence["decisions"][0]["selection_rule"] == "parallel_navigation_minimum_turn"
    task = result.metadata["driving_tasks"]["actors"]["ego"]
    assert task["endpoint"] == "support_boundary"
    assert task["route"]["points"][-1] == [60.0, 0.0]
    np.testing.assert_array_equal(result.poses, scene.poses)


def test_parallel_maneuver_does_not_choose_between_unknown_and_proven_dead_end(straight_scene):
    scene, graph = goal_fork(straight_scene)
    graph.nodes["connector:dead"]["outgoing"] = ["lane:parallel"]
    graph.nodes["lane:parallel"] = dict(graph.nodes["lane:goal"], points=[[40, 0.1], [60, 0.1]])
    graph.nodes["lane:goal"]["outgoing"] = ["connector:missing"]
    result = extend_ego_route(scene, graph, ["1", "2", "3", "4", "5"], None)
    assert result.metadata["ego_route_continuation"]["end_reason"] == "ambiguous_navigation_branch"


def test_explicit_bounded_profile_uses_only_local_non_dead_end_without_goal_claim(straight_scene):
    scene, graph = goal_fork(straight_scene)
    graph.nodes["connector:dead"]["outgoing"] = ["lane:parallel"]
    graph.nodes["lane:parallel"] = dict(graph.nodes["lane:goal"], points=[[40, 0.1], [60, 0.1]])
    graph.nodes["lane:goal"]["outgoing"] = ["connector:missing"]
    result = extend_ego_route(
        scene, graph, ["1", "2", "3", "4", "5"], None, support_profile="bounded_navigation_v2"
    )
    ev = result.metadata["ego_route_continuation"]
    assert ev["end_reason"] == "map_boundary"
    assert ev["nodes"] == ["lane:1", "connector:2", "lane:goal"]
    assert ev["decisions"][0]["selection_rule"] == "unique_locally_supported_navigation"
    assert not ev["destination"]
    np.testing.assert_array_equal(result.poses, scene.poses)


def test_initial_navigation_resolves_only_matching_declared_branch(straight_scene):
    from driversi.data.road_network.route_continuation import initial_navigation_routes

    scene, graph = fixture(straight_scene)
    graph.nodes["lane:other"] = dict(
        graph.nodes["lane:1"], roadblock_id="other", points=[[0, 0.05], [20, 0.05]]
    )
    routes, audit = initial_navigation_routes(scene, graph, ["1", "2"])
    assert routes["ego"].node_ids == ("lane:1",)
    assert audit["ego"]["chosen_initial_node"] == "lane:1"


def test_initial_navigation_near_exit_uses_real_adjacent_support(straight_scene):
    from driversi.data.road_network.route_continuation import initial_navigation_routes

    scene, graph = fixture(straight_scene)
    scene.poses[0, 0, :2] = [19.0, 0.0]
    scene.poses[0, 0, 3] = 0.0
    routes, _ = initial_navigation_routes(scene, graph, ["1", "2"])
    assert routes["ego"].node_ids == ("lane:1", "connector:2")
    assert routes["ego"].points[-1].tolist() == [40.0, 0.0]
