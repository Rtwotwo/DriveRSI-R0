import json

import numpy as np
import pytest

from driversi.agents.traffic_control.routing import RouteResolutionError, inspect_routes, resolve_routes
from driversi.contracts.road import RoadGraph
from driversi.contracts.route_tasks import RouteTasks


def test_task_identity_and_actor_ownership(straight_scene):
    tasks = RouteTasks("fixture-straight", "synthetic", {"a": "lane:1"})
    tasks.validate_scene(straight_scene)
    with pytest.raises(ValueError, match="different source"):
        RouteTasks("other", "synthetic", {}).validate_scene(straight_scene)
    with pytest.raises(ValueError, match="controlled actors"):
        RouteTasks("fixture-straight", "synthetic", {"replay": "lane:1"}).validate_scene(straight_scene)
    with pytest.raises(ValueError, match="node IDs"):
        RouteTasks("fixture-straight", "synthetic", {"a": "unknown"}).validate_scene(straight_scene)
    with pytest.raises(ValueError, match="controlled actors"):
        RouteTasks("fixture-straight", "synthetic", successors={"replay": {"lane:1": "lane:2"}}).validate_scene(straight_scene)


def task_graph(scene):
    scene.metadata["map_version"] = "fixture"
    nodes = {}
    for actor, y in (("a", 0), ("b", 8)):
        nodes[f"lane:{actor}"] = {"kind": "lane", "points": [[0, y], [5, y]],
                                   "outgoing": [f"connector:{actor}1", f"connector:{actor}2"], "speed_limit_mps": None}
        for number, dy in ((1, 0), (2, 4)):
            nodes[f"connector:{actor}{number}"] = {
                "kind": "connector", "points": [[5, y], [30, y+dy]], "outgoing": [], "speed_limit_mps": None}
    return RoadGraph({"schema_version": 1, "map_name": "straight", "map_version": "fixture",
                      "origin_xyz": [0, 0, 0], "bounds_local": [-10, -10, 100, 100]}, nodes)


def test_successor_only_schema_round_trip_and_route_diagnostics(straight_scene, tmp_path):
    graph = task_graph(straight_scene)
    straight_scene.velocities[0, 0, 0] = 8
    before = inspect_routes(straight_scene, graph)
    short = before["actors"]["a"]["initial_route_feasibility"]
    assert short["available_length_m"] == pytest.approx(3)
    assert short["immediate_max_braking_lower_bound_m"] == pytest.approx(6.4)
    assert short["insufficient_stopping_distance"]
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"schema_version": 1, "scenario_id": "fixture-straight", "source_log": "synthetic",
                               "successors": {"a": {"lane:a": "connector:a1"}}}))
    tasks = RouteTasks.load(path)
    assert tasks.initial_nodes == {}
    routes, report = resolve_routes(straight_scene, graph, tasks)
    assert routes["a"].node_ids == ("lane:a", "connector:a1")
    assert report["actors"]["a"]["route"]["end_reason"] == "dead_end"
    assert not report["actors"]["a"]["initial_route_feasibility"]["insufficient_stopping_distance"]
    assert report["route_tasks"]["successors"] == tasks.successors
    # Neither legal observations nor route feasibility depend on log futures.
    straight_scene.poses[1:] += 1000
    straight_scene.velocities[1:] *= 9
    assert inspect_routes(straight_scene, graph, tasks) == report


def test_invalid_successors_are_structured_resolution_failures(straight_scene):
    graph = task_graph(straight_scene)
    tasks = RouteTasks("fixture-straight", "synthetic", successors={"a": {"lane:a": "lane:b"}})
    report = inspect_routes(straight_scene, graph, tasks)
    assert not report["all_resolved"]
    assert "not adjacent" in report["actors"]["a"]["route_error"]
    with pytest.raises(RouteResolutionError):
        resolve_routes(straight_scene, graph, tasks)


def test_stopping_diagnostic_uses_vehicle_footprint_projection(straight_scene):
    graph = task_graph(straight_scene)
    straight_scene.poses[0, 0, 3] = np.pi/6
    report = inspect_routes(straight_scene, graph)
    diagnostic = report["actors"]["a"]["initial_route_feasibility"]
    assert diagnostic["front_extent_m"] == pytest.approx(2*np.cos(np.pi/6)+np.sin(np.pi/6))
