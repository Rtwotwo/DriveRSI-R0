"""Resolve declared tasks and retain all unresolved actor diagnostics."""

import numpy as np

from driversi.contracts.road import RoadGraph, Route
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene


class RouteResolutionError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        failed = [actor for actor, item in report["actors"].items() if not item["resolved"]]
        super().__init__("Unresolved route tasks for: " + ", ".join(failed))


def inspect_routes(scene: Scene, graph: RoadGraph, tasks: RouteTasks | None = None) -> dict:
    graph.validate_scene(scene)
    if tasks:
        tasks.validate_scene(scene)
    report = {"schema_version": 1, "scenario_id": scene.metadata["scenario_id"],
              "source_log": scene.metadata["source_log"], "actors": {},
              "route_tasks": tasks.to_dict() if tasks else None}
    for i, actor in enumerate(scene.actor_ids):
        if not scene.controlled[i]:
            continue
        choice = tasks.initial_nodes.get(actor) if tasks else None
        match = graph.match_pose(*scene.poses[0, i, [0, 1, 3]], initial_node_id=choice)
        diagnostic = match.to_dict()
        diagnostic["requested_initial_node_id"] = choice
        successors = tasks.successors.get(actor, {}) if tasks else {}
        diagnostic["requested_successors"] = dict(successors)
        diagnostic["resolved"] = match.selected_node_id is not None
        if diagnostic["resolved"]:
            try:
                route = graph.route_from_pose(*scene.poses[0, i, [0, 1, 3]],
                                              initial_node_id=choice, successors=successors)
            except ValueError as error:
                diagnostic.update(resolved=False, route_error=str(error))
            else:
                progress, _, heading = route.project(scene.poses[0, i, :2])
                yaw = float(scene.poses[0, i, 3])
                angle = yaw-heading
                front = float((scene.dimensions[i, 0]*abs(np.cos(angle))
                               + scene.dimensions[i, 1]*abs(np.sin(angle)))/2)
                length = float(route.distances[-1])
                available = length-progress-front
                velocity = scene.velocities[0, i, :2]
                forward = float(velocity @ np.asarray([np.cos(yaw), np.sin(yaw)]))
                speed = float(np.linalg.norm(velocity)) if forward > 0 else 0.0
                minimum_stopping_distance = speed**2/10
                diagnostic["route"] = {"node_ids": list(route.node_ids), "end_reason": route.end_reason,
                                       "length_m": length, "speed_limit_mps": route.speed_limit_mps}
                diagnostic["initial_route_feasibility"] = {
                    "available_length_m": available, "initial_arc_length_m": progress,
                    "front_extent_m": front, "forward_speed_mps": speed,
                    "immediate_max_deceleration_mps2": 5.0,
                    "immediate_max_braking_lower_bound_m": minimum_stopping_distance,
                    "insufficient_stopping_distance": bool(available < minimum_stopping_distance),
                    "model": "one_dimensional_forward_motion_immediate_constant_deceleration",
                    "interpretation": "necessary only in this idealized model; not a sufficient feasibility or safety check",
                }
        report["actors"][actor] = diagnostic
    report["all_resolved"] = all(item["resolved"] for item in report["actors"].values())
    return report


def resolve_routes(scene: Scene, graph: RoadGraph, tasks: RouteTasks | None = None) -> tuple[dict[str, Route], dict]:
    report = inspect_routes(scene, graph, tasks)
    if not report["all_resolved"]:
        raise RouteResolutionError(report)
    routes = {actor: graph.route_from_pose(*scene.poses[0, i, [0, 1, 3]],
                                          initial_node_id=tasks.initial_nodes.get(actor) if tasks else None,
                                          successors=tasks.successors.get(actor) if tasks else None)
              for i, actor in enumerate(scene.actor_ids) if scene.controlled[i]}
    return routes, report
