"""Explicit geometric driving tasks; no inferred signal or right-of-way rules."""

import copy
import json

import numpy as np

from driversi.contracts.road import Route

TASK_VERSION = "route-task-v1"
REWARD_VERSION = "event-plus-supported-route-progress-v1"


def tasks_from_metadata(metadata, actor_ids):
    value = metadata.get("driving_tasks")
    if value is None:
        return {}
    if (
        not isinstance(value, dict)
        or value.get("version") != TASK_VERSION
        or set(value) != {"version", "actors"}
        or not isinstance(value["actors"], dict)
        or set(value["actors"]) != set(actor_ids)
    ):
        raise ValueError("Driving tasks must cover every controlled actor with a supported version")
    result = {}
    for actor, task in value["actors"].items():
        if (
            not isinstance(task, dict)
            or set(task) != {"route", "endpoint", "corridor_m", "rule_status"}
            or task["endpoint"] not in {"destination", "support_boundary"}
            or task["rule_status"] != "unknown"
        ):
            raise ValueError("Invalid driving task; traffic legality is not established")
        width = task["corridor_m"]
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, float))
            or not np.isfinite(width)
            or not 0.25 <= width <= 4
        ):
            raise ValueError("Task corridor tolerance must be between0.25and4metres")
        route = Route(**task["route"])
        if len(route.points) > 8192:
            raise ValueError("Task route exceeds8192points")
        if task["endpoint"] == "destination" and route.end_reason != "explicit_endpoint":
            raise ValueError("A destination requires an explicitly declared endpoint")
        goals = metadata.get("task_goals_xy", {})
        if actor not in goals or not np.array_equal(goals[actor], route.points[-1]):
            raise ValueError("Task endpoint and world goal differ")
        # Avoid native radius completion on an earlier pass near the endpoint.
        arc = route.distances
        cutoff = max(0.0, arc[-1] - (2.0 + width))
        if cutoff > 0:
            prefix = np.vstack((route.points[arc < cutoff], route.at(cutoff)))
            if len(prefix) >= 2 and Route(prefix).project(route.points[-1])[1] < 2.0 - 1e-6:
                raise ValueError("Route passes within goal radius before final approach")
        result[actor] = (route, task)
    return result


def bind_driving_tasks(scene, routes, *, destinations=(), corridor_m=2.0):
    """Return a new scene. Routes are declared intentions, never inferred futures."""
    result = copy.deepcopy(scene)
    ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    if set(routes) != set(ids) or not set(destinations) <= set(ids):
        raise ValueError("Routes must cover controlled actors and destinations must be controlled")
    actors = {}
    for actor, route in routes.items():
        if actor in destinations:
            route = Route(route.points, route.node_ids, "explicit_endpoint", route.speed_limit_mps)
        actors[actor] = {
            "route": route.to_dict(),
            "endpoint": "destination" if actor in destinations else "support_boundary",
            "corridor_m": corridor_m,
            "rule_status": "unknown",
        }
    result.metadata["driving_tasks"] = {"version": TASK_VERSION, "actors": actors}
    result.metadata["task_goals_xy"] = {
        a: item["route"]["points"][-1] for a, item in actors.items()
    }
    result.validate()
    for i, actor in enumerate(result.actor_ids):
        if actor not in routes:
            continue
        route = routes[actor]
        arc, distance, heading = route.project(result.poses[0, i, :2])
        error = abs(
            np.arctan2(
                np.sin(heading - result.poses[0, i, 3]), np.cos(heading - result.poses[0, i, 3])
            )
        )
        if distance > corridor_m or error > np.pi / 3 or route.distances[-1] - arc <= 2:
            raise ValueError(f"Initial pose is outside task support: {actor}")
        if np.linalg.norm(result.poses[0, i, :2] - route.points[-1]) <= 2:
            raise ValueError(f"Initial pose is already at task endpoint: {actor}")
    return result


def task_identity(scene):
    value = scene.metadata.get("driving_tasks")
    return (
        b""
        if value is None
        else b"driving-tasks:" + json.dumps(value, sort_keys=True, allow_nan=False).encode()
    )


def spec_options(metadata):
    if "driving_tasks" not in metadata:
        return {}
    return {
        "goal_semantics": "explicit-destination-or-route-support-truncation-v1",
        "reward_version": REWARD_VERSION,
    }


def project_supported(route, xy):
    edges = np.diff(route.points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    fractions = np.clip(np.sum((xy - route.points[:-1]) * edges, axis=1) / lengths**2, 0, 1)
    distances = np.linalg.norm(route.points[:-1] + fractions[:, None] * edges - xy, axis=1)
    arcs = route.distances[:-1] + fractions * lengths
    nearest = int(np.argmin(distances))
    ambiguous = np.any((distances <= distances[nearest] + 0.05) & (abs(arcs - arcs[nearest]) > 2.0))
    return float(arcs[nearest]), float(distances[nearest]), bool(ambiguous)


def route_progress(route, before_xy, after_xy, *, corridor_m):
    """Stateless signed arc progress with conservative shortcut/jump rejection.

    Five chord samples are a bounded geometric check, not swept-road compliance.
    Rejected progress is zero (not a navigation reward or safety certificate).
    """
    before_xy, after_xy = np.asarray(before_xy), np.asarray(after_xy)
    projections = [project_supported(route, p) for p in np.linspace(before_xy, after_xy, 5)]
    start, end = projections[0], projections[-1]
    travel = float(np.linalg.norm(after_xy - before_xy))
    delta = end[0] - start[0]
    reason = (
        "ambiguous_projection"
        if any(p[2] for p in projections)
        else "outside_corridor"
        if any(p[1] > corridor_m for p in projections)
        else "projection_jump"
        if abs(delta) > travel * 1.2 + 1e-4
        else "supported"
    )
    return {
        "progress_m": delta if reason == "supported" else 0.0,
        "raw_progress_m": delta,
        "arc_m": end[0],
        "remaining_m": float(route.distances[-1] - end[0]),
        "cross_track_m": end[1],
        "travel_m": travel,
        "status": reason,
    }
