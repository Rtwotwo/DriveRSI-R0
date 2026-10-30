"""Extend simulator task support on directed map topology and declared navigation."""

import copy

import numpy as np

from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import Route


def _goal_support(graph, start, cursor, navigation, goal):
    """True/False/None: target reachable, known dead end, or incomplete search.

    Without an explicit mission goal, the target is the end of the ordered
    navigation. Reaching it proves lane continuity, not task completion.

    Use only navigation and loaded map geometry. Missing map data or a bounded
    search cannot prove an alternative branch impossible.
    """
    pending = [(start, cursor)]
    seen = set()
    unknown = False
    while pending:
        node_id, index = pending.pop()
        if (node_id, index) in seen:
            continue
        if len(seen) >= 1024:
            return None
        seen.add((node_id, index))
        if node_id not in graph.nodes:
            unknown = True
            continue
        node = graph.nodes[node_id]
        block = node.get("roadblock_id")
        if block == navigation[index]:
            pass
        elif index + 1 < len(navigation) and block == navigation[index + 1]:
            index += 1
        else:
            continue
        route = Route(node["points"])
        if goal is None:
            if index == len(navigation) - 1:
                return True
        else:
            arc, distance, _ = route.project(goal)
            if distance <= 2 and arc > 2:
                return True
        for child in node["outgoing"]:
            if child not in graph.nodes:
                unknown = True
                continue
            next_node = graph.nodes[child]
            if next_node.get("roadblock_id") not in navigation[index : index + 2]:
                continue
            if np.linalg.norm(np.asarray(next_node["points"][0]) - route.points[-1]) > 0.25:
                unknown = True
                continue
            pending.append((child, index))
    return None if unknown else False


def _parallel_lane_choice(graph, candidates, support):
    """Resolve lane choice within one proven navigation maneuver, not a road fork.

    All alternatives must join loaded lanes of the same downstream roadblock
    with aligned exit headings; proven navigation dead ends are rejected. An
    unknown distant continuation does not invalidate that local maneuver.
    Select minimum total
    heading change, then length and stable ID. This is a route-support policy,
    not a claim about the source driver's intended lane or traffic right-of-way.
    """
    if len(candidates) < 2 or any(
        key not in support or support[key] is False for _, key in candidates
    ):
        return None, []
    nodes = [graph.nodes[key] for _, key in candidates]
    if len({n.get("roadblock_id") for n in nodes}) != 1 or any(
        n["kind"] != "connector" for n in nodes
    ):
        return None, []
    destinations, exits, costs = set(), [], []
    for _, key in candidates:
        node = graph.nodes[key]
        if len(node["outgoing"]) != 1 or node["outgoing"][0] not in graph.nodes:
            return None, []
        target = graph.nodes[node["outgoing"][0]]
        if target["kind"] != "lane" or not target.get("roadblock_id"):
            return None, []
        destinations.add(target["roadblock_id"])
        points = np.asarray(node["points"])
        delta = np.diff(points, axis=0)
        headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        exits.append(headings[-1])
        costs.append(
            dict(
                node=key,
                total_turn_rad=float(np.abs(np.diff(headings)).sum()),
                length_m=float(np.linalg.norm(delta, axis=1).sum()),
            )
        )
    if len(destinations) != 1 or any(
        abs(np.arctan2(np.sin(h - exits[0]), np.cos(h - exits[0]))) > 0.15 for h in exits
    ):
        return None, []
    costs.sort(key=lambda x: (x["total_turn_rad"], x["length_m"], x["node"]))
    return costs[0]["node"], costs


def extend_ego_route(
    scene, graph, roadblocks, mission_goal_local, *, support_profile="strict_navigation_v1"
):
    """Retain the original supported prefix; never reinterpret its end as arrival.

    Roadblock navigation is privileged task information, not future detections.
    Ambiguous or missing topology remains an explicit support boundary.
    """
    if support_profile not in ("strict_navigation_v1", "bounded_navigation_v2"):
        raise ValueError("Unknown route support profile")
    graph.validate_scene(scene)
    original = Route(**scene.metadata["driving_tasks"]["actors"]["ego"]["route"])
    tangent = original.points[-1] - original.at(max(0, original.distances[-1] - 3))
    yaw = float(np.arctan2(tangent[1], tangent[0]))
    match = graph.match_pose(*original.points[-1], yaw)
    navigation = list(map(str, roadblocks))
    ambiguity_group = set(match.to_dict().get("ambiguous_node_ids", []))
    compatible = [
        c
        for c in match.candidates
        if c.node_id in ambiguity_group and graph.nodes[c.node_id].get("roadblock_id") in navigation
    ]
    previous = original.node_ids[-1] if original.node_ids else None
    prior = [c for c in compatible if c.node_id == previous and c.distance_m <= 0.25]
    if match.selected_node_id is None and len(prior) == 1:
        match = graph.match_pose(*original.points[-1], yaw, initial_node_id=previous)
    elif match.selected_node_id is None and len(compatible) == 1:
        match = graph.match_pose(*original.points[-1], yaw, initial_node_id=compatible[0].node_id)
    evidence = {
        "version": 1,
        "support_profile": support_profile,
        "original_length_m": float(original.distances[-1]),
        "original_endpoint": original.points[-1].tolist(),
        "roadblocks": list(roadblocks),
        "matching": match.to_dict(),
        "status": "not_extended",
        "decisions": [],
    }
    allowed = set(map(str, roadblocks))
    goal = np.asarray(mission_goal_local, float)[:2] if mission_goal_local is not None else None
    result = copy.deepcopy(scene)
    if match.selected_node_id is None:
        evidence["reason"] = match.reason
        result.metadata["ego_route_continuation"] = evidence
        return result
    node_id = match.selected_node_id
    occurrences = [
        i for i, block in enumerate(navigation) if block == graph.nodes[node_id].get("roadblock_id")
    ]
    if len(occurrences) != 1:
        evidence["reason"] = (
            "ambiguous_navigation_occurrence" if occurrences else "outside_declared_navigation"
        )
        result.metadata["ego_route_continuation"] = evidence
        return result
    cursor = occurrences[0]
    evidence["initial_navigation_index"] = cursor
    points = original.points.tolist()
    visited = []
    end_reason = "node_budget"
    destination = False
    for _ in range(128):
        if node_id in visited:
            end_reason = "cycle"
            break
        node = graph.nodes[node_id]
        if node.get("roadblock_id") not in allowed:
            end_reason = "outside_declared_navigation"
            break
        route = Route(node["points"])
        arc, distance, _ = route.project(points[-1])
        if not visited:
            arc = min(route.distances[-1], arc + max(3.0, 3 * distance))
        else:
            arc = 0.0
            if np.linalg.norm(np.asarray(points[-1]) - route.points[0]) > 0.25:
                end_reason = "geometry_gap"
                break
        stop = float(route.distances[-1])
        if goal is not None:
            ga, gd, _ = route.project(goal)
            if gd <= 2 and ga > arc + 2:
                stop = ga
                destination = True
        part = np.vstack(
            (
                route.at(arc),
                route.points[(route.distances > arc) & (route.distances < stop)],
                route.at(stop),
            )
        )
        for point in part:
            if np.linalg.norm(point - np.asarray(points[-1])) > 1e-4:
                points.append(point.tolist())
        visited.append(node_id)
        if destination:
            end_reason = "explicit_endpoint"
            break
        children = [
            key
            for key in node["outgoing"]
            if key in graph.nodes
            and graph.nodes[key].get("roadblock_id") in allowed
            and graph.nodes[key]["roadblock_id"] in navigation[cursor : cursor + 2]
        ]
        if not children:
            end_reason = (
                "map_boundary"
                if any(key not in graph.nodes for key in node["outgoing"])
                else "navigation_end"
            )
            break
        if len(children) > 1:
            # Continue only when one compatible branch is distinguishable by heading.
            tail = route.points[-1] - route.points[-2]
            candidates = []
            for child in children:
                head = np.asarray(graph.nodes[child]["points"])
                if np.linalg.norm(head[0] - route.points[-1]) > 0.25:
                    continue
                direction = head[min(2, len(head) - 1)] - head[0]
                cross = tail[0] * direction[1] - tail[1] * direction[0]
                candidates.append((abs(float(np.arctan2(cross, tail @ direction))), child))
            candidates.sort()
            goal_support = {}
            lane_costs = []
            ambiguous = not candidates or (
                len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.05
            )
            if ambiguous:
                goal_support = {
                    child: _goal_support(graph, child, cursor, navigation, goal)
                    for _, child in candidates
                }
                viable = [child for child, supported in goal_support.items() if supported is True]
                locally_possible = [
                    child for child, supported in goal_support.items() if supported is not False
                ]
                if (
                    support_profile == "bounded_navigation_v2"
                    and goal is None
                    and len(locally_possible) == 1
                ):
                    # One local navigation continuation remains; unknown distant map is not a goal proof.
                    chosen, ambiguous = locally_possible[0], False
                elif len(viable) == 1 and all(value is not None for value in goal_support.values()):
                    chosen = viable[0]
                    ambiguous = False
                elif len(viable) == len(candidates) or (
                    goal is None and all(value is not False for value in goal_support.values())
                ):
                    lane_choice, lane_costs = _parallel_lane_choice(graph, candidates, goal_support)
                    if lane_choice is not None:
                        chosen, ambiguous = lane_choice, False
            elif not ambiguous:
                chosen = candidates[0][1]
            if ambiguous:
                end_reason = "ambiguous_navigation_branch"
                break
            evidence["decisions"].append(
                {
                    "selection_rule": "unique_locally_supported_navigation"
                    if support_profile == "bounded_navigation_v2"
                    and goal_support.get(chosen) is None
                    else "parallel_navigation_minimum_turn"
                    if lane_costs
                    else "unique_navigation_or_heading",
                    "parallel_lane_costs": lane_costs,
                    "node": node_id,
                    "selected": chosen,
                    "heading_candidates": candidates,
                    "goal_support": goal_support if goal is not None else {},
                    "navigation_support": goal_support if goal is None else {},
                }
            )
        else:
            chosen = children[0]
        next_block = graph.nodes[chosen]["roadblock_id"]
        if next_block != navigation[cursor]:
            cursor += 1
        node_id = chosen
    extended = Route(points, tuple(visited), end_reason)
    if extended.distances[-1] > original.distances[-1] + 2:
        result = bind_driving_tasks(
            result, {"ego": extended}, destinations=("ego",) if destination else ()
        )
        evidence["status"] = "extended"
    evidence.update(
        end_reason=end_reason,
        extended_length_m=float(extended.distances[-1]),
        endpoint=extended.points[-1].tolist(),
        nodes=visited,
        destination=destination,
        mission_goal_local=None if goal is None else goal.tolist(),
    )
    result.metadata["ego_route_continuation"] = evidence
    return result


def initial_navigation_routes(scene, graph, roadblocks):
    """Start with one supported lane; continuation follows navigation, not greedy forks."""
    from driversi.contracts.road import LaneMatchError

    routes, diagnostics = {}, {}
    for i, actor in enumerate(scene.actor_ids):
        if not scene.controlled[i]:
            continue
        pose = scene.poses[0, i, [0, 1, 3]]
        match = graph.match_pose(*pose)
        choice = match.selected_node_id
        if choice is None:
            candidates = [
                c.node_id
                for c in match.candidates
                if c.node_id in match.ambiguous_node_ids
                and graph.nodes[c.node_id].get("roadblock_id") in roadblocks
            ]
            if len(candidates) == 1:
                choice = candidates[0]
        if choice is None:
            raise LaneMatchError(match)
        route = graph.route_from_pose(*pose, initial_node_id=choice, max_nodes=1)
        # A rear axle near a connector exit needs the adjacent loaded lane before binding.
        for budget in range(2, 33):
            arc, _, _ = route.project(pose[:2])
            if route.distances[-1] - arc > 2 and np.linalg.norm(pose[:2] - route.points[-1]) > 2:
                break
            candidate = graph.route_from_pose(*pose, initial_node_id=choice, max_nodes=budget)
            if candidate.distances[-1] <= route.distances[-1]:
                break
            if any(
                graph.nodes[n].get("roadblock_id") not in roadblocks for n in candidate.node_ids
            ):
                break
            route = candidate
        routes[actor] = route
        diagnostics[actor] = dict(
            matching=match.to_dict(),
            chosen_initial_node=choice,
            rule="initial_lane_then_ordered_navigation",
        )
    return routes, diagnostics
