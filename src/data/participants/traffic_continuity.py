"""Opt-in traffic continuity with declared map exits and retained source identities."""

import copy

import numpy as np

from driversi.contracts.road import LaneMatchError, Route
from driversi.contracts.scene import CYCLIST, OTHER_OBJECT, VEHICLE
from driversi.data.participants.background import propose_logged_tasks

# Exact native nuPlan STATIC_OBJECT_TYPES; unknown/other categories are not included.
STATIC_SOURCE_KINDS = frozenset({"traffic_cone", "barrier", "czone_sign", "generic_object"})

def configure_traffic_continuity(scene, graph):
    graph.validate_scene(scene)
    result = copy.deepcopy(scene)
    # A scene can contain pedestrians/objects but no passive vehicles. Fleet
    # setup then has no assignments; continuity must still own every object.
    if "reactive_background" not in result.metadata:
        if result.controlled.all():
            return result
        result.metadata["reactive_background"] = {
            "version": 1,
            "lifetime": "first_annotation_to_horizon",
            "assignments": {},
            "source_presence": np.stack(
                [result.presence_at(t) for t in range(len(result.times))]
            ).tolist(),
        }
    config = result.metadata["reactive_background"]
    tasks = config["assignments"]
    if result.presence is None:
        result.presence = np.asarray(config["source_presence"], bool).copy()
    source = copy.deepcopy(result)
    source.presence = np.asarray(config["source_presence"], bool)
    source.metadata.pop("reactive_background", None)
    source.metadata.pop("vehicle_fleet", None)
    extra, _ = propose_logged_tasks(source)
    evidence = {}
    # Every passive object is owned after admission. Stationary/unknown objects
    # are retained; no gap-based disappearance, no invented pedestrian destination.
    for i, actor in enumerate(result.actor_ids):
        if result.controlled[i]:
            continue
        hits = np.flatnonzero(source.presence[:, i])
        if not len(hits):
            continue
        spawn = int(hits[0])
        measured = (
            source.metadata.get("first_observations", {}).get(actor, {}).get("velocity_world")
        )
        if measured is not None:
            measured = np.asarray(measured, float)
            if measured.shape != (3,) or not np.isfinite(measured).all():
                raise ValueError(f"Invalid measured velocity: {actor}")
        source_kind = source.metadata.get("source_actor_kinds", {}).get(actor)
        semantic_static = result.kinds[i] == OTHER_OBJECT and source_kind in STATIC_SOURCE_KINDS
        compiled_spawn_velocity = source.velocities[spawn, i].copy()
        if semantic_static:
            # The native category declares a static SceneObject, not an Agent.
            # Noisy box differences and finite estimated velocities must not
            # induce a braking displacement before this fixed object is held.
            start = source.poses[spawn, i, :2]
            yaw = source.poses[spawn, i, 3]
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            tasks[actor] = {
                "points": [start.tolist(), (start + direction * 0.01).tolist()],
                "desired_speed": 0.0,
                "navigation_source": "observed_stationary_hold",
            }
        elif actor not in tasks:
            if actor in extra:
                tasks[actor] = extra[actor]
            else:
                v = measured[:2] if measured is not None else source.velocities[spawn, i, :2]
                speed = float(np.linalg.norm(v))
                direction = (
                    v / speed
                    if speed > 1e-6
                    else np.array(
                        [np.cos(source.poses[spawn, i, 3]), np.sin(source.poses[spawn, i, 3])]
                    )
                )
                start = source.poses[spawn, i, :2]
                tasks[actor] = {
                    "points": [
                        start.tolist(),
                        (start + direction * max(0.01, speed * speed / 6 + 0.3)).tolist(),
                    ],
                    "desired_speed": 0.0,
                    "navigation_source": "observed_stationary_hold"
                    if speed <= 0.2
                    else "uncertain_motion_braking",
                }
        result.presence[spawn:, i] = True
        task = tasks[actor]
        measured_braking = measured is not None and task["navigation_source"] in {
            "uncertain_motion_braking",
            "observed_stationary_hold",
        }
        if semantic_static:
            result.velocities[spawn:, i] = 0.0
        elif measured_braking:
            result.velocities[spawn, i] = measured
        evidence[actor] = {
            "status": "preserved_intent",
            "departure_allowed": False,
            "spawn_velocity_source": "native_static_category" if semantic_static else
                "measured" if measured_braking else "existing_compiled_state",
        }
        if semantic_static:
            evidence[actor]["semantic_static"] = {
                "rule": "native_nuplan_static_object_type_v1",
                "source_category": source_kind,
                "measured_spawn_velocity_world": None if measured is None else measured.tolist(),
                "compiled_spawn_velocity_world": compiled_spawn_velocity.tolist(),
                "source_observations_preserved": True,
                "physical_pose": "first_admitted_pose_fixed_without_annotation_snapping",
            }
        if result.kinds[i] not in (VEHICLE, CYCLIST) or task["desired_speed"] <= 0:
            continue
        original = Route(task["points"])
        tangent = original.points[-1] - original.at(max(0, original.distances[-1] - 3))
        yaw = float(np.arctan2(tangent[1], tangent[0]))
        try:
            # Declared traffic intention at branches, fixed for the whole rollout.
            # Only existing topology edges can be chosen.
            choices = {}
            extension = graph.route_from_pose(*original.points[-1], yaw)
            for _ in range(16):
                if extension.end_reason != "branch":
                    break
                node_id = extension.node_ids[-1]
                node = graph.nodes[node_id]
                tail = np.asarray(node["points"])
                tangent = tail[-1] - tail[-2]
                candidates = []
                for child in node["outgoing"]:
                    if child not in graph.nodes:
                        continue
                    head = np.asarray(graph.nodes[child]["points"])
                    direction = head[min(2, len(head) - 1)] - head[0]
                    if np.linalg.norm(head[0] - tail[-1]) > 0.25:
                        continue
                    angle = abs(np.arctan2(np.cross(tangent, direction), tangent @ direction))
                    candidates.append((float(angle), child))
                if not candidates:
                    break
                choices[node_id] = min(candidates)[1]
                extension = graph.route_from_pose(*original.points[-1], yaw, successors=choices)
        except (ValueError, LaneMatchError) as error:
            evidence[actor].update(status="unresolved_map_match", reason=str(error))
            continue
        progress, distance, heading = extension.project(original.points[-1])
        # A gradual bridge to a compatible baseline, never a lateral teleport.
        begin = min(extension.distances[-1], progress + max(5, 3 * distance))
        part = extension.points[extension.distances > begin]
        if not len(part):
            evidence[actor].update(
                status="no_forward_map_support",
                end_reason=extension.end_reason,
                generated_branch_choices=choices,
            )
            continue
        bridge = extension.at(begin)
        points = np.vstack((original.points, bridge, part))
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]]
        task["points"] = points.tolist()
        task["navigation_source"] = "map_continuation_privileged"
        evidence[actor].update(
            status="map_extended",
            original_length_m=float(original.distances[-1]),
            extended_length_m=float(Route(points).distances[-1]),
            nodes=list(extension.node_ids),
            end_reason=extension.end_reason,
            generated_branch_choices=choices,
        )
    origin = np.asarray(result.metadata["origin_xyz"])[:2]
    bounds = np.asarray(result.metadata["map_bounds_global"]) - np.tile(origin, 2)
    # Keep a physical/map buffer outside the simulation exit domain.
    domain = bounds + np.array([20, 20, -20, -20])
    if np.any(domain[2:] - domain[:2] <= 40):
        raise ValueError("Traffic continuity needs a nonempty buffered map domain")
    for actor, row in evidence.items():
        if row["status"] == "map_extended":
            i = result.actor_ids.index(actor)
            end = np.asarray(tasks[actor]["points"][-1])
            radius = 0.5 * np.linalg.norm(result.dimensions[i, :2])
            sides = [
                k
                for k in range(4)
                if (end[k % 2] < domain[k] - radius if k < 2 else end[k % 2] > domain[k] + radius)
            ]
            row["departure_allowed"] = bool(sides)
            row["exit_sides"] = sides
            row["exit_route_max_distance_m"] = 2.0
    result.metadata["traffic_continuity"] = {
        "version": 1,
        "forward_only": True,
        "initial_overlap_policy": "retain_and_report",
        "domain": domain.tolist(),
        "admission_horizon_seconds": 2.0,
        "admission_margin_m": 0.5,
        "navigation": evidence,
        "departure": "whole_footprint_outward_supported_map_exit",
        "pending": "retain_identity_at_original_entry_no_timeout_deletion",
        "trainable": False,
    }
    result.validate()
    return result
