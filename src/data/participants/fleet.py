"""Explicit traffic ownership for every vehicle, separate from learning slots."""

import copy
from collections import Counter

import numpy as np

from driversi.contracts.scene import VEHICLE
from driversi.data.participants.background import configure_background, propose_logged_tasks


def configure_vehicle_fleet(scene):
    """Own all vehicles from first annotation; never invent a parked departure.

    Logged routes are privileged navigation intentions. Unsupported intentions
    use bounded braking/holding and remain visible as such, not qualified drivers.
    Existing source support and nonvehicle reactive tasks are preserved.
    """
    if scene.metadata.get("vehicle_fleet", {}).get("version") == 2:
        scene.validate()
        return copy.deepcopy(scene)
    source = copy.deepcopy(scene)
    source.metadata.pop("vehicle_fleet", None)
    old = source.metadata.pop("reactive_background", None)
    if old is not None:
        source.presence = np.asarray(old["source_presence"], bool)
    source.metadata.pop("observed_motion_inference", None)
    tasks, rejected = propose_logged_tasks(source)
    tasks = {
        a: task for a, task in tasks.items() if source.kinds[source.actor_ids.index(a)] == VEHICLE
    }
    # Preserve explicitly configured navigation and pedestrian control.
    if old is not None:
        tasks.update(copy.deepcopy(old["assignments"]))
    roles = {}
    for i, actor in enumerate(source.actor_ids):
        if source.kinds[i] != VEHICLE:
            continue
        if source.controlled[i]:
            roles[actor] = {"mode": "policy_action", "learning_slot": True}
            continue
        support = np.array([source.presence_at(t)[i] for t in range(len(source.times))])
        hits = np.flatnonzero(support)
        if not len(hits):
            roles[actor] = {"mode": "outside_window", "learning_slot": False, "spawn_tick": None}
            tasks.pop(actor, None)
            continue
        spawn = int(hits[0])
        end = next((t for t in range(spawn + 1, len(support)) if not support[t]), len(support))
        raw = source.metadata.get("first_observations", {}).get(actor, {}).get("velocity_world")
        measured = np.asarray(raw, float) if raw is not None else None
        if measured is not None and (measured.shape != (3,) or not np.isfinite(measured).all()):
            raise ValueError(f"Invalid measured velocity: {actor}")
        evidence = (
            "measured_velocity"
            if measured is not None
            else "continuous_position_difference"
            if end - spawn > 6
            else "unknown_velocity"
        )
        velocity = measured if measured is not None else source.velocities[spawn, i].copy()
        speed = float(np.linalg.norm(velocity[:2]))
        yaw = source.poses[spawn, i, 3]
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        points = source.poses[spawn:end, i, :2]
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]]
        moving_path = (
            len(points) > 1 and np.linalg.norm(np.diff(points, axis=0), axis=1).sum() >= 0.5
        )
        aligned = speed <= 0.2 or velocity[:2] @ forward >= 0.5 * speed
        tangent_ok = moving_path and (points[1] - points[0]) @ forward >= 0.5 * np.linalg.norm(
            points[1] - points[0]
        )
        mode = "route_feedback"
        motion = source.metadata.get("source_motion_evidence", {})
        motion = (
            motion.get("actors", {}).get(actor, {})
            if (
                motion.get("version") == 1
                and motion.get("scope") == "first_contiguous_run_privileged_navigation"
            )
            else {}
        )
        # Position jitter is not departure intent. Require complete measured motion
        # evidence and bounded positional drift; do not override explicit prior tasks.
        confirmed_hold = (
            motion.get("samples", 0) >= 2
            and motion.get("finite_velocity_samples") == motion["samples"]
            and 0 <= motion.get("max_reported_speed_mps", float("inf")) <= 0.2
            and 0 <= motion.get("max_position_excursion_m", float("inf")) <= 1.5
            and speed <= 0.2
            and (old is None or actor not in old["assignments"])
        )
        if confirmed_hold:
            start = source.poses[spawn, i, :2]
            tasks[actor] = {
                "points": [start.tolist(), (start + 0.01 * forward).tolist()],
                "desired_speed": 0.0,
                "navigation_source": "observed_stationary_hold",
            }
        if actor not in tasks:
            # Starting from rest does not invalidate an otherwise supported route.
            desired = float(np.median(np.linalg.norm(source.velocities[spawn:end, i, :2], axis=1)))
            if tangent_ok and aligned and 0.1 <= desired <= 20:
                tasks[actor] = {
                    "points": points.tolist(),
                    "desired_speed": desired,
                    "navigation_source": "logged_path_privileged",
                }
            else:
                known_stationary = (
                    evidence != "unknown_velocity" and speed <= 0.2 and not moving_path
                )
                mode = "stationary_hold" if known_stationary else "uncertain_brake"
                direction = velocity[:2] / speed if speed > 1e-6 else forward
                start = source.poses[spawn, i, :2]
                distance = max(0.01, speed * speed / 10 + 0.3) if speed > 0.2 else 0.01
                tasks[actor] = {
                    "points": [start.tolist(), (start + distance * direction).tolist()],
                    "desired_speed": 0.0,
                    "navigation_source": "observed_stationary_hold"
                    if known_stationary
                    else "uncertain_motion_braking",
                }
        if tasks[actor]["navigation_source"] == "observed_stationary_hold":
            mode = "stationary_hold"
        elif tasks[actor]["navigation_source"] == "uncertain_motion_braking":
            mode = "uncertain_brake"
        elif tasks[actor]["navigation_source"] == "observed_velocity_extrapolation":
            mode = "bounded_motion_feedback"
        # Preserve measured spawn velocity rather than the single-frame zero fill.
        source.velocities[spawn, i] = velocity
        roles[actor] = {
            "mode": mode,
            "learning_slot": False,
            "spawn_tick": spawn,
            "velocity_evidence": evidence,
            "stationary_evidence": {
                "accepted": bool(confirmed_hold),
                "rule": "complete_source_speed_le_0.2_and_excursion_le_1.5_v1",
                "source": motion,
            },
            "previous_rejection": rejected.get(actor),
            "navigation_source": tasks[actor]["navigation_source"],
        }
    result = configure_background(source, tasks) if tasks else source
    result.metadata["vehicle_fleet"] = {
        "version": 2,
        "controller": "traffic-fleet-v2",
        "roles": roles,
        "counts": dict(Counter(v["mode"] for v in roles.values())),
        "lifetime": "first_annotation_to_horizon",
        "trainable": False,
        "learning_slots_unchanged": True,
        "cooperation": "moving_traffic_only; learner_policy_is_not_assumed_cooperative",
        "prediction": "current_and_intended_velocity_swept_boxes_and_declared_routes",
    }
    result.validate()
    return result


def validate_vehicle_fleet(scene):
    config = scene.metadata.get("vehicle_fleet")
    if config is None:
        return
    if (config.get("version"), config.get("controller")) not in {
        (1, "traffic-fleet-v1"),
        (2, "traffic-fleet-v2"),
    }:
        raise ValueError("Unsupported vehicle fleet")
    vehicles = {a for a, kind in zip(scene.actor_ids, scene.kinds) if kind == VEHICLE}
    if set(config.get("roles", {})) != vehicles:
        raise ValueError("Fleet must assign every vehicle")
    tasks = scene.metadata.get("reactive_background", {}).get("assignments", {})
    for i, actor in enumerate(scene.actor_ids):
        if actor not in vehicles:
            continue
        mode = config["roles"][actor]["mode"]
        if scene.controlled[i]:
            if mode != "policy_action":
                raise ValueError("Learning policy ownership differs from fleet role")
        elif mode == "outside_window":
            if actor in tasks or any(scene.presence_at(t)[i] for t in range(len(scene.times))):
                raise ValueError("Outside-window vehicle cannot be spawned or assigned motion")
        elif actor not in tasks or mode not in {
            "route_feedback",
            "bounded_motion_feedback",
            "stationary_hold",
            "uncertain_brake",
        }:
            raise ValueError("Fleet vehicle lacks executable traffic task")
        elif (mode == "stationary_hold") != (
            tasks[actor]["navigation_source"] == "observed_stationary_hold"
        ) or (mode == "uncertain_brake") != (
            tasks[actor]["navigation_source"] == "uncertain_motion_braking"
        ):
            raise ValueError("Fleet role disagrees with controller")
