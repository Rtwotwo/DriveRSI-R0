"""Explicit finite-lifetime background tasks; raw annotation support is retained."""

import copy
import json

import numpy as np

from driversi.contracts.road import Route
from driversi.contracts.scene import CYCLIST, OTHER_OBJECT, PEDESTRIAN, VEHICLE


def identity_suffix(scene):
    config = scene.metadata.get("reactive_background")
    encoded = (
        b"" if config is None else json.dumps(config, sort_keys=True, allow_nan=False).encode()
    )
    fleet = scene.metadata.get("vehicle_fleet")
    encoded = (
        encoded
        if fleet is None
        else encoded + b"fleet:" + json.dumps(fleet, sort_keys=True, allow_nan=False).encode()
    )
    prior = scene.metadata.get("static_scene_context")
    if prior is not None:
        encoded += b"static_scene_context:" + json.dumps(prior, sort_keys=True, allow_nan=False).encode()
    continuity = scene.metadata.get("traffic_continuity")
    return (
        encoded
        if continuity is None
        else encoded
        + b"continuity:"
        + json.dumps(continuity, sort_keys=True, allow_nan=False).encode()
    )


def validate_background(scene):
    config = scene.metadata.get("reactive_background")
    if config is None:
        return {}
    if (
        not isinstance(config, dict)
        or set(config) != {"version", "assignments", "source_presence", "lifetime"}
        or config["version"] != 1
        or config["lifetime"] != "first_annotation_to_horizon"
    ):
        raise ValueError("Unsupported reactive background configuration")
    support = np.asarray(config["source_presence"])
    if support.dtype != np.bool_ or support.shape != scene.poses.shape[:2]:
        raise ValueError("Background source support must be boolean time x actor")
    from driversi.data.scenes.static_context import static_prior_actors

    surveyed_static = static_prior_actors(scene)
    assignments = config["assignments"]
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Background requires explicit assignments")
    for actor, task in assignments.items():
        if actor not in scene.actor_ids:
            raise ValueError("Unknown background actor")
        i = scene.actor_ids.index(actor)
        if scene.controlled[i] or scene.kinds[i] not in (
            VEHICLE,
            PEDESTRIAN,
            CYCLIST,
            OTHER_OBJECT,
        ):
            raise ValueError("Reactive background supports nonlearner participants only")
        if set(task) != {"points", "desired_speed", "navigation_source"}:
            raise ValueError("Background task fields mismatch")
        Route(task["points"])
        speed = task["desired_speed"]
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not np.isfinite(speed)
            or not 0 <= speed <= (2.5 if scene.kinds[i] == PEDESTRIAN else 20)
            or (
                speed == 0
                and task["navigation_source"]
                not in {"observed_stationary_hold", "uncertain_motion_braking"}
            )
        ):
            raise ValueError("Background desired speed outside supported range")
        if task["navigation_source"] not in (
            "explicit_task",
            "logged_path_privileged",
            "observed_velocity_extrapolation",
            "observed_stationary_hold",
            "uncertain_motion_braking",
            "map_continuation_privileged",
        ):
            raise ValueError("Declare navigation provenance")
        indices = np.flatnonzero(support[:, i])
        expected = np.arange(len(scene.times)) >= (0 if actor in surveyed_static else
            indices[0] if len(indices) else len(scene.times))
        if not len(indices) or not np.array_equal(scene.presence_at(indices[0])[i], True):
            raise ValueError("Background task has no supported spawn")
        if scene.presence is None or not np.array_equal(scene.presence[:, i], expected):
            raise ValueError("Reactive lifetime differs from declared ownership")
        start = 0 if actor in surveyed_static else indices[0]
        if np.linalg.norm(np.asarray(task["points"])[0] - scene.poses[start, i, :2]) > 0.05:
            raise ValueError("Background path must start at its logged spawn")
    return assignments


def configure_background(scene, assignments):
    """Copy a scene, retaining IDs and raw support; never merge fragmented tracks."""
    if "reactive_background" in scene.metadata:
        raise ValueError("Configure background from the unmodified source scene")
    scene = copy.deepcopy(scene)
    support = np.stack([scene.presence_at(t) for t in range(len(scene.times))])
    scene.presence = support.copy()
    for actor in assignments:
        if actor not in scene.actor_ids:
            raise ValueError("Unknown background actor")
        i = scene.actor_ids.index(actor)
        hits = np.flatnonzero(support[:, i])
        if not len(hits):
            raise ValueError("No spawn within simulation horizon")
        scene.presence[hits[0] :, i] = True
        if assignments[actor]["navigation_source"] == "uncertain_motion_braking":
            measured = (
                scene.metadata.get("first_observations", {}).get(actor, {}).get("velocity_world")
            )
            if measured is not None:
                measured = np.asarray(measured, float)
                if measured.shape != (3,) or not np.isfinite(measured).all():
                    raise ValueError(f"Invalid measured velocity: {actor}")
                # Braking starts from observed motion for all participant classes,
                # rather than acceleration implied by noisy position differences.
                scene.velocities[hits[0], i] = measured

        # Finite fallback reference for records and native initialization. The
        # reactive runtime never snaps back to it after taking ownership.
        for t in range(hits[0] + 1, len(scene.times)):
            if not support[t, i]:
                scene.poses[t, i] = scene.poses[t - 1, i]
                scene.velocities[t, i] = 0
    scene.metadata["reactive_background"] = {
        "version": 1,
        "lifetime": "first_annotation_to_horizon",
        "assignments": copy.deepcopy(assignments),
        "source_presence": support.tolist(),
    }
    scene.validate()
    return scene


def propose_logged_tasks(scene):
    """Use only the first continuous supported run as an offline navigation task.

    Moving tracks need >=0.5m of path. Short/stationary/unsupported tracks stay
    replay, with explicit rejection reasons, not silently promoted or dropped.
    """
    assignments, rejected = {}, {}
    support = np.stack([scene.presence_at(t) for t in range(len(scene.times))])
    for i, actor in enumerate(scene.actor_ids):
        if scene.controlled[i]:
            continue
        hits = np.flatnonzero(support[:, i])
        if scene.kinds[i] not in (VEHICLE, PEDESTRIAN, CYCLIST) or len(hits) < 2:
            rejected[actor] = "unsupported_kind_or_insufficient_support"
            continue
        end = next((j for j in range(hits[0] + 1, len(support)) if not support[j, i]), len(support))
        evidence = scene.metadata.get("source_motion_evidence", {})
        motion = (
            evidence.get("actors", {}).get(actor, {})
            if (
                evidence.get("version") == 1
                and evidence.get("scope") == "first_contiguous_run_privileged_navigation"
            )
            else {}
        )
        measured = scene.metadata.get("first_observations", {}).get(actor, {}).get("velocity_world")
        if measured is not None:
            velocity = np.asarray(measured, float)[:2]
            speed = float(np.linalg.norm(velocity))
            contradicted = (
                motion.get("samples", 0) >= 2
                and motion.get("finite_velocity_samples") == motion.get("samples")
                and motion.get("last_timestamp_us", 0) - motion.get("first_timestamp_us", 0)
                >= 2_000_000
                and 0 <= motion.get("reported_travel_m", float("inf")) <= 0.25
                and motion.get("position_path_length_m", 0) >= 1.0
                and np.isfinite(speed)
                and speed <= 0.5
            )
            if contradicted:
                start = scene.poses[hits[0], i, :2]
                yaw = scene.poses[hits[0], i, 3]
                direction = (
                    velocity / speed if speed > 1e-6 else np.array([np.cos(yaw), np.sin(yaw)])
                )
                assignments[actor] = {
                    "points": [
                        start.tolist(),
                        (start + direction * max(0.01, speed * speed / 10 + 0.3)).tolist(),
                    ],
                    "desired_speed": 0.0,
                    "navigation_source": "uncertain_motion_braking",
                }
                rejected[actor] = "position_motion_contradicts_reported_travel"
                continue
        points = scene.poses[hits[0] : end, i, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
        points = points[keep]
        if len(points) < 2 or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() < 0.5:
            rejected[actor] = "no_supported_moving_path"
            continue
        speed = np.median(np.linalg.norm(scene.velocities[hits[0] : end, i, :2], axis=1))
        if speed < (0.1 if scene.kinds[i] == PEDESTRIAN else 0.5) or speed > (
            2.5 if scene.kinds[i] == PEDESTRIAN else 20
        ):
            rejected[actor] = "unsupported_speed"
            continue
        if scene.kinds[i] in (VEHICLE, CYCLIST):
            # Box-axis continuity is not front-direction evidence. Require both
            # measured motion and initial path tangent to agree with box yaw.
            yaw = scene.poses[hits[0], i, 3]
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            velocity = scene.velocities[hits[0], i, :2]
            tangent = points[1] - points[0]
            if (
                np.linalg.norm(velocity) < 0.5
                or velocity @ forward < 0.5 * np.linalg.norm(velocity)
                or tangent @ forward < 0.5 * np.linalg.norm(tangent)
            ):
                rejected[actor] = "unverified_forward_direction"
                continue
        assignments[actor] = {
            "points": points.tolist(),
            "desired_speed": float(speed),
            "navigation_source": "logged_path_privileged",
        }
    return assignments, rejected


def audit_presence(scene):
    support = np.asarray(
        scene.metadata.get("reactive_background", {}).get(
            "source_presence", [scene.presence_at(t) for t in range(len(scene.times))]
        )
    )
    rows = []
    for i, actor in enumerate(scene.actor_ids):
        mask = support[:, i]
        edges = np.diff(np.r_[False, mask, False].astype(int))
        intervals = [
            {
                "start_tick": int(a),
                "end_tick_exclusive": int(b),
                "duration_seconds": float((b - a) * 0.1),
            }
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        ]
        rows.append(
            {
                "actor_id": actor,
                "kind": int(scene.kinds[i]),
                "intervals": intervals,
                "short_run": any(x["end_tick_exclusive"] - x["start_tick"] <= 6 for x in intervals),
                "internal_gaps": max(0, len(intervals) - 1),
                "reactive": actor
                in scene.metadata.get("reactive_background", {}).get("assignments", {}),
            }
        )
    return {"version": 1, "membership_is_annotation_support": True, "actors": rows}


def configure_observed_background(scene, *, extrapolation_seconds=2.0):
    """Retain short vehicle tracks using measured velocity, never infer missing velocity as zero.

    This is explicit simulator ownership, not interpolation of missing truth.
    Extrapolated navigation is bounded to two seconds of travel before braking;
    the physical object remains until episode end. No ID merging or early spawn.
    """
    if not np.isfinite(extrapolation_seconds) or not 0 < extrapolation_seconds <= 3:
        raise ValueError("Observed-motion intent requires a budget in (0,3] seconds")
    source = copy.deepcopy(scene)
    tasks, rejected = propose_logged_tasks(source)
    inferred = {}
    for i, actor in enumerate(source.actor_ids):
        if source.controlled[i] or actor in tasks or source.kinds[i] != VEHICLE:
            continue
        observation = source.metadata.get("first_observations", {}).get(actor, {})
        measured = observation.get("velocity_world")
        if measured is None:
            continue
        velocity = np.asarray(measured, float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("Invalid measured source velocity")
        speed = float(np.linalg.norm(velocity[:2]))
        hits = np.flatnonzero([source.presence_at(t)[i] for t in range(len(source.times))])
        if not len(hits) or not (0 <= speed <= 0.2 or 0.5 <= speed <= 20):
            continue
        spawn = int(hits[0])
        yaw = source.poses[spawn, i, 3]
        stationary = speed <= 0.2
        direction = np.array([np.cos(yaw), np.sin(yaw)]) if stationary else velocity[:2] / speed
        if not stationary and direction @ np.array([np.cos(yaw), np.sin(yaw)]) < 0.5:
            rejected[actor] = "source_velocity_disagrees_with_forward_axis"
            continue
        # Only fill missing intent from genuinely short support; a longer observed
        # path rejected for orientation/behavior must not bypass its qualification.
        first_end = next(
            (t for t in range(spawn + 1, len(source.times)) if not source.presence_at(t)[i]),
            len(source.times),
        )
        if first_end - spawn > 6:
            continue
        start = source.poses[spawn, i, :2]
        distance = (
            0.01 if stationary else max(speed * extrapolation_seconds, speed * speed / 10 + 0.3)
        )
        tasks[actor] = {
            "points": [start.tolist(), (start + direction * distance).tolist()],
            "desired_speed": 0.0 if stationary else speed,
            "navigation_source": "observed_stationary_hold"
            if stationary
            else "observed_velocity_extrapolation",
        }
        source.velocities[spawn:first_end, i] = velocity
        inferred[actor] = {
            "spawn_tick": spawn,
            "source": observation,
            "travel_budget_seconds": 0.0 if stationary else extrapolation_seconds,
            "inference": "measured_low_speed_hold"
            if stationary
            else "measured_velocity_navigation",
            "path_length_m": distance,
            "source_truth_after_gap": False,
            "lifetime": "simulation_owned_until_horizon",
        }
        rejected.pop(actor, None)
    if not tasks:
        raise ValueError("No supported background navigation or measured-velocity task")
    result = configure_background(source, tasks)
    result.metadata["observed_motion_inference"] = inferred
    return result, rejected


def boundary_identity(scene) -> bytes:
    """Bind explicit boundary semantics identically in world and replay artifacts."""
    policy = scene.metadata.get("road_boundary_policy")
    if policy is None:
        return b""
    if policy != "nuplan_strict_corners_v1":
        raise ValueError("Unsupported road boundary policy")
    return policy.encode("ascii")
