"""Deterministic current-state background interaction baseline (not a learned policy).

Assigned paths are navigation intent. No neighbor future trajectories are read.
All proposals are computed from the same pre-step world; native applies them jointly.
"""

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, idm_acceleration
from driversi.agents.traffic_control.traffic import parallel_clearance_margin, route_crossing, swept_contact
from driversi.contracts.road import Route
from driversi.contracts.scene import PEDESTRIAN, VEHICLE
from driversi.data.participants.background import validate_background


def first_conflict(index, state, present, dimensions, preferred_velocity, horizon=3.0):
    """Constant-velocity forecast with oriented box extents in ego axes.

    Conservative for angled neighbors. Discrete forecast, no guaranteed safety.
    Includes stopped learners and passive objects; excludes absent identities.
    """
    ego = state[index]
    direction = np.array([np.cos(ego[3]), np.sin(ego[3])])
    lateral = np.array([-direction[1], direction[0]])
    best = None
    for j in np.flatnonzero(present):
        if j == index or np.linalg.norm(state[j, :2] - ego[:2]) > 50:
            continue
        angle = state[j, 3] - ego[3]
        long_extent = (
            0.5
            * (
                dimensions[index, 0]
                + abs(np.cos(angle)) * dimensions[j, 0]
                + abs(np.sin(angle)) * dimensions[j, 1]
            )
            + 0.4
        )
        lat_extent = (
            0.5
            * (
                dimensions[index, 1]
                + abs(np.sin(angle)) * dimensions[j, 0]
                + abs(np.cos(angle)) * dimensions[j, 1]
            )
            + 0.25
        )
        times = np.arange(0.1, horizon + 0.05, 0.1)
        relative = state[j, :2] - ego[:2] + times[:, None] * (state[j, 4:6] - preferred_velocity)
        hit = (abs(relative @ direction) <= long_extent) & (abs(relative @ lateral) <= lat_extent)
        if hit.any():
            t = float(times[np.flatnonzero(hit)[0]])
            if best is None or (t, j) < best:
                best = (t, int(j))
    return best


class ReactiveBackground:
    """State is entirely world-owned; no RNG, recurrent memory or mutable path index."""

    def __init__(self, scene):
        self.tasks = validate_background(scene)
        self.scene = scene
        self.routes = {a: Route(t["points"]) for a, t in self.tasks.items()}
        self.indices = {a: scene.actor_ids.index(a) for a in self.tasks}
        self.fleet = scene.metadata.get("vehicle_fleet")
        self.continuity = bool(scene.metadata.get("traffic_continuity"))
        self.fleet_v2 = bool(self.fleet and self.fleet.get("version") == 2)
        self.cooperative = {
            a
            for a, task in self.tasks.items()
            if scene.kinds[self.indices[a]] == VEHICLE and task["desired_speed"] > 0
        }

    def vehicle_conflict(self, index, state, present, preferred):
        """All vehicle decisions read one frozen state; no other's new action leaks."""
        actor = self.scene.actor_ids[index]
        metric = np.column_stack((state[:, :2], state[:, 3:6], self.scene.dimensions[:, :2]))
        ego = metric[index]
        best = None
        reason = None
        speed_limit = None
        for j in np.flatnonzero(present):
            if j == index or np.linalg.norm(metric[j, :2] - ego[:2]) > 50:
                continue
            other, oid = metric[j], self.scene.actor_ids[j]
            priority = None
            if self.continuity:
                # A following vehicle is responsible for its own longitudinal
                # braking. Braking the leader too creates an artificial cycle.
                # This does not disable any physical contact test.
                direction = np.array([np.cos(ego[2]), np.sin(ego[2])])
                ahead = float((other[:2]-ego[:2]) @ direction)
                behind = ahead < -(ego[5]+other[5])/2 - .5
                aligned = np.cos(other[2]-ego[2]) > .94
                if behind and aligned:
                    continue
            if actor in self.cooperative and oid in self.cooperative:
                crossing = route_crossing(self.routes[actor], ego, self.routes[oid], other, 40)
                if crossing is not None:
                    da, db, ca, cb = crossing
                    inside_a, inside_b = abs(da) <= ca, abs(db) <= cb
                    priority = (
                        actor
                        if inside_a and not inside_b
                        else oid
                        if inside_b and not inside_a
                        else min(actor, oid)
                    )
                    if priority != actor:
                        # Route intent can anticipate a stationary car starting, unlike velocity-only checks.
                        when = max(0.0, da - ca) / max(float(np.linalg.norm(preferred)), 0.1)
                        if when <= 4 and (best is None or (when, j) < best):
                            best, reason = (when, int(j)), "cooperative_crossing_yield"
                            speed_limit = float(np.sqrt(3.0 * max(0.0, da - ca - 0.6)))
            # Declared cooperative crossing uses a distance-based approach speed,
            # with immediate swept collision checks; unknown actors retain 3s checks.
            horizon = 1.0 if priority is not None else 3.0
            margin = parallel_clearance_margin(ego, other, .4) if self.continuity else .4
            when = swept_contact(ego, other, preferred, horizon, margin)
            if self.fleet_v2:
                # A desired braking/turning velocity is not an instantaneous change
                # of momentum. Check actual motion as well as navigation intent.
                actual = swept_contact(ego, other, ego[3:5], horizon, margin)
                if actual is not None:
                    when = actual if when is None else min(when, actual)
            if when is not None and (best is None or (when, j) < best):
                best, reason = (float(when), int(j)), "swept_box_conflict"
                speed_limit = 0.0
        return best, reason, speed_limit

    def propose(self, state, present, next_present):
        rows, diagnostics = [], []
        for actor, task in self.tasks.items():
            i = self.indices[actor]
            if not present[i] or not next_present[i]:
                continue  # New entrants use the immutable logged spawn at this step.
            current = state[i]
            pedestrian = self.scene.kinds[i] == PEDESTRIAN
            route = self.routes[actor]
            progress, _, _ = route.project(current[:2])
            speed = float(np.linalg.norm(current[4:6]))
            acceleration, deceleration = (1.5, 3.0) if pedestrian else (1.5, 5.0)
            remaining = max(0.0, float(route.distances[-1] - progress))
            desired = min(
                task["desired_speed"], np.sqrt(2 * deceleration * max(0, remaining - 0.15 - (speed*.1 if self.continuity else 0)))
            )
            target = route.at(progress + max(0.5 if pedestrian else 2.0, speed)) - current[:2]
            heading = np.arctan2(target[1], target[0])
            support_exhausted = remaining <= 0.15
            if (self.fleet_v2 or self.continuity) and support_exhausted:
                # Never turn back towards a support endpoint passed under inertia.
                # Continue bounded braking without fabricating a new route.
                heading = current[3]
                desired = 0.0
            error = float(np.arctan2(np.sin(heading - current[3]), np.cos(heading - current[3])))
            if not pedestrian:
                curvature = 2 * np.sin(error) / max(np.linalg.norm(target), 1.0)
                desired = min(desired, np.sqrt(2.5 / max(abs(curvature), 1e-6)))
            following_id = None
            following_gap = None
            if self.continuity:
                # Maintain a time headway before a lead begins braking. TTC alone
                # cannot protect a close, equal-speed convoy from a later stop.
                tangent = np.asarray([np.cos(current[3]), np.sin(current[3])])
                neighbors = np.flatnonzero(present)
                projections = route.project_many(state[neighbors, :2])
                for j, projected in zip(neighbors, projections):
                    if i == j:
                        continue
                    other_progress, lateral_distance, _ = projected
                    angle = state[j,3]-current[3]
                    half_long = .5*(abs(np.cos(angle))*self.scene.dimensions[j,0]+abs(np.sin(angle))*self.scene.dimensions[j,1])
                    half_lat = .5*(abs(np.sin(angle))*self.scene.dimensions[j,0]+abs(np.cos(angle))*self.scene.dimensions[j,1])
                    gap = other_progress-progress-.5*self.scene.dimensions[i,0]-half_long
                    if other_progress > progress and lateral_distance <= half_lat+.5*self.scene.dimensions[i,1]:
                        if following_gap is None or gap < following_gap:
                            following_gap, following_id = float(gap), self.scene.actor_ids[j]
                            closing = speed-float(state[j,4:6]@tangent)
                if following_id is not None:
                    limit = idm_acceleration(speed,max(desired,.1),following_gap,closing,FollowingConfig())
                    desired = min(desired, max(0.,speed+.1*limit))
            preferred = desired * np.array([np.cos(heading), np.sin(heading)])
            if self.fleet and not pedestrian:
                conflict, conflict_reason, conflict_limit = self.vehicle_conflict(
                    i, state, present, preferred
                )
            else:
                conflict = first_conflict(i, state, present, self.scene.dimensions, preferred)
                conflict_reason = "sampled_current_state_conflict" if conflict else None
                conflict_limit = 0.0 if conflict else None
            if conflict is not None:
                desired = min(desired, conflict_limit)
            if self.continuity and speed < .02 and (conflict is not None or
                    (following_gap is not None and following_gap <= 2.05 and desired <= speed)):
                desired = 0.0  # Declared standstill deadband under a blocking constraint.
            next_speed = max(
                0.0, speed + np.clip(desired - speed, -deceleration * 0.1, acceleration * 0.1)
            )
            # A stopped actor never rotates. Pedestrian turn rate is a geometric
            # approximation; vehicle rate respects a bicycle steering envelope.
            max_yaw_rate = (
                1.5
                if pedestrian
                else min(0.6, next_speed * np.tan(0.55) / (0.6 * self.scene.dimensions[i, 0]))
            )
            yaw = current[3] + (
                np.clip(error, -max_yaw_rate * 0.1, max_yaw_rate * 0.1)
                if next_speed > 1e-4
                else 0.0
            )
            velocity = next_speed * np.array([np.cos(yaw), np.sin(yaw)])
            if self.fleet and task["navigation_source"] in {
                "uncertain_motion_braking",
                "observed_stationary_hold",
            }:
                # Ambiguous box axes are not a licence to rotate/reverse measured motion.
                yaw = current[3]
                velocity = current[4:6] * (next_speed / speed) if speed > 1e-8 else np.zeros(2)
            xy = current[:2] + 0.05 * (current[4:6] + velocity)
            rows.append([i, *xy, current[2], yaw, *velocity, 0.0])
            diagnostics.append(
                {
                    "actor": actor,
                    "controller": "reactive-pedestrian-v1" if pedestrian else "reactive-vehicle-v1",
                    "target_speed_mps": float(desired),
                    "following_id": following_id, "following_gap_m": following_gap,
                    "acceleration_mps2": float((next_speed - speed) / 0.1),
                    "conflict_id": self.scene.actor_ids[conflict[1]] if conflict else None,
                    "predicted_conflict_seconds": conflict[0] if conflict else None,
                    "remaining_path_m": remaining,
                    "endpoint_distance_m": float(np.linalg.norm(current[:2] - route.points[-1])),
                    "stopping_distance_exceeds_path": bool(
                        speed * speed / (2 * deceleration) > remaining
                    ),
                    "navigation_source": task["navigation_source"],
                    **(
                        {
                            "control_mode": self.fleet["roles"]
                            .get(actor, {})
                            .get("mode", ("stationary_hold" if task["navigation_source"] == "observed_stationary_hold" else "uncertain_brake" if task["navigation_source"] == "uncertain_motion_braking" else "route_feedback") if self.continuity else "pedestrian_feedback"),
                            "conflict_reason": conflict_reason,
                            "learning_slot": False,
                            "next_state": rows[-1][1:],
                            "actuator": "bounded_kinematic_traffic_v1",
                            **(
                                {
                                    "controller_version": "traffic-fleet-v2",
                                    "route_support_exhausted": bool(support_exhausted),
                                    "braking_distance_m": float(speed * speed / (2 * deceleration)),
                                    "prediction_velocities": "actual_and_intended",
                                }
                                if self.fleet_v2
                                else {}
                            ),
                        }
                        if self.fleet
                        else {}
                    ),
                }
            )
        return np.asarray(rows, dtype=np.float64).reshape(-1, 8), diagnostics
