"""Staged route/box conflict baseline inspired by CARLA Traffic Manager.

Independent implementation; no CARLA runtime. Static route intents are privileged
navigation input, never logged future motion. This is not traffic-law certification.
"""

from dataclasses import asdict, dataclass

import numpy as np

from driversi.agents.traffic_control.following import (
    FollowingConfig,
    RouteFollower,
    _actuator_choice,
    idm_acceleration,
)


@dataclass(frozen=True)
class TrafficConfig:
    horizon_seconds: float = 4.0
    margin_m: float = 0.6
    emergency_seconds: float = 1.0
    route_preview_m: float = 40.0

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Traffic limits must be positive and finite")
        if (
            self.horizon_seconds > 6
            or self.route_preview_m > 60
            or self.emergency_seconds > self.horizon_seconds
        ):
            raise ValueError("Traffic prediction exceeds supported bounded horizon")


def swept_contact(ego, other, velocity, horizon, margin=0.0):
    """Exact swept SAT for fixed-heading boxes moving at constant velocity.

    No point sampling/tunnelling; acceleration and rotation are not predicted.
    Metric rows: x,y,yaw,vx,vy,length,width. Return first overlap time or None.
    """
    delta = other[:2] - ego[:2]
    dv = other[3:5] - velocity
    e = np.array([np.cos(ego[2]), np.sin(ego[2])])
    o = np.array([np.cos(other[2]), np.sin(other[2])])
    el, ol = np.array([-e[1], e[0]]), np.array([-o[1], o[0]])
    enter, leave = 0.0, float(horizon)
    for axis in (e, el, o, ol):
        extent = (
            abs(axis @ e) * ego[5]
            + abs(axis @ el) * ego[6]
            + abs(axis @ o) * other[5]
            + abs(axis @ ol) * other[6]
        ) / 2 + margin
        p, v = float(delta @ axis), float(dv @ axis)
        if abs(v) < 1e-10:
            if abs(p) > extent:
                return None
        else:
            a, b = sorted(((-extent - p) / v, (extent - p) / v))
            enter, leave = max(enter, a), min(leave, b)
            if enter > leave:
                return None
    return enter if leave >= 0 and enter <= horizon else None


def parallel_clearance_margin(ego, other, margin):
    """Never inflate physically separated parallel lanes into permanent overlap."""
    angle=other[2]-ego[2]
    if np.cos(angle) < .995:
        return margin
    lateral=np.array([-np.sin(ego[2]), np.cos(ego[2])])
    separation=abs(float((other[:2]-ego[:2])@lateral))
    half_width=.5*(ego[6]+abs(np.cos(angle))*other[6]+abs(np.sin(angle))*other[5])
    clearance=separation-half_width
    return min(margin,.5*clearance) if clearance>0 else margin


def route_crossing(route, ego, other_route, other, preview, margin=0.6):
    """First non-parallel centreline intersection within bounded assigned routes.

    Keep a vehicle-length tail so an actor crossing the centre does not instantly
    release the corridor while its body still occupies it. Parallel merges retain
    the existing following and swept-box stages instead of invented right-of-way.
    """

    def segments(r, state):
        s, distance, _ = r.project(state[:2])
        if distance > 2.0:
            return None
        lo, hi = max(0, s - state[5] - 2), min(r.distances[-1], s + preview)
        arcs = np.unique(np.r_[lo, r.distances[(r.distances > lo) & (r.distances < hi)], hi])
        points = np.column_stack([np.interp(arcs, r.distances, r.points[:, k]) for k in (0, 1)])
        return s, arcs, points

    a, b = segments(route, ego), segments(other_route, other)
    if a is None or b is None:
        return None
    sa, aa, pa = a
    sb, ab, pb = b
    if len(pa) < 2 or len(pb) < 2:
        return None
    p, q = pa[:-1, None, :], pb[None, :-1, :]
    u, v = np.diff(pa, axis=0)[:, None, :], np.diff(pb, axis=0)[None, :, :]

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    den = cross(u, v)
    length = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    valid = abs(den) > np.maximum(1e-9, 0.34 * length)
    safe = np.where(valid, den, 1)
    ta, tb = cross(q - p, v) / safe, cross(q - p, u) / safe
    valid &= (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)
    rows = []
    for i, j in zip(*np.nonzero(valid)):
        da = float(aa[i] + ta[i, j] * (aa[i + 1] - aa[i]) - sa)
        db = float(ab[j] + tb[i, j] * (ab[j + 1] - ab[j]) - sb)
        clearance_a, clearance_b = (
            (ego[5] + other[6]) / 2 + margin,
            (other[5] + ego[6]) / 2 + margin,
        )
        if da >= -clearance_a and db >= -clearance_b:
            rows.append((max(da, 0) + max(db, 0), da, db, clearance_a, clearance_b))
    return min(rows)[1:] if rows else None


class TrafficController(RouteFollower):
    """Stateless staged controller, opt-in cooperation; never edits external actions."""

    def __init__(self, routes, config=None, *, cooperative_ids=(), traffic=None, forward_only=False):
        super().__init__(routes, config or FollowingConfig(), forward_only=forward_only)
        self.cooperative_ids = frozenset(cooperative_ids)
        if not self.cooperative_ids <= set(routes):
            raise ValueError("Cooperative actors need explicit route intents")
        self.traffic = traffic or TrafficConfig()

    def describe(self):
        result = super().describe()
        result.update(
            controller="route-conflict-v1",
            traffic=asdict(self.traffic),
            cooperative_ids=sorted(self.cooperative_ids),
            priority="occupied_crossing_then_stable_actor_id_not_legal_right_of_way",
            prediction="assigned_route_crossings_and_current_constant_velocity_boxes",
            stateful=False,
            trainable=False,
        )
        return result

    def commands(self, observation):
        commands = super().commands(observation)
        metric, cfg = observation.metric, self.traffic
        for i, command in enumerate(commands):
            actor, ego = command["actor"], metric.ego[i]
            command.update(
                traffic_hazards=[],
                traffic_priority="none",
                traffic_modified=False,
                traffic_observation_tick=observation.step,
            )
            if not observation.alive[i]:
                continue
            speed = max(0.0, command["actual_speed_mps"])
            target = command["desired_acceleration"]
            for j in np.flatnonzero(metric.visible[i]):
                other, oid = metric.neighbors[i, j], metric.neighbor_ids[i][j]
                cooperative = actor in self.cooperative_ids and oid in self.cooperative_ids
                priority = None
                crossing = (
                    route_crossing(
                        self.routes[actor],
                        ego,
                        self.routes[oid],
                        other,
                        cfg.route_preview_m,
                        cfg.margin_m,
                    )
                    if cooperative
                    else None
                )
                if crossing is not None:
                    da, db, ca, cb = crossing
                    # An actor already occupying the conflict clears it first.
                    # Otherwise stable IDs avoid mutual yielding/ordering dependence.
                    inside_a, inside_b = abs(da) <= ca, abs(db) <= cb
                    priority = (
                        actor
                        if inside_a and not inside_b
                        else oid
                        if inside_b and not inside_a
                        else min(actor, oid)
                    )
                    if priority != actor:
                        gap = max(0.05, da - ca - cfg.margin_m)
                        limit = idm_acceleration(
                            speed, max(command["target_speed_mps"], 0.1), gap, speed, self.config
                        )
                        target = min(target, limit)
                        command["traffic_hazards"].append(
                            {
                                "actor": oid,
                                "reason": "crossing_yield",
                                "gap_m": gap,
                                "priority_actor": priority,
                                "target_acceleration": limit,
                            }
                        )
                        command["traffic_priority"] = "yield"
                    elif command["traffic_priority"] == "none":
                        command["traffic_priority"] = "proceed_subject_to_collision_check"
                horizon = cfg.emergency_seconds if priority == actor else cfg.horizon_seconds
                # Use current motion, not commands from another policy or its log future.
                velocity = ego[3:5]
                margin = parallel_clearance_margin(ego, other, cfg.margin_m) if self.forward_only else cfg.margin_m
                when = swept_contact(ego, other, velocity, horizon, margin)
                if when is not None:
                    forward = np.array([np.cos(ego[2]), np.sin(ego[2])])
                    ahead = float((other[:2] - ego[:2]) @ forward)
                    # Do not brake for a car catching up from directly behind.
                    aligned = np.cos(other[2] - ego[2]) > 0.94
                    if ahead < -(ego[5] + other[5]) / 2 and aligned:
                        continue
                    gap = max(0.05, speed * when - cfg.margin_m)
                    limit = idm_acceleration(
                        speed, max(command["target_speed_mps"], 0.1), gap, speed, self.config
                    )
                    target = min(target, limit)
                    command["traffic_hazards"].append(
                        {
                            "actor": oid,
                            "reason": "swept_box_conflict",
                            "time_seconds": when,
                            "gap_m": gap,
                            "target_acceleration": limit,
                        }
                    )
            if not self.forward_only and speed < 0.2 and target < 0:
                target = 0.0
            if target < command["desired_acceleration"] - 1e-12:
                command["traffic_modified"] = True
                command["unconstrained_acceleration"] = command["desired_acceleration"]
                command["desired_acceleration"] = target
                command["action"], actuator = _actuator_choice(
                    target,
                    command["desired_lateral_acceleration"],
                    ego[7:9],
                    command["actual_speed_mps"],
                    steering=ego[9],
                    wheelbase=0.6 * ego[5],
                    forward_only=self.forward_only,
                )
                command.update(actuator)
        return commands
