"""Stateless IDM following and geometric route tracking for short local tasks.

IDM: https://traffic-simulation.de/info/info_IDM.html
Not a right-of-way, signal, lane-change or collision-avoidance guarantee.
"""

from dataclasses import asdict, dataclass

import numpy as np

from driversi.contracts.road import Route
from driversi.contracts.transition import Observation


@dataclass(frozen=True)
class FollowingConfig:
    desired_speed: float = 8.0
    headway: float = 1.5
    standstill_gap: float = 2.0
    acceleration: float = 1.5
    deceleration: float = 2.0
    lookahead_seconds: float = 1.0
    minimum_lookahead_m: float = 5.0
    lateral_acceleration_limit: float = 2.5
    curvature_preview_seconds: float = 3.0
    recovery_heading_radians: float = .35
    recovery_distance_m: float = .75
    recovery_speed_floor_mps: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(value) and value > 0 for value in asdict(self).values()):
            raise ValueError("Following parameters must be positive and finite")
        if self.lateral_acceleration_limit > 4 or self.recovery_heading_radians >= np.pi/2:
            raise ValueError("Lateral limit must fit native bounds; recovery heading must be below pi/2")


def _preview_speed(route: Route, progress: float, speed: float, acceleration: float,
                   config: FollowingConfig) -> tuple[float, float]:
    """Geometric speed envelope with room for the native braking jerk ramp.

    Uses only the assigned route, sampled at 1 m. Curvature is a three-point
    circumcircle estimate; it is not a guarantee for discontinuous polylines.
    The 0.1 s clock and -15 m/s^3 braking jerk belong to the pinned action space.
    """
    horizon = max(2*config.minimum_lookahead_m, max(0, speed)*config.curvature_preview_seconds)
    arcs = np.minimum(progress + np.arange(0, horizon+1, 1), route.distances[-1]-1)
    arcs = np.maximum(arcs, 1)
    distances = route.distances
    samples = np.stack((arcs-1, arcs, arcs+1))
    before, points, after = np.stack([np.interp(samples, distances, route.points[:, axis])
                                    for axis in (0, 1)], axis=-1)
    left, right = points-before, after-points
    denominator = np.linalg.norm(left, axis=1)*np.linalg.norm(right, axis=1)*np.linalg.norm(after-before, axis=1)
    curvature = 2*np.abs(left[:, 0]*right[:, 1]-left[:, 1]*right[:, 0])/np.maximum(denominator, 1e-6)
    curve_speeds_squared = config.lateral_acceleration_limit/np.maximum(curvature, 1e-6)
    ramp_time = .1 + max(0, acceleration + config.deceleration)/15
    ramp_distance = max(0, speed)*ramp_time + .5*max(0, acceleration)*ramp_time**2
    usable_distance = np.maximum(0, arcs-progress-ramp_distance)
    speed_limit = np.sqrt(curve_speeds_squared + 2*config.deceleration*usable_distance)
    return float(speed_limit.min()), float(curvature.max())


def idm_acceleration(speed: float, desired_speed: float, gap: float | None,
                     closing_speed: float, config: FollowingConfig) -> float:
    speed = max(0.0, speed)
    free = 1 - (speed / max(desired_speed, .1))**4
    interaction = 0.0
    if gap is not None:
        desired_gap = config.standstill_gap + max(
            0.0, speed * config.headway + speed * closing_speed / (2*np.sqrt(config.acceleration*config.deceleration)))
        interaction = (desired_gap / max(gap, .1))**2
    return float(np.clip(config.acceleration * (free - interaction), -5, 2.5))


def _next_acceleration(current: float, jerk: np.ndarray, bounds) -> np.ndarray:
    proposed = current + jerk * .1
    return np.where(current * proposed < 0, 0, np.clip(proposed, *bounds))


def _actuator_choice(longitudinal: float, lateral: float, current: np.ndarray, speed: float,
                     *, steering: float | None = None, wheelbase: float | None = None, forward_only=False) -> tuple[int, dict]:
    """Predict pinned discrete actuators; do not change the world's dynamics."""
    if (steering is None) != (wheelbase is None):
        raise ValueError("Steering and wheelbase must be supplied together")
    if wheelbase is not None and (not np.isfinite([steering, wheelbase]).all() or wheelbase <= 0):
        raise ValueError("Steering must be finite and wheelbase positive")
    long_values = _next_acceleration(current[0], np.asarray([-15, -4, 0, 4]), (-5, 2.5))
    lat_values = _next_acceleration(current[1], np.asarray([-4, 0, 4]), (-4, 4))
    predicted_speed = speed + .05 * (current[0] + long_values)
    predicted_speed = np.where(speed*predicted_speed < 0, 0, np.clip(predicted_speed, -2, 20))
    # Discrete trapezoidal speed loss while releasing braking with +4 jerk.
    # Checking only the next tick releases braking too late at low speed.
    braking = np.maximum(0, -long_values)
    release_steps = np.ceil(braking/.4)
    release_loss = .1*((release_steps-.5)*braking-.2*release_steps*(release_steps-1))
    release_speed = predicted_speed-release_loss
    if forward_only:
        predicted_speed = np.maximum(0, predicted_speed)
        release_speed = np.maximum(0, release_speed)
    feasible = release_speed >= -1e-8
    if forward_only and 0 < speed < .2 and longitudinal < 0:
        longitudinal = min(longitudinal, -.4)
    long_cost = abs(long_values-longitudinal)
    if feasible.any():
        long_index = min(np.flatnonzero(feasible), key=lambda i: (long_cost[i], i != 2, i))
    else:
        # Some externally supplied states already cannot avoid reverse under
        # the jerk limit. Report it; choose the least reverse-prone action.
        long_index = min(range(4), key=lambda i: (-release_speed[i], long_cost[i], i != 2, i))
    if wheelbase is not None:
        requested_steering = np.arctan(lat_values/max(predicted_speed[long_index]**2, 1e-5)*wheelbase)
        next_steering = np.clip(steering+np.clip(requested_steering-steering, -.06, .06), -.55, .55)
        lat_values = predicted_speed[long_index]**2*np.tan(next_steering)/wheelbase
    lat_index = min(range(3), key=lambda i: (abs(lat_values[i] - lateral), i != 1, i))
    return int(long_index*3+lat_index), {
        "forward_release_feasible": bool(feasible.any()),
        "predicted_minimum_release_speed_mps": float(release_speed[long_index]),
        "predicted_next_lateral_acceleration": float(lat_values[lat_index]),
    }


def quantize_jerk(longitudinal: float, lateral: float, current: np.ndarray, speed: float,
                  *, steering: float | None = None, wheelbase: float | None = None) -> int:
    """Closest feasible jerk action, preferring zero jerk ties.

    Supplying both steering and wheelbase enables lateral actuator prediction;
    the legacy four-argument form quantizes lateral acceleration directly.
    """
    return _actuator_choice(longitudinal, lateral, current, speed,
                            steering=steering, wheelbase=wheelbase)[0]


class RouteFollower:
    """No hidden memory/RNG: same current observation and routes => same action."""

    def __init__(self, routes: dict[str, Route], config: FollowingConfig | None = None, *, forward_only=False):
        self.forward_only = bool(forward_only)
        self.routes = dict(routes)
        self.config = config or FollowingConfig()

    def describe(self) -> dict:
        return {"forward_only": self.forward_only, "controller": "route-idm-v2", "parameters": asdict(self.config),
                "tracking": {"speed": "signed_vehicle_speed", "curvature_spacing_m": 1.0,
                             "preview": "assigned_route_only", "braking_jerk_mps3": -15.0,
                             "release_jerk_mps3": 4.0, "forward_release_tolerance_mps": 1e-8,
                             "steering_rate_radians_per_second": .6, "steering_limit_radians": .55,
                             "wheelbase_length_ratio": .6,
                             "actuator_prediction": "discrete-jerk-and-steering-v1",
                             "infeasible_forward_release": "report_and_minimize_reverse_speed",
                             "route_exhaustion": "nearest_endpoint_and_positive_terminal_projection",
                             "exhausted_reference": "terminal_tangent_extension_with_braking",
                             "exhausted_target_acceleration_mps2": -5.0,
                             "dt_seconds": .1, "stateful": False},
                "routes": {actor: route.to_dict() for actor, route in self.routes.items()}}

    def commands(self, observation: Observation) -> list[dict]:
        if observation.metric is None:
            raise ValueError("RouteFollower requires metric-v1 current observations")
        commands = []
        metric = observation.metric
        for index, actor in enumerate(observation.actor_ids):
            if actor not in self.routes:
                raise ValueError(f"No explicit route for {actor}")
            route = self.routes[actor]
            ego = metric.ego[index]
            progress, distance, tangent = route.project(ego[:2])
            direction = np.asarray([np.cos(tangent), np.sin(tangent)])
            path_speed = float(np.dot(ego[3:5], direction))
            signed = np.dot(ego[3:5], [np.cos(ego[2]), np.sin(ego[2])])
            speed = float(np.copysign(np.linalg.norm(ego[3:5]), signed))
            ego_angle = float(np.arctan2(np.sin(ego[2]-tangent), np.cos(ego[2]-tangent)))
            ego_half_long = .5*(abs(np.cos(ego_angle))*ego[5] + abs(np.sin(ego_angle))*ego[6])
            ego_half_lat = .5*(abs(np.sin(ego_angle))*ego[5] + abs(np.cos(ego_angle))*ego[6])
            lookahead_distance = max(self.config.minimum_lookahead_m,
                                     max(0, speed)*self.config.lookahead_seconds)
            end_direction = route.points[-1]-route.points[-2]
            end_direction = end_direction/np.linalg.norm(end_direction)
            signed_end_distance = float((ego[:2]-route.points[-1]) @ end_direction)
            exhausted = progress >= route.distances[-1]-1e-6 and signed_end_distance > 0
            if exhausted:
                # The task no longer specifies the road ahead. Continue its
                # terminal tangent while braking, never chase the point behind.
                lookahead = route.points[-1]+(signed_end_distance+lookahead_distance)*end_direction-ego[:2]
            else:
                lookahead = route.at(progress+lookahead_distance)-ego[:2]
            lateral = -np.sin(ego[2])*lookahead[0] + np.cos(ego[2])*lookahead[1]
            curvature = 2*lateral/max(float(lookahead @ lookahead), 1)
            tracking_speed = np.sqrt(self.config.lateral_acceleration_limit/max(abs(curvature), 1e-6))
            desired_speed = min(self.config.desired_speed, route.speed_limit_mps or self.config.desired_speed)
            preview_speed, preview_curvature = _preview_speed(route, progress, speed, ego[7], self.config)
            desired_speed = min(desired_speed, preview_speed, tracking_speed)
            recovery = abs(ego_angle) > self.config.recovery_heading_radians or distance > self.config.recovery_distance_m
            if recovery:
                # Keep moving to recover heading, but do not accelerate while
                # displaced. A stationary vehicle may start at a low crawl speed.
                desired_speed = min(desired_speed, max(self.config.recovery_speed_floor_mps, speed))
            target = idm_acceleration(speed, desired_speed, None, 0, self.config)
            lead, gap, closing = None, None, 0.0
            for j in np.flatnonzero(metric.visible[index]):
                neighbor = metric.neighbors[index, j]
                arc, lateral_distance, _ = route.project(neighbor[:2])
                angle = neighbor[2] - tangent
                half_long = .5*(abs(np.cos(angle))*neighbor[5] + abs(np.sin(angle))*neighbor[6])
                half_lat = .5*(abs(np.sin(angle))*neighbor[5] + abs(np.cos(angle))*neighbor[6])
                if arc <= progress or lateral_distance > half_lat + ego_half_lat:
                    continue
                candidate_gap = arc - progress - ego_half_long - half_long
                if gap is None or candidate_gap < gap:
                    gap = float(candidate_gap)
                    closing = float(path_speed - np.dot(neighbor[3:5], direction))
                    lead = metric.neighbor_ids[index][j]
            if gap is not None:
                target = min(target, idm_acceleration(speed, desired_speed, gap, closing, self.config))
            remaining = float(route.distances[-1] - progress - ego_half_long)
            endpoint_gap = remaining
            if self.forward_only:
                # Route endpoints are vehicle-centre task goals, not obstacle
                # bumpers. Stop at 0.5 m from the centre endpoint so a clear
                # approach can enter the native 2 m completion radius.
                remaining = float(route.distances[-1] - progress)
                endpoint_gap = remaining + self.config.standstill_gap - .5
            target = min(target, idm_acceleration(speed, desired_speed, endpoint_gap, speed, self.config))
            if not self.forward_only and speed < .2 and target < 0:
                target = 0.0
            terminal_hold = self.forward_only and speed < .2 and remaining <= .5
            if terminal_hold:
                target = min(target, -.4)
            if exhausted:
                desired_speed, target = 0.0, -5.0
            anticipated_speed = max(0, speed + .1*ego[7])
            target_lat = float(np.clip(anticipated_speed**2 * curvature,
                                       -self.config.lateral_acceleration_limit,
                                       self.config.lateral_acceleration_limit))
            action, actuator = _actuator_choice(target, target_lat, ego[7:9], speed,
                                                steering=ego[9], wheelbase=.6*ego[5], forward_only=self.forward_only)
            if not observation.alive[index]:
                action = 7
                actuator = {key: None for key in actuator}
            commands.append({"actor": actor, "action": action, "desired_acceleration": target,
                             "desired_lateral_acceleration": target_lat, "lead_id": lead,
                             "bumper_gap_m": gap, "ttc_seconds": gap/closing if gap is not None and gap > 0 and closing > 1e-6 else None,
                             "route_progress_m": progress, "route_distance_m": distance,
                             "actual_speed_mps": speed, "path_speed_mps": path_speed,
                             "target_speed_mps": desired_speed, "heading_error_radians": ego_angle,
                             "recovery_mode": bool(recovery), "preview_curvature_per_m": preview_curvature,
                             "curve_speed_limit_mps": preview_speed,
                             "tracking_speed_limit_mps": float(tracking_speed),
                             "route_exhausted": bool(exhausted),
                             "route_end_signed_distance_m": signed_end_distance,
                             "tracking_reference": "terminal_tangent" if exhausted else "route_lookahead",
                             "lookahead_forward_m": float(lookahead @ np.asarray([np.cos(ego[2]), np.sin(ego[2])])),
                             **actuator,
                             "remaining_m": remaining})
        return commands

    def act(self, observation: Observation) -> np.ndarray:
        return np.asarray([command["action"] for command in self.commands(observation)], dtype=np.int32)
