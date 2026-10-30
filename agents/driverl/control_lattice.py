"""Finite recovery search from the actual state using the native bicycle dynamics.

Separate teacher backend; never edits model outputs, source poses or score rules.
"""

import gc
import itertools
import time
import numpy as np
from driversi_agents.driverl.repair import (
    PDMRouteRepair,
    nonreversing_tracking_control,
    select_admissible_proposal,
)


def schedules():
    feedback = (
        ("feedback", *v)
        for v in itertools.product((0.5, 1.0, 2.0, 3.0), (4.0, 8.0, 12.0), (-3.0, 0.0))
    )
    reference = (
        ("reference", *v)
        for v in itertools.product(
            (0.4, 0.6, 0.8, 1.0, 1.5, 2.0), (-6.0, -3.0, 0.0), (-0.3, 0.0, 0.3)
        )
    )
    return tuple(feedback) + tuple(reference)


def propagate_control(motion, ego, acceleration, rate):
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint

    v = ego.dynamic_car_state
    acceleration, rate = nonreversing_tracking_control(
        v.rear_axle_acceleration_2d.x, acceleration, rate, v.rear_axle_velocity_2d.x, 0.1
    )
    dynamic = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=ego.car_footprint.rear_axle_to_center_dist,
        rear_axle_velocity_2d=v.rear_axle_velocity_2d,
        rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
        tire_steering_rate=rate,
    )
    nxt = motion.propagate_state(ego, dynamic, TimePoint(100000))
    return nxt, [(acceleration - v.rear_axle_acceleration_2d.x) / 0.1, rate]


def rollout_feedback(motion, ego, path, schedule, tracker):
    from shapely.geometry import Point
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
    )

    cross_gain, response, brake = schedule
    states = [ego_state_to_state_array(ego)]
    actions = []
    first = None
    for tick in range(40):
        speed = ego.dynamic_car_state.rear_axle_velocity_2d.x
        heading = ego.rear_axle.heading
        wb = ego.car_footprint.vehicle_parameters.wheel_base
        front = ego.rear_axle.array[:2] + wb * np.array([np.cos(heading), np.sin(heading)])
        progress = float(path.project(Point(*front)))
        target = path.interpolate([progress], as_array=True)[0]
        error = float(np.dot(target[:2] - front, [-np.sin(heading), np.cos(heading)]))
        yaw = float(np.arctan2(np.sin(target[2] - heading), np.cos(target[2] - heading)))
        angle = yaw + np.arctan2(cross_gain * error, max(speed, 1.0))
        rate = (angle - ego.tire_steering_angle) * response
        ego, action = propagate_control(motion, ego, brake, rate)
        if first is None:
            first = ego
        states.append(ego_state_to_state_array(ego))
        actions.append(action)
    return np.asarray(states), np.asarray(actions), first


def rollout_reference(motion, ego, path, schedule, tracker):
    from shapely.geometry import Point
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
        state_array_to_ego_state,
    )

    duration, brake, target_offset = schedule
    speed = ego.dynamic_car_state.rear_axle_velocity_2d.x
    progress = float(path.project(Point(ego.rear_axle.x, ego.rear_axle.y)))
    foot = path.interpolate([progress], as_array=True)[0]
    yaw = ego.rear_axle.heading - foot[2]
    lateral = float(np.dot(ego.rear_axle.array[:2] - foot[:2], [-np.sin(foot[2]), np.cos(foot[2])]))
    lateral_speed = speed * np.sin(yaw)
    # Quintic endpoint constraints; initial acceleration is measured, not a future hint.
    lateral_accel = ego.dynamic_car_state.rear_axle_acceleration_2d.x * np.sin(
        yaw
    ) + speed * ego.dynamic_car_state.angular_velocity * np.cos(yaw)
    c = np.array([lateral, lateral_speed, lateral_accel / 2.0, 0.0, 0.0, 0.0])
    T = duration
    matrix = np.array(
        [[T**3, T**4, T**5], [3 * T * T, 4 * T**3, 5 * T**4], [6 * T, 12 * T * T, 20 * T**3]]
    )
    c[3:] = np.linalg.solve(
        matrix, [target_offset - c[0] - c[1] * T - c[2] * T * T, -c[1] - 2 * c[2] * T, -2 * c[2]]
    )
    times = np.arange(81) * 0.1
    speeds = np.maximum(0.1, speed * np.cos(yaw) + brake * times)
    distances = np.concatenate(([0.0], np.cumsum((speeds[:-1] + speeds[1:]) * 0.05)))
    reference = path.interpolate(progress + distances, as_array=True)
    lateral_values = np.polynomial.polynomial.polyval(np.minimum(times, T), c)
    reference[:, :2] += lateral_values[:, None] * np.stack(
        (-np.sin(reference[:, 2]), np.cos(reference[:, 2])), axis=1
    )
    tangents = np.gradient(reference[:, :2], 0.1, axis=0)
    reference[:, 2] = np.arctan2(tangents[:, 1], tangents[:, 0])
    reference[0] = [ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading]
    refs = []
    for i, p in enumerate(reference):
        state = np.zeros(11)
        state[:3] = p
        state[3] = float(np.linalg.norm(tangents[i]))
        state[5] = brake
        refs.append(
            state_array_to_ego_state(
                state, TimePoint(ego.time_us + i * 100000), ego.car_footprint.vehicle_parameters
            )
        )
    refs[0] = ego
    trajectory = InterpolatedTrajectory(refs)
    states = [ego_state_to_state_array(ego)]
    actions = []
    first = None
    for i in range(40):
        current = SimulationIteration(ego.time_point, i)
        nxt = SimulationIteration(TimePoint(ego.time_us + 100000), i + 1)
        desired = tracker.track_trajectory(current, nxt, ego, trajectory)
        ego, action = propagate_control(
            motion, ego, desired.rear_axle_acceleration_2d.x, desired.tire_steering_rate
        )
        if first is None:
            first = ego
        states.append(ego_state_to_state_array(ego))
        actions.append(action)
    return np.asarray(states), np.asarray(actions), first


class ControlLatticeRepair(PDMRouteRepair):
    def __init__(self, root, initialization):
        super().__init__(root, initialization)
        self.description = {
            **self.description,
            "backend": "native_recovery_search_v1",
            "proposals": len(schedules()),
            "families": ["front_axle_feedback", "continuous_start_quintic_lqr"],
            "schedules": schedules(),
            "forecast_seconds": 4.0,
            "future_source_poses_used": False,
            "stateful": False,
            "actual_start_preserved": True,
            "command_limits": {
                "acceleration": [-6.5, 4.0],
                "jerk": [-8.0, 5.0],
                "steering_rate": [-0.8, 0.8],
            },
            "unsupported_forecasts_are_rejected": True,
        }
        self.description["context_proposal_bank"] = {
            k: self.description.pop(k)
            for k in ("lateral_offsets_m", "speed_limit_fractions", "trajectory_horizon_s")
        }
        self.last_forecast = None

        import hashlib
        from pathlib import Path

        self.description["implementation_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()

    def forecast(self, planner_input, ego):
        started = time.monotonic()
        from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
            BatchIDMPolicy,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            WeightedMetricIndex,
        )
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        planner = self.planner_type(
            TrajectorySampling(num_poses=80, interval_length=0.1),
            TrajectorySampling(num_poses=40, interval_length=0.1),
            BatchIDMPolicy(10.0, [0.6, 0.8, 1.0], 2.0, 1.5, 1.5, 3.0),
            [-2.0, -1.0, 1.0, 2.0],
            100.0,
        )
        enabled = gc.isenabled()
        try:
            planner.initialize(self.initialization)
            planner.compute_planner_trajectory(planner_input)
        finally:
            if enabled:
                gc.enable()
        original = dict(
            direction=planner._scorer._weighted_metrics[
                WeightedMetricIndex.DRIVING_DIRECTION
            ].tolist(),
            starting_lane=planner._get_starting_lane(ego).id,
        )
        states = []
        controls = []
        first = []
        keys = []
        rejected = []
        for schedule in schedules():
            try:
                fn = rollout_feedback if schedule[0] == "feedback" else rollout_reference
                a, b, c = fn(self.motion, ego, planner._centerline, schedule[1:], self.tracker)
                states.append(a)
                controls.append(b)
                first.append(c)
                keys.append(schedule)
            except ValueError as exc:
                rejected.append(dict(schedule=schedule, reason=str(exc)))
        if states:
            states = np.asarray(states)
            scores = planner._scorer.score_proposals(
                states,
                planner._observation,
                planner._centerline,
                list(planner._route_lane_dict),
                planner._drivable_area_map,
            )
            scorer = planner._scorer
            direction = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION]
            ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC]
            selected, valid = select_admissible_proposal(
                scores, scorer._multi_metrics, direction, ttc, scorer._progress_raw
            )
            metrics = dict(
                proposal_scores=np.where(np.isfinite(scores), scores, -1.0).tolist(),
                proposal_direction=direction.tolist(),
                proposal_ttc=ttc.tolist(),
                proposal_multiplicative=scorer._multi_metrics.tolist(),
                proposal_progress_m=scorer._progress_raw.tolist(),
            )
            self.last_forecast = dict(
                states=states.copy(), controls=np.asarray(controls), areas=scorer._ego_areas.copy()
            )
        else:
            states = np.empty((0, 41, 11))
            selected = None
            valid = np.zeros(0, bool)
            metrics = dict(
                proposal_scores=[],
                proposal_direction=[],
                proposal_ttc=[],
                proposal_multiplicative=[[], []],
                proposal_progress_m=[],
            )
            self.last_forecast = dict(
                states=states, controls=np.empty((0, 40, 2)), areas=np.empty((0, 41, 3), bool)
            )
        evidence = dict(
            backend="native_recovery_search_v1",
            proposals=len(states),
            schedules=keys,
            attempted_schedules=len(schedules()),
            rejected_schedules=rejected,
            selected_proposal=selected,
            admissible_proposals=int(valid.sum()),
            **metrics,
            original_pdm=original,
            search_wall_seconds=time.monotonic() - started,
            completed_candidate_forecast_steps=len(states) * 40,
            cost_scope="wall_time_includes_context_and_rejections; step_count_excludes_context_and_partial_rejections",
            implementation_sha256=self.description["implementation_sha256"],
            qualification="not_admitted",
            fallback_reason="bounded_search_found_no_admissible_forecast"
            if selected is None
            else None,
        )
        return states, controls, first, evidence

    def step(self, planner_input, next_iteration, ego):
        if next_iteration.time_point.time_us - ego.time_us != 100000:
            raise ValueError("Recovery search requires 10 Hz")
        _, controls, first, evidence = self.forecast(planner_input, ego)
        selected = evidence["selected_proposal"]
        if selected is None:
            state, action = propagate_control(self.motion, ego, -3.0, 0.0)
        else:
            state, action = first[selected], controls[selected][0].tolist()
        return state, action, evidence
