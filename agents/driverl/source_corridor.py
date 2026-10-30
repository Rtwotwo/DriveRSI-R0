"""Privileged source-corridor recovery using unchanged native LQR and dynamics.

The raw training log provides a geometric/velocity reference, never student input.
Physical history is preserved; only bounded controls create the future states.
"""

import hashlib
from pathlib import Path

import numpy as np

from driversi_agents.driverl.control_lattice import propagate_control
from driversi_agents.driverl.repair import PDMRouteRepair


def recovery_offsets(times, position, velocity, acceleration, duration):
    """C2 offset joins the actual anchor to the logged motion after fixed duration."""
    values = [np.asarray(v, dtype=float) for v in (position, velocity, acceleration)]
    times = np.asarray(times, dtype=float)
    if duration not in (2.0, 3.0) or any(v.shape != (2,) for v in values):
        raise ValueError("Recovery requires two-dimensional state and declared 2/3s duration")
    if not np.isfinite(np.concatenate([times, *values])).all() or np.any(times < 0):
        raise ValueError("Recovery inputs must be finite nonnegative times")
    c = np.zeros((6, 2))
    c[:3] = [values[0], values[1], values[2] / 2]
    t = duration
    matrix = np.array(
        [[t**3, t**4, t**5], [3 * t * t, 4 * t**3, 5 * t**4], [6 * t, 12 * t * t, 20 * t**3]]
    )
    c[3:] = np.linalg.solve(
        matrix, [-c[0] - c[1] * t - c[2] * t * t, -c[1] - 2 * c[2] * t, -2 * c[2]]
    )
    return np.polynomial.polynomial.polyval(np.minimum(times, t), c).T


def corridor_evidence(actual_xy, source_xy, *, limit_m=2.0):
    if limit_m != 2.0:
        raise ValueError("Source-corridor coverage limit is fixed at 2m")
    actual, source = np.asarray(actual_xy), np.asarray(source_xy)
    if actual.shape != source.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError("Actual and source corridor samples must align")
    if not np.isfinite(actual).all() or not np.isfinite(source).all():
        raise ValueError("Corridor evidence must be finite")
    distances = np.linalg.norm(actual - source, axis=1)
    return dict(
        limit_m=limit_m,
        distances_m=distances.tolist(),
        max_deviation_m=float(distances.max()),
        passed=bool((distances <= limit_m).all()),
        scope="time_aligned_asset_coverage_not_driving_score",
    )


class SourceCorridorRepair(PDMRouteRepair):
    def __init__(
        self, root, initialization, scene, anchor, recovery_seconds=2.0, position_gain=0.0
    ):
        if type(position_gain) not in (int, float) or position_gain not in (0.0, 0.5, 1.0):
            raise ValueError("Unplanned source-corridor position gain")
        self.position_gain = float(position_gain)
        super().__init__(root, initialization)
        from navsim.common.dataclasses import Scene as NativeScene
        from navsim.common.dataclasses import SensorConfig
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
            state_array_to_ego_state,
        )
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
            InterpolatedTrajectory,
        )

        from driversi.data.sources.navsim import load_log

        raw = load_log(Path(scene.metadata["navsim_source"]["log_path"]), trusted=True)
        start = scene.metadata["source_start_frame"]
        if scene.metadata["split"] != "train" or start < 3:
            raise ValueError("Privileged reference is restricted to training sources")
        # Eight seconds of real source after the two-second arrival, plus one
        # actual 2Hz sample for timestamp jitter. No invented endpoint extension.
        source_rows = raw[start - 3 : start + 22]
        if len(source_rows) != 25:
            raise ValueError("Eight-second real source lookahead unavailable")
        native = NativeScene.from_scene_dict_list(
            source_rows,
            Path(scene.metadata["navsim_source"]["sensor_root"]),
            4,
            21,
            SensorConfig.build_no_sensors(),
        )
        scenario = NavSimScenario(native, "", "nuplan-maps-v1.0")
        self.source = InterpolatedTrajectory(
            [scenario.get_ego_state_at_iteration(i) for i in range(22)]
        )
        self.anchor_us = anchor.time_us
        self.times = np.arange(81) * 0.1
        source_states = [
            self.source.get_state_at_time(TimePoint(self.anchor_us + i * 100000)) for i in range(81)
        ]
        xy = np.asarray([[s.rear_axle.x, s.rear_axle.y] for s in source_states])
        velocity = np.gradient(xy, 0.1, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, 0.1, axis=0, edge_order=2)
        yaw = anchor.rear_axle.heading
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        dynamic = anchor.dynamic_car_state
        av = rot @ np.array([dynamic.rear_axle_velocity_2d.x, dynamic.rear_axle_velocity_2d.y])
        aa = rot @ np.array(
            [dynamic.rear_axle_acceleration_2d.x, dynamic.rear_axle_acceleration_2d.y]
        )
        xy += recovery_offsets(
            self.times,
            anchor.rear_axle.array[:2] - xy[0],
            av - velocity[0],
            aa - acceleration[0],
            recovery_seconds,
        )
        velocity = np.gradient(xy, 0.1, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, 0.1, axis=0, edge_order=2)
        headings = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
        states = []
        for i in range(len(xy)):
            c, s = np.cos(headings[i]), np.sin(headings[i])
            rotation = np.array([[c, s], [-s, c]])
            value = np.zeros(11)
            value[:3] = [*xy[i], headings[i]]
            value[3:5] = rotation @ velocity[i]
            value[5:7] = rotation @ acceleration[i]
            states.append(
                state_array_to_ego_state(
                    value,
                    TimePoint(self.anchor_us + i * 100000),
                    anchor.car_footprint.vehicle_parameters,
                )
            )
        states[0] = anchor
        self.trajectory = InterpolatedTrajectory(states)
        self.description = dict(
            backend="source_corridor_lqr_v1",
            stateful=False,
            recovery_seconds=recovery_seconds,
            position_gain_s_inverse=position_gain,
            future_source_poses_used=True,
            privilege="training_source_future_teacher_reference_only",
            asset_corridor_limit_m=2.0,
            actual_anchor_preserved=True,
            source_log=scene.metadata["source_log"],
            source_start_frame=start,
            raw_reference_frame_interval=[start, start + 21],
            reference_horizon_s=8.0,
            tracker="unchanged_native_LQR",
            dynamics="unchanged_native_kinematic_bicycle",
            qualification="independent_executed_PDMS_and_physics_and_visual_required",
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def source_xy(self, egos):
        return np.asarray(
            [
                [s.rear_axle.x, s.rear_axle.y]
                for s in (self.source.get_state_at_time(e.time_point) for e in egos)
            ]
        )

    def step(self, planner_input, next_iteration, ego):
        if next_iteration.time_point.time_us - ego.time_us != 100000:
            raise ValueError("Source-corridor teacher requires 10Hz actual state")
        if not self.anchor_us <= ego.time_us < self.anchor_us + 5000000:
            raise ValueError("Source-corridor teacher outside declared execution window")
        desired = self.tracker.track_trajectory(
            planner_input.iteration, next_iteration, ego, self.trajectory
        )
        acceleration = desired.rear_axle_acceleration_2d.x
        longitudinal_error = 0.0
        reference_velocity = corrected_velocity = None
        if self.position_gain:
            source = self.source.get_state_at_time(ego.time_point)
            yaw = source.rear_axle.heading
            longitudinal_error = float(
                (ego.rear_axle.array[:2] - source.rear_axle.array[:2])
                @ np.array([np.cos(yaw), np.sin(yaw)])
            )
            reference_velocity, _ = self.tracker._compute_reference_velocity_and_curvature_profile(
                planner_input.iteration, self.trajectory
            )
            corrected_velocity = max(
                0.0, reference_velocity - self.position_gain * longitudinal_error
            )
            acceleration = self.tracker._longitudinal_lqr_controller(
                ego.dynamic_car_state.rear_axle_velocity_2d.x, corrected_velocity
            )
        predicted, action = propagate_control(
            self.motion, ego, acceleration, desired.tire_steering_rate
        )
        return (
            predicted,
            action,
            dict(
                backend="source_corridor_lqr_v1",
                recovery_seconds=self.description["recovery_seconds"],
                source_future_used=True,
                qualification="not_admitted",
                desired_acceleration=float(acceleration),
                native_reference_velocity=reference_velocity,
                corrected_reference_velocity=corrected_velocity,
                source_longitudinal_error_m=longitudinal_error,
                position_gain_s_inverse=self.position_gain,
                bounded_action=action,
            ),
        )
