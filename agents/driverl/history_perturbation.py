"""Execute a privileged source-relative history proposal with native dynamics."""

import numpy as np

from driversi.rsi.correction.history_perturbation import (
    BACKEND,
    quintic_reference,
    validate_perturbation,
)
from driversi_agents.driverl.control_lattice import propagate_control
from driversi_agents.driverl.repair import PDMRouteRepair


class HistoryPerturbation(PDMRouteRepair):
    def __init__(self, root, initialization, scene, ego, perturbation):
        params = validate_perturbation(perturbation)
        if scene.metadata.get("split") != "train" or len(scene.times) != 71:
            raise ValueError("Historical proposals require a complete training source")
        super().__init__(root, initialization)
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
            state_array_to_ego_state,
        )
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
            InterpolatedTrajectory,
        )

        idx = scene.actor_ids.index("ego")
        source = scene.poses[:, idx]
        origin = np.asarray(scene.metadata["origin_xyz"][:2])
        heading = source[:, 3]
        d = ego.car_footprint.vehicle_parameters.rear_axle_to_center
        xy = source[:, :2] + origin - d * np.column_stack([np.cos(heading), np.sin(heading)])
        sv = np.gradient(xy, 0.1, axis=0, edge_order=2)[20]
        yaw = ego.rear_axle.heading
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        dynamics = ego.dynamic_car_state
        start = np.array(
            [
                ego.rear_axle.array[:2],
                rot @ dynamics.rear_axle_velocity_2d.array,
                rot @ dynamics.rear_axle_acceleration_2d.array,
            ]
        )
        eyaw = heading[20]
        forward = np.array([np.cos(eyaw), np.sin(eyaw)])
        left = np.array([-np.sin(eyaw), np.cos(eyaw)])
        endpoint = xy[20] + params["longitudinal_m"] * forward + params["lateral_m"] * left
        speed = max(0.0, float(sv @ forward) + params["speed_delta_mps"])
        end = np.array([endpoint, speed * forward, np.zeros(2)])
        times = np.arange(41) * 0.1
        positions, velocities, accelerations = quintic_reference(start, end, times)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        stopped = np.linalg.norm(velocities, axis=1) < 0.05
        angles[stopped] = np.interp(
            times[stopped], [0, 2], [yaw, yaw + np.arctan2(np.sin(eyaw - yaw), np.cos(eyaw - yaw))]
        )
        angles = np.unwrap(angles)
        states = []
        for i in range(len(times)):
            c, s = np.cos(angles[i]), np.sin(angles[i])
            local = np.array([[c, s], [-s, c]])
            value = np.zeros(11)
            value[:3] = [*positions[i], angles[i]]
            value[3:5], value[5:7] = local @ velocities[i], local @ accelerations[i]
            states.append(
                state_array_to_ego_state(
                    value, TimePoint(ego.time_us + i * 100000), ego.car_footprint.vehicle_parameters
                )
            )
        states[0] = ego
        self.start_us = ego.time_us
        self.trajectory = InterpolatedTrajectory(states)
        self.description = dict(
            backend=BACKEND,
            parameters=params,
            actual_control_ticks=[0, 20],
            reference_frame="source_endpoint_rear_axle_local_heading",
            requested_endpoint_xy=endpoint.tolist(),
            requested_endpoint_speed_mps=speed,
            stateful=False,
            source_future_used=True,
            privilege="training_source_reference_only",
            tracker="unchanged_native_LQR",
            dynamics="unchanged_native_kinematic_bicycle",
            commanded_reference_is_not_executed_trajectory=True,
        )
        self.reference = dict(
            times_s=times.tolist(),
            xy=positions.tolist(),
            velocity_xy=velocities.tolist(),
            acceleration_xy=accelerations.tolist(),
            description=self.description,
        )

    def step(self, planner_input, next_iteration, ego):
        if not self.start_us <= ego.time_us < self.start_us + 2000000:
            raise ValueError("Historical controller outside [0,2s)")
        if next_iteration.time_point.time_us - ego.time_us != 100000:
            raise ValueError("Historical controller requires actual 10Hz steps")
        desired = self.tracker.track_trajectory(
            planner_input.iteration, next_iteration, ego, self.trajectory
        )
        predicted, action = propagate_control(
            self.motion, ego, desired.rear_axle_acceleration_2d.x, desired.tire_steering_rate
        )
        return (
            predicted,
            action,
            dict(
                backend=BACKEND,
                parameters=self.description["parameters"],
                bounded_action=action,
                qualification="not_admitted",
            ),
        )
