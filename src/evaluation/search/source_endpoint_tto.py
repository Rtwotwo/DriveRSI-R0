"""Track a declared endpoint candidate in the authoritative reactive world."""
import hashlib
from pathlib import Path
import numpy as np

from driversi.rsi.correction.endpoint_tto import BACKEND, warp_reference
from driversi_agents.driverl.source_corridor import SourceCorridorRepair


class SourceEndpointTTO(SourceCorridorRepair):
    def __init__(self, root, initialization, scene, anchor, offset):
        super().__init__(root, initialization, scene, anchor)
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        source_states = self.trajectory.get_sampled_trajectory()
        xy = np.array([[s.rear_axle.x, s.rear_axle.y] for s in source_states])
        xy = warp_reference(xy, self.times, offset)
        velocity = np.gradient(xy, .1, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, .1, axis=0, edge_order=2)
        yaw = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
        states = []
        for i in range(len(xy)):
            c, s = np.cos(yaw[i]), np.sin(yaw[i])
            rotation = np.array([[c, s], [-s, c]])
            value = np.zeros(11)
            value[:3] = [*xy[i], yaw[i]]
            value[3:5], value[5:7] = rotation @ velocity[i], rotation @ acceleration[i]
            states.append(state_array_to_ego_state(value,
                TimePoint(anchor.time_us+i*100000), anchor.car_footprint.vehicle_parameters))
        states[0] = anchor
        self.trajectory = InterpolatedTrajectory(states)
        self.description.update(backend=BACKEND, endpoint_offset=dict(offset),
            construction="C2_bounded_training_source_endpoint_warp",
            official_drivetto_implementation=False, pure_driverl=False,
            future_control_ticks=[20, 70], source_reference_horizon_s=8.,
            constructor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def step(self, planner_input, next_iteration, ego):
        predicted, action, details = super().step(planner_input, next_iteration, ego)
        details.update(backend=BACKEND, endpoint_offset=self.description["endpoint_offset"])
        return predicted, action, details
