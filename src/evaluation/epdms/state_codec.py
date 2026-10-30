"""Decode the generated world's sealed rear-axle history with native types."""
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


def unpack_ego(v):
    return EgoState.build_from_rear_axle(
        StateSE2(*v[:3]), StateVector2D(*v[3:5]), StateVector2D(*v[5:7]),
        v[7], TimePoint(int(v[11])), get_pacifica_parameters(),
        angular_vel=v[8], angular_accel=v[9], tire_steering_rate=v[10])
