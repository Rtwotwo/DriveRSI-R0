"""Torch implementation of NAVSIM's batched PDM LQR simulator.

The implementation follows ``batch_lqr.py``, ``batch_lqr_utils.py``, and
``batch_kinematic_bicycle.py`` while preserving arbitrary leading batch
dimensions.  Use float64 during CPU/GPU parity validation; float32 can be
enabled only after the scorer's discrete metrics match the CPU reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .ops import (
    STATE_ACCELERATION_X,
    STATE_ACCELERATION_Y,
    STATE_ANGULAR_ACCELERATION,
    STATE_ANGULAR_VELOCITY,
    STATE_HEADING,
    STATE_SIZE,
    STATE_STEERING_ANGLE,
    STATE_STEERING_RATE,
    STATE_VELOCITY_X,
    STATE_VELOCITY_Y,
    STATE_X,
    STATE_Y,
    wrap_angle,
)


@dataclass(frozen=True)
class SimulatorConfig:
    discretization_time: float = 0.1
    tracking_horizon: int = 10
    q_longitudinal: float = 10.0
    r_longitudinal: float = 1.0
    q_lateral: tuple[float, float, float] = (1.0, 10.0, 0.0)
    r_lateral: float = 1.0
    jerk_penalty: float = 1e-4
    curvature_rate_penalty: float = 1e-2
    initial_curvature_penalty: float = 1e-10
    stopping_proportional_gain: float = 0.5
    stopping_velocity: float = 0.2
    max_steering_angle: float = torch.pi / 3
    accel_time_constant: float = 0.2
    steering_angle_time_constant: float = 0.05


def _difference_matrix(rows: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    matrix = torch.zeros((rows, rows + 1), device=device, dtype=dtype)
    if rows > 0:
        eye = torch.eye(rows, device=device, dtype=dtype)
        matrix[:, 1:] = eye
        matrix[:, :-1] = -eye
    return matrix


def _profile(initial: Tensor, derivatives: Tensor, dt: float) -> Tensor:
    cumulative = torch.cumsum(derivatives * dt, dim=-1)
    zero = torch.zeros_like(cumulative[..., :1])
    return initial[..., None] + torch.cat((zero, cumulative), dim=-1)


def velocity_curvature_profiles(
    poses: Tensor,
    dt: float,
    jerk_penalty: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Torch equivalent of NAVSIM's regularized profile fitting."""

    if poses.ndim != 3 or poses.shape[-1] != 3 or poses.shape[1] < 4:
        raise ValueError("poses must have shape [N, T>=4, 3]")
    batch_size, num_poses, _ = poses.shape
    num_displacements = num_poses - 1
    pose_diff = poses[:, 1:] - poses[:, :-1]
    xy_displacements = pose_diff[..., :2]
    heading_displacements = wrap_angle(pose_diff[..., 2])

    y_velocity = xy_displacements.reshape(batch_size, -1)
    headings = poses[:, :-1, 2]
    a_column = torch.empty_like(y_velocity)
    a_column[:, 0::2] = torch.cos(headings)
    a_column[:, 1::2] = torch.sin(headings)
    a_velocity = (a_column[..., None] * (dt**2)).repeat(1, 1, num_displacements)
    a_velocity[..., 0] = a_column * dt
    upper = torch.triu(
        torch.ones((num_displacements, num_displacements), device=poses.device, dtype=torch.bool), diagonal=1
    ).repeat_interleave(2, dim=0)
    a_velocity = a_velocity.masked_fill(upper.unsqueeze(0), 0.0)

    banded = _difference_matrix(num_displacements - 2, device=poses.device, dtype=poses.dtype)
    regularizer = torch.cat(
        (torch.zeros((banded.shape[0], 1), device=poses.device, dtype=poses.dtype), banded), dim=-1
    )
    regularizer = regularizer.unsqueeze(0).expand(batch_size, -1, -1)
    a_t = a_velocity.transpose(-1, -2)
    r_t = regularizer.transpose(-1, -2)
    normal = a_t @ a_velocity + jerk_penalty * (r_t @ regularizer)
    velocity_rhs = a_t @ y_velocity.unsqueeze(-1)
    velocity_solution = torch.linalg.solve(normal, velocity_rhs).squeeze(-1)
    initial_velocity = velocity_solution[:, 0]
    acceleration_profile = velocity_solution[:, 1:]
    velocity_profile = _profile(initial_velocity, acceleration_profile, dt)

    dim = num_displacements
    a_curvature = torch.tril(torch.ones((dim, dim), device=poses.device, dtype=poses.dtype))
    a_curvature = a_curvature.unsqueeze(0).repeat(batch_size, 1, 1)
    a_curvature[:, :, 0] = velocity_profile * dt
    velocity_dt2 = velocity_profile * (dt**2)
    a_curvature[:, 1:, 1:] *= velocity_dt2[:, None, 1:].transpose(1, 2)
    q_curvature = torch.eye(dim, device=poses.device, dtype=poses.dtype) * curvature_rate_penalty
    q_curvature[0, 0] = initial_curvature_penalty
    a_t = a_curvature.transpose(-1, -2)
    curvature_normal = a_t @ a_curvature + q_curvature
    curvature_rhs = a_t @ heading_displacements.unsqueeze(-1)
    curvature_solution = torch.linalg.solve(curvature_normal, curvature_rhs).squeeze(-1)
    initial_curvature = curvature_solution[:, 0]
    curvature_rate_profile = curvature_solution[:, 1:]
    curvature_profile = _profile(initial_curvature, curvature_rate_profile, dt)
    return velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile


class TorchPDMSimulator(nn.Module):
    """Rank-local GPU PDM simulator with the same 11-value state layout."""

    def __init__(self, wheel_base: float, config: SimulatorConfig = SimulatorConfig()) -> None:
        super().__init__()
        if wheel_base <= 0.0:
            raise ValueError("wheel_base must be positive")
        if config.tracking_horizon <= 1 or config.discretization_time <= 0.0:
            raise ValueError("invalid simulator horizon or discretization time")
        self.wheel_base = float(wheel_base)
        self.config = config

    def _lateral_controller(
        self,
        lateral_state: Tensor,
        velocity_profile: Tensor,
        curvature_profile: Tensor,
    ) -> Tensor:
        config = self.config
        batch_size = lateral_state.shape[0]
        eye = torch.eye(3, device=lateral_state.device, dtype=lateral_state.dtype)
        state_transition = eye.view(1, 1, 3, 3).repeat(config.tracking_horizon, batch_size, 1, 1)
        state_transition[:, :, 0, 1] = velocity_profile.transpose(0, 1) * config.discretization_time
        state_transition[:, :, 1, 2] = (
            velocity_profile.transpose(0, 1) * config.discretization_time / self.wheel_base
        )
        affine = torch.zeros(
            (config.tracking_horizon, batch_size, 3), device=lateral_state.device, dtype=lateral_state.dtype
        )
        affine[:, :, 1] = (
            -velocity_profile.transpose(0, 1)
            * curvature_profile.transpose(0, 1)
            * config.discretization_time
        )
        input_matrix = torch.zeros((3, 1), device=lateral_state.device, dtype=lateral_state.dtype)
        input_matrix[2, 0] = config.discretization_time
        a_matrix = eye.unsqueeze(0).repeat(batch_size, 1, 1)
        b_matrix = torch.zeros((batch_size, 3, 1), device=lateral_state.device, dtype=lateral_state.dtype)
        g_vector = torch.zeros((batch_size, 3), device=lateral_state.device, dtype=lateral_state.dtype)
        for step in range(config.tracking_horizon):
            transition = state_transition[step]
            a_matrix = transition @ a_matrix
            b_matrix = transition @ b_matrix + input_matrix
            g_vector = (transition @ g_vector.unsqueeze(-1)).squeeze(-1) + affine[step]

        state_error = (a_matrix @ lateral_state.unsqueeze(-1)).squeeze(-1) + g_vector
        state_error = state_error.clone()
        state_error[:, 1:3] = wrap_angle(state_error[:, 1:3])
        q_matrix = torch.diag(
            lateral_state.new_tensor(config.q_lateral)
        )
        b_t_q = b_matrix.transpose(-1, -2) @ q_matrix
        inverse = -1.0 / ((b_t_q @ b_matrix).squeeze(-1).squeeze(-1) + config.r_lateral)
        tail = (b_t_q @ state_error.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return inverse * tail

    def _commands(
        self,
        current_index: int,
        current_states: Tensor,
        proposal_states: Tensor,
        velocity_profile: Tensor,
        curvature_profile: Tensor,
    ) -> Tensor:
        config = self.config
        reference_pose = proposal_states[:, current_index]
        x_error = current_states[:, STATE_X] - reference_pose[:, STATE_X]
        y_error = current_states[:, STATE_Y] - reference_pose[:, STATE_Y]
        reference_heading = reference_pose[:, STATE_HEADING]
        lateral_error = -x_error * torch.sin(reference_heading) + y_error * torch.cos(reference_heading)
        heading_error = wrap_angle(current_states[:, STATE_HEADING] - reference_heading)
        lateral_state = torch.stack(
            (lateral_error, heading_error, current_states[:, STATE_STEERING_ANGLE]), dim=-1
        )

        reference_index = min(current_index + config.tracking_horizon, velocity_profile.shape[1] - 1)
        reference_velocity = velocity_profile[:, reference_index]
        curvature_indices = torch.arange(
            config.tracking_horizon, device=current_states.device
        ).add(current_index).clamp(max=reference_index)
        reference_curvature = curvature_profile.index_select(1, curvature_indices)
        current_velocity = current_states[:, STATE_VELOCITY_X]
        should_stop = (reference_velocity <= config.stopping_velocity) & (
            current_velocity <= config.stopping_velocity
        )

        horizon_dt = config.tracking_horizon * config.discretization_time
        zero_input_error = current_velocity - reference_velocity
        acceleration = (
            -horizon_dt
            * config.q_longitudinal
            * zero_input_error
            / (horizon_dt * config.q_longitudinal * horizon_dt + config.r_longitudinal)
        )
        stopping_acceleration = -config.stopping_proportional_gain * zero_input_error
        acceleration = torch.where(should_stop, stopping_acceleration, acceleration)

        profile_steps = torch.arange(
            config.tracking_horizon, device=current_states.device, dtype=current_states.dtype
        )
        controller_velocity = current_velocity[:, None] + (
            acceleration[:, None] * config.discretization_time * profile_steps[None]
        )
        steering_rate = self._lateral_controller(lateral_state, controller_velocity, reference_curvature)
        steering_rate = torch.where(should_stop, torch.zeros_like(steering_rate), steering_rate)
        return torch.stack((acceleration, steering_rate), dim=-1)

    def _propagate(self, states: Tensor, commands: Tensor) -> Tensor:
        config = self.config
        dt = config.discretization_time
        acceleration = states[:, STATE_ACCELERATION_X]
        steering_angle = states[:, STATE_STEERING_ANGLE]
        updated_acceleration = dt / (dt + config.accel_time_constant) * (
            commands[:, 0] - acceleration
        ) + acceleration
        ideal_steering = steering_angle + dt * commands[:, 1]
        updated_steering = dt / (dt + config.steering_angle_time_constant) * (
            ideal_steering - steering_angle
        ) + steering_angle
        updated_steering_rate = (updated_steering - steering_angle) / dt

        velocity = states[:, STATE_VELOCITY_X]
        heading = states[:, STATE_HEADING]
        output = states.clone()
        output[:, STATE_X] = states[:, STATE_X] + dt * velocity * torch.cos(heading)
        output[:, STATE_Y] = states[:, STATE_Y] + dt * velocity * torch.sin(heading)
        output[:, STATE_HEADING] = wrap_angle(
            heading + dt * velocity * torch.tan(steering_angle) / self.wheel_base
        )
        output[:, STATE_VELOCITY_X] = velocity + dt * updated_acceleration
        output[:, STATE_VELOCITY_Y] = 0.0
        output[:, STATE_STEERING_ANGLE] = torch.clamp(
            steering_angle + dt * updated_steering_rate,
            -config.max_steering_angle,
            config.max_steering_angle,
        )
        output[:, STATE_ANGULAR_VELOCITY] = (
            output[:, STATE_VELOCITY_X]
            * torch.tan(output[:, STATE_STEERING_ANGLE])
            / self.wheel_base
        )
        output[:, STATE_ACCELERATION_X] = updated_acceleration
        output[:, STATE_ACCELERATION_Y] = 0.0
        output[:, STATE_ANGULAR_ACCELERATION] = (
            output[:, STATE_ANGULAR_VELOCITY] - states[:, STATE_ANGULAR_VELOCITY]
        ) / dt
        output[:, STATE_STEERING_RATE] = updated_steering_rate
        return output

    def forward(self, proposal_states: Tensor, initial_states: Tensor) -> Tensor:
        """Simulate proposal reference states.

        Args:
            proposal_states: ``[..., T, 11]`` reference states.
            initial_states: ``[..., 11]`` or states broadcastable over proposal
                leading dimensions.
        """

        if proposal_states.ndim < 3 or proposal_states.shape[-1] != STATE_SIZE:
            raise ValueError("proposal_states must have shape [..., T, 11]")
        leading_shape = proposal_states.shape[:-2]
        num_steps = proposal_states.shape[-2]
        if num_steps < 2:
            raise ValueError("proposal_states requires at least two time steps")
        if initial_states.shape[-1] != STATE_SIZE:
            raise ValueError("initial_states must end in 11 values")
        while initial_states.ndim < len(leading_shape) + 1:
            initial_states = initial_states.unsqueeze(-2)
        expanded_initial = torch.broadcast_to(initial_states, (*leading_shape, STATE_SIZE))
        flat_proposals = proposal_states.reshape(-1, num_steps, STATE_SIZE)
        flat_initial = expanded_initial.reshape(-1, STATE_SIZE)
        velocity_profile, _, curvature_profile, _ = velocity_curvature_profiles(
            flat_proposals[..., :3],
            self.config.discretization_time,
            self.config.jerk_penalty,
            self.config.curvature_rate_penalty,
            self.config.initial_curvature_penalty,
        )
        simulated = torch.zeros_like(flat_proposals)
        simulated[:, 0] = flat_initial
        for time_index in range(num_steps - 1):
            commands = self._commands(
                time_index,
                simulated[:, time_index],
                flat_proposals,
                velocity_profile,
                curvature_profile,
            )
            simulated[:, time_index + 1] = self._propagate(simulated[:, time_index], commands)
        return simulated.reshape(*leading_shape, num_steps, STATE_SIZE)
