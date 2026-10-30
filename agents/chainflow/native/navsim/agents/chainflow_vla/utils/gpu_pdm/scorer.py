"""GPU implementations of the seven ChainFlow V1 PDM training targets.

Scene objects are packed into :class:`PreparedPDMScene` once by the data
pipeline.  Scoring then uses only Torch tensors and never transfers proposal
states to NumPy/Shapely/Ray.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial, pi

import torch
from torch import Tensor

from .ops import (
    STATE_ACCELERATION_X,
    STATE_ACCELERATION_Y,
    STATE_HEADING,
    STATE_SIZE,
    STATE_VELOCITY_X,
    STATE_VELOCITY_Y,
    V1ScoreConfig,
    VehicleGeometry,
    aggregate_v1_scores,
    oriented_boxes_intersect,
    points_in_padded_polygon_edges,
    project_points_to_polyline,
    state_array_to_coords,
)


@dataclass(frozen=True)
class PreparedPDMScene:
    """Padded rank-local tensors required by the V1 scorer.

    Agent properties use a stable agent axis across time. Map polygon masks
    describe columns in ``map_polygons`` and may overlap.
    """

    origin: Tensor  # [B, 2], subtracted from all scene coordinates
    agent_corners: Tensor  # [B, To, A, 4, 2]
    agent_valid: Tensor  # [B, To, A]
    agent_is_dynamic: Tensor  # [B, A], tracked-object type in AGENT_TYPES
    agent_is_stopped: Tensor  # [B, A]
    agent_is_red_light: Tensor  # [B, A]
    agent_initially_collided: Tensor  # [B, A]
    map_edge_start: Tensor  # [B, E, 2]
    map_edge_end: Tensor  # [B, E, 2]
    map_edge_polygon: Tensor  # [B, E]
    map_edge_valid: Tensor  # [B, E]
    map_polygon_min: Tensor  # [B, K, 2]
    map_polygon_max: Tensor  # [B, K, 2]
    lane_polygon_mask: Tensor  # [B, K]
    drivable_polygon_mask: Tensor  # [B, K]
    route_lane_polygon_mask: Tensor  # [B, K]
    intersection_polygon_mask: Tensor  # [B, K]
    centerline: Tensor  # [B, L, 2]
    centerline_point_mask: Tensor  # [B, L]


@dataclass(frozen=True)
class MetricConfig:
    interval_s: float = 0.1
    driving_direction_horizon_s: float = 1.0
    driving_direction_compliance_m: float = 2.0
    driving_direction_violation_m: float = 6.0
    collision_stopped_speed_mps: float = 5e-2
    ttc_stopped_speed_mps: float = 5e-3


def _validate_inputs(states: Tensor, scene: PreparedPDMScene) -> None:
    if states.ndim != 4 or states.shape[-1] != STATE_SIZE:
        raise ValueError("states must have shape [B, P, T, 11]")
    batch_size = states.shape[0]
    if scene.origin.shape != (batch_size, 2):
        raise ValueError("scene origin must have shape [B, 2]")
    if scene.agent_corners.ndim != 5 or scene.agent_corners.shape[-2:] != (4, 2):
        raise ValueError("agent_corners must have shape [B, To, A, 4, 2]")
    if scene.agent_corners.shape[0] != batch_size:
        raise ValueError("state and scene batch dimensions must match")
    if scene.agent_valid.shape != scene.agent_corners.shape[:-2]:
        raise ValueError("agent_valid must have shape [B, To, A]")
    agent_shape = (batch_size, scene.agent_corners.shape[2])
    for value in (
        scene.agent_is_dynamic,
        scene.agent_is_stopped,
        scene.agent_is_red_light,
        scene.agent_initially_collided,
    ):
        if value.shape != agent_shape:
            raise ValueError("agent property tensors must have shape [B, A]")
    if scene.map_edge_start.shape != scene.map_edge_end.shape:
        raise ValueError("map edge endpoints must share shape [B, E, 2]")
    if scene.map_edge_start.ndim != 3 or scene.map_edge_start.shape[-1] != 2:
        raise ValueError("map edge endpoints must have shape [B, E, 2]")
    if scene.map_edge_polygon.shape != scene.map_edge_start.shape[:-1]:
        raise ValueError("map_edge_polygon must have shape [B, E]")
    if scene.map_edge_valid.shape != scene.map_edge_polygon.shape:
        raise ValueError("map_edge_valid must have shape [B, E]")
    polygon_shape = scene.lane_polygon_mask.shape
    if scene.map_edge_start.shape[0] != batch_size or len(polygon_shape) != 2 or polygon_shape[0] != batch_size:
        raise ValueError("map tensors must share the state batch dimension")
    if scene.map_polygon_min.shape != (*polygon_shape, 2) or scene.map_polygon_max.shape != (*polygon_shape, 2):
        raise ValueError("map polygon bounds must have shape [B, K, 2]")
    for value in (
        scene.lane_polygon_mask,
        scene.drivable_polygon_mask,
        scene.route_lane_polygon_mask,
        scene.intersection_polygon_mask,
    ):
        if value.shape != polygon_shape:
            raise ValueError("map layer masks must have shape [B, K]")
    if scene.centerline.ndim != 3 or scene.centerline.shape[-1] != 2:
        raise ValueError("centerline must have shape [B, L, 2]")
    if scene.centerline_point_mask.shape != scene.centerline.shape[:-1]:
        raise ValueError("centerline_point_mask must have shape [B, L]")
    device = states.device
    if any(value.device != device for value in scene.__dict__.values()):
        raise ValueError("all prepared scene tensors must be on the states device")


def _cross_2d(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


def _segments_intersect(
    p0: Tensor,
    p1: Tensor,
    q0: Tensor,
    q1: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Broadcasted closed line-segment intersection."""

    p0, p1, q0, q1 = torch.broadcast_tensors(p0, p1, q0, q1)
    p_delta = p1 - p0
    q_delta = q1 - q0
    o1 = _cross_2d(p_delta, q0 - p0)
    o2 = _cross_2d(p_delta, q1 - p0)
    o3 = _cross_2d(q_delta, p0 - q0)
    o4 = _cross_2d(q_delta, p1 - q0)
    proper = ((o1 > eps) & (o2 < -eps) | (o1 < -eps) & (o2 > eps)) & (
        (o3 > eps) & (o4 < -eps) | (o3 < -eps) & (o4 > eps)
    )

    def on_segment(start: Tensor, end: Tensor, point: Tensor, orientation: Tensor) -> Tensor:
        lower = torch.minimum(start, end) - eps
        upper = torch.maximum(start, end) + eps
        return (orientation.abs() <= eps) & ((point >= lower) & (point <= upper)).all(dim=-1)

    return (
        proper
        | on_segment(p0, p1, q0, o1)
        | on_segment(p0, p1, q1, o2)
        | on_segment(q0, q1, p0, o3)
        | on_segment(q0, q1, p1, o4)
    )


def _points_in_oriented_boxes(points: Tensor, corners: Tensor, eps: float = 1e-7) -> Tensor:
    points, first_corner = torch.broadcast_tensors(points, corners[..., 0, :])
    corners = torch.broadcast_to(corners, (*points.shape[:-1], 4, 2))
    axes = torch.stack(
        (corners[..., 1, :] - first_corner, corners[..., 3, :] - first_corner), dim=-2
    )
    point_projection = (points.unsqueeze(-2) * axes).sum(dim=-1)
    corner_projection = torch.einsum("...ci,...ai->...ca", corners, axes)
    return (
        (point_projection >= corner_projection.amin(dim=-2) - eps)
        & (point_projection <= corner_projection.amax(dim=-2) + eps)
    ).all(dim=-1)


def _front_edge_intersects_boxes(front_left: Tensor, front_right: Tensor, corners: Tensor) -> Tensor:
    edge_start = corners
    edge_end = torch.roll(corners, shifts=-1, dims=-2)
    edge_hits = _segments_intersect(
        front_left.unsqueeze(-2), front_right.unsqueeze(-2), edge_start, edge_end
    ).any(dim=-1)
    return (
        edge_hits
        | _points_in_oriented_boxes(front_left, corners)
        | _points_in_oriented_boxes(front_right, corners)
    )


def _relative_angle(ego_states: Tensor, agent_centers: Tensor) -> Tensor:
    relative = agent_centers - ego_states[..., :2]
    norm = torch.linalg.vector_norm(relative, dim=-1)
    heading = ego_states[..., STATE_HEADING]
    forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    cosine = (forward * relative).sum(dim=-1) / norm.clamp_min(
        torch.finfo(relative.dtype).eps
    )
    cosine = torch.where(norm > 0, cosine, torch.zeros_like(cosine))
    return torch.acos(cosine.clamp(-1.0, 1.0))


def _gather_time(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B, P, T, ...]`` at ``[B, P, A]`` time indices."""

    tail_shape = values.shape[3:]
    gather_index = indices.reshape(*indices.shape, *((1,) * len(tail_shape)))
    gather_index = gather_index.expand(*indices.shape, *tail_shape)
    expanded = values.unsqueeze(2).expand(values.shape[0], values.shape[1], indices.shape[2], *values.shape[2:])
    return expanded.gather(3, gather_index.unsqueeze(3)).squeeze(3)


def _gather_agent_time(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B, T, A, ...]`` using ``[B, P, A]`` time indices."""

    batch_size, _, num_agents = values.shape[:3]
    batch_idx = torch.arange(batch_size, device=values.device)[:, None, None]
    agent_idx = torch.arange(num_agents, device=values.device)[None, None, :]
    return values[batch_idx, indices, agent_idx]


def calculate_ego_area(
    states: Tensor,
    ego_coords: Tensor,
    scene: PreparedPDMScene,
) -> tuple[Tensor, Tensor]:
    """Return V1 ego areas ``[..., 3]`` and rear-axle-in-intersection."""

    batch_size, num_proposals, num_steps = states.shape[:3]
    query_points = torch.cat((ego_coords, states[..., None, :2]), dim=-2)
    membership = points_in_padded_polygon_edges(
        query_points.reshape(batch_size, -1, 2),
        scene.map_edge_start,
        scene.map_edge_end,
        scene.map_edge_polygon,
        scene.map_edge_valid,
        scene.map_polygon_min,
        scene.map_polygon_max,
        scene.lane_polygon_mask.shape[1],
    ).reshape(batch_size, num_proposals, num_steps, 6, -1)
    corner_membership = membership[..., :4, :]

    lane_membership = corner_membership & scene.lane_polygon_mask[:, None, None, None, :]
    polygons_with_corner = lane_membership.any(dim=-2).sum(dim=-1) > 1
    contained_by_one_lane = (lane_membership.sum(dim=-2) == 4).any(dim=-1)
    multiple_lanes = polygons_with_corner & ~contained_by_one_lane

    drivable_membership = corner_membership & scene.drivable_polygon_mask[:, None, None, None, :]
    non_drivable = ~drivable_membership.any(dim=-1).all(dim=-1)

    center_membership = membership[..., 4, :]
    on_route = (
        center_membership & scene.route_lane_polygon_mask[:, None, None, :]
    ).any(dim=-1)
    oncoming = ~on_route

    rear_membership = membership[..., 5, :]
    rear_in_intersection = (
        rear_membership & scene.intersection_polygon_mask[:, None, None, :]
    ).any(dim=-1)
    return torch.stack((multiple_lanes, non_drivable, oncoming), dim=-1), rear_in_intersection


def _no_at_fault_collision(
    states: Tensor,
    ego_coords: Tensor,
    ego_area: Tensor,
    scene: PreparedPDMScene,
    config: MetricConfig,
) -> Tensor:
    batch_size, num_proposals, num_steps = states.shape[:3]
    num_agents = scene.agent_corners.shape[2]
    if num_agents == 0:
        return states.new_ones((batch_size, num_proposals))

    agent_corners = scene.agent_corners[:, :num_steps]
    intersections = oriented_boxes_intersect(
        ego_coords[..., None, :4, :], agent_corners[:, None, :, :, :, :]
    )
    valid = scene.agent_valid[:, :num_steps]
    intersections &= valid[:, None]
    intersections &= ~scene.agent_is_red_light[:, None, None, :]
    intersections &= ~scene.agent_initially_collided[:, None, None, :]

    time_index = torch.arange(num_steps, device=states.device).view(1, 1, -1, 1)
    first_time = torch.where(intersections, time_index, num_steps).amin(dim=2)
    has_collision = first_time < num_steps
    safe_time = first_time.clamp(max=num_steps - 1)
    event_state = _gather_time(states, safe_time)
    event_coords = _gather_time(ego_coords, safe_time)
    event_agent_corners = _gather_agent_time(agent_corners, safe_time)
    event_agent_centers = event_agent_corners.mean(dim=-2)
    event_area = _gather_time(ego_area, safe_time)

    speed = torch.linalg.vector_norm(event_state[..., STATE_VELOCITY_X : STATE_VELOCITY_Y + 1], dim=-1)
    ego_stopped = speed <= config.collision_stopped_speed_mps
    track_stopped = scene.agent_is_stopped[:, None, :]
    behind = _relative_angle(event_state, event_agent_centers) > (150.0 * pi / 180.0)
    front_hit = _front_edge_intersects_boxes(
        event_coords[..., 0, :], event_coords[..., 3, :], event_agent_corners
    )
    lateral_fault_area = event_area[..., 0] | event_area[..., 1]
    at_fault = has_collision & ~ego_stopped & (
        track_stopped | (~track_stopped & ~behind & (front_hit | lateral_fault_area))
    )

    dynamic_fault = (at_fault & scene.agent_is_dynamic[:, None, :]).any(dim=-1)
    static_fault = (at_fault & ~scene.agent_is_dynamic[:, None, :]).any(dim=-1)
    return torch.where(
        dynamic_fault,
        states.new_zeros(()),
        torch.where(static_fault, states.new_full((), 0.5), states.new_ones(())),
    )


def _driving_direction(ego_coords: Tensor, ego_area: Tensor, config: MetricConfig) -> Tensor:
    centers = ego_coords[..., 4, :]
    distance = torch.linalg.vector_norm(centers[..., 1:, :] - centers[..., :-1, :], dim=-1)
    progress = torch.cat((torch.zeros_like(distance[..., :1]), distance), dim=-1)
    progress = torch.where(ego_area[..., 2], progress, torch.zeros_like(progress))

    num_steps = progress.shape[-1]
    horizon = int(config.driving_direction_horizon_s / config.interval_s)
    prefix = torch.cat((torch.zeros_like(progress[..., :1]), torch.cumsum(progress, dim=-1)), dim=-1)
    end = torch.arange(1, num_steps + 1, device=progress.device)
    start = (torch.arange(num_steps, device=progress.device) - horizon).clamp_min(0)
    rolling = prefix.index_select(-1, end) - prefix.index_select(-1, start)
    maximum = rolling.amax(dim=-1)
    return torch.where(
        maximum < config.driving_direction_compliance_m,
        progress.new_ones(()),
        torch.where(
            maximum < config.driving_direction_violation_m,
            progress.new_full((), 0.5),
            progress.new_zeros(()),
        ),
    )


def _progress(ego_coords: Tensor, scene: PreparedPDMScene) -> Tensor:
    endpoints = torch.stack((ego_coords[:, :, 0, 4], ego_coords[:, :, -1, 4]), dim=2)
    batch_size, num_proposals = endpoints.shape[:2]
    # Progress is especially sensitive to nearly tied adjacent centerline
    # segments. Keep this small (two points per proposal) projection in FP64 on
    # the same device so its segment choice agrees with the official Shapely
    # scorer without moving trajectories to CPU.
    along, _ = project_points_to_polyline(
        endpoints.reshape(batch_size, num_proposals * 2, 2).double(),
        scene.centerline.double(),
        scene.centerline_point_mask,
    )
    along = along.reshape(batch_size, num_proposals, 2)
    return (along[..., 1] - along[..., 0]).clamp_min(0.0).to(ego_coords.dtype)


def _ttc(
    states: Tensor,
    ego_coords: Tensor,
    ego_area: Tensor,
    rear_in_intersection: Tensor,
    scene: PreparedPDMScene,
    config: MetricConfig,
) -> Tensor:
    batch_size, num_proposals, num_steps = states.shape[:3]
    num_agents = scene.agent_corners.shape[2]
    if num_agents == 0:
        return states.new_ones((batch_size, num_proposals))

    future_indices = torch.tensor((0, 3, 6, 9), device=states.device)
    observation_indices = torch.arange(num_steps, device=states.device)[:, None] + future_indices[None]
    if int(observation_indices[-1, -1]) >= scene.agent_corners.shape[1]:
        raise ValueError("prepared agent horizon must include nine TTC look-ahead steps")
    num_future = len(future_indices)
    future_agent_corners = scene.agent_corners.index_select(1, observation_indices.flatten()).reshape(
        batch_size, num_steps, num_future, num_agents, 4, 2
    )
    future_valid = scene.agent_valid.index_select(1, observation_indices.flatten()).reshape(
        batch_size, num_steps, num_future, num_agents
    )

    speeds = torch.linalg.vector_norm(
        states[..., STATE_VELOCITY_X : STATE_VELOCITY_Y + 1], dim=-1
    )
    heading = states[..., STATE_HEADING]
    velocity_xy = torch.stack((torch.cos(heading) * speeds, torch.sin(heading) * speeds), dim=-1)
    delta_t = future_indices.to(states.dtype) * config.interval_s
    projected = ego_coords[..., :4, :].unsqueeze(3) + (
        velocity_xy[..., None, None, :] * delta_t.view(1, 1, 1, -1, 1, 1)
    )
    intersections = oriented_boxes_intersect(
        projected[..., None, :, :], future_agent_corners[:, None]
    )
    intersections &= future_valid[:, None]
    intersections &= ~scene.agent_is_red_light[:, None, None, None, :]
    intersections &= ~scene.agent_initially_collided[:, None, None, None, :]
    intersections &= speeds[..., None, None] >= config.ttc_stopped_speed_mps

    flattened = intersections.permute(0, 1, 4, 2, 3).reshape(
        batch_size, num_proposals, num_agents, num_steps * num_future
    )
    query_index = torch.arange(num_steps * num_future, device=states.device)
    first_query = torch.where(flattened, query_index, num_steps * num_future).amin(dim=-1)
    has_collision = first_query < num_steps * num_future
    safe_query = first_query.clamp(max=num_steps * num_future - 1)
    current_time = torch.div(safe_query, num_future, rounding_mode="floor")
    future_slot = torch.remainder(safe_query, num_future)
    observation_time = current_time + future_indices[future_slot]

    event_state = _gather_time(states, current_time)
    event_area = _gather_time(ego_area, current_time)
    event_intersection = _gather_time(rear_in_intersection[..., None], current_time).squeeze(-1)
    event_agent_centers = _gather_agent_time(
        scene.agent_corners.mean(dim=-2), observation_time
    )
    relative_angle = _relative_angle(event_state, event_agent_centers)
    ahead = relative_angle < (30.0 * pi / 180.0)
    behind = relative_angle > (150.0 * pi / 180.0)
    permissive_area = event_area[..., 0] | event_area[..., 1] | event_intersection
    ttc_fault = has_collision & (ahead | (permissive_area & ~behind))
    return (~ttc_fault.any(dim=-1)).to(states.dtype)


@lru_cache(maxsize=32)
def _savgol_matrix(
    length: int,
    window_length: int,
    poly_order: int,
    derivative_order: int,
    delta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build SciPy ``savgol_filter(mode='interp')`` as a Torch matrix."""

    window_length = min(window_length, length)
    if poly_order >= window_length:
        raise ValueError("poly_order must be smaller than window_length")
    sample = torch.arange(window_length, device=device, dtype=dtype)
    vandermonde = torch.stack([sample**power for power in range(poly_order + 1)], dim=-1)
    pseudo_inverse = torch.linalg.pinv(vandermonde)
    matrix = torch.zeros((length, length), device=device, dtype=dtype)
    half = window_length // 2
    center_position = half if window_length % 2 else half - 0.5

    def weights(position: float) -> Tensor:
        evaluation = torch.zeros(poly_order + 1, device=device, dtype=dtype)
        for power in range(derivative_order, poly_order + 1):
            evaluation[power] = (
                factorial(power)
                / factorial(power - derivative_order)
                * position ** (power - derivative_order)
                / delta**derivative_order
            )
        return evaluation @ pseudo_inverse

    center_weights = weights(center_position)
    for output_index in range(length):
        if output_index < half:
            start = 0
            row_weights = weights(float(output_index))
        elif output_index >= length - half:
            start = length - window_length
            row_weights = weights(float(output_index - start))
        else:
            start = output_index - half + (0 if window_length % 2 else 1)
            row_weights = center_weights
        matrix[output_index, start : start + window_length] = row_weights
    return matrix


def _savgol(values: Tensor, window: int, poly: int, deriv: int = 0, delta: float = 1.0) -> Tensor:
    matrix = _savgol_matrix(
        values.shape[-1], window, poly, deriv, delta, values.device, values.dtype
    )
    return values @ matrix.transpose(0, 1)


def _round8(values: Tensor) -> Tensor:
    return torch.round(values * 1e8) / 1e8


def _within(values: Tensor, minimum: float, maximum: float) -> Tensor:
    return ((values > minimum) & (values < maximum)).all(dim=-1)


def _comfort_metrics(states: Tensor, interval_s: float) -> Tensor:
    """Return the six official comfort checks before final conjunction."""

    num_steps = states.shape[-2]
    acceleration_x = states[..., STATE_ACCELERATION_X]
    acceleration_y = states[..., STATE_ACCELERATION_Y]

    lon_acceleration = _round8(_savgol(acceleration_x, num_steps, 2))
    lat_acceleration = _round8(_savgol(acceleration_y, num_steps, 2))

    acceleration_magnitude = torch.hypot(acceleration_x, acceleration_y)
    smoothed_magnitude = _round8(_savgol(acceleration_magnitude, 8, 2))
    magnitude_jerk = _round8(_savgol(smoothed_magnitude, num_steps, 2, 1, interval_s))
    smoothed_lon = _round8(_savgol(acceleration_x, 8, 2))
    lon_jerk = _round8(_savgol(smoothed_lon, num_steps, 2, 1, interval_s))

    headings = states[..., STATE_HEADING]
    difference = headings[..., 1:] - headings[..., :-1]
    adjustment = torch.cumsum(torch.round(difference / (2.0 * pi)), dim=-1)
    adjustment = torch.cat((torch.zeros_like(adjustment[..., :1]), adjustment), dim=-1)
    unwrapped = headings - 2.0 * pi * adjustment
    # The official implementation accepts a window_length argument in
    # _extract_ego_yaw_rate but does not forward it to _approximate_derivatives.
    # Consequently both yaw derivatives use the latter's default 5-point window.
    yaw_acceleration = _round8(_savgol(unwrapped, 5, 3, 2, interval_s))
    yaw_rate = _round8(_savgol(unwrapped, 5, 2, 1, interval_s))

    return torch.stack(
        (
            _within(lon_acceleration, -4.05, 2.40),
            _within(lat_acceleration, -4.89, 4.89),
            _within(magnitude_jerk, -8.37, 8.37),
            _within(lon_jerk, -4.13, 4.13),
            _within(yaw_acceleration, -1.93, 1.93),
            _within(yaw_rate, -0.95, 0.95),
        ),
        dim=-1,
    )


def _comfort(states: Tensor, interval_s: float) -> Tensor:
    return _comfort_metrics(states, interval_s).all(dim=-1).to(states.dtype)


def score_v1_gpu(
    states: Tensor,
    scene: PreparedPDMScene,
    geometry: VehicleGeometry,
    pdm_progress: Tensor,
    score_config: V1ScoreConfig = V1ScoreConfig(),
    metric_config: MetricConfig = MetricConfig(),
) -> tuple[Tensor, Tensor]:
    """Compute V1 targets and ego-area flags entirely on the tensor device.

    Returns:
        Targets ``[B, P, 7]`` and ego-area flags ``[B, P, T, 3]``.
    """

    _validate_inputs(states, scene)
    ego_coords = state_array_to_coords(
        states,
        geometry.half_length,
        geometry.half_width,
        geometry.rear_axle_to_center,
    )
    ego_area, rear_in_intersection = calculate_ego_area(states, ego_coords, scene)
    no_collision = _no_at_fault_collision(states, ego_coords, ego_area, scene, metric_config)
    drivable_area = (~ego_area[..., 1].any(dim=-1)).to(states.dtype)
    driving_direction = _driving_direction(ego_coords, ego_area, metric_config)
    progress_raw = _progress(ego_coords, scene)
    ttc = _ttc(
        states,
        ego_coords,
        ego_area,
        rear_in_intersection,
        scene,
        metric_config,
    )
    comfort = _comfort(states, metric_config.interval_s)
    targets = aggregate_v1_scores(
        no_collision,
        drivable_area,
        progress_raw,
        ttc,
        comfort,
        driving_direction,
        pdm_progress,
        score_config,
    )
    return targets, ego_area
