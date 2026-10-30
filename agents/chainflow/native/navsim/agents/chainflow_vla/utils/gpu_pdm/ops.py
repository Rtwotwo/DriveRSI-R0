"""Tensor-only building blocks for the ChainFlow V1 GPU PDM scorer.

The production CPU scorer in :mod:`pdm_score` remains the correctness
reference.  This module deliberately contains no NumPy, Shapely, Ray, or
metric-cache I/O so every operation can stay on the training rank's GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


STATE_SIZE = 11
STATE_X = 0
STATE_Y = 1
STATE_HEADING = 2
STATE_VELOCITY_X = 3
STATE_VELOCITY_Y = 4
STATE_ACCELERATION_X = 5
STATE_ACCELERATION_Y = 6
STATE_STEERING_ANGLE = 7
STATE_STEERING_RATE = 8
STATE_ANGULAR_VELOCITY = 9
STATE_ANGULAR_ACCELERATION = 10


@dataclass(frozen=True)
class VehicleGeometry:
    """Vehicle dimensions used by the PDM rear-axle state representation."""

    half_length: float
    half_width: float
    rear_axle_to_center: float
    wheel_base: float


@dataclass(frozen=True)
class V1ScoreConfig:
    """Weights and thresholds matching ``utils/pdm_score.py``."""

    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    comfort_weight: float = 2.0
    driving_direction_weight: float = 0.0
    progress_distance_threshold: float = 5.0


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angles to ``[-pi, pi]`` without leaving the tensor device."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def local_poses_to_global(local_poses: Tensor, initial_pose: Tensor) -> Tensor:
    """Transform local ``(x, y, heading)`` poses to the global SE(2) frame.

    Args:
        local_poses: ``[B, ..., T, 3]`` local rear-axle poses.
        initial_pose: ``[B, 3]`` global rear-axle pose.
    """

    if local_poses.ndim < 3 or local_poses.shape[-1] != 3:
        raise ValueError(f"local_poses must have shape [B, ..., T, 3], got {tuple(local_poses.shape)}")
    if initial_pose.ndim != 2 or initial_pose.shape != (local_poses.shape[0], 3):
        raise ValueError(f"initial_pose must have shape {(local_poses.shape[0], 3)}, got {tuple(initial_pose.shape)}")

    expand_dims = (1,) * (local_poses.ndim - 2)
    x0 = initial_pose[:, 0].reshape(-1, *expand_dims)
    y0 = initial_pose[:, 1].reshape(-1, *expand_dims)
    heading0 = initial_pose[:, 2].reshape(-1, *expand_dims)
    cos_heading = torch.cos(heading0)
    sin_heading = torch.sin(heading0)

    local_x, local_y, local_heading = local_poses.unbind(dim=-1)
    global_x = x0 + local_x * cos_heading - local_y * sin_heading
    global_y = y0 + local_x * sin_heading + local_y * cos_heading
    global_heading = wrap_angle(heading0 + local_heading)
    return torch.stack((global_x, global_y, global_heading), dim=-1)


def proposals_to_state_array(
    local_poses: Tensor,
    initial_states: Tensor,
    input_interval_s: float = 0.5,
    output_interval_s: float = 0.1,
) -> Tensor:
    """Convert local proposals to interpolated PDM state arrays on-device.

    The interpolation mirrors NAVSIM's trajectory construction: the full
    initial ego state is prepended, future poses contain zero dynamic state,
    and the sequence is linearly interpolated. Heading uses shortest-arc
    interpolation.

    Args:
        local_poses: ``[B, P, T, 3]`` future local poses.
        initial_states: ``[B, 11]`` global ego state arrays.

    Returns:
        ``[B, P, output_steps + 1, 11]`` reference states.
    """

    if local_poses.ndim != 4 or local_poses.shape[-1] != 3:
        raise ValueError(f"local_poses must have shape [B, P, T, 3], got {tuple(local_poses.shape)}")
    if initial_states.shape != (local_poses.shape[0], STATE_SIZE):
        raise ValueError(
            f"initial_states must have shape {(local_poses.shape[0], STATE_SIZE)}, "
            f"got {tuple(initial_states.shape)}"
        )
    if input_interval_s <= 0.0 or output_interval_s <= 0.0:
        raise ValueError("sampling intervals must be positive")

    batch_size, num_proposals, num_input_poses, _ = local_poses.shape
    horizon_s = num_input_poses * input_interval_s
    num_output_poses = int(round(horizon_s / output_interval_s))
    if abs(num_output_poses * output_interval_s - horizon_s) > 1e-8:
        raise ValueError("trajectory horizon must be divisible by output_interval_s")

    global_poses = local_poses_to_global(local_poses, initial_states[:, :3])
    source = local_poses.new_zeros((batch_size, num_proposals, num_input_poses + 1, STATE_SIZE))
    # InterpolatedTrajectory preserves pose, velocity, acceleration, and tire
    # angle, but its interpolated state representation zeros steering rate and
    # angular derivatives. Match get_trajectory_as_array exactly here; the
    # simulator still receives the complete initial state separately.
    source[:, :, 0, :STATE_STEERING_RATE] = initial_states[:, None, :STATE_STEERING_RATE]
    source[:, :, 1:, :3] = global_poses

    output_times = torch.arange(
        num_output_poses + 1,
        device=local_poses.device,
        dtype=local_poses.dtype,
    ) * output_interval_s
    source_position = output_times / input_interval_s
    left_idx = torch.floor(source_position).to(torch.long).clamp(max=num_input_poses)
    right_idx = (left_idx + 1).clamp(max=num_input_poses)
    alpha = (source_position - left_idx.to(source_position.dtype)).view(1, 1, -1, 1)

    left = source.index_select(2, left_idx)
    right = source.index_select(2, right_idx)
    output = left + alpha * (right - left)
    heading_delta = wrap_angle(right[..., STATE_HEADING] - left[..., STATE_HEADING])
    output[..., STATE_HEADING] = wrap_angle(
        left[..., STATE_HEADING] + alpha.squeeze(-1) * heading_delta
    )
    return output


def state_array_to_coords(
    states: Tensor,
    half_length: float,
    half_width: float,
    rear_axle_to_center: float,
) -> Tensor:
    """Return PDM box coordinates ordered as FL, RL, RR, FR, center."""

    if states.shape[-1] != STATE_SIZE:
        raise ValueError(f"states must end in {STATE_SIZE} values, got {tuple(states.shape)}")

    heading = states[..., STATE_HEADING]
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    rear_axle = states[..., :2]
    center = rear_axle + torch.stack(
        (rear_axle_to_center * cos_heading, rear_axle_to_center * sin_heading), dim=-1
    )

    def translate(longitudinal: float, lateral: float) -> Tensor:
        return center + torch.stack(
            (
                longitudinal * cos_heading - lateral * sin_heading,
                longitudinal * sin_heading + lateral * cos_heading,
            ),
            dim=-1,
        )

    front_left = translate(half_length, half_width)
    rear_left = translate(-half_length, half_width)
    rear_right = translate(-half_length, -half_width)
    front_right = translate(half_length, -half_width)
    return torch.stack((front_left, rear_left, rear_right, front_right, center), dim=-2)


def oriented_boxes_intersect(corners_a: Tensor, corners_b: Tensor, eps: float = 0.0) -> Tensor:
    """Batched exact SAT intersection for convex oriented rectangles.

    Leading dimensions follow normal PyTorch broadcasting. Touching boxes are
    intersections, matching Shapely's ``intersects`` predicate.
    """

    if corners_a.shape[-2:] != (4, 2) or corners_b.shape[-2:] != (4, 2):
        raise ValueError("oriented boxes must have trailing shape [4, 2]")

    corners_a, corners_b = torch.broadcast_tensors(corners_a, corners_b)

    edges_a = torch.stack(
        (corners_a[..., 1, :] - corners_a[..., 0, :], corners_a[..., 3, :] - corners_a[..., 0, :]),
        dim=-2,
    )
    edges_b = torch.stack(
        (corners_b[..., 1, :] - corners_b[..., 0, :], corners_b[..., 3, :] - corners_b[..., 0, :]),
        dim=-2,
    )
    axes = torch.cat((edges_a, edges_b), dim=-2)
    axes = torch.stack((-axes[..., 1], axes[..., 0]), dim=-1)

    projections_a = torch.einsum("...ci,...ai->...ca", corners_a, axes)
    projections_b = torch.einsum("...ci,...ai->...ca", corners_b, axes)
    separated = (projections_a.amax(dim=-2) < projections_b.amin(dim=-2) - eps) | (
        projections_b.amax(dim=-2) < projections_a.amin(dim=-2) - eps
    )
    return ~separated.any(dim=-1)


def pairwise_ego_agent_intersections(ego_corners: Tensor, agent_corners: Tensor) -> Tensor:
    """Intersect ego proposals with agents at matching time indices.

    Args:
        ego_corners: ``[B, P, T, 4, 2]``.
        agent_corners: ``[B, T, A, 4, 2]``.

    Returns:
        Boolean tensor ``[B, P, T, A]``.
    """

    if ego_corners.ndim != 5 or agent_corners.ndim != 5:
        raise ValueError("ego_corners and agent_corners must be rank-5 tensors")
    if ego_corners.shape[0] != agent_corners.shape[0] or ego_corners.shape[2] != agent_corners.shape[1]:
        raise ValueError("ego and agent batch/time dimensions must match")
    return oriented_boxes_intersect(ego_corners.unsqueeze(3), agent_corners.unsqueeze(1))


def points_in_padded_polygons(
    points: Tensor,
    polygon_vertices: Tensor,
    vertex_mask: Tensor,
    boundary_inside: bool = False,
    boundary_eps: float = 1e-7,
    point_chunk_size: Optional[int] = 2048,
    polygon_chunk_size: Optional[int] = 32,
) -> Tensor:
    """Ray-crossing point-in-polygon for padded, variable-length polygons.

    Args:
        points: ``[B, N, 2]``.
        polygon_vertices: ``[B, K, V, 2]`` with valid vertices first.
        vertex_mask: ``[B, K, V]``.

    Returns:
        Boolean membership tensor ``[B, N, K]``.
    """

    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2]")
    if polygon_vertices.ndim != 4 or polygon_vertices.shape[-1] != 2:
        raise ValueError("polygon_vertices must have shape [B, K, V, 2]")
    if vertex_mask.shape != polygon_vertices.shape[:-1]:
        raise ValueError("vertex_mask must match polygon_vertices leading dimensions")
    if points.shape[0] != polygon_vertices.shape[0]:
        raise ValueError("point and polygon batch dimensions must match")

    batch_size, num_polygons, max_vertices, _ = polygon_vertices.shape
    if num_polygons == 0:
        return torch.empty(
            (batch_size, points.shape[1], 0), device=points.device, dtype=torch.bool
        )
    if (
        (point_chunk_size is not None and points.shape[1] > point_chunk_size)
        or (polygon_chunk_size is not None and num_polygons > polygon_chunk_size)
    ):
        point_step = point_chunk_size or points.shape[1]
        polygon_step = polygon_chunk_size or num_polygons
        point_results = []
        for point_start in range(0, points.shape[1], point_step):
            polygon_results = []
            point_slice = points[:, point_start : point_start + point_step]
            for polygon_start in range(0, num_polygons, polygon_step):
                polygon_results.append(
                    points_in_padded_polygons(
                        point_slice,
                        polygon_vertices[:, polygon_start : polygon_start + polygon_step],
                        vertex_mask[:, polygon_start : polygon_start + polygon_step],
                        boundary_inside=boundary_inside,
                        boundary_eps=boundary_eps,
                        point_chunk_size=None,
                        polygon_chunk_size=None,
                    )
                )
            point_results.append(torch.cat(polygon_results, dim=-1))
        return torch.cat(point_results, dim=1)

    counts = vertex_mask.sum(dim=-1)
    if torch.any((counts > 0) & (counts < 3)):
        raise ValueError("valid polygons must contain at least three vertices")

    vertex_idx = torch.arange(max_vertices, device=points.device).view(1, 1, -1)
    safe_counts = counts.clamp(min=1)
    next_idx = torch.remainder(vertex_idx + 1, safe_counts.unsqueeze(-1))
    next_vertices = polygon_vertices.gather(
        2, next_idx.unsqueeze(-1).expand(batch_size, num_polygons, max_vertices, 2)
    )

    start = polygon_vertices[:, None, :, :, :]
    end = next_vertices[:, None, :, :, :]
    point = points[:, :, None, None, :]
    edge = end - start
    edge_length_sq = (edge * edge).sum(dim=-1)
    edge_valid = vertex_mask[:, None, :, :] & (edge_length_sq > boundary_eps**2)

    start_x, start_y = start[..., 0], start[..., 1]
    end_x, end_y = end[..., 0], end[..., 1]
    point_x, point_y = point[..., 0], point[..., 1]
    denominator = end_y - start_y
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    crosses = ((start_y > point_y) != (end_y > point_y)) & (
        point_x < (end_x - start_x) * (point_y - start_y) / safe_denominator + start_x
    )
    inside = ((crosses & edge_valid).sum(dim=-1) % 2) == 1

    relative = point - start
    cross = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
    dot = relative[..., 0] * edge[..., 0] + relative[..., 1] * edge[..., 1]
    on_boundary = (
        (cross.abs() <= boundary_eps)
        & (dot >= -boundary_eps)
        & (dot <= edge_length_sq + boundary_eps)
        & edge_valid
    ).any(dim=-1)
    return inside | on_boundary if boundary_inside else inside & ~on_boundary


def points_in_padded_polygon_edges(
    points: Tensor,
    edge_start: Tensor,
    edge_end: Tensor,
    edge_polygon: Tensor,
    edge_valid: Tensor,
    polygon_min: Tensor,
    polygon_max: Tensor,
    num_polygons: int,
    boundary_inside: bool = False,
    boundary_eps: float = 1e-7,
    point_chunk_size: int = 2048,
    edge_chunk_size: int = 2048,
) -> Tensor:
    """Point-in-polygon using flat valid edges and polygon-id scatter.

    Exterior and interior-ring edges share the same polygon id, so the even-
    odd rule handles holes without a separate pass. This avoids the expensive
    ``num_polygons * max_vertices`` padding used by the reference helper.
    """

    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2]")
    if edge_start.shape != edge_end.shape or edge_start.ndim != 3 or edge_start.shape[-1] != 2:
        raise ValueError("edge_start and edge_end must have shape [B, E, 2]")
    if edge_polygon.shape != edge_start.shape[:-1] or edge_valid.shape != edge_polygon.shape:
        raise ValueError("edge_polygon and edge_valid must have shape [B, E]")
    if polygon_min.shape != (points.shape[0], num_polygons, 2) or polygon_max.shape != polygon_min.shape:
        raise ValueError("polygon bounds must have shape [B, K, 2]")
    if points.shape[0] != edge_start.shape[0] or num_polygons < 0:
        raise ValueError("invalid batch dimension or polygon count")

    batch_size, num_points, _ = points.shape
    if num_polygons == 0:
        return torch.empty((batch_size, num_points, 0), device=points.device, dtype=torch.bool)
    output = torch.empty((batch_size, num_points, num_polygons), device=points.device, dtype=torch.bool)
    for batch_idx in range(batch_size):
        for point_start in range(0, num_points, point_chunk_size):
            point_block = points[batch_idx, point_start : point_start + point_chunk_size]
            block_size = point_block.shape[0]
            polygon_candidate = (
                (point_block[:, None] >= polygon_min[batch_idx, None])
                & (point_block[:, None] <= polygon_max[batch_idx, None])
            ).all(dim=-1)
            active_polygon = polygon_candidate.any(dim=0)
            active_edge = edge_valid[batch_idx] & active_polygon[edge_polygon[batch_idx]]
            selected_start = edge_start[batch_idx, active_edge]
            selected_end = edge_end[batch_idx, active_edge]
            selected_polygon = edge_polygon[batch_idx, active_edge]
            crossing_count = torch.zeros(
                (block_size, num_polygons), device=points.device, dtype=torch.int32
            )
            boundary_count = torch.zeros_like(crossing_count)
            point = point_block[:, None, :]
            for edge_offset in range(0, len(selected_start), edge_chunk_size):
                start = selected_start[None, edge_offset : edge_offset + edge_chunk_size]
                end = selected_end[None, edge_offset : edge_offset + edge_chunk_size]
                polygon_ids = selected_polygon[edge_offset : edge_offset + edge_chunk_size]
                polygon = polygon_ids[None].expand(block_size, -1)
                valid = polygon_candidate.index_select(1, polygon_ids)

                start_x, start_y = start[..., 0], start[..., 1]
                end_x, end_y = end[..., 0], end[..., 1]
                point_x, point_y = point[..., 0], point[..., 1]
                denominator = end_y - start_y
                safe_denominator = torch.where(
                    denominator == 0, torch.ones_like(denominator), denominator
                )
                crosses = ((start_y > point_y) != (end_y > point_y)) & (
                    point_x
                    < (end_x - start_x) * (point_y - start_y) / safe_denominator + start_x
                )
                crossing_count.scatter_add_(-1, polygon, (crosses & valid).to(torch.int32))

                edge = end - start
                relative = point - start
                cross = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
                dot = (relative * edge).sum(dim=-1)
                edge_length_sq = (edge * edge).sum(dim=-1)
                on_boundary = (
                    (cross.abs() <= boundary_eps)
                    & (dot >= -boundary_eps)
                    & (dot <= edge_length_sq + boundary_eps)
                    & valid
                )
                boundary_count.scatter_add_(-1, polygon, on_boundary.to(torch.int32))
            inside = crossing_count.remainder(2) == 1
            on_boundary = boundary_count > 0
            output[batch_idx, point_start : point_start + block_size] = (
                inside | on_boundary if boundary_inside else inside & ~on_boundary
            )
    return output


def project_points_to_polyline(
    points: Tensor,
    polyline: Tensor,
    point_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Project points onto padded polylines.

    Args:
        points: ``[B, N, 2]``.
        polyline: ``[B, L, 2]``.
        point_mask: ``[B, L]`` with valid points first.

    Returns:
        Tuple of along-line progress and lateral distance, each ``[B, N]``.
    """

    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2]")
    if polyline.ndim != 3 or polyline.shape[-1] != 2 or point_mask.shape != polyline.shape[:-1]:
        raise ValueError("polyline/point_mask must have shapes [B, L, 2] and [B, L]")
    if polyline.shape[1] < 2:
        raise ValueError("polyline requires at least two padded points")
    if torch.any(point_mask.sum(dim=-1) < 2):
        raise ValueError("every polyline must contain at least two valid points")

    start = polyline[:, :-1]
    end = polyline[:, 1:]
    segment = end - start
    segment_length_sq = (segment * segment).sum(dim=-1)
    segment_length = torch.sqrt(segment_length_sq.clamp_min(torch.finfo(polyline.dtype).eps))
    segment_valid = point_mask[:, :-1] & point_mask[:, 1:] & (segment_length_sq > 0)

    relative = points[:, :, None, :] - start[:, None, :, :]
    fraction = (relative * segment[:, None, :, :]).sum(dim=-1) / segment_length_sq[:, None, :].clamp_min(
        torch.finfo(polyline.dtype).eps
    )
    fraction = fraction.clamp(0.0, 1.0)
    projected = start[:, None, :, :] + fraction[..., None] * segment[:, None, :, :]
    distance_sq = ((points[:, :, None, :] - projected) ** 2).sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~segment_valid[:, None, :], torch.inf)
    best_segment = distance_sq.argmin(dim=-1)

    cumulative = torch.cat(
        (torch.zeros_like(segment_length[:, :1]), torch.cumsum(segment_length, dim=-1)), dim=-1
    )
    cumulative_start = cumulative[:, :-1]
    progress_per_segment = cumulative_start[:, None, :] + fraction * segment_length[:, None, :]
    progress = progress_per_segment.gather(-1, best_segment.unsqueeze(-1)).squeeze(-1)
    distance = torch.sqrt(distance_sq.gather(-1, best_segment.unsqueeze(-1)).squeeze(-1))
    return progress, distance


def aggregate_v1_scores(
    no_collision: Tensor,
    drivable_area: Tensor,
    progress_raw: Tensor,
    ttc: Tensor,
    comfort: Tensor,
    driving_direction: Tensor,
    pdm_progress: Tensor,
    config: V1ScoreConfig = V1ScoreConfig(),
) -> Tensor:
    """Aggregate V1 metrics and return the seven training targets.

    All proposal tensors have shape ``[B, P]``. ``pdm_progress`` may be
    ``[B]``, ``[B, 1]``, or ``[B, P]``.
    """

    expected_shape = no_collision.shape
    metrics = (drivable_area, progress_raw, ttc, comfort, driving_direction)
    if any(metric.shape != expected_shape for metric in metrics):
        raise ValueError("all proposal metrics must share shape [B, P]")
    if no_collision.ndim != 2:
        raise ValueError("proposal metrics must have shape [B, P]")

    multiplicative = no_collision * drivable_area
    raw_progress = progress_raw * multiplicative
    if pdm_progress.ndim == 1:
        pdm_progress = pdm_progress[:, None]
    pdm_progress = torch.broadcast_to(pdm_progress, raw_progress.shape)
    max_progress = torch.maximum(raw_progress, pdm_progress)
    fast = max_progress > config.progress_distance_threshold
    normalized_progress = torch.ones_like(raw_progress)
    normalized_progress = torch.where(
        fast,
        raw_progress / max_progress.clamp_min(torch.finfo(raw_progress.dtype).eps),
        normalized_progress,
    )
    normalized_progress = torch.where((~fast) & (multiplicative == 0), 0.0, normalized_progress)

    weight_sum = (
        config.progress_weight
        + config.ttc_weight
        + config.comfort_weight
        + config.driving_direction_weight
    )
    weighted = (
        normalized_progress * config.progress_weight
        + ttc * config.ttc_weight
        + comfort * config.comfort_weight
        + driving_direction * config.driving_direction_weight
    ) / weight_sum
    final_score = multiplicative * weighted
    return torch.stack(
        (
            no_collision,
            drivable_area,
            normalized_progress,
            ttc,
            comfort,
            driving_direction,
            final_score,
        ),
        dim=-1,
    )
