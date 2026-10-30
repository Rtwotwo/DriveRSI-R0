"""Experimental native-CUDA backend for the complete V1 PDM pipeline.

Triton kernels cover the geometry-heavy operations. The LQR matrix solves and
Savitzky-Golay filters stay in Torch so they use CUDA library kernels without
copying proposals back to CPU. This module is inference-only and is not wired
to the training agent.
"""

from __future__ import annotations

from math import pi
from typing import Optional

import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..ops import (
    STATE_HEADING,
    STATE_VELOCITY_X,
    STATE_VELOCITY_Y,
    VehicleGeometry,
    project_points_to_polyline,
)
from ..scorer import (
    MetricConfig,
    PreparedPDMScene,
    _comfort,
    _driving_direction,
    _front_edge_intersects_boxes,
    _gather_agent_time,
    _gather_time,
    _relative_angle,
    _validate_inputs,
)
from ..simulator import TorchPDMSimulator


@triton.jit
def _wrap_angle_cuda(angle):
    return libdevice.atan2(tl.sin(angle), tl.cos(angle))


@triton.jit
def _source_state_value(
    local_ptr,
    initial_ptr,
    batch_idx,
    proposal_idx,
    source_idx,
    state_idx,
    NUM_PROPOSALS: tl.constexpr,
    NUM_INPUT_POSES: tl.constexpr,
):
    initial_base = batch_idx * 11
    initial_value = tl.load(
        initial_ptr + initial_base + state_idx,
        mask=state_idx < 8,
        other=0.0,
    )
    safe_input_idx = tl.maximum(source_idx - 1, 0)
    local_base = (
        ((batch_idx * NUM_PROPOSALS + proposal_idx) * NUM_INPUT_POSES + safe_input_idx)
        * 3
    )
    has_future_pose = source_idx > 0
    local_x = tl.load(local_ptr + local_base, mask=has_future_pose, other=0.0)
    local_y = tl.load(local_ptr + local_base + 1, mask=has_future_pose, other=0.0)
    local_heading = tl.load(local_ptr + local_base + 2, mask=has_future_pose, other=0.0)
    x0 = tl.load(initial_ptr + initial_base)
    y0 = tl.load(initial_ptr + initial_base + 1)
    heading0 = tl.load(initial_ptr + initial_base + 2)
    cos_heading = tl.cos(heading0)
    sin_heading = tl.sin(heading0)
    global_x = x0 + local_x * cos_heading - local_y * sin_heading
    global_y = y0 + local_x * sin_heading + local_y * cos_heading
    global_heading = _wrap_angle_cuda(heading0 + local_heading)
    pose_value = tl.where(
        state_idx == 0,
        global_x,
        tl.where(state_idx == 1, global_y, tl.where(state_idx == 2, global_heading, 0.0)),
    )
    return tl.where(source_idx == 0, initial_value, pose_value)


@triton.jit
def _interpolate_transform_kernel(
    local_ptr,
    initial_ptr,
    output_ptr,
    NUM_PROPOSALS: tl.constexpr,
    NUM_INPUT_POSES: tl.constexpr,
    UPSAMPLE: tl.constexpr,
    NUM_OUTPUT_STATES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    batch_idx = program_idx // NUM_PROPOSALS
    proposal_idx = program_idx % NUM_PROPOSALS
    offsets = tl.arange(0, BLOCK)
    valid = offsets < NUM_OUTPUT_STATES * 11
    time_idx = offsets // 11
    state_idx = offsets % 11
    left_idx = time_idx // UPSAMPLE
    right_idx = tl.minimum(left_idx + 1, NUM_INPUT_POSES)
    alpha = (time_idx % UPSAMPLE).to(tl.float32) / UPSAMPLE
    left = _source_state_value(
        local_ptr,
        initial_ptr,
        batch_idx,
        proposal_idx,
        left_idx,
        state_idx,
        NUM_PROPOSALS,
        NUM_INPUT_POSES,
    )
    right = _source_state_value(
        local_ptr,
        initial_ptr,
        batch_idx,
        proposal_idx,
        right_idx,
        state_idx,
        NUM_PROPOSALS,
        NUM_INPUT_POSES,
    )
    value = left + alpha * (right - left)
    heading_delta = _wrap_angle_cuda(right - left)
    heading_value = _wrap_angle_cuda(left + alpha * heading_delta)
    value = tl.where(state_idx == 2, heading_value, value)
    output_base = program_idx * NUM_OUTPUT_STATES * 11
    tl.store(output_ptr + output_base + offsets, value, mask=valid)


def triton_proposals_to_state_array(
    local_poses: Tensor,
    initial_states: Tensor,
    input_interval_s: float = 0.5,
    output_interval_s: float = 0.1,
) -> Tensor:
    """Fused local-to-scene transform and 8-to-40 interpolation kernel."""

    _check_cuda_float(local_poses, "local_poses")
    _check_cuda_float(initial_states, "initial_states")
    if local_poses.ndim != 4 or local_poses.shape[-1] != 3:
        raise ValueError("local_poses must have shape [B, P, T, 3]")
    if initial_states.shape != (local_poses.shape[0], 11):
        raise ValueError("initial_states must have shape [B, 11]")
    ratio = input_interval_s / output_interval_s
    upsample = int(round(ratio))
    if upsample <= 0 or abs(ratio - upsample) > 1e-8:
        raise ValueError("the CUDA experiment requires an integer sampling ratio")
    local_poses = local_poses.contiguous()
    initial_states = initial_states.contiguous()
    batch_size, num_proposals, num_input_poses, _ = local_poses.shape
    num_output_states = num_input_poses * upsample + 1
    output = torch.empty(
        (batch_size, num_proposals, num_output_states, 11),
        device=local_poses.device,
        dtype=local_poses.dtype,
    )
    block = triton.next_power_of_2(num_output_states * 11)
    _interpolate_transform_kernel[(batch_size * num_proposals,)](
        local_poses,
        initial_states,
        output,
        NUM_PROPOSALS=num_proposals,
        NUM_INPUT_POSES=num_input_poses,
        UPSAMPLE=upsample,
        NUM_OUTPUT_STATES=num_output_states,
        BLOCK=block,
        num_warps=8,
    )
    return output


@triton.jit
def _state_coords_kernel(
    states_ptr,
    coords_ptr,
    num_states,
    half_length: tl.constexpr,
    half_width: tl.constexpr,
    rear_axle_to_center: tl.constexpr,
):
    state_idx = tl.program_id(0)
    valid_state = state_idx < num_states
    base = state_idx * 11
    x = tl.load(states_ptr + base, mask=valid_state)
    y = tl.load(states_ptr + base + 1, mask=valid_state)
    heading = tl.load(states_ptr + base + 2, mask=valid_state)
    cos_heading = tl.cos(heading)
    sin_heading = tl.sin(heading)
    center_x = x + rear_axle_to_center * cos_heading
    center_y = y + rear_axle_to_center * sin_heading
    offsets = tl.arange(0, 16)
    point_idx = offsets // 2
    coordinate_idx = offsets % 2
    valid = valid_state & (offsets < 10)
    longitudinal = tl.where(
        point_idx == 0,
        half_length,
        tl.where(
            point_idx == 1,
            -half_length,
            tl.where(point_idx == 2, -half_length, tl.where(point_idx == 3, half_length, 0.0)),
        ),
    )
    lateral = tl.where(
        point_idx == 0,
        half_width,
        tl.where(
            point_idx == 1,
            half_width,
            tl.where(point_idx == 2, -half_width, tl.where(point_idx == 3, -half_width, 0.0)),
        ),
    )
    point_x = center_x + longitudinal * cos_heading - lateral * sin_heading
    point_y = center_y + longitudinal * sin_heading + lateral * cos_heading
    value = tl.where(coordinate_idx == 0, point_x, point_y)
    tl.store(coords_ptr + state_idx * 10 + offsets, value, mask=valid)


def triton_state_array_to_coords(states: Tensor, geometry: VehicleGeometry) -> Tensor:
    """Create FL/RL/RR/FR/center coordinates with one program per state."""

    _check_cuda_float(states, "states")
    if states.shape[-1] != 11:
        raise ValueError("states must end in 11 values")
    states = states.contiguous()
    output = torch.empty((*states.shape[:-1], 5, 2), device=states.device, dtype=states.dtype)
    num_states = states.numel() // 11
    _state_coords_kernel[(num_states,)](
        states,
        output,
        num_states,
        half_length=geometry.half_length,
        half_width=geometry.half_width,
        rear_axle_to_center=geometry.rear_axle_to_center,
        num_warps=1,
    )
    return output


@triton.jit
def _axis_separated(
    ax0,
    ay0,
    ax1,
    ay1,
    ax2,
    ay2,
    ax3,
    ay3,
    bx0,
    by0,
    bx1,
    by1,
    bx2,
    by2,
    bx3,
    by3,
    axis_x,
    axis_y,
):
    ap0 = ax0 * axis_x + ay0 * axis_y
    ap1 = ax1 * axis_x + ay1 * axis_y
    ap2 = ax2 * axis_x + ay2 * axis_y
    ap3 = ax3 * axis_x + ay3 * axis_y
    bp0 = bx0 * axis_x + by0 * axis_y
    bp1 = bx1 * axis_x + by1 * axis_y
    bp2 = bx2 * axis_x + by2 * axis_y
    bp3 = bx3 * axis_x + by3 * axis_y
    a_min = tl.minimum(tl.minimum(ap0, ap1), tl.minimum(ap2, ap3))
    a_max = tl.maximum(tl.maximum(ap0, ap1), tl.maximum(ap2, ap3))
    b_min = tl.minimum(tl.minimum(bp0, bp1), tl.minimum(bp2, bp3))
    b_max = tl.maximum(tl.maximum(bp0, bp1), tl.maximum(bp2, bp3))
    return (a_max < b_min) | (b_max < a_min)


@triton.jit
def _pairwise_box_kernel(
    ego_ptr,
    agent_ptr,
    valid_ptr,
    output_ptr,
    total_pairs,
    NUM_PROPOSALS: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    NUM_FUTURE: tl.constexpr,
    NUM_AGENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = pair_idx < total_pairs
    agent_idx = pair_idx % NUM_AGENTS
    quotient = pair_idx // NUM_AGENTS
    future_idx = quotient % NUM_FUTURE
    quotient = quotient // NUM_FUTURE
    time_idx = quotient % NUM_STEPS
    quotient = quotient // NUM_STEPS
    proposal_idx = quotient % NUM_PROPOSALS
    batch_idx = quotient // NUM_PROPOSALS
    ego_base = (
        (((batch_idx * NUM_PROPOSALS + proposal_idx) * NUM_STEPS + time_idx) * NUM_FUTURE + future_idx)
        * 8
    )
    agent_base = (
        (((batch_idx * NUM_STEPS + time_idx) * NUM_FUTURE + future_idx) * NUM_AGENTS + agent_idx)
        * 8
    )
    ax0 = tl.load(ego_ptr + ego_base, mask=active)
    ay0 = tl.load(ego_ptr + ego_base + 1, mask=active)
    ax1 = tl.load(ego_ptr + ego_base + 2, mask=active)
    ay1 = tl.load(ego_ptr + ego_base + 3, mask=active)
    ax2 = tl.load(ego_ptr + ego_base + 4, mask=active)
    ay2 = tl.load(ego_ptr + ego_base + 5, mask=active)
    ax3 = tl.load(ego_ptr + ego_base + 6, mask=active)
    ay3 = tl.load(ego_ptr + ego_base + 7, mask=active)
    bx0 = tl.load(agent_ptr + agent_base, mask=active)
    by0 = tl.load(agent_ptr + agent_base + 1, mask=active)
    bx1 = tl.load(agent_ptr + agent_base + 2, mask=active)
    by1 = tl.load(agent_ptr + agent_base + 3, mask=active)
    bx2 = tl.load(agent_ptr + agent_base + 4, mask=active)
    by2 = tl.load(agent_ptr + agent_base + 5, mask=active)
    bx3 = tl.load(agent_ptr + agent_base + 6, mask=active)
    by3 = tl.load(agent_ptr + agent_base + 7, mask=active)
    separated = _axis_separated(
        ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3,
        bx0, by0, bx1, by1, bx2, by2, bx3, by3,
        -(ay1 - ay0), ax1 - ax0,
    )
    separated |= _axis_separated(
        ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3,
        bx0, by0, bx1, by1, bx2, by2, bx3, by3,
        -(ay3 - ay0), ax3 - ax0,
    )
    separated |= _axis_separated(
        ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3,
        bx0, by0, bx1, by1, bx2, by2, bx3, by3,
        -(by1 - by0), bx1 - bx0,
    )
    separated |= _axis_separated(
        ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3,
        bx0, by0, bx1, by1, bx2, by2, bx3, by3,
        -(by3 - by0), bx3 - bx0,
    )
    valid_offset = (
        ((batch_idx * NUM_STEPS + time_idx) * NUM_FUTURE + future_idx) * NUM_AGENTS
        + agent_idx
    )
    agent_valid = tl.load(valid_ptr + valid_offset, mask=active, other=0) != 0
    tl.store(output_ptr + pair_idx, agent_valid & ~separated, mask=active)


def triton_pairwise_box_intersections(
    ego_corners: Tensor,
    agent_corners: Tensor,
    agent_valid: Tensor,
) -> Tensor:
    """SAT collision for ``[B,P,T,F]`` ego-agent pairs."""

    _check_cuda_float(ego_corners, "ego_corners")
    _check_cuda_float(agent_corners, "agent_corners")
    if ego_corners.ndim != 6 or ego_corners.shape[-2:] != (4, 2):
        raise ValueError("ego_corners must have shape [B, P, T, F, 4, 2]")
    if agent_corners.ndim != 6 or agent_corners.shape[-2:] != (4, 2):
        raise ValueError("agent_corners must have shape [B, T, F, A, 4, 2]")
    batch_size, num_proposals, num_steps, num_future = ego_corners.shape[:4]
    if agent_corners.shape[:3] != (batch_size, num_steps, num_future):
        raise ValueError("ego and agent batch/time/future dimensions must match")
    num_agents = agent_corners.shape[3]
    if agent_valid.shape != (batch_size, num_steps, num_future, num_agents):
        raise ValueError("agent_valid has an invalid shape")
    if num_agents == 0:
        return torch.empty(
            (batch_size, num_proposals, num_steps, num_future, 0),
            device=ego_corners.device,
            dtype=torch.bool,
        )
    ego_corners = ego_corners.contiguous()
    agent_corners = agent_corners.contiguous()
    agent_valid = agent_valid.contiguous()
    output = torch.empty(
        (batch_size, num_proposals, num_steps, num_future, num_agents),
        device=ego_corners.device,
        dtype=torch.bool,
    )
    total_pairs = output.numel()
    block = 256
    _pairwise_box_kernel[(triton.cdiv(total_pairs, block),)](
        ego_corners,
        agent_corners,
        agent_valid,
        output,
        total_pairs,
        NUM_PROPOSALS=num_proposals,
        NUM_STEPS=num_steps,
        NUM_FUTURE=num_future,
        NUM_AGENTS=num_agents,
        BLOCK=block,
        num_warps=4,
    )
    return output


@triton.jit
def _polygon_crossing_kernel(
    points_ptr,
    edge_start_ptr,
    edge_end_ptr,
    edge_polygon_ptr,
    edge_valid_ptr,
    polygon_min_ptr,
    polygon_max_ptr,
    crossing_ptr,
    boundary_ptr,
    NUM_POINTS: tl.constexpr,
    NUM_EDGES: tl.constexpr,
    NUM_POLYGONS: tl.constexpr,
    boundary_eps,
    BLOCK_EDGES: tl.constexpr,
):
    point_flat_idx = tl.program_id(0)
    edge_idx = tl.program_id(1) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
    batch_idx = point_flat_idx // NUM_POINTS
    point_idx = point_flat_idx % NUM_POINTS
    valid_edge = edge_idx < NUM_EDGES
    edge_offset = batch_idx * NUM_EDGES + edge_idx
    polygon_idx = tl.load(edge_polygon_ptr + edge_offset, mask=valid_edge, other=0)
    valid_edge &= tl.load(edge_valid_ptr + edge_offset, mask=valid_edge, other=0) != 0
    valid_edge &= (polygon_idx >= 0) & (polygon_idx < NUM_POLYGONS)
    point_base = (batch_idx * NUM_POINTS + point_idx) * 2
    point_x = tl.load(points_ptr + point_base)
    point_y = tl.load(points_ptr + point_base + 1)
    polygon_base = (batch_idx * NUM_POLYGONS + polygon_idx) * 2
    min_x = tl.load(polygon_min_ptr + polygon_base, mask=valid_edge, other=0.0)
    min_y = tl.load(polygon_min_ptr + polygon_base + 1, mask=valid_edge, other=0.0)
    max_x = tl.load(polygon_max_ptr + polygon_base, mask=valid_edge, other=0.0)
    max_y = tl.load(polygon_max_ptr + polygon_base + 1, mask=valid_edge, other=0.0)
    candidate = valid_edge & (point_x >= min_x) & (point_x <= max_x)
    candidate &= (point_y >= min_y) & (point_y <= max_y)
    edge_base = edge_offset * 2
    start_x = tl.load(edge_start_ptr + edge_base, mask=candidate, other=0.0)
    start_y = tl.load(edge_start_ptr + edge_base + 1, mask=candidate, other=0.0)
    end_x = tl.load(edge_end_ptr + edge_base, mask=candidate, other=0.0)
    end_y = tl.load(edge_end_ptr + edge_base + 1, mask=candidate, other=0.0)
    edge_x = end_x - start_x
    edge_y = end_y - start_y
    edge_length_sq = edge_x * edge_x + edge_y * edge_y
    candidate &= edge_length_sq > boundary_eps * boundary_eps
    vertical_crossing = (start_y > point_y) != (end_y > point_y)
    denominator = tl.where(edge_y == 0.0, 1.0, edge_y)
    intersection_x = edge_x * (point_y - start_y) / denominator + start_x
    crosses = candidate & vertical_crossing & (point_x < intersection_x)
    output_offset = (batch_idx * NUM_POINTS + point_idx) * NUM_POLYGONS + polygon_idx
    tl.atomic_xor(crossing_ptr + output_offset, 1, mask=crosses)
    relative_x = point_x - start_x
    relative_y = point_y - start_y
    cross = edge_x * relative_y - edge_y * relative_x
    dot = relative_x * edge_x + relative_y * edge_y
    on_boundary = candidate & (tl.abs(cross) <= boundary_eps)
    on_boundary &= (dot >= -boundary_eps) & (dot <= edge_length_sq + boundary_eps)
    tl.atomic_or(boundary_ptr + output_offset, 1, mask=on_boundary)


@triton.jit
def _polygon_finalize_kernel(
    crossing_ptr,
    boundary_ptr,
    output_ptr,
    num_values,
    BOUNDARY_INSIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_values
    inside = (tl.load(crossing_ptr + offsets, mask=valid, other=0) & 1) != 0
    boundary = tl.load(boundary_ptr + offsets, mask=valid, other=0) != 0
    result = inside | boundary if BOUNDARY_INSIDE else inside & ~boundary
    tl.store(output_ptr + offsets, result, mask=valid)


def triton_points_in_polygon_edges(
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
) -> Tensor:
    """Even-odd map query with CUDA atomic parity accumulation."""

    _check_cuda_float(points, "points")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2]")
    batch_size, num_points, _ = points.shape
    if edge_start.shape != edge_end.shape or edge_start.ndim != 3:
        raise ValueError("edge tensors must have shape [B, E, 2]")
    num_edges = edge_start.shape[1]
    if edge_start.shape != (batch_size, num_edges, 2):
        raise ValueError("edge batch dimensions must match points")
    if edge_polygon.shape != (batch_size, num_edges) or edge_valid.shape != edge_polygon.shape:
        raise ValueError("edge metadata has an invalid shape")
    if polygon_min.shape != (batch_size, num_polygons, 2) or polygon_max.shape != polygon_min.shape:
        raise ValueError("polygon bounds have an invalid shape")
    if num_polygons == 0:
        return torch.empty((batch_size, num_points, 0), device=points.device, dtype=torch.bool)
    tensors = (
        points,
        edge_start,
        edge_end,
        edge_polygon,
        edge_valid,
        polygon_min,
        polygon_max,
    )
    points, edge_start, edge_end, edge_polygon, edge_valid, polygon_min, polygon_max = (
        tensor.contiguous() for tensor in tensors
    )
    shape = (batch_size, num_points, num_polygons)
    crossing = torch.zeros(shape, device=points.device, dtype=torch.int32)
    boundary = torch.zeros_like(crossing)
    output = torch.empty(shape, device=points.device, dtype=torch.bool)
    block_edges = 256
    _polygon_crossing_kernel[(batch_size * num_points, triton.cdiv(num_edges, block_edges))](
        points,
        edge_start,
        edge_end,
        edge_polygon,
        edge_valid,
        polygon_min,
        polygon_max,
        crossing,
        boundary,
        NUM_POINTS=num_points,
        NUM_EDGES=num_edges,
        NUM_POLYGONS=num_polygons,
        boundary_eps=boundary_eps,
        BLOCK_EDGES=block_edges,
        num_warps=4,
    )
    block = 256
    _polygon_finalize_kernel[(triton.cdiv(output.numel(), block),)](
        crossing,
        boundary,
        output,
        output.numel(),
        BOUNDARY_INSIDE=boundary_inside,
        BLOCK=block,
        num_warps=4,
    )
    return output


@triton.jit
def _polyline_projection_kernel(
    points_ptr,
    polyline_ptr,
    point_mask_ptr,
    cumulative_ptr,
    progress_ptr,
    distance_ptr,
    NUM_POINTS: tl.constexpr,
    NUM_LINE_POINTS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
):
    point_flat_idx = tl.program_id(0)
    batch_idx = point_flat_idx // NUM_POINTS
    point_idx = point_flat_idx % NUM_POINTS
    point_base = point_flat_idx * 2
    point_x = tl.load(points_ptr + point_base)
    point_y = tl.load(points_ptr + point_base + 1)
    segment_idx = tl.arange(0, BLOCK_SEGMENTS)
    segment_valid = segment_idx < NUM_LINE_POINTS - 1
    line_base = batch_idx * NUM_LINE_POINTS
    segment_valid &= (
        tl.load(point_mask_ptr + line_base + segment_idx, mask=segment_valid, other=0) != 0
    )
    segment_valid &= (
        tl.load(point_mask_ptr + line_base + segment_idx + 1, mask=segment_valid, other=0)
        != 0
    )
    start_base = (line_base + segment_idx) * 2
    end_base = (line_base + segment_idx + 1) * 2
    start_x = tl.load(polyline_ptr + start_base, mask=segment_valid, other=0.0)
    start_y = tl.load(polyline_ptr + start_base + 1, mask=segment_valid, other=0.0)
    end_x = tl.load(polyline_ptr + end_base, mask=segment_valid, other=0.0)
    end_y = tl.load(polyline_ptr + end_base + 1, mask=segment_valid, other=0.0)
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    length_sq = segment_x * segment_x + segment_y * segment_y
    segment_valid &= length_sq > 0.0
    safe_length_sq = tl.maximum(length_sq, EPS)
    fraction = ((point_x - start_x) * segment_x + (point_y - start_y) * segment_y) / safe_length_sq
    fraction = tl.maximum(0.0, tl.minimum(1.0, fraction))
    projected_x = start_x + fraction * segment_x
    projected_y = start_y + fraction * segment_y
    distance_sq = (point_x - projected_x) * (point_x - projected_x)
    distance_sq += (point_y - projected_y) * (point_y - projected_y)
    distance_sq = tl.where(segment_valid, distance_sq, float("inf"))
    best_distance_sq = tl.min(distance_sq, axis=0)
    best_segment = tl.min(
        tl.where(distance_sq == best_distance_sq, segment_idx, BLOCK_SEGMENTS), axis=0
    )
    segment_length = tl.sqrt(tl.maximum(length_sq, EPS))
    cumulative_base = batch_idx * (NUM_LINE_POINTS - 1)
    cumulative = tl.load(cumulative_ptr + cumulative_base + best_segment)
    best_fraction = tl.sum(tl.where(segment_idx == best_segment, fraction, 0.0), axis=0)
    best_length = tl.sum(tl.where(segment_idx == best_segment, segment_length, 0.0), axis=0)
    tl.store(progress_ptr + point_flat_idx, cumulative + best_fraction * best_length)
    tl.store(distance_ptr + point_flat_idx, tl.sqrt(best_distance_sq))


@triton.jit
def _polyline_projection_chunk_kernel(
    points_ptr,
    polyline_ptr,
    point_mask_ptr,
    cumulative_ptr,
    chunk_distance_ptr,
    chunk_segment_ptr,
    chunk_progress_ptr,
    NUM_POINTS: tl.constexpr,
    NUM_LINE_POINTS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
):
    program_idx = tl.program_id(0)
    point_flat_idx = program_idx // NUM_CHUNKS
    chunk_idx = program_idx % NUM_CHUNKS
    batch_idx = point_flat_idx // NUM_POINTS
    point_base = point_flat_idx * 2
    point_x = tl.load(points_ptr + point_base)
    point_y = tl.load(points_ptr + point_base + 1)

    segment_idx = chunk_idx * BLOCK_SEGMENTS + tl.arange(0, BLOCK_SEGMENTS)
    in_bounds = segment_idx < NUM_LINE_POINTS - 1
    line_base = batch_idx * NUM_LINE_POINTS
    segment_valid = in_bounds
    segment_valid &= (
        tl.load(point_mask_ptr + line_base + segment_idx, mask=in_bounds, other=0) != 0
    )
    segment_valid &= (
        tl.load(point_mask_ptr + line_base + segment_idx + 1, mask=in_bounds, other=0)
        != 0
    )
    start_base = (line_base + segment_idx) * 2
    end_base = (line_base + segment_idx + 1) * 2
    start_x = tl.load(polyline_ptr + start_base, mask=segment_valid, other=0.0)
    start_y = tl.load(polyline_ptr + start_base + 1, mask=segment_valid, other=0.0)
    end_x = tl.load(polyline_ptr + end_base, mask=segment_valid, other=0.0)
    end_y = tl.load(polyline_ptr + end_base + 1, mask=segment_valid, other=0.0)
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    length_sq = segment_x * segment_x + segment_y * segment_y
    segment_valid &= length_sq > 0.0
    safe_length_sq = tl.maximum(length_sq, EPS)
    fraction = (
        (point_x - start_x) * segment_x + (point_y - start_y) * segment_y
    ) / safe_length_sq
    fraction = tl.maximum(0.0, tl.minimum(1.0, fraction))
    projected_x = start_x + fraction * segment_x
    projected_y = start_y + fraction * segment_y
    distance_sq = (point_x - projected_x) * (point_x - projected_x)
    distance_sq += (point_y - projected_y) * (point_y - projected_y)
    distance_sq = tl.where(segment_valid, distance_sq, float("inf"))
    best_distance_sq = tl.min(distance_sq, axis=0)
    best_segment = tl.min(
        tl.where(distance_sq == best_distance_sq, segment_idx, NUM_LINE_POINTS), axis=0
    )
    segment_length = tl.sqrt(tl.maximum(length_sq, EPS))
    cumulative_base = batch_idx * (NUM_LINE_POINTS - 1)
    cumulative = tl.load(
        cumulative_ptr + cumulative_base + segment_idx,
        mask=in_bounds,
        other=0.0,
    )
    best_progress = tl.sum(
        tl.where(
            segment_idx == best_segment,
            cumulative + fraction * segment_length,
            0.0,
        ),
        axis=0,
    )
    tl.store(chunk_distance_ptr + program_idx, best_distance_sq)
    tl.store(chunk_segment_ptr + program_idx, best_segment)
    tl.store(chunk_progress_ptr + program_idx, best_progress)


@triton.jit
def _polyline_projection_finalize_kernel(
    chunk_distance_ptr,
    chunk_segment_ptr,
    chunk_progress_ptr,
    progress_ptr,
    distance_ptr,
    NUM_CHUNKS: tl.constexpr,
    NUM_LINE_POINTS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    point_flat_idx = tl.program_id(0)
    chunk_idx = tl.arange(0, BLOCK_CHUNKS)
    valid = chunk_idx < NUM_CHUNKS
    base = point_flat_idx * NUM_CHUNKS
    chunk_distance = tl.load(
        chunk_distance_ptr + base + chunk_idx, mask=valid, other=float("inf")
    )
    chunk_segment = tl.load(
        chunk_segment_ptr + base + chunk_idx, mask=valid, other=NUM_LINE_POINTS
    )
    best_distance_sq = tl.min(chunk_distance, axis=0)
    best_segment = tl.min(
        tl.where(
            chunk_distance == best_distance_sq,
            chunk_segment,
            NUM_LINE_POINTS,
        ),
        axis=0,
    )
    chunk_progress = tl.load(
        chunk_progress_ptr + base + chunk_idx, mask=valid, other=0.0
    )
    selected = (chunk_distance == best_distance_sq) & (chunk_segment == best_segment)
    progress = tl.sum(tl.where(selected, chunk_progress, 0.0), axis=0)
    tl.store(progress_ptr + point_flat_idx, progress)
    tl.store(distance_ptr + point_flat_idx, tl.sqrt(best_distance_sq))


def triton_project_points_to_polyline(
    points: Tensor,
    polyline: Tensor,
    point_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project points to a padded centerline in one CUDA program per point."""

    _check_cuda_float(points, "points")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2]")
    batch_size, num_points, _ = points.shape
    if polyline.ndim != 3 or polyline.shape[0] != batch_size or polyline.shape[-1] != 2:
        raise ValueError("polyline must have shape [B, L, 2]")
    if point_mask.shape != polyline.shape[:-1]:
        raise ValueError("point_mask must have shape [B, L]")
    num_line_points = polyline.shape[1]
    if num_line_points < 2:
        raise ValueError("polyline requires at least two points")
    points = points.contiguous()
    polyline = polyline.contiguous()
    point_mask = point_mask.contiguous()
    progress = torch.empty((batch_size, num_points), device=points.device, dtype=points.dtype)
    distance = torch.empty_like(progress)
    eps = torch.finfo(points.dtype).eps
    segment = polyline[:, 1:] - polyline[:, :-1]
    segment_length = torch.sqrt(
        (segment * segment).sum(dim=-1).clamp_min(eps)
    )
    cumulative = torch.cat(
        (torch.zeros_like(segment_length[:, :1]), torch.cumsum(segment_length, dim=-1)),
        dim=-1,
    )[:, :-1].contiguous()
    block_segments = triton.next_power_of_2(num_line_points - 1)
    if block_segments <= 4096:
        _polyline_projection_kernel[(batch_size * num_points,)](
            points,
            polyline,
            point_mask,
            cumulative,
            progress,
            distance,
            NUM_POINTS=num_points,
            NUM_LINE_POINTS=num_line_points,
            EPS=eps,
            BLOCK_SEGMENTS=block_segments,
            num_warps=8 if block_segments >= 512 else 4,
        )
        return progress, distance

    # A Triton program cannot safely materialize an unbounded power-of-two block.
    # Long centerlines use a chunked nearest-segment search followed by a second
    # reduction. The cumulative prefix is computed once per centerline, instead
    # of redundantly once per query point.
    chunk_segments = 1024
    num_segments = num_line_points - 1
    num_chunks = triton.cdiv(num_segments, chunk_segments)
    chunk_shape = (batch_size * num_points, num_chunks)
    chunk_distance = torch.empty(chunk_shape, device=points.device, dtype=points.dtype)
    chunk_segment = torch.empty(chunk_shape, device=points.device, dtype=torch.int32)
    chunk_progress = torch.empty_like(chunk_distance)
    _polyline_projection_chunk_kernel[(batch_size * num_points * num_chunks,)](
        points,
        polyline,
        point_mask,
        cumulative,
        chunk_distance,
        chunk_segment,
        chunk_progress,
        NUM_POINTS=num_points,
        NUM_LINE_POINTS=num_line_points,
        NUM_CHUNKS=num_chunks,
        EPS=eps,
        BLOCK_SEGMENTS=chunk_segments,
        num_warps=8,
    )
    block_chunks = triton.next_power_of_2(num_chunks)
    _polyline_projection_finalize_kernel[(batch_size * num_points,)](
        chunk_distance,
        chunk_segment,
        chunk_progress,
        progress,
        distance,
        NUM_CHUNKS=num_chunks,
        NUM_LINE_POINTS=num_line_points,
        BLOCK_CHUNKS=block_chunks,
        num_warps=4,
    )
    return progress, distance


@triton.jit
def _aggregate_kernel(
    no_collision_ptr,
    drivable_ptr,
    progress_ptr,
    ttc_ptr,
    comfort_ptr,
    direction_ptr,
    pdm_progress_ptr,
    output_ptr,
    num_proposals_total,
    progress_weight: tl.constexpr,
    ttc_weight: tl.constexpr,
    comfort_weight: tl.constexpr,
    direction_weight: tl.constexpr,
    progress_threshold: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_proposals_total
    no_collision = tl.load(no_collision_ptr + offsets, mask=valid)
    drivable = tl.load(drivable_ptr + offsets, mask=valid)
    raw_progress_input = tl.load(progress_ptr + offsets, mask=valid)
    ttc = tl.load(ttc_ptr + offsets, mask=valid)
    comfort = tl.load(comfort_ptr + offsets, mask=valid)
    direction = tl.load(direction_ptr + offsets, mask=valid)
    pdm_progress = tl.load(pdm_progress_ptr + offsets, mask=valid)
    multiplicative = no_collision * drivable
    raw_progress = raw_progress_input * multiplicative
    max_progress = tl.maximum(raw_progress, pdm_progress)
    fast = max_progress > progress_threshold
    normalized = tl.where(fast, raw_progress / tl.maximum(max_progress, EPS), 1.0)
    normalized = tl.where((~fast) & (multiplicative == 0.0), 0.0, normalized)
    weight_sum = progress_weight + ttc_weight + comfort_weight + direction_weight
    weighted = (
        normalized * progress_weight
        + ttc * ttc_weight
        + comfort * comfort_weight
        + direction * direction_weight
    ) / weight_sum
    final_score = multiplicative * weighted
    output_base = offsets * 7
    tl.store(output_ptr + output_base, no_collision, mask=valid)
    tl.store(output_ptr + output_base + 1, drivable, mask=valid)
    tl.store(output_ptr + output_base + 2, normalized, mask=valid)
    tl.store(output_ptr + output_base + 3, ttc, mask=valid)
    tl.store(output_ptr + output_base + 4, comfort, mask=valid)
    tl.store(output_ptr + output_base + 5, direction, mask=valid)
    tl.store(output_ptr + output_base + 6, final_score, mask=valid)


def triton_aggregate_v1_scores(
    no_collision: Tensor,
    drivable_area: Tensor,
    progress_raw: Tensor,
    ttc: Tensor,
    comfort: Tensor,
    driving_direction: Tensor,
    pdm_progress: Tensor,
    progress_weight: float = 5.0,
    ttc_weight: float = 5.0,
    comfort_weight: float = 2.0,
    driving_direction_weight: float = 0.0,
    progress_distance_threshold: float = 5.0,
) -> Tensor:
    """Fuse progress normalization and seven-target aggregation."""

    expected = no_collision.shape
    metrics = (drivable_area, progress_raw, ttc, comfort, driving_direction)
    if no_collision.ndim != 2 or any(value.shape != expected for value in metrics):
        raise ValueError("all metrics must share shape [B, P]")
    _check_cuda_float(no_collision, "no_collision")
    if pdm_progress.ndim == 1:
        pdm_progress = pdm_progress[:, None]
    pdm_progress = torch.broadcast_to(pdm_progress, expected).contiguous()
    inputs = [value.contiguous() for value in (no_collision, *metrics)]
    output = torch.empty((*expected, 7), device=no_collision.device, dtype=no_collision.dtype)
    block = 256
    _aggregate_kernel[(triton.cdiv(no_collision.numel(), block),)](
        *inputs,
        pdm_progress,
        output,
        no_collision.numel(),
        progress_weight=progress_weight,
        ttc_weight=ttc_weight,
        comfort_weight=comfort_weight,
        direction_weight=driving_direction_weight,
        progress_threshold=progress_distance_threshold,
        EPS=torch.finfo(no_collision.dtype).eps,
        BLOCK=block,
        num_warps=4,
    )
    return output


def _calculate_ego_area_triton(
    states: Tensor,
    ego_coords: Tensor,
    scene: PreparedPDMScene,
) -> tuple[Tensor, Tensor]:
    batch_size, num_proposals, num_steps = states.shape[:3]
    query_points = torch.cat((ego_coords, states[..., None, :2]), dim=-2)
    membership = triton_points_in_polygon_edges(
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
    multiple_lanes = lane_membership.any(dim=-2).sum(dim=-1) > 1
    contained_by_one_lane = (lane_membership.sum(dim=-2) == 4).any(dim=-1)
    multiple_lanes &= ~contained_by_one_lane
    drivable_membership = corner_membership & scene.drivable_polygon_mask[:, None, None, None, :]
    non_drivable = ~drivable_membership.any(dim=-1).all(dim=-1)
    center_membership = membership[..., 4, :]
    on_route = (center_membership & scene.route_lane_polygon_mask[:, None, None, :]).any(dim=-1)
    rear_membership = membership[..., 5, :]
    rear_in_intersection = (
        rear_membership & scene.intersection_polygon_mask[:, None, None, :]
    ).any(dim=-1)
    return torch.stack((multiple_lanes, non_drivable, ~on_route), dim=-1), rear_in_intersection


def _no_at_fault_collision_triton(
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
    valid = scene.agent_valid[:, :num_steps]
    intersections = triton_pairwise_box_intersections(
        ego_coords[..., None, :4, :],
        agent_corners[:, :, None],
        valid[:, :, None],
    ).squeeze(3)
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
    speed = torch.linalg.vector_norm(
        event_state[..., STATE_VELOCITY_X : STATE_VELOCITY_Y + 1], dim=-1
    )
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


def _progress_triton(ego_coords: Tensor, scene: PreparedPDMScene) -> Tensor:
    endpoints = torch.stack((ego_coords[:, :, 0, 4], ego_coords[:, :, -1, 4]), dim=2)
    batch_size, num_proposals = endpoints.shape[:2]
    # Use CUDA FP64 for this small projection. Float32 can change the selected
    # segment when adjacent segments differ by only one ULP, which amplifies a
    # millimetre along-line change during normalized score aggregation.
    along, _ = project_points_to_polyline(
        endpoints.reshape(batch_size, num_proposals * 2, 2).double(),
        scene.centerline.double(),
        scene.centerline_point_mask,
    )
    along = along.reshape(batch_size, num_proposals, 2)
    return (along[..., 1] - along[..., 0]).clamp_min(0.0).to(ego_coords.dtype)


def _ttc_triton(
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
    num_future = len(future_indices)
    future_agent_corners = scene.agent_corners.index_select(
        1, observation_indices.flatten()
    ).reshape(batch_size, num_steps, num_future, num_agents, 4, 2)
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
    intersections = triton_pairwise_box_intersections(projected, future_agent_corners, future_valid)
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
    event_intersection = _gather_time(
        rear_in_intersection[..., None], current_time
    ).squeeze(-1)
    event_agent_centers = _gather_agent_time(scene.agent_corners.mean(dim=-2), observation_time)
    relative_angle = _relative_angle(event_state, event_agent_centers)
    ahead = relative_angle < (30.0 * pi / 180.0)
    behind = relative_angle > (150.0 * pi / 180.0)
    permissive_area = event_area[..., 0] | event_area[..., 1] | event_intersection
    ttc_fault = has_collision & (ahead | (permissive_area & ~behind))
    return (~ttc_fault.any(dim=-1)).to(states.dtype)


def score_v1_triton(
    states: Tensor,
    scene: PreparedPDMScene,
    geometry: VehicleGeometry,
    pdm_progress: Tensor,
    metric_config: MetricConfig = MetricConfig(),
) -> tuple[Tensor, Tensor]:
    """Run every V1 scorer stage on CUDA, using native kernels for geometry."""

    _validate_inputs(states, scene)
    _check_cuda_float(states, "states")
    ego_coords = triton_state_array_to_coords(states, geometry)
    ego_area, rear_in_intersection = _calculate_ego_area_triton(states, ego_coords, scene)
    no_collision = _no_at_fault_collision_triton(
        states, ego_coords, ego_area, scene, metric_config
    )
    drivable_area = (~ego_area[..., 1].any(dim=-1)).to(states.dtype)
    driving_direction = _driving_direction(ego_coords, ego_area, metric_config)
    progress_raw = _progress_triton(ego_coords, scene)
    ttc = _ttc_triton(
        states, ego_coords, ego_area, rear_in_intersection, scene, metric_config
    )
    comfort = _comfort(states, metric_config.interval_s)
    targets = triton_aggregate_v1_scores(
        no_collision,
        drivable_area,
        progress_raw,
        ttc,
        comfort,
        driving_direction,
        pdm_progress,
    )
    return targets, ego_area


def run_v1_triton_pipeline(
    local_poses: Tensor,
    initial_states: Tensor,
    scene: PreparedPDMScene,
    geometry: VehicleGeometry,
    pdm_progress: Tensor,
    simulator: Optional[TorchPDMSimulator] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run interpolation, LQR simulation and V1 scoring without CPU transfer."""

    reference_states = triton_proposals_to_state_array(local_poses, initial_states)
    if simulator is None:
        simulator = TorchPDMSimulator(geometry.wheel_base).to(
            device=local_poses.device, dtype=local_poses.dtype
        )
    simulated_states = simulator(reference_states, initial_states)
    targets, ego_area = score_v1_triton(
        simulated_states, scene, geometry, pdm_progress
    )
    return reference_states, simulated_states, targets, ego_area


def _check_cuda_float(tensor: Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use float32 in the CUDA experiment")
