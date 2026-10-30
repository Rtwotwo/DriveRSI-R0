"""Adversarial operator parity checks for Torch and Triton GPU PDM backends."""

from __future__ import annotations

import json
from math import pi
from typing import Union

import numpy as np
import torch
from shapely.geometry import LineString, Point, Polygon

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.chainflow_vla.utils.gpu_pdm.cuda.triton_backend import (
    triton_aggregate_v1_scores,
    triton_pairwise_box_intersections,
    triton_points_in_polygon_edges,
    triton_project_points_to_polyline,
    triton_proposals_to_state_array,
    triton_state_array_to_coords,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    VehicleGeometry,
    aggregate_v1_scores,
    oriented_boxes_intersect,
    points_in_padded_polygon_edges,
    project_points_to_polyline,
    proposals_to_state_array,
    state_array_to_coords,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scorer import _comfort_metrics
from navsim.agents.chainflow_vla.utils.gpu_pdm.simulator import TorchPDMSimulator
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)


def _max_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().amax().cpu())


def _geometry() -> VehicleGeometry:
    vehicle = get_pacifica_parameters()
    return VehicleGeometry(
        vehicle.half_length,
        vehicle.half_width,
        vehicle.rear_axle_to_center,
        vehicle.wheel_base,
    )


def check_interpolation(device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(1949)
    local = torch.randn((4, 64, 8, 3), generator=generator, device=device) * 0.4
    local[..., 0] += torch.arange(1, 9, device=device) * 1.3
    local[..., 2] *= 0.25
    initial = torch.randn((4, 11), generator=generator, device=device) * 0.2
    initial[:, :2] *= 30.0
    initial[:, 2] = torch.tensor(
        [pi - 1e-3, -pi + 1e-3, pi / 2, -pi / 2], device=device
    )
    reference = proposals_to_state_array(local, initial)
    candidate = triton_proposals_to_state_array(local, initial)
    pose_error = _max_error(reference[..., :2], candidate[..., :2])
    state_error = _max_error(reference[..., 3:], candidate[..., 3:])
    heading_error = float(
        torch.atan2(
            torch.sin(candidate[..., 2] - reference[..., 2]),
            torch.cos(candidate[..., 2] - reference[..., 2]),
        )
        .abs()
        .amax()
        .cpu()
    )
    assert pose_error <= 2e-5
    assert heading_error <= 2e-6
    assert state_error <= 2e-6
    return {
        "xy_max_abs_error": pose_error,
        "heading_wrapped_max_abs_error": heading_error,
        "dynamic_state_max_abs_error": state_error,
    }


def check_state_coords(device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(7)
    states = torch.randn((3, 17, 41, 11), generator=generator, device=device)
    states[..., :2] *= 50.0
    states[..., 2] *= pi
    geometry = _geometry()
    reference = state_array_to_coords(
        states,
        geometry.half_length,
        geometry.half_width,
        geometry.rear_axle_to_center,
    )
    candidate = triton_state_array_to_coords(states, geometry)
    error = _max_error(reference, candidate)
    # One float32 ULP at the tested |xy|~50 scale; formulas are checked again
    # through exact SAT decisions and real-cache scorer parity below.
    assert error <= 2e-5, error
    return {"max_abs_error": error}


def _boxes(center: torch.Tensor, heading: torch.Tensor, half_length: TensorLike, half_width: TensorLike) -> torch.Tensor:
    direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    lateral = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
    half_length_tensor = torch.as_tensor(half_length, device=center.device, dtype=center.dtype)
    half_width_tensor = torch.as_tensor(half_width, device=center.device, dtype=center.dtype)
    while half_length_tensor.ndim < heading.ndim:
        half_length_tensor = half_length_tensor.unsqueeze(-1)
        half_width_tensor = half_width_tensor.unsqueeze(-1)
    half_length_tensor = torch.broadcast_to(half_length_tensor, heading.shape)
    half_width_tensor = torch.broadcast_to(half_width_tensor, heading.shape)
    longitudinal_sign = center.new_tensor((1.0, -1.0, -1.0, 1.0))
    lateral_sign = center.new_tensor((1.0, 1.0, -1.0, -1.0))
    return (
        center[..., None, :]
        + half_length_tensor[..., None, None]
        * longitudinal_sign[:, None]
        * direction[..., None, :]
        + half_width_tensor[..., None, None]
        * lateral_sign[:, None]
        * lateral[..., None, :]
    )


TensorLike = Union[float, torch.Tensor]


def check_box_collision(device: torch.device) -> dict[str, int]:
    generator = torch.Generator(device=device).manual_seed(13)
    batch_size, proposals, steps, future, agents = 2, 11, 7, 4, 19
    ego_center = torch.randn(
        (batch_size, proposals, steps, future, 2), generator=generator, device=device
    ) * 6.0
    ego_heading = torch.randn(
        (batch_size, proposals, steps, future), generator=generator, device=device
    ) * pi
    agent_center = torch.randn(
        (batch_size, steps, future, agents, 2), generator=generator, device=device
    ) * 6.0
    agent_heading = torch.randn(
        (batch_size, steps, future, agents), generator=generator, device=device
    ) * pi
    ego = _boxes(ego_center, ego_heading, 2.4, 1.0)
    agent = _boxes(agent_center, agent_heading, 1.5, 0.8)
    valid = torch.rand(
        (batch_size, steps, future, agents), generator=generator, device=device
    ) > 0.2
    reference = oriented_boxes_intersect(ego.unsqueeze(4), agent[:, None])
    reference &= valid[:, None]
    candidate = triton_pairwise_box_intersections(ego, agent, valid)
    random_mismatches = int((reference != candidate).sum().cpu())
    assert random_mismatches == 0

    centers_a = torch.zeros((1, 1, 1, 1, 2), device=device)
    headings_a = torch.zeros(centers_a.shape[:-1], device=device)
    edge_cases_a = _boxes(centers_a, headings_a, 1.0, 1.0)
    centers_b = torch.tensor(
        [[[[[0.0, 0.0], [2.0, 0.0], [2.0001, 0.0], [0.0, 0.0]]]]],
        device=device,
    )
    headings_b = torch.tensor([[[[0.0, 0.0, 0.0, pi / 4]]]], device=device)
    edge_cases_b = _boxes(centers_b, headings_b, 1.0, 1.0)
    edge_valid = torch.ones((1, 1, 1, 4), device=device, dtype=torch.bool)
    edge_reference = oriented_boxes_intersect(
        edge_cases_a.unsqueeze(4), edge_cases_b[:, None]
    ) & edge_valid[:, None]
    edge_candidate = triton_pairwise_box_intersections(
        edge_cases_a, edge_cases_b, edge_valid
    )
    edge_mismatches = int((edge_reference != edge_candidate).sum().cpu())
    assert edge_mismatches == 0
    assert edge_candidate.flatten().tolist() == [True, True, False, True]

    for _ in range(5):
        repeated = triton_pairwise_box_intersections(ego, agent, valid)
        assert torch.equal(repeated, candidate)
    return {
        "random_pairs": candidate.numel(),
        "random_mismatches": random_mismatches,
        "edge_case_mismatches": edge_mismatches,
    }


def _pack_polygons(
    batches: list[list[Polygon]], device: torch.device
) -> tuple[torch.Tensor, ...]:
    edge_batches: list[list[tuple[tuple[float, float], tuple[float, float], int]]] = []
    for polygons in batches:
        edges = []
        for polygon_idx, polygon in enumerate(polygons):
            for ring in (polygon.exterior, *polygon.interiors):
                vertices = list(ring.coords)[:-1]
                for idx, start in enumerate(vertices):
                    edges.append((start, vertices[(idx + 1) % len(vertices)], polygon_idx))
        edge_batches.append(edges)
    max_edges = max(map(len, edge_batches))
    max_polygons = max(map(len, batches))
    edge_start = torch.zeros((len(batches), max_edges, 2), device=device)
    edge_end = torch.zeros_like(edge_start)
    edge_polygon = torch.zeros((len(batches), max_edges), device=device, dtype=torch.long)
    edge_valid = torch.zeros((len(batches), max_edges), device=device, dtype=torch.bool)
    polygon_min = torch.full((len(batches), max_polygons, 2), torch.inf, device=device)
    polygon_max = torch.full((len(batches), max_polygons, 2), -torch.inf, device=device)
    for batch_idx, (polygons, edges) in enumerate(zip(batches, edge_batches)):
        for edge_idx, (start, end, polygon_idx) in enumerate(edges):
            edge_start[batch_idx, edge_idx] = torch.tensor(start, device=device)
            edge_end[batch_idx, edge_idx] = torch.tensor(end, device=device)
            edge_polygon[batch_idx, edge_idx] = polygon_idx
            edge_valid[batch_idx, edge_idx] = True
        for polygon_idx, polygon in enumerate(polygons):
            min_x, min_y, max_x, max_y = polygon.bounds
            polygon_min[batch_idx, polygon_idx] = torch.tensor((min_x, min_y), device=device)
            polygon_max[batch_idx, polygon_idx] = torch.tensor((max_x, max_y), device=device)
    return edge_start, edge_end, edge_polygon, edge_valid, polygon_min, polygon_max


def check_polygon_query(device: torch.device) -> dict[str, int]:
    batches = [
        [
            Polygon(((0, 0), (10, 0), (10, 10), (0, 10))),
            Polygon(
                ((20, 0), (30, 0), (30, 10), (20, 10)),
                holes=[((23, 3), (27, 3), (27, 7), (23, 7))],
            ),
            Polygon(((0, 20), (8, 20), (4, 28))),
        ],
        [
            Polygon(((-5, -5), (5, -5), (5, 5), (-5, 5))),
            Polygon(((10, 10), (18, 10), (18, 14), (10, 14))),
        ],
    ]
    packed = _pack_polygons(batches, device)
    generator = torch.Generator(device=device).manual_seed(23)
    points = torch.rand((2, 2048, 2), generator=generator, device=device) * 40.0 - 8.0
    edge_points = torch.tensor(
        ((0, 0), (5, 0), (10, 10), (25, 3), (23, 5), (25, 5), (20, 5), (4, 20)),
        device=device,
        dtype=torch.float32,
    )
    points[:, : len(edge_points)] = edge_points
    num_polygons = max(map(len, batches))
    torch_inside = points_in_padded_polygon_edges(
        points, *packed, num_polygons, boundary_inside=False
    )
    cuda_inside = triton_points_in_polygon_edges(
        points, *packed, num_polygons, boundary_inside=False
    )
    torch_covers = points_in_padded_polygon_edges(
        points, *packed, num_polygons, boundary_inside=True
    )
    cuda_covers = triton_points_in_polygon_edges(
        points, *packed, num_polygons, boundary_inside=True
    )
    assert torch.equal(torch_inside, cuda_inside)
    assert torch.equal(torch_covers, cuda_covers)

    shapely_inside = torch.zeros_like(torch_inside)
    shapely_covers = torch.zeros_like(torch_covers)
    points_cpu = points.cpu().numpy()
    for batch_idx, polygons in enumerate(batches):
        for point_idx, xy in enumerate(points_cpu[batch_idx]):
            point = Point(float(xy[0]), float(xy[1]))
            for polygon_idx, polygon in enumerate(polygons):
                shapely_inside[batch_idx, point_idx, polygon_idx] = polygon.contains(point)
                shapely_covers[batch_idx, point_idx, polygon_idx] = polygon.covers(point)
    inside_mismatches = int((torch_inside != shapely_inside).sum().cpu())
    covers_mismatches = int((torch_covers != shapely_covers).sum().cpu())
    assert inside_mismatches == 0
    assert covers_mismatches == 0
    for _ in range(5):
        repeated = triton_points_in_polygon_edges(
            points, *packed, num_polygons, boundary_inside=False
        )
        assert torch.equal(repeated, cuda_inside)
    return {
        "queries": torch_inside.numel(),
        "inside_mismatches": inside_mismatches,
        "covers_mismatches": covers_mismatches,
    }


def check_polyline_projection(device: torch.device) -> dict[str, float]:
    polylines = [
        [(0.0, 0.0), (4.0, 0.0), (7.0, 2.0), (8.0, 8.0), (12.0, 9.0)],
        [(-3.0, -1.0), (2.0, -1.0), (2.0, 5.0)],
    ]
    max_points = max(map(len, polylines))
    line = torch.zeros((2, max_points, 2), device=device)
    mask = torch.zeros((2, max_points), device=device, dtype=torch.bool)
    for batch_idx, values in enumerate(polylines):
        line[batch_idx, : len(values)] = torch.tensor(values, device=device)
        mask[batch_idx, : len(values)] = True
    generator = torch.Generator(device=device).manual_seed(101)
    points = torch.randn((2, 1024, 2), generator=generator, device=device) * 9.0
    reference_progress, reference_distance = project_points_to_polyline(points, line, mask)
    candidate_progress, candidate_distance = triton_project_points_to_polyline(points, line, mask)
    progress_error = _max_error(reference_progress, candidate_progress)
    distance_error = _max_error(reference_distance, candidate_distance)
    assert progress_error <= 4e-6
    assert distance_error <= 4e-6

    shapely_progress = np.zeros((2, points.shape[1]), dtype=np.float32)
    shapely_distance = np.zeros_like(shapely_progress)
    points_cpu = points.cpu().numpy()
    for batch_idx, values in enumerate(polylines):
        polyline = LineString(values)
        for point_idx, xy in enumerate(points_cpu[batch_idx]):
            point = Point(float(xy[0]), float(xy[1]))
            shapely_progress[batch_idx, point_idx] = polyline.project(point)
            shapely_distance[batch_idx, point_idx] = polyline.distance(point)
    shapely_progress_error = float(
        np.max(np.abs(reference_progress.cpu().numpy() - shapely_progress))
    )
    shapely_distance_error = float(
        np.max(np.abs(reference_distance.cpu().numpy() - shapely_distance))
    )
    assert shapely_progress_error <= 4e-6
    assert shapely_distance_error <= 4e-6

    # Force the chunked path used by centerlines longer than one Triton block.
    long_length = 6003
    parameter = torch.linspace(0.0, 300.0, long_length, device=device)
    long_line = torch.zeros((2, long_length, 2), device=device)
    long_line[0, :, 0] = parameter
    long_line[0, :, 1] = 4.0 * torch.sin(parameter * 0.03)
    second_length = 4789
    long_line[1, :second_length, 0] = parameter[:second_length]
    long_line[1, :second_length, 1] = 3.0 * torch.cos(parameter[:second_length] * 0.04)
    long_mask = torch.zeros((2, long_length), device=device, dtype=torch.bool)
    long_mask[0] = True
    long_mask[1, :second_length] = True
    long_points = torch.rand((2, 256, 2), generator=generator, device=device)
    long_points[..., 0] *= 300.0
    long_points[..., 1] = long_points[..., 1] * 20.0 - 10.0
    long_reference = project_points_to_polyline(long_points, long_line, long_mask)
    long_candidate = triton_project_points_to_polyline(long_points, long_line, long_mask)
    long_progress_error = _max_error(long_reference[0], long_candidate[0])
    long_distance_error = _max_error(long_reference[1], long_candidate[1])
    assert long_progress_error <= 2e-4
    assert long_distance_error <= 1e-5
    return {
        "torch_cuda_progress_max_abs_error": progress_error,
        "torch_cuda_distance_max_abs_error": distance_error,
        "torch_shapely_progress_max_abs_error": shapely_progress_error,
        "torch_shapely_distance_max_abs_error": shapely_distance_error,
        "long_torch_cuda_progress_max_abs_error": long_progress_error,
        "long_torch_cuda_distance_max_abs_error": long_distance_error,
    }


def check_aggregation(device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(37)
    shape = (5, 64)
    no_collision = torch.randint(0, 3, shape, generator=generator, device=device).float() * 0.5
    drivable = torch.randint(0, 2, shape, generator=generator, device=device).float()
    progress = torch.rand(shape, generator=generator, device=device) * 20.0
    ttc = torch.randint(0, 2, shape, generator=generator, device=device).float()
    comfort = torch.randint(0, 2, shape, generator=generator, device=device).float()
    direction = torch.randint(0, 3, shape, generator=generator, device=device).float() * 0.5
    pdm_progress = torch.rand((shape[0],), generator=generator, device=device) * 15.0
    progress[0, :3] = torch.tensor((0.0, 5.0, 5.000001), device=device)
    reference = aggregate_v1_scores(
        no_collision, drivable, progress, ttc, comfort, direction, pdm_progress
    )
    candidate = triton_aggregate_v1_scores(
        no_collision, drivable, progress, ttc, comfort, direction, pdm_progress
    )
    error = _max_error(reference, candidate)
    assert error <= 2e-7
    return {"max_abs_error": error}


def _ego_state() -> EgoState:
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(12.0, -7.0, pi - 0.02),
        rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.1, 0.0),
        tire_steering_angle=0.02,
        time_point=TimePoint(1_000_000),
        vehicle_parameters=get_pacifica_parameters(),
        angular_vel=0.03,
        angular_accel=0.01,
        tire_steering_rate=0.005,
    )


def check_cpu_simulator_and_comfort(device: torch.device) -> dict[str, float | int]:
    ego_state = _ego_state()
    time_s = np.arange(1, 9, dtype=np.float64) * 0.5
    proposals = np.zeros((32, 8, 3), dtype=np.float64)
    for idx in range(len(proposals)):
        speed = 2.0 + idx * 0.2
        curvature = -0.08 + 0.16 * idx / (len(proposals) - 1)
        distance = speed * time_s
        proposals[idx, :, 0] = distance
        proposals[idx, :, 1] = 0.5 * curvature * distance**2
        proposals[idx, :, 2] = np.arctan(curvature * distance)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    reference = np.stack(
        [
            get_trajectory_as_array(
                transform_trajectory(Trajectory(poses), ego_state),
                sampling,
                ego_state.time_point,
            )
            for poses in proposals
        ]
    )
    initial = ego_state_to_state_array(ego_state)
    cpu_simulator = PDMSimulator(sampling)
    cpu_states = cpu_simulator.simulate_proposals(reference, ego_state)
    torch_simulator = TorchPDMSimulator(
        ego_state.car_footprint.vehicle_parameters.wheel_base
    ).to(device=device, dtype=torch.float64)
    with torch.inference_mode():
        torch_states = torch_simulator(
            torch.from_numpy(reference).to(device),
            torch.from_numpy(initial).to(device),
        )
    simulation_error = float(
        np.max(np.abs(torch_states.cpu().numpy() - cpu_states))
    )
    assert simulation_error <= 1e-7
    cpu_comfort = ego_is_comfortable(
        cpu_states, np.arange(41, dtype=np.float64) * 0.1
    )
    torch_comfort = (
        _comfort_metrics(torch_states[None], 0.1)[0].cpu().numpy().astype(bool)
    )
    comfort_mismatches = int(np.count_nonzero(cpu_comfort != torch_comfort))
    assert comfort_mismatches == 0
    return {
        "simulation_max_abs_error": simulation_error,
        "comfort_mismatch_count": comfort_mismatches,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    results = {
        "interpolation": check_interpolation(device),
        "state_coords": check_state_coords(device),
        "box_collision": check_box_collision(device),
        "polygon_query": check_polygon_query(device),
        "polyline_projection": check_polyline_projection(device),
        "aggregation": check_aggregation(device),
        "cpu_simulator_comfort": check_cpu_simulator_and_comfort(device),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
