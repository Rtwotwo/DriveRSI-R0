"""Real metric-cache parity runner for the isolated GPU PDM backend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.chainflow_vla.utils.pdm_score import PDMScorer, PDMScorerConfig
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)

from .ops import (
    VehicleGeometry,
    points_in_padded_polygon_edges,
    proposals_to_state_array,
    state_array_to_coords,
)
from .scene import load_metric_cache, prepare_pdm_scene
from .scorer import score_v1_gpu
from .simulator import TorchPDMSimulator


TARGET_NAMES = (
    "no_collision",
    "drivable_area",
    "progress",
    "ttc",
    "comfort",
    "driving_direction",
    "score",
)


def _summarize_values(values: np.ndarray) -> Any:
    unique = np.unique(values)
    if len(unique) <= 8:
        return unique.tolist()
    return {"count": len(unique), "min": float(unique[0]), "max": float(unique[-1])}


def _column_report(cpu_targets: np.ndarray, gpu_targets: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for column_idx, name in enumerate(TARGET_NAMES):
        difference = np.abs(gpu_targets[:, column_idx] - cpu_targets[:, column_idx])
        report[name] = {
            "mismatch_count": int(np.count_nonzero(difference > 1e-6)),
            "max_abs_error": float(difference.max(initial=0.0)),
            "cpu_values": _summarize_values(cpu_targets[:, column_idx]),
            "gpu_values": _summarize_values(gpu_targets[:, column_idx]),
        }
    return report


def generate_proposals(metric_cache: Any, num_proposals: int = 64) -> np.ndarray:
    """Generate a deterministic mix of speeds and curvatures in ego frame."""

    initial_state = ego_state_to_state_array(metric_cache.ego_state)
    initial_speed = max(float(np.hypot(initial_state[3], initial_state[4])), 2.0)
    time_s = np.arange(1, 9, dtype=np.float64) * 0.5
    proposals = np.zeros((num_proposals, len(time_s), 3), dtype=np.float64)
    for proposal_idx in range(num_proposals):
        speed_scale = 0.4 + 1.2 * ((proposal_idx % 8) / 7.0)
        acceleration = -1.5 + 3.0 * (((proposal_idx // 8) % 4) / 3.0)
        curvature = -0.10 + 0.20 * ((proposal_idx // 32) % 2)
        distance = np.maximum(
            initial_speed * speed_scale * time_s + 0.5 * acceleration * time_s**2,
            0.0,
        )
        proposals[proposal_idx, :, 0] = distance
        proposals[proposal_idx, :, 1] = 0.5 * curvature * distance**2
        proposals[proposal_idx, :, 2] = np.arctan(curvature * distance)
    return proposals


def _cpu_pipeline(
    metric_cache: Any, proposals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the current official CPU transform, simulator, and scorer path."""

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling, PDMScorerConfig())
    initial_ego_state = metric_cache.ego_state
    reference_states = np.stack(
        [
            get_trajectory_as_array(
                transform_trajectory(Trajectory(proposal), initial_ego_state),
                sampling,
                initial_ego_state.time_point,
            )
            for proposal in proposals
        ]
    )
    simulated_states = simulator.simulate_proposals(reference_states, initial_ego_state)
    final_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress,
    )
    targets = np.stack(
        (
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION],
            scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA],
            scorer._weighted_metrics[WeightedMetricIndex.PROGRESS],
            scorer._weighted_metrics[WeightedMetricIndex.TTC],
            scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE],
            scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION],
            final_scores,
        ),
        axis=-1,
    )
    return targets, reference_states, simulated_states, scorer._ego_areas.copy()


def _cpu_score_states(
    metric_cache: Any, simulated_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run only the official CPU scorer on caller-supplied global states."""

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    scorer = PDMScorer(sampling, PDMScorerConfig())
    final_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress,
    )
    targets = np.stack(
        (
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION],
            scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA],
            scorer._weighted_metrics[WeightedMetricIndex.PROGRESS],
            scorer._weighted_metrics[WeightedMetricIndex.TTC],
            scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE],
            scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION],
            final_scores,
        ),
        axis=-1,
    )
    return targets, scorer._ego_areas.copy()


def _cpu_targets(
    metric_cache: Any, proposals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper used by the existing compact parity runner."""

    targets, reference_states, _, ego_areas = _cpu_pipeline(metric_cache, proposals)
    return targets, reference_states, ego_areas


def _pdm_progress_tensor(
    metric_cache: Any,
    num_proposals: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    progress = torch.as_tensor(metric_cache.pdm_progress, device=device, dtype=dtype)
    if progress.numel() == 1:
        return progress.reshape(1)
    if progress.numel() == num_proposals:
        return progress.reshape(1, num_proposals)
    raise ValueError(f"unexpected pdm_progress shape: {tuple(progress.shape)}")


def compare_cache(
    cache_path: str | Path,
    device: torch.device,
    num_proposals: int = 64,
    warm_repeats: int = 3,
    test_float32_local: bool = False,
) -> dict[str, Any]:
    """Compare all seven CPU/GPU targets for one real metric cache."""

    metric_cache = load_metric_cache(cache_path)
    proposals = generate_proposals(metric_cache, num_proposals)
    cpu_start = time.perf_counter()
    cpu_targets, reference_states, cpu_ego_area = _cpu_targets(metric_cache, proposals)
    cpu_seconds = time.perf_counter() - cpu_start

    initial_array = ego_state_to_state_array(metric_cache.ego_state)
    vehicle = metric_cache.ego_state.car_footprint.vehicle_parameters
    geometry = VehicleGeometry(
        vehicle.half_length,
        vehicle.half_width,
        vehicle.rear_axle_to_center,
        vehicle.wheel_base,
    )
    local_proposals = torch.from_numpy(proposals).to(device)
    initial_states = torch.from_numpy(initial_array).to(device)
    gpu_reference = proposals_to_state_array(local_proposals[None], initial_states[None])[0]
    cpu_reference = torch.from_numpy(reference_states).to(device)
    trajectory_max_abs_error = float((gpu_reference - cpu_reference).abs().amax().cpu())

    scene = prepare_pdm_scene(
        metric_cache, device=device, dtype=torch.float64, local_frame=False
    )
    simulator = TorchPDMSimulator(vehicle.wheel_base).to(device=device, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_start = time.perf_counter()

    def run_gpu() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        simulated = simulator(cpu_reference, initial_states)
        targets, ego_area = score_v1_gpu(
            simulated[None],
            scene,
            geometry,
            _pdm_progress_tensor(metric_cache, num_proposals, device),
        )
        return targets, ego_area, simulated

    with torch.inference_mode():
        gpu_targets, gpu_ego_area, simulated_states = run_gpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - gpu_start
    warm_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(warm_repeats):
            gpu_targets, gpu_ego_area, simulated_states = run_gpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_warm_seconds = (time.perf_counter() - warm_start) / max(warm_repeats, 1)
    gpu_targets_array = gpu_targets[0].cpu().numpy()

    columns = _column_report(cpu_targets, gpu_targets_array)
    gpu_ego_area_array = gpu_ego_area[0].cpu().numpy()
    ego_area_names = ("multiple_lanes", "non_drivable", "oncoming")
    ego_area = {
        name: {
            "mismatch_count": int(
                np.count_nonzero(cpu_ego_area[..., idx] != gpu_ego_area_array[..., idx])
            ),
            "cpu_true": int(np.count_nonzero(cpu_ego_area[..., idx])),
            "gpu_true": int(np.count_nonzero(gpu_ego_area_array[..., idx])),
        }
        for idx, name in enumerate(ego_area_names)
    }
    valid_map_points = scene.map_edge_start[scene.map_edge_valid]
    sample_coords = state_array_to_coords(
        simulated_states[:1, :1],
        geometry.half_length,
        geometry.half_width,
        geometry.rear_axle_to_center,
    )[0, 0]
    gpu_membership = points_in_padded_polygon_edges(
        sample_coords[None],
        scene.map_edge_start,
        scene.map_edge_end,
        scene.map_edge_polygon,
        scene.map_edge_valid,
        scene.map_polygon_min,
        scene.map_polygon_max,
        scene.lane_polygon_mask.shape[1],
    )[0]
    cpu_membership = metric_cache.drivable_area_map.points_in_polygons(
        sample_coords.cpu().numpy()
    )
    cpu_hit_indices = np.flatnonzero(cpu_membership[:, 0])
    scene_stats = {
        "agents": int(scene.agent_valid.any(dim=1).sum().cpu()),
        "polygons": int(scene.lane_polygon_mask.shape[1]),
        "edges": int(scene.map_edge_valid.sum().cpu()),
        "lane_polygons": int(scene.lane_polygon_mask.sum().cpu()),
        "drivable_polygons": int(scene.drivable_polygon_mask.sum().cpu()),
        "route_lane_polygons": int(scene.route_lane_polygon_mask.sum().cpu()),
        "map_xy_min": valid_map_points.amin(dim=0).cpu().tolist(),
        "map_xy_max": valid_map_points.amax(dim=0).cpu().tolist(),
        "state_xy_min": simulated_states[..., :2].amin(dim=(0, 1)).cpu().tolist(),
        "state_xy_max": simulated_states[..., :2].amax(dim=(0, 1)).cpu().tolist(),
        "cpu_polygon_hits_per_ego_point": cpu_membership.sum(axis=0).tolist(),
        "gpu_polygon_hits_per_ego_point": gpu_membership.sum(dim=-1).cpu().tolist(),
        "cpu_hit_polygon_indices": cpu_hit_indices.tolist(),
        "gpu_hit_polygon_indices": torch.nonzero(gpu_membership[0]).flatten().cpu().tolist(),
        "sample_ego_point": sample_coords[0].cpu().tolist(),
    }
    report = {
        "cache": str(cache_path),
        "num_proposals": num_proposals,
        "trajectory_max_abs_error": trajectory_max_abs_error,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "gpu_warm_seconds": gpu_warm_seconds,
        "scene": scene_stats,
        "ego_area_mismatch_count": ego_area,
        "columns": columns,
    }
    if test_float32_local:
        local_scene = prepare_pdm_scene(
            metric_cache, device=device, dtype=torch.float32, local_frame=True
        )
        local_initial = torch.from_numpy(initial_array).to(device=device, dtype=torch.float64)
        local_initial[:2] -= local_scene.origin[0]
        local_initial = local_initial.float()
        local_reference = proposals_to_state_array(local_proposals.float()[None], local_initial[None])[0]
        local_simulator = TorchPDMSimulator(vehicle.wheel_base).to(device=device, dtype=torch.float32)

        def run_float32_local() -> torch.Tensor:
            local_simulated = local_simulator(local_reference, local_initial)
            targets, _ = score_v1_gpu(
                local_simulated[None],
                local_scene,
                geometry,
                _pdm_progress_tensor(
                    metric_cache, num_proposals, device, torch.float32
                ),
            )
            return targets

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        local_start = time.perf_counter()
        with torch.inference_mode():
            local_targets = run_float32_local()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        local_first_seconds = time.perf_counter() - local_start
        local_warm_start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(warm_repeats):
                local_targets = run_float32_local()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        local_warm_seconds = (time.perf_counter() - local_warm_start) / max(warm_repeats, 1)
        local_targets_array = local_targets[0].cpu().numpy()
        cpu_reference_local = torch.from_numpy(reference_states).to(device=device, dtype=torch.float64)
        cpu_reference_local[..., :2] -= local_scene.origin[0]
        report["float32_local"] = {
            "trajectory_max_abs_error": float(
                (local_reference - cpu_reference_local.float()).abs().amax().cpu()
            ),
            "gpu_seconds": local_first_seconds,
            "gpu_warm_seconds": local_warm_seconds,
            "columns": _column_report(cpu_targets, local_targets_array),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--float32-local", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    reports = [
        compare_cache(
            path,
            device,
            args.num_proposals,
            args.warm_repeats,
            args.float32_local,
        )
        for path in args.cache
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
