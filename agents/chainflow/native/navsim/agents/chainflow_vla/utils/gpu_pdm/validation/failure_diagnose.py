"""Diagnose progress projection and ego-area parity failures on one cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.cuda.triton_backend import (
    run_v1_triton_pipeline,
    score_v1_triton,
    triton_project_points_to_polyline,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    VehicleGeometry,
    project_points_to_polyline,
    proposals_to_state_array,
    state_array_to_coords,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import (
    _cpu_pipeline,
    _cpu_score_states,
    generate_proposals,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scene import (
    load_metric_cache,
    prepare_pdm_scene,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scorer import score_v1_gpu
from navsim.agents.chainflow_vla.utils.gpu_pdm.simulator import TorchPDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)


def _progress_input(cache: Any, count: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(cache.pdm_progress, device=device, dtype=torch.float32)
    if value.numel() == 1:
        return value.reshape(1)
    if value.numel() == count:
        return value.reshape(1, count)
    raise ValueError(f"unexpected pdm_progress shape: {tuple(value.shape)}")


def _projection_report(
    states: torch.Tensor,
    scene: Any,
    geometry: VehicleGeometry,
    proposal: int,
) -> dict[str, Any]:
    coords = state_array_to_coords(
        states,
        geometry.half_length,
        geometry.half_width,
        geometry.rear_axle_to_center,
    )
    endpoints = torch.stack((coords[:, :, 0, 4], coords[:, :, -1, 4]), dim=2)
    flattened = endpoints.reshape(1, -1, 2)
    torch_along, torch_distance = project_points_to_polyline(
        flattened, scene.centerline, scene.centerline_point_mask
    )
    triton_along, triton_distance = triton_project_points_to_polyline(
        flattened, scene.centerline, scene.centerline_point_mask
    )
    float64_along, float64_distance = project_points_to_polyline(
        flattened.double(), scene.centerline.double(), scene.centerline_point_mask
    )
    torch_along = torch_along.reshape(1, -1, 2)
    triton_along = triton_along.reshape(1, -1, 2)
    torch_distance = torch_distance.reshape(1, -1, 2)
    triton_distance = triton_distance.reshape(1, -1, 2)
    float64_along = float64_along.reshape(1, -1, 2)
    float64_distance = float64_distance.reshape(1, -1, 2)

    line = scene.centerline[0]
    mask = scene.centerline_point_mask[0]
    start = line[:-1]
    segment = line[1:] - line[:-1]
    length_sq = (segment * segment).sum(dim=-1)
    length = torch.sqrt(length_sq.clamp_min(torch.finfo(line.dtype).eps))
    valid = mask[:-1] & mask[1:] & (length_sq > 0)
    cumulative = torch.cat(
        (torch.zeros_like(length[:1]), torch.cumsum(length, dim=0)), dim=0
    )[:-1]

    endpoint_reports = []
    for endpoint_index in range(2):
        point = endpoints[0, proposal, endpoint_index]
        fraction = ((point - start) * segment).sum(dim=-1) / length_sq.clamp_min(
            torch.finfo(line.dtype).eps
        )
        fraction = fraction.clamp(0.0, 1.0)
        projected = start + fraction[:, None] * segment
        distance_sq = ((point - projected) ** 2).sum(dim=-1).masked_fill(
            ~valid, torch.inf
        )
        progress = cumulative + fraction * length
        nearest = torch.argsort(distance_sq)[:8]
        endpoint_reports.append(
            {
                "endpoint": endpoint_index,
                "point_xy": point.cpu().tolist(),
                "torch_along": float(torch_along[0, proposal, endpoint_index]),
                "triton_along": float(triton_along[0, proposal, endpoint_index]),
                "float64_along_from_float32_inputs": float(
                    float64_along[0, proposal, endpoint_index]
                ),
                "along_difference": float(
                    triton_along[0, proposal, endpoint_index]
                    - torch_along[0, proposal, endpoint_index]
                ),
                "torch_distance": float(
                    torch_distance[0, proposal, endpoint_index]
                ),
                "triton_distance": float(
                    triton_distance[0, proposal, endpoint_index]
                ),
                "float64_distance_from_float32_inputs": float(
                    float64_distance[0, proposal, endpoint_index]
                ),
                "nearest_segments": [
                    {
                        "segment": int(index),
                        "distance_sq": float(distance_sq[index]),
                        "progress": float(progress[index]),
                        "fraction": float(fraction[index]),
                    }
                    for index in nearest
                ],
            }
        )
    return {
        "valid_centerline_points": int(mask.sum()),
        "padded_centerline_points": int(mask.numel()),
        "torch_progress_raw": float(
            (torch_along[0, proposal, 1] - torch_along[0, proposal, 0]).clamp_min(0)
        ),
        "triton_progress_raw": float(
            (triton_along[0, proposal, 1] - triton_along[0, proposal, 0]).clamp_min(0)
        ),
        "float64_progress_raw_from_float32_inputs": float(
            (
                float64_along[0, proposal, 1]
                - float64_along[0, proposal, 0]
            ).clamp_min(0)
        ),
        "endpoints": endpoint_reports,
    }


def diagnose(
    cache_path: Path,
    proposal: int,
    num_proposals: int,
    device: torch.device,
) -> dict[str, Any]:
    cache = load_metric_cache(cache_path)
    proposals_np = generate_proposals(cache, num_proposals)
    cpu_targets, _, cpu_simulated, cpu_area = _cpu_pipeline(cache, proposals_np)
    scene = prepare_pdm_scene(cache, device=device, dtype=torch.float32, local_frame=True)
    initial = torch.as_tensor(
        ego_state_to_state_array(cache.ego_state), device=device, dtype=torch.float64
    )
    initial[:2] -= scene.origin[0]
    initial = initial.float().unsqueeze(0)
    proposals = torch.as_tensor(
        proposals_np, device=device, dtype=torch.float32
    ).unsqueeze(0)
    pdm_progress = _progress_input(cache, num_proposals, device)
    vehicle = cache.ego_state.car_footprint.vehicle_parameters
    geometry = VehicleGeometry(
        vehicle.half_length,
        vehicle.half_width,
        vehicle.rear_axle_to_center,
        vehicle.wheel_base,
    )
    simulator = TorchPDMSimulator(vehicle.wheel_base).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        torch_reference = proposals_to_state_array(proposals, initial)
        torch_simulated = simulator(torch_reference, initial)
        torch_targets, torch_area = score_v1_gpu(
            torch_simulated, scene, geometry, pdm_progress
        )
        triton_reference, triton_simulated, triton_targets, triton_area = (
            run_v1_triton_pipeline(
                proposals, initial, scene, geometry, pdm_progress, simulator
            )
        )
        same_targets, same_area = score_v1_triton(
            torch_simulated, scene, geometry, pdm_progress
        )

    origin = scene.origin[0].cpu().numpy()
    torch_simulated_global = (
        torch_simulated[0].cpu().numpy().astype(np.float64, copy=True)
    )
    torch_simulated_global[..., :2] += origin
    same_cpu_targets, same_cpu_area = _cpu_score_states(
        cache, torch_simulated_global
    )
    cpu_simulated_local = cpu_simulated.copy()
    cpu_simulated_local[..., :2] -= origin
    cpu_states = torch.as_tensor(
        cpu_simulated_local, device=device, dtype=torch.float32
    ).unsqueeze(0)
    area_difference = np.argwhere(cpu_area != triton_area[0].cpu().numpy())
    area_reports = []
    for proposal_index, time_index, channel in area_difference[:32]:
        area_reports.append(
            {
                "proposal": int(proposal_index),
                "time": int(time_index),
                "channel": int(channel),
                "cpu": bool(cpu_area[proposal_index, time_index, channel]),
                "torch": bool(torch_area[0, proposal_index, time_index, channel]),
                "triton": bool(triton_area[0, proposal_index, time_index, channel]),
                "same_state_triton": bool(
                    same_area[0, proposal_index, time_index, channel]
                ),
                "cpu_state": cpu_states[0, proposal_index, time_index].cpu().tolist(),
                "torch_state": torch_simulated[
                    0, proposal_index, time_index
                ].cpu().tolist(),
                "triton_state": triton_simulated[
                    0, proposal_index, time_index
                ].cpu().tolist(),
                "cpu_triton_xy_max_abs_error": float(
                    (
                        cpu_states[0, proposal_index, time_index, :2]
                        - triton_simulated[0, proposal_index, time_index, :2]
                    )
                    .abs()
                    .amax()
                ),
            }
        )
    return {
        "cache": str(cache_path),
        "proposal": proposal,
        "targets": {
            "cpu": cpu_targets[proposal].tolist(),
            "torch": torch_targets[0, proposal].cpu().tolist(),
            "triton": triton_targets[0, proposal].cpu().tolist(),
            "same_state_triton": same_targets[0, proposal].cpu().tolist(),
            "official_cpu_same_states": same_cpu_targets[proposal].tolist(),
        },
        "projection_on_torch_states": _projection_report(
            torch_simulated, scene, geometry, proposal
        ),
        "projection_on_triton_states": _projection_report(
            triton_simulated, scene, geometry, proposal
        ),
        "projection_on_cpu_states_float32": _projection_report(
            cpu_states, scene, geometry, proposal
        ),
        "state_max_abs_error": {
            "cpu_torch": float((cpu_states - torch_simulated).abs().amax()),
            "cpu_triton": float((cpu_states - triton_simulated).abs().amax()),
            "torch_triton": float(
                (torch_simulated - triton_simulated).abs().amax()
            ),
        },
        "ego_area_difference_count_cpu_triton": int(area_difference.shape[0]),
        "ego_area_differences": area_reports,
        "same_state_ego_area_difference_count": int(
            (torch_area != same_area).sum().cpu()
        ),
        "official_cpu_same_state_ego_area_difference_count": int(
            np.count_nonzero(
                same_cpu_area != same_area[0].cpu().numpy()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--proposal", type=int, required=True)
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = diagnose(
        args.cache,
        args.proposal,
        args.num_proposals,
        torch.device(args.device),
    )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
