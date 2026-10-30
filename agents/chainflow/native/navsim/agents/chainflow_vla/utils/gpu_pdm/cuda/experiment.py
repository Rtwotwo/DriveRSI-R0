"""Real-cache parity and timing experiment for the native CUDA PDM backend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    VehicleGeometry,
    proposals_to_state_array,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import generate_proposals
from navsim.agents.chainflow_vla.utils.gpu_pdm.scene import (
    load_metric_cache,
    prepare_pdm_scene,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scorer import score_v1_gpu
from navsim.agents.chainflow_vla.utils.gpu_pdm.simulator import TorchPDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)

from .triton_backend import run_v1_triton_pipeline, score_v1_triton


TARGET_NAMES = (
    "no_collision",
    "drivable_area",
    "progress",
    "ttc",
    "comfort",
    "driving_direction",
    "score",
)


def _timed_cuda(
    function: Callable[[], object],
    device: torch.device,
    warm_repeats: int,
    repeats: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warm_repeats):
            function()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            function()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / repeats


def _progress(cache: object, num_proposals: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(cache.pdm_progress, device=device, dtype=torch.float32)
    if value.numel() == 1:
        return value.reshape(1)
    if value.numel() == num_proposals:
        return value.reshape(1, num_proposals)
    raise ValueError(f"unexpected pdm_progress shape: {tuple(value.shape)}")


def _column_errors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    difference = (candidate - reference).abs()
    return {
        name: {
            "max_abs_error": float(difference[..., idx].amax().cpu()),
            "mismatch_count_1e-5": int((difference[..., idx] > 1e-5).sum().cpu()),
        }
        for idx, name in enumerate(TARGET_NAMES)
    }


def compare_cache(
    cache_path: Path,
    device: torch.device,
    num_proposals: int,
    warm_repeats: int,
    repeats: int,
) -> dict[str, object]:
    cache = load_metric_cache(cache_path)
    scene = prepare_pdm_scene(
        cache, device=device, dtype=torch.float32, local_frame=True
    )
    initial = torch.as_tensor(
        ego_state_to_state_array(cache.ego_state), device=device, dtype=torch.float64
    )
    initial[:2] -= scene.origin[0]
    initial = initial.float().unsqueeze(0)
    proposals = torch.as_tensor(
        generate_proposals(cache, num_proposals), device=device, dtype=torch.float32
    ).unsqueeze(0)
    vehicle = cache.ego_state.car_footprint.vehicle_parameters
    geometry = VehicleGeometry(
        vehicle.half_length,
        vehicle.half_width,
        vehicle.rear_axle_to_center,
        vehicle.wheel_base,
    )
    pdm_progress = _progress(cache, num_proposals, device)
    simulator = TorchPDMSimulator(vehicle.wheel_base).to(device=device, dtype=torch.float32)

    def torch_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = proposals_to_state_array(proposals, initial)
        simulated = simulator(reference, initial)
        targets, areas = score_v1_gpu(simulated, scene, geometry, pdm_progress)
        return reference, simulated, targets, areas

    def cuda_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return run_v1_triton_pipeline(
            proposals,
            initial,
            scene,
            geometry,
            pdm_progress,
            simulator,
        )

    with torch.inference_mode():
        torch_result = torch_pipeline()
        cuda_result = cuda_pipeline()  # includes one-time Triton compilation
        cuda_score_same_states = score_v1_triton(
            torch_result[1], scene, geometry, pdm_progress
        )
    torch.cuda.synchronize(device)

    torch_seconds = _timed_cuda(torch_pipeline, device, warm_repeats, repeats)
    cuda_seconds = _timed_cuda(cuda_pipeline, device, warm_repeats, repeats)
    reference_error = float((cuda_result[0] - torch_result[0]).abs().amax().cpu())
    simulation_error = float((cuda_result[1] - torch_result[1]).abs().amax().cpu())
    scorer_area_mismatch = int(
        (cuda_score_same_states[1] != torch_result[3]).sum().cpu()
    )
    pipeline_area_mismatch = int((cuda_result[3] != torch_result[3]).sum().cpu())
    return {
        "cache": str(cache_path),
        "scene": {
            "agents": int(scene.agent_valid.any(dim=1).sum().cpu()),
            "polygons": int(scene.lane_polygon_mask.shape[1]),
            "edges": int(scene.map_edge_valid.sum().cpu()),
            "centerline_points": int(scene.centerline_point_mask.sum().cpu()),
        },
        "shape": {
            "proposals": list(proposals.shape),
            "states": list(cuda_result[1].shape),
            "targets": list(cuda_result[2].shape),
        },
        "parity": {
            "reference_max_abs_error": reference_error,
            "simulation_max_abs_error": simulation_error,
            "scorer_same_states": {
                "ego_area_mismatch_count": scorer_area_mismatch,
                "columns": _column_errors(torch_result[2], cuda_score_same_states[0]),
            },
            "end_to_end": {
                "ego_area_mismatch_count": pipeline_area_mismatch,
                "columns": _column_errors(torch_result[2], cuda_result[2]),
            },
        },
        "timing_seconds": {
            "torch_gpu": torch_seconds,
            "native_cuda": cuda_seconds,
            "speedup": torch_seconds / cuda_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the native CUDA experiment requires a CUDA device")
    reports = [
        compare_cache(
            path,
            device,
            args.num_proposals,
            args.warm_repeats,
            args.repeats,
        )
        for path in args.cache
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
