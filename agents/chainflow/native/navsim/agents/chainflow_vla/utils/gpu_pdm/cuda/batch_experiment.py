"""Multi-scene batch experiment for the native CUDA PDM backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    VehicleGeometry,
    proposals_to_state_array,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import generate_proposals
from navsim.agents.chainflow_vla.utils.gpu_pdm.scene import (
    load_metric_cache,
    prepare_pdm_scenes,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scorer import score_v1_gpu
from navsim.agents.chainflow_vla.utils.gpu_pdm.simulator import TorchPDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)

from .experiment import _column_errors, _timed_cuda
from .triton_backend import run_v1_triton_pipeline, score_v1_triton


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if len(args.cache) < 2:
        raise ValueError("batch_experiment requires at least two caches")
    device = torch.device(args.device)
    caches = [load_metric_cache(path) for path in args.cache]
    scene = prepare_pdm_scenes(
        caches, device=device, dtype=torch.float32, local_frame=True
    )
    initials = []
    proposals = []
    progresses = []
    for idx, cache in enumerate(caches):
        initial = torch.as_tensor(
            ego_state_to_state_array(cache.ego_state),
            device=device,
            dtype=torch.float64,
        )
        initial[:2] -= scene.origin[idx]
        initials.append(initial.float())
        proposals.append(
            torch.as_tensor(
                generate_proposals(cache, args.num_proposals),
                device=device,
                dtype=torch.float32,
            )
        )
        progress = torch.as_tensor(
            cache.pdm_progress, device=device, dtype=torch.float32
        ).reshape(-1)
        if progress.numel() == 1:
            progress = progress.expand(args.num_proposals)
        elif progress.numel() != args.num_proposals:
            raise ValueError(f"unexpected pdm_progress shape: {tuple(progress.shape)}")
        progresses.append(progress)
    initials_tensor = torch.stack(initials)
    proposals_tensor = torch.stack(proposals)
    progresses_tensor = torch.stack(progresses)
    vehicle = caches[0].ego_state.car_footprint.vehicle_parameters
    geometry = VehicleGeometry(
        vehicle.half_length,
        vehicle.half_width,
        vehicle.rear_axle_to_center,
        vehicle.wheel_base,
    )
    simulator = TorchPDMSimulator(vehicle.wheel_base).to(device=device, dtype=torch.float32)

    def torch_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = proposals_to_state_array(proposals_tensor, initials_tensor)
        simulated = simulator(reference, initials_tensor)
        targets, areas = score_v1_gpu(
            simulated, scene, geometry, progresses_tensor
        )
        return reference, simulated, targets, areas

    def cuda_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return run_v1_triton_pipeline(
            proposals_tensor,
            initials_tensor,
            scene,
            geometry,
            progresses_tensor,
            simulator,
        )

    with torch.inference_mode():
        torch_result = torch_pipeline()
        cuda_result = cuda_pipeline()
        cuda_score_same_states = score_v1_triton(
            torch_result[1], scene, geometry, progresses_tensor
        )
    torch.cuda.synchronize(device)
    torch_seconds = _timed_cuda(
        torch_pipeline, device, args.warm_repeats, args.repeats
    )
    cuda_seconds = _timed_cuda(
        cuda_pipeline, device, args.warm_repeats, args.repeats
    )
    report = {
        "caches": [str(path) for path in args.cache],
        "batch_size": len(caches),
        "shape": {
            "proposals": list(proposals_tensor.shape),
            "states": list(cuda_result[1].shape),
            "targets": list(cuda_result[2].shape),
        },
        "parity": {
            "reference_max_abs_error": float(
                (cuda_result[0] - torch_result[0]).abs().amax().cpu()
            ),
            "simulation_max_abs_error": float(
                (cuda_result[1] - torch_result[1]).abs().amax().cpu()
            ),
            "scorer_same_states": {
                "ego_area_mismatch_count": int(
                    (cuda_score_same_states[1] != torch_result[3]).sum().cpu()
                ),
                "columns": _column_errors(
                    torch_result[2], cuda_score_same_states[0]
                ),
            },
            "end_to_end": {
                "ego_area_mismatch_count": int(
                    (cuda_result[3] != torch_result[3]).sum().cpu()
                ),
                "columns": _column_errors(torch_result[2], cuda_result[2]),
            },
        },
        "timing_seconds": {
            "torch_gpu": torch_seconds,
            "native_cuda": cuda_seconds,
            "speedup": torch_seconds / cuda_seconds,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
