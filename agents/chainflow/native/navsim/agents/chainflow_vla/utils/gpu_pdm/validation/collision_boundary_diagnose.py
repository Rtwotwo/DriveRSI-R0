"""Inspect a CPU/Torch/Triton collision-boundary disagreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.cuda.triton_backend import (
    run_v1_triton_pipeline,
    triton_pairwise_box_intersections,
    triton_state_array_to_coords,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    VehicleGeometry,
    oriented_boxes_intersect,
    proposals_to_state_array,
    state_array_to_coords,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import (
    _cpu_pipeline,
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


def _sat_margin(box_a: torch.Tensor, box_b: torch.Tensor) -> dict[str, float]:
    """Return positive separation or negative overlap along the worst SAT axis."""

    edges_a = torch.stack((box_a[1] - box_a[0], box_a[3] - box_a[0]))
    edges_b = torch.stack((box_b[1] - box_b[0], box_b[3] - box_b[0]))
    edges = torch.cat((edges_a, edges_b))
    axes = torch.stack((-edges[:, 1], edges[:, 0]), dim=-1)
    projection_a = box_a @ axes.T
    projection_b = box_b @ axes.T
    signed = torch.maximum(
        projection_b.amin(dim=0) - projection_a.amax(dim=0),
        projection_a.amin(dim=0) - projection_b.amax(dim=0),
    )
    normalized = signed / torch.linalg.vector_norm(axes, dim=-1).clamp_min(1e-12)
    worst = int(normalized.argmax())
    return {
        "signed_margin_m": float(normalized[worst]),
        "axis": worst,
        "raw_projection_margin": float(signed[worst]),
        "axis_norm": float(torch.linalg.vector_norm(axes[worst])),
    }


def diagnose(
    cache_path: Path,
    proposal_index: int,
    device: torch.device,
    num_proposals: int,
) -> dict[str, object]:
    cache = load_metric_cache(cache_path)
    proposals_np = generate_proposals(cache, num_proposals)
    cpu_targets, _, cpu_simulated, _ = _cpu_pipeline(cache, proposals_np)
    scene = prepare_pdm_scene(
        cache, device=device, dtype=torch.float32, local_frame=True
    )
    initial = torch.as_tensor(
        ego_state_to_state_array(cache.ego_state), device=device, dtype=torch.float64
    )
    initial[:2] -= scene.origin[0]
    initial = initial.float().unsqueeze(0)
    proposals = torch.as_tensor(
        proposals_np, device=device, dtype=torch.float32
    ).unsqueeze(0)
    progress = torch.as_tensor(cache.pdm_progress, device=device, dtype=torch.float32)
    if progress.numel() == 1:
        progress = progress.reshape(1)
    elif progress.numel() == num_proposals:
        progress = progress.reshape(1, num_proposals)
    else:
        raise ValueError(f"unexpected pdm_progress shape: {tuple(progress.shape)}")
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
        torch_targets, _ = score_v1_gpu(
            torch_simulated, scene, geometry, progress
        )
        triton_reference, triton_simulated, triton_targets, _ = (
            run_v1_triton_pipeline(
                proposals, initial, scene, geometry, progress, simulator
            )
        )
        torch_coords = state_array_to_coords(
            torch_simulated,
            geometry.half_length,
            geometry.half_width,
            geometry.rear_axle_to_center,
        )
        triton_coords = triton_state_array_to_coords(triton_simulated, geometry)
        agents = scene.agent_corners[:, : torch_simulated.shape[2]]
        valid = scene.agent_valid[:, : torch_simulated.shape[2]]
        torch_intersections = oriented_boxes_intersect(
            torch_coords[..., None, :4, :], agents[:, None]
        )
        triton_intersections = triton_pairwise_box_intersections(
            triton_coords[..., None, :4, :],
            agents[:, :, None],
            valid[:, :, None],
        ).squeeze(3)
    with torch.inference_mode():
        torch_intersections &= valid[:, None]
        torch_intersections &= ~scene.agent_initially_collided[:, None, None, :]
        triton_intersections &= ~scene.agent_initially_collided[:, None, None, :]
    difference = torch_intersections[0, proposal_index] != triton_intersections[
        0, proposal_index
    ]
    differing_pairs = torch.nonzero(difference).cpu().tolist()

    origin = scene.origin[0].cpu().numpy()
    cpu_simulated_local = cpu_simulated.copy()
    cpu_simulated_local[..., :2] -= origin
    cpu_states = torch.as_tensor(cpu_simulated_local, dtype=torch.float64)
    cpu_coords = state_array_to_coords(
        cpu_states,
        geometry.half_length,
        geometry.half_width,
        geometry.rear_axle_to_center,
    )
    scene64 = prepare_pdm_scene(
        cache, device="cpu", dtype=torch.float64, local_frame=True
    )

    pair_reports = []
    for time_index, agent_index in differing_pairs:
        boxes = {
            "cpu": (
                cpu_coords[proposal_index, time_index, :4],
                scene64.agent_corners[0, time_index, agent_index],
            ),
            "torch": (
                torch_coords[0, proposal_index, time_index, :4].double().cpu(),
                scene.agent_corners[0, time_index, agent_index].double().cpu(),
            ),
            "triton": (
                triton_coords[0, proposal_index, time_index, :4].double().cpu(),
                scene.agent_corners[0, time_index, agent_index].double().cpu(),
            ),
        }
        margins = {name: _sat_margin(*pair) for name, pair in boxes.items()}
        epsilon_intersections = {
            f"{epsilon:.0e}": bool(
                oriented_boxes_intersect(
                    boxes["torch"][0], boxes["torch"][1], eps=epsilon
                )
            )
            for epsilon in (0.0, 1e-7, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
        }
        pair_reports.append(
            {
                "time_index": int(time_index),
                "agent_index": int(agent_index),
                "agent_valid": bool(valid[0, time_index, agent_index]),
                "margins": margins,
                "torch_epsilon_intersections": epsilon_intersections,
                "ego_center_difference_torch_triton": float(
                    torch.linalg.vector_norm(
                        boxes["torch"][0].mean(dim=0)
                        - boxes["triton"][0].mean(dim=0)
                    )
                ),
            }
        )
    return {
        "cache": str(cache_path),
        "proposal": proposal_index,
        "cpu_targets": cpu_targets[proposal_index].tolist(),
        "torch_targets": torch_targets[0, proposal_index].cpu().tolist(),
        "triton_targets": triton_targets[0, proposal_index].cpu().tolist(),
        "reference_max_abs_error_torch_triton": float(
            (torch_reference - triton_reference).abs().amax().cpu()
        ),
        "simulation_max_abs_error_torch_triton": float(
            (torch_simulated - triton_simulated).abs().amax().cpu()
        ),
        "differing_pairs": pair_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--proposal", type=int, required=True)
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(
        json.dumps(
            diagnose(
                args.cache,
                args.proposal,
                torch.device(args.device),
                args.num_proposals,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
