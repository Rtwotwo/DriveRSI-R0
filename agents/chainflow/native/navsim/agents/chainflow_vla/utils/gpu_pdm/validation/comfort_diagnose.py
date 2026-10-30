"""Diagnose CPU/Torch comfort differences on one real metric cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    STATE_ACCELERATION_X,
    STATE_ACCELERATION_Y,
    STATE_HEADING,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import (
    _cpu_targets,
    generate_proposals,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scene import load_metric_cache
from navsim.agents.chainflow_vla.utils.gpu_pdm.scorer import _round8, _savgol
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    _extract_ego_acceleration,
    _extract_ego_jerk,
    _extract_ego_yaw_rate,
    ego_is_comfortable,
)


NAMES = (
    "lon_acceleration",
    "lat_acceleration",
    "magnitude_jerk",
    "lon_jerk",
    "yaw_acceleration",
    "yaw_rate",
)
BOUNDS = ((-4.05, 2.40), (-4.89, 4.89), (-8.37, 8.37), (-4.13, 4.13), (-1.93, 1.93), (-0.95, 0.95))


def _torch_raw(states: torch.Tensor, interval_s: float) -> tuple[torch.Tensor, ...]:
    num_steps = states.shape[-2]
    acceleration_x = states[..., STATE_ACCELERATION_X]
    acceleration_y = states[..., STATE_ACCELERATION_Y]
    magnitude = torch.hypot(acceleration_x, acceleration_y)
    headings = states[..., STATE_HEADING]
    difference = headings[..., 1:] - headings[..., :-1]
    adjustment = torch.cumsum(torch.round(difference / (2.0 * np.pi)), dim=-1)
    adjustment = torch.cat((torch.zeros_like(adjustment[..., :1]), adjustment), dim=-1)
    unwrapped = headings - 2.0 * np.pi * adjustment
    return (
        _round8(_savgol(acceleration_x, num_steps, 2)),
        _round8(_savgol(acceleration_y, num_steps, 2)),
        _round8(_savgol(_round8(_savgol(magnitude, 8, 2)), num_steps, 2, 1, interval_s)),
        _round8(_savgol(_round8(_savgol(acceleration_x, 8, 2)), num_steps, 2, 1, interval_s)),
        _round8(_savgol(unwrapped, 5, 3, 2, interval_s)),
        _round8(_savgol(unwrapped, 5, 2, 1, interval_s)),
    )


def _cpu_raw(states: np.ndarray, time_steps: np.ndarray) -> tuple[np.ndarray, ...]:
    num_steps = states.shape[-2]
    return (
        _extract_ego_acceleration(states, "x", window_length=num_steps),
        _extract_ego_acceleration(states, "y", window_length=num_steps),
        _extract_ego_jerk(states, "magnitude", time_steps, window_length=num_steps),
        _extract_ego_jerk(states, "x", time_steps, window_length=num_steps),
        _extract_ego_yaw_rate(states, time_steps, deriv_order=2, poly_order=3, window_length=num_steps),
        _extract_ego_yaw_rate(states, time_steps, window_length=num_steps),
    )


def _within(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.all((values > bounds[0]) & (values < bounds[1]), axis=-1)


def diagnose(cache_path: Path, device: torch.device, num_proposals: int) -> dict[str, object]:
    cache = load_metric_cache(cache_path)
    proposals = generate_proposals(cache, num_proposals)
    _, reference_states, _ = _cpu_targets(cache, proposals)

    # Use the exact official simulated states so this isolates comfort math from
    # interpolation and simulator differences.
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulated = PDMSimulator(sampling).simulate_proposals(reference_states, cache.ego_state)
    time_steps = np.arange(simulated.shape[1], dtype=np.float64) * sampling.interval_length
    cpu_raw = _cpu_raw(simulated, time_steps)
    torch_raw = tuple(
        value.cpu().numpy()
        for value in _torch_raw(torch.from_numpy(simulated).to(device), sampling.interval_length)
    )
    cpu_flags = ego_is_comfortable(simulated, time_steps)
    torch_flags = np.stack(
        [_within(values, bounds) for values, bounds in zip(torch_raw, BOUNDS)], axis=-1
    )
    mismatch_proposals = np.flatnonzero(np.any(cpu_flags != torch_flags, axis=-1))
    details = []
    for proposal in mismatch_proposals.tolist():
        metrics = []
        for name, bounds, cpu_values, torch_values in zip(NAMES, BOUNDS, cpu_raw, torch_raw):
            difference = np.abs(torch_values[proposal] - cpu_values[proposal])
            metrics.append(
                {
                    "name": name,
                    "bounds": list(bounds),
                    "cpu_pass": bool(_within(cpu_values[proposal][None], bounds)[0]),
                    "torch_pass": bool(_within(torch_values[proposal][None], bounds)[0]),
                    "cpu_min": float(np.min(cpu_values[proposal])),
                    "cpu_max": float(np.max(cpu_values[proposal])),
                    "torch_min": float(np.min(torch_values[proposal])),
                    "torch_max": float(np.max(torch_values[proposal])),
                    "max_abs_error": float(np.max(difference)),
                    "max_error_step": int(np.argmax(difference)),
                }
            )
        details.append({"proposal": proposal, "metrics": metrics})
    return {
        "cache": str(cache_path),
        "mismatch_proposals": mismatch_proposals.tolist(),
        "cpu_flags": cpu_flags[mismatch_proposals].astype(int).tolist(),
        "torch_flags": torch_flags[mismatch_proposals].astype(int).tolist(),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-proposals", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.cache, torch.device(args.device), args.num_proposals), indent=2))


if __name__ == "__main__":
    main()
