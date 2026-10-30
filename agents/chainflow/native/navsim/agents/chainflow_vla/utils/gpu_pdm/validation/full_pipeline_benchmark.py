"""Large-scale official-CPU versus Torch/Triton GPU PDM benchmark.

The script calls the current NAVSIM ``PDMSimulator`` and ``PDMScorer`` as the
reference. It is standalone validation code and is never imported by training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.cuda.experiment import _timed_cuda
from navsim.agents.chainflow_vla.utils.gpu_pdm.cuda.triton_backend import (
    run_v1_triton_pipeline,
    score_v1_triton,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.ops import (
    VehicleGeometry,
    proposals_to_state_array,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import (
    TARGET_NAMES,
    _cpu_pipeline,
    _cpu_score_states,
    generate_proposals,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scene import (
    load_metric_cache,
    prepare_pdm_scene,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.scorer import (
    PreparedPDMScene,
    score_v1_gpu,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.simulator import TorchPDMSimulator
from navsim.agents.chainflow_vla.utils.gpu_pdm.validation.real_cache_sweep import (
    select_caches,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)


DISCRETE_TARGETS = {
    "no_collision",
    "drivable_area",
    "ttc",
    "comfort",
    "driving_direction",
}


def _new_worst() -> dict[str, Any]:
    return {"value": 0.0, "cache": None, "proposal": None}


def _new_pair_accumulator() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "count": 0,
            "reference_sum": 0.0,
            "candidate_sum": 0.0,
            "absolute_error_sum": 0.0,
            "squared_error_sum": 0.0,
            "exact_mismatch_count": 0,
            "mismatch_count_1e-6": 0,
            "mismatch_count_1e-5": 0,
            "non_finite_count": 0,
            "max_abs_error": _new_worst(),
            "absolute_errors": [],
        }
        for name in TARGET_NAMES
    }


def _accumulate_pair(
    accumulator: dict[str, dict[str, Any]],
    reference: np.ndarray,
    candidate: np.ndarray,
    cache: str,
) -> dict[str, dict[str, Any]]:
    if reference.shape != candidate.shape or reference.shape[-1] != len(TARGET_NAMES):
        raise ValueError(
            f"target shapes differ: reference={reference.shape}, candidate={candidate.shape}"
        )
    per_cache: dict[str, dict[str, Any]] = {}
    for column, name in enumerate(TARGET_NAMES):
        reference_column = reference[:, column].astype(np.float64, copy=False)
        candidate_column = candidate[:, column].astype(np.float64, copy=False)
        finite = np.isfinite(reference_column) & np.isfinite(candidate_column)
        difference = np.full(reference_column.shape, np.inf, dtype=np.float64)
        difference[finite] = np.abs(candidate_column[finite] - reference_column[finite])
        non_finite = int(np.count_nonzero(~finite))
        exact_mismatch = int(np.count_nonzero(candidate_column != reference_column))
        mismatch_1e6 = int(np.count_nonzero(difference > 1e-6))
        mismatch_1e5 = int(np.count_nonzero(difference > 1e-5))
        maximum = float(np.max(difference, initial=0.0))
        slot = accumulator[name]
        slot["count"] += int(reference_column.size)
        slot["reference_sum"] += float(np.sum(reference_column[finite]))
        slot["candidate_sum"] += float(np.sum(candidate_column[finite]))
        slot["absolute_error_sum"] += float(np.sum(difference[finite]))
        slot["squared_error_sum"] += float(np.sum(difference[finite] ** 2))
        slot["exact_mismatch_count"] += exact_mismatch
        slot["mismatch_count_1e-6"] += mismatch_1e6
        slot["mismatch_count_1e-5"] += mismatch_1e5
        slot["non_finite_count"] += non_finite
        slot["absolute_errors"].extend(difference[finite].tolist())
        if maximum > slot["max_abs_error"]["value"]:
            proposal = int(np.argmax(difference))
            slot["max_abs_error"].update(
                value=maximum, cache=cache, proposal=proposal
            )
        per_cache[name] = {
            "max_abs_error": maximum,
            "exact_mismatch_count": exact_mismatch,
            "mismatch_count_1e-6": mismatch_1e6,
            "mismatch_count_1e-5": mismatch_1e5,
            "non_finite_count": non_finite,
        }
    return per_cache


def _finalize_pair(
    accumulator: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, values in accumulator.items():
        finite_count = values["count"] - values["non_finite_count"]
        denominator = max(finite_count, 1)
        result[name] = {
            "count": values["count"],
            "reference_mean": values["reference_sum"] / denominator,
            "candidate_mean": values["candidate_sum"] / denominator,
            "mean_abs_error": values["absolute_error_sum"] / denominator,
            "root_mean_squared_error": math.sqrt(
                values["squared_error_sum"] / denominator
            ),
            "exact_mismatch_count": values["exact_mismatch_count"],
            "mismatch_count_1e-6": values["mismatch_count_1e-6"],
            "mismatch_count_1e-5": values["mismatch_count_1e-5"],
            "non_finite_count": values["non_finite_count"],
            "max_abs_error": values["max_abs_error"],
            "absolute_error_percentiles": {
                f"p{percentile:g}": float(
                    np.percentile(values["absolute_errors"], percentile)
                )
                for percentile in (50, 90, 95, 99, 99.9)
            },
        }
    return result


def _timing_summary(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _wall_cuda(
    function: Callable[[], tuple[torch.Tensor, ...]], device: torch.device
) -> tuple[tuple[torch.Tensor, ...], float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = function()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def _progress(cache: Any, num_proposals: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(cache.pdm_progress, device=device, dtype=torch.float32)
    if value.numel() == 1:
        return value.reshape(1)
    if value.numel() == num_proposals:
        return value.reshape(1, num_proposals)
    raise ValueError(f"unexpected pdm_progress shape: {tuple(value.shape)}")


def _max_error(
    reference: torch.Tensor, candidate: torch.Tensor, cache: str
) -> dict[str, Any]:
    difference = (candidate - reference).abs()
    maximum = float(difference.amax().cpu())
    flat_index = int(difference.argmax().cpu())
    return {"value": maximum, "cache": cache, "flat_index": flat_index}


def _update_worst(destination: dict[str, Any], candidate: dict[str, Any]) -> None:
    if candidate["value"] > destination["value"]:
        destination.update(candidate)


def _validate(summary: dict[str, Any]) -> list[str]:
    failures = [
        f"{value['cache']}: {value['type']}: {value['message']}"
        for value in summary["exceptions"]
    ]
    # The production candidate is the full Triton path. The pure-Torch path is
    # still reported as a diagnostic, but it is not an acceptance oracle: tiny
    # float32 simulator differences can legitimately flip a boundary collision.
    for name in DISCRETE_TARGETS:
        metric = summary["comparisons"]["cpu_triton"][name]
        if metric["exact_mismatch_count"]:
            failures.append(
                f"cpu_triton.{name}: {metric['exact_mismatch_count']} exact mismatches"
            )
    # End-to-end float32 interpolation/simulation can move a point across an
    # exactly tied centerline segment or map boundary. Keep it bounded while
    # using the scorer-only official comparison as the stricter scorer oracle.
    tolerances = {"progress": 1e-3, "score": 5e-4}
    for name, tolerance in tolerances.items():
        maximum = summary["comparisons"]["cpu_triton"][name]["max_abs_error"]["value"]
        if maximum > tolerance:
            failures.append(
                f"cpu_triton.{name}: {maximum:.9g} > {tolerance:.9g}"
            )

    # Official CPU and Triton scorers consume the exact same GPU-produced states.
    for name in DISCRETE_TARGETS:
        metric = summary["comparisons"]["cpu_same_triton"][name]
        if metric["exact_mismatch_count"]:
            failures.append(
                f"cpu_same_triton.{name}: "
                f"{metric['exact_mismatch_count']} exact mismatches"
            )
    for name in ("progress", "score"):
        maximum = summary["comparisons"]["cpu_same_triton"][name][
            "max_abs_error"
        ]["value"]
        if maximum > 1e-4:
            failures.append(f"cpu_same_triton.{name}: {maximum:.9g} > 1e-4")
    for path_name in ("cpu_same_triton", "torch_triton_same_states"):
        count = summary["ego_area_mismatch_count"][path_name]
        if count:
            failures.append(f"{path_name} ego-area mismatches: {count}")
    for path_name, values in summary["comparisons"].items():
        for name, metric in values.items():
            if metric["non_finite_count"]:
                failures.append(
                    f"{path_name}.{name}: {metric['non_finite_count']} non-finite values"
                )
    if summary["intermediate_max_abs_error"]["cpu_triton_simulation"]["value"] > 2e-3:
        failures.append("CPU/Triton simulation max error exceeds 2e-3")
    if summary["completed"] != summary["requested"]:
        failures.append(
            f"only {summary['completed']} of {summary['requested']} caches completed"
        )
    return failures


def benchmark(
    caches: list[Path],
    device: torch.device,
    num_proposals: int,
    warm_repeats: int,
    repeats: int,
) -> dict[str, Any]:
    pair_accumulators = {
        "cpu_torch": _new_pair_accumulator(),
        "cpu_triton": _new_pair_accumulator(),
        "torch_triton": _new_pair_accumulator(),
        "torch_triton_same_states": _new_pair_accumulator(),
        "cpu_same_torch": _new_pair_accumulator(),
        "cpu_same_triton": _new_pair_accumulator(),
    }
    timing_values: dict[str, list[float]] = {
        name: []
        for name in (
            "cache_load",
            "proposal_generation",
            "official_cpu_pipeline",
            "official_cpu_score_same_torch_states",
            "scene_pack_cpu",
            "scene_h2d",
            "scene_pack_and_h2d",
            "input_prepare_h2d",
            "torch_first_call",
            "triton_first_call",
            "torch_gpu_steady",
            "triton_gpu_steady",
            "triton_end_to_end_with_scene_pack",
            "official_cpu_over_triton_compute_speedup",
            "official_cpu_over_triton_end_to_end_speedup",
            "torch_over_triton_compute_speedup",
        )
    }
    summary: dict[str, Any] = {
        "device": str(device),
        "num_proposals_per_cache": num_proposals,
        "requested": len(caches),
        "completed": 0,
        "exceptions": [],
        "ego_area_mismatch_count": {
            "cpu_torch": 0,
            "cpu_triton": 0,
            "torch_triton": 0,
            "torch_triton_same_states": 0,
            "cpu_same_torch": 0,
            "cpu_same_triton": 0,
        },
        "intermediate_max_abs_error": {
            name: {"value": 0.0, "cache": None, "flat_index": None}
            for name in (
                "cpu_torch_reference",
                "cpu_triton_reference",
                "torch_triton_reference",
                "cpu_torch_simulation",
                "cpu_triton_simulation",
                "torch_triton_simulation",
            )
        },
        "maximum_scene_size": {
            name: {"value": 0, "cache": None}
            for name in ("agents", "polygons", "edges", "centerline_points")
        },
        "per_cache": [],
    }

    for index, cache_path in enumerate(caches, start=1):
        cache_name = str(cache_path)
        print(f"[{index}/{len(caches)}] {cache_name}", flush=True)
        try:
            start = time.perf_counter()
            cache = load_metric_cache(cache_path)
            cache_load_seconds = time.perf_counter() - start

            start = time.perf_counter()
            proposals_np = generate_proposals(cache, num_proposals)
            proposal_seconds = time.perf_counter() - start

            start = time.perf_counter()
            cpu_targets, cpu_reference, cpu_simulated, cpu_ego_area = _cpu_pipeline(
                cache, proposals_np
            )
            cpu_seconds = time.perf_counter() - start

            start = time.perf_counter()
            scene_cpu = prepare_pdm_scene(
                cache, device="cpu", dtype=torch.float32, local_frame=True
            )
            scene_pack_cpu_seconds = time.perf_counter() - start

            torch.cuda.synchronize(device)
            start = time.perf_counter()
            scene = PreparedPDMScene(
                **{
                    name: getattr(scene_cpu, name).to(device=device)
                    for name in scene_cpu.__dataclass_fields__
                }
            )
            torch.cuda.synchronize(device)
            scene_h2d_seconds = time.perf_counter() - start
            scene_seconds = scene_pack_cpu_seconds + scene_h2d_seconds

            torch.cuda.synchronize(device)
            start = time.perf_counter()
            initial = torch.as_tensor(
                ego_state_to_state_array(cache.ego_state),
                device=device,
                dtype=torch.float64,
            )
            initial[:2] -= scene.origin[0]
            initial = initial.float().unsqueeze(0)
            proposals = torch.as_tensor(
                proposals_np, device=device, dtype=torch.float32
            ).unsqueeze(0)
            pdm_progress = _progress(cache, num_proposals, device)
            torch.cuda.synchronize(device)
            input_seconds = time.perf_counter() - start

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

            def torch_pipeline() -> tuple[torch.Tensor, ...]:
                reference = proposals_to_state_array(proposals, initial)
                simulated = simulator(reference, initial)
                targets, areas = score_v1_gpu(
                    simulated, scene, geometry, pdm_progress
                )
                return reference, simulated, targets, areas

            def triton_pipeline() -> tuple[torch.Tensor, ...]:
                return run_v1_triton_pipeline(
                    proposals,
                    initial,
                    scene,
                    geometry,
                    pdm_progress,
                    simulator,
                )

            torch_result, torch_first_seconds = _wall_cuda(torch_pipeline, device)
            triton_result, triton_first_seconds = _wall_cuda(triton_pipeline, device)
            with torch.inference_mode():
                same_state_targets, same_state_areas = score_v1_triton(
                    torch_result[1], scene, geometry, pdm_progress
                )
            torch.cuda.synchronize(device)
            origin = scene.origin[0].cpu().numpy()
            torch_simulated_global = (
                torch_result[1][0].cpu().numpy().astype(np.float64, copy=True)
            )
            torch_simulated_global[..., :2] += origin
            start = time.perf_counter()
            same_cpu_targets, same_cpu_area = _cpu_score_states(
                cache, torch_simulated_global
            )
            same_cpu_seconds = time.perf_counter() - start
            torch_steady_seconds = _timed_cuda(
                torch_pipeline, device, warm_repeats, repeats
            )
            triton_steady_seconds = _timed_cuda(
                triton_pipeline, device, warm_repeats, repeats
            )

            cpu_targets_array = np.asarray(cpu_targets)
            torch_targets_array = torch_result[2][0].cpu().numpy()
            triton_targets_array = triton_result[2][0].cpu().numpy()
            cache_columns = {
                "cpu_torch": _accumulate_pair(
                    pair_accumulators["cpu_torch"],
                    cpu_targets_array,
                    torch_targets_array,
                    cache_name,
                ),
                "cpu_triton": _accumulate_pair(
                    pair_accumulators["cpu_triton"],
                    cpu_targets_array,
                    triton_targets_array,
                    cache_name,
                ),
                "torch_triton": _accumulate_pair(
                    pair_accumulators["torch_triton"],
                    torch_targets_array,
                    triton_targets_array,
                    cache_name,
                ),
                "torch_triton_same_states": _accumulate_pair(
                    pair_accumulators["torch_triton_same_states"],
                    torch_targets_array,
                    same_state_targets[0].cpu().numpy(),
                    cache_name,
                ),
                "cpu_same_torch": _accumulate_pair(
                    pair_accumulators["cpu_same_torch"],
                    same_cpu_targets,
                    torch_targets_array,
                    cache_name,
                ),
                "cpu_same_triton": _accumulate_pair(
                    pair_accumulators["cpu_same_triton"],
                    same_cpu_targets,
                    same_state_targets[0].cpu().numpy(),
                    cache_name,
                ),
            }

            torch_areas = torch_result[3][0].cpu().numpy()
            triton_areas = triton_result[3][0].cpu().numpy()
            area_mismatches = {
                "cpu_torch": int(np.count_nonzero(cpu_ego_area != torch_areas)),
                "cpu_triton": int(np.count_nonzero(cpu_ego_area != triton_areas)),
                "torch_triton": int(np.count_nonzero(torch_areas != triton_areas)),
                "torch_triton_same_states": int(
                    np.count_nonzero(torch_areas != same_state_areas[0].cpu().numpy())
                ),
                "cpu_same_torch": int(
                    np.count_nonzero(same_cpu_area != torch_areas)
                ),
                "cpu_same_triton": int(
                    np.count_nonzero(
                        same_cpu_area != same_state_areas[0].cpu().numpy()
                    )
                ),
            }
            for name, count in area_mismatches.items():
                summary["ego_area_mismatch_count"][name] += count

            cpu_reference_local = cpu_reference.copy()
            cpu_reference_local[..., :2] -= origin
            cpu_simulated_local = cpu_simulated.copy()
            cpu_simulated_local[..., :2] -= origin
            cpu_reference_tensor = torch.as_tensor(
                cpu_reference_local, device=device, dtype=torch.float32
            ).unsqueeze(0)
            cpu_simulated_tensor = torch.as_tensor(
                cpu_simulated_local, device=device, dtype=torch.float32
            ).unsqueeze(0)
            intermediate = {
                "cpu_torch_reference": _max_error(
                    cpu_reference_tensor, torch_result[0], cache_name
                ),
                "cpu_triton_reference": _max_error(
                    cpu_reference_tensor, triton_result[0], cache_name
                ),
                "torch_triton_reference": _max_error(
                    torch_result[0], triton_result[0], cache_name
                ),
                "cpu_torch_simulation": _max_error(
                    cpu_simulated_tensor, torch_result[1], cache_name
                ),
                "cpu_triton_simulation": _max_error(
                    cpu_simulated_tensor, triton_result[1], cache_name
                ),
                "torch_triton_simulation": _max_error(
                    torch_result[1], triton_result[1], cache_name
                ),
            }
            for name, value in intermediate.items():
                _update_worst(summary["intermediate_max_abs_error"][name], value)

            scene_size = {
                "agents": int(scene.agent_valid.any(dim=1).sum().cpu()),
                "polygons": int(scene.lane_polygon_mask.shape[1]),
                "edges": int(scene.map_edge_valid.sum().cpu()),
                "centerline_points": int(scene.centerline_point_mask.sum().cpu()),
            }
            for name, value in scene_size.items():
                if value > summary["maximum_scene_size"][name]["value"]:
                    summary["maximum_scene_size"][name].update(
                        value=value, cache=cache_name
                    )

            end_to_end_seconds = scene_seconds + input_seconds + triton_steady_seconds
            timings = {
                "cache_load": cache_load_seconds,
                "proposal_generation": proposal_seconds,
                "official_cpu_pipeline": cpu_seconds,
                "official_cpu_score_same_torch_states": same_cpu_seconds,
                "scene_pack_cpu": scene_pack_cpu_seconds,
                "scene_h2d": scene_h2d_seconds,
                "scene_pack_and_h2d": scene_seconds,
                "input_prepare_h2d": input_seconds,
                "torch_first_call": torch_first_seconds,
                "triton_first_call": triton_first_seconds,
                "torch_gpu_steady": torch_steady_seconds,
                "triton_gpu_steady": triton_steady_seconds,
                "triton_end_to_end_with_scene_pack": end_to_end_seconds,
                "official_cpu_over_triton_compute_speedup": cpu_seconds
                / triton_steady_seconds,
                "official_cpu_over_triton_end_to_end_speedup": cpu_seconds
                / end_to_end_seconds,
                "torch_over_triton_compute_speedup": torch_steady_seconds
                / triton_steady_seconds,
            }
            for name, value in timings.items():
                timing_values[name].append(value)

            score_error = cache_columns["cpu_triton"]["score"]["max_abs_error"]
            print(
                "  "
                f"cpu={cpu_seconds:.3f}s pack={scene_seconds:.3f}s "
                f"triton_first={triton_first_seconds:.3f}s "
                f"triton_steady={triton_steady_seconds:.3f}s "
                f"score_err={score_error:.3g}",
                flush=True,
            )
            summary["per_cache"].append(
                {
                    "cache": cache_name,
                    "scene": scene_size,
                    "timing_seconds": timings,
                    "columns": cache_columns,
                    "ego_area_mismatch_count": area_mismatches,
                    "intermediate_max_abs_error": intermediate,
                }
            )
            summary["completed"] += 1
        except Exception as error:  # continue to expose all unsupported scenes
            summary["exceptions"].append(
                {
                    "cache": cache_name,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)

    summary["total_proposals"] = summary["completed"] * num_proposals
    summary["comparisons"] = {
        name: _finalize_pair(values) for name, values in pair_accumulators.items()
    }
    summary["timing_seconds"] = {
        name: _timing_summary(values) for name, values in timing_values.items()
    }
    summary["failures"] = _validate(summary)
    summary["passed"] = not summary["failures"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="*", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cache and args.cache_root:
        parser.error("pass explicit cache paths or --cache-root, not both")
    if args.cache_root:
        caches = select_caches(args.cache_root, args.sample_count)
    elif args.cache:
        caches = args.cache
    else:
        parser.error("cache paths or --cache-root are required")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("the Triton comparison requires a CUDA device")

    summary = benchmark(
        caches,
        device,
        args.num_proposals,
        args.warm_repeats,
        args.repeats,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"REPORT_JSON={args.output}")
    compact = {name: value for name, value in summary.items() if name != "per_cache"}
    print("SUMMARY_JSON")
    print(json.dumps(compact, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
