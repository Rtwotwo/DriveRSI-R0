"""Sweep real metric caches across the CPU, Torch, and Triton PDM paths.

This module is deliberately kept below ``gpu_pdm/validation``.  It is not
imported by an agent, model, loss, or training entry point.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

from navsim.agents.chainflow_vla.utils.gpu_pdm.cuda.experiment import (
    compare_cache as compare_cuda_cache,
)
from navsim.agents.chainflow_vla.utils.gpu_pdm.parity import (
    TARGET_NAMES,
    compare_cache as compare_cpu_cache,
)


DISCRETE_TARGETS = {
    "no_collision",
    "drivable_area",
    "ttc",
    "comfort",
    "driving_direction",
}


def _first_cache(log_directory: Path) -> Optional[Path]:
    # Stop at the first cache: materializing and sorting every token in a log
    # directory causes avoidable metadata traffic when the cache lives on NAS.
    return next(log_directory.rglob("metric_cache.pkl"), None)


def select_caches(cache_root: Path, count: int) -> list[Path]:
    """Select one cache from evenly spaced log directories."""

    logs = sorted(path for path in cache_root.iterdir() if path.is_dir())
    if not logs:
        raise ValueError(f"no log directories below {cache_root}")
    sample_count = min(count, len(logs))
    target_indices = np.linspace(0, len(logs) - 1, sample_count, dtype=np.int64)
    caches = []
    used_indices: set[int] = set()
    for target in target_indices.tolist():
        # Some log directories contain no cache. Search symmetrically around the
        # desired index so the sweep retains both its size and temporal spread.
        candidate_indices = [target]
        for offset in range(1, len(logs)):
            candidate_indices.extend((target + offset, target - offset))
        for index in candidate_indices:
            if index < 0 or index >= len(logs) or index in used_indices:
                continue
            cache = _first_cache(logs[index])
            if cache is not None:
                used_indices.add(index)
                caches.append(cache)
                break
    if not caches:
        raise ValueError(f"no metric_cache.pkl below {cache_root}")
    return caches


def _empty_worst() -> dict[str, Any]:
    return {"value": -math.inf, "cache": None}


def _update_worst(slot: dict[str, Any], value: float, cache: str) -> None:
    if value > slot["value"]:
        slot.update(value=value, cache=cache)


def _new_column_summary() -> dict[str, dict[str, Any]]:
    return {
        name: {"mismatch_count": 0, "max_abs_error": _empty_worst()}
        for name in TARGET_NAMES
    }


def _accumulate_columns(
    summary: dict[str, dict[str, Any]],
    columns: dict[str, dict[str, Any]],
    cache: str,
    mismatch_key: str,
) -> None:
    for name, values in columns.items():
        summary[name]["mismatch_count"] += int(values[mismatch_key])
        _update_worst(
            summary[name]["max_abs_error"],
            float(values["max_abs_error"]),
            cache,
        )


def _finite_worst(slot: dict[str, Any]) -> None:
    if slot["value"] == -math.inf:
        slot["value"] = 0.0


def _validate(summary: dict[str, Any]) -> list[str]:
    failures = list(summary["exceptions"])
    for path_name in ("cpu_torch_float64", "cpu_torch_float32"):
        path = summary[path_name]
        for name in DISCRETE_TARGETS:
            if path[name]["mismatch_count"]:
                failures.append(
                    f"{path_name}.{name}: {path[name]['mismatch_count']} mismatches"
                )
        for name in ("progress", "score"):
            error = path[name]["max_abs_error"]["value"]
            if error > 1e-4:
                failures.append(f"{path_name}.{name}: max error {error:.9g} > 1e-4")

    for path_name in ("torch_triton_same_states", "torch_triton_end_to_end"):
        path = summary[path_name]
        for name in TARGET_NAMES:
            if path[name]["mismatch_count"]:
                failures.append(
                    f"{path_name}.{name}: {path[name]['mismatch_count']} errors > 1e-5"
                )

    for name in (
        "cpu_torch_ego_area_mismatch_count",
        "torch_triton_same_states_ego_area_mismatch_count",
        "torch_triton_end_to_end_ego_area_mismatch_count",
    ):
        if summary[name]:
            failures.append(f"{name}: {summary[name]}")
    if summary["completed"] != summary["requested"]:
        failures.append(
            f"only {summary['completed']} of {summary['requested']} caches completed"
        )
    return failures


def sweep(
    caches: Iterable[Path],
    device: torch.device,
    num_proposals: int,
) -> dict[str, Any]:
    cache_list = list(caches)
    summary: dict[str, Any] = {
        "device": str(device),
        "requested": len(cache_list),
        "completed": 0,
        "exceptions": [],
        "cpu_torch_float64": _new_column_summary(),
        "cpu_torch_float32": _new_column_summary(),
        "torch_triton_same_states": _new_column_summary(),
        "torch_triton_end_to_end": _new_column_summary(),
        "cpu_torch_ego_area_mismatch_count": 0,
        "torch_triton_same_states_ego_area_mismatch_count": 0,
        "torch_triton_end_to_end_ego_area_mismatch_count": 0,
        "trajectory_max_abs_error": _empty_worst(),
        "triton_reference_max_abs_error": _empty_worst(),
        "triton_simulation_max_abs_error": _empty_worst(),
        "maximum_scene_size": {
            name: {"value": 0, "cache": None}
            for name in ("agents", "polygons", "edges", "centerline_points")
        },
        "timing_seconds": {"torch_gpu": [], "triton": [], "speedup": []},
        "caches": [],
    }

    for index, cache in enumerate(cache_list, start=1):
        cache_name = str(cache)
        print(f"[{index}/{len(cache_list)}] {cache_name}", flush=True)
        try:
            cpu_report = compare_cpu_cache(
                cache,
                device,
                num_proposals=num_proposals,
                warm_repeats=0,
                test_float32_local=True,
            )
            cuda_report = compare_cuda_cache(
                cache,
                device,
                num_proposals=num_proposals,
                warm_repeats=0,
                repeats=1,
            )
        except Exception as error:  # keep the sweep running to expose all bad scenes
            summary["exceptions"].append(
                {"cache": cache_name, "type": type(error).__name__, "message": str(error)}
            )
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            continue

        _accumulate_columns(
            summary["cpu_torch_float64"],
            cpu_report["columns"],
            cache_name,
            "mismatch_count",
        )
        _accumulate_columns(
            summary["cpu_torch_float32"],
            cpu_report["float32_local"]["columns"],
            cache_name,
            "mismatch_count",
        )
        same_states = cuda_report["parity"]["scorer_same_states"]
        end_to_end = cuda_report["parity"]["end_to_end"]
        _accumulate_columns(
            summary["torch_triton_same_states"],
            same_states["columns"],
            cache_name,
            "mismatch_count_1e-5",
        )
        _accumulate_columns(
            summary["torch_triton_end_to_end"],
            end_to_end["columns"],
            cache_name,
            "mismatch_count_1e-5",
        )
        summary["cpu_torch_ego_area_mismatch_count"] += sum(
            value["mismatch_count"]
            for value in cpu_report["ego_area_mismatch_count"].values()
        )
        summary["torch_triton_same_states_ego_area_mismatch_count"] += int(
            same_states["ego_area_mismatch_count"]
        )
        summary["torch_triton_end_to_end_ego_area_mismatch_count"] += int(
            end_to_end["ego_area_mismatch_count"]
        )
        _update_worst(
            summary["trajectory_max_abs_error"],
            float(cpu_report["float32_local"]["trajectory_max_abs_error"]),
            cache_name,
        )
        _update_worst(
            summary["triton_reference_max_abs_error"],
            float(cuda_report["parity"]["reference_max_abs_error"]),
            cache_name,
        )
        _update_worst(
            summary["triton_simulation_max_abs_error"],
            float(cuda_report["parity"]["simulation_max_abs_error"]),
            cache_name,
        )
        for name, value in cuda_report["scene"].items():
            scene_slot = summary["maximum_scene_size"][name]
            if int(value) > scene_slot["value"]:
                scene_slot.update(value=int(value), cache=cache_name)
        timing = cuda_report["timing_seconds"]
        summary["timing_seconds"]["torch_gpu"].append(float(timing["torch_gpu"]))
        summary["timing_seconds"]["triton"].append(float(timing["native_cuda"]))
        summary["timing_seconds"]["speedup"].append(float(timing["speedup"]))
        summary["caches"].append(cache_name)
        summary["completed"] += 1

    for path_name in (
        "cpu_torch_float64",
        "cpu_torch_float32",
        "torch_triton_same_states",
        "torch_triton_end_to_end",
    ):
        for values in summary[path_name].values():
            _finite_worst(values["max_abs_error"])
    for name in (
        "trajectory_max_abs_error",
        "triton_reference_max_abs_error",
        "triton_simulation_max_abs_error",
    ):
        _finite_worst(summary[name])

    timings = summary["timing_seconds"]
    summary["timing_seconds"] = {
        name: {
            "median": float(np.median(values)) if values else None,
            "minimum": float(np.min(values)) if values else None,
            "maximum": float(np.max(values)) if values else None,
        }
        for name, values in timings.items()
    }
    summary["failures"] = _validate(summary)
    summary["passed"] = not summary["failures"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="*", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--num-proposals", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
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
    summary = sweep(caches, device, args.num_proposals)
    print("SUMMARY_JSON")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
