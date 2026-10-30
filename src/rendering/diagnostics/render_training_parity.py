"""Native export/loader and fresh history-encoder parity after render comparison."""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from driversi.rendering.diagnostics.render_benchmark import branch_inputs, write


def equal(a, b):
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return (
            isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        )
    return bool(a == b)


def history_images(directory, config):
    from PIL import Image

    images = []
    for tick in (5, 10, 15, 20):
        for camera in config["cameras"]:
            with Image.open(directory / f"{tick:04d}-{camera}.png") as image:
                image = image.convert("RGB")
                image.thumbnail((config["max_side"], config["max_side"]), Image.Resampling.LANCZOS)
                images.append(image.copy())
    return images


def main(config_path, benchmark, output):
    from driversi.agents.backends.native import digest, project_root
    from driversi_agents.chainflow.synthetic import export_native_frames

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = project_root()
    sys.path[:0] = [
        str(root / "agents/chainflow/native"),
        str(root / "src/vendor/driving/nuplan-devkit"),
    ]
    os.environ["NUPLAN_MAPS_ROOT"] = str(root.parent / "dataset/maps")
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    from navsim.common.dataclasses import Scene, SensorConfig

    config = json.loads(Path(config_path).read_text())
    benchmark = Path(benchmark)
    rows = []
    for i, branch in enumerate(config["branches"]):
        kwargs, _, _, _ = branch_inputs(branch["render_report"])
        reports = {
            mode: benchmark / f"r0-{mode}" / str(i) / "render/report.json"
            for mode in ("scalar", "optimized")
        }
        exports, scenes, frames = {}, {}, {}
        for mode, path in reports.items():
            started = time.perf_counter()
            exported = export_native_frames(
                kwargs["scene_path"],
                kwargs["rollout_path"],
                path,
                output / str(i) / mode,
                anchor_tick=20,
                future_points=10,
            )
            with Path(exported["log_path"]).open("rb") as f:
                frames[mode] = pickle.load(f)
            scenes[mode] = Scene.from_scene_dict_list(
                frames[mode],
                Path(exported["sensor_root"]),
                4,
                10,
                SensorConfig(
                    cam_f0=True,
                    cam_l0=True,
                    cam_r0=True,
                    cam_b0=True,
                    cam_l1=False,
                    cam_l2=False,
                    cam_r1=False,
                    cam_r2=False,
                    lidar_pc=False,
                ),
            )
            exports[mode] = dict(
                export_seconds=time.perf_counter() - started,
                export=str(output / str(i) / mode / "export.json"),
                counts=exported["image_counts"],
            )
        archived = json.loads(Path(branch["native_report"]).read_text())
        with Path(archived["row"]["log_path"]).open("rb") as handle:
            original_frames = pickle.load(handle)
        label_equal = equal(frames["scalar"], frames["optimized"]) and equal(
            frames["optimized"], original_frames
        )
        future_equal = all(
            np.array_equal(
                scenes["scalar"].get_future_trajectory(n).poses,
                scenes["optimized"].get_future_trajectory(n).poses,
            )
            for n in (8, 10)
        )
        rows.append(
            dict(
                branch=i,
                native_frames_equal=label_equal,
                native_future_targets_equal=future_equal,
                exports=exports,
                physical_input_sha256={
                    str(kwargs[k]): digest(kwargs[k]) for k in ("scene_path", "rollout_path")
                },
            )
        )
        if not label_equal or not future_equal:
            write(output / "native-parity.json", dict(status="failed", rows=rows))
            raise ValueError("Native geometry changed")
    write(
        output / "native-parity.json", dict(status="verified", rows=rows, new_production_samples=0)
    )

    import torch

    from driversi.data.catalog.features import VisionEncoder, feature_profile

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.15, 0)
    original = json.loads(Path(config["branches"][0]["feature_report"]).read_text())
    encoder_config = original["actual_encoder_profile"]["config"]
    for branch in config["branches"]:
        other = json.loads(Path(branch["feature_report"]).read_text())["actual_encoder_profile"]
        if (
            other["config"] != encoder_config
            or other["checkpoint_files"] != original["actual_encoder_profile"]["checkpoint_files"]
        ):
            raise ValueError(
                "Benchmark branches require the same verified feature model/configuration"
            )
    profile = feature_profile(encoder_config)
    if profile["checkpoint_files"] != original["actual_encoder_profile"]["checkpoint_files"]:
        raise ValueError("Feature checkpoint changed")
    backend = VisionEncoder(encoder_config)
    feature_rows = []
    for i in range(len(rows)):
        vectors, costs = {}, {}
        for mode in ("scalar", "optimized"):
            images = history_images(benchmark / f"r0-{mode}" / str(i) / "render", encoder_config)
            vectors[mode], costs[mode] = backend(images)
            np.save(output / f"{i}-{mode}-features.npy", vectors[mode])
        exact = np.array_equal(vectors["scalar"], vectors["optimized"])
        feature_rows.append(
            dict(
                branch=i,
                fresh_features_exact=bool(exact),
                max_absolute_error=float(np.abs(vectors["scalar"] - vectors["optimized"]).max()),
                costs=costs,
                shape=list(vectors["scalar"].shape),
            )
        )
    result = dict(
        status="verified" if all(x["fresh_features_exact"] for x in feature_rows) else "failed",
        native_rows=rows,
        feature_rows=feature_rows,
        profile=profile,
        new_production_samples=0,
        scope="native export, native scene loading and 4+10 trajectory targets, fresh visual encoder outputs; no production admission or index writes",
    )
    write(output / "summary.json", result)
    if result["status"] != "verified":
        raise ValueError("Fresh history feature mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.config, args.benchmark, args.output)
