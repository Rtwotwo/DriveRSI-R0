"""Bounded full MTGS rendering of a sealed executed teacher window.

The complete upstream checkpoint, optics and measured surface are inputs.
This producer cannot certify its own pixels or authorize training.
"""

import os
import shutil
from pathlib import Path

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.native_branch import validate_trajectory_evidence
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.rsi.plugins.worldengine import CAMERAS, FUTURE, HISTORY, validate_render


def prepare(config, base, registry):
    _keys(
        config,
        (
            "config_path",
            "checkpoint_path",
            "road_block_config",
            "travel_id",
            "python",
            "gpu",
            "cuda_home",
            "max_wall_seconds",
        ),
        "MTGS executed window",
    )
    if type(config["travel_id"]) is not int or config["travel_id"] < 0:
        raise ValueError("Explicit native traversal required")
    if type(config["gpu"]) is not int or config["gpu"] < 0:
        raise ValueError("Explicit GPU required")
    if type(config["max_wall_seconds"]) is not int or not 1 <= config["max_wall_seconds"] <= 600:
        raise ValueError("Bounded MTGS rendering wall time required")
    resolved = dict(config)
    for key in ("config_path", "checkpoint_path", "road_block_config", "python", "cuda_home"):
        resolved[key] = str(path_at(base, config[key]))
    if not (Path(resolved["cuda_home"]) / "bin/nvcc").is_file():
        raise ValueError("CUDA compiler missing")
    from driversi.rendering.gaussian.mtgs_training_window import asset_inputs

    assets = asset_inputs(
        resolved["config_path"],
        resolved["checkpoint_path"],
        resolved["road_block_config"],
        resolved["travel_id"],
    )
    code = [
        project_root() / path
        for path in (
            "src/rsi/plugins/mtgs_window.py",
            "src/rendering/gaussian/mtgs_training_window.py",
            "src/rendering/gaussian/mtgs_runtime.py",
            "src/rendering/gaussian/mtgs_road_surface.py",
            "src/rendering/gaussian/mtgs_surface_registration.py",
            "src/rendering/gaussian/mtgs_bindings.py",
            "src/data/scenes/pufferdrive.py",
            "src/rendering/gaussian/worldengine_replay.py",
            "src/rendering/gaussian/mtgs_registration.py",
        )
    ]
    resolved["asset_files"] = assets
    resolved["code_files"] = {str(path): digest(path) for path in code}
    check_files({**assets, **resolved["code_files"]})
    return resolved, 0


def execute(config, inputs, output, registry):
    selected = inputs["branch"]
    branch = read_mapping(selected["artifacts"]["report"])
    sealed = {
        **config["asset_files"],
        **config["code_files"],
        **branch["files"],
        **branch["upstream_source_files"],
    }
    check_files(sealed)
    protect([branch["source_log"]], registry)
    future_points = branch.get("future_frames")
    if (
        branch.get("teacher_qualified") is not True
        or branch.get("anchor_tick") != 20
        or type(future_points) is not int
        or future_points not in (8, 10)
    ):
        raise ValueError("Qualified four-history eight/ten-future teacher branch required")
    validate_trajectory_evidence(branch["trajectory_evidence"])
    execution = Path(branch["execution"])
    scene_path, rollout = execution / "scene.npz", execution / (branch["branch"] + ".npz")
    scene = Scene.load(scene_path)
    raw_log = scene.metadata["navsim_source"]["log_path"]
    if scene.metadata["split"] != "train" or raw_log not in branch["upstream_source_files"]:
        raise ValueError("Sealed NAVSIM training source required")
    window = Path(selected["artifacts"]["trajectory_window"])
    if digest(window) != branch["trajectory_window_sha256"]:
        raise ValueError("Selected trajectory window changed")
    if len(read_mapping(window)["future"]) != future_points:
        raise ValueError("Declared actual future samples required")
    if not output.resolve().is_relative_to(project_root() / "exp"):
        raise ValueError("MTGS pilot outputs must live in exp")
    output.mkdir(parents=True)
    destination = output / "render"
    run_process(
        [
            config["python"],
            "-B",
            "-m",
            "driversi.rendering.gaussian.mtgs_training_window",
            "--scene",
            str(scene_path),
            "--rollout",
            str(rollout),
            "--raw-log",
            raw_log,
            "--config",
            config["config_path"],
            "--checkpoint",
            config["checkpoint_path"],
            "--road-block",
            config["road_block_config"],
            "--travel-id",
            str(config["travel_id"]),
            "--output",
            str(destination),
            "--future-points",
            str(future_points),
        ],
        output,
        config["max_wall_seconds"],
        environment={
            "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
            "CUDA_HOME": config["cuda_home"],
            "PATH": str(Path(config["python"]).parent)
            + ":"
            + config["cuda_home"]
            + "/bin:"
            + os.environ.get("PATH", ""),
        },
    )
    check_files(sealed)
    report = read_mapping(destination / "report.json")
    expected = {
        **config["asset_files"],
        **{str(p.resolve()): digest(p) for p in (scene_path, rollout, Path(raw_log))},
    }
    images = validate_render(report, destination, expected, future_points=future_points)
    if (
        report.get("renderer_backend") != "mtgs_native"
        or report["source_log"] != branch["source_log"]
        or report.get("travel_id") != config["travel_id"]
    ):
        raise ValueError("Rendered native source identity differs")
    shutil.copyfile(window, output / "trajectory-window.json")
    payload = {
        **images,
        str(destination / "report.json"): digest(destination / "report.json"),
        str(output / "trajectory-window.json"): digest(output / "trajectory-window.json"),
    }
    receipt = dict(
        kind="worldengine_inspection_window",
        renderer_backend="mtgs_native",
        branch_identity=selected["summary"]["artifact_reference"]["identity"],
        trajectory_evidence=branch["trajectory_evidence"],
        source_files=sealed,
        payload_files=payload,
        trajectory_window=read_mapping(window),
        render_report=report,
        history_images=[
            str(destination / f"{tick:04d}-{cam}.png") for tick in HISTORY for cam in CAMERAS
        ],
        future_images=[
            str(destination / f"{tick:04d}-{cam}.png")
            for tick in range(25, 21 + future_points * 5, 5)
            for cam in CAMERAS
        ],
        primary_future_ticks=FUTURE,
        native_future_points=future_points,
        image_purpose={"history": "student_input_after_admission", "future": "supervision_only"},
        measured_new_simulation_steps=0,
        render_image_count=len(images),
        physics_verification="upstream_restore_evidence_not_new_full_replay",
        terrain_fully_supported=report.get("terrain_fully_supported") is True,
        visual_quality_certified=False,
        trainable=False,
        rejection_reasons=["independent_visual_admission_missing"],
    )
    return result(
        output,
        receipt,
        artifacts={
            "trajectory_window": str(output / "trajectory-window.json"),
            "render_report": str(destination / "report.json"),
        },
    )


def register(registry, Provider):
    registry.register(
        Provider(
            "mtgs_selected_branch_render",
            "2",
            (("branch", "selected_native_branch"),),
            "worldengine_inspection_window",
            prepare,
            execute,
            ("report", "trajectory_window", "render_report"),
        )
    )
