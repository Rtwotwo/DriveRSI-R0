"""Sealed teacher branch -> bounded WorldEngine diagnostic images and window.

A complete rendering receipt is not a visual certificate. Its typed output
cannot be consumed by training services requiring an admitted dataset.
"""

import math
import os
import shutil
from pathlib import Path

from driversi.agents.backends.native import digest, path_at
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.data.assets.worldengine import bundle_files
from driversi.rsi.contracts import _keys
from driversi.rsi.plugins.native_branch import validate_trajectory_evidence
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.rsi.workflow.results import result

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
HISTORY = [5, 10, 15, 20]
FUTURE = list(range(25, 61, 5))


def prepare(config, base, registry):
    _keys(
        {k: v for k, v in config.items() if k != "width"},
        ("bundle", "python", "gpu", "cuda_home", "max_wall_seconds", "max_render_images"),
        "WorldEngine renderer",
    )
    if type(config["max_wall_seconds"]) is not int or not 1 <= config["max_wall_seconds"] <= 600:
        raise ValueError("Bounded renderer wall time required")
    if type(config["max_render_images"]) is not int or config["max_render_images"] not in (52, 60):
        raise ValueError("Explicit 13- or 15-frame four-camera pilot budget required")
    if type(config["gpu"]) is not int or config["gpu"] < 0:
        raise ValueError("Explicit GPU index required")
    width = config.get("width", 960)
    if type(width) is not int or width not in (960, 1920):
        raise ValueError("Explicit half or native reconstruction width required")
    bundle = (base / config["bundle"]).resolve(strict=True)
    files = bundle_files(bundle.parent, bundle.name)
    if not all(row["present"] for row in files.values()):
        raise ValueError("Complete reconstruction bundle required")
    cuda = (base / config["cuda_home"]).resolve(strict=True)
    if not (cuda / "bin/nvcc").is_file():
        raise ValueError("CUDA compiler missing")
    return {
        **config,
        "bundle": str(bundle),
        "width": width,
        "python": str(path_at(base, config["python"])),
        "cuda_home": str(cuda),
        "files": {row["path"]: row["sha256"] for row in files.values()},
    }, 0


def validate_render(report, root, expected_inputs, *, future_points=8):
    """Fail closed for stale source, omitted/repeated ticks or camera frames."""
    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Expected eight primary or ten native future points")
    end_tick = 20 + future_points * 5
    future = list(range(25, end_tick + 1, 5))
    contract = report.get("image_contract", "history_future_rgb_v1")
    if contract not in {"history_future_rgb_v1", "history_rgb_future_states_v1", "anchor_rgb_history_future_states_v1"}:
        raise ValueError("Unknown image contract")
    ticks = ([20] if contract == "anchor_rgb_history_future_states_v1" else HISTORY if contract == "history_rgb_future_states_v1" else list(range(0, end_tick + 1, 5)))
    image_count = len(ticks) * len(CAMERAS)
    if report.get("status") != "rendered_diagnostic" or report.get("image_count") != image_count:
        raise ValueError("Incomplete diagnostic render")
    if report.get("history_ticks") != HISTORY or report.get("future_ticks") != future:
        raise ValueError("History/future window differs")
    if report.get("quality_admitted") is not False or report.get("trainable") is not False:
        raise ValueError("Renderer cannot grant visual or training admission")
    if report.get("inputs") != expected_inputs:
        raise ValueError("Renderer did not consume the sealed branch and asset")
    output_size = report.get("output_image_size")
    if output_size is not None and output_size not in ([960, 540], [1920, 1080]):
        raise ValueError("Invalid sealed output image size")
    rows = report.get("frames", [])
    if [row["tick"] for row in rows] != ticks:
        raise ValueError("Missing or repeated physical tick")
    images = {}
    support_flags = []
    for row in rows:
        if set(row["cameras"]) != set(CAMERAS):
            raise ValueError("Camera coverage differs")
        for camera, item in row["cameras"].items():
            name = f"{row['tick']:04d}-{camera}.png"
            if item["file"] != name or item["source_token"] != row["source_token"]:
                raise ValueError("Camera and source identity differ")
            terrain = item.get("terrain_evidence", {})
            if not terrain or set(terrain) != set(row.get("motions", {})):
                raise ValueError("Missing or mismatched terrain motion evidence")
            supported = all(
                value.get("status") in {"supported", "source_identity"}
                for value in terrain.values()
            )
            if item.get("terrain_fully_supported") is not supported:
                raise ValueError("Camera terrain summary disagrees with evidence")
            for token, value in terrain.items():
                if value.get("status") == "source_identity":
                    from driversi.rendering.gaussian.mtgs_bindings import exact_float32_identity

                    binding = item.get("object_bindings", {}).get(token)
                    if (
                        report.get("renderer_backend") != "mtgs_native"
                        or not binding
                        or value.get("method") != "source_identity_transport_v1"
                        or value.get("source_token") != binding.get("source_token")
                        or value.get("reference_tick") != binding.get("reference_tick")
                    ):
                        raise ValueError("Native source identity transport lacks exact binding")
                    identity = exact_float32_identity(
                        binding["reference_physical_pose"], binding["target_physical_pose"]
                    )
                    if not identity["equal"] or value.get("identity") != identity:
                        raise ValueError("Native source identity transport actually moves geometry")
                if value.get("status") == "supported":
                    if value.get("method") not in {
                        "finite_bilinear_source_clearance_v1",
                        "measured_road_source_clearance_v1",
                    }:
                        raise ValueError("Unsupported terrain lifting method")
                    if value.get("method") == "measured_road_source_clearance_v1":
                        for side in ("source", "target"):
                            sample = value.get(side, {})
                            independent = sample.get("independent_surface_validation", {})
                            if independent.get("method") == "traversal_consensus_measured_road_v1":
                                from driversi.rendering.gaussian.mtgs_road_surface import (
                                    validate_traversal_consensus_evidence,
                                )

                                validate_traversal_consensus_evidence(sample)
                                continue
                            if (
                                sample.get("status") != "supported"
                                or not independent
                                or independent.get("method")
                                != "dominant_measured_road_layer_disjoint_traversal_validation_v1"
                                or any(
                                    type(independent.get(key)) not in (int, float)
                                    or not math.isfinite(independent[key])
                                    or independent[key] < 0
                                    for key in (
                                        "independent_center_difference_m",
                                        "independent_max_plane_difference_m",
                                    )
                                )
                                or independent.get("independent_center_difference_m", math.inf)
                                > 0.03
                                or independent.get("independent_max_plane_difference_m", math.inf)
                                > 0.08
                            ):
                                raise ValueError(
                                    "Measured terrain lacks independent traversal validation"
                                )
                            from driversi.rendering.gaussian.mtgs_road_surface import (
                                validate_independent_surface_evidence,
                            )

                            validate_independent_surface_evidence(sample)
                    numbers = [
                        value.get("source_clearance_m"),
                        value.get("vertical_correction_m"),
                        value.get("vertical_displacement_m"),
                    ]
                    if any(
                        type(number) not in (int, float) or not math.isfinite(number)
                        for number in numbers
                    ):
                        raise ValueError("Nonfinite terrain lifting evidence")
            support_flags.append(supported)
            path = root / name
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Missing or foreign frame")
            if output_size is not None:
                from PIL import Image

                with Image.open(path) as pixels:
                    if list(pixels.size) != output_size or item.get("output_image_size") != output_size:
                        raise ValueError("Rendered camera pixels and declared output size differ")
            images[str(path)] = digest(path)
    if report.get("terrain_fully_supported") is not all(support_flags):
        raise ValueError("Render terrain summary disagrees with per-camera evidence")
    return images


def execute(config, inputs, output, registry, *, process_runner=None):
    selected = inputs["branch"]
    branch = read_mapping(selected["artifacts"]["report"])
    sealed = {**config["files"], **branch["files"], **branch["upstream_source_files"]}
    check_files(sealed)
    protect([branch["source_log"]], registry)
    if not branch["teacher_qualified"] or branch["anchor_tick"] != 20:
        raise ValueError("Qualified same-anchor teacher branch required")
    validate_trajectory_evidence(branch["trajectory_evidence"])
    execution = Path(branch["execution"])
    scene_path = execution / "scene.npz"
    scene = Scene.load(scene_path)
    raw_log = scene.metadata["navsim_source"]["log_path"]
    if scene.metadata["split"] != "train" or raw_log not in branch["upstream_source_files"]:
        raise ValueError("Sealed training source required")
    window = Path(selected["artifacts"]["trajectory_window"])
    if digest(window) != branch["trajectory_window_sha256"]:
        raise ValueError("Selected trajectory window changed")
    output.mkdir(parents=True)
    destination = output / "render"
    future_points = 10 if config["max_render_images"] == 60 else 8
    if branch.get("future_frames") != future_points:
        raise ValueError("Selected branch and renderer native horizon differ")
    (process_runner or run_process)(
        [
            config["python"],
            "-B",
            "-m",
            "driversi.rendering.gaussian.worldengine_replay",
            "--scene",
            str(scene_path),
            "--rollout",
            str(execution / (branch["branch"] + ".npz")),
            "--bundle",
            config["bundle"],
            "--raw-log",
            raw_log,
            "--output",
            str(destination),
            "--future-points",
            str(future_points),
            "--width",
            str(config.get("width", 960)),
        ],
        output,
        config["max_wall_seconds"],
        environment={
            "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
            "CUDA_HOME": config["cuda_home"],
            "PATH": config["cuda_home"]
            + "/bin:"
            + str(Path(config["python"]).parent)
            + ":"
            + os.environ.get("PATH", ""),
        },
    )
    check_files(sealed)
    report = read_mapping(destination / "report.json")
    bundle = Path(config["bundle"])
    expected = {
        str(p.resolve()): digest(p)
        for p in (
            scene_path,
            execution / (branch["branch"] + ".npz"),
            bundle / "video_scene_dict.pkl",
            bundle / "background" / (bundle.name + ".ckpt"),
            Path(raw_log),
            bundle / "road_height_map" / "road_height_map.npy",
            bundle / "road_height_map" / "sim2.json",
        )
    }
    images = validate_render(report, destination, expected, future_points=future_points)
    if report["source_log"] != branch["source_log"] or report["asset_id"] != bundle.name:
        raise ValueError("Rendered source identity differs")
    shutil.copyfile(window, output / "trajectory-window.json")
    window_report = read_mapping(output / "trajectory-window.json")
    payload = {
        **images,
        str(destination / "report.json"): digest(destination / "report.json"),
        str(output / "trajectory-window.json"): digest(output / "trajectory-window.json"),
    }
    receipt = dict(
        kind="worldengine_inspection_window",
        branch_identity=selected["summary"]["artifact_reference"]["identity"],
        trajectory_evidence=branch["trajectory_evidence"],
        source_files=sealed,
        payload_files=payload,
        trajectory_window=window_report,
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
        terrain_scope="finite_vertical_support_not_independent_visual_or_surface_orientation_certificate",
        static_background_candidate_tokens=report.get("static_background_candidate_tokens", []),
        visual_quality_certified=False,
        trainable=False,
        rejection_reasons=[
            "independent_visual_admission_missing",
        ]
        + (
            []
            if report.get("terrain_fully_supported") is True
            else ["finite_terrain_support_missing"]
        )
        + (
            ["physical_object_appearance_unresolved"] if report["missing_appearance_tokens"] else []
        ),
    )
    return result(
        output,
        receipt,
        artifacts={
            "trajectory_window": str(output / "trajectory-window.json"),
            "render_report": str(destination / "report.json"),
        },
    )


def prepare_source(config, base, registry):
    _keys(
        config, ("python", "gpu", "cuda_home", "max_wall_seconds"), "WorldEngine source comparison"
    )
    if (
        type(config["max_wall_seconds"]) is not int
        or not 1 <= config["max_wall_seconds"] <= 600
        or type(config["gpu"]) is not int
        or config["gpu"] < 0
    ):
        raise ValueError("Explicit source render GPU and bounded wall time required")
    cuda = (base / config["cuda_home"]).resolve(strict=True)
    if not (cuda / "bin/nvcc").is_file():
        raise ValueError("CUDA compiler missing")
    return {**config, "python": str(path_at(base, config["python"])), "cuda_home": str(cuda)}, 0


def execute_source(config, inputs, output, registry, *, process_runner=None):
    rendered = inputs["render"]
    receipt = read_mapping(rendered["artifacts"]["report"])
    if receipt.get("native_future_points") not in (8, 10):
        raise ValueError("Explicit eight- or ten-future selected window required")
    check_files({**receipt["source_files"], **receipt["payload_files"]})
    output.mkdir(parents=True)
    native = receipt["render_report"].get("renderer_backend") == "mtgs_native"
    worker = "mtgs_training_window" if native else "worldengine_source"
    argument = "--source-render-report" if native else "--render-report"
    (process_runner or run_process)(
        [
            config["python"],
            "-B",
            "-m",
            "driversi.rendering.gaussian." + worker,
            argument,
            rendered["artifacts"]["render_report"],
            "--output",
            str(output / "source"),
        ],
        output,
        config["max_wall_seconds"],
        environment={
            "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
            "CUDA_HOME": config["cuda_home"],
            "PATH": config["cuda_home"]
            + "/bin:"
            + str(Path(config["python"]).parent)
            + ":"
            + os.environ.get("PATH", ""),
        },
    )
    report = read_mapping(output / "source/report.json")
    check_files(
        {
            **receipt["source_files"],
            **receipt["payload_files"],
            **report["files"],
            **report["inputs"],
            **report["producer_code"],
        }
    )
    report.update(
        measured_new_simulation_steps=0,
        render_image_count=len(report["frames"]),
        trainable=False,
        render_identity=rendered["summary"]["artifact_reference"]["identity"],
    )
    return result(output, report)


def prepare_admission(config, base, registry):
    from driversi.rendering.diagnostics.worldengine_admission import visual_policy

    _keys(
        {k: v for k, v in config.items() if k != "visual_quality_profile"},
        ("visual_review",),
        "WorldEngine independent admission",
    )
    policy = visual_policy(config.get("visual_quality_profile", "source27_v1"))
    config = dict(config)
    path = (base / config["visual_review"]).resolve(strict=True)
    config["visual_review"] = str(path)
    config["visual_quality_profile"] = policy["name"]
    config["files"] = {str(path): digest(path)}
    review = read_mapping(path)
    # These pixel masks are part of the review, not unsigned runtime inputs.
    for frame in review.get("frames", []):
        if frame.get("void_mask"):
            mask = Path(frame["void_mask"])
            if not mask.is_absolute():
                raise ValueError("Review void masks require absolute evidence paths")
            mask = mask.resolve(strict=True)
            config["files"][str(mask)] = digest(mask)
    # Supplemental scope evidence is auditable; it never replaces the checks.
    for value in review.get("supporting_evidence", []):
        evidence = Path(value)
        if not evidence.is_absolute():
            raise ValueError("Review supporting evidence requires absolute paths")
        evidence = evidence.resolve(strict=True)
        config["files"][str(evidence)] = digest(evidence)
    return config, 0


def execute_admission(config, inputs, output, registry):
    from driversi.rendering.diagnostics.worldengine_admission import (
        validate_admission,
        visual_policy,
    )
    from driversi.runtime.budget import write_json

    selected, rendered, source = inputs["branch"], inputs["render"], inputs["source"]
    branch_path = selected["artifacts"]["report"]
    branch = read_mapping(branch_path)
    render_receipt = read_mapping(rendered["artifacts"]["report"])
    identity = selected["summary"]["artifact_reference"]["identity"]
    if render_receipt["branch_identity"] != identity:
        raise ValueError("Visual admission cannot mix selected and rendered branches")
    comparison_path = source["artifacts"]["report"]
    comparison = read_mapping(comparison_path)
    if comparison["render_identity"] != rendered["summary"]["artifact_reference"]["identity"]:
        raise ValueError("Independent source render references another branch")
    protect([branch["source_log"]], registry)
    render_path = rendered["artifacts"]["render_report"]
    report = read_mapping(render_path)
    window = selected["artifacts"]["trajectory_window"]
    files = {
        **config["files"],
        **branch["files"],
        **branch["upstream_source_files"],
        **render_receipt["payload_files"],
        **report["inputs"],
        **report.get("producer_code", {}),
        **comparison["files"],
        **comparison["inputs"],
        **comparison["producer_code"],
    }
    for path in (branch_path, render_path, window, comparison_path):
        files[str(Path(path).resolve())] = digest(path)
    check_files(files)
    anchor = next(row for row in report["frames"] if row["tick"] == 20)
    certificate = dict(
        schema_version=1,
        kind="worldengine_visual_admission",
        branch_report=branch_path,
        render_report=render_path,
        trajectory_window=window,
        source_comparison=comparison_path,
        visual_review=config["visual_review"],
        visual_quality_profile=config["visual_quality_profile"],
        branch_identity=identity,
        source_log=branch["source_log"],
        parent_token=anchor["source_token"],
        files=files,
    )
    output.mkdir(parents=True)
    path = output / "admission-input.json"
    write_json(path, certificate)
    checked = validate_admission(path)
    check_files(files)
    checked.update(
        measured_new_simulation_steps=0,
        render_image_count=0,
        source_view_psnr_threshold=visual_policy(config["visual_quality_profile"])[
            "source_psnr_min_db"
        ],
        trainable=checked["admitted"],
    )
    return result(output, checked)


def prepare_stored_source(config, base, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    _keys(config, ("source_task", "source_stage", "source_config"), "stored render comparison")
    reference = {key: config[key] for key in ("source_task", "source_stage")}
    _, _, ref = _stored_stage(reference, "worldengine_inspection_window", registry)
    prepared, cap = prepare_source(config["source_config"], base, registry)
    return {"reference": reference, "identity": ref["identity"], "source_config": prepared}, cap


def execute_stored_source(config, inputs, output, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    if inputs:
        raise ValueError("Stored source comparison requires no live dependency")
    _, rendered, ref = _stored_stage(config["reference"], "worldengine_inspection_window", registry)
    if ref["identity"] != config["identity"]:
        raise ValueError("Stored render identity changed")
    return execute_source(config["source_config"], {"render": rendered}, output, registry)


def register(registry, Provider):
    registry.register(
        Provider(
            "stored_worldengine_source_reconstruction",
            "1",
            (),
            "worldengine_source_comparison",
            prepare_stored_source,
            execute_stored_source,
            ("report",),
        )
    )
    registry.register(
        Provider(
            "worldengine_selected_branch_render",
            "2",
            (("branch", "selected_native_branch"),),
            "worldengine_inspection_window",
            prepare,
            execute,
            ("report", "trajectory_window", "render_report"),
        )
    )

    registry.register(
        Provider(
            "worldengine_visual_admission",
            "1",
            (
                ("branch", "selected_native_branch"),
                ("render", "worldengine_inspection_window"),
                ("source", "worldengine_source_comparison"),
            ),
            "worldengine_visual_admission",
            prepare_admission,
            execute_admission,
            ("report",),
        )
    )

    registry.register(
        Provider(
            "worldengine_source_reconstruction",
            "1",
            (("render", "worldengine_inspection_window"),),
            "worldengine_source_comparison",
            prepare_source,
            execute_source,
            ("report",),
        )
    )
