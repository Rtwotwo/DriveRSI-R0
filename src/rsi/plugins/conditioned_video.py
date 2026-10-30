"""Bounded conditioned-video inference in the existing DriveHarness graph."""

from pathlib import Path

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.vendor import path as dependency_path
from driversi.rendering.video.bundle import validate_bundle
from driversi.rendering.video.validation import validate_output
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files, protect, run_process


def prepare(config, base, registry):
    _keys(
        config,
        ("backend", "python", "conditions", "weights", "gpu", "max_wall_seconds"),
        "conditioned video",
    )
    if config["backend"] not in ("worlddreamer", "streetcrafter"):
        raise ValueError("Unknown native video backend")
    if type(config["gpu"]) is not int or config["gpu"] < 0:
        raise ValueError("Explicit GPU required")
    if type(config["max_wall_seconds"]) is not int or not 1 <= config["max_wall_seconds"] <= 1800:
        raise ValueError("Small-sample video wall budget required")
    value = dict(config)
    for key in ("python", "conditions", "weights"):
        value[key] = str(path_at(base, config[key]))
    conditions = read_mapping(value["conditions"])
    if conditions["kind"] != config["backend"] + "_native_conditions":
        raise ValueError("Backend condition contract differs")
    geometry = read_mapping(conditions["geometry"])
    bundle = validate_bundle(geometry["bundle"])
    protect([bundle["source_log"]], registry)
    files = {
        value["conditions"]: digest(value["conditions"]),
        **conditions["files"],
        **geometry["files"],
    }
    weights = Path(value["weights"])
    if not weights.exists():
        raise ValueError("Official video weights missing: " + str(weights))
    if weights.is_file():
        files[str(weights)] = digest(weights)
    else:
        for p in sorted(weights.rglob("*")):
            if p.is_file() and not any(x.startswith(".") for x in p.relative_to(weights).parts):
                files[str(p)] = digest(p)
    for folder in (
        project_root() / "src/rendering/video" / config["backend"],
        project_root() / "src/rendering/video",
        dependency_path(config["backend"]),
    ):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".yaml"):
                files[str(p)] = digest(p)
    plugin = Path(__file__).resolve()
    files[str(plugin)] = digest(plugin)
    value["files"] = files
    value["source_groups"] = [bundle["source_log"]]
    return value, 0


def execute(config, inputs, output, registry):
    check_files(config["files"])
    selected = inputs["branch"]
    branch = read_mapping(selected["artifacts"]["report"])
    condition = read_mapping(config["conditions"])
    geometry = read_mapping(condition["geometry"])
    bundle = validate_bundle(geometry["bundle"])
    if bundle["branch_identity"] != digest(selected["artifacts"]["report"]):
        raise ValueError("Conditions are not from the selected Harness branch")
    protect([branch["source_log"]], registry)
    if not output.resolve().is_relative_to(project_root() / "exp"):
        raise ValueError("Video outputs must live in exp")
    output.mkdir(parents=True, exist_ok=False)
    worker = project_root() / "src/rendering/video" / config["backend"] / "runtime.py"
    flag = "--weights" if config["backend"] == "worlddreamer" else "--checkpoint"
    run_process(
        [
            config["python"],
            "-B",
            str(worker),
            "--prepared",
            config["conditions"],
            flag,
            config["weights"],
            "--output",
            str(output / "video"),
        ],
        output,
        config["max_wall_seconds"],
        environment={
            "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    check_files(config["files"])
    rendered, _, images = validate_output(output / "video/report.json")
    return result(
        output,
        dict(
            kind="conditioned_video_inspection_window",
            backend=config["backend"],
            branch_identity=selected["summary"]["artifact_reference"]["identity"],
            physical_branch_sha256=bundle["branch_identity"],
            files=config["files"],
            image_files=images,
            source_control=rendered["source_control"],
            history_ticks=rendered["history_ticks"],
            future_ticks=rendered["future_ticks"],
            generated_images=len(images),
            trainable=False,
            visual_admitted=False,
            rejection_reasons=["independent_generated_pixel_admission_required"],
        ),
        artifacts={"render_report": str(output / "video/report.json")},
    )


def prepare_stored(config, base, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    _keys(config, ("branch", "render"), "stored conditioned video")
    state, original, reference = _stored_stage(config["branch"], "selected_native_branch", registry)
    native, cap = prepare(config["render"], base, registry)
    geometry = read_mapping(read_mapping(native["conditions"])["geometry"])
    bundle = validate_bundle(geometry["bundle"])
    if bundle["branch_identity"] != digest(original["artifacts"]["report"]):
        raise ValueError("Prepared conditions do not match stored physical branch")
    return dict(
        native=native,
        branch={**config["branch"], "identity": reference["identity"]},
        source_groups=sorted(set(state["task"]["source_groups"] + native["source_groups"])),
    ), cap


def execute_stored(config, inputs, output, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    if inputs:
        raise ValueError("Stored renderer accepts only sealed references")
    item = config["branch"]
    _, original, reference = _stored_stage(
        {key: item[key] for key in ("source_task", "source_stage")},
        "selected_native_branch",
        registry,
    )
    if reference["identity"] != item["identity"]:
        raise ValueError("Stored physical branch changed after registration")
    return execute(config["native"], {"branch": original}, output, registry)


def prepare_quality(config, base, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    _keys(config, ("novel", "source", "python", "max_wall_seconds"), "stored video quality")
    if type(config["max_wall_seconds"]) is not int or not 1 <= config["max_wall_seconds"] <= 300:
        raise ValueError("Bounded image quality budget required")
    references, groups, files = {}, set(), {}
    for side in ("novel", "source"):
        state, original, ref = _stored_stage(
            config[side], "conditioned_video_inspection_window", registry
        )
        rendered, _, images = validate_output(original["artifacts"]["render_report"])
        if rendered["source_control"] != (side == "source"):
            raise ValueError("Video quality source/novel roles differ")
        references[side] = {**config[side], "identity": ref["identity"]}
        groups.update(state["task"]["source_groups"])
        files.update(images)
    for name in (
        "src/rendering/video/worker.py",
        "src/rendering/video/validation.py",
        "src/rsi/plugins/conditioned_video.py",
    ):
        p = project_root() / name
        files[str(p)] = digest(p)
    return dict(
        references=references,
        source_groups=sorted(groups),
        files=files,
        python=str(path_at(base, config["python"])),
        max_wall_seconds=config["max_wall_seconds"],
    ), 0


def execute_quality(config, inputs, output, registry):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    if inputs:
        raise ValueError("Stored video quality accepts sealed references only")
    check_files(config["files"])
    paths = {}
    for side, item in config["references"].items():
        _, original, ref = _stored_stage(
            {key: item[key] for key in ("source_task", "source_stage")},
            "conditioned_video_inspection_window",
            registry,
        )
        if ref["identity"] != item["identity"]:
            raise ValueError("Video artifact changed after quality registration")
        paths[side] = original["artifacts"]["render_report"]
    if not output.resolve().is_relative_to(project_root() / "exp"):
        raise ValueError("Video quality outputs must live in exp")
    output.mkdir(parents=True, exist_ok=False)
    run_process(
        [
            config["python"],
            "-B",
            str(project_root() / "src/rendering/video/worker.py"),
            "--novel",
            paths["novel"],
            "--source",
            paths["source"],
            "--output",
            str(output / "quality.json"),
        ],
        output,
        config["max_wall_seconds"],
    )
    check_files(config["files"])
    report = read_mapping(output / "quality.json")
    if report.get("admitted") is not False:
        raise ValueError("Diagnostic image evaluator cannot authorize synthetic training")
    report.update(
        source_references=config["references"],
        training_started=False,
        promoted=False,
        failure_domain="visual_generation",
        next_action="repair_or_replace_visual_provider_before_training",
    )
    return result(
        output,
        report,
        status="rejected",
        reason="independent_video_visual_admission_not_passed",
        artifacts={"quality": str(output / "quality.json")},
    )


def register(registry, Provider):
    registry.register(
        Provider(
            "conditioned_video_stored_quality",
            "1",
            (),
            "conditioned_video_quality_feedback",
            prepare_quality,
            execute_quality,
            ("report", "quality"),
        )
    )
    registry.register(
        Provider(
            "conditioned_video_render",
            "1",
            (("branch", "selected_native_branch"),),
            "conditioned_video_inspection_window",
            prepare,
            execute,
            ("report", "render_report"),
        )
    )
    registry.register(
        Provider(
            "conditioned_video_stored_branch",
            "1",
            (),
            "conditioned_video_inspection_window",
            prepare_stored,
            execute_stored,
            ("report", "render_report"),
        )
    )
