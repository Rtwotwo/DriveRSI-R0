"""Sealed native execution -> exact GS replay -> NAVSIM inspection dataset.

Rejected teacher branches remain renderable for diagnosis. Neither a completed
render nor an exported inspection package grants a training-data capability.
"""

import os
import shutil
from pathlib import Path

from driversi.agents.backends.native import digest, path_at
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files, protect, run_process


def prepare_select(config, base, registry):
    config = {"future_frames": 8, **config}
    _keys(config, ("branch", "future_frames"), "native branch selection")
    if type(config["future_frames"]) is not int or config["future_frames"] not in (8, 10):
        raise ValueError("Native branch requires eight or ten actual future points")
    if config["branch"] not in ("baseline", "teacher"):
        raise ValueError("Explicit recorded native branch required")
    return dict(config), 0


def run_select(config, inputs, output, registry):
    upstream = inputs["world"]
    report = read_mapping(upstream["artifacts"]["report"])
    if report.get("status") != "completed" or not report.get("restore_exact"):
        raise ValueError("Complete reproducible native teacher execution required")
    check_files(report["source_files"])
    branch = config["branch"]
    execution = Path(upstream["artifacts"]["execution/scene.npz"]).parent
    scene = Scene.load(execution / "scene.npz")
    protect([scene.metadata["source_log"]], registry)
    if scene.metadata["split"] != "train" or not report.get("history_changed"):
        raise ValueError("Changed-history training-source branch required")
    row = report["branches"][branch]
    if row["completed_steps"] != 50 or row["failure"] is not None:
        from driversi.rsi.decisions.generation_outcomes import incomplete_physical_branch
        if incomplete_physical_branch(report, branch):
            output.mkdir(parents=True)
            return result(output, dict(kind="rejected_native_branch", branch=branch,
                native_report_sha256=digest(upstream["artifacts"]["report"]),
                completed_steps=row["completed_steps"], events=row["events"],
                trainable=False, trajectory_window_available=False),
                status="failed", reason="incomplete_physical_branch")
        raise ValueError("Incomplete branch cannot supply a complete RGB window")
    paths = [
        execution / n
        for n in ("scene.npz", "road-graph.json", branch + ".npz", branch + "-targets.json")
    ]
    cache_context = execution / "teacher-counterfactual-context.pkl"
    if branch == "teacher" and config.get("future_frames", 8) == 10:
        context_ref = row.get("counterfactual_context", {})
        if not cache_context.is_file() or digest(cache_context) != context_ref.get("sha256"):
            raise ValueError("Ten-point teacher transport requires its exact native context")
        paths.append(cache_context)
    output.mkdir(parents=True)
    from driversi.data.datasets.executed_window import export_executed_window

    window = output / "trajectory-window.json"
    export_executed_window(
        execution / "scene.npz",
        execution / (branch + ".npz"),
        window,
        anchor_tick=report["arrival_tick"],
        future_frames=config.get("future_frames", 8),
    )
    return result(
        output,
        dict(
            kind="selected_native_branch",
            branch=branch,
            execution=str(execution),
            files={str(p): digest(p) for p in paths},
            anchor_tick=report["arrival_tick"],
            future_frames=config.get("future_frames", 8),
            teacher_qualified=bool(branch == "teacher" and row["trajectory_admitted"]),
            trajectory_evidence={
                "teacher": report.get("teacher"),
                "branch": branch,
                "score": row["score"],
                "admission_profile": report.get("admission_profile", "strict_world95_v1"),
                "failure": row["failure"],
                "completed_steps": row["completed_steps"],
                "events": row["events"],
                "arrival_contact_records": len(report.get("arrival_contacts", [])),
                "future_contact_records": len(row["object_contacts"]),
                "history_changed": report["history_changed"],
                "restore_exact": report["restore_exact"],
                "producer_identity": upstream["summary"]["artifact_reference"]["identity"],
            },
            render_purpose="diagnostic_until_visual_admission",
            trainable=False,
            upstream_source_files=report["source_files"],
            execution_identity=upstream["summary"]["artifact_reference"]["identity"],
            source_log=scene.metadata["source_log"],
            trajectory_window_sha256=digest(window),
        ),
        artifacts={"trajectory_window": str(window)},
    )


def prepare_render(config, base, registry):
    _keys(
        config,
        ("config", "python", "gpu", "cuda_home", "max_wall_seconds", "max_sim_agent_steps"),
        "native GS replay",
    )
    if type(config["max_wall_seconds"]) is not int or not 1 <= config["max_wall_seconds"] <= 3600:
        raise ValueError("Bounded native GS wall time required")
    if type(config["max_sim_agent_steps"]) is not int or config["max_sim_agent_steps"] <= 0:
        raise ValueError("Explicit replay step budget required")
    if type(config["gpu"]) is not int or config["gpu"] < 0:
        raise ValueError("Explicit GPU index required")
    p = (base / config["config"]).resolve(strict=True)
    renderer = read_mapping(p)
    files = {str(p): digest(p)}
    for key in ("binding", "config_path", "checkpoint_path", "road_block_config", "native_path"):
        files[renderer[key]] = digest(renderer[key])
    cuda = Path(config["cuda_home"]).resolve(strict=True)
    if not (cuda / "bin/nvcc").is_file():
        raise ValueError("CUDA compiler missing")
    return {
        **config,
        "config": str(p),
        "python": str(path_at(base, config["python"])),
        "cuda_home": str(cuda),
        "files": files,
    }, config["max_sim_agent_steps"]


def run_render(config, inputs, output, registry):
    source = inputs["branch"]
    branch = read_mapping(source["artifacts"]["report"])
    check_files({**config["files"], **branch["files"], **branch["upstream_source_files"]})
    protect([branch["source_log"]], registry)
    scene = Scene.load(Path(branch["execution"]) / "scene.npz")
    # Both the full TTC replay and requested RGB replay count against budget.
    binding = read_mapping(read_mapping(config["config"])["binding"])
    end = binding.get("render_end_tick")
    end = len(scene.times) - 1 if end is None else end
    steps = (len(scene.times) - 1 + end) * len(scene.actor_ids)
    if steps > config["max_sim_agent_steps"]:
        raise ValueError("Exact replays exceed reservation")
    output.mkdir(parents=True)
    renderer = read_mapping(config["config"])
    renderer.update(
        scene=str(Path(branch["execution"]) / "scene.npz"),
        road_graph=str(Path(branch["execution"]) / "road-graph.json"),
    )
    import json

    local_config = output / "renderer.json"
    local_config.write_text(json.dumps(renderer, indent=2))
    run_process(
        [
            config["python"],
            "-B",
            "-m",
            "driversi.rendering.execution.native_replay",
            "--config",
            str(local_config),
            "--execution",
            branch["execution"],
            "--branch",
            branch["branch"],
            "--output",
            str(output / "render"),
        ],
        output,
        config["max_wall_seconds"],
        environment={
            "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
            "CUDA_HOME": config["cuda_home"],
            "PATH": config["cuda_home"] + "/bin:" + os.environ.get("PATH", ""),
        },
    )
    check_files({**config["files"], **branch["files"], **branch["upstream_source_files"]})
    verification = read_mapping(output / "render/report.json")
    if (
        verification["status"] != "completed"
        or verification["verified_cpu_steps"] != len(scene.times) - 1
        or verification["verified_render_steps"] != end
    ):
        raise ValueError("Incomplete native render verification")
    scene_path = output / "scene.npz"
    shutil.copyfile(renderer["scene"], scene_path)
    session = output / "render/session"
    report = dict(
        kind="selected_native_branch_render",
        upstream_source_files=branch["upstream_source_files"],
        verification=verification,
        branch_identity=source["summary"]["artifact_reference"]["identity"],
        anchor_tick=branch["anchor_tick"],
        teacher_qualified=branch["teacher_qualified"],
        trajectory_evidence=branch["trajectory_evidence"],
        asset_preflight=verification.get("asset_preflight"),
        registration_mode=renderer.get("registration_mode", "legacy-map-axes"),
        visual_quality_certified=False,
        trainable=False,
        render_purpose=branch["render_purpose"],
        payload_files={str(p): digest(p) for p in session.rglob("*") if p.is_file()},
    )
    report["payload_files"][str(scene_path)] = digest(scene_path)
    return result(
        output,
        report,
        steps=steps,
        artifacts=dict(
            session=str(session / "manifest.json"),
            scene=str(scene_path),
            viewer=str(session / "viewer.html"),
            asset_preflight=str(output / "render/asset-preflight.json"),
            asset_catalog=str(output / "render/asset-catalog.json"),
        ),
    )


def run_export(config, inputs, output, registry):
    from driversi.data.datasets.navsim_synthetic import export_navsim_sample
    from driversi.data.sources.navsim import load_log
    from driversi.runtime.budget import write_json

    rendered = inputs["rendered"]
    receipt = read_mapping(rendered["artifacts"]["report"])
    check_files({**receipt["payload_files"], **receipt["upstream_source_files"]})
    scene = Scene.load(Path(rendered["artifacts"]["scene"]))
    protect([scene.metadata["source_log"]], registry)
    context = scene.metadata["navsim_source"]
    path = Path(context["log_path"])
    if (
        scene.metadata["split"] != "train"
        or "nuplan_source" in scene.metadata
        or str(path) not in receipt["upstream_source_files"]
    ):
        raise ValueError("Sealed NAVSIM training source required for conditioning")
    frames = load_log(path, trusted=True)
    initial = frames[scene.metadata["source_start_frame"]]
    if (
        initial["token"] != scene.metadata["scenario_id"]
        or initial["log_name"] != scene.metadata["source_log"]
        or int(initial["timestamp"]) != context["anchor_timestamp_us"]
    ):
        raise ValueError("Source navigation anchor identity mismatch")
    navigation = {
        "roadblock_ids": [str(x) for x in initial["roadblock_ids"]],
        "driving_command": initial["driving_command"].tolist(),
    }
    timeline = [
        dict(
            timestamp_us=int(context["anchor_timestamp_us"] + round(v["time"] * 1e6)),
            records=[[str(i), bool(red)] for i, red in v["records"]],
            available=v.get("available", False),
        )
        for v in scene.metadata["traffic_lights"]
    ]
    package = export_navsim_sample(
        rendered["artifacts"]["session"],
        rendered["artifacts"]["scene"],
        output,
        anchor_tick=receipt["anchor_tick"],
        cameras=tuple(config["cameras"]),
        include_future_images=True,
        navigation=navigation,
        signal_timeline=timeline,
    )
    check_files({**receipt["payload_files"], **receipt["upstream_source_files"]})
    evidence = receipt["trajectory_evidence"]
    package["trajectory_evidence"] = evidence
    package["counterfactual_score"] = evidence["score"]
    if receipt["teacher_qualified"]:
        validate_trajectory_evidence(evidence)
        package["rejection_reasons"] = [
            reason
            for reason in package["rejection_reasons"]
            if reason not in ("generator_not_approved_teacher", "independent_teacher_score_missing")
        ]
    package["conditioning"] = {
        "navigation": "frozen_source_task_not_teacher_future",
        "signals": "latest_causal_source_record",
        "source_log_sha256": receipt["upstream_source_files"][str(path)],
    }
    package["rsi_usage"] = {
        "history": "student_input",
        "future_rgb": "supervision_only",
        "return_pool": "navtrain_after_independent_admission",
        "training_started": False,
    }
    write_json(output / "manifest.json", package)
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": {"report": str(output / "manifest.json")},
        "summary": {
            "render_identity": rendered["summary"]["artifact_reference"]["identity"],
            "image_counts": package["image_counts"],
            "rejection_reasons": package["rejection_reasons"],
        },
        "admission": {
            "trainable": False,
            "visual": "inspection_only",
            "trajectory": "qualified" if receipt["teacher_qualified"] else "unqualified",
        },
    }


def register(registry, Provider):
    from driversi.rsi.plugins.navsim import prepare_export

    for provider in (
        Provider(
            "native_branch_selection",
            "3",
            (("world", "navsim_teacher_execution"),),
            "selected_native_branch",
            prepare_select,
            run_select,
            ("report", "trajectory_window"),
        ),
        Provider(
            "native_branch_gs",
            "2",
            (("branch", "selected_native_branch"),),
            "rendered_native_branch",
            prepare_render,
            run_render,
            ("report", "session", "scene", "viewer"),
        ),
        Provider(
            "native_branch_history_future_export",
            "2",
            (("rendered", "rendered_native_branch"),),
            "navsim_inspection_dataset",
            prepare_export,
            run_export,
            ("report",),
        ),
    ):
        registry.register(provider)
