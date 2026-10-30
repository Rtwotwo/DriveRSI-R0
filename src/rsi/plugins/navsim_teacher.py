"""Native NAVSIM teacher skill. Sealed execution precedes separate data admission."""

from pathlib import Path

from driversi.agents.backends.native import digest, path_at
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.rsi.workflow.results import result


def prepare(config, base, registry):
    config = {
        "teacher_backend": "driverl_native",
        "recovery_seconds": 2.0,
        "position_gain": 0.0,
        "arrival_backend": "bounded_control",
        "seed": 42,
        "sampling_method": "argmax",
        "admission_profile": "strict_world95_v1",
        "history_perturbation": None,
        **config,
    }
    _keys(
        config,
        (
            "history_perturbation",
            "teacher_backend",
            "seed",
            "admission_profile",
            "recovery_seconds",
            "position_gain",
            "scene",
            "road_graph",
            "python",
            "checkpoint",
            "maps_root",
            "arrival_action",
            "arrival_backend",
            "search_candidates",
            "sampling_method",
            "device",
            "max_wall_seconds",
        ),
        "NAVSIM native teacher",
    )
    c = dict(config)
    from driversi_agents.driverl.random_state import validate_sampling
    validate_sampling(c["sampling_method"], search_candidates=c["search_candidates"])
    if c["sampling_method"] == "sample" and (c["arrival_backend"] != "driverl_native_causal_history" or c["teacher_backend"] != "driverl_native"):
        raise ValueError("Stochastic native generation requires full-window native control")
    from driversi.rsi.correction.trajectory_admission import trajectory_policy

    trajectory_policy(c["admission_profile"])
    if c["arrival_backend"] not in (
        "bounded_control",
        "driverl_native",
        "driverl_native_causal_history",
        "source_history_perturbation_v1",
    ):
        raise ValueError("Unknown arrival backend")
    if c["arrival_backend"] == "source_history_perturbation_v1":
        from driversi.rsi.correction.history_perturbation import validate_perturbation

        c["history_perturbation"] = validate_perturbation(c["history_perturbation"])
        if c["teacher_backend"] != "driverl_native" or c["search_candidates"] not in (8, 64):
            raise ValueError("Historical perturbation requires native future TTS")
    elif c["history_perturbation"] is not None:
        raise ValueError("Historical proposal supplied to a different backend")
    if c["teacher_backend"] not in (
        "driverl_native",
        "pdm_route_v1",
        "bounded_native_search_v1",
        "source_corridor_lqr_v1",
    ):
        raise ValueError("Unknown same-anchor teacher backend")
    if c["teacher_backend"] != "driverl_native" and c["search_candidates"] != 0:
        raise ValueError("Alternate teacher cannot mix DriveRL TTS")
    if (
        type(c["recovery_seconds"]) not in (int, float)
        or c["recovery_seconds"] not in (2.0, 3.0)
        or (c["teacher_backend"] != "source_corridor_lqr_v1" and c["recovery_seconds"] != 2.0)
    ):
        raise ValueError("Recovery duration requires declared source-corridor backend")
    if (
        type(c["position_gain"]) not in (int, float)
        or c["position_gain"] not in (0.0, 0.5, 1.0)
        or (c["teacher_backend"] != "source_corridor_lqr_v1" and c["position_gain"] != 0.0)
    ):
        raise ValueError("Position gain requires declared source-corridor backend")
    for k in ("scene", "road_graph", "checkpoint", "maps_root"):
        c[k] = str((base / config[k]).resolve(strict=True))
    c["python"] = str(path_at(base, config["python"]))
    scene = Scene.load(Path(c["scene"]))
    protect([scene.metadata["source_log"]], registry)
    if scene.metadata["split"] != "train" or "navsim_source" not in scene.metadata:
        raise ValueError("NAVSIM training source required")
    if len(scene.times) != 71 or tuple(
        a for a, v in zip(scene.actor_ids, scene.controlled) if v
    ) != ("ego",):
        raise ValueError("Bounded two-second arrival and five-second ego correction required")
    from driversi_agents.driverl.search import search_window

    search_window(
        c["search_candidates"],
        20 if c["search_candidates"] else None,
        70 if c["search_candidates"] else None,
        seed=c["seed"],
    )
    if type(c["max_wall_seconds"]) is not int or not 310 <= c["max_wall_seconds"] <= 1800:
        raise ValueError("Bounded wall budget required")
    import math

    if (
        not isinstance(c["arrival_action"], list)
        or len(c["arrival_action"]) != 2
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 0.25
            for v in c["arrival_action"]
        )
    ):
        raise ValueError("Arrival perturbation must be bounded native raw actions")
    if c["arrival_backend"] in (
        "driverl_native",
        "driverl_native_causal_history",
        "source_history_perturbation_v1",
    ) and any(v != 0 for v in c["arrival_action"]):
        raise ValueError("Native arrival requires zero raw-action placeholder")
    if c["arrival_backend"] in ("driverl_native_causal_history", "source_history_perturbation_v1"):
        if (
            c["teacher_backend"] != "driverl_native"
            or scene.metadata.get("driverl_history_mode") != "causal_resampled_10hz_v1"
        ):
            raise ValueError("Full-window DriveRL history/controller contract differs")
    c["files"] = {c[k]: digest(c[k]) for k in ("scene", "road_graph", "checkpoint")}
    source = scene.metadata["navsim_source"]["log_path"]
    c["files"][source] = digest(source)
    if c["teacher_backend"] == "source_corridor_lqr_v1":
        from driversi.agents.backends.native import project_root

        for relative in (
            "agents/driverl/source_corridor.py",
            "agents/driverl/repair.py",
            "agents/driverl/control_lattice.py",
            "agents/driverl/closed_loop.py",
            "src/evaluation/search/navsim_teacher.py",
        ):
            path = project_root() / relative
            c["files"][str(path)] = digest(path)
    if c["arrival_backend"] == "source_history_perturbation_v1":
        from driversi.agents.backends.native import project_root

        for relative in (
            "agents/driverl/history_perturbation.py",
            "agents/driverl/closed_loop.py",
            "agents/driverl/repair.py",
            "agents/driverl/control_lattice.py",
            "src/rsi/correction/history_perturbation.py",
            "src/evaluation/search/navsim_teacher.py",
        ):
            path = project_root() / relative
            c["files"][str(path)] = digest(path)
    from driversi.rsi.correction.epdms_admission import required_for
    from driversi.agents.backends.native import project_root
    if c["sampling_method"] == "sample":
        for relative in ("agents/driverl/closed_loop.py", "agents/driverl/navsim_context.py", "agents/driverl/random_state.py"):
            path = project_root() / relative
            c["files"][str(path)] = digest(path)
    c["capture_planning"] = required_for(base)
    if c["capture_planning"]:
        for relative in ("src/evaluation/search/navsim_teacher.py",
                         "src/evaluation/search/driverl_planning.py"):
            path = project_root() / relative
            c["files"][str(path)] = digest(path)
    c["source_groups"] = [scene.metadata["source_log"]]
    c["actor_count"] = len(scene.actor_ids)
    return c, ((575 if c["teacher_backend"] == "bounded_native_search_v1" else 125)
               + (80 if c["capture_planning"] else 0)) * len(
        scene.actor_ids
    )


def execute(config, inputs, output, registry):
    check_files(config["files"])
    protect(config["source_groups"], registry)
    output.mkdir(parents=True)
    args = [
        config["python"],
        "-B",
        "-m",
        "driversi.evaluation.search.navsim_teacher",
        "--scene",
        config["scene"],
        "--road-graph",
        config["road_graph"],
        "--output",
        str(output / "execution"),
        "--checkpoint",
        config["checkpoint"],
        "--maps-root",
        config["maps_root"],
        "--seed",
        str(config.get("seed", 42)),
        "--device",
        config["device"],
        "--sampling-method",
        config.get("sampling_method", "argmax"),
        "--search-candidates",
        str(config["search_candidates"]),
        "--teacher-backend",
        config["teacher_backend"],
        "--recovery-seconds",
        str(config["recovery_seconds"]),
        "--position-gain",
        str(config["position_gain"]),
        "--arrival-action",
        *[str(v) for v in config["arrival_action"]],
        "--arrival-backend",
        config["arrival_backend"],
        "--admission-profile",
        config.get("admission_profile", "strict_world95_v1"),
        "--max-wall-seconds",
        str(config["max_wall_seconds"] - 10),
    ]
    if config.get("history_perturbation") is not None:
        import json

        args += [
            "--history-perturbation",
            json.dumps(config["history_perturbation"], allow_nan=False),
        ]
    try:
        run_process(args, output, config["max_wall_seconds"])
    except Exception:
        # A partial JSON receipt does not prove process completion or permit retry.
        raise
    check_files(config["files"])
    report = read_mapping(output / "execution/report.json")
    artifacts = {
        str(p.relative_to(output)): str(p) for p in (output / "execution").iterdir() if p.is_file()
    }
    report["source_files"] = config["files"]
    return result(
        output,
        report,
        artifacts=artifacts,
        steps=report.get("charged_joint_steps", 0) * config["actor_count"],
        status="completed" if report["status"] == "completed" else "rejected",
        reason=None
        if report["status"] == "completed"
        else report.get("reason", "incomplete_execution"),
    )


def _completed_teacher(config, registry):
    from driversi.rsi.workflow.artifacts import verify
    from driversi.rsi.workflow.store import HarnessStore

    state = HarnessStore(registry).get(config["source_task"])
    declaration = next(
        (x for x in state["task"]["steps"] if x["id"] == config["source_stage"]), None
    )
    stage = next((x for x in state["stages"] if x["id"] == config["source_stage"]), None)
    if (
        not declaration
        or not stage
        or stage["status"] != "completed"
        or declaration["output_kind"] != "navsim_teacher_execution"
    ):
        raise ValueError("Completed sealed teacher execution required")
    ref = verify(stage["result"], state["id"], declaration)
    protect(state["task"]["source_groups"], registry)
    report = read_mapping(stage["result"]["artifacts"]["report"])
    if report["status"] != "completed" or not report.get("restore_exact"):
        raise ValueError("Completed reproducible teacher report required")
    check_files(report["source_files"])
    return state, stage["result"], ref


def prepare_reference(config, base, registry):
    _keys(config, ("source_task", "source_stage"), "completed teacher reference")
    state, _, ref = _completed_teacher(config, registry)
    return {
        **config,
        "identity": ref["identity"],
        "source_groups": state["task"]["source_groups"],
    }, 0


def run_reference(config, inputs, output, registry):
    import shutil

    state, original, ref = _completed_teacher(config, registry)
    if ref["identity"] != config["identity"]:
        raise ValueError("Referenced teacher changed after registration")
    output.mkdir(parents=True)
    artifacts = {}
    for name, source in original["artifacts"].items():
        relative = Path(name + ".json") if name == "report" else Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Teacher artifact name is not relative")
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if digest(dest) != ref["files"][name]["sha256"]:
            raise ValueError("Copied teacher artifact differs")
        artifacts[name] = str(dest)
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": 0,
        "artifacts": artifacts,
        "summary": {
            "reused": True,
            "origin_identity": ref["identity"],
            "source_task": state["id"],
            "new_simulation_steps": 0,
            "original_sim_agent_steps": original["sim_agent_steps"],
        },
        "admission": {"trainable": False},
    }


def register(registry, Provider):
    registry.register(
        Provider(
            "completed_navsim_teacher_reference",
            "1",
            (),
            "navsim_teacher_execution",
            prepare_reference,
            run_reference,
            ("report",),
        )
    )
    registry.register(
        Provider(
            "navsim_native_teacher_probe",
            "5",
            (),
            "navsim_teacher_execution",
            prepare,
            execute,
            ("report",),
        )
    )
