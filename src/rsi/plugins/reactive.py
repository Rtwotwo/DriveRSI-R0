"""Artifact-bound reactive world -> selected execution -> GS -> inspection export.

Reactive admission is deliberately distinct from logged-world DriveTTO scores.
The existing geometry and visual gates remain in charge; a finished render is
not evidence of a qualified teacher or of a usable visual training sample.
"""

from pathlib import Path

from driversi.agents.backends.native import digest, path_at
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.runtime.budget import write_json


def prepare_world(config, base, registry):
    _keys(
        config,
        (
            "scene",
            "road_graph",
            "log",
            "sensor_root",
            "explore_steps",
            "explore_speed",
            "desired_speed",
        ),
        "reactive synthesis skill",
    )
    config = dict(config)
    for key in ("scene", "road_graph", "log", "sensor_root"):
        config[key] = str((base / config[key]).resolve(strict=True))
    scene = Scene.load(Path(config["scene"]))
    from driversi.training.sampling.environment import validate_development_split

    validate_development_split(scene.metadata["split"])
    protect([scene.metadata["source_log"]], registry)
    if type(config["explore_steps"]) is not int or not 15 <= config["explore_steps"] <= 40:
        raise ValueError("A full four-frame reached history requires a bounded exploration prefix")
    if any(
        type(config[k]) not in (int, float) or not 1 <= config[k] <= 15
        for k in ("explore_speed", "desired_speed")
    ):
        raise ValueError("Bounded physical speed intents required")
    config.update(
        files={config[k]: digest(config[k]) for k in ("scene", "road_graph", "log")},
        source_groups=[scene.metadata["source_log"]],
        actor_capacity=len(scene.actor_ids),
    )
    return config, 200 * len(scene.actor_ids)


def run_world(config, inputs, output, registry):
    from driversi.evaluation.search.augmentation import run

    check_files(config["files"])
    protect(config["source_groups"], registry)
    report = run(
        config["scene"],
        config["road_graph"],
        config["log"],
        config["sensor_root"],
        output,
        explore_steps=config["explore_steps"],
        explore_speed=config["explore_speed"],
        desired_speed=config["desired_speed"],
        cameras=("CAM_F0", "CAM_L0", "CAM_R0"),
        trusted=True,
    )
    artifacts = {"bundle": str(output / "report.json")}
    artifacts.update(
        {
            p.name: str(p)
            for p in output.iterdir()
            if p.is_file() and p.suffix in (".npz", ".json", ".html")
        }
    )
    # Keep report.json intact: renderer validates its exact native source schema.
    write_json(
        output / "receipt.json",
        {
            "kind": "reactive_synthesis",
            "world": report,
            "source_groups": config["source_groups"],
            "trainable": False,
        },
    )
    return {
        "status": "completed",
        "reason": None,
        "sim_agent_steps": report["charged_joint_steps"]
        * (1 + len(report["other_controlled_actors"])),
        "artifacts": {**artifacts, "report": str(output / "receipt.json")},
        "summary": {"responding_actors": report["responding_actors"], "trainable": False},
        "admission": {"trainable": False},
    }


def prepare_select(config, base, registry):
    _keys(config, ("branch",), "reactive branch selection")
    if config["branch"] not in ("baseline", "challenge"):
        raise ValueError("Explicit completed branch required")
    return config, 0


def run_select(config, inputs, output, registry):
    world = inputs["world"]
    bundle = Path(world["artifacts"]["bundle"])
    report = read_mapping(bundle)
    branch = config["branch"]
    if report.get("export_status") != "complete":
        raise ValueError("Incomplete reactive execution")
    row = report[branch]
    source = report["branches"][branch]
    rollout = bundle.parent / source["rollout"]
    if digest(rollout) != source["sha256"]:
        raise ValueError("Selected execution changed")
    from driversi.contracts.transition import Snapshot
    from driversi.rsi.correction.reached_state import verify_reached_state
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    snapshot_path = bundle.parent / source["stage_snapshot"]["path"]
    if digest(snapshot_path) != source["stage_snapshot"]["sha256"]:
        raise ValueError("Reached-state snapshot changed")
    scene_path = bundle.parent / report["input_artifacts"]["scene.npz"]["path"]
    if digest(scene_path) != report["input_artifacts"]["scene.npz"]["sha256"]:
        raise ValueError("Reached-state scene changed")
    scene = Scene.load(scene_path)
    physical = PufferDriveWorld(scene)
    try:
        physical.restore(Snapshot.load(snapshot_path))
        reached = verify_reached_state(
            scene,
            rollout,
            physical,
            report["phases"]["exploration_end_tick"],
            actor=report["actor"],
        )
    finally:
        physical.close()
    reached.update(
        world_snapshot=str(snapshot_path),
        producer="rule_perturbation",
        policy_memory="stateless_route_follower",
        learned_self_play=False,
    )
    output.mkdir(parents=True)
    write_json(output / "recovery-task.json", reached)
    return result(
        output,
        {
            "kind": "selected_reactive_branch",
            "branch": branch,
            "bundle": str(bundle),
            "bundle_sha256": digest(bundle),
            "world_identity": world["summary"]["artifact_reference"]["identity"],
            "files": {path: digest(path) for path in world["artifacts"].values()},
            "anchor_tick": report["phases"]["exploration_end_tick"],
            "engineering_pass": not row["bounds_failures"]
            and row["collision_ticks"] == 0
            and row["offroad_ticks"] == 0,
            "metrics": row,
            "teacher_qualified": False,
            "official_score": None,
            "render_purpose": "diagnostic_only",
            "trainable": False,
            "reached_state": reached,
        },
        artifacts={"recovery_task": str(output / "recovery-task.json")},
    )


def prepare_render(config, base, registry):
    _keys(
        config,
        ("config", "python", "gpu", "max_wall_seconds", "max_sim_agent_steps"),
        "selected branch GS",
    )
    if type(config["gpu"]) is not int or config["gpu"] < 0:
        raise ValueError("Explicit nonnegative render GPU required")
    if (
        type(config["max_sim_agent_steps"]) is not int
        or not 0 < config["max_sim_agent_steps"] <= 100000
        or not 0 < config["max_wall_seconds"] <= 3600
    ):
        raise ValueError("Explicit render/replay budget required")
    path = (base / config["config"]).resolve(strict=True)
    renderer = read_mapping(path)
    files = {str(path): digest(path)}
    for key in (
        "scene",
        "road_graph",
        "binding",
        "native_path",
        "config_path",
        "checkpoint_path",
        "road_block_config",
    ):
        files[renderer[key]] = digest(renderer[key])
    return {
        **config,
        "config": str(path),
        "python": str(path_at(base, config["python"])),
        "files": files,
    }, config["max_sim_agent_steps"]


def run_render(config, inputs, output, registry):
    branch = read_mapping(inputs["branch"]["artifacts"]["report"])
    check_files({**config["files"], **branch["files"]})
    source = read_mapping(branch["bundle"])
    steps = source[branch["branch"]]["ticks"] * (1 + len(source["other_controlled_actors"])) * 2
    if steps > config["max_sim_agent_steps"]:
        raise ValueError("Both exact CPU replay and GS world replay must fit reservation")
    output.mkdir(parents=True)
    run_process(
        [
            config["python"],
            "-B",
            "-m",
            "driversi.rendering.execution.challenge_replay",
            "--config",
            config["config"],
            "--report",
            branch["bundle"],
            "--candidate",
            branch["branch"],
            "--output",
            str(output / "render"),
            "--allow-diagnostic-failure",
        ],
        output,
        config["max_wall_seconds"],
        environment={"CUDA_VISIBLE_DEVICES": str(config["gpu"])},
    )
    verification = read_mapping(output / "render/verification.json")
    manifest = output / "render/session/manifest.json"
    report = {
        "kind": "selected_branch_render",
        "verification": verification,
        "branch_identity": inputs["branch"]["summary"]["artifact_reference"]["identity"],
        "anchor_tick": branch["anchor_tick"],
        "teacher_qualified": False,
        "render_purpose": "diagnostic_sequence",
        "future_rgb_rendered": True,
        "trainable": False,
        "visual_quality_certified": False,
        "payload_files": {
            str(p): digest(p) for p in (output / "render/session").rglob("*") if p.is_file()
        },
    }
    return result(
        output,
        report,
        steps=steps,
        artifacts={
            "session": str(manifest),
            "scene": str(output / "render/provenance/scene.npz"),
            "viewer": str(output / "render/session/viewer.html"),
        },
    )


def prepare_export(config, base, registry):
    _keys(config, ("cameras", "future_images"), "branch inspection export")
    if type(config["future_images"]) is not bool or not config["cameras"]:
        raise ValueError("Explicit sensor purpose required")
    return config, 0


def run_export(config, inputs, output, registry):
    from driversi.data.datasets.navsim_synthetic import export_navsim_sample

    rendered = inputs["rendered"]
    receipt = read_mapping(rendered["artifacts"]["report"])
    check_files(receipt["payload_files"])
    package = export_navsim_sample(
        rendered["artifacts"]["session"],
        rendered["artifacts"]["scene"],
        output,
        anchor_tick=receipt["anchor_tick"],
        cameras=tuple(config["cameras"]),
        include_future_images=config["future_images"],
    )
    check_files(receipt["payload_files"])
    # Inspection packages retain valid examples and failure reasons. They cannot
    # satisfy native_training_dataset ports until a registered visual verifier passes.
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
        "admission": {"trainable": False, "visual": "inspection_only", "trajectory": "unqualified"},
    }


def register(registry, Provider):
    for p in (
        Provider(
            "reactive_synthesis_skill",
            "1",
            (),
            "reactive_execution",
            prepare_world,
            run_world,
            ("report", "bundle"),
        ),
        Provider(
            "reactive_branch_selection",
            "1",
            (("world", "reactive_execution"),),
            "selected_reactive_branch",
            prepare_select,
            run_select,
            ("report",),
        ),
        Provider(
            "selected_branch_gs",
            "1",
            (("branch", "selected_reactive_branch"),),
            "rendered_reactive_branch",
            prepare_render,
            run_render,
            ("report", "session", "scene", "viewer"),
        ),
        Provider(
            "selected_branch_navsim_export",
            "1",
            (("rendered", "rendered_reactive_branch"),),
            "navsim_inspection_dataset",
            prepare_export,
            run_export,
            ("report",),
        ),
    ):
        registry.register(p)
