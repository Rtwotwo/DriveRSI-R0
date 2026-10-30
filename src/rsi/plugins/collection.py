"""Training artifact -> actual candidate execution -> pending recovery task.

Explicit diagnostic reuse does not activate or promote a student. Native
sampling collectors, not these argmax rollouts, supply on-policy RL batches.
"""

from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.runtime.budget import write_json
from driversi.training.sampling.environment import source_log_group


def prepare(config, base, registry):
    _keys(
        config,
        ("scene", "road_graph", "source_registry", "anchor_tick", "max_wall_seconds"),
        "learned collection",
    )
    if type(config["anchor_tick"]) is not int or not 15 <= config["anchor_tick"] <= 40:
        raise ValueError("Bounded actual history required")
    if (
        type(config["max_wall_seconds"]) not in (int, float)
        or not 0 < config["max_wall_seconds"] <= 900
    ):
        raise ValueError("Bounded native worker time required")
    paths = {
        k: str((base / config[k]).resolve(strict=True))
        for k in ("scene", "road_graph", "source_registry")
    }
    scene = Scene.load(Path(paths["scene"]))
    declared = read_mapping(paths["source_registry"])
    if (
        scene.metadata["split"] not in ("train", "unassigned")
        or "nuplan_source" not in scene.metadata
        or declared["split"] != "train"
        or declared["source_log"] != scene.metadata["source_log"]
        or declared["scene_sha256"] != digest(paths["scene"])
    ):
        raise ValueError("Native collection requires an exact registered training source")
    source = scene.metadata["nuplan_source"]
    db = Path(source["db_path"]).resolve(strict=True)
    if (
        any(x in ("val", "test") for x in db.parts)
        or db.stem != scene.metadata["source_log"]
        or db != Path(declared["db_path"]).resolve(strict=True)
    ):
        raise ValueError("Collection DB is heldout or mismatched")
    protect([scene.metadata["source_log"]], registry, extra=declared["protected_logs"])
    steps = config["anchor_tick"] + 40
    if len(scene.times) <= config["anchor_tick"]:
        raise ValueError("Scene must support the actual reached anchor")
    steps = min(steps, len(scene.times) - 1)
    cap = (2 * steps + 40) * len(scene.actor_ids)
    return {
        **config,
        **paths,
        "steps": steps,
        "sim_cap": cap,
        "files": {p: digest(p) for p in paths.values()},
        "source_groups": [scene.metadata["source_log"]],
    }, cap


def validate_student(student, plan, source_log):
    if (
        student["status"] != "completed"
        or student["model"] != "driverl"
        or student["phase"] not in ("train", "resume")
        or plan["model_config"]["model"] != "driverl"
    ):
        raise ValueError("A completed DriveRL learning artifact is required")
    if student["context"] != plan["context"]:
        raise ValueError("Student recipe mismatch")
    group = source_log_group(source_log)
    if group in student["source_groups"]["development"]:
        raise ValueError("Development logs cannot be recycled into synthesis")
    checkpoint = student["checkpoint"]
    if checkpoint not in student["checkpoint_files"]:
        raise ValueError("Selected student weight is not part of its training publication")
    check_files(student["checkpoint_files"])


def execute(config, inputs, output, registry):
    check_files(config["files"])
    student = read_mapping(inputs["student"]["artifacts"]["report"])
    training = read_mapping(inputs["student"]["artifacts"]["plan"])
    validate_student(student, training, config["source_groups"][0])
    protect(config["source_groups"], registry, extra=student["source_groups"]["development"])
    cfg = training["model_config"]
    if cfg["repository"] != "agents/driverl/native":
        raise ValueError("Model must use the owned native implementation")
    check_files(training["files"])
    parent_files = {}
    if training["previous"]:
        parent_receipt = read_mapping(training["previous"])
        validate_student(parent_receipt, training, config["source_groups"][0])
        parent = parent_receipt["checkpoint"]
        parent_files[training["previous"]] = digest(training["previous"])
        parent_role = "immediate_training_parent"
    else:
        parent = str(project_root() / cfg["weights"]["checkpoint"])
        parent_role = "released_initialization"
    checkpoint = student["checkpoint"]
    files = {
        **config["files"],
        **parent_files,
        checkpoint: student["checkpoint_files"][checkpoint],
        parent: digest(parent),
    }
    files.update({p: digest(p) for p in inputs["student"]["artifacts"].values()})
    output.mkdir(parents=True)
    plan = {
        **config,
        "output": str(output),
        "files": files,
        "model_config": cfg,
        "checkpoint": checkpoint,
        "parent_checkpoint": parent,
        "parent_role": parent_role,
        "student_identity": inputs["student"]["summary"]["artifact_reference"]["identity"],
    }
    write_json(output / "collection-plan.json", plan)
    argv = [
        cfg["python"],
        "-B",
        "-m",
        "driversi_agents.driverl.collection",
        str(output / "collection-plan.json"),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": str(cfg["settings"]["gpu"]),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    import time

    start = time.monotonic()
    run_process(argv, output, config["max_wall_seconds"], environment=env)
    (output / "worker.log").rename(output / "collection.log")
    preliminary = read_mapping(output / "collection.json")
    if not preliminary["branches"]["candidate"]["recovery_available"]:
        return result(
            output,
            preliminary,
            status="rejected",
            reason="candidate_did_not_reach_live_anchor",
            steps=preliminary["charged_agent_steps"],
            artifacts={
                "plan": str(output / "collection-plan.json"),
                **{str(p.relative_to(output)): str(p) for p in output.rglob("*") if p.is_file()},
            },
        )
    remaining = config["max_wall_seconds"] - (time.monotonic() - start)
    if remaining <= 0:
        raise ValueError("No remaining budget for independent restore verification")
    run_process([*argv, "--verify"], output, remaining, environment=env)
    check_files(files)
    report = read_mapping(output / "collection.json")
    report.update(
        kind="learned_reached_state_collection",
        restore=read_mapping(output / "restore.json"),
        student_identity=plan["student_identity"],
        policy_activated=False,
        trainable=False,
        promoted=False,
        cost_scope="all_world_slots_each_step_including_restore",
    )
    artifacts = {
        "recovery_task": str(output / "candidate/recovery-task.json"),
        "plan": str(output / "collection-plan.json"),
    }
    for p in output.rglob("*"):
        if p.is_file():
            artifacts[str(p.relative_to(output))] = str(p)
    charged = (
        report["charged_agent_steps"]
        + report["restore"]["steps"] * report["branches"]["candidate"]["all_world_objects"]
    )
    report["charged_agent_steps"] = charged
    return result(output, report, steps=charged, artifacts=artifacts)


def register(registry, Provider):
    registry.register(
        Provider(
            "learned_recovery_collection",
            "1",
            (("student", "unified_training_result"),),
            "learned_recovery_execution",
            prepare,
            execute,
            ("report", "recovery_task", "plan"),
        )
    )
