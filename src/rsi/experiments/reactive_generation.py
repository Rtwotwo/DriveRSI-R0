"""DriveRL actual history/future and same-branch GS, through durable Harness tasks."""

import argparse
import copy
import os
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.data.catalog.store import identity
from driversi.rsi.decisions.capability_skills import SKILLS
from driversi.rsi.experiments.continuation import (
    _reconcile_assignment,
    frozen_json,
    render_task,
    task_state,
)
from driversi.runtime.budget import write_json


def assignment_plan(rows, context, *, skill="reactive_new_history"):
    if skill not in (
        "reactive_new_history",
        "reactive_new_history_tts",
        "driverl_full_window",
        "driverl_sample_full_window",
        "perturbed_history_tts",
    ):
        raise ValueError("Native new-history skill required")
    assignments = []
    for i, row in enumerate(rows):
        assignments.append(
            dict(
                id=f"assignment-{i}",
                token=row["token"],
                source_log=row["source_log"],
                source_group=row["source_group"],
                capability=row.get("capability", "recovery"),
                skill=skill,
                skill_version=1,
                recipe=SKILLS[skill],
                max_wall_seconds=900,
                trainable=False,
            )
        )
    plan = dict(
        kind="capability_skill_plan",
        schema_version=1,
        context=context,
        assignments=assignments,
        selection_role=context["selection_role"],
        automatic_training=False,
    )
    return {**plan, "plan_id": identity(plan)}


def generate(
    plan_path,
    assignment_id,
    asset,
    database,
    output,
    *,
    gpu=6,
    render=True,
    runtime=None,
    render_gpu=None,
):
    from driversi.rsi.experiments.skill_execution import run_assignment

    runtime = runtime or {}
    root = project_root().resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(root / "exp"):
        raise ValueError("Production artifacts belong under exp")
    output.mkdir(parents=True, exist_ok=True)
    plan = read_mapping(plan_path)
    chosen = next(r for r in plan["assignments"] if r["id"] == assignment_id)
    if chosen["token"] != asset["token"] or chosen["source_log"] != asset["source_log"]:
        raise ValueError("Asset candidate and selected training source differ")
    registry = output / "harness.sqlite"
    prefix = output.name
    source = dict(
        kind="assignment",
        plan=str(Path(plan_path).resolve()),
        assignment=assignment_id,
        database=str(Path(database).resolve()),
        memory=str(output.parent / "execution-memory.sqlite"),
        maps_root=runtime.get("maps_root", os.environ.get("NAVSIM_MAP_ROOT", str(root / "data/maps"))),
        python=runtime.get(
            "driverl_python", str(root / ".venvs/driverl/bin/python")
        ),
        checkpoint=str(root / "weights/driverl/checkpoint_2400.pt"),
        device="cpu",
    )
    render_config = dict(
        provider="worldengine_selected_branch_render",
        config=dict(
            bundle=asset["bundle_root"],
            python=runtime.get("gs_python", str(root / ".venvs/mtgs/bin/python")),
            gpu=gpu,
            cuda_home=runtime.get("cuda_home", "/usr/local/cuda-12.4"),
            max_wall_seconds=600,
            max_render_images=60,
        ),
        source_config=dict(
            python=runtime.get("gs_python", str(root / ".venvs/mtgs/bin/python")),
            gpu=gpu,
            cuda_home=runtime.get("cuda_home", "/usr/local/cuda-12.4"),
            max_wall_seconds=600,
        ),
    )
    config = dict(id=prefix, registry=str(registry), input=source, render=render_config)
    frozen_json(
        output / "generation.json", dict(config=config, asset=asset, plan_sha256=digest(plan_path))
    )
    destination = output / (prefix + "-teacher")
    receipt = destination / "report.json"
    outcome = read_mapping(receipt) if receipt.exists() else None
    if outcome and (
        outcome["plan_sha256"] != digest(plan_path) or outcome["assignment_id"] != assignment_id
    ):
        raise ValueError("Teacher receipt belongs to another selection")
    if destination.exists() and (
        outcome is None
        or outcome["status"] in ("running", "interrupted", "preparation_or_execution_failed")
    ):
        outcome = _reconcile_assignment(config, destination, plan) or outcome
        if outcome is None or outcome["status"] in ("running", "interrupted"):
            return dict(status="teacher_outcome_unknown", trainable=False)
    if outcome is None:
        outcome = run_assignment(
            plan_path,
            assignment_id,
            database,
            destination,
            source["memory"],
            maps_root=source["maps_root"],
            python=source["python"],
            checkpoint=source["checkpoint"],
            device="cpu",
            registry=registry,
        )
    result = dict(
        kind="reactive_gs_generation",
        token=chosen["token"],
        source_log=chosen["source_log"],
        teacher=outcome,
        trainable=False,
        selection_role=plan["selection_role"],
    )
    if outcome["status"] != "trajectory_admitted":
        result.update(status="teacher_rejected")
    elif not render:
        result.update(status="trajectory_ready")
    else:
        frozen_json(
            output / "render-reservation.json",
            dict(
                plan_sha256=digest(plan_path),
                assignment=assignment_id,
                reason="charge_attempt_even_if_source_comparison_fails",
            ),
        )
        dispatched = config
        if render_gpu is not None:
            if type(render_gpu) is not int or not 0 <= render_gpu <= 7:
                raise ValueError("Invalid rendering GPU")
            frozen_json(output / "render-routing.json", dict(gpu=render_gpu))
            dispatched = copy.deepcopy(config)
            dispatched["render"]["config"]["gpu"] = render_gpu
            dispatched["render"]["source_config"]["gpu"] = render_gpu
        elif (output / "render-routing.json").exists():
            raise ValueError("Existing explicit render routing must be preserved")
        state = task_state(
            registry,
            render_task(dispatched),
            output / "render-task.json",
            output / "tasks",
            execute=True,
        )
        result.update(
            status="rendered" if state["status"] == "completed" else "render_" + state["status"],
            render_status=state["status"],
            render_reason=state.get("failure_reason"),
        )
        if state["status"] == "completed":
            for stage, key in (
                ("branch", "branch_report"),
                ("render", "render_report"),
                ("source", "source_report"),
            ):
                row = next(s for s in state["stages"] if s["id"] == stage)
                result[key] = row["result"]["artifacts"]["report"]
            result["render_payload"] = next(s for s in state["stages"] if s["id"] == "render")[
                "result"
            ]["artifacts"]["render_report"]
            result["files"] = {
                result[k]: digest(result[k])
                for k in ("branch_report", "render_report", "source_report")
            }
            result["files"][result["render_payload"]] = digest(result["render_payload"])
    write_json(output / "report.json", result)
    return result


def rerender(generation_path, asset, output, *, gpu=6, runtime=None):
    """Derive a new render receipt while keeping the completed teacher immutable."""
    from driversi.rsi.plugins.synthesis import check_files

    parent_path = Path(generation_path).resolve(strict=True)
    parent = read_mapping(parent_path)
    if parent.get("status") not in ("rendered", "trajectory_ready"):
        raise ValueError("A settled qualified physical generation is required")
    if (parent["token"], parent["source_log"]) != (asset["token"], asset["source_log"]):
        raise ValueError("Child asset belongs to another training source")
    check_files(parent.get("files", {}))
    old = read_mapping(parent_path.parent / "generation.json")
    teacher = parent["teacher"]
    if teacher.get("status") != "trajectory_admitted":
        raise ValueError("Teacher admission cannot be bypassed by asset repair")
    if digest(teacher["plan_path"]) != teacher["plan_sha256"]:
        raise ValueError("Original teacher plan changed")
    out = Path(output).resolve()
    if not out.is_relative_to(project_root() / "exp"):
        raise ValueError("New rendering artifacts belong under exp")
    out.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(old["config"])
    config["id"] = out.name
    if config["input"]["kind"] == "assignment":
        config["input"] = dict(kind="completed_teacher",
            source_task=old["config"]["id"]+"-teacher", source_stage="teacher")
    config["render"]["config"]["bundle"] = asset["bundle_root"]
    runtime = runtime or {}
    for section in ("config", "source_config"):
        target = config["render"][section]
        target["gpu"] = gpu
        if runtime.get("gs_python"):
            target["python"] = runtime["gs_python"]
        if runtime.get("cuda_home"):
            target["cuda_home"] = runtime["cuda_home"]
    frozen_json(out / "generation.json", dict(config=config, asset=asset,
        plan_sha256=teacher["plan_sha256"], parent_generation=str(parent_path),
        parent_generation_sha256=digest(parent_path), new_physical_steps=0))
    frozen_json(out / "render-reservation.json", dict(reason="repair_rerender", new_physical_steps=0))
    state = task_state(Path(config["registry"]), render_task(config), out / "render-task.json",
                       out / "tasks", execute=True)
    value = dict(kind="reactive_gs_generation", token=parent["token"],
        source_log=parent["source_log"], teacher=teacher,
        selection_role=parent["selection_role"], trainable=False,
        status="rendered" if state["status"] == "completed" else "render_"+state["status"],
        render_status=state["status"], render_reason=state.get("failure_reason"),
        parent_generation=str(parent_path), new_physical_steps=0)
    if state["status"] == "completed":
        for stage, key in (("branch", "branch_report"), ("render", "render_report"), ("source", "source_report")):
            value[key] = next(s for s in state["stages"] if s["id"] == stage)["result"]["artifacts"]["report"]
        value["render_payload"] = next(s for s in state["stages"] if s["id"] == "render")["result"]["artifacts"]["render_report"]
        value["files"] = {value[k]: digest(value[k]) for k in
                          ("branch_report", "render_report", "source_report", "render_payload")}
    write_json(out / "report.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = read_mapping(args.job)
    operation = job.pop("operation", "generate")
    if operation not in ("generate", "rerender"):
        raise ValueError("Unknown generation operation")
    print((rerender if operation == "rerender" else generate)(**job), flush=True)


if __name__ == "__main__":
    main()
