"""Dispatch a selected new-history skill through existing physical Harness services.

No student scorer is imported. Teacher scoring remains the unchanged admission
service. Every terminal attempt, including preparation rejection, enters memory.
"""

import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.data.catalog.store import Catalog
from driversi.rsi.decisions.capability_skills import (
    NEW_HISTORY_SKILLS,
    SKILLS,
    SkillMemory,
    validate_plan,
)
from driversi.runtime.budget import write_json


def execution_evidence(teacher):
    """Report failed gates without declaring a collision's causal origin."""
    branch = teacher.get("branches", {}).get("teacher") or {}
    arrival = len(teacher.get("arrival_contacts", []))
    future = len(branch.get("object_contacts", []))
    checks = {
        "arrival_contacts_absent": arrival == 0,
        "future_contacts_absent": future == 0,
        "native_teacher_score_pass": (branch.get("score") or {}).get("score_pass") is True,
        "future_execution_complete": branch.get("completed_steps") == 50
        and branch.get("failure") is None,
        "continuous_new_history": teacher.get("history_changed") is True,
        "restore_exact": teacher.get("restore_exact") is True,
    }
    return {
        "failed_checks": [k for k, v in checks.items() if not v],
        "checks_scope": "diagnostics_not_all_admission_vetoes",
        "admission_profile": teacher.get("admission_profile", "strict_world95_v1"),
        "arrival_contact_records": arrival,
        "future_contact_records": future,
        "failure_domain": "world_validity_pending" if arrival or future else "branch_admission",
        "causal_attribution": "unresolved",
        "teacher_failure": branch.get("failure"),
    }


def run_assignment(
    plan_path,
    assignment_id,
    database,
    output,
    memory_path,
    *,
    maps_root,
    python,
    checkpoint,
    device="cpu",
    registry=None,
):
    from driversi.agents.traffic_control.routing import resolve_routes
    from driversi.contracts.driving_task import bind_driving_tasks
    from driversi.data.participants.fleet import configure_vehicle_fleet
    from driversi.data.participants.traffic_continuity import configure_traffic_continuity
    from driversi.data.road_network.lane_graph import compile_lane_graph
    from driversi.data.road_network.maps import attach_nuplan_map
    from driversi.data.road_network.route_continuation import extend_ego_route
    from driversi.data.sources.navsim import compile_frames, load_log
    from driversi.rsi.workflow import engine
    from driversi_agents.chainflow.benchmark import official_filter

    root = project_root().resolve()
    output = Path(output).resolve()
    memory_path = Path(memory_path).resolve()
    # The task spec lives under exp; relative runtime inputs belong to the
    # project, not to that per-attempt directory.
    if (
        output.exists()
        or not output.is_relative_to(root / "exp")
        or not memory_path.is_relative_to(root / "exp")
    ):
        raise ValueError("New execution and memory must live under exp")
    registry = Path(registry).resolve() if registry is not None else output / "harness.sqlite"
    if not registry.is_relative_to(root / "exp"):
        raise ValueError("Skill Harness registry must live under exp")
    plan = validate_plan(read_mapping(plan_path))
    chosen = next((a for a in plan["assignments"] if a["id"] == assignment_id), None)
    if chosen is None or chosen["skill"] not in NEW_HISTORY_SKILLS:
        raise ValueError("No native new-history execution bridge for this assignment")
    recipe = SKILLS[chosen["skill"]]
    if chosen["recipe"] != recipe:
        raise ValueError("Declared skill recipe version changed")
    output.mkdir(parents=True)
    started = time.monotonic()
    attempt = {
        "kind": "capability_skill_execution",
        "plan_id": plan["plan_id"],
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": digest(plan_path),
        "assignment_id": assignment_id,
        "token": chosen["token"],
        "source_log": chosen["source_log"],
        "skill": chosen["skill"],
        "skill_version": chosen["skill_version"],
        "status": "running",
        "training_executed": False,
        "trainable": False,
        "registry": str(registry),
    }
    try:
        checkpoint = (root / Path(checkpoint)).resolve(strict=True)
        python = (root / Path(python)).resolve(strict=True)
        maps_root = (root / Path(maps_root)).resolve(strict=True)
        catalog = Catalog(database, "navtrain", readonly=True)
        try:
            row = catalog.scene(chosen["token"])["record"]
        finally:
            catalog.close()
        _, membership = official_filter("navtrain")
        if (
            row["token"] not in membership["tokens"]
            or row["source_log"] not in membership["log_names"]
            or row["source_log"] != chosen["source_log"]
        ):
            raise ValueError("Assignment outside exact training membership")
        frames = load_log(Path(row["log_path"]), trusted=True)
        anchor = row["anchor_index"]
        if anchor < 4 or anchor + 10 >= len(frames):
            raise ValueError("Insufficient continuous source support for arrival and future")
        start = anchor - 4
        scene = compile_frames(
            frames, start=start, duration=7.0, split="train", max_actors=1, population_mode="logged"
        )
        attach_nuplan_map(scene, Path(maps_root), margin=120)
        graph = compile_lane_graph(scene, Path(maps_root), full_map_coverage=True)
        stamp = int(frames[start]["timestamp"])
        scene.metadata["navsim_source"] = {
            "log_path": row["log_path"],
            "sensor_root": row["sensor_root"],
            "anchor_timestamp_us": stamp,
            "signals": [
                {
                    "timestamp_us": int(stamp + v["time"] * 1e6),
                    "states": [
                        {"lane_connector_id": i, "status": "RED" if red else "GREEN"}
                        for i, red in v["records"]
                    ],
                }
                for v in scene.metadata["traffic_lights"]
            ],
        }
        route_profile = chosen.get("route_support_profile")
        navigation = frames[start]["roadblock_ids"]
        if route_profile == "driverl_navigation_v2":
            import os

            from driversi.data.road_network.route_continuation import initial_navigation_routes
            from driversi.rsi.plugins.synthesis import run_process

            repair = output / "route-support"
            repair.mkdir()
            raw_scene = repair / "input-scene.npz"
            scene.save(raw_scene)
            env = {
                "LD_LIBRARY_PATH": str(python.parent.parent / "lib")
                + ":"
                + os.environ.get("LD_LIBRARY_PATH", "")
            }
            run_process(
                [
                    str(python),
                    "-m",
                    "driversi_agents.driverl.route_support",
                    "--scene",
                    str(raw_scene),
                    "--maps-root",
                    str(maps_root),
                    "--output",
                    str(repair / "navigation.json"),
                ],
                repair,
                180,
                environment=env,
            )
            correction = read_mapping(repair / "navigation.json")
            if correction["scene_sha256"] != digest(raw_scene):
                raise ValueError("Navigation correction belongs to another scene")
            navigation = correction["corrected"]
            scene.metadata["native_navigation_correction"] = correction
            routes, routing = initial_navigation_routes(scene, graph, navigation)
        elif route_profile is None:
            routes, routing = resolve_routes(scene, graph)
        else:
            raise ValueError("Unknown route support profile")
        scene = configure_vehicle_fleet(bind_driving_tasks(scene, routes))
        scene = extend_ego_route(
            scene,
            graph,
            navigation,
            None,
            support_profile="bounded_navigation_v2" if route_profile else "strict_navigation_v1",
        )
        scene = configure_traffic_continuity(scene, graph)
        scene.metadata["source_train_membership"] = {
            "source_log": row["source_log"],
            "token": row["token"],
            "anchor_index": anchor,
            "reference_role": "official_split_membership_not_physical_anchor",
        }
        scene.metadata["capability_skill"] = {
            "plan_id": plan["plan_id"],
            "assignment_id": assignment_id,
            "skill": chosen["skill"],
            "skill_version": chosen["skill_version"],
        }
        if recipe.get("history_bootstrap"):
            scene.metadata["driverl_history_mode"] = recipe["history_bootstrap"]
        scene.save(output / "scene.npz")
        graph.save(output / "road-graph.json")
        remaining = int(chosen["max_wall_seconds"] - (time.monotonic() - started))
        if remaining < 325:
            raise TimeoutError("Preparation exhausted assignment wall budget")
        from driversi.rsi.correction.epdms_admission import required_for
        complete_epdms = required_for(output)
        teacher_wall = min(900 if complete_epdms else 360, remaining - 15)
        task = {
            "schema_version": 2,
            "id": output.name,
            "label": "Capability-selected native teacher skill",
            "budget": {
                "max_wall_seconds": remaining,
                "max_sim_agent_steps": (205 if complete_epdms else 125) * len(scene.actor_ids),
            },
            "steps": [
                {
                    "id": "teacher",
                    "provider": recipe["provider"],
                    "inputs": {},
                    "max_wall_seconds": teacher_wall,
                    "config": {
                        "scene": str(output / "scene.npz"),
                        "road_graph": str(output / "road-graph.json"),
                        "python": str(python),
                        "checkpoint": str(checkpoint),
                        "maps_root": str(maps_root),
                        "arrival_backend": recipe["arrival_backend"],
                        **(
                            {"history_perturbation": chosen["history_perturbation"]}
                            if chosen["skill"] == "perturbed_history_tts"
                            else {}
                        ),
                        "arrival_action": recipe["arrival_action"],
                        "search_candidates": recipe.get("search_candidates", 0),
                        **({"sampling_method": recipe["sampling_method"]} if "sampling_method" in recipe else {}),
                        "teacher_backend": recipe.get("teacher_backend", "driverl_native"),
                        "device": device,
                        "seed": chosen.get("sampling_seed", 42),
                        "admission_profile": chosen.get("admission_profile", "strict_world95_v1"),
                        "max_wall_seconds": teacher_wall,
                    },
                },
                {
                    "id": "admission",
                    "provider": "native_branch_selection",
                    "inputs": {"world": "teacher"},
                    "max_wall_seconds": 15,
                    "config": {"branch": "teacher", "future_frames": 10},
                },
            ],
        }
        write_json(output / "task.json", task)
        engine.register(registry, output / "task.json", output / "tasks")
        state = engine.run(registry, task["id"])
        report_path = output / "tasks" / task["id"] / "teacher/report.json"
        teacher = read_mapping(report_path) if report_path.exists() else {}
        diagnostics = execution_evidence(teacher) if teacher else {}
        admitted = (
            state["status"] == "completed"
            and teacher.get("branches", {}).get("teacher", {}).get("trajectory_admitted") is True
        )
        attempt.update(
            status="trajectory_admitted" if admitted else "execution_or_admission_rejected",
            harness_status=state["status"],
            failure_reason=state.get("failure_reason"),
            failure_domain="none" if admitted else diagnostics.get("failure_domain", "platform"),
            diagnostic_evidence=diagnostics,
            teacher_report=str(report_path) if report_path.exists() else None,
            same_anchor_score=teacher.get("branches", {}).get("teacher", {}).get("score"),
            history_changed=teacher.get("history_changed"),
            restore_exact=teacher.get("restore_exact"),
        )
    except Exception as exc:
        attempt.update(
            status="preparation_or_execution_failed",
            failure_domain="platform",
            failure_reason=type(exc).__name__ + ": " + str(exc),
        )
    except BaseException:
        attempt.update(
            status="interrupted", failure_domain="platform", failure_reason="execution_interrupted"
        )
        raise
    finally:
        attempt["wall_seconds"] = time.monotonic() - started
        write_json(output / "report.json", attempt)
        outcome = {
            "evidence_id": digest(output / "report.json"),
            "outcome_type": "execution",
            "plan_id": plan["plan_id"],
            "context": plan["context"],
            "capability": chosen["capability"],
            "skill": chosen["skill"],
            "skill_version": chosen["skill_version"],
            "token": chosen["token"],
            "execution_status": attempt["status"],
            "cost_wall_seconds": attempt["wall_seconds"],
            "failure_domain": attempt.get("failure_domain", "platform"),
            "diagnostic_evidence": attempt.get("diagnostic_evidence", {}),
            "credited": False,
            "reasons": ["teacher_attempt_is_not_student_training_gain"],
            "report": str(output / "report.json"),
            "trainable": False,
        }
        memory = SkillMemory(memory_path)
        try:
            memory.record(outcome)
        finally:
            memory.close()
    return attempt
