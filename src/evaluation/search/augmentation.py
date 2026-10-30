"""SimScale-inspired two-stage reactive candidate generation, diagnostic rules only.

Four source frames are context/provenance. Stage one varies one actor's desired
speed, stage two resumes an explicit stateless route policy from the reached
world. Neither stage teleports vehicles or copies other actors' future actions.
This bounded prototype does not implement SimScale's vocabulary or PDM expert.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.data.datasets.history_windows import executed_label, source_history
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.data.sources.navsim import load_log
from driversi.evaluation.search.lateral_search import SearchBudget, StepAccount
from driversi.rendering.bev.replay import export_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi.training.sampling.environment import validate_development_split

VERSION = "two-stage-reactive-augmentation-v1"


def generate_pair(
    scene,
    routes,
    *,
    actor="ego",
    explore_steps=20,
    explore_speed=6.0,
    config=FollowingConfig(),
    step_guard=None,
):
    validate_development_split(scene.metadata["split"])
    ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    if (
        set(routes) != set(ids)
        or actor not in ids
        or len(ids) < 2
        or type(explore_steps) is not int
        or not 1 <= explore_steps < len(scene.times) - 1
        or isinstance(explore_speed, bool)
        or not np.isfinite(explore_speed)
        or not 1 <= explore_speed <= 15
        or not 1 <= config.desired_speed <= 15
    ):
        raise ValueError("Invalid two-stage actor, horizon or bounded desired speed")
    ordinary = RouteFollower(routes, config)
    perturb = RouteFollower(routes, replace(config, desired_speed=explore_speed))
    slot = ids.index(actor)
    records, boundaries = [], []
    initial = None
    for branch in ("baseline", "challenge"):
        with PufferDriveWorld(scene) as world:
            if not world.observe().alive.all():
                raise ValueError("Initially invalid controlled actor")
            initial = world.snapshot()
            record = RolloutRecord.start(world)
            boundary = None
            while not world.episode_done:
                observation = world.observe()
                if observation.step == explore_steps:
                    boundary = world.snapshot()
                commands = list(ordinary.commands(observation))
                if branch == "challenge" and observation.step < explore_steps:
                    commands[slot] = list(perturb.commands(observation))[slot]
                actions = {c["actor"]: c["action"] for c in commands}
                if step_guard:
                    step_guard()
                record.append(world, actions, world.step(actions), commands)
            records.append(record)
            boundaries.append(boundary)
    a, b = records
    common = min(len(a.actions), len(b.actions))
    action_diff = np.any(np.asarray(a.actions[:common]) != np.asarray(b.actions[:common]), axis=0)
    other_ids = [ids[i] for i in range(len(ids)) if i != slot]
    changed_others = [ids[i] for i in range(len(ids)) if i != slot and action_diff[i]]
    phases = {
        "exploration_end_tick": explore_steps,
        "continuation_start_tick": explore_steps,
        "continuation_policy": "stateless-idm-route-rule-diagnostic-v1",
        "teacher_qualified": False,
        "opponents_reobserve_each_tick": True,
    }
    branch_stats = {}
    for name, record, boundary in zip(("baseline", "challenge"), records, boundaries):
        collision = sum(int(t.events["collision"].any()) for t in record.transitions)
        offroad = sum(int(t.events["offroad"].any()) for t in record.transitions)
        failures = []
        if boundary is None or not boundary.alive[slot]:
            failures.append("owner_not_alive_at_stage_boundary")
        if collision:
            failures.append("collision")
        if offroad:
            failures.append("offroad")
        if len(record.actions) != len(scene.times) - 1:
            failures.append("early_world_end")
        branch_stats[name] = {
            "ticks": len(record.actions),
            "bounds_failures": failures,
            "collision_ticks": collision,
            "offroad_ticks": offroad,
        }
    report = {
        "schema_version": 1,
        "kind": VERSION,
        "trainable": False,
        "actor": actor,
        "phases": phases,
        "explore_speed": float(explore_speed),
        "continuation_configuration": asdict(config),
        "official_score": None,
        "input_use": "rules_consume_current_structured_state_and_map_route_not_rgb",
        "other_controlled_actors": other_ids,
        "responding_actors": changed_others,
        "changed_actor_actions": bool(action_diff[slot]),
        **branch_stats,
    }
    return report, records, initial, boundaries


def run(
    scene_path,
    graph_path,
    log_path,
    sensor_root,
    output,
    *,
    actor="ego",
    explore_steps=20,
    explore_speed=6.0,
    desired_speed=13.0,
    cameras=("CAM_F0", "CAM_R0"),
    trusted=False,
    budget=SearchBudget(max_attempts=1, max_joint_steps=200, max_wall_seconds=120),
):
    original = Scene.load(Path(scene_path))
    validate_development_split(original.metadata["split"])
    history = source_history(
        load_log(Path(log_path), trusted=trusted), original, sensor_root, camera_ids=cameras
    )
    graph = RoadGraph.load(Path(graph_path))
    routes, resolution = resolve_routes(original, graph)
    scene = bind_driving_tasks(original, routes)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    account = StepAccount(budget)

    def dump(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False))

    dump("source-history.json", history)
    dump("report.json", {"kind": VERSION, "status": "running", "trainable": False})
    try:
        config = FollowingConfig(desired_speed=desired_speed)
        report, records, initial, boundaries = generate_pair(
            scene,
            routes,
            actor=actor,
            explore_steps=explore_steps,
            explore_speed=explore_speed,
            config=config,
            step_guard=account.consume,
        )
        report.update(
            status="completed",
            export_status="writing",
            source_scene=str(Path(scene_path).resolve()),
            scenario_id=scene.metadata["scenario_id"],
            scene_key=initial.scene_key,
            backend_build=initial.build_id,
            source_history_sha256=file_sha256(output / "source-history.json"),
            route_resolution=resolution,
            budget=asdict(budget),
            charged_joint_steps=account.steps,
        )
        scene.save(output / "scene.npz")
        graph.save(output / "road_graph.json")
        branches = {}
        for name, record, boundary in zip(("baseline", "challenge"), records, boundaries):
            path = output / (name + ".npz")
            write_rollout(
                scene,
                path,
                record=record,
                snapshot=initial,
                controlled_ids=tuple(routes),
                policy=VERSION,
                controller_configuration=RouteFollower(routes, config).describe(),
                protocol={**report, "branch": name},
            )
            branches[name] = {"rollout": path.name, "sha256": file_sha256(path)}
            if boundary is not None:
                boundary.save(output / (name + "-stage-boundary.npz"))
                branches[name]["stage_snapshot"] = {
                    "path": name + "-stage-boundary.npz",
                    "sha256": file_sha256(output / (name + "-stage-boundary.npz")),
                }
                if len(record.actions) > explore_steps:
                    label = executed_label(
                        record.states,
                        scene.actor_ids.index(actor),
                        explore_steps,
                        list(range(explore_steps + 1, len(record.states))),
                        0.1,
                    )
                    dump(name + "-continuation.json", label)
        export_replay(
            scene,
            output / "baseline.npz",
            output / "viewer.html",
            road_graph=graph,
            compare_path=output / "challenge.npz",
        )
        report.update(
            branches=branches,
            export_status="complete",
            input_artifacts={
                name: {"path": name, "sha256": file_sha256(output / name)}
                for name in ("scene.npz", "road_graph.json", "source-history.json")
            },
        )
        dump("report.json", report)
        return report
    except Exception as error:
        dump(
            "report.json",
            {
                "kind": VERSION,
                "status": "failed",
                "trainable": False,
                "charged_joint_steps": account.steps,
                "error": str(error),
            },
        )
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "road-graph", "log", "sensor-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--actor", default="ego")
    p.add_argument("--explore-steps", type=int, default=20)
    p.add_argument("--explore-speed", type=float, default=6.0)
    p.add_argument("--desired-speed", type=float, default=13.0)
    p.add_argument("--trusted-log", action="store_true")
    a = p.parse_args(argv)
    result = run(
        a.scene,
        a.road_graph,
        a.log,
        a.sensor_root,
        a.output,
        actor=a.actor,
        explore_steps=a.explore_steps,
        explore_speed=a.explore_speed,
        desired_speed=a.desired_speed,
        trusted=a.trusted_log,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "responding_actors", "charged_joint_steps", "trainable")
            }
        )
    )


if __name__ == "__main__":
    main()
