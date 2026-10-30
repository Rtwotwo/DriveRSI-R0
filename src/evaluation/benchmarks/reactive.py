"""Source-preserving paired reactive background audit; no training qualification."""

import copy
import json
from pathlib import Path

import numpy as np

from driversi.contracts.population import EpisodePopulation
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.data.participants.background import (
    audit_presence,
    configure_background,
    configure_observed_background,
    propose_logged_tasks,
)
from driversi.rendering.bev.replay import export_replay
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi.sim.workers.worker import WorldWorker


def evaluate_background(scene_path, graph_path, output, *, context_path=None, population_path=None, brake_actor=None, observation_memory=True):
    from driversi.contracts.context import SceneContext, scene_fingerprint

    source = Scene.load(Path(scene_path))
    graph = RoadGraph.load(Path(graph_path))
    context = SceneContext.load(Path(context_path)) if context_path else None
    if context:
        context.validate_scene(source)
    population = EpisodePopulation.load(Path(population_path)) if population_path else None
    tasks, rejected = propose_logged_tasks(source)
    if not tasks and not observation_memory:
        raise ValueError("No supported moving background tasks; source remains replay")
    if observation_memory:
        scene, rejected = configure_observed_background(source)
        tasks = scene.metadata["reactive_background"]["assignments"]
    else:
        scene = configure_background(source, tasks)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene.save(output / "scene.npz")
    report = {
        "version": 1,
        "scenario_id": scene.metadata["scenario_id"],
        "trainable": False,
        "assignments": tasks,
        "observed_motion_inference": scene.metadata.get("observed_motion_inference", {}),
        "replay_reasons": rejected,
        "source_audit": audit_presence(source),
        "protocol": "logged_navigation_current_world_reaction_v1",
        "runs": {},
    }
    records = {}
    brake_actor = brake_actor or next(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    if brake_actor not in {a for a,c in zip(scene.actor_ids,scene.controlled) if c}:
        raise ValueError("Intervention actor must be a controlled vehicle")
    report["intervention"] = {"actor": brake_actor, "start_tick": 10, "action": 1}
    for name, active in (
        ("source", source),
        ("source_brake", source),
        ("reactive", scene),
        ("reactive_brake", scene),
    ):
        active_context = copy.deepcopy(context)
        if active_context:
            active_context.metadata["scene_fingerprint"] = scene_fingerprint(active)
            active_context.metadata["derived_from_scene_fingerprint"] = scene_fingerprint(source)
            active_context.validate_scene(active)
        worker = WorldWorker(active, graph, population=population)
        try:
            record = RolloutRecord.start(worker.world)
            actions = []
            saved = None
            while not worker.world.episode_done:
                tick = worker.world.observe().step
                if tick == 10:
                    saved = worker.world.snapshot()
                commands = worker.controller.commands(worker.world.observe())
                if name.endswith("_brake") and tick >= 10:
                    commands = copy.deepcopy(commands)
                    for command in commands:
                        if command["actor"] == brake_actor:
                            command["action"] = 1
                            command["intervention"] = "brake_action"
                action = {c["actor"]: c["action"] for c in commands}
                result = worker.world.step(action)
                record.append(worker.world, action, result, commands)
                actions.append(action)
            final = worker.world.state()
            restored = None
            if saved is not None:
                worker.world.restore(saved)
                with worker.world.fork() as branch:
                    for action in actions[10:]:
                        worker.world.step(action)
                        branch.step(action)
                        np.testing.assert_array_equal(worker.world.state(), branch.state())
                        assert worker.world.lifecycle() == branch.lifecycle()
                    np.testing.assert_array_equal(worker.world.state(), final)
                    restored = True
            # Recording uses the full, original run including reactive diagnostics.
            write_rollout(
                active,
                output / (name + ".npz"),
                record=record,
                snapshot=worker.initial,
                controlled_ids=worker.world.actor_ids,
                policy="episode-population-v1",
                controller_configuration=worker.configuration,
            )
            export_replay(
                active,
                output / (name + ".npz"),
                output / (name + ".html"),
                road_graph=graph,
                scene_context=active_context,
                compare_path=output / (name.removesuffix("_brake") + ".npz")
                if name.endswith("_brake")
                else None,
            )
            records[name] = record
            decisions = [
                d for frame in record.lifecycle for d in frame.get("background_decisions", [])
            ]
            report["runs"][name] = {
                "steps": len(record.actions),
                "restore_and_fork_exact": restored,
                "controlled_collisions": sum(
                    int(t.events["collision"].sum()) for t in record.transitions
                ),
                "background_conflict_decisions": sum(
                    d["conflict_id"] is not None for d in decisions
                ),
                "contact_ticks": sum(bool(f["contacts"]) for f in record.lifecycle),
                "present_min": int(np.asarray(record.presence).sum(1).min()),
                "present_max": int(np.asarray(record.presence).sum(1).max()),
            }
        finally:
            worker.close()
    report["paired_response"] = {}
    for mode in ("source", "reactive"):
        left, right = records[mode], records[mode + "_brake"]
        count = min(len(left.states), len(right.states))
        changes = []
        for i, actor in enumerate(source.actor_ids):
            if source.controlled[i]:
                continue
            delta = np.linalg.norm(
                np.asarray(left.states)[:count, i, :2] - np.asarray(right.states)[:count, i, :2],
                axis=1,
            )
            hits = np.flatnonzero(delta > 0.01)
            if len(hits):
                changes.append(
                    {
                        "actor": actor,
                        "kind": int(source.kinds[i]),
                        "first_tick": int(hits[0]),
                        "max_delta_m": float(delta.max()),
                    }
                )
        report["paired_response"][mode] = {
            "compared_states": count,
            "background_position_changes_over_1cm": changes,
            "qualification": "position_difference_only_not_causal_interaction_acceptance",
            "pre_intervention_equal": bool(
                np.array_equal(np.asarray(left.states)[:11], np.asarray(right.states)[:11])
            ),
        }
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>DriveRSI 反应式参与者</title>'
        "<h1>真实 NAVSIM · 日志与反应式背景对照</h1><p>背景路径来自日志，但速度与位置由当前世界反馈推进。"
        "短暂漏标与真正离场不能混同；单帧对象仍保留原始回放，未擅自补全。</p>"
        '<p><a href="source.html">原始日志人口</a> · <a href="reactive.html">反应式背景</a> · '
        '<a href="reactive_brake.html">制动配对对照</a> · <a href="validation.json">完整审计与边界</a></p><p>规则基线；不是已验证的人类行为模型。GS 缺失资产不代表物理对象不存在。</p>'
    )
    return {
        "output": str(output),
        "scenario_id": report["scenario_id"],
        "reactive_actors": len(tasks),
        "runs": report["runs"],
        "trainable": False,
    }
