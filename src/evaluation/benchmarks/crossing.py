"""Paired crossing diagnostic from real source geometry; not human behavior validation."""
import copy
import json
from pathlib import Path

import numpy as np

from driversi.contracts.context import SceneContext, scene_fingerprint
from driversi.contracts.scene import PEDESTRIAN, VEHICLE, Scene
from driversi.data.participants.background import configure_background, propose_logged_tasks
from driversi.rendering.bev.replay import export_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout


def evaluate_crossing(scene_path, output, *, vehicle_id, pedestrian_id, context_path=None):
    """Keep all other actors; vary one pedestrian presence or one car's action.

    Initial measured speed is an explicit free-flow task, not a claimed human
    intent. Other vehicles retain source replay except ego and the probe car.
    No trajectory is relabelled as a successful teacher rollout.
    """
    source = Scene.load(Path(scene_path))
    context = SceneContext.load(Path(context_path)) if context_path else None
    if context:
        context.validate_scene(source)
    for actor, kind in ((vehicle_id, VEHICLE), (pedestrian_id, PEDESTRIAN)):
        i = source.actor_ids.index(actor)
        if source.kinds[i] != kind or not source.presence_at(0)[i] or actor == "ego":
            raise ValueError("Crossing probe requires present distinct background car and pedestrian")
    scene = copy.deepcopy(source)
    scene.controlled[:] = False
    scene.controlled[scene.actor_ids.index("ego")] = True
    scene.metadata.pop("task_goals_xy", None)
    tasks, rejected = propose_logged_tasks(scene)
    if vehicle_id not in tasks or pedestrian_id not in tasks:
        raise ValueError(f"Unsupported probe task: {rejected}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source.save(output / "source-scene.npz")
    if context:
        context.save(output / "source-context.json")
    report = {"trainable": False, "scenario_id": source.metadata["scenario_id"],
              "vehicle": vehicle_id, "pedestrian": pedestrian_id, "brake_start_tick": 5,
              "task_intent": "explicit_initial_measured_speed_on_logged_path",
              "counterfactuals": {"without_pedestrian": "named pedestrian removed only in this diagnostic branch",
                                  "pedestrian_brake": "probe car brake from tick5; other learner ego coasts"},
              "runs": {}, "paired": {}}
    records = {}
    for name in ("vehicle_yield", "without_pedestrian", "pedestrian_coast", "pedestrian_brake"):
        active = copy.deepcopy(scene)
        actor = pedestrian_id if name.startswith("pedestrian") else vehicle_id
        i = active.actor_ids.index(actor)
        task = copy.deepcopy(tasks[actor])
        task["desired_speed"] = float(np.linalg.norm(active.velocities[0, i, :2]))
        task["navigation_source"] = "explicit_task"
        if name.startswith("pedestrian"):
            active.controlled[active.actor_ids.index(vehicle_id)] = True
        if name == "without_pedestrian":
            if active.presence is None:
                active.presence = np.ones(active.poses.shape[:2], bool)
            j = active.actor_ids.index(pedestrian_id)
            active.presence[:, j] = False
            active.poses[:, j] = 0
            active.velocities[:, j] = 0
        active = configure_background(active, {actor: task})
        active.save(output / (name + "-scene.npz"))
        derived_context = copy.deepcopy(context)
        if derived_context:
            derived_context.metadata["derived_from_scene_fingerprint"] = scene_fingerprint(source)
            derived_context.metadata["scene_fingerprint"] = scene_fingerprint(active)
        with PufferDriveWorld(active) as world:
            initial = world.snapshot()
            record = RolloutRecord.start(world)
            checkpoint = None
            actions = []
            while not world.episode_done:
                tick = world.observe().step
                if tick == 5:
                    checkpoint = world.snapshot()
                action = {a: (1 if name == "pedestrian_brake" and a == vehicle_id and tick >= 5 else 7)
                          for a in world.actor_ids}
                transition = world.step(action)
                record.append(world, action, transition)
                actions.append(action)
            final = world.state()
            if checkpoint:
                world.restore(checkpoint)
                with world.fork() as branch:
                    for action in actions[5:]:
                        world.step(action)
                        branch.step(action)
                        np.testing.assert_array_equal(world.state(), branch.state())
                        if world.lifecycle() != branch.lifecycle():
                            raise AssertionError("Fork lifecycle differs")
                np.testing.assert_array_equal(world.state(), final)
            write_rollout(active, output / (name + ".npz"), record=record, snapshot=initial,
                          controlled_ids=world.actor_ids, policy="crossing_counterfactual_diagnostic",
                          controller_configuration={"controller": "coast_with_explicit_brake_probe"},
                          protocol=report["counterfactuals"])
            # Counterfactual presence differs: the general replay viewer correctly
            # rejects overlaying incompatible scenes. Each branch gets its own view.
            export_replay(active, output / (name + ".npz"), output / (name + ".html"),
                          scene_context=derived_context)
            records[name] = record
            report["runs"][name] = {"steps": len(actions), "restore_fork_exact": checkpoint is not None,
                                    "controlled_collisions": sum(int(t.events["collision"].sum()) for t in record.transitions),
                                    "goal_events": sum(int(t.events["goal"].sum()) for t in record.transitions),
                                    "offroad_events": sum(int(t.events["offroad"].sum()) for t in record.transitions),
                                    "actual_seconds": len(actions) * 0.1,
                                    "requested_seconds": float(active.times[-1]),
                                    "contact_ticks": sum(bool(f["contacts"]) for f in record.lifecycle),
                                    "probe_pair_contact_ticks": sum(any(set(pair) == {vehicle_id, pedestrian_id}
                                        for pair in f["contacts"]) for f in record.lifecycle)}
    for name, other, actor, target in (("vehicle_yield", "without_pedestrian", vehicle_id, pedestrian_id),
                                      ("pedestrian_coast", "pedestrian_brake", pedestrian_id, vehicle_id)):
        left, right = records[name], records[other]
        i = scene.actor_ids.index(actor)
        n = min(len(left.states), len(right.states))
        delta = np.linalg.norm(np.asarray(left.states)[:n, i, :2] - np.asarray(right.states)[:n, i, :2], axis=1)
        decisions = []
        for record in (left, right):
            decisions.append([{d["actor"]: d for d in f.get("background_decisions", [])}.get(actor)
                              for f in record.lifecycle[:n-1]])
        changed, targeted = [], []
        for tick, (a, b) in enumerate(zip(*decisions)):
            if a and b and abs(a["acceleration_mps2"] - b["acceleration_mps2"]) > 1e-3:
                changed.append(tick)
                if target in (a["conflict_id"], b["conflict_id"]):
                    targeted.append(tick)
        accepted = bool(delta.max() > 0.01 and targeted)
        report["paired"][name] = {"max_position_difference_m": float(delta.max()),
                                  "changed_decision_ticks": changed, "target_conflict_change_ticks": targeted,
                                  "paired_controller_response": accepted,
                                  "qualification": "target-specific decision change and >1cm displacement" if accepted else "insufficient_causal_evidence",
                                  "thresholds": {"position_m": 0.01, "acceleration_mps2": 1e-3},
                                  "pre_brake_states_equal": bool(np.array_equal(np.asarray(left.states)[:6], np.asarray(right.states)[:6])) if name.startswith("pedestrian") else None}
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "index.html").write_text('<!doctype html><meta charset=utf-8><h1>真实 NAVSIM 横穿与避让对照</h1><p>真实初态、地图、对象与路径；速度任务和干预为显式反事实。规则基线，不代表人类行为验证。</p>' +
        ''.join('<p><a href="' + name + '.html">' + name + '</a></p>' for name in records) + '<a href="validation.json">决策级配对验收</a>')
    return report
