"""Frozen real-data coverage audits with separate execution and behavior records."""

import json
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.routing import RouteResolutionError, inspect_routes
from driversi.contracts.road import Route
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene
from driversi.data.road_network.lane_graph import compile_lane_graph
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.execution.rollout import rollout


def summarize_rollout(scene: Scene, path: Path) -> dict:
    """Counts are actor-episodes. Never turn stopped into an inferred collision."""
    with np.load(path, allow_pickle=False) as bundle:
        states = bundle["states"]
        valid = bundle["valid"]
        metadata = json.loads(str(bundle["metadata"]))
        controlled = np.flatnonzero(scene.controlled)
        if states.shape != (len(valid)+1, len(scene.actor_ids), 13):
            raise ValueError("Rollout state and transition clocks do not align")
        if not np.isfinite(states).all() or not np.isfinite(bundle["motion"]).all():
            raise ValueError("Nonfinite recorded dynamics")
        report = {"steps": len(valid), "valid_agent_steps": int(valid.sum()),
                  "controlled_actor_episodes": len(controlled), "events": {}, "actors": {}}
        report["initial_inactive_actors"] = int(np.count_nonzero(states[0, controlled, 9]))
        report["initial_events"] = {name: int(np.count_nonzero(states[0, controlled, column]))
                                    for name, column in (("collision", 10), ("offroad", 11), ("goal", 12))}
        for event in ("collision", "offroad", "goal"):
            values = bundle[event]
            if values.shape != valid.shape:
                raise ValueError("Event and validity shapes do not match")
            report["events"][event] = int(values.any(axis=0).sum())
        presence = bundle["presence"] if "presence" in bundle else np.ones(states.shape[:2], bool)
        if scene.presence is not None and "presence" not in bundle:
            raise ValueError("Logged-population audit requires recorded presence")
        if presence.shape != states.shape[:2] or presence.dtype != bool:
            raise ValueError("Recorded presence must match world states")
        expected_presence = np.stack([scene.presence_at(tick) for tick in range(len(states))])
        if not np.array_equal(presence, expected_presence):
            raise ValueError("Recorded presence differs from scene lifecycle")
        initial_contacts = set()
        for i in np.flatnonzero(presence[0]):
            a = rectangle_corners(*states[0, i, :2], states[0, i, 3], *scene.dimensions[i, :2])
            for j in np.flatnonzero(presence[0, :i]):
                b = rectangle_corners(*states[0, j, :2], states[0, j, 3], *scene.dimensions[j, :2])
                if rectangles_overlap(a, b):
                    initial_contacts.update((i, j))
        report["initial_controlled_2d_contacts"] = sum(i in initial_contacts for i in controlled)
        configurations = metadata["controller_configuration"].get("routes", {})
        for column, i in enumerate(controlled):
            actor = scene.actor_ids[i]
            indices = np.unique(np.r_[0, np.flatnonzero(valid[:, column])+1])
            body_speed = (states[indices, i, 4]*np.cos(states[indices, i, 3])
                          + states[indices, i, 5]*np.sin(states[indices, i, 3]))
            item = {"valid_steps": int(valid[:, column].sum()),
                    "initial_inactive": bool(states[0, i, 9]),
                    "initial_events": {name: bool(states[0, i, col]) for name, col in
                                       (("collision", 10), ("offroad", 11), ("goal", 12))},
                    "displacement_m": float(np.linalg.norm(states[-1, i, :2]-states[0, i, :2])),
                    "distance_traveled_m": float(np.linalg.norm(np.diff(states[:, i, :2], axis=0), axis=1).sum()),
                    "min_body_speed_mps": float(body_speed.min()),
                    "reverse_state_samples": int(np.count_nonzero(body_speed < -1e-3)),
                    "initial_2d_contact": i in initial_contacts,
                    "events": {name: bool(bundle[name][:, column].any()) for name in report["events"]}}
            if actor in configurations:
                route = Route(**configurations[actor])
                # Include the successor of each valid action, including terminal state.
                projected = np.asarray([route.project(states[k, i, :2]) for k in indices])
                item.update(initial_route_distance_m=float(projected[0, 1]),
                            final_route_distance_m=float(projected[-1, 1]),
                            max_route_distance_m=float(projected[:, 1].max()),
                            route_progress_m=float(projected[-1, 0]-projected[0, 0]))
            report["actors"][actor] = item
        return report


def verify_replay(scene: Scene, path: Path) -> dict:
    with np.load(path, allow_pickle=False) as bundle, PufferDriveWorld(scene) as world:
        actions, states, motion = bundle["actions"], bundle["states"], bundle["motion"]
        exact = np.array_equal(world.state(), states[0]) and np.array_equal(world.motion(), motion[0])
        def matches(step, transition):
            return (np.array_equal(world.state(), states[step+1])
                    and np.array_equal(world.motion(), motion[step+1])
                    and all(np.array_equal(getattr(transition, key), bundle[key][step])
                            for key in ("valid", "terminated", "truncated", "rewards"))
                    and all(np.array_equal(transition.events[key], bundle[key][step])
                            for key in ("collision", "offroad", "goal")))
        midway = len(actions)//2
        snapshot = None
        for step, action in enumerate(actions):
            if step == midway:
                snapshot = world.snapshot()
            transition = world.step(action)
            exact &= matches(step, transition)
        restored = None
        if snapshot:
            observation = world.restore(snapshot)
            restored = (np.array_equal(world.state(), states[midway])
                        and np.array_equal(world.motion(), motion[midway])
                        and np.array_equal(observation.alive, snapshot.alive)
                        and observation.step == midway)
            for step in range(midway, len(actions)):
                restored &= matches(step, world.step(actions[step]))
        return {
            "joint_action_replay_exact": bool(exact) if len(actions) else None,
            "mid_episode_restore_exact": bool(restored) if restored is not None else None}


def evaluate_scenes(selection_path: Path, scene_dir: Path, maps_root: Path, output: Path) -> dict:
    """CLI success means the audit ran, not that all behavior metrics are acceptable."""
    from driversi.sim import _native

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != 1 or not selection.get("cases"):
        raise ValueError("Require a nonempty frozen scene selection schema v1")
    names = [case["scene"] for case in selection["cases"]]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("Frozen cases must be unique scene filenames")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "selection": selection,
              "backend_build": _native.build_id, "requested": len(names), "completed": 0,
              "unresolved_tasks": 0, "runtime_errors": 0, "contract_failures": 0, "unexercised": 0,
              "behavior_scope": "native event counts and local geometric tracking; not traffic-rule certification",
              "records": []}

    def save():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output / "report.json")

    save()
    for case in selection["cases"]:
        directory = output / Path(case["scene"]).stem
        directory.mkdir()
        record = {"scene": case["scene"], "expected_scenario_id": case["scenario_id"]}
        try:
            scene = Scene.load(scene_dir / case["scene"])
            if scene.metadata["scenario_id"] != case["scenario_id"]:
                raise ValueError("Frozen case identity mismatch")
            record.update(map_name=scene.metadata["map_name"], source_log=scene.metadata["source_log"],
                          source_log_token=scene.metadata.get("source_log_token"),
                          duration_seconds=float(scene.times[-1]))
            graph = compile_lane_graph(scene, maps_root)
            graph.save(directory / "lanes.json")
            tasks = RouteTasks.load(selection_path.parent / case["route_tasks"]) if case.get("route_tasks") else None
            routing = inspect_routes(scene, graph, tasks)
            (directory / "routes.json").write_text(json.dumps(routing, indent=2, allow_nan=False), encoding="utf-8")
            if not routing["all_resolved"]:
                raise RouteResolutionError(routing)
            path = directory / "rollout.npz"
            rollout(scene, path, policy="follow", road_graph=graph, route_tasks=tasks)
            record.update(status="completed", behavior=summarize_rollout(scene, path),
                          contracts=verify_replay(scene, path))
            if record["behavior"]["valid_agent_steps"] == 0:
                record["status"] = "no_valid_steps"
                report["unexercised"] += 1
            else:
                report["completed"] += 1
            report["contract_failures"] += int(any(value is False for value in record["contracts"].values()))
        except RouteResolutionError as error:
            record.update(status="needs_route_task", routing=error.report)
            report["unresolved_tasks"] += 1
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            record.update(status="runtime_error", error_type=type(error).__name__, error=str(error))
            report["runtime_errors"] += 1
        report["records"].append(record)
        save()
        print(json.dumps({"scene": case["scene"], "status": record["status"]}), flush=True)
    report["status"] = "complete"
    report["events"] = {event: sum(item.get("behavior", {}).get("events", {}).get(event, 0)
                                         for item in report["records"]) for event in ("collision", "offroad", "goal")}
    save()
    return {key: report[key] for key in ("requested", "completed", "unresolved_tasks", "runtime_errors",
                                       "contract_failures", "unexercised", "events")} | {"report": str(output / "report.json")}
