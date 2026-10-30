"""Bounded, paired causal probes on existing scene tasks; no traffic certification."""

import json
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.following import RouteFollower, quantize_jerk
from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.road import RoadGraph
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout

DT = .1
TOLERANCE = 1e-5
BRAKE_TARGET_MPS2 = -3.0


def _brake_protocol(intervention_seconds):
    return {
        "target_mps2": BRAKE_TARGET_MPS2,
        "rule": "min(current_controller_target, target_mps2)",
        "start_seconds": intervention_seconds,
        "duration": "each active tick until branch termination",
        "execution": "quantized jerk control; target is not instantaneous actual acceleration",
    }


def _advance(world, controller, *, steps=None, lead=None, observer=None, replay=None):
    record = RolloutRecord.start(world)
    stop_reason = "episode_boundary"
    while not world.episode_done and (steps is None or len(record.transitions) < steps):
        observation = world.observe()
        commands = controller.commands(observation)
        offset = len(record.transitions)
        for index, command in enumerate(commands):
            actor = command["actor"]
            if actor == lead and observation.alive[index]:
                ego = observation.metric.ego[index]
                target = min(command["desired_acceleration"], BRAKE_TARGET_MPS2)
                command["desired_acceleration"] = target
                command["action"] = quantize_jerk(
                    target, command["desired_lateral_acceleration"], ego[7:9],
                    command["actual_speed_mps"], steering=ego[9], wheelbase=.6 * ego[5])
                command["intervention"] = "longitudinal_brake_target"
            if actor == observer and replay is not None:
                if offset >= len(replay.actions):
                    stop_reason = "baseline_action_reference_exhausted"
                    return record, stop_reason
                # Replay only this actor's executed action, not a policy decision
                # using the new world. Keep diagnostics from that reference too.
                commands[index] = dict(replay.diagnostics[offset][index])
                commands[index]["action"] = replay.actions[offset][index]
                commands[index]["controller"] = "baseline_action_replay"
        actions = {command["actor"]: command["action"] for command in commands}
        transition = world.step(actions)
        record.append(world, actions, transition, commands)
    if steps is not None and len(record.transitions) == steps and not world.episode_done:
        stop_reason = "probe_prefix_complete"
    return record, stop_reason


def _exact(first: RolloutRecord, second: RolloutRecord) -> bool:
    if not all(np.array_equal(getattr(first, key), getattr(second, key))
               for key in ("states", "motion", "actions")):
        return False
    if len(first.transitions) != len(second.transitions):
        return False
    return all(
        all(np.array_equal(getattr(a, key), getattr(b, key))
            for key in ("valid", "terminated", "truncated", "rewards"))
        and all(np.array_equal(a.events[key], b.events[key]) for key in a.events)
        for a, b in zip(first.transitions, second.transitions, strict=True))


def _usable_states(scene: Scene, record: RolloutRecord, pair: tuple[int, int]) -> int:
    """Exclude first stopped/contact state, including contacts with other objects."""
    for tick, state in enumerate(record.states):
        for index in pair:
            if state[index, 9]:
                return tick
            box = rectangle_corners(*state[index, :2], state[index, 3],
                                    *scene.dimensions[index, :2])
            for other in range(len(scene.actor_ids)):
                if other == index:
                    continue
                neighbor = rectangle_corners(*state[other, :2], state[other, 3],
                                             *scene.dimensions[other, :2])
                if rectangles_overlap(box, neighbor):
                    return tick
    return len(record.states)


def probe_interaction(scene: Scene, road_graph: RoadGraph, output: Path, *,
                      intervention_seconds: float = .5, response_seconds: float = 2.,
                      route_tasks: RouteTasks | None = None) -> dict:
    """Test one deterministically selected, currently observed controlled lead pair.

    Verified means a bounded longitudinal response with action-replay and
    determinism controls. It says nothing about other pairs or driving success.
    """
    for value in (intervention_seconds, response_seconds):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0
                or not np.isclose(value / DT, round(value / DT), rtol=0, atol=1e-8)):
            raise ValueError("Probe times must be positive finite multiples of 0.1 seconds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "protocol": "scene-following-probe-v1", "status": "running",
        "scenario_id": scene.metadata["scenario_id"], "planned_seconds": float(scene.times[-1]),
        "intervention_seconds": intervention_seconds, "response_budget_seconds": response_seconds,
        "response_tolerance_mps2": TOLERANCE, "checks": {}, "metrics": {}, "recordings": {},
        "scope": "one selected following pair; not a driving, junction or whole-scene certificate",
        "negative_control": "baseline follower actions; not logged trajectory replay",
        "brake_intervention": _brake_protocol(intervention_seconds),
        "selection": "lexicographically first alive follower with an alive controlled lead at the fixed probe tick",
    }

    def save(status, reason):
        report.update(status=status, reason=reason)
        pending = output / "report.pending"
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
        pending.replace(output / "report.json")
        return report

    save("running", "initializing")
    try:
        if scene.times[-1] + 1e-9 < intervention_seconds + response_seconds:
            return save("evidence_insufficient", "window_shorter_than_prefix_and_response_budget")
        if np.count_nonzero(scene.controlled) < 2:
            return save("not_applicable", "fewer_than_two_controlled_actors")
        routes, routing = resolve_routes(scene, road_graph, route_tasks)
        controller = RouteFollower(routes)
        with PufferDriveWorld(scene) as root:
            initial_snapshot = root.snapshot()
            prefix, _ = _advance(root, controller, steps=round(intervention_seconds / DT))
            if root.episode_done or len(prefix.transitions) != round(intervention_seconds / DT):
                return save("evidence_insufficient", "episode_ended_before_intervention")
            observation = root.observe()
            alive = {actor for actor, active in zip(root.actor_ids, observation.alive, strict=True) if active}
            candidates = sorted((command["actor"], command["lead_id"])
                                for command in controller.commands(observation)
                                if command["actor"] in alive and command["lead_id"] in alive)
            report["candidate_pairs"] = [list(pair) for pair in candidates]
            if not candidates:
                return save("not_applicable", "no_alive_controlled_following_pair_at_probe_tick")
            observer, lead = candidates[0]
            report.update(observer_id=observer, lead_id=lead)
            actor_index = scene.actor_ids.index(observer)
            observer_column = root.actor_ids.index(observer)
            pair = (actor_index, scene.actor_ids.index(lead))
            parent = root.snapshot()
            report["backend_build"] = parent.build_id
            parent.save(output / "intervention-snapshot.npz")
            branches = {}
            for name, changed, open_loop in (
                ("baseline", None, False), ("lead_brake", lead, False),
                ("action_replay_keep", None, True), ("action_replay_brake", lead, True),
            ):
                replay = branches["baseline"] if open_loop else None
                with root.fork() as world:
                    record, stop_reason = _advance(world, controller, lead=changed,
                                                   observer=observer, replay=replay)
                branches[name] = record
                report["checks"][name + "_parent_unchanged"] = root.snapshot() == parent
                with root.fork() as repeated:
                    repeated.restore(parent)
                    again, _ = _advance(repeated, controller, lead=changed,
                                        observer=observer, replay=replay)
                report["checks"][name + "_exact_repeat"] = _exact(record, again)
                path = output / (name + ".npz")
                configuration = controller.describe()
                configuration["routing"] = routing
                configuration["probe_overrides"] = {"braking_actor": changed,
                                                      "action_replay_actor": observer if open_loop else None,
                                                      "brake_intervention": _brake_protocol(intervention_seconds)
                                                      if changed is not None else None}
                write_rollout(
                    scene, path, record=prefix.followed_by(record), snapshot=initial_snapshot,
                    controlled_ids=root.actor_ids, policy="scene-probe-" + name,
                    controller_configuration=configuration,
                    protocol={"name": report["protocol"], "branch": name,
                              "observer_id": observer, "lead_id": lead,
                              "intervention_seconds": intervention_seconds,
                              "brake_intervention": _brake_protocol(intervention_seconds)
                              if changed is not None else None,
                              "stop_reason": stop_reason, "planned_seconds": float(scene.times[-1])})
                report["recordings"][name] = path.name
            baseline, changed = branches["baseline"], branches["lead_brake"]
            usable = min(_usable_states(scene, record, pair) for record in branches.values())
            usable = min(usable, round(response_seconds / DT) + 1)
            report["metrics"]["common_precontact_seconds"] = max(0, usable - 1) * DT
            if usable < 2:
                return save("evidence_insufficient", "no_common_precontact_interval")
            target_count = min(usable, len(baseline.diagnostics), len(changed.diagnostics))
            delta_target = np.asarray([
                changed.diagnostics[t][observer_column]["desired_acceleration"]
                - baseline.diagnostics[t][observer_column]["desired_acceleration"]
                for t in range(target_count)])
            delta_actual = np.asarray(changed.motion[:usable])[:, actor_index, 0] \
                - np.asarray(baseline.motion[:usable])[:, actor_index, 0]
            checks = report["checks"]
            checks["no_same_tick_anticipation"] = bool(
                abs(delta_target[0]) <= TOLERANCE
                and baseline.actions[0][observer_column] == changed.actions[0][observer_column])
            keep, brake = branches["action_replay_keep"], branches["action_replay_brake"]
            checks["action_replay_precontact_unchanged"] = bool(
                np.array_equal(np.asarray(keep.states[:usable])[:, actor_index, :7],
                               np.asarray(brake.states[:usable])[:, actor_index, :7])
                and np.array_equal(np.asarray(keep.motion[:usable])[:, actor_index],
                                   np.asarray(brake.motion[:usable])[:, actor_index]))
            checks["action_replay_keep_matches_baseline"] = _exact(baseline, keep)
            target_hits = np.flatnonzero(np.abs(delta_target) > TOLERANCE)
            actual_hits = np.flatnonzero(np.abs(delta_actual) > TOLERANCE)
            target_delay = float(target_hits[0] * DT) if len(target_hits) else None
            actual_delay = float(actual_hits[0] * DT) if len(actual_hits) else None
            report["metrics"].update(
                desired_response_delay_s=target_delay, actual_response_delay_s=actual_delay,
                target_acceleration_delta=delta_target.tolist(),
                actual_acceleration_delta=delta_actual.tolist())
            if not all(checks.values()):
                return save("failed", "synchronization_replay_or_fork_contract_failed")
            if usable < round(response_seconds / DT) + 1:
                return save("evidence_insufficient", "contact_or_actor_stop_censored_response_budget")
            if target_delay is None or actual_delay is None:
                return save("evidence_insufficient", "no_measurable_response_within_budget")
            responding = bool(target_delay >= DT - 1e-9 and actual_delay >= 2 * DT - 1e-9
                              and delta_target[target_hits[0]] < 0 and delta_actual[actual_hits[0]] < 0)
            report["checks"]["causal_braking_direction_and_delay"] = responding
            return save("verified" if responding else "failed",
                        "selected_pair_response_verified" if responding else "response_direction_or_timing_failed")
    except Exception as error:
        save("failed", f"{type(error).__name__}: {error}")
        raise
