"""Budgeted, same-state longitudinal challenges; engineering evidence, not PDMS.

All traffic recomputes current-state actions. The bounded producer searches a
finite brake-pulse list; it does not learn an adversary or certify road rights.
Explicit heldout/test/navtest/navhard splits are refused; original-log isolation
and navtrain membership certification remain the upstream caller's responsibility.
"""
import argparse
import copy
import hashlib
import html
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower, quantize_jerk
from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.context import SceneContext
from driversi.contracts.road import RoadGraph
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene
from driversi.rendering.bev.replay import export_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi.training.sampling.environment import validate_development_split

DT = .1
VERSION = "bounded-longitudinal-challenge-v1"


@dataclass(frozen=True)
class BrakeCandidate:
    delay_steps: int
    duration_steps: int
    target_acceleration_mps2: float

    def __post_init__(self):
        if (type(self.delay_steps) is not int or self.delay_steps < 0
                or type(self.duration_steps) is not int or self.duration_steps < 1
                or type(self.target_acceleration_mps2) not in (int, float)
                or not np.isfinite(self.target_acceleration_mps2)
                or not -5 <= self.target_acceleration_mps2 < 0):
            raise ValueError("Brake pulses require nonnegative delay, positive duration, target in [-5,0)")


@dataclass(frozen=True)
class ChallengeBounds:
    minimum_gap_m: float = 2.
    maximum_gap_m: float = 40.
    minimum_speed_mps: float = .5
    maximum_speed_mps: float = 15.
    maximum_heading_difference_rad: float = .35
    maximum_route_distance_m: float = .75
    minimum_initial_ttc_seconds: float = 1.5
    maximum_pulse_steps: int = 40

    def __post_init__(self):
        values = asdict(self)
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0
               for v in values.values()):
            raise ValueError("Challenge bounds must be finite positive numbers")
        if (self.minimum_gap_m >= self.maximum_gap_m
                or self.minimum_speed_mps >= self.maximum_speed_mps
                or self.maximum_heading_difference_rad >= np.pi/2
                or self.maximum_speed_mps > 20
                or type(self.maximum_pulse_steps) is not int):
            raise ValueError("Invalid bounded following domain")


class _Budget:
    def __init__(self, max_world_steps, wall_seconds):
        if type(max_world_steps) is not int or max_world_steps < 1:
            raise ValueError("max_world_steps must be a positive integer")
        if type(wall_seconds) not in (int, float) or not np.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("wall_seconds must be finite and positive")
        self.limit, self.wall_limit = max_world_steps, float(wall_seconds)
        self.started = time.monotonic()
        self.attempted = self.completed = self.agent_steps = 0

    def remaining(self):
        return self.limit - self.attempted

    def reason(self):
        if self.remaining() <= 0:
            return "world_step_budget_exhausted"
        if time.monotonic() - self.started >= self.wall_limit:
            return "wall_budget_exhausted"
        return None

    def report(self):
        elapsed = time.monotonic() - self.started
        return {"max_world_steps": self.limit, "world_steps_attempted": self.attempted,
                "world_steps_completed": self.completed, "physical_agent_steps": self.agent_steps,
                "wall_seconds_limit": self.wall_limit, "wall_seconds_elapsed": elapsed,
                "wall_overshoot_seconds": max(0., elapsed-self.wall_limit),
                "accounting": "prefix, baseline, every attempted candidate; no uncharged repeats",
                "wall_enforcement": "checked before control and physical step; one call or persistence can overshoot",
                "branch_admission": "require remaining full-horizon step allowance; charge only attempted native steps"}


def _dump(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    pending.replace(path)


def _support(world, controller, follower_id, challenger_id, bounds, commands=None):
    observation = world.observe()
    commands = controller.commands(observation) if commands is None else commands
    rows = {c["actor"]: c for c in commands}
    ids = observation.actor_ids
    f, c = ids.index(follower_id), ids.index(challenger_id)
    follower, challenger = rows[follower_id], rows[challenger_id]
    angle = observation.metric.ego[f, 2] - observation.metric.ego[c, 2]
    angle = float(abs(np.arctan2(np.sin(angle), np.cos(angle))))
    gap, ttc = follower["bumper_gap_m"], follower["ttc_seconds"]
    reasons = []
    if not (observation.alive[f] and observation.alive[c]):
        reasons.append("pair_not_alive")
    if follower["lead_id"] != challenger_id:
        reasons.append("challenger_not_current_route_lead")
    if gap is None or not bounds.minimum_gap_m <= gap <= bounds.maximum_gap_m:
        reasons.append("gap_outside_initial_domain")
    if ttc is not None and ttc < bounds.minimum_initial_ttc_seconds:
        reasons.append("initial_ttc_below_bound")
    if angle > bounds.maximum_heading_difference_rad:
        reasons.append("heading_difference_outside_domain")
    for name, command in (("follower", follower), ("challenger", challenger)):
        if not bounds.minimum_speed_mps <= command["actual_speed_mps"] <= bounds.maximum_speed_mps:
            reasons.append(name+"_speed_outside_domain")
        if command["route_distance_m"] > bounds.maximum_route_distance_m:
            reasons.append(name+"_route_distance_outside_domain")
        if command["route_exhausted"]:
            reasons.append(name+"_route_exhausted")
    return {"supported": not reasons, "reasons": reasons, "gap_m": gap, "ttc_seconds": ttc,
            "heading_difference_rad": angle,
            "follower_speed_mps": follower["actual_speed_mps"],
            "challenger_speed_mps": challenger["actual_speed_mps"]}


def _run(world, controller, budget, *, end_tick, follower_id, challenger_id, bounds,
         start_tick=None, candidate=None):
    started = time.monotonic()
    before = (budget.attempted, budget.completed, budget.agent_steps)
    start_world_tick = world.observe().step
    record = RolloutRecord.start(world)
    result = {"stop_reason": "episode_boundary", "intervention_steps": 0,
              "action_override_steps": 0, "error": None}
    while not world.episode_done and world.observe().step < end_tick:
        reason = budget.reason()
        if reason:
            result["stop_reason"] = reason
            break
        try:
            observation = world.observe()
            commands = controller.commands(observation)
            tick = observation.step
            active = (candidate is not None and start_tick <= tick < start_tick+candidate.duration_steps)
            if active and tick == start_tick:
                result["intervention_start_support"] = _support(
                    world, controller, follower_id, challenger_id, bounds, commands)
                if not result["intervention_start_support"]["supported"]:
                    result["stop_reason"] = "intervention_start_unsupported"
                    break
            for index, command in enumerate(commands):
                command["policy_action"] = command["action"]
                if active and command["actor"] == challenger_id and observation.alive[index]:
                    ego = observation.metric.ego[index]
                    target = min(command["desired_acceleration"], candidate.target_acceleration_mps2)
                    command["baseline_desired_acceleration"] = command["desired_acceleration"]
                    command["desired_acceleration"] = float(target)
                    command["action"] = quantize_jerk(
                        target, command["desired_lateral_acceleration"], ego[7:9],
                        command["actual_speed_mps"], steering=ego[9], wheelbase=.6*ego[5])
                    command["intervention"] = "bounded_longitudinal_brake_pulse"
                    # Baseline actuator predictions do not describe a changed command.
                    for field in ("forward_release_feasible", "predicted_minimum_release_speed_mps",
                                  "predicted_next_lateral_acceleration"):
                        command["baseline_"+field] = command[field]
                        command[field] = None
                    command["actuator_diagnostics_scope"] = "override_actual_motion_recorded_predictions_unavailable"
            reason = budget.reason()
            if reason:
                result["stop_reason"] = reason
                break
            actions = {c["actor"]: c["action"] for c in commands}
            # Charge before the native call: a failed call is never free search.
            budget.attempted += 1
            transition = world.step(actions)
            budget.completed += 1
            budget.agent_steps += int(transition.valid.sum())
            record.append(world, actions, transition, commands)
            applied = [c for c in commands if c.get("intervention")]
            result["intervention_steps"] += len(applied)
            result["action_override_steps"] += sum(c["action"] != c["policy_action"] for c in applied)
        except Exception as error:
            result.update(stop_reason="runtime_error", error=f"{type(error).__name__}: {error}")
            break
    else:
        result["stop_reason"] = "episode_boundary" if world.episode_done else "prefix_complete"
    result.update(steps=len(record.transitions), requested_world_steps=end_tick-start_world_tick,
                  world_steps_attempted=budget.attempted-before[0],
                  world_steps_completed=budget.completed-before[1], physical_agent_steps=budget.agent_steps-before[2],
                  rollout_wall_seconds=time.monotonic()-started)
    return record, result


def _metrics(scene, record, controller, follower_id, challenger_id, bounds):
    indices = {a: scene.actor_ids.index(a) for a in (follower_id, challenger_id)}
    controlled = tuple(a for a, c in zip(scene.actor_ids, scene.controlled, strict=True) if c)
    columns = {a: controlled.index(a) for a in indices}
    states, motion = np.asarray(record.states), np.asarray(record.motion)
    events = {name: {actor: int(sum(t.events[name][columns[actor]] for t in record.transitions))
                     for actor in indices} for name in ("collision", "offroad", "goal")}
    gaps, ttcs = [], []
    route = controller.routes[follower_id]
    for state in states:
        f, c = (state[indices[a]] for a in (follower_id, challenger_id))
        fs, _, tangent = route.project(f[:2])
        cs, lateral, _ = route.project(c[:2])
        ef, ec = f[3]-tangent, c[3]-tangent
        fl, fw = scene.dimensions[indices[follower_id], :2]
        cl, cw = scene.dimensions[indices[challenger_id], :2]
        front = .5*(abs(np.cos(ef))*fl+abs(np.sin(ef))*fw)
        back = .5*(abs(np.cos(ec))*cl+abs(np.sin(ec))*cw)
        width = .5*(abs(np.sin(ef))*fl+abs(np.cos(ef))*fw
                    + abs(np.sin(ec))*cl+abs(np.cos(ec))*cw)
        if cs <= fs or lateral > width:
            continue
        gap = float(cs-fs-front-back)
        closing = float((f[4:6]-c[4:6]) @ [np.cos(tangent), np.sin(tangent)])
        gaps.append(gap)
        if gap > 0 and closing > 1e-6 and not (f[9] or c[9]):
            ttcs.append(gap/closing)
    violations = []
    controlled_indices = [indices[a] for a in indices]
    if not np.isfinite(states).all() or not np.isfinite(motion).all():
        violations.append("nonfinite_dynamics")
    for actor in indices:
        index = indices[actor]
        speed = np.sum(states[:, index, 4:6]*np.column_stack((
            np.cos(states[:, index, 3]), np.sin(states[:, index, 3]))), axis=1)
        if (speed < -1e-5).any():
            violations.append(actor+":reverse_motion")
        if events["offroad"][actor]:
            violations.append(actor+":offroad")
        commands = [row[columns[actor]] for row in record.diagnostics]
        # Check the geometry actually reached by every valid action, including
        # the final successor and actions that terminate the actor. Diagnostics
        # describe only pre-action states and cannot certify that last state.
        valid = np.asarray([t.valid[columns[actor]] for t in record.transitions], dtype=bool)
        assigned_route = controller.routes[actor]
        end_direction = assigned_route.points[-1]-assigned_route.points[-2]
        end_direction = end_direction/np.linalg.norm(end_direction)
        reached = []
        for xy in states[1:, index, :2][valid]:
            progress, distance, _ = assigned_route.project(xy)
            exhausted = (progress >= assigned_route.distances[-1]-1e-6
                         and float((xy-assigned_route.points[-1]) @ end_direction) > 0)
            reached.append((distance, exhausted))
        if any(c["route_exhausted"] for c in commands) or any(exhausted for _, exhausted in reached):
            violations.append(actor+":route_exhausted")
        corridor = max(1.5, bounds.maximum_route_distance_m)
        if (any(c["route_distance_m"] > corridor for c in commands)
                or any(distance > corridor for distance, _ in reached)):
            violations.append(actor+":route_tracking_outside_supported_corridor")
    if (motion[:, controlled_indices, 0] < -5.0001).any() or (motion[:, controlled_indices, 0] > 2.5001).any():
        violations.append("native_longitudinal_acceleration_bound")
    actor_metrics = {}
    for actor, index in indices.items():
        valid = np.asarray([t.valid[columns[actor]] for t in record.transitions], dtype=bool)
        # Each valid action produces the successor motion, including the final tick.
        active_motion = motion[1:, index, 0][valid]
        actor_metrics[actor] = {"active_seconds": float(valid.sum()*DT),
                               "minimum_actual_acceleration_mps2": float(active_motion.min()) if len(active_motion) else None,
                               "travel_m": float(np.linalg.norm(np.diff(states[:, index, :2], axis=0), axis=1).sum())}
    all_events = {name: {actor: int(sum(t.events[name][i] for t in record.transitions))
                        for i, actor in enumerate(controlled)} for name in ("collision", "offroad", "goal")}
    return {"events": events, "all_controlled_events": all_events, "actors": actor_metrics,
            "minimum_pair_gap_m": min(gaps) if gaps else None,
            "minimum_pretermination_pair_ttc_seconds": min(ttcs) if ttcs else None,
            "pair_geometry_samples": len(gaps), "constraint_violations": violations,
            "pair_metric_scope": "assigned-route projection; TTC only positive gap/closing and both alive; all events retained",
            "collision_fault": "not_determined"}


def _compare(scene, baseline, candidate, follower_id, challenger_id, onset):
    f = scene.actor_ids.index(follower_id)
    c = scene.actor_ids.index(challenger_id)
    controlled = [a for a, flag in zip(scene.actor_ids, scene.controlled, strict=True) if flag]
    fc, cc = controlled.index(follower_id), controlled.index(challenger_id)
    common = min(len(baseline.transitions), len(candidate.transitions))
    precontact = 0
    for t in range(common):
        if any(record.states[t][[f, c], 9:12].any() for record in (baseline, candidate)):
            break
        precontact += 1
    difference = [i for i in range(precontact)
                  if candidate.actions[i][fc] != baseline.actions[i][fc]]
    prefix = min(onset, common)
    unchanged_prefix = (np.array_equal(baseline.states[:prefix+1], candidate.states[:prefix+1])
                        and np.array_equal(baseline.actions[:prefix], candidate.actions[:prefix]))
    same_tick = (baseline.actions[onset][fc] == candidate.actions[onset][fc]
                 if onset < precontact else None)
    return {"common_steps": common, "common_precontact_action_steps": precontact,
            "actual_intervention_action_differences": sum(
                baseline.actions[i][cc] != candidate.actions[i][cc] for i in range(common)),
            "unmodified_prefix_exact": unchanged_prefix,
            "follower_same_onset_action": same_tick,
            "first_follower_action_response_seconds_after_onset":
                (difference[0]-onset)*DT if difference else None,
            "response_scope": "common active precontact interval, no mandatory response claim"}


def _risk_key(metrics, follower_id):
    ttc = metrics["minimum_pretermination_pair_ttc_seconds"]
    gap = metrics["minimum_pair_gap_m"]
    decel = metrics["actors"][follower_id]["minimum_actual_acceleration_mps2"]
    return (-metrics["events"]["collision"][follower_id],
            float("inf") if ttc is None else ttc, float("inf") if gap is None else gap,
            float("inf") if decel is None else decel)


def _gallery(output, report):
    rows = []
    for row in [report["baseline"], *report["attempts"]]:
        label = html.escape(row["id"])
        link = f'<a href="{html.escape(row["viewer"])}">{label}</a>' if row.get("viewer") else label
        rows.append(f'<tr><td>{link}</td><td>{html.escape(row["status"])}</td>'
                    f'<td>{html.escape(str(row.get("reason")))}</td><td>{row.get("steps", 0)}</td></tr>')
    title = html.escape(report["scenario_id"])
    page = ('<!doctype html><meta charset="utf-8"><title>Bounded challenge</title>'
            '<style>body{font:16px system-ui;background:#15222f;color:#eef4fa;padding:32px;max-width:1200px}'
            'a{color:#81d7ff}td,th{padding:12px;border-bottom:1px solid #435363;text-align:left}</style>'
            f'<h1>{title} · 有限纵向挑战</h1>'
            '<p>规则策略共同响应；工程难例候选，不是 PDM 分数、教师正例或高质量图像数据。'
            '每页保留实际时长，不同结束时刻不补帧。</p>'
            f'<p>选择结果：{html.escape(str(report.get("selected_candidate")))}</p>'
            f'<p>完整来源上下文：{html.escape(report.get("source_context", {}).get("status", "unavailable"))}；参考对象不参与碰撞。物理人口尚不完整。</p>'
            '<p><a href="report.json">完整协议、预算、拒绝原因与指标</a></p>'
            '<table><tr><th>分支</th><th>状态</th><th>原因</th><th>新模拟步</th></tr>'
            + ''.join(rows) + '</table>')
    (output / "index.html").write_text(page, encoding="utf-8")


def run_challenge_search(scene, road_graph, output, *, follower_id, challenger_id, fork_tick,
                         candidates, max_world_steps=2000, wall_seconds=120.,
                         route_tasks=None, following_config=None, bounds=None, scene_context=None):
    """Run a finite registered cohort; only complete supported candidates may rank.

    Candidate delays are relative to fork_tick. Bounds describe a deliberately
    narrow engineering domain, not a guarantee that every generated case is safe.
    """
    scene = copy.deepcopy(scene)
    scene.validate()
    if scene_context is not None:
        scene_context = copy.deepcopy(scene_context)
        scene_context.validate_scene(scene)
    try:
        validate_development_split(scene.metadata["split"])
    except ValueError as error:
        raise ValueError("Adaptive challenge search cannot use explicitly protected evaluation splits") from error
    graph = copy.deepcopy(road_graph)
    graph.validate_scene(scene)
    candidates = tuple(candidates)
    if not 1 <= len(candidates) <= 64 or any(not isinstance(c, BrakeCandidate) for c in candidates):
        raise ValueError("Declare 1..64 BrakeCandidate pulses before search")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidates inflate the search cohort")
    controlled = tuple(a for a, flag in zip(scene.actor_ids, scene.controlled, strict=True) if flag)
    if follower_id == challenger_id or any(a not in controlled for a in (follower_id, challenger_id)):
        raise ValueError("Follower and challenger must be distinct controlled actors")
    if type(fork_tick) is not int or not 0 <= fork_tick < len(scene.times)-1:
        raise ValueError("fork_tick must be an integer within the physical horizon")
    if not np.allclose(np.diff(scene.times), DT, atol=1e-9, rtol=0):
        raise ValueError("Challenge producer requires the declared 0.1 s physics clock")
    bounds = bounds or ChallengeBounds()
    if not isinstance(bounds, ChallengeBounds):
        raise ValueError("Expected ChallengeBounds")
    budget = _Budget(max_world_steps, wall_seconds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "kind": VERSION, "status": "running", "trainable": False,
              "scenario_id": scene.metadata["scenario_id"], "source": copy.deepcopy(scene.metadata),
              "follower_id": follower_id, "challenger_id": challenger_id, "fork_tick": fork_tick,
              "dt_seconds": DT, "bounds": asdict(bounds), "selected_candidate": None,
              "source_scope": "explicit protected split names rejected; global log separation is an upstream obligation",
              "baseline": {"id": "baseline", "status": "not_run", "reason": "pending", "steps": 0},
              "attempts": [{"id": f"candidate-{i:03d}", "candidate": asdict(c),
                            "status": "not_run", "reason": "pending", "steps": 0}
                           for i, c in enumerate(candidates)],
              "limits": ["Finite rule search, not learned adversary, PFSP, cut-in or right-of-way",
                         "No independent PDMS, teacher correction, solvability or image-quality certification",
                         "Safety bounds describe applicability; collision fault is not determined"],
              "selection_rule": "complete supported effective candidates harder than baseline; lexicographic follower collision, smaller TTC, smaller gap, stronger braking"}

    def publish(status, reason):
        report.update(status=status, reason=reason, budget=budget.report())
        for row in [report["baseline"], *report["attempts"]]:
            if row["status"] == "running":
                row.update(status="failed", reason=reason)
            elif row["reason"] == "pending":
                row["reason"] = reason
        report["attempt_counts"] = {state: sum(row["status"] == state for row in report["attempts"])
                                    for state in ("completed", "rejected", "failed", "incomplete", "no_effect", "not_run")}
        _dump(output / "report.json", report)
        _gallery(output, report)
        return report

    _dump(output / "report.json", report)
    scene.save(output / "scene.npz")
    graph.save(output / "road_graph.json")
    if scene_context is not None:
        scene_context.save(output / "context.json")
    report["source_context"] = {"status": "provided" if scene_context else "unavailable",
        "semantics": "source_only_objects_are_reference_not_collision_participants",
        "physics_population_complete": False}
    if route_tasks is not None:
        _dump(output / "route_tasks.json", route_tasks.to_dict())
    report["input_artifacts"] = {
        name: {"path": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
        for name in ("scene.npz", "road_graph.json", "route_tasks.json", "context.json") if (output / name).exists()}
    report["input_artifact_scope"] = "frozen normalized input copies; source-log files are not independently hashed"
    try:
        routes, routing = resolve_routes(scene, graph, route_tasks)
        controller = RouteFollower(routes, following_config)
        report["controller"] = controller.describe()
        report["controller_sha256"] = hashlib.sha256(json.dumps(report["controller"], sort_keys=True,
                                                               allow_nan=False).encode()).hexdigest()
        report["routing"] = routing
    except Exception as error:
        return publish("rejected", f"route_resolution:{type(error).__name__}:{error}")

    def store(name, full, initial, details, *, compare=None):
        record_path = output / (name+".npz")
        configuration = controller.describe()
        configuration["routing"] = routing
        write_rollout(scene, record_path, record=full, snapshot=initial,
                      controlled_ids=controlled, policy=VERSION, controller_configuration=configuration,
                      protocol={"kind": VERSION, "branch": name, "fork_tick": fork_tick,
                                "follower_id": follower_id, "challenger_id": challenger_id,
                                "planned_seconds": float(scene.times[-1]), "trainable": False,
                                "details": details})
        artifact = {"rollout": record_path.name,
                    "rollout_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                    "actual_seconds": len(full.transitions)*DT,
                    "comparison": "single_branch"}
        # The existing viewer rejects nonterminal zero-step recordings; retain raw evidence.
        if full.transitions:
            comparison = None
            if compare is not None and len(full.transitions) == len(compare.transitions):
                comparison = output / "baseline.npz"
                artifact["comparison"] = "same_length_baseline"
            export_replay(scene, record_path, output / (name+".html"),
                          compare_path=comparison, road_graph=graph, scene_context=scene_context)
            artifact["viewer"] = name+".html"
        return artifact

    try:
        with PufferDriveWorld(scene) as root:
            initial = root.snapshot()
            initial.save(output / "initial-snapshot.npz")
            if not root.observe().alive.all():
                return publish("rejected", "initial_controlled_actor_invalid")
            if budget.remaining() < fork_tick:
                return publish("budget_exhausted", "insufficient_prefix_step_budget")
            prefix, prefix_result = _run(root, controller, budget, end_tick=fork_tick,
                                        follower_id=follower_id, challenger_id=challenger_id, bounds=bounds)
            report["prefix"] = prefix_result
            if prefix.transitions:
                report["prefix"].update(store("prefix", prefix, initial, prefix_result))
            if root.observe().step != fork_tick or root.episode_done:
                return publish("incomplete", "prefix_ended_before_fork")
            report["pair_support"] = _support(root, controller, follower_id, challenger_id, bounds)
            if not report["pair_support"]["supported"]:
                return publish("rejected", "following_pair_outside_supported_domain")
            parent = root.snapshot()
            parent.save(output / "fork-snapshot.npz")
            report["scene_key"], report["backend_build"] = parent.scene_key, parent.build_id
            report["fork_snapshot_sha256"] = hashlib.sha256(parent.native_state).hexdigest()
            remaining = root.max_steps - fork_tick
            if budget.remaining() < remaining:
                return publish("budget_exhausted", "insufficient_complete_baseline_step_budget")
            with root.fork() as branch:
                baseline, result = _run(branch, controller, budget, end_tick=root.max_steps,
                                        follower_id=follower_id, challenger_id=challenger_id, bounds=bounds)
            full_baseline = prefix.followed_by(baseline)
            report["baseline"].update(result, status="completed" if result["stop_reason"] == "episode_boundary"
                                      else "incomplete", reason=result["stop_reason"])
            report["baseline"].update(store("baseline", full_baseline, initial, result))
            report["baseline"]["metrics"] = _metrics(scene, baseline, controller, follower_id, challenger_id, bounds)
            report["baseline"]["parent_unchanged"] = root.snapshot() == parent
            if not report["baseline"]["parent_unchanged"]:
                return publish("failed", "baseline_mutated_parent")
            if result["stop_reason"] != "episode_boundary":
                return publish("incomplete", "baseline_incomplete")
            ranked = []
            for spec, row in zip(candidates, report["attempts"], strict=True):
                if spec.duration_steps > bounds.maximum_pulse_steps:
                    row.update(status="rejected", reason="pulse_duration_outside_bound")
                    continue
                if spec.delay_steps + spec.duration_steps > remaining:
                    row.update(status="rejected", reason="pulse_exceeds_remaining_horizon")
                    continue
                reason = budget.reason()
                if reason or budget.remaining() < remaining:
                    row["reason"] = reason or "insufficient_complete_candidate_step_budget"
                    continue
                row.update(status="running", reason="executing_candidate")
                report["budget"] = budget.report()
                _dump(output / "report.json", report)
                with root.fork() as branch:
                    record, result = _run(branch, controller, budget, end_tick=root.max_steps,
                                         follower_id=follower_id, challenger_id=challenger_id, bounds=bounds,
                                         start_tick=fork_tick+spec.delay_steps, candidate=spec)
                row.update(result, reason=result["stop_reason"])
                row["parent_unchanged"] = root.snapshot() == parent
                full = prefix.followed_by(record)
                row.update(store(row["id"], full, initial, {**asdict(spec), **result}, compare=full_baseline))
                row["metrics"] = _metrics(scene, record, controller, follower_id, challenger_id, bounds)
                row["comparison_metrics"] = _compare(scene, baseline, record, follower_id, challenger_id, spec.delay_steps)
                if result["stop_reason"] != "episode_boundary":
                    row["status"] = ("rejected" if result["stop_reason"] == "intervention_start_unsupported"
                                     else "failed" if result["stop_reason"] == "runtime_error" else "incomplete")
                elif (not row["parent_unchanged"] or not row["comparison_metrics"]["unmodified_prefix_exact"]
                      or row["comparison_metrics"]["follower_same_onset_action"] is False):
                    row.update(status="failed", reason="fork_or_shared_prefix_changed")
                elif result["action_override_steps"] == 0:
                    row.update(status="no_effect", reason="no_executed_action_override")
                elif row["metrics"]["constraint_violations"]:
                    row.update(status="rejected", reason="executed_constraints_failed")
                else:
                    row.update(status="completed", reason="effective_bounded_candidate")
                    # Baseline invalidity cannot be credited as newly induced challenge.
                    if not report["baseline"]["metrics"]["constraint_violations"]:
                        harder = _risk_key(row["metrics"], follower_id) < _risk_key(report["baseline"]["metrics"], follower_id)
                        row["harder_than_baseline"] = harder
                        if harder:
                            ranked.append((_risk_key(row["metrics"], follower_id), row["id"]))
                report["budget"] = budget.report()
                _dump(output / "report.json", report)
            if ranked:
                report["selected_candidate"] = min(ranked)[1]
            report["parent_unchanged"] = root.snapshot() == parent
    except Exception as error:
        return publish("failed", f"{type(error).__name__}:{error}")
    counts = {status: sum(row["status"] == status for row in report["attempts"])
              for status in ("completed", "rejected", "failed", "incomplete", "no_effect", "not_run")}
    report["attempt_counts"] = counts
    status = "partial" if counts["not_run"] or counts["incomplete"] or counts["failed"] else "completed"
    return publish(status, "finite_cohort_finished" if status == "completed" else "finite_cohort_incomplete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--road-graph", type=Path, required=True)
    parser.add_argument("--route-overrides", type=Path)
    parser.add_argument("--context", type=Path, help="Complete source/map context; omission is explicitly incomplete")
    parser.add_argument("--follower", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--fork-tick", type=int, default=5)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="JSON list of delay_steps/duration_steps/target_acceleration_mps2")
    parser.add_argument("--max-world-steps", type=int, default=2000)
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--desired-speed", type=float, default=8.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pulses = json.loads(args.candidates.read_text())
    if not isinstance(pulses, list):
        raise ValueError("Candidates must be a JSON list")
    report = run_challenge_search(
        Scene.load(args.scene), RoadGraph.load(args.road_graph), args.output,
        follower_id=args.follower, challenger_id=args.challenger, fork_tick=args.fork_tick,
        candidates=[BrakeCandidate(**item) for item in pulses],
        max_world_steps=args.max_world_steps, wall_seconds=args.wall_seconds,
        following_config=FollowingConfig(desired_speed=args.desired_speed),
        scene_context=SceneContext.load(args.context) if args.context else None,
        route_tasks=RouteTasks.load(args.route_overrides) if args.route_overrides else None)
    print(json.dumps({"status": report["status"], "selected_candidate": report["selected_candidate"],
                      "budget": report["budget"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
