"""Paired lead-braking and negative controls; no learning or policy ranking."""

import json
import time
import tomllib
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower, quantize_jerk
from driversi.contracts.road import Route
from driversi.contracts.transition import Observation
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout

RESPONSE_TOLERANCE_MPS2 = 1e-5


def _decide(world, follower, speed, changed, reduction):
    observation = world.observe()
    joint, commands = {}, []
    for i, actor in enumerate(observation.actor_ids):
        if actor == "follower":
            part = Observation((actor,), observation.features[[i]], observation.alive[[i]],
                               observation.step, observation.metric.select([i]))
            diagnostic = follower.commands(part)[0]
            joint[actor] = diagnostic["action"]
            commands.append(diagnostic)
        else:
            ego = observation.metric.ego[i]
            current_speed = float(ego[3]*np.cos(ego[2]) + ego[4]*np.sin(ego[2]))
            target = speed - (reduction if actor == changed else 0)
            desired = float(np.clip(1.5*(target-current_speed), -3, 1.5))
            joint[actor] = quantize_jerk(desired, 0, ego[7:9], current_speed)
            commands.append({"actor": actor, "action": joint[actor],
                             "controller": "longitudinal-speed-v1", "desired_acceleration": desired,
                             "desired_lateral_acceleration": 0.0, "actual_speed_mps": current_speed,
                             "target_speed_mps": target})
    return joint, commands


def _run(world, follower, speed, changed, reduction, *, steps=None):
    record = RolloutRecord.start(world)
    while not world.episode_done and (steps is None or len(record.transitions) < steps):
        joint, commands = _decide(world, follower, speed, changed, reduction)
        transition = world.step(joint)
        record.append(world, joint, transition, commands)
    result = {"states": np.asarray(record.states), "motion": np.asarray(record.motion),
              "actions": np.asarray(record.actions),
              "valid": np.asarray([transition.valid for transition in record.transitions]),
              "diagnostics": [next((command for command in commands if command["actor"] == "follower"), None)
                              for commands in record.diagnostics]}
    return result, record


def _overlapping_pair(state):
    boxes = [rectangle_corners(*state[i, :2], state[i, 3], 4, 2) for i in (0, 1)]
    return rectangles_overlap(*boxes)


def _comparison_end(states):
    """First contact OR stopped actor; neither is included in causal comparison."""
    for t, state in enumerate(states):
        if _overlapping_pair(state) or state[0, 9] or state[1, 9]:
            return t
    return len(states)


def _metrics(result):
    states = result["states"]
    gaps = states[:, 1, 0] - states[:, 0, 0] - 4
    closing = states[:, 0, 4] - states[:, 1, 4]
    ttc = gaps[(closing > 1e-6) & (gaps > 0)] / closing[(closing > 1e-6) & (gaps > 0)]
    return {"minimum_bumper_gap_m": float(gaps.min()),
            "minimum_ttc_seconds": float(ttc.min()) if len(ttc) else None,
            "native_collision_any": bool(states[:, :2, 10].any()),
            "geometric_contact_any": any(_overlapping_pair(state) for state in states),
            "follower_displacement_m": float(states[-1, 0, 0]-states[0, 0, 0])}


def run_case(speed: float, gap: float, *, duration: float = 9, intervention: float = 1,
             reduction: float = 3, maximum_response: float = 2, output: Path | None = None) -> dict:
    if not 0 < intervention < duration or not np.isclose(intervention*10, round(intervention*10)):
        raise ValueError("Intervention must precede the horizon on the 0.1 s clock")
    if not np.isfinite([reduction, maximum_response]).all() or not 0 < reduction < speed or maximum_response <= 0:
        raise ValueError("Require a positive speed reduction below initial speed and a positive response budget")
    follower = RouteFollower({"follower": Route([[-30, 0], [250, 0]])}, FollowingConfig(desired_speed=speed))
    checks, modes, valid_steps = {}, {}, 0
    for replay in (False, True):
        mode = "replay" if replay else "reactive"
        scene = following_scene(speed, gap, replay=replay, duration=duration)
        branches = {}
        with PufferDriveWorld(scene) as root:
            initial_snapshot = root.snapshot()
            _, common_prefix = _run(root, follower, speed, None, reduction, steps=round(intervention*10))
            snapshot = root.snapshot()
            parent = root.state()
            if output:
                scene.save(output / f"{mode}-scene.npz")
                snapshot.save(output / f"{mode}-snapshot.npz")
            for branch, changed in (("keep", None), ("brake", "lead"), ("far", "far")):
                with root.fork() as world:
                    result, branch_record = _run(world, follower, speed, changed, reduction)
                    branches[branch] = result
                    valid_steps += int(result["valid"].sum())
                checks[f"{mode}_{branch}_parent_unchanged"] = np.array_equal(root.state(), parent)
                with root.fork() as repeated:
                    if output:
                        from driversi.contracts.transition import Snapshot

                        repeated.restore(Snapshot.load(output / f"{mode}-snapshot.npz"))
                    again, _ = _run(repeated, follower, speed, changed, reduction)
                    checks[f"{mode}_{branch}_exact_replay"] = all(
                        np.array_equal(result[key], again[key]) for key in ("states", "motion", "actions", "valid"))
                if output:
                    output.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output / f"{mode}-{branch}.npz",
                                        **{key: value for key, value in result.items() if key != "diagnostics"},
                                        diagnostics=json.dumps(result["diagnostics"], allow_nan=False))
                    configuration = {"controller": "paired-following-v1",
                                     "controlled_actor_controllers": {
                                         actor: "route-idm-v2" if actor == "follower" else "longitudinal-speed-v1"
                                         for actor in root.actor_ids},
                                     "longitudinal_speed_parameters": {"gain": 1.5, "acceleration_bounds": [-3, 1.5]},
                                     "routes": {} if replay else follower.describe()["routes"]}
                    if not replay:
                        configuration["follower"] = follower.describe()
                    write_rollout(
                        scene, output / f"{mode}-{branch}-rollout.npz",
                        record=common_prefix.followed_by(branch_record), snapshot=initial_snapshot,
                        controlled_ids=root.actor_ids, policy=f"paired-following-{mode}-{branch}",
                        controller_configuration=configuration,
                        protocol={"name": "paired-following-v1", "mode": mode, "branch": branch,
                                  "changed_actor": changed, "intervention_seconds": intervention,
                                  "common_prefix_steps": len(common_prefix.transitions),
                                  "speed_mps": speed, "initial_bumper_gap_m": gap,
                                  "speed_reduction_mps": reduction,
                                  "metric_interval_start_seconds": intervention,
                                  "intervention_timing": "joint_action_from_current_observation"})
            keep, brake, far = (branches[name] for name in ("keep", "brake", "far"))
            checks[f"{mode}_far_negative"] = np.array_equal(keep["states"][:, 0], far["states"][:, 0])
            prefix = min(_comparison_end(keep["states"]), _comparison_end(brake["states"]))
            checks[f"{mode}_nonempty_precontact"] = prefix >= 2
            metrics = {name: _metrics(value) for name, value in branches.items()}
            if replay:
                checks["replay_precontact_unchanged"] = np.array_equal(
                    keep["states"][:prefix, 0, :7], brake["states"][:prefix, 0, :7])
                metrics["precontact_comparison_seconds"] = max(0, prefix-1)*.1
            else:
                delta_target = np.asarray([b["desired_acceleration"]-a["desired_acceleration"]
                                           for a, b in zip(keep["diagnostics"], brake["diagnostics"], strict=True)])
                delta_actual = brake["motion"][:prefix, 0, 0] - keep["motion"][:prefix, 0, 0]
                target_indices = np.flatnonzero(abs(delta_target[:prefix]) > RESPONSE_TOLERANCE_MPS2)
                actual_indices = np.flatnonzero(abs(delta_actual) > RESPONSE_TOLERANCE_MPS2)
                target_delay = float(target_indices[0]*.1) if len(target_indices) else None
                actual_delay = float(actual_indices[0]*.1) if len(actual_indices) else None
                checks["no_same_tick_anticipation"] = (abs(delta_target[0]) < 1e-10
                                                      and keep["actions"][0, 0] == brake["actions"][0, 0])
                checks["target_response_next_observation"] = bool(
                    target_delay is not None and np.isclose(target_delay, .1) and delta_target[target_indices[0]] < 0)
                checks["executed_response_after_observation"] = bool(
                    actual_delay is not None and .2-1e-9 <= actual_delay <= maximum_response
                    and delta_actual[actual_indices[0]] < 0)
                metrics.update({"desired_response_delay_s": target_delay, "actual_response_delay_s": actual_delay,
                                "acceleration_comparison_state_samples": prefix,
                                "acceleration_comparison_last_time_s": max(0, prefix-1)*.1,
                                "mean_absolute_acceleration_difference": float(np.abs(delta_actual).mean())})
            modes[mode] = metrics
            checks[f"{mode}_finite"] = all(np.isfinite(value[key]).all()
                                          for value in branches.values() for key in ("states", "motion"))
    checks = {name: bool(value) for name, value in checks.items()}
    report = {"speed_mps": speed, "initial_bumper_gap_m": gap, "checks": checks,
              "passed": all(checks.values()), "modes": modes, "valid_agent_steps_primary_branches": valid_steps,
              "controller": follower.describe()}
    if output:
        (output / "case.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def evaluate_following(config_path: Path, output: Path) -> dict:
    from driversi.sim import _native

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation configuration")
    speeds, gaps = config["speeds_mps"], config["bumper_gaps_m"]
    if not speeds or not gaps:
        raise ValueError("An empty suite cannot pass")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.toml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    report = {"schema_version": 1, "protocol": "paired-following-v1", "status": "running",
              "backend_build": _native.build_id, "dt_seconds": .1, "config": config,
              "response_tolerance_mps2": RESPONSE_TOLERANCE_MPS2,
              "independent_templates": 1, "cases": []}
    started = time.perf_counter()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for speed in speeds:
        for gap in gaps:
            index = len(report["cases"])
            report["cases"].append(run_case(
                speed, gap, duration=config["duration_seconds"], intervention=config["intervention_seconds"],
                reduction=config["speed_reduction_mps"], maximum_response=config["maximum_response_seconds"],
                output=output / f"case-{index:03d}"))
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report.update(status="complete", passed=all(case["passed"] for case in report["cases"]),
                  elapsed_seconds=time.perf_counter()-started)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {"protocol": report["protocol"], "cases": len(report["cases"]),
            "passed": report["passed"], "output": str(output)}
