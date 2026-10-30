"""Bounded same-state recipe trials over executed histories.

Only the stateless route controller is registered. These are engineering
candidates, never PDM-qualified labels. Models, metrics and assets are not edited.
"""

import copy
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.contracts.driving_task import route_progress, tasks_from_metadata
from driversi.contracts.road import Route
from driversi.contracts.scene import Scene
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.runtime.budget import BudgetExceeded, StepBudget, write_json
from driversi.rsi.contracts import _identity, _keys
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi.training.sampling.environment import source_log_group, validate_development_split

VERSION = "same-state-repair-v1"
VERIFIER = "supported-route-engineering-v1"


def recipe(value):
    _keys(value, ("id", "backend", "desired_speed", "headway"), "repair recipe")
    _identity(value["id"], "recipe version")
    if value["id"] in {"scene", "anchor"}:
        raise ValueError("Recipe identity conflicts with a reserved artifact")
    if value["backend"] != "route_following_v1":
        raise ValueError("Unregistered repair backend; learned teachers are not connected")
    for key, lo, hi in (("desired_speed", 1, 15), ("headway", 0.5, 4)):
        x = value[key]
        if type(x) not in (float, int) or not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError("Recipe parameter outside fixed bounds: " + key)
    return copy.deepcopy(value)


def verifier(value):
    _keys(value, ("version", "min_progress_m"), "repair verifier")
    if value["version"] != VERIFIER:
        raise ValueError("Unknown independent verifier")
    x = value["min_progress_m"]
    if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
        raise ValueError("Invalid minimum route progress")
    return copy.deepcopy(value)


def prepare(raw, base):
    _keys(
        raw,
        ("schema_version", "id", "case", "baseline", "recipes", "verifier", "budget"),
        "repair run",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported repair schema")
    _identity(raw["id"], "repair run id")
    case = raw["case"]
    _keys(
        case,
        ("scene", "rollout", "actor", "anchor_tick", "future_steps", "failure_type"),
        "repair case",
    )
    if case["failure_type"] not in (
        "collision",
        "offroad",
        "low_progress",
        "interaction",
        "diagnostic",
    ):
        raise ValueError("Unknown operator-declared failure category")
    for key in ("anchor_tick", "future_steps"):
        if type(case[key]) is not int or case[key] < (0 if key == "anchor_tick" else 1):
            raise ValueError("Invalid anchor/horizon")
    result = copy.deepcopy(raw)
    for key in ("scene", "rollout"):
        result["case"][key] = str((Path(base) / case[key]).resolve(strict=True))
    scene = Scene.load(Path(result["case"]["scene"]))
    validate_development_split(scene.metadata["split"])
    ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    tasks = tasks_from_metadata(scene.metadata, ids)
    if not tasks or case["actor"] not in ids or len(ids) < 2:
        raise ValueError("Repair requires explicit route tasks and multiple controlled actors")
    if case["anchor_tick"] + case["future_steps"] > len(scene.times) - 1:
        raise ValueError("Requested suffix exceeds source-supported scene horizon")
    result["baseline"] = recipe(raw["baseline"])
    if not isinstance(raw["recipes"], list) or not raw["recipes"]:
        raise ValueError("At least one bounded candidate required")
    result["recipes"] = [recipe(r) for r in raw["recipes"]]
    if (
        len({r["id"] for r in [result["baseline"], *result["recipes"]]})
        != len(result["recipes"]) + 1
    ):
        raise ValueError("Recipe identities must be distinct")
    result["verifier"] = verifier(raw["verifier"])
    b = raw["budget"]
    _keys(b, ("max_attempts", "max_sim_agent_steps", "max_wall_seconds"), "repair budget")
    for key in ("max_attempts", "max_sim_agent_steps"):
        if type(b[key]) is not int or b[key] < 1:
            raise ValueError("Positive integral budget required")
    if (
        type(b["max_wall_seconds"]) not in (int, float)
        or not math.isfinite(b["max_wall_seconds"])
        or b["max_wall_seconds"] <= 0
    ):
        raise ValueError("Positive wall budget required")
    result["source_log"] = source_log_group(scene.metadata["source_log"])
    result["scenario_id"] = scene.metadata["scenario_id"]
    result["inputs"] = {k: file_sha256(result["case"][k]) for k in ("scene", "rollout")}
    return result


def check_suffix(scene, record, anchor, actor, horizon, policy):
    """Fixed geometric checks computed from actual execution, not recipe claims."""
    ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    index, slot = scene.actor_ids.index(actor), ids.index(actor)
    route, task = tasks_from_metadata(scene.metadata, ids)[actor]
    suffix = record.transitions[anchor:]
    progress, support = 0.0, []
    for before, after in zip(record.states[anchor:-1], record.states[anchor + 1 :]):
        item = route_progress(
            route, before[index, :2], after[index, :2], corridor_m=task["corridor_m"]
        )
        progress += item["progress_m"]
        if item["status"] != "supported":
            support.append(item["status"])
    collisions = sum(bool(t.events["collision"].any()) for t in suffix)
    contacts = sum(bool(row.get("contacts")) for row in record.lifecycle[anchor + 1 :])
    offroad = sum(bool(t.events["offroad"].any()) for t in suffix)
    checks = {
        "full_horizon": len(suffix) == horizon,
        "owner_valid": bool(suffix) and all(bool(t.valid[slot]) for t in suffix),
        "no_collision": collisions == 0 and contacts == 0,
        "no_offroad": offroad == 0,
        "route_supported": not support
        and not any(
            bool(t.events.get("route_support_end", np.zeros(len(ids), bool))[slot]) for t in suffix
        ),
        "minimum_progress": progress >= policy["min_progress_m"],
    }
    return {
        "verifier": policy,
        "checks": checks,
        "engineering_pass": all(checks.values()),
        "route_progress_m": progress,
        "collision_ticks": collisions,
        "contact_ticks": contacts,
        "offroad_ticks": offroad,
        "support_failures": sorted(set(support)),
        "executed_steps": len(suffix),
        "reasons": [k for k, v in checks.items() if not v],
        "official_score": None,
        "trainable": False,
    }


def run_prepared(store, spec, output, *, role="development", gate_id=None, use_memory=True):
    """Execute fixed versions; failed trials remain in the ledger and on disk."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = store.register_run(spec, output, role=role, gate_id=gate_id)
    started = time.monotonic()
    account = StepBudget(output / "budget.json", spec["budget"]["max_sim_agent_steps"], "repair")
    deadline = started + spec["budget"]["max_wall_seconds"]
    state["status"] = "running"

    def persist(terminal=False):
        state["elapsed_seconds"] = time.monotonic() - started
        state["admitted_agent_steps"] = account.state["admitted_agent_steps"]
        write_json(output / "report.json", state)
        stored = copy.deepcopy(state)
        if not terminal:
            stored["status"] = "running"
        store.update_run(stored)

    def advance(world, record, actions, commands):
        if time.monotonic() >= deadline:
            raise BudgetExceeded("Wall deadline exhausted before world advance")
        account.before_step(int(world.observe().alive.sum()))
        record.append(world, actions, world.step(actions), commands)

    try:
        persist()
        for key in spec["inputs"]:
            if file_sha256(spec["case"][key]) != spec["inputs"][key]:
                raise ValueError("Source changed after run declaration")
        scene = Scene.load(Path(spec["case"]["scene"]))
        ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
        anchor, horizon = spec["case"]["anchor_tick"], spec["case"]["future_steps"]
        with np.load(spec["case"]["rollout"], allow_pickle=False) as source:
            source_meta = json.loads(str(source["metadata"]))
            if tuple(source["controlled_ids"]) != ids or len(source["actions"]) < anchor:
                raise ValueError("Source action order or history is incomplete")
            recorded_config = source_meta["controller_configuration"]
            protocol = source_meta.get("protocol", {})
            kind = protocol.get("kind")
            if kind not in (None, "two-stage-reactive-augmentation-v1", "bounded-lateral-pair"):
                raise ValueError("Source protocol requires an explicit continuation adapter")
            if kind == "two-stage-reactive-augmentation-v1":
                phases = protocol["phases"]
                if (
                    anchor < phases["exploration_end_tick"]
                    or anchor < phases["continuation_start_tick"]
                    or phases["continuation_policy"] != "stateless-idm-route-rule-diagnostic-v1"
                    or protocol["continuation_configuration"] != recorded_config["parameters"]
                ):
                    raise ValueError(
                        "Source protocol has an unsupported active continuation schedule"
                    )
            if recorded_config.get("controller") != "route-idm-v2":
                raise ValueError("Only recorded stateless route controllers are supported")
            if set(recorded_config.get("routes", {})) != set(ids):
                raise ValueError("Recorded controller routes must cover every controlled actor")
            if recorded_config.get("probe_overrides"):
                raise ValueError("Scheduled replay/brake overrides require an explicit adapter")
            lateral = recorded_config.get("lateral_intervention", {})
            if lateral.get("applied") and lateral.get("tick", 0) > anchor:
                raise ValueError("Future scheduled route changes require an explicit adapter")
            routes = {
                a: Route(
                    v["points"],
                    tuple(v.get("node_ids", ())),
                    v.get("end_reason", "unspecified"),
                    v.get("speed_limit_mps"),
                )
                for a, v in recorded_config["routes"].items()
            }
            parameters = recorded_config["parameters"]
            ordinary = RouteFollower(routes, FollowingConfig(**parameters))
            recorded_diagnostics = json.loads(str(source["controller_diagnostics"]))
            if len(recorded_diagnostics) < anchor:
                raise ValueError("Source controller diagnostics do not cover the executed prefix")
            actions = source["actions"].copy()
            expected = {k: source[k].copy() for k in ("states", "motion", "presence")}
        candidates, evidence = store.order(spec) if use_memory else (spec["recipes"], [])
        state["selection_evidence"] = evidence
        state["execution_order"] = [r["id"] for r in [spec["baseline"], *candidates]]
        with PufferDriveWorld(scene) as world:
            initial = world.snapshot()
            if (
                source_meta.get("scene_key") != initial.scene_key
                or source_meta.get("backend_build") != initial.build_id
            ):
                raise ValueError("Source recording belongs to another scene/native build")
            prefix = RolloutRecord.start(world)
            for tick in range(anchor + 1):
                for field, actual in (
                    ("states", world.state()),
                    ("motion", world.motion()),
                    ("presence", world.presence()),
                ):
                    if not np.array_equal(actual, expected[field][tick]):
                        raise ValueError(
                            "Source prefix differs at tick " + str(tick) + ": " + field
                        )
                if tick < anchor:
                    act = dict(zip(ids, map(int, actions[tick])))
                    advance(
                        world,
                        prefix,
                        act,
                        copy.deepcopy(recorded_diagnostics[tick]),
                    )
            slot = ids.index(spec["case"]["actor"])
            if world.episode_done or not world.observe().alive[slot]:
                raise ValueError("Repair owner is invalid at anchor")
            snapshot = world.snapshot()
            snapshot.save(output / "anchor.npz")
            scene.save(output / "scene.npz")
            for configuration in [spec["baseline"], *candidates]:
                remaining = (
                    spec["budget"]["max_sim_agent_steps"] - account.state["admitted_agent_steps"]
                )
                if (
                    len(state["attempts"]) >= spec["budget"]["max_attempts"]
                    or remaining < len(ids) * horizon
                    or time.monotonic() >= deadline
                ):
                    state["status"] = "budget_exhausted"
                    state["failure_reason"] = "Insufficient remaining budget for a complete suffix"
                    break
                row = {
                    "recipe": configuration,
                    "status": "running",
                    "failure_reason": None,
                    "metrics": None,
                    "charged_agent_steps": 0,
                }
                state["attempts"].append(row)
                before_charge, trial_start = account.state["admitted_agent_steps"], time.monotonic()
                persist()
                try:
                    world.restore(snapshot)
                    changed = RouteFollower(
                        routes,
                        replace(
                            ordinary.config,
                            desired_speed=configuration["desired_speed"],
                            headway=configuration["headway"],
                        ),
                    )
                    suffix = RolloutRecord.start(world)
                    for _ in range(horizon):
                        if world.episode_done:
                            break
                        observation = world.observe()
                        commands = list(ordinary.commands(observation))
                        commands[slot] = changed.commands(observation)[slot]
                        for i, command in enumerate(commands):
                            cfg = changed.config if i == slot else ordinary.config
                            command["configured_desired_speed_mps"] = cfg.desired_speed
                            command["configured_headway_seconds"] = cfg.headway
                        commands[slot]["intervention"] = "declared_repair_recipe"
                        act = {c["actor"]: c["action"] for c in commands}
                        advance(world, suffix, act, commands)
                    full = prefix.followed_by(suffix)
                    metrics = check_suffix(
                        scene, full, anchor, spec["case"]["actor"], horizon, spec["verifier"]
                    )
                    filename = configuration["id"] + ".npz"
                    write_rollout(
                        scene,
                        output / filename,
                        record=full,
                        snapshot=initial,
                        controlled_ids=ids,
                        policy=VERSION,
                        controller_configuration=copy.deepcopy(recorded_config),
                        protocol={
                            "kind": VERSION,
                            "recipe": configuration,
                            "actor": spec["case"]["actor"],
                            "anchor_tick": anchor,
                            "future_steps": horizon,
                            "trainable": False,
                            "input_use": "current_structured_state_and_declared_routes",
                            "other_actors_reobserve": True,
                        },
                    )
                    row.update(status="completed", metrics=metrics, rollout=filename)
                except BudgetExceeded as exc:
                    row.update(status="budget_exhausted", failure_reason=str(exc))
                    state.update(status="budget_exhausted", failure_reason=str(exc))
                except Exception as exc:
                    row.update(status="error", failure_reason=str(exc))
                finally:
                    row["charged_agent_steps"] = (
                        account.state["admitted_agent_steps"] - before_charge
                    )
                    row["wall_seconds"] = time.monotonic() - trial_start
                    persist()
                if state["status"] == "budget_exhausted":
                    break
            if state["status"] == "running":
                state["status"] = "completed"
            eligible = [
                r
                for r in state["attempts"]
                if r["status"] == "completed" and r["metrics"]["engineering_pass"]
            ]
            state["selected_recipe"] = (
                max(eligible, key=lambda r: r["metrics"]["route_progress_m"])["recipe"]["id"]
                if eligible
                else None
            )
            state["anchor_verified"] = True
            state["untested_recipes"] = [
                r
                for r in state["execution_order"]
                if r not in {a["recipe"]["id"] for a in state["attempts"]}
            ]
    except BudgetExceeded as exc:
        state.update(status="budget_exhausted", failure_reason=str(exc))
    except Exception as exc:
        state.update(status="failed", failure_reason=str(exc))
    if state["status"] == "completed" and time.monotonic() > deadline:
        state.update(
            status="budget_exhausted", failure_reason="Complete execution exceeded wall budget"
        )
    persist(terminal=True)
    from driversi.rsi.correction.viewer import write_viewer

    write_viewer(output, state)
    return state


def run(store, raw, output, *, base=Path(".")):
    return run_prepared(store, prepare(raw, base), output)
