"""Executed traffic comparisons and exact-fork anchor refinement.

INTERACT-inspired intent separation/local trust regions. This uses actual rule
responses for every candidate, not the paper's SMART predictor or CEM algorithm.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.population import PopulationController
from driversi.contracts.driving_task import route_progress
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.road import Route
from driversi.contracts.scene import LANE, ROAD_EDGE, Scene
from driversi.runtime.budget import BudgetExceeded, StepBudget
from driversi.sim.persistence.recording import RolloutRecord, write_rollout


def crossing_scene():
    times = np.arange(91) * 0.1
    poses = np.zeros((len(times), 3, 4))
    poses[:, 0, 0] = -22 + 6 * times
    poses[:, 1, 1] = -22 + 6 * times
    poses[:, 1, 3] = np.pi / 2
    poses[:, 2, 0], poses[:, 2, 1] = -22 + 6 * times, 45
    velocity = np.zeros((len(times), 3, 3))
    velocity[:, 0, 0], velocity[:, 1, 1], velocity[:, 2, 0] = 6, 6, 6
    ids = ("a", "b", "far")
    routes = {
        "a": Route([[-40, 0], [100, 0]]),
        "b": Route([[0, -40], [0, 100]]),
        "far": Route([[-40, 45], [100, 45]]),
    }
    scene = Scene(
        {
            "schema_version": 1,
            "scenario_id": "cross-stage-v1",
            "source_log": "synthetic-crossing-staged-v1",
            "split": "canonical",
            "map_name": "crossing",
            "origin_xyz": [0, 0, 0],
            "map_complete": True,
            "task_goals_xy": {a: r.points[-1].tolist() for a, r in routes.items()},
            "heading_qualification": dict.fromkeys(ids, "declared_synthetic"),
        },
        ids,
        np.ones(3, np.int32),
        np.tile([4, 2, 1.5], (3, 1)),
        times,
        poses,
        velocity,
        np.ones(3, bool),
        np.array(
            [
                [-40, 0, 0],
                [100, 0, 0],
                [0, -40, 0],
                [0, 100, 0],
                [-40, 45, 0],
                [100, 45, 0],
                [-45, -45, 0],
                [105, -45, 0],
                [105, 105, 0],
                [-45, 105, 0],
                [-45, -45, 0],
            ],
            float,
        ),
        np.array([0, 2, 4, 6, 11]),
        np.array([LANE, LANE, LANE, ROAD_EDGE]),
    )
    scene.validate()
    return scene, routes


def population(routes, *, actor=None, speed=8.0, controller="route-conflict-v1"):
    policies = [PopulationPolicy("ordinary", controller, controller)]
    if actor is not None:
        policies.append(
            PopulationPolicy(
                "candidate",
                f"{controller}:speed={speed:g}",
                controller,
                (("desired_speed", float(speed)),),
            )
        )
    return PopulationController(
        routes,
        EpisodePopulation(
            tuple(policies),
            tuple(ActorPolicyBinding(a, "candidate" if a == actor else "ordinary") for a in routes),
        ),
    )


def advance(
    world, controller, steps, *, budget=None, deadline=None, intervention=None, record=None
):
    record = RolloutRecord.start(world) if record is None else record
    for _ in range(steps):
        if world.episode_done:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExceeded("Traffic search wall limit")
        obs = world.observe()
        commands = controller.commands(obs)
        if intervention is not None:
            intervention(obs, commands)
        actions = {c["actor"]: int(c["action"]) for c in commands}
        if budget is not None:
            budget.before_step(int(obs.alive.sum()))
        transition = world.step(actions)
        record.append(world, actions, transition, commands)
        if budget is not None:
            budget.progress(
                valid_transitions=budget.state["valid_transitions"] + int(transition.valid.sum())
            )
    return record


def metrics(scene, record, routes, actor):
    ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    i, slot = scene.actor_ids.index(actor), ids.index(actor)
    p = np.asarray(record.states)[:, i, :2]
    raw_progress = float(routes[actor].project(p[-1])[0] - routes[actor].project(p[0])[0])
    increments = [
        route_progress(routes[actor], a, b, corridor_m=2.0) for a, b in zip(p[:-1], p[1:])
    ]
    progress = sum(item["progress_m"] for item in increments)
    progress_supported = all(item["status"] == "supported" for item in increments)
    from driversi.evaluation.metrics.contacts import contact_report

    evidence = contact_report(scene, record.states, record.presence, record.lifecycle, actor)
    per_actor = {
        a: {
            "collision_events": sum(int(t.events["collision"][j]) for t in record.transitions),
            "offroad_events": sum(int(t.events["offroad"][j]) for t in record.transitions),
            "valid_steps": sum(int(t.valid[j]) for t in record.transitions),
        }
        for j, a in enumerate(ids)
    }
    collision = sum(int(t.events["collision"].sum()) for t in record.transitions)
    offroad = sum(int(t.events["offroad"].sum()) for t in record.transitions)
    contacts = sum(bool(f.get("contacts")) for f in record.lifecycle)
    controlled_contacts = sum(
        any(set(pair) & set(ids) for pair in f.get("contacts", [])) for f in record.lifecycle
    )
    valid = sum(bool(t.valid[slot]) for t in record.transitions)
    correction = sum(bool(c.get("traffic_modified")) for row in record.diagnostics for c in row)
    distances = [routes[actor].project(x)[1] for x in p]
    return {
        "owner": actor,
        "owner_collision_events": per_actor[actor]["collision_events"],
        "owner_offroad_events": per_actor[actor]["offroad_events"],
        "owner_contact_ticks": evidence["owner_contact_ticks"],
        "other_contact_ticks": evidence["other_contact_ticks"],
        "per_policy_actor": per_actor,
        "contact_evidence": evidence,
        "collision_events": collision,
        "offroad_events": offroad,
        "contact_ticks": contacts,
        "controlled_contact_ticks": controlled_contacts,
        "background_only_contact_ticks": contacts - controlled_contacts,
        "executed_steps": len(record.transitions),
        "owner_valid_steps": valid,
        "route_progress_m": progress,
        "raw_endpoint_progress_m": raw_progress,
        "route_progress_supported": progress_supported,
        "max_owner_route_error_m": float(max(distances)),
        "traffic_modified_commands": correction,
        "trainable": False,
        "official_score": None,
    }


@dataclass(frozen=True)
class AnchorSearchConfig:
    steps: int = 40
    anchor_speeds: tuple = (2.0, 5.0, 8.0)
    refinements: tuple = (-0.5, 0.5)
    max_deviation_m: float = 2.0
    max_attempts: int = 9
    max_agent_steps: int = 20000
    max_wall_seconds: float = 120.0

    def __post_init__(self):
        if any(
            type(x) is not int or x < 1
            for x in (self.steps, self.max_attempts, self.max_agent_steps)
        ):
            raise ValueError("Positive integer search budgets required")
        if self.steps > 90 or self.max_attempts > 100:
            raise ValueError("Search beyond bounded support")
        if (
            not self.anchor_speeds
            or not np.isfinite(self.anchor_speeds).all()
            or any(not 1 <= x <= 15 for x in self.anchor_speeds)
        ):
            raise ValueError("Anchors require speeds 1..15 m/s")
        if not np.isfinite(self.refinements).all() or any(abs(x) > 1 for x in self.refinements):
            raise ValueError("Local speed refinements must stay within 1 m/s")
        if (
            not np.isfinite([self.max_deviation_m, self.max_wall_seconds]).all()
            or min(self.max_deviation_m, self.max_wall_seconds) <= 0
        ):
            raise ValueError("Finite positive search bounds required")


def search(world, routes, actor, output, config=AnchorSearchConfig(), *, prefix=None):
    """Fork one world with fixed rule populations; never mutate the parent.

    A nonzero anchor requires its recorded prefix for time-correct replay. Search
    metrics and budget cover the newly executed suffix only. This standalone
    baseline intentionally assigns the declared rule population to every branch;
    it does not preserve arbitrary learned policies or unrecorded model memories.
    """
    from driversi.rendering.bev.replay import export_replay

    if set(routes) != set(world.actor_ids) or actor not in routes:
        raise ValueError("Search requires explicit route tasks for every controlled actor")
    if world.episode_done or not world.observe().alive[world.actor_ids.index(actor)]:
        raise ValueError("Search owner must be alive")
    if world.observe().step + config.steps > world.max_steps:
        raise ValueError("Full candidate suffix exceeds remaining scene support")
    tick = world.observe().step
    if tick:
        if prefix is None or len(prefix.transitions) != tick:
            raise ValueError("Nonzero anchor requires the actual contiguous prefix")
        if any(t.observation.step != i + 1 for i, t in enumerate(prefix.transitions)):
            raise ValueError("Search prefix must start at tick zero")
        # Check the full boundary before creating files or spending search budget.
        prefix.followed_by(RolloutRecord.start(world))
    elif prefix is not None and prefix.transitions:
        raise ValueError("Tick zero cannot have a nonempty prefix")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = world.snapshot()
    snapshot.save(output / "anchor.npz")
    world._scene.save(output / "scene.npz")
    budget = StepBudget(output / "budget.json", config.max_agent_steps, "reactive-anchor-search")
    deadline = time.monotonic() + config.max_wall_seconds
    report = {
        "version": "reactive-anchor-search-v1",
        "actor": actor,
        "anchor_tick": world.observe().step,
        "configuration": asdict(config),
        "attempts": [],
        "selected": None,
        "status": "running",
        "prediction": "exact_authoritative_world_fork_with_reactive_rule_population",
        "refinement": "bounded_speed_grid_not_CEM_or_learned_prediction",
        "selection_objective": "maximize_owner_supported_route_progress_then_smaller_absolute_refinement_then_candidate_id",
        "traffic_law_verified": False,
        "trainable": False,
        "budget_scope": "agent steps admitted before joint execution; wall checks at step boundaries; final export may overrun",
    }

    def save():
        report["agent_steps"] = budget.state["admitted_agent_steps"]
        report["parent_unchanged"] = world.snapshot() == snapshot
        (output / "search.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    records = {}
    stopped = False
    for ai, speed in enumerate(config.anchor_speeds):
        anchor_record = None
        for ri, delta in enumerate((0.0, *config.refinements)):
            if not 1 <= speed + delta <= 15:
                continue
            if len(report["attempts"]) >= config.max_attempts or time.monotonic() >= deadline:
                report["status"] = "budget_exhausted"
                stopped = True
                break
            required = len(world.actor_ids) * config.steps
            if budget.state["admitted_agent_steps"] + required > config.max_agent_steps:
                report["status"] = "budget_exhausted"
                stopped = True
                break
            name = f"anchor-{ai:02d}-variant-{ri:02d}"
            controller = population(routes, actor=actor, speed=speed + delta)
            row = {
                "id": name,
                "anchor_speed_mps": speed,
                "delta_speed_mps": delta,
                "status": "running",
            }
            report["attempts"].append(row)
            save()
            record = None
            try:
                with world.fork() as branch:
                    record = RolloutRecord.start(branch)
                    advance(
                        branch,
                        controller,
                        config.steps,
                        budget=budget,
                        deadline=deadline,
                        record=record,
                    )
                if ri == 0:
                    anchor_record = record
                m = metrics(world._scene, record, routes, actor)
                equal_length = len(record.states) == len(anchor_record.states)
                index = world._scene.actor_ids.index(actor)
                deviation = (
                    float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(record.states)[:, index, :2]
                                - np.asarray(anchor_record.states)[:, index, :2],
                                axis=1,
                            )
                        )
                    )
                    if equal_length
                    else None
                )
                failures = []
                for failed, reason in (
                    (m["executed_steps"] != config.steps, "incomplete_suffix"),
                    (m["owner_valid_steps"] != config.steps, "owner_ended_early"),
                    (bool(m["collision_events"]), "controlled_collision"),
                    (bool(m["offroad_events"]), "controlled_offroad"),
                    (bool(m["contact_ticks"]), "world_contact_including_background"),
                    (m["max_owner_route_error_m"] > 2.0, "owner_route_error"),
                    (not m["route_progress_supported"], "unsupported_route_projection"),
                    (not equal_length or deviation > config.max_deviation_m, "trust_region"),
                ):
                    if failed:
                        failures.append(reason)
                owner_failures = []
                for failed, reason in (
                    (m["executed_steps"] != config.steps, "incomplete_suffix"),
                    (m["owner_valid_steps"] != config.steps, "owner_ended_early"),
                    (bool(m["owner_collision_events"]), "owner_collision"),
                    (bool(m["owner_offroad_events"]), "owner_offroad"),
                    (bool(m["owner_contact_ticks"]), "owner_geometric_contact"),
                    (m["max_owner_route_error_m"] > 2.0, "owner_route_error"),
                    (not m["route_progress_supported"], "unsupported_route_projection"),
                    (not equal_length or deviation > config.max_deviation_m, "trust_region"),
                ):
                    if failed:
                        owner_failures.append(reason)
                # Owner engineering status is not official PDM or a dataset pass.
                # Retain the strict all-world gate and its original rejection reasons.
                accepted = not failures
                row.update(
                    owner_engineering_pass=not owner_failures,
                    owner_rejection_reasons=owner_failures,
                    world_contact_check_pass=(
                        m["executed_steps"] == config.steps
                        and not (m["collision_events"] or m["offroad_events"] or m["contact_ticks"])
                    ),
                    world_contact_check_scope="complete_suffix_geometric_contacts_and_policy_slot_events_not_all_traffic_road_legality",
                    status="completed",
                    metrics=m,
                    rejection_reasons=failures,
                    accepted=bool(accepted),
                    max_anchor_deviation_m=deviation,
                    trust_region_pass=bool(equal_length and deviation <= config.max_deviation_m),
                )
                path = output / (name + ".npz")
                write_rollout(
                    world._scene,
                    path,
                    record=prefix.followed_by(record) if prefix is not None else record,
                    snapshot=snapshot,
                    controlled_ids=world.actor_ids,
                    policy="reactive-anchor-candidate",
                    controller_configuration=controller.describe(),
                    protocol={
                        "kind": report["version"],
                        "candidate": name,
                        "anchor_tick": report["anchor_tick"],
                        "metrics": m,
                        "trainable": False,
                    },
                )
                export_replay(world._scene, path, output / (name + ".html"))
                records[name] = record
            except BudgetExceeded as exc:
                row.update(status="budget_exhausted", reason=str(exc), accepted=False)
                if record is not None:
                    row["metrics"] = metrics(world._scene, record, routes, actor)
                    complete = prefix.followed_by(record) if prefix is not None else record
                    if complete.transitions:
                        partial_path = output / (name + "-partial.npz")
                        write_rollout(
                            world._scene,
                            partial_path,
                            record=complete,
                            snapshot=snapshot,
                            controlled_ids=world.actor_ids,
                            policy="partial-reactive-anchor-candidate",
                            controller_configuration=controller.describe(),
                            protocol={
                                "kind": report["version"],
                                "candidate": name,
                                "anchor_tick": tick,
                                "status": "budget_exhausted",
                                "trainable": False,
                            },
                        )
                        export_replay(world._scene, partial_path, output / (name + "-partial.html"))
                        row["partial_rollout"] = partial_path.name
                    records[name] = record
                report["status"] = "budget_exhausted"
                stopped = True
                save()
                break
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                report["status"] = "failed"
                save()
                raise
            save()
        if stopped:
            break
    accepted = [a for a in report["attempts"] if a.get("accepted")]
    if accepted:
        report["selected"] = max(
            accepted,
            key=lambda a: (a["metrics"]["route_progress_m"], -abs(a["delta_speed_mps"]), a["id"]),
        )["id"]
    if report["status"] == "running":
        report["status"] = "completed"
    save()
    if not report["parent_unchanged"]:
        raise AssertionError("Search mutated its parent world")
    return report, records


def compare(scene, routes, output, *, steps=None):
    """Identical scene/tasks for both providers; retain every failure and actor."""
    from driversi.agents.traffic_control.following import RouteFollower
    from driversi.rendering.bev.replay import export_replay
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene.save(output / "scene.npz")
    actor = "ego" if "ego" in routes else next(iter(routes))
    report = {
        "scenario_id": scene.metadata["scenario_id"],
        "source_log": scene.metadata["source_log"],
        "map_name": scene.metadata["map_name"],
        "controlled": len(routes),
        "world_objects": len(scene.actor_ids),
        "trainable": False,
        "traffic_law_verified": False,
        "modes": {},
    }
    records = {}
    with PufferDriveWorld(scene) as world:
        root = world.snapshot()
        for name, controller in [
            ("following", RouteFollower(routes)),
            ("conflict", population(routes)),
        ]:
            world.restore(root)
            record = advance(world, controller, world.max_steps if steps is None else steps)
            report["modes"][name] = metrics(scene, record, routes, actor)
            report["modes"][name]["per_actor_progress_m"] = {
                a: float(
                    r.project(record.states[-1][scene.actor_ids.index(a), :2])[0]
                    - r.project(record.states[0][scene.actor_ids.index(a), :2])[0]
                )
                for a, r in routes.items()
            }
            records[name] = record
            write_rollout(
                scene,
                output / (name + ".npz"),
                record=record,
                snapshot=root,
                controlled_ids=world.actor_ids,
                policy=name,
                controller_configuration=controller.describe(),
                protocol={"kind": "traffic-comparison-v1", "trainable": False},
            )
            export_replay(scene, output / (name + ".npz"), output / (name + ".html"))
        world.restore(root)
        repeat = advance(world, population(routes), world.max_steps if steps is None else steps)
        from driversi.evaluation.benchmarks.interaction import _exact

        report["restore_exact"] = _exact(records["conflict"], repeat)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report, records


def main():
    import argparse

    from driversi.agents.traffic_control.routing import resolve_routes
    from driversi.contracts.road import RoadGraph
    from driversi.contracts.route_tasks import RouteTasks
    from driversi.sim.backends.pufferdrive import PufferDriveWorld

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--road-graph", type=Path, required=True)
    parser.add_argument("--route-tasks", type=Path)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--prefix-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--anchor-speeds", type=float, nargs="+", default=[2.0, 5.0, 8.0])
    parser.add_argument("--max-agent-steps", type=int, default=20000)
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix_steps < 0:
        parser.error("prefix-steps must be nonnegative")
    scene = Scene.load(args.scene)
    rs, _ = resolve_routes(
        scene,
        RoadGraph.load(args.road_graph),
        RouteTasks.load(args.route_tasks) if args.route_tasks else None,
    )
    cfg = AnchorSearchConfig(
        steps=args.steps,
        anchor_speeds=tuple(args.anchor_speeds),
        max_agent_steps=args.max_agent_steps,
        max_wall_seconds=args.max_wall_seconds,
    )
    with PufferDriveWorld(scene) as world:
        if args.prefix_steps + cfg.steps > world.max_steps:
            parser.error("prefix and suffix exceed available scene support")
        prefix = advance(world, population(rs), args.prefix_steps)
        report, _ = search(world, rs, args.actor, args.output, cfg, prefix=prefix)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
