"""Bounded action-executed lateral interaction probes, not learned adversaries.

The paired worlds start identically. After intervention_tick, only one actor's
controller route changes. Everyone observes the new world at each later tick.
Results retain failed bounds and unknown legality; no automatic training admission.
"""

from dataclasses import asdict, dataclass

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.contracts.road import Route
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord
from driversi.training.sampling.environment import validate_development_split


@dataclass(frozen=True)
class LateralBounds:
    maximum_speed_mps: float = 15.0
    maximum_lateral_acceleration_mps2: float = 3.0
    maximum_route_error_m: float = 2.0
    minimum_initial_gap_m: float = 2.0
    maximum_initial_gap_m: float = 45.0

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Lateral bounds must be finite and positive")
        if self.minimum_initial_gap_m >= self.maximum_initial_gap_m:
            raise ValueError("Initial gap bounds are reversed")


def blend_lane_change(origin: Route, target: Route, start_xy, *, length_m=30.0):
    """Quintic lateral-intent blend across two locally aligned route baselines.

    Checks geometry only. It does not authorize crossing a lane marking or create
    road surface. Actual tracking, accelerations and contacts are checked later.
    """
    if not np.isfinite(length_m) or not 15 <= length_m <= 80:
        raise ValueError("Lane change length must be15..80m")
    start, dist, heading = origin.project(start_xy)
    start_target, separation, heading_target = target.project(origin.at(start))
    error = abs(np.arctan2(np.sin(heading - heading_target), np.cos(heading - heading_target)))
    if dist > 1 or not 2 <= separation <= 5 or error > 0.25:
        raise ValueError("Lane change requires nearby, locally aligned routes")
    if min(origin.distances[-1] - start, target.distances[-1] - start_target) < length_m + 8:
        raise ValueError("Insufficient route support for lane change and continuation")
    distances = np.linspace(0, length_m, int(np.ceil(length_m)) + 1)
    u = distances / length_m
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    points = np.asarray(
        [
            (1 - w) * origin.at(start + d) + w * target.at(start_target + d)
            for d, w in zip(distances, blend)
        ]
    )
    tail = np.arange(start_target + length_m + 1, target.distances[-1], 1.0)
    points = (
        np.vstack((points, [target.at(s) for s in tail], target.points[-1]))
        if len(tail)
        else np.vstack((points, target.points[-1]))
    )
    return Route(points, target.node_ids, "lateral_intent_support", target.speed_limit_mps)


def run_lateral_pair(
    scene,
    routes,
    candidate_route,
    *,
    challenger,
    responder,
    maneuver="cut_in",
    intervention_tick=10,
    steps=None,
    config=FollowingConfig(),
    bounds=LateralBounds(),
    step_guard=None,
):
    if maneuver not in {"cut_in", "merge"}:
        raise ValueError("Supported lateral probes are cut_in and merge")
    validate_development_split(scene.metadata["split"])
    ids = tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c)
    if (
        set(routes) != set(ids)
        or challenger == responder
        or not {challenger, responder} <= set(ids)
    ):
        raise ValueError("Pair and routes must reference distinct controlled actors")
    steps = len(scene.times) - 1 if steps is None else steps
    if (
        type(steps) is not int
        or not 2 <= steps < len(scene.times)
        or type(intervention_tick) is not int
        or not 0 <= intervention_tick < steps - 1
    ):
        raise ValueError("Probe needs a bounded horizon with post-intervention evidence")
    ci, ri = (scene.actor_ids.index(a) for a in (challenger, responder))
    cslot, rslot = (ids.index(a) for a in (challenger, responder))
    # Declared route interaction geometry, no unknown signal/right-of-way claims.
    initial_c = scene.poses[0, ci, :2]
    initial_r = scene.poses[0, ri, :2]
    forward_gap = routes[responder].project(initial_c)[0] - routes[responder].project(initial_r)[0]
    gap = forward_gap - (scene.dimensions[ci, 0] + scene.dimensions[ri, 0]) / 2
    if not bounds.minimum_initial_gap_m <= gap <= bounds.maximum_initial_gap_m:
        raise ValueError("Initial pair gap outside declared challenge bounds")
    if maneuver == "merge":
        shared = set(candidate_route.node_ids) & set(routes[responder].node_ids)
        if not shared or candidate_route.node_ids[-1] != routes[responder].node_ids[-1]:
            raise ValueError("Merge requires a declared shared downstream lane")
        if routes[responder].project(candidate_route.points[-1])[1] > 0.25:
            raise ValueError("Merge downstream geometry does not meet responder route")
    controller = RouteFollower(routes, config)
    alternative = RouteFollower({**routes, challenger: candidate_route}, config)
    records = []
    for branch in range(2):
        with PufferDriveWorld(scene) as world:
            if not world.observe().alive.all():
                raise ValueError("Lateral probe has an initially invalid actor")
            record = RolloutRecord.start(world)
            snapshot = world.snapshot()
            while not world.episode_done and world.observe().step < steps:
                observation = world.observe()
                commands = list(controller.commands(observation))
                if branch and observation.step >= intervention_tick:
                    commands[cslot] = list(alternative.commands(observation))[cslot]
                actions = {cmd["actor"]: cmd["action"] for cmd in commands}
                if step_guard is not None:
                    step_guard()
                transition = world.step(actions)
                record.append(world, actions, transition, commands)
            records.append(record)
    baseline, challenge = records
    common = min(len(baseline.actions), len(challenge.actions))
    a, b = np.asarray(baseline.actions), np.asarray(challenge.actions)
    changed_c = np.flatnonzero(a[:common, cslot] != b[:common, cslot])
    changed_r = np.flatnonzero(a[:common, rslot] != b[:common, rslot])
    first_c = int(changed_c[0]) if len(changed_c) else None
    first_r = int(changed_r[0]) if len(changed_r) else None
    metrics = []
    for branch, record in enumerate(records):
        state, motion = np.asarray(record.states), np.asarray(record.motion)
        valid = np.asarray([t.valid for t in record.transitions])
        pair_valid = valid[:, cslot] & valid[:, rslot]
        present = np.asarray(record.presence)
        tracking_route = candidate_route if branch else routes[challenger]
        errors = [tracking_route.project(p)[1] for p in state[intervention_tick + 1 :, ci, :2]]
        collision = any(t.events["collision"].any() for t in record.transitions)
        offroad = any(t.events["offroad"].any() for t in record.transitions)
        speed = np.linalg.norm(state[:, :, 4:6], axis=-1)
        measured_speed = float(speed[:, scene.controlled].max())
        measured_lat = float(abs(motion[:, scene.controlled, 1]).max())
        target_distance = routes[responder].project(state[-1, ci, :2])[1]
        end_heading = routes[responder].project(state[-1, ci, :2])[2]
        heading_error = float(
            abs(
                np.arctan2(
                    np.sin(state[-1, ci, 3] - end_heading), np.cos(state[-1, ci, 3] - end_heading)
                )
            )
        )
        failures = []
        if collision:
            failures.append("collision_in_world")
        if offroad:
            failures.append("offroad_in_world")
        if measured_speed > bounds.maximum_speed_mps:
            failures.append("speed_bound")
        if measured_lat > bounds.maximum_lateral_acceleration_mps2:
            failures.append("lateral_acceleration_bound")
        if errors and max(errors) > bounds.maximum_route_error_m:
            failures.append("route_tracking_bound")
        metrics.append(
            {
                "ticks": len(record.actions),
                "pair_valid_seconds": float(pair_valid.sum() * 0.1),
                "collision": collision,
                "offroad": offroad,
                "maximum_speed_mps": measured_speed,
                "maximum_lateral_acceleration_mps2": measured_lat,
                "target_route_distance_m": float(target_distance),
                "target_heading_error_rad": heading_error,
                "all_actor_presence_samples": int(present.sum()),
                "maneuver_completed": bool(target_distance < 0.75 and heading_error < 0.2),
                "bounds_failures": failures,
            }
        )
    prefix_equal = all(
        np.array_equal(x, y)
        for x, y in zip(
            baseline.states[: intervention_tick + 1], challenge.states[: intervention_tick + 1]
        )
    )
    causal = first_r is None or first_c is not None and first_r > first_c
    report = {
        "schema_version": 1,
        "kind": "bounded-lateral-pair",
        "maneuver": maneuver,
        "challenger": challenger,
        "responder": responder,
        "intervention_tick": intervention_tick,
        "initial_gap_m": float(gap),
        "bounds": asdict(bounds),
        "requested_ticks": steps,
        "first_challenger_action_difference_tick": first_c,
        "first_responder_action_difference_tick": first_r,
        "common_prefix_equal": prefix_equal,
        "causal_action_order": causal,
        "baseline": metrics[0],
        "challenge": metrics[1],
        "trainable": False,
        "interaction_evidence": bool(
            prefix_equal
            and causal
            and first_c is not None
            and first_r is not None
            and not metrics[1]["bounds_failures"]
        ),
        "limitations": [
            "route-intent intervention only; not a learned adversary",
            "lane marking permission and right-of-way unknown",
            "no human-behavior realism or official benchmark qualification",
        ],
    }
    report["task_semantics"] = {
        "authoritative_tasks": "frozen_identically_across_branches",
        "controller_intent": "only_challenger_reference_changes_after_intervention",
        "completion": "candidate_does_not_change_world_goal_or_termination",
    }
    report["evidence_semantics"] = "bounded_causal_action_response_not_completed_maneuver"
    report["maneuver_validation_pass"] = bool(
        report["interaction_evidence"]
        and not metrics[0]["bounds_failures"]
        and all(m["ticks"] == steps for m in metrics)
        and metrics[1]["maneuver_completed"]
    )
    return report, records, snapshot


def export_lateral_pair(scene, graph, routes, candidate, report, records, snapshot, output,
                        *, source_scene_path, route_report=None, scene_context=None, config=FollowingConfig()):
    """One durable source package for BEV inspection and exact GS action replay."""
    import copy
    import json
    from pathlib import Path

    from driversi.data.assets.mtgs_assets import file_sha256
    from driversi.rendering.bev.replay import export_replay
    from driversi.sim.persistence.recording import write_rollout

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = copy.deepcopy(report)
    report.update(scenario_id=scene.metadata["scenario_id"], scene_key=snapshot.scene_key,
                  backend_build=snapshot.build_id, source_scene=str(Path(source_scene_path).resolve()),
                  candidate_route=candidate.to_dict(), route_resolution=route_report,
                  export_status="writing")
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    scene.save(output / "scene.npz")
    graph.save(output / "road_graph.json")
    controller = RouteFollower(routes, config)
    branches = {}
    for name, record in zip(("baseline", "challenge"), records, strict=True):
        configuration = controller.describe()
        if name == "challenge":
            configuration["routes"][report["challenger"]] = candidate.to_dict()
        configuration["lateral_intervention"] = {
            "actor": report["challenger"], "tick": report["intervention_tick"],
            "candidate_route": candidate.to_dict(), "applied": name == "challenge",
        }
        path = output / (name + ".npz")
        write_rollout(scene, path, record=record, snapshot=snapshot, controlled_ids=tuple(routes),
                      policy="bounded-lateral-route-intent-v1", controller_configuration=configuration,
                      protocol={**report, "branch": name})
        branches[name] = {"rollout": path.name, "sha256": file_sha256(path)}
    export_replay(scene, output / "baseline.npz", output / "viewer.html", road_graph=graph,
                  compare_path=output / "challenge.npz", scene_context=scene_context)
    report.update(export_status="complete", branches=branches, input_artifacts={
        name: {"path": name, "sha256": file_sha256(output / name)}
        for name in ("scene.npz", "road_graph.json")
    })
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    """Explicit, bounded candidate; caller selects actor IDs and route intent."""
    import argparse
    import json
    from pathlib import Path

    from driversi.agents.traffic_control.routing import resolve_routes
    from driversi.contracts.context import SceneContext
    from driversi.contracts.driving_task import bind_driving_tasks
    from driversi.contracts.road import RoadGraph
    from driversi.contracts.route_tasks import RouteTasks
    from driversi.contracts.scene import Scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--road-graph", type=Path, required=True)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--scene-context", type=Path)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--responder", required=True)
    parser.add_argument("--length-m", type=float, default=30.0)
    parser.add_argument("--intervention-tick", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = Scene.load(args.scene)
    graph = RoadGraph.load(args.road_graph)
    routes, route_report = resolve_routes(
        scene, graph, RouteTasks.load(args.intent) if args.intent else None
    )
    source_scene = scene
    scene = bind_driving_tasks(scene, routes)
    context = (
        SceneContext.load(args.scene_context).with_driving_tasks(source_scene, scene)
        if args.scene_context else None
    )
    if args.challenger not in routes or args.responder not in routes:
        raise ValueError("Actor IDs must have resolved controlled routes")
    if not 0 <= args.intervention_tick < len(scene.times) - 2:
        raise ValueError("Intervention tick outside source horizon")
    controller = RouteFollower(routes)
    with PufferDriveWorld(scene) as preview:
        for _ in range(args.intervention_tick):
            if preview.episode_done:
                raise ValueError("World ended before intervention")
            preview.step({c["actor"]: c["action"] for c in controller.commands(preview.observe())})
        xy = preview.state()[scene.actor_ids.index(args.challenger), :2]
    candidate = blend_lane_change(
        routes[args.challenger], routes[args.responder], xy, length_m=args.length_m
    )
    report, records, snapshot = run_lateral_pair(
        scene,
        routes,
        candidate,
        challenger=args.challenger,
        responder=args.responder,
        intervention_tick=args.intervention_tick,
    )
    export_lateral_pair(
        scene, graph, routes, candidate, report, records, snapshot, args.output,
        source_scene_path=args.scene, route_report=route_report, scene_context=context,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "common_prefix_equal",
                    "causal_action_order",
                    "interaction_evidence",
                    "baseline",
                    "challenge",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
