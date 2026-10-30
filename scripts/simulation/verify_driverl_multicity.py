"""Fixed eight-log preparation and native evaluation; failure cases are retained."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import RoadGraph, Route
from driversi.contracts.scene import CYCLIST, PEDESTRIAN, Scene
from driversi.data.participants.background import configure_background, propose_logged_tasks
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.data.road_network.lane_graph import compile_lane_graph
from driversi.data.road_network.maps import attach_nuplan_map
from driversi.data.sources.nuplan_db import compile_window, read_window
from driversi.data.road_network.route_continuation import extend_ego_route
from driversi.data.participants.traffic_continuity import configure_traffic_continuity
from driversi_agents.driverl.closed_loop import native_scenario


def prepare(manifest, output, maps_root, seconds=60, reuse_source=None):
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads(manifest.read_text())
    if reuse_source is None:
        with sqlite3.connect("file:" + source["db_path"] + "?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT token,timestamp FROM lidar_pc ORDER BY timestamp"
            ).fetchall()
        anchor = next(row for row in rows if row[1] - rows[0][1] >= 1_500_000)
        duration = float(np.floor(min(seconds, (rows[-1][1] - anchor[1]) / 1e6) * 10) / 10)
        window = read_window(source["db_path"], token=anchor[0].hex(), duration=duration)
        scene = compile_window(window, duration=duration, max_actors=1, split="unassigned")
        attach_nuplan_map(scene, maps_root, margin=500)
    else:
        scene = Scene.load(reuse_source / "source.npz")
        duration = float(scene.times[-1])
    scene.save(output / "source.npz")
    tasks, _ = propose_logged_tasks(scene)
    peds = {
        a: t
        for a, t in tasks.items()
        if scene.kinds[scene.actor_ids.index(a)] in (PEDESTRIAN, CYCLIST)
    }
    scene = configure_vehicle_fleet(configure_background(scene, peds) if peds else scene)
    points = scene.poses[:, scene.actor_ids.index("ego"), :2]
    # Route waypoints describe navigation, not sub-centimetre stationary label drift.
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= 1.0:
            kept.append(point)
    if np.linalg.norm(points[-1] - kept[-1]) > 1e-4:
        kept.append(points[-1])
    points = np.asarray(kept)
    scene = bind_driving_tasks(scene, {"ego": Route(points, end_reason="source_support_end")})
    graph = (
        compile_lane_graph(scene, maps_root, full_map_coverage=True)
        if reuse_source is None
        else RoadGraph.load(reuse_source / "road-graph.json")
    )
    graph.save(output / "road-graph.json")
    scene = configure_traffic_continuity(scene, graph)
    native = native_scenario(scene, maps_root)
    goal = native.get_mission_goal()
    goal_local = (
        None
        if goal is None
        else np.asarray([goal.x, goal.y]) - np.asarray(scene.metadata["origin_xyz"][:2])
    )
    scene = extend_ego_route(scene, graph, native.get_route_roadblock_ids(), goal_local)
    scene.save(output / "scene.npz")
    result = dict(
        status="prepared",
        source_log=source["source_log"],
        city=source["map_name"],
        duration=duration,
        actors=len(scene.actor_ids),
        route=scene.metadata["ego_route_continuation"],
        trainable=False,
    )
    (output / "preparation.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparation-name", default="prepared")
    p.add_argument("--reuse-preparation")
    p.add_argument("--logs", nargs="+", default=[f"log-{i:02}" for i in range(1, 9)])
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    for log in a.logs:
        output = a.output / log / a.preparation_name
        try:
            result = prepare(
                root / "outputs/demonstrations/nuplan-capacity-v1/logs" / log / "manifest.json",
                output,
                Path("/data/xinyao/navsim_workspace/dataset/maps"),
                reuse_source=None
                if a.reuse_preparation is None
                else a.output / log / a.reuse_preparation,
            )
        except Exception as e:
            result = dict(status="preparation_failed", error=repr(e), trainable=False)
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(json.dumps(result, indent=2))
        print(log, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
