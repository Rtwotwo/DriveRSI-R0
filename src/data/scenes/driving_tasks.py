"""Bind declared lane-graph intentions into executable, versioned scene tasks."""

import argparse
import json
from pathlib import Path

from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import RoadGraph
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--road-graph", required=True, type=Path)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--destination", nargs="*", default=[])
    parser.add_argument("--corridor-m", type=float, default=2.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = Scene.load(args.scene)
    routes, report = resolve_routes(
        scene,
        RoadGraph.load(args.road_graph),
        RouteTasks.load(args.intent) if args.intent else None,
    )
    result = bind_driving_tasks(
        scene, routes, destinations=args.destination, corridor_m=args.corridor_m
    )
    result.save(args.output)
    print(json.dumps({"output": str(args.output), "version": "route-task-v1", "routing": report}))


if __name__ == "__main__":
    main()
