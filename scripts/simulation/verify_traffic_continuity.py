"""Prepare and verify a continuity correction against an immutable source case."""

import argparse
import json
import shutil
from pathlib import Path

from verify_continuous_nuplan import execute

from driversi.contracts.scene import Scene
from driversi.data.road_network.lane_graph import compile_lane_graph
from driversi.data.participants.traffic_continuity import configure_traffic_continuity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = Scene.load(args.source_dir / "scene.npz")
    graph = compile_lane_graph(
        scene, Path(scene.metadata["map_file"]).parents[2], full_map_coverage=True
    )
    graph.save(args.output / "road-graph.json")
    scene = configure_traffic_continuity(scene, graph)
    scene.save(args.output / "scene.npz")
    shutil.copyfile(args.source_dir / "source.npz", args.output / "source.npz")
    preparation = json.loads((args.source_dir / "preparation.json").read_text())
    preparation.update(
        comparison="same source, ego intent, initial state and seed; explicit traffic continuity",
        baseline=str(args.source_dir),
        feedback_actors=len(scene.metadata["reactive_background"]["assignments"]),
        continuity=scene.metadata["traffic_continuity"],
        trainable=False,
    )
    (args.output / "preparation.json").write_text(json.dumps(preparation, indent=2))
    execute(args.output)


if __name__ == "__main__":
    main()
