"""Resume export/scoring of a completed rollout without resimulating it."""

import argparse
import hashlib
import json
from pathlib import Path

from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.evaluation.metrics.nuplan_metrics import score_record
from driversi.rendering.inspection.driverl import export_driverl_gallery
from driversi.rendering.bev.replay import export_replay, export_source_replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    scene = Scene.load(a.output / "scene.npz")
    report = json.loads((a.output / "report.json").read_text())
    rows = json.loads((a.output / "native-history.json").read_text())
    if len(rows) != report["steps"]:
        raise ValueError("Execution/history lengths differ")
    graph = (
        RoadGraph.load(a.output / "road-graph.json")
        if (a.output / "road-graph.json").exists()
        else None
    )
    # Strict replay validation checks scene identity and real executed states first.
    export_replay(
        scene,
        a.output / "rollout.npz",
        a.output / "viewer.html",
        road_graph=graph,
        compress_payload=True,
    )
    export_source_replay(
        Scene.load(a.output / "source.npz"),
        a.output / "source.html",
        inspection={"title": "nuPlan源日志；非控制器仿真"},
    )
    target = a.output / "official-score.json"
    if not target.exists():
        result = score_record(
            scene,
            rows,
            a.output / "official-metrics",
            root / "agents/driverl/native/DriveRL",
            Path("/data/xinyao/navsim_workspace/dataset/maps"),
        )
        target.write_text(json.dumps(result, indent=2, allow_nan=False))
    export_driverl_gallery(a.output)
    evidence = {
        "resimulated": False,
        "existing_score_preserved": True,
        "rollout_sha256": hashlib.sha256((a.output / "rollout.npz").read_bytes()).hexdigest(),
        "status": "finalized",
    }
    (a.output / "finalization.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
