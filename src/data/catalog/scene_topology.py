"""Anchor map facts for scene clustering, independent of image-vector retrieval."""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .scene_hierarchy import digest, save_json

TOPOLOGY_FEATURES = (
    "inside_intersection", "intersection_distance_m", "nearby_lanes_30m",
    "nearby_connectors_30m", "crosswalk_distance_m", "inside_lane_connector",
    "lane_incoming_edges", "lane_outgoing_edges", "lane_group_lanes",
)


def read_anchors(item):
    from driversi.data.sources.navsim import load_log

    path, rows = item
    frames = load_log(Path(path), trusted=True)
    result = []
    for row in rows:
        frame = frames[row["anchor_index"]]
        if str(frame["token"]) != row["token"] or frame["log_name"] != row["source_log"]:
            raise ValueError("Anchor identity does not match source log")
        result.append({"token": row["token"], "map": frame["map_location"],
                       "xy": list(map(float, frame["ego2global_translation"][:2]))})
    return result


def read_layer(path, layer, columns, projection):
    import pyogrio
    import pyogrio.raw
    import shapely
    from pyproj import Transformer

    meta, fids, wkb, arrays = pyogrio.raw.read(path, layer=layer, columns=columns, return_fids=True)
    transform = Transformer.from_crs(meta["crs"], projection, always_xy=True)
    geometries = shapely.transform(shapely.from_wkb(wkb), transform.transform, interleaved=False)
    return geometries, fids, dict(zip(meta["fields"], arrays))


def counts(tree, points, predicate, distance=None):
    indices = tree.query(points, predicate=predicate, distance=distance)
    return np.bincount(indices[0], minlength=len(points)), indices


def nearest_distance(tree, points):
    if not len(tree.geometries):
        return np.full(len(points), np.nan)
    indices, distances = tree.query_nearest(points, return_distance=True, all_matches=False)
    result = np.full(len(points), np.nan)
    result[indices[0]] = np.minimum(distances, 100)
    return result


def map_facts(path, rows):
    import pyogrio.raw
    import shapely

    _, _, _, fields = pyogrio.raw.read(path, layer="meta", columns=["key", "value"])
    projection = dict(zip(fields[0], fields[1]))["projectedCoordSystem"]
    lanes, lane_fids, lane_data = read_layer(path, "lanes_polygons", ["lane_fid", "lane_group_fid"], projection)
    connectors, _, connector_data = read_layer(path, "lane_connectors", ["entry_lane_fid", "exit_lane_fid"], projection)
    intersections, _, _ = read_layer(path, "intersections", [], projection)
    crossings, _, _ = read_layer(path, "crosswalks", [], projection)
    connector_polygons, _, _ = read_layer(path, "gen_lane_connectors_scaled_width_polygons", [], projection)
    points = shapely.points(np.array([row["xy"] for row in rows]))
    lane_tree, conn_tree = shapely.STRtree(lanes), shapely.STRtree(connectors)
    intersection_tree = shapely.STRtree(intersections)
    inside, _ = counts(intersection_tree, points, "covered_by")
    nearby_lanes, _ = counts(lane_tree, points, "dwithin", 30)
    nearby_connectors, _ = counts(conn_tree, points, "dwithin", 30)
    inside_connector, _ = counts(shapely.STRtree(connector_polygons), points, "covered_by")
    lane_matches, lane_indices = counts(lane_tree, points, "covered_by")
    intersection_dist = nearest_distance(intersection_tree, points)
    crosswalk_dist = nearest_distance(shapely.STRtree(crossings), points)
    # Same edge convention as upstream NuPlanLane: entry is incoming, exit outgoing.
    incoming = Counter(str(int(v)) for v in connector_data["entry_lane_fid"] if v is not None and np.isfinite(float(v)))
    outgoing = Counter(str(int(v)) for v in connector_data["exit_lane_fid"] if v is not None and np.isfinite(float(v)))
    lane_groups = Counter(str(v) for v in lane_data["lane_group_fid"])
    single = {int(i): int(j) for i, j in lane_indices.T if lane_matches[i] == 1}
    result = []
    for i, row in enumerate(rows):
        f = {"inside_intersection": int(inside[i] > 0), "intersection_distance_m": float(intersection_dist[i]),
             "nearby_lanes_30m": int(nearby_lanes[i]), "nearby_connectors_30m": int(nearby_connectors[i]),
             "crosswalk_distance_m": float(crosswalk_dist[i]), "inside_lane_connector": int(inside_connector[i] > 0),
             "lane_incoming_edges": None, "lane_outgoing_edges": None, "lane_group_lanes": None}
        if i in single:
            j = single[i]
            lane_id = str(int(lane_fids[j]))
            f.update(lane_incoming_edges=incoming[lane_id], lane_outgoing_edges=outgoing[lane_id],
                     lane_group_lanes=lane_groups[str(lane_data["lane_group_fid"][j])])
        f = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in f.items()}
        labels = []
        if f["inside_intersection"]:
            labels.append("地图路口内部")
        elif f["intersection_distance_m"] is not None and f["intersection_distance_m"] <= 25:
            labels.append("地图路口邻近25m")
        if f["inside_lane_connector"]:
            labels.append("地图车道连接区域")
        if f["crosswalk_distance_m"] is not None and f["crosswalk_distance_m"] <= 15:
            labels.append("地图横道邻近15m")
        if (f["lane_outgoing_edges"] or 0) > 1:
            labels.append("车道多后继连接")
        if (f["lane_incoming_edges"] or 0) > 1:
            labels.append("车道多前驱连接")
        result.append({"token": row["token"], "features": f, "labels": labels,
                       "lane_association": "unique_polygon" if i in single else "ambiguous" if lane_matches[i] > 1 else "not_on_lane_polygon",
                       "map_name": row["map"], "map_version": path.parent.name,
                       "scope": "current_anchor_map_facts_not_maneuver_or_causal_labels"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "source": "native_NAVSIM_logs_and_nuPlan_map", "image_vectors_used": False, "splits": {}, "map_hashes": {}}
    maps_root = args.maps_root.resolve()
    for split, expected in (("navtrain", 103288), ("navtest", 12146)):
        index_path = args.index_root / split / "index.json"
        index = json.loads(index_path.read_text())
        if index["split"] != split or not index["complete"] or len(index["records"]) != expected:
            raise ValueError("Invalid original scene index")
        groups = defaultdict(list)
        for row in index["records"]:
            groups[row["log_path"]].append(row)
        anchors = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, result in enumerate(executor.map(read_anchors, sorted(groups.items()))):
                anchors.extend(result)
                if i % 20 == 0:
                    save_json(output / "progress.json", {"status": "reading_anchor_logs", "split": split, "logs_done": i + 1, "logs_total": len(groups), "anchors": len(anchors)})
        cities = defaultdict(list)
        for row in anchors:
            cities[row["map"]].append(row)
        result = []
        for city, rows in sorted(cities.items()):
            save_json(output / "progress.json", {"status": "querying_map", "split": split, "city": city, "anchors": len(rows)})
            paths = list((maps_root / city).glob("*/map.gpkg"))
            if len(paths) != 1:
                raise ValueError("Map version ambiguous")
            summary["map_hashes"][city] = digest(paths[0])
            result.extend(map_facts(paths[0], rows))
        target = output / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as stream:
            for row in sorted(result, key=lambda v: v["token"]):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary["splits"][split] = {"members": len(result), "lane_association": dict(Counter(v["lane_association"] for v in result)), "index_sha256": digest(index_path), "facts_sha256": digest(target)}
        save_json(output / "summary.json", summary)
    save_json(output / "progress.json", {"status": "complete"})


if __name__ == "__main__":
    main()
