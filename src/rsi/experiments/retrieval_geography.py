"""Audit saved retrievals using native map poses; never a semantic relevance score."""

import argparse
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from driversi.data.catalog.scene_hierarchy import digest, save_json
from driversi.data.sources.navsim import load_log, yaw_from_quaternion

METHODS = ("visual", "hybrid", "cluster")
THRESHOLDS = (25, 50, 100, 500, 1000)


def read_json(path):
    return json.loads(Path(path).read_text())


def pose_distance(a, b):
    """Only compare anchors in the exact same native map coordinate frame."""
    if a["map"] != b["map"]:
        return {"same_map": False, "distance_m": None, "heading_difference_deg": None}
    av, bv = np.array(a["xy"], dtype=float), np.array(b["xy"], dtype=float)
    if av.shape != (2,) or bv.shape != (2,) or not np.isfinite([*av, *bv, a["yaw"], b["yaw"]]).all():
        raise ValueError("Invalid map pose")
    delta = (a["yaw"] - b["yaw"] + math.pi) % (2 * math.pi) - math.pi
    return {"same_map": True, "distance_m": float(np.linalg.norm(av - bv)),
            "heading_difference_deg": abs(math.degrees(delta))}


def summarize(pairs, query_count):
    groups = defaultdict(list)
    for row in pairs:
        groups[row["query"]].append(row)
    if len(groups) != query_count:
        raise ValueError("Missing query in metric denominator")
    distances = [r["distance_m"] for r in pairs if r["same_map"]]
    nearest = [min((r["distance_m"] for r in rows if r["same_map"]), default=None)
               for rows in groups.values()]
    valid_nearest = [d for d in nearest if d is not None]
    return {"queries": query_count, "pairs": len(pairs), "same_map_pairs": len(distances),
            "same_map_pair_fraction": len(distances) / len(pairs),
            "same_map_query_count": len(valid_nearest),
            "same_map_distance_median_m": float(np.median(distances)) if distances else None,
            "same_map_nearest_median_m": float(np.median(valid_nearest)) if valid_nearest else None,
            "distance_quantiles_m": dict(zip(("p10", "p50", "p90"), map(float, np.quantile(distances, [.1, .5, .9])))) if distances else {},
            "within_radius": {str(radius): {
                "pairs": sum(r["same_map"] and r["distance_m"] <= radius for r in pairs),
                "queries": sum(any(r["same_map"] and r["distance_m"] <= radius for r in rows) for rows in groups.values()),
                "queries_heading_30deg": sum(any(r["same_map"] and r["distance_m"] <= radius and r["heading_difference_deg"] <= 30 for r in rows) for rows in groups.values()),
            } for radius in THRESHOLDS}}


def extract_log(item):
    path, records = item
    frames = load_log(Path(path), trusted=True)
    frame_indices = {str(frame["token"]): i for i, frame in enumerate(frames)}
    if len(frame_indices) != len(frames):
        raise ValueError("Duplicate source frame token")
    poses = []
    for record in records:
        anchor_index = frame_indices[record["token"]]
        frame = frames[anchor_index]
        if frame["token"] != record["token"] or frame["log_name"] != record["source_log"]:
            raise ValueError("Indexed anchor identity mismatch")
        xy = list(map(float, frame["ego2global_translation"][:2]))
        yaw = yaw_from_quaternion(frame["ego2global_rotation"])
        if not np.isfinite([*xy, yaw]).all():
            raise ValueError("Invalid anchor pose")
        images = {}
        for camera in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            source = str(Path(record["sensor_root"]) / frame["cams"][camera]["data_path"])
            images[camera] = source
        poses.append({"token": record["token"], "split": record["kind"].removeprefix("original_"),
                      "map": frame["map_location"], "xy": xy, "yaw": yaw,
                      "source_log": record["source_log"], "timestamp_us": int(frame["timestamp"]),
                      "anchor_index": anchor_index, "images": images})
    return poses, {"path": path, "sha256": digest(path)}


def coordinates(project, out, workers):
    groups, token_sets, indices = defaultdict(list), {}, {}
    for split, count in (("navtrain", 103288), ("navtest", 12146)):
        path = project / "lib/cluster_caption/clusters" / split / "members.jsonl"
        members = [json.loads(line) for line in path.open()]
        dataset = project.parent / "dataset"
        log_root = dataset / "navsim_logs" / ("trainval" if split == "navtrain" else "test")
        sensor_root = dataset / "sensor_blobs" / ("trainval_all" if split == "navtrain" else "test")
        records = [{"token": m["token"], "source_log": m["source_log"], "kind": "original_" + split,
                    "log_path": str(log_root / (m["source_log"] + ".pkl")), "sensor_root": str(sensor_root)} for m in members]
        if len(records) != count or len({r["token"] for r in records}) != count or any(m["split"] != split for m in members):
            raise ValueError("Original caption/cluster membership differs")
        token_sets[split] = {m["token"] for m in members}
        indices[split] = records
        for record in records:
            groups[record["log_path"]].append(record)
    if token_sets["navtrain"] & token_sets["navtest"]:
        raise ValueError("Train/test token overlap")
    fingerprint = __import__("hashlib").sha256(json.dumps(indices, sort_keys=True).encode()).hexdigest()
    saved = out / "private/poses.json"
    if saved.exists():
        cache = read_json(saved)
        if cache["index_sha256"] != fingerprint or cache["module_sha256"] != digest(__file__):
            raise ValueError("Pose extraction contract changed")
        if any(digest(row["path"]) != row["sha256"] for row in cache["logs"]):
            raise ValueError("Native log changed")
        return cache["rows"], cache["logs"]
    rows, logs = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (values, provenance) in enumerate(pool.map(extract_log, sorted(groups.items()))):
            rows.extend(values)
            logs.append(provenance)
            if i % 20 == 0:
                save_json(out / "progress.json", {"status": "native_pose_extraction", "logs": i + 1, "total_logs": len(groups), "anchors": len(rows)})
    save_json(saved, {"index_sha256": fingerprint, "module_sha256": digest(__file__), "rows": rows, "logs": logs})
    return rows, logs


def map_metadata(project, names):
    from pyproj import CRS

    result = {}
    for name in sorted(names):
        paths = list((project.parent / "dataset/maps" / name).glob("*/map.gpkg"))
        if len(paths) != 1:
            raise ValueError("Map version must be unambiguous")
        path = paths[0]
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            metadata = dict(db.execute('SELECT key,value FROM meta'))
        crs = CRS.from_user_input(metadata["projectedCoordSystem"])
        if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]):
            raise ValueError("Native map coordinates must be projected metres")
        result[name] = {"version": path.parent.name, "projection": crs.to_string(), "unit": "metre", "sha256": digest(path)}
    return result


def run(config):
    start = time.perf_counter()
    project, out = Path(config["project"]), Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(exist_ok=True, mode=0o700)
    os.chmod(out / "private", 0o700)
    current = project / "exp/chainflow9405-cluster-mapping-v1"
    old = project / "exp/chainflow-restart-32000-v1/mapping"
    files = [current / "private/query-mapping.json", old / "private-vector-retrieval.json", old / "private-hybrid-mapping.json"]
    qrows = read_json(files[0])["rows"]
    methods = {
        "cluster": {r["token"]: r["training_candidates"][:16] for r in qrows},
        "visual": {r["query_token"]: [c["token"] for c in r["candidates"][:16]] for r in read_json(files[1])["rows"]},
        "hybrid": {r["query_token"]: [c["token"] if isinstance(c, dict) else c for c in r["union"][:16]] for r in read_json(files[2])["rows"]},
    }
    queries = sorted(methods["cluster"])
    if len(queries) != 527 or any(set(methods[m]) != set(queries) for m in METHODS):
        raise ValueError("Identical 527 query population required")
    poses, logs = coordinates(project, out, config.get("workers", 6))
    by_token = {r["token"]: r for r in poses}
    if len(by_token) != len(poses):
        raise ValueError("Duplicated pose identity")
    maps = map_metadata(project, {r["map"] for r in poses})
    pairs, result = [], {}
    for method in METHODS:
        for q in queries:
            if by_token[q]["split"] != "navtest" or not methods[method][q] or len(methods[method][q]) != len(set(methods[method][q])):
                raise ValueError("Invalid query/candidates")
            for rank, t in enumerate(methods[method][q], 1):
                if by_token[t]["split"] != "navtrain":
                    raise ValueError("Candidate is not a training member")
                pairs.append({"method": method, "query": q, "candidate": t, "rank": rank,
                              **pose_distance(by_token[q], by_token[t])})
        subset = [p for p in pairs if p["method"] == method]
        result[method] = {"top1": summarize([p for p in subset if p["rank"] == 1], len(queries)),
                          "top16": summarize(subset, len(queries)),
                          "by_map": {name: summarize([p for p in subset if by_token[p["query"]]["map"] == name], sum(by_token[q]["map"] == name for q in queries)) for name in sorted({by_token[q]["map"] for q in queries})}}
    # A geographic opportunity ceiling, not a visual/semantic retrieval baseline.
    trees = {}
    for name in maps:
        members = sorted((r for r in poses if r["split"] == "navtrain" and r["map"] == name), key=lambda r: r["token"])
        trees[name] = cKDTree([r["xy"] for r in members]), members
    nearest = []
    for q in queries:
        tree, members = trees[by_token[q]["map"]]
        distance, i = tree.query(by_token[q]["xy"])
        nearest.append({"query": q, "candidate": members[int(i)]["token"], "distance_m": float(distance),
                        "heading_difference_deg": pose_distance(by_token[q], members[int(i)])["heading_difference_deg"]})
    summary = {"status": "complete", "queries": len(queries), "pose_coverage": dict(Counter(r["split"] for r in poses)),
               "query_maps": dict(Counter(by_token[q]["map"] for q in queries)), "maps": maps,
               "methods": result, "geographic_availability": {str(radius): sum(r["distance_m"] <= radius for r in nearest) for radius in THRESHOLDS},
               "metrics_meaning": "same_map_anchor_proximity_not_same_road_or_semantic_relevance",
               "cross_map_distances": "null; never compared", "coordinate_scope": "native ego2global XY; projected metres; anchor only",
               "same_budget": "saved ranking capped16; hybrid has16 queries with15 candidates; source diversity policies were not rerun",
               "files": {str(p): digest(p) for p in files}, "native_logs": len(logs),
               "student_scoring_calls": 0, "api_calls": 0, "elapsed_seconds": time.perf_counter() - start}
    save_json(out / "private/pairs.json", {"publishable": False, "rows": pairs})
    save_json(out / "private/geographic-nearest.json", {"publishable": False, "rows": nearest})
    save_json(out / "summary.json", summary)
    save_json(out / "progress.json", {"status": "complete", "queries": len(queries)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(read_json(args.config)), ensure_ascii=False))


if __name__ == "__main__":
    main()
