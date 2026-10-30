"""Exact geographic recall over full training membership, not semantic relevance."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from driversi.data.catalog.scene_hierarchy import digest, save_json


def read(path):
    return json.loads(Path(path).read_text())


def metrics(retrieved, truth):
    """Macro over queries; unknown/empty geographic opportunities are explicit."""
    if set(retrieved) != set(truth) or not truth:
        raise ValueError("Identical nonempty query sets required")
    eligible = [q for q in truth if truth[q]]
    rows = []
    for q in sorted(truth):
        selected, relevant = set(retrieved[q]), set(truth[q])
        matched = selected & relevant
        rows.append({"query": q, "retrieved": len(selected), "relevant": len(relevant),
                     "matched": len(matched), "recall": len(matched) / len(relevant) if relevant else None,
                     "precision": len(matched) / len(selected) if selected else 0.0})
    n = len(rows)
    total_relevant = sum(r["relevant"] for r in rows)
    total_selected = sum(r["retrieved"] for r in rows)
    total_matched = sum(r["matched"] for r in rows)
    recalled = [r["recall"] for r in rows if r["recall"] is not None]
    return {
        "queries": n, "queries_with_geographic_ground_truth": len(eligible),
        "queries_without_ground_truth": n - len(eligible),
        "queries_with_any_match": sum(r["matched"] > 0 for r in rows),
        "hit_fraction_all_queries": sum(r["matched"] > 0 for r in rows) / n,
        "hit_fraction_eligible_queries": sum(r["matched"] > 0 for r in rows) / len(eligible) if eligible else None,
        "mean_candidates_per_query": total_selected / n,
        "empty_retrieval_queries": sum(r["retrieved"] == 0 for r in rows),
        "total_query_candidate_pairs": total_selected,
        "total_relevant_pairs": total_relevant, "matched_pairs": total_matched,
        "macro_recall": float(np.mean(recalled)) if recalled else None,
        "micro_recall": total_matched / total_relevant if total_relevant else None,
        "macro_precision_empty_is_zero": sum(r["precision"] for r in rows) / n,
        "micro_precision": total_matched / total_selected if total_selected else None,
    }, rows


def geographic_truth(queries, poses, radius, max_heading):
    trees = {}
    for city in sorted({poses[q]["map"] for q in queries}):
        train = sorted((r for r in poses.values() if r["split"] == "navtrain" and r["map"] == city), key=lambda r: r["token"])
        trees[city] = cKDTree([r["xy"] for r in train]), train
    truth = {}
    for q in queries:
        pose = poses[q]
        if pose["split"] != "navtest":
            raise ValueError("Expected test query")
        tree, train = trees[pose["map"]]
        selected = tree.query_ball_point(pose["xy"], radius)
        relevant = set()
        for i in selected:
            target = train[i]
            heading = abs(math.degrees((pose["yaw"] - target["yaw"] + math.pi) % (2 * math.pi) - math.pi))
            if max_heading is None or heading <= max_heading:
                relevant.add(target["token"])
        truth[q] = relevant
    return truth


def run(result):
    start = time.perf_counter()
    result = Path(result).resolve()
    geography = read(result / "summary.json")
    if geography["status"] != "complete" or geography["pose_coverage"] != {"navtrain": 103288, "navtest": 12146}:
        raise ValueError("Complete geographic pose inventory required")
    poses = {r["token"]: r for r in read(result / "private/poses.json")["rows"]}
    pairs = read(result / "private/pairs.json")["rows"]
    queries = sorted({r["query"] for r in pairs})
    if len(queries) != 527:
        raise ValueError("Expected same 527 queries")
    methods = {m: {q: set() for q in queries} for m in ("visual", "cluster", "hybrid")}
    for row in pairs:
        if row["rank"] > 16 or poses[row["candidate"]]["split"] != "navtrain":
            raise ValueError("Only saved Top16 training candidates allowed")
        methods[row["method"]][row["query"]].add(row["candidate"])
    methods["intersection"] = {q: methods["visual"][q] & methods["cluster"][q] for q in queries}
    methods["union"] = {q: methods["visual"][q] | methods["cluster"][q] for q in queries}
    schemes = [(25, None), (50, None), (100, None), (500, None), (50, 30), (100, 30)]
    reports = {}
    private = result / "private/geographic-recall"
    private.mkdir(exist_ok=True, mode=0o700)
    for radius, heading in schemes:
        name = f"radius_{radius}m" + (f"_heading_{heading}deg" if heading else "")
        truth = geographic_truth(queries, poses, radius, heading)
        values = [len(v) for v in truth.values()]
        report = {"radius_m": radius, "max_heading_difference_deg": heading,
                  "relevant_candidates": {"min": min(values), "median": float(np.median(values)), "max": max(values), "sum": sum(values)},
                  "recall_at16_cardinality_upper_bound": float(np.mean([min(16, n) / n for n in values if n])) if any(values) else None,
                  "methods": {}}
        rows = {}
        for method, selected in methods.items():
            report["methods"][method], rows[method] = metrics(selected, truth)
        for q in queries:
            common = methods["intersection"][q] & truth[q]
            union = methods["union"][q] & truth[q]
            assert len(common) <= min(len(methods[m][q] & truth[q]) for m in ("visual", "cluster"))
            assert len(union) >= max(len(methods[m][q] & truth[q]) for m in ("visual", "cluster"))
        save_json(private / (name + ".json"), {"publishable": False, "truth": {q: sorted(t) for q, t in truth.items()}, "per_query_metrics": rows})
        os.chmod(private / (name + ".json"), 0o600)
        reports[name] = report
    summary = {"status": "complete", "queries": len(queries), "train_members": 103288,
               "definition": "all_training_anchors_in_same_map_within_declared_radius_and_optional_heading",
               "truth_scope": "geographic_relevance_only_not_same_lane_interaction_failure_cause",
               "primary_protocol": "radius_100m_heading_30deg",
               "protocols": reports,
               "set_scope": "per_query_intersection_and_union_of_visual_top16_and_cluster_top16",
               "union_budget": "up_to32; not equal to a ranked Top16 fusion",
               "empty_truth_policy": "excluded_from_macro_recall_and_reported; hit_all_keeps_all527",
               "precision_policy": "macro_empty_retrieval_is_zero; micro_precision_counts_only_returned_pairs",
               "source_files": {str(result / "private" / f): digest(result / "private" / f) for f in ("poses.json", "pairs.json")},
               "module_sha256": digest(__file__),
               "api_calls": 0, "student_scoring_calls": 0, "elapsed_seconds": time.perf_counter() - start}
    save_json(result / "geographic-recall.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.result)
    print(json.dumps({k: v for k, v in result.items() if k != "protocols"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
