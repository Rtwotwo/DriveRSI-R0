"""Training-only synthesis proposals from measured weakness and retrieval evidence."""

import json
import random
from collections import Counter

from driversi.data.catalog.vectors import VectorStore

METRICS = {
    "collision": ("no_at_fault_collisions", 1.0),
    "road": ("drivable_area_compliance", 1.0),
    "direction": ("driving_direction_compliance", 1.0),
    "interaction": ("time_to_collision_within_bound", 1.0),
    "progress": ("ego_progress", .95),
    "comfort": ("comfort", 1.0),
}


def categories(metrics):
    return [name for name, (key, threshold) in METRICS.items()
            if key in metrics and metrics[key] < threshold]


def expansion_plan(catalog, score_run, profile, *, per_category=8, probes=2, neighbors=5):
    if catalog.split != "navtrain":
        raise ValueError("Expansion proposals must use the training catalog only")
    if not 1 <= per_category <= 32 or not 0 <= probes <= 4 or not 1 <= neighbors <= 20:
        raise ValueError("Bounded proposal/retrieval budgets required")
    provenance = catalog.db.execute("SELECT provenance FROM score_runs WHERE id=?", (score_run,)).fetchone()
    if provenance is None:
        raise ValueError("An identified training score run is required")
    provenance = json.loads(provenance[0])
    if provenance.get("split") != "navtrain" or provenance.get("purpose") != "training_pool_mining":
        raise ValueError("Test feedback cannot enter this training miner")
    from driversi_agents.chainflow.benchmark import official_filter
    _, official = official_filter("navtrain")
    members, logs = set(official["tokens"]), set(official["log_names"])
    scored = {}
    for row in catalog.db.execute(
        "SELECT x.token,x.metrics,s.source_log,s.source_group,s.descriptor FROM scores x "
        "JOIN scenes s ON s.token=x.token WHERE x.run=? AND x.valid=1", (score_run,)):
        if row["token"] not in members or row["source_log"] not in logs:
            raise ValueError("Score outside official training membership")
        scored[row["token"]] = {"token": row["token"], "source_log": row["source_log"],
            "source_group": row["source_group"], "metrics": json.loads(row["metrics"]),
            "descriptor": json.loads(row["descriptor"])}
    ready = {r[0] for r in catalog.db.execute(
        "SELECT token FROM vectors WHERE profile=? AND status='ready'", (profile,))}
    store = VectorStore(catalog)
    store.profile_data(profile)
    ready_sources = {x["token"]: x["source_group"] for x in catalog.db.execute(
        "SELECT s.token,s.source_group FROM scenes s JOIN vectors v ON v.token=s.token WHERE v.profile=? AND v.status='ready'", (profile,))}
    selected, validation = {}, []
    distribution = Counter()
    for category, (key, threshold) in METRICS.items():
        weak = [row for row in scored.values() if row["metrics"].get(key, 1) < threshold]
        distribution[category] = len(weak)
        weak.sort(key=lambda row: (row["metrics"][key], row["metrics"]["score"], row["token"]))
        used, kept = set(), 0
        for row in weak:
            if row["source_group"] in used:
                continue
            used.add(row["source_group"])
            item = selected.setdefault(row["token"], {**row, "selection_categories": []})
            item["selection_categories"].append(category)
            kept += 1
            if kept >= per_category:
                break
        queries, used = [], set()
        for row in weak:
            if row["token"] in ready and row["source_group"] not in used:
                queries.append(row)
                used.add(row["source_group"])
            if len(queries) >= probes:
                break
        if probes == 0:
            queries = []
        for query in queries:
            result = store.search("training", query["token"], profile, limit=neighbors, shortlist=128)
            comparisons = []
            for x in result["results"]:
                measured = scored.get(x["token"])
                comparisons.append({"token": x["token"], "cosine": x["aligned_view_cosine"],
                    "measured": measured is not None,
                    "same_weakness_metric": measured["metrics"].get(key, 1) < threshold if measured else None,
                    "metrics": measured["metrics"] if measured else None})
            # A measured proxy, not a claim that nearest visual neighbors share a cause.
            eligible = [{"token": t, "source_group": g, "metrics": scored[t]["metrics"] if t in scored else None} for t,g in ready_sources.items() if g != query["source_group"]]
            rng = random.Random("scene-expansion-v1:" + query["token"] + category)
            rng.shuffle(eligible)
            random_groups, baseline = set(), []
            for row in eligible:
                if row["source_group"] in random_groups:
                    continue
                random_groups.add(row["source_group"])
                baseline.append(row["metrics"].get(key, 1) < threshold if row["metrics"] is not None else None)
                if len(baseline) >= neighbors:
                    break
            validation.append({"query_token": query["token"], "category": category,
                "query_metrics": query["metrics"], "neighbors": comparisons,
                "unfiltered_vector_same_metric_hits": sum(x["same_weakness_metric"] is True for x in comparisons),
                "measured_neighbors": sum(x["measured"] for x in comparisons),
                "random_distinct_source_hits": sum(x is True for x in baseline), "random_count": len(baseline),
                "random_measured_neighbors": sum(x is not None for x in baseline),
                "scanned": result["scanned_vectors"], "same_cause_established": False})
    candidates = []
    for row in selected.values():
        continuous = row["descriptor"].get("history", {}).get("continuous") is True
        candidates.append({**row, "has_current_feature": row["token"] in ready,
            "status": "needs_same_anchor_diagnosis" if continuous else "source_history_repair_required",
            "synthesis_requested": continuous, "teacher_pdms_minimum": 95,
            "required_output": "4 new historical frames and 10 physical future points with matching images",
            "admission_required": ["same_anchor_teacher_execution", "counterfactual_pdms_ge_95", "physical_integrity", "visual_alignment"],
            "trainable": False})
    return {"kind": "navtrain_synthesis_proposals", "split": "navtrain", "score_run": score_run,
        "score_provenance": provenance, "vector_profile": profile, "scored_scenes": len(scored),
        "ready_vectors": len(ready), "weakness_counts": dict(distribution), "candidates": candidates,
        "retrieval_validation": validation, "test_samples_consumed": 0, "test_details_exported": False,
        "claim": "same_metric_weakness_proxy_not_causal_retrieval_accuracy", "trainable": False}
