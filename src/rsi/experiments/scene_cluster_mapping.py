"""Versioned scene pools and bounded expansion into the existing Harness queue."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.data.catalog.centroid_cache import open_cache, write_json
from driversi.data.catalog.scene_clusters import (
    assign, candidate_indices, cluster_summaries, fit, rank_pool, transform,
)
from driversi.data.catalog.store import identity


def _verify_membership(manifest, rows):
    from driversi_agents.chainflow.benchmark import official_filter

    path, spec = official_filter(manifest["split"])
    if {r["token"] for r in rows} != set(spec["tokens"]):
        raise ValueError("Pool differs from complete official split")
    if not {r["source_log"] for r in rows} <= set(spec["log_names"]):
        raise ValueError("Pool source log outside official split")
    return dict(path=str(path), sha256=digest(path))


def validate_priority(report, plan, plan_digest, tokens, test_tokens):
    if (not report.get("complete") or report.get("plan_sha256") != plan_digest
            or report.get("expected") != len(test_tokens)
            or report.get("completed") != len(test_tokens)):
        raise ValueError("Complete parent evaluation with bound plan required")
    rows = report["rows"]
    if (len(rows) != len(test_tokens) or {r["token"] for r in rows} != test_tokens
            or {r["token"] for r in plan["records"]} != test_tokens):
        raise ValueError("Evaluation population differs from the test pool")
    scores = {r["token"]: r["metrics"]["PDMS"] for r in rows}
    if any(type(x) not in (int, float) or not np.isfinite(x) or not 0 <= x <= 1
           for x in scores.values()):
        raise ValueError("Normalized finite PDMS required")
    expected = {t for t, score in scores.items() if score < 0.6}
    if len(tokens) != len(set(tokens)) or set(tokens) != expected:
        raise ValueError("Priority queue must match this parent's complete zero/low set")
    return sum(x == 0 for x in scores.values()), sum(0 < x < 0.6 for x in scores.values())


def _top(scores, indices, rows, limit, per_source=None):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.lexsort((indices, -scores[indices]))
    found, counts = [], {}
    for pos in order:
        i = int(indices[pos])
        group = rows[i]["source_group"]
        if per_source is not None and counts.get(group, 0) >= per_source:
            continue
        counts[group] = counts.get(group, 0) + 1
        found.append(i)
        if len(found) == limit:
            break
    return found


def run(config_path, output):
    import fcntl

    cfg, out = read_mapping(config_path), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(cfg, Path(config_path), out)


def _run(cfg, config_path, out):
    start = time.monotonic()
    tm, train, tv = open_cache(cfg["train_cache"], verify=True)
    qm, test, qv = open_cache(cfg["test_cache"], verify=True)
    if tm["split"] != "navtrain" or qm["split"] != "navtest":
        raise ValueError("Explicit training/test pool roles required")
    if tm["profile"] != qm["profile"] or tm["encoder"] != qm["encoder"]:
        raise ValueError("Pool encoder profiles differ")
    if {r["token"] for r in train} & {r["token"] for r in test}:
        raise ValueError("Split overlap")
    memberships = [_verify_membership(tm, train), _verify_membership(qm, test)]
    base = read_mapping(cfg["baseline_config"])
    baseline = Path(base["output"]) / "baseline"
    request = dict(
        config_sha256=digest(config_path), train_cache=tm["cache_id"],
        test_cache=qm["cache_id"], memberships=memberships,
        baseline_sha256=digest(baseline / "report.json"),
        baseline_config_sha256=digest(cfg["baseline_config"]),
        queries_sha256=digest(cfg["priority_queue"]),
        implementation={str(Path(p).name): digest(p) for p in [
            __file__, Path(__file__).parents[2] / "data/catalog/scene_clusters.py"]},
    )
    if (out / "complete.json").exists():
        seal = read_mapping(out / "complete.json")
        if seal["request"] != request:
            raise ValueError("Frozen cluster inputs changed; use a new run")
        for name, sha in seal["files"].items():
            if digest(out / name) != sha:
                raise ValueError("Frozen cluster output changed")
        return dict(read_mapping(out / "summary.json"), reused=True,
                    reload_seconds=time.monotonic() - start)
    if (out / "inputs.json").exists() and read_mapping(out / "inputs.json") != request:
        raise ValueError("Incomplete run has different inputs")
    write_json(out / "inputs.json", request)
    k = int(cfg.get("clusters", 128))
    probes = int(cfg.get("probes", 8))
    limit = int(cfg.get("expansion_per_query", 64))
    if not 1 <= probes <= k or not 1 <= limit <= 256:
        raise ValueError("Invalid bounded mapping budget")
    write_json(out / "progress.json", dict(phase="fitting_train_contexts"))
    fit_start = time.monotonic()
    model, tx, labels, distances = fit(
        tv, train, split=tm["split"], clusters=k,
        fit_samples=cfg.get("fit_samples", 8192), iterations=cfg.get("iterations", 20),
        seed=cfg.get("seed", 42))
    fit_seconds = time.monotonic() - fit_start
    qx = transform(qv, test, model)
    qlabels, qdistances = assign(qx, model["centers"])
    with (out / "model.npz").open("wb") as f:
        np.savez(f, **model)
    os.chmod(out / "model.npz", 0o600)
    model_id = identity(dict(train_cache=tm["cache_id"], model_sha256=digest(out / "model.npz")))
    write_json(out / "model.json", dict(
        model_id=model_id, fitted_split="navtrain", config=cfg,
        training_mean_and_scale_only=True, test_used_for_fit=False,
        meaning="history_scene_context_not_verified_interaction_or_failure_cause"))
    for name, rows, lab, dist in [
        ("navtrain", train, labels, distances), ("navtest", test, qlabels, qdistances),
    ]:
        write_json(out / name / "pool.json", dict(
            split=name, model_id=model_id, publishable=False,
            rows=[dict(token=r["token"], cluster=int(c), distance=float(d),
                       outlier=bool(d > model["outlier_distance"]),
                       fact_subgroup=identity(sorted(r["conditions"])),
                       history_continuous=r["history_continuous"], trainable=False)
                  for r, c, d in zip(rows, lab, dist, strict=True)]))
    write_json(out / "navtrain/clusters.json", dict(
        model_id=model_id, rows=cluster_summaries(train, labels, distances, k)))
    priorities = read_mapping(cfg["priority_queue"])
    tokens = [r["token"] for r in priorities["rows"]]
    test_index = {r["token"]: i for i, r in enumerate(test)}
    if not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= test_index.keys():
        raise ValueError("Priority queries must be unique members of the test pool")
    if len(tokens) > 1024:
        raise ValueError("Bounded initial priority query budget exceeded")
    zero_count, low_count = validate_priority(
        read_mapping(baseline / "report.json"), read_mapping(baseline / "plan.json"),
        digest(baseline / "plan.json"), tokens, set(test_index))
    indices = [test_index[t] for t in tokens]
    # Independent fixed pilot for algorithmic ANN recall, not semantic recall.
    rng = np.random.default_rng(cfg.get("seed", 42))
    bench_ids = rng.choice(len(test), min(cfg.get("benchmark_queries", 64), len(test)), replace=False)
    all_queries = sorted(set(indices) | set(map(int, bench_ids)))
    col = {q: j for j, q in enumerate(all_queries)}
    write_json(out / "progress.json", dict(phase="exact_reference_batch"))
    exact_start = time.monotonic()
    scores = np.asarray(tv) @ np.asarray(qv[all_queries]).T
    exact_seconds = time.monotonic() - exact_start
    all_train = np.arange(len(train))
    benchmarks = []
    for probe_count in sorted({min(p, k) for p in (4, 8, 16, 32)}):
        recalls, scanned = [], []
        timed = time.monotonic()
        for q in bench_ids:
            pool, _ = candidate_indices(qx[q], model["centers"], labels, probe_count)
            # Recompute real pool cosine for an honest query-only timing.
            approx = [i for i, _ in rank_pool(tv, qv[q], pool, train, 20)]
            exact = _top(scores[:, col[q]], all_train, train, 20)
            recalls.append(len(set(approx) & set(exact)) / 20)
            scanned.append(len(pool))
        benchmarks.append(dict(
            probes=probe_count, centroid_recall_at_20=float(np.mean(recalls)),
            min_query_recall_at_20=float(min(recalls)),
            mean_scanned=float(np.mean(scanned)),
            seconds_including_reference_sort=time.monotonic() - timed))
    # Fixed production recipe; benchmark on test does not tune probe count.
    mapping, production, seen = [], [], set()
    for n, q in enumerate(indices):
        pool, chosen = candidate_indices(qx[q], model["centers"], labels, probes)
        query_scores = scores[:, col[q]]
        expanded = _top(query_scores, pool, train, limit, per_source=3)
        fallback = _top(query_scores, all_train, train, 16, per_source=1)
        union = list(dict.fromkeys(fallback + expanded))
        candidates = []
        for i in union:
            row = train[i]
            candidate = dict(
                token=row["token"], source_log=row["source_log"],
                source_group=row["source_group"], input_id=row["input_id"],
                cluster=int(labels[i]), similarity=float(query_scores[i]),
                evidence_channels=[tag for ok, tag in [
                    (i in fallback, "exact_centroid_fallback"),
                    (i in expanded, "scene_context_cluster_expansion")] if ok],
                state="asset_preflight_pending", trainable=False,
                observed_irregular_history=row["history_continuous"] is False,
            )
            candidates.append(candidate)
        mapping.append(dict(query_token=test[q]["token"], clusters=chosen.tolist(),
                            outlier=bool(qdistances[q] > model["outlier_distance"]),
                            candidates=candidates))
        if n % 50 == 0:
            write_json(out / "progress.json", dict(phase="mapping", completed=n + 1,
                                                    expected=len(indices)))
    # Fair rank rounds across priority queries, matching the existing Harness queue.
    for rank in range(max(len(r["candidates"]) for r in mapping)):
        for row in mapping:
            if rank < len(row["candidates"]):
                candidate = row["candidates"][rank]
                if candidate["token"] not in seen:
                    seen.add(candidate["token"])
                    production.append(dict(candidate, vector_rank=rank + 1))
    write_json(out / "private-mapping.json", dict(
        rows=mapping, model_id=model_id, publishable=False, adaptive_evaluation=True,
        training_samples_from_test=False, claim="scene_context_similarity_not_failure_diagnosis"))
    dest = out / "production"
    write_json(dest / "inputs.json", dict(
        population="official_navtrain", config_sha256=digest(cfg["baseline_config"]),
        baseline_sha256=request["baseline_sha256"], cluster_model_id=model_id,
        mapping_sha256=digest(out / "private-mapping.json")))
    write_json(dest / "training-production-queue.json", dict(
        rows=production, contains_test_tokens=False, retrieval_is_not_failure_diagnosis=True))
    summary = dict(
        status="completed", training_members=len(train), test_members=len(test),
        ready_vectors=len(train), missing_training_vectors=0, mapped_queries=len(indices),
        training_candidates=len(production), zero_count=zero_count,
        nonzero_below_06=low_count,
        clusters=k, probes=probes, fit_sample_count=len(model["fit_indices"]),
        fit_seconds=fit_seconds, exact_batch_seconds=exact_seconds,
        exact_batch_queries=len(all_queries), benchmark_queries=len(bench_ids),
        benchmark=benchmarks, semantic_recall=None,
        semantic_recall_reason="No independently annotated relevant scene sets",
        text_channel="disabled_no_reviewed_caption_dependency",
        context_channel="history_structured_facts_plus_projected_visual_centroid",
        historical_scene_groups_not_full_temporal_taxonomy=True,
        trainable=False, adaptive_evaluation=True, production_recipe_tuned_on_test=False,
        elapsed_seconds=time.monotonic() - start,
    )
    write_json(out / "summary.json", summary)
    write_json(dest / "summary.json", summary)
    files = {str(p.relative_to(out)): digest(p) for p in out.rglob("*")
             if p.is_file() and p.name not in {"run.lock", "progress.json", "complete.json"}}
    write_json(out / "complete.json", dict(request=request, files=files))
    write_json(out / "progress.json", dict(phase="complete"))
    return dict(summary, reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
