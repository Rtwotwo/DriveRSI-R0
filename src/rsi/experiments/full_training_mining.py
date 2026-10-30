"""Private full-test failure mapping into the complete official training corpus.

Vector + historical structural text recall. Missing reviewed VLM captions are
reported, never invented. Assets affect production readiness, not retrieval rank.
"""

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.data.catalog.retrieval_collection import active_collection_identity
from driversi.data.catalog.store import Catalog, identity
from driversi.rsi.experiments.continuation import frozen_json
from driversi.rsi.experiments.navval_batch_retrieval import retrieve
from driversi.rsi.experiments.navval_retrieval import training_members
from driversi.rsi.experiments.reactive_selection import ordered_failures
from driversi.runtime.budget import write_json


def describe(descriptor):
    """Only history facts; omit token, path, timestamp and future supervision."""
    if descriptor.get("history", {}).get("information_set") != "current_state_history":
        raise ValueError("Historical-only descriptor required")
    terms = list(descriptor.get("conditions", []))
    for name, value in descriptor.get("features", {}).items():
        if name.endswith("count_30m") or name.endswith("count_15m"):
            if value is None:
                terms.append(name + "_unknown")
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid history count")
            terms.append(
                name + "_" + ("none" if value == 0 else "sparse" if value <= 2 else "dense")
            )
    return " ".join(sorted(terms))


def lexical_index(catalog, path, members):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-262144")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute(
        "CREATE TABLE IF NOT EXISTS documents(token TEXT PRIMARY KEY, source_log TEXT, source_group TEXT, text TEXT, cluster TEXT, caption_attempt INTEGER)"
    )
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(token UNINDEXED, text)")
    labels = {}
    for row in catalog.db.execute(
        "SELECT a.token,a.id,a.observation FROM attempts a WHERE a.status='validated' AND a.id=(SELECT max(b.id) FROM attempts b WHERE b.token=a.token AND b.profile=a.profile) AND (SELECT verdict FROM reviews r WHERE r.attempt=a.id ORDER BY r.id DESC LIMIT 1)='spot_check_pass'"
    ):
        value = json.loads(row["observation"] or "{}")
        if value.get("caption"):
            labels[row["token"]] = (row["id"], value["caption"])
    with db:
        for row in catalog.db.execute(
            "SELECT token,source_log,source_group,descriptor FROM scenes"
        ):
            if row["token"] not in members:
                continue
            desc = json.loads(row["descriptor"])
            text = describe(desc)
            label = labels.get(row["token"])
            if label:
                text += " " + label[1]
            cluster = identity(sorted(desc.get("conditions", [])))
            prior = db.execute(
                "SELECT text,caption_attempt FROM documents WHERE token=?", (row["token"],)
            ).fetchone()
            if prior:
                if prior != (text, label[0] if label else None):
                    raise ValueError("Frozen text index input changed")
                continue
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (
                    row["token"],
                    row["source_log"],
                    row["source_group"],
                    text,
                    cluster,
                    label[0] if label else None,
                ),
            )
            db.execute("INSERT INTO search VALUES (?,?)", (row["token"], text))
    if db.execute("SELECT count(*) FROM documents").fetchone()[0] != len(members):
        raise ValueError("Text index coverage mismatch")
    return db, labels


def text_search(db, descriptor, limit=64):
    words = sorted(set(re.findall(r"[a-zA-Z0-9_]+", describe(descriptor))))
    if not words:
        return []
    query = " OR ".join('"' + x + '"' for x in words)
    return [
        dict(token=t, text_bm25=float(score))
        for t, score in db.execute(
            "SELECT token,bm25(search) FROM search WHERE search MATCH ? ORDER BY bm25(search),token LIMIT ?",
            (query, limit),
        )
    ]


def cache_collection_matches(cached, target_identity, query_identity, *, existing_mapping):
    """Never overwrite a completed mapping referenced by frozen production inputs."""
    matches = (
        cached.get("target_collection_id") == target_identity
        and cached.get("query_collection_id") == query_identity
    )
    if not matches and existing_mapping:
        raise ValueError(
            "Retrieval collection changed; use a new output directory; existing mapping preserved"
        )
    return matches


def run(config_path, output, coverage_path, max_queries=64, *, use_text=True, vector_seed=None):
    if not 2 <= max_queries <= 1024:
        raise ValueError("Bounded first-wave query budget required")
    cfg = read_mapping(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    full = Path(cfg["output"]) / "baseline"
    plan, report = read_mapping(full / "plan.json"), read_mapping(full / "report.json")
    if len(plan["records"]) != 12146 or report["plan_sha256"] != digest(full / "plan.json"):
        raise ValueError("Full official parent benchmark required")
    failures = ordered_failures(report, [r["token"] for r in plan["records"]])
    zeros = [r for r in failures if r["metrics"]["PDMS"] == 0]
    lows = [r for r in failures if 0 < r["metrics"]["PDMS"] < 0.6]
    queue = [
        row
        for i in range(max(len(zeros), len(lows)))
        for group in (zeros, lows)
        if i < len(group)
        for row in [group[i]]
    ]
    frozen_json(
        out / "inputs.json",
        dict(
            config=str(Path(config_path).resolve()),
            config_sha256=digest(config_path),
            baseline_sha256=digest(full / "report.json"),
            coverage=str(Path(coverage_path).resolve()),
            coverage_sha256=digest(coverage_path),
            population="official_navtrain",
            max_queries=max_queries,
            threshold=0.6,
            use_text=use_text,
            vector_seed=str(Path(vector_seed).resolve()) if vector_seed else None,
            vector_seed_sha256=digest(vector_seed) if vector_seed else None,
        ),
    )
    write_json(
        out / "private-failure-queue.json",
        dict(rows=queue, publishable=False, zero_count=len(zeros), nonzero_below_06=len(lows)),
    )
    q, t = (
        Catalog(cfg["test_catalog"], "navtest", readonly=True),
        Catalog(cfg["train_catalog"], "navtrain", readonly=True),
    )
    text_db = None
    try:
        members, membership_files = training_members(t, "navtrain")
        labels = {}
        if use_text:
            text_db, labels = lexical_index(t, out / "training-text.sqlite", members)
        write_json(
            out / "progress.json",
            dict(
                status="full_training_vector_scan",
                training_members=len(members),
                zero_count=len(zeros),
                nonzero_below_06=len(lows),
                reviewed_caption_members=len(labels),
            ),
        )
        selected, missing = [], []
        for row in queue:
            v = q.db.execute(
                "SELECT status FROM vectors WHERE profile=? AND token=?",
                (cfg["profile"], row["token"]),
            ).fetchone()
            if v and v[0] == "ready":
                selected.append(row)
            else:
                missing.append(row["token"])
            if len(selected) >= max_queries:
                break
        vp = out / "private-vector-retrieval.json"
        tokens = [r["token"] for r in selected]
        cached = (
            read_mapping(vp) if vp.exists() else read_mapping(vector_seed) if vector_seed else None
        )
        # A supplemental collection changes the candidate universe even if all
        # old vectors are byte-identical. Do not mix old top-k with new rankings.
        if cached and not cache_collection_matches(
            cached,
            active_collection_identity(t, cfg["profile"]),
            active_collection_identity(q, cfg["profile"]),
            existing_mapping=vp.exists(),
        ):
            write_json(
                out / "cache-reuse.json",
                dict(
                    reused_queries=0,
                    reason="collection_changed_recompute_ranking_without_feature_extraction",
                    old_results_preserved=True,
                ),
            )
            cached = None
        cached_rows = {}
        if cached:
            if (
                cached.get("target_population") != "navtrain"
                or cached.get("profile") != cfg["profile"]
                or cached.get("target_members") != len(members)
            ):
                raise ValueError("Cached vector population/profile mismatch")
            from driversi.data.catalog.vectors import VectorStore

            store = VectorStore(q)
            for row in cached["rows"]:
                if row["query_token"] not in tokens:
                    continue
                if (
                    store.get(row["query_token"], cfg["profile"])["input_id"]
                    != row["query_input_id"]
                ):
                    raise ValueError("Cached query pixels changed")
                cached_rows[row["query_token"]] = row
        remaining = [t for t in tokens if t not in cached_rows]
        if remaining:
            vector = retrieve(
                q,
                t,
                cfg["profile"],
                remaining,
                limit=16,
                shortlist=64,
                progress=out / "vector-progress.json",
                population="navtrain",
                **(
                    {"centroid_cache_path": cfg["centroid_cache"]}
                    if cfg.get("centroid_cache")
                    else {}
                ),
            )
            if cached and vector["ready_vectors"] != cached["ready_vectors"]:
                raise ValueError("Training vector population changed during mapping")
            cached_rows.update({row["query_token"]: row for row in vector["rows"]})
        else:
            vector = cached
        if vector is None:
            raise ValueError("No ready priority query vectors")
        vector = dict(
            vector,
            rows=[cached_rows[t] for t in tokens],
            reused_query_results=len(tokens) - len(remaining),
        )
        write_json(vp, vector)
        if not use_text:
            return vector_only_snapshot(config_path, out, out / "production")
        from driversi.data.catalog.hybrid import fuse

        coverage = read_mapping(coverage_path)
        assets = {}
        for a in coverage["rows"]:
            assets.setdefault(a["token"], []).append(a)
        write_json(
            out / "progress.json",
            dict(
                status="text_rank_fusion",
                training_members=len(members),
                ready_vectors=vector["ready_vectors"],
                mapped_queries=0,
            ),
        )
        mappings = []
        production = []
        used = set()
        for visual in vector["rows"]:
            desc = json.loads(
                q.db.execute(
                    "SELECT descriptor FROM scenes WHERE token=?", (visual["query_token"],)
                ).fetchone()[0]
            )
            semantic = text_search(text_db, desc)
            fused = fuse([visual["candidates"], semantic])
            retained = {x["token"] for x in visual["candidates"][:8] + semantic[:8]}
            union = [x for x in fused if x["token"] in retained]
            for x in union:
                x.update(
                    source_log=members[x["token"]],
                    asset_available=x["token"] in assets,
                    trainable=False,
                )
                if x["token"] not in used:
                    used.add(x["token"])
                    production.append(
                        {
                            **x,
                            "assets": assets.get(x["token"], []),
                            "state": "ready_for_rollout_preflight"
                            if x["token"] in assets
                            else "needs_reconstruction_asset",
                        }
                    )
            mappings.append(
                dict(
                    query_token=visual["query_token"],
                    visual=visual["candidates"],
                    text=semantic[:16],
                    union=union,
                )
            )
        write_json(
            out / "private-hybrid-mapping.json",
            dict(
                rows=mappings,
                publishable=False,
                adaptive_evaluation=True,
                training_pixels_from_test=False,
            ),
        )
        write_json(
            out / "training-production-queue.json",
            dict(
                rows=production, contains_test_tokens=False, retrieval_is_not_failure_diagnosis=True
            ),
        )
        summary = dict(
            status="completed",
            training_members=len(members),
            ready_vectors=vector["ready_vectors"],
            missing_training_vectors=len(members) - vector["ready_vectors"],
            vector_coverage_fraction=vector["ready_vectors"] / len(members),
            zero_count=len(zeros),
            nonzero_below_06=len(lows),
            mapped_queries=len(mappings),
            pending_failure_queries=len(queue) - len(mappings),
            missing_query_vectors=len(missing),
            training_candidates=len(production),
            asset_ready_candidates=sum(x["asset_available"] for x in production),
            reviewed_vlm_captions=len(labels),
            text_channel="historical_structured_text_plus_reviewed_captions_if_present",
            recall_at_k=None,
            recall_reason="No independently annotated cross-split relevant-set ground truth",
            policy="vector_top8_preserved_plus_text_top8_RRF",
            trainable=False,
        )
        write_json(out / "summary.json", summary)
        write_json(out / "progress.json", summary)
        return summary
    finally:
        if text_db:
            text_db.close()
        q.close()
        t.close()


def vector_only_snapshot(config_path, completed_mining, output):
    """Use saved full-population vector matches without waiting for captions."""
    source, out = Path(completed_mining), Path(output)
    cfg = read_mapping(config_path)
    origin = read_mapping(source / "inputs.json")
    full = Path(cfg["output"]) / "baseline"
    if origin["config_sha256"] != digest(config_path) or origin["baseline_sha256"] != digest(
        full / "report.json"
    ):
        raise ValueError("Cached vector mapping belongs to a different parent")
    vector = read_mapping(source / "private-vector-retrieval.json")
    if vector.get("target_population") != "navtrain" or vector.get("target_members") != 103288:
        raise ValueError("Full official training search scope required")
    out.mkdir(parents=True, exist_ok=True)
    frozen_json(
        out / "inputs.json",
        dict(
            origin,
            vector_mapping_sha256=digest(source / "private-vector-retrieval.json"),
            vector_source=str((source / "private-vector-retrieval.json").resolve()),
            text_disabled=True,
        ),
    )
    failure = read_mapping(source / "private-failure-queue.json")
    expected = {r["token"] for r in failure["rows"]}
    queried = [q["query_token"] for q in vector["rows"]]
    if len(queried) != len(set(queried)) or not set(queried) <= expected:
        raise ValueError("Vector queries do not belong to the recorded zero/low queue")
    rows, seen = [], set()
    for rank in range(max((len(q["candidates"]) for q in vector["rows"]), default=0)):
        for q in vector["rows"]:
            if rank >= len(q["candidates"]):
                continue
            row = q["candidates"][rank]
            if row["token"] in seen:
                continue
            seen.add(row["token"])
            rows.append(
                dict(row, vector_rank=rank + 1, state="asset_preflight_pending", trainable=False)
            )
    frozen_json(
        out / "training-production-queue.json",
        dict(rows=rows, contains_test_tokens=False, retrieval_is_not_failure_diagnosis=True),
    )
    summary = dict(
        status="completed",
        training_members=vector["target_members"],
        ready_vectors=vector["ready_vectors"],
        missing_training_vectors=vector["target_members"] - vector["ready_vectors"],
        vector_coverage_fraction=vector["ready_vectors"] / vector["target_members"],
        zero_count=failure["zero_count"],
        nonzero_below_06=failure["nonzero_below_06"],
        mapped_queries=len(queried),
        pending_failure_queries=len(expected) - len(queried),
        training_candidates=len(rows),
        text_channel="disabled_no_reviewed_caption_dependency",
        recall_at_k=None,
        recall_reason="No relevance ground truth",
        trainable=False,
    )
    frozen_json(out / "summary.json", summary)
    write_json(out / "progress.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--coverage", required=True, type=Path)
    p.add_argument("--max-queries", type=int, default=64)
    p.add_argument("--vector-only", action="store_true")
    p.add_argument("--vector-seed", type=Path)
    a = p.parse_args()
    print(
        run(
            a.config,
            a.output,
            a.coverage,
            a.max_queries,
            use_text=not a.vector_only,
            vector_seed=a.vector_seed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
