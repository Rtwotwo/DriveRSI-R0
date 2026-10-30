"""One read of navval centroids for many private test queries, same similarity."""

import time

import numpy as np

from driversi.data.catalog.retrieval_collection import active_collection_identity
from driversi.data.catalog.vectors import VectorStore, normalize
from driversi.rsi.experiments.navval_retrieval import training_members
from driversi.runtime.budget import write_json


def retrieve(
    query_catalog,
    train_catalog,
    profile,
    tokens,
    *,
    limit=4,
    shortlist=64,
    progress=None,
    population="navval",
    centroid_cache_path=None,
):
    if query_catalog.split != "navtest" or train_catalog.split != "navtrain":
        raise ValueError("Explicit test query / training target catalogs required")
    if not tokens or len(tokens) != len(set(tokens)) or not 1 <= limit <= shortlist <= 256:
        raise ValueError("Unique bounded queries and candidates required")
    query, target = VectorStore(query_catalog), VectorStore(train_catalog)
    if query.profile_data(profile) != target.profile_data(profile):
        raise ValueError("Query and target embedding profiles differ")
    members, files = training_members(train_catalog, population)
    start = time.monotonic()
    if centroid_cache_path is not None:
        from pathlib import Path

        from driversi.configuration import read_mapping
        from driversi.data.catalog.centroid_cache import open_cache, signature

        manifest, cached_rows, matrix = open_cache(centroid_cache_path, verify=True)
        binding = read_mapping(Path(centroid_cache_path) / "location-binding.json")
        if (
            binding["catalog_signature"] != signature(train_catalog.path)
            or binding["cache_id"] != manifest["cache_id"]
            or manifest["profile"] != profile
            or manifest["split"] != "navtrain"
            or manifest["collection_id"] != active_collection_identity(train_catalog, profile)
        ):
            raise ValueError("Compact retrieval cache has stale source or collection identity")
        indices = [i for i, x in enumerate(cached_rows) if x["token"] in members]
        rows = [cached_rows[i] for i in indices]
        if len(rows) != len(members) or {x["token"] for x in rows} != set(members):
            raise ValueError("Compact retrieval cache does not cover declared training membership")
        values = np.asarray(matrix[indices])
        if progress:
            write_json(
                progress,
                dict(
                    status="reused_compact_centroids",
                    loaded_vectors=len(rows),
                    target_members=len(members),
                    elapsed_seconds=time.monotonic() - start,
                ),
            )
    else:
        # Sequential pages once, rather than scattered IN probes repeated per query.
        rows = []
        cursor = train_catalog.db.execute(
            "SELECT v.token,v.centroid,s.source_group FROM vectors v NOT INDEXED JOIN scenes s ON v.token=s.token "
            "WHERE v.profile=? AND v.status='ready'",
            (profile,),
        )
        while page := cursor.fetchmany(512):
            rows.extend(x for x in page if x["token"] in members)
            if progress:
                write_json(
                    progress,
                    dict(
                        status="loading_centroids",
                        loaded_vectors=len(rows),
                        target_members=len(members),
                        target_population=population,
                        target_collection_id=active_collection_identity(train_catalog, profile),
                        elapsed_seconds=time.monotonic() - start,
                    ),
                )
        rows.sort(key=lambda x: x["token"])
        if not rows:
            raise ValueError("No ready official training vectors")
        values = normalize(np.stack([np.frombuffer(x["centroid"], dtype="<f4") for x in rows]))
    cache = {}
    results = []
    for token in tokens:
        row = query.get(token, profile)
        q = normalize(np.frombuffer(row["centroid"], dtype="<f4"))
        best = []
        for offset in range(0, len(rows), 256):
            part = rows[offset : offset + 256]
            scores = values[offset : offset + 256] @ q
            best.extend(
                (float(s), x["token"], x["source_group"]) for s, x in zip(scores, part, strict=True)
            )
            best.sort(key=lambda x: (-x[0], x[1]))
            seen = set()
            best = [x for x in best if not (x[2] in seen or seen.add(x[2]))][:shortlist]
        qv = normalize(
            np.frombuffer(row["image_vectors"], dtype="<f2").reshape(row["views"], row["dimension"])
        )
        ranked = []
        for coarse, candidate, group in best:
            if candidate not in cache:
                v = target.get(candidate, profile)
                cache[candidate] = (
                    v,
                    normalize(
                        np.frombuffer(v["image_vectors"], dtype="<f2").reshape(
                            v["views"], v["dimension"]
                        )
                    ),
                )
            v, vv = cache[candidate]
            if vv.shape != qv.shape:
                raise ValueError("View layout mismatch")
            ranked.append(
                dict(
                    token=candidate,
                    source_log=members[candidate],
                    source_group=group,
                    similarity=float((qv * vv).sum(-1).mean()),
                    coarse_similarity=coarse,
                    input_id=v["input_id"],
                )
            )
        ranked.sort(key=lambda x: (-x["similarity"], x["token"]))
        results.append(
            dict(
                query_token=token,
                query_input_id=row["input_id"],
                scanned=len(rows),
                candidates=ranked[:limit],
            )
        )
        if progress:
            write_json(
                progress,
                dict(
                    status="retrieving",
                    completed_queries=len(results),
                    expected_queries=len(tokens),
                    target_population=population,
                    ready_vectors=len(rows),
                    centroid_scans=1,
                    elapsed_seconds=time.monotonic() - start,
                ),
            )
    return dict(
        schema_version=1,
        kind="private_navtest_to_training_retrieval",
        target_population=population,
        profile=profile,
        target_collection_id=active_collection_identity(train_catalog, profile),
        query_collection_id=active_collection_identity(query_catalog, profile),
        target_members=len(members),
        ready_vectors=len(rows),
        membership_files=files,
        rows=results,
        adaptive_evaluation=True,
        training_samples_from_test=False,
        claim="visual_similarity_not_confirmed_failure",
        publishable=False,
    )
