"""Private cross-split visual retrieval with an explicit official navval allowlist."""

import numpy as np
import yaml

from driversi.agents.backends.native import digest, project_root
from driversi.data.catalog.vectors import VectorStore, normalize
from driversi_agents.chainflow.benchmark import official_filter


def navval_members(catalog):
    if catalog.split != "navtrain":
        raise ValueError("navval must be drawn only from the training catalog")
    root = project_root()
    logs_path = (
        root
        / "agents/chainflow/native/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
    )
    logs = set(yaml.safe_load(logs_path.read_text())["val_logs"])
    split_path, spec = official_filter("navtrain")
    allowed = set(spec["tokens"])
    members = {
        r["token"]: r["source_log"]
        for r in catalog.db.execute("SELECT token,source_log FROM scenes")
        if r["source_log"] in logs and r["token"] in allowed
    }
    return members, {str(p): digest(p) for p in (logs_path, split_path)}


def training_members(catalog, population="navtrain"):
    """Exact official 103k membership; navval is an explicit optional subset."""
    if population == "navval":
        return navval_members(catalog)
    if population != "navtrain" or catalog.split != "navtrain":
        raise ValueError("Only declared official training sources are allowed")
    path, spec = official_filter("navtrain")
    expected, logs = set(spec["tokens"]), set(spec["log_names"])
    members = {
        r["token"]: r["source_log"]
        for r in catalog.db.execute("SELECT token,source_log FROM scenes")
        if r["token"] in expected and r["source_log"] in logs
    }
    if set(members) != expected:
        raise ValueError("Training catalog does not cover the exact full official split")
    return members, {str(path): digest(path)}


def retrieve(query_catalog, train_catalog, profile, tokens, *, limit=4, shortlist=64):
    if query_catalog.split != "navtest" or train_catalog.split != "navtrain":
        raise ValueError("Explicit test query / training target catalogs required")
    if not tokens or len(tokens) != len(set(tokens)) or not 1 <= limit <= shortlist <= 256:
        raise ValueError("Unique bounded queries and candidates required")
    query, target = VectorStore(query_catalog), VectorStore(train_catalog)
    config = query.profile_data(profile)
    if config != target.profile_data(profile):
        raise ValueError("Query and target embedding profiles differ")
    members, files = navval_members(train_catalog)
    results = []
    for token in tokens:
        row = query.get(token, profile)
        q = normalize(np.frombuffer(row["centroid"], dtype="<f4"))
        best, scanned = [], 0
        keys = sorted(members)

        def batches():
            for start in range(0, len(keys), 256):
                part = keys[start : start + 256]
                marks = ",".join("?" for _ in part)
                yield train_catalog.db.execute(
                    "SELECT v.token,v.centroid,s.source_group FROM vectors v JOIN scenes s ON v.token=s.token WHERE v.profile=? AND v.status='ready' AND v.token IN ("
                    + marks
                    + ")",
                    (profile, *part),
                ).fetchall()

        for batch in batches():
            batch = [b for b in batch if b["token"] in members]
            if not batch:
                continue
            values = normalize(np.stack([np.frombuffer(b["centroid"], dtype="<f4") for b in batch]))
            best.extend(
                (float(s), b["token"], b["source_group"])
                for s, b in zip(values @ q, batch, strict=True)
            )
            best.sort(key=lambda x: (-x[0], x[1]))
            seen = set()
            best = [b for b in best if not (b[2] in seen or seen.add(b[2]))][:shortlist]
            scanned += len(batch)
        qv = normalize(
            np.frombuffer(row["image_vectors"], dtype="<f2").reshape(row["views"], row["dimension"])
        )
        ranked = []
        for coarse, candidate, group in best:
            v = target.get(candidate, profile)
            vv = normalize(
                np.frombuffer(v["image_vectors"], dtype="<f2").reshape(v["views"], v["dimension"])
            )
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
                scanned=scanned,
                candidates=ranked[:limit],
            )
        )
    return dict(
        schema_version=1,
        kind="private_navtest_to_navval_retrieval",
        profile=profile,
        navval_members=len(members),
        membership_files=files,
        rows=results,
        adaptive_evaluation=True,
        training_samples_from_test=False,
        claim="visual_similarity_not_confirmed_failure",
        publishable=False,
    )
