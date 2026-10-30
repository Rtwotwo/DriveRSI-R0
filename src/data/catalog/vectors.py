"""Split-local image embeddings, exact coarse retrieval and aligned-view reranking.

These features describe visual similarity; they are not failure probabilities or
text-aligned CLIP embeddings. Keep the encoder and layout explicit in profiles.
"""

import json
import time

import numpy as np

from .store import encode, identity


def normalize(array):
    value = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or not np.isfinite(norm).all() or np.any(norm < 1e-8):
        raise ValueError("Nonfinite or zero-norm embedding")
    return value / norm


class VectorStore:
    def __init__(self, catalog, *, writable=False):
        self.catalog, self.db = catalog, catalog.db
        if writable:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS vector_profiles (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS vectors (
                    profile TEXT REFERENCES vector_profiles(id),
                    token TEXT REFERENCES scenes(token), input_id TEXT NOT NULL,
                    status TEXT NOT NULL, evidence TEXT NOT NULL, cost TEXT NOT NULL,
                    error TEXT, dimension INTEGER, views INTEGER, centroid BLOB, image_vectors BLOB,
                    PRIMARY KEY(profile,token));
            """)

    def profile(self, config):
        key = identity(config)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO vector_profiles VALUES (?,?)", (key, encode(config)))
        return key

    def profile_data(self, profile):
        row = self.db.execute("SELECT config FROM vector_profiles WHERE id=?", (profile,)).fetchone()
        if row is None:
            raise ValueError("Unknown embedding profile")
        return json.loads(row[0])

    def completed(self, token, profile, evidence):
        row = self.db.execute("SELECT input_id,status FROM vectors WHERE profile=? AND token=?",
                              (profile, token)).fetchone()
        return row is not None and tuple(row) == (identity(evidence), "ready")

    def put(self, token, profile, evidence, values, cost, error=None):
        self.catalog.scene(token)
        config = self.profile_data(profile)
        expected = [f"h{i}_{c}" for i in range(4) for c in config["cameras"]]
        if values is not None:
            if [e["id"] for e in evidence] != expected:
                raise ValueError("Image feature order differs from declared history/cameras")
            values = normalize(values)
            if values.shape != (len(expected), config["dimension"]):
                raise ValueError("Embedding dimensions differ from profile")
            centroid = normalize(values.mean(axis=0)).astype("<f4").tobytes()
            packed = values.astype("<f2").tobytes()
            status = "ready"
        else:
            if not error:
                raise ValueError("Failed extraction requires a reason")
            centroid, packed, status = None, None, "failed"
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                profile, token, identity(evidence), status, encode(evidence), encode(cost), error,
                config["dimension"], len(expected), centroid, packed,
            ))

    def get(self, token, profile):
        row = self.db.execute("SELECT * FROM vectors WHERE profile=? AND token=? AND status='ready'",
                              (profile, token)).fetchone()
        if row is None:
            raise ValueError("No ready embedding for query in this catalog/profile")
        return row

    def search(self, purpose, token, profile, *, limit=10, shortlist=128,
               exclude_source=True, diverse_sources=True, filters=None):
        if type(limit) is not int or type(shortlist) is not int or not 1 <= limit <= shortlist <= 4096:
            raise ValueError("Require 1 <= limit <= shortlist <= 4096")
        if type(exclude_source) is not bool or type(diverse_sources) is not bool:
            raise ValueError("Source exclusion and diversity must be booleans")
        config = self.profile_data(profile)
        query = self.get(token, profile)
        query_scene = self.catalog.scene(token)
        filters = filters or {}
        if set(filters) - {"tags", "profile", "visual_tags", "text", "score_run", "score_below"}:
            raise ValueError("Unknown vector retrieval filter")
        # The same split/purpose and label-quality checks as lexical retrieval.
        sql, params = self.catalog.token_query(purpose, limit=None, **filters)
        where = "v.profile=? AND v.status='ready' AND v.token!=? AND v.token IN (" + sql + ")"
        params = [profile, token, *params]
        if exclude_source:
            where += " AND s.source_group!=?"
            params.append(query_scene["descriptor"]["source_group"])
        cursor = self.db.execute(
            "SELECT v.token,v.centroid,s.source_group FROM vectors v NOT INDEXED JOIN scenes s ON s.token=v.token WHERE "
            + where, params,
        )
        query_centroid = np.frombuffer(query["centroid"], dtype="<f4")
        started, scanned, best = time.monotonic(), 0, []
        while rows := cursor.fetchmany(512):
            matrix = np.stack([np.frombuffer(r["centroid"], dtype="<f4") for r in rows])
            scores = matrix @ query_centroid
            best.extend((float(s), r["token"], r["source_group"]) for s, r in zip(scores, rows, strict=True))
            best.sort(key=lambda x: (-x[0], x[1]))
            if diverse_sources:
                # Keep the best coarse representative of each source group.
                # This prevents one near-duplicate log from filling the shortlist.
                used = set()
                best = [x for x in best if not (x[2] in used or used.add(x[2]))]
            best = best[:shortlist]
            scanned += len(rows)
        query_views = normalize(np.frombuffer(query["image_vectors"], dtype="<f2").reshape(
            query["views"], query["dimension"]))
        ranked = []
        for coarse, candidate, source in best:
            row = self.get(candidate, profile)
            views = normalize(np.frombuffer(row["image_vectors"], dtype="<f2").reshape(
                row["views"], row["dimension"]))
            similarities = (query_views * views).sum(axis=-1)
            ranked.append({
                "token": candidate, "source_group": source,
                "coarse_cosine": coarse, "aligned_view_cosine": float(similarities.mean()),
                "per_image_cosine": similarities.tolist(), "input_id": row["input_id"],
                "evidence": json.loads(row["evidence"]),
            })
        ranked.sort(key=lambda r: (-r["aligned_view_cosine"], r["token"]))
        report = {
            "split": self.catalog.split, "purpose": purpose, "profile": profile,
            "encoder": config, "query_token": token, "query_input_id": query["input_id"],
            "query_evidence": json.loads(query["evidence"]), "filters": filters,
            "exclude_source": exclude_source, "diverse_sources": diverse_sources,
            "limit": limit, "shortlist": shortlist, "scanned_vectors": scanned,
            "search_seconds": time.monotonic() - started, "results": ranked[:limit],
            "claim": "visual_similarity_not_causal_failure_or_synthesis_admission",
            "ranking": "exact_centroid_shortlist_then_aligned_image_cosine",
            "diversity": "best_coarse_representative_per_source_before_rerank" if diverse_sources else "none",
            "trainable": False,
        }
        recovery = getattr(self.catalog, "_history_recovery_view", None)
        if recovery and profile == recovery[0]:
            from .history_recovery import retrieval_provenance
            retrieval_provenance(self.catalog, report, *recovery)
        return report

    def stats(self, profile):
        return dict(self.db.execute("SELECT status,count(*) FROM vectors WHERE profile=? GROUP BY status",
                                    (profile,)))
