"""Training-only vector collection for qualified same-observation target updates.

A relabel is a new supervision identity, not a new image. The packed source
embedding is preserved exactly; no test query or test score is stored here.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.data.catalog.store import Catalog, encode, identity
from driversi.data.catalog.vectors import VectorStore, normalize


def register(source_catalog, profile, source_manifest, relabel_report, database):
    from driversi_agents.augustus.data import manifest, read_cache, row_digest

    proof = read_mapping(relabel_report)
    if proof.get("kind") != "augustus_relabel_dataset" or proof.get("new_images") != 0:
        raise ValueError("Only verified unchanged-observation relabel can reuse features")
    if (
        digest(source_manifest) != proof["source_manifest_sha256"]
        or digest(proof["manifest"]) != proof["manifest_sha256"]
    ):
        raise ValueError("Generated supervision changed")
    for name, sha in {**proof["source_cache_files"], **proof["cache_files"]}.items():
        if digest(name) != sha:
            raise ValueError("Supervision cache changed")
    original = manifest(source_manifest, allow_smoke=True)
    generated = manifest(proof["manifest"], allow_smoke=True)
    before = {x["frame_id"]: x for x in read_cache(original["splits"]["train"])}
    after = {x["frame_id"]: x for x in read_cache(generated["splits"]["train"])}
    cat = Catalog(source_catalog, "navtrain", readonly=True)
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        vectors = VectorStore(cat)
        config = vectors.profile_data(profile)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS generated_samples (
                sample_id TEXT PRIMARY KEY, source_token TEXT NOT NULL,
                source_log TEXT NOT NULL, profile TEXT NOT NULL,
                source_input_id TEXT NOT NULL, metadata TEXT NOT NULL,
                centroid BLOB NOT NULL, image_vectors BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS generated_profile ON generated_samples(profile);
        """)
        records = []
        for replacement in proof["replacements"]:
            token = replacement["token"]
            left, right = before[token], after[token]
            fields = {"future_x_y_angle", "solution_planning", "reasoning_trace"}
            if any(left[k] != right[k] for k in left if k not in fields):
                raise ValueError("Changed image/history requires fresh feature extraction")
            row = vectors.get(token, profile)
            scene = cat.scene(token)
            evidence = json.loads(row["evidence"])
            for image in evidence:
                if digest(image["path"]) != image["sha256"]:
                    raise ValueError("Source image changed; cannot reuse vector")
            sample_id = "relabel-" + identity(
                dict(token=token, row=row_digest(right), profile=profile)
            )
            metadata = dict(
                kind="original_anchor_relabel",
                split="navtrain",
                partition="navval",
                source_token=token,
                source_log=scene["record"]["source_log"],
                manifest=proof["manifest"],
                manifest_sha256=proof["manifest_sha256"],
                native_frame_id=token,
                row_sha256=row_digest(right),
                profile=config,
                image_evidence=evidence,
                feature_method="exact_source_vector_reuse_unchanged_observations",
                encoder_calls=0,
                new_images=0,
                admission_report=str(relabel_report),
                admission_sha256=digest(relabel_report),
            )
            values = (
                sample_id,
                token,
                scene["record"]["source_log"],
                profile,
                row["input_id"],
                encode(metadata),
                row["centroid"],
                row["image_vectors"],
            )
            old = db.execute(
                "SELECT * FROM generated_samples WHERE sample_id=?", (sample_id,)
            ).fetchone()
            if old is not None and old != values:
                raise ValueError("Generated record changed")
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO generated_samples VALUES (?,?,?,?,?,?,?,?)", values
                )
            records.append(
                dict(sample_id=sample_id, source_token=token, source_input_id=row["input_id"])
            )
        return dict(
            status="completed",
            database=str(path),
            records=records,
            registered=len(records),
            encoder_calls=0,
            new_images=0,
            feature_method="exact_source_vector_reuse_unchanged_observations",
        )
    finally:
        db.close()
        cat.close()


def search(database, profile, query_vector, limit=10):
    """Training callers retrieve generated sample IDs plus native consumer references."""
    if not 1 <= limit <= 100:
        raise ValueError("Bounded search required")
    q = normalize(np.asarray(query_vector, dtype=np.float32))
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        result = []
        for sid, metadata, blob in db.execute(
            "SELECT sample_id,metadata,centroid FROM generated_samples WHERE profile=?", (profile,)
        ):
            vector = np.frombuffer(blob, dtype="<f4")
            if q.shape != vector.shape:
                raise ValueError("Query profile shape mismatch")
            result.append(
                dict(sample_id=sid, cosine=float(q @ vector), sample=json.loads(metadata))
            )
        return sorted(result, key=lambda x: (-x["cosine"], x["sample_id"]))[:limit]
    finally:
        db.close()
