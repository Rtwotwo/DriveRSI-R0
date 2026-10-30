"""Fresh features of actual synthesized history; separate from source vectors.

Unadmitted pixels can be encoded for inspection, but are quarantined. Only an
independently validated native sample can make an entry training-searchable.
"""

import io
import json
import pickle
import sqlite3
import time
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.data.catalog.store import Catalog, identity
from driversi.data.catalog.vectors import VectorStore, normalize
from driversi.runtime.budget import write_json


def history(export_path, cameras, max_side):
    from PIL import Image

    from driversi.rsi.plugins.synthesis import check_files

    export = json.loads(Path(export_path).read_text())
    check_files(export["files"])
    log = Path(export["log_path"]).resolve(strict=True)
    if export["files"].get(str(log)) != digest(log):
        raise ValueError("Unsealed synthetic log")
    with log.open("rb") as f:
        frames = pickle.load(f)
    if len(frames) < 12:
        raise ValueError("Full native supervision window required")
    root = Path(export["sensor_root"]).resolve(strict=True)
    times = [int(x["timestamp"]) for x in frames[:4]]
    if any(not 450000 <= b - a <= 550000 for a, b in zip(times, times[1:])):
        raise ValueError("Four continuous 2 Hz generated history frames required")
    images, evidence = [], []
    for i, frame in enumerate(frames[:4]):
        for cam in cameras:
            path = (root / frame["cams"][cam]["data_path"]).resolve(strict=True)
            if not path.is_relative_to(root) or export["files"].get(str(path)) != digest(path):
                raise ValueError("Unsealed generated pixel input")
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as img:
                img = img.convert("RGB")
                original = list(img.size)
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                images.append(img.copy())
                evidence.append(
                    dict(
                        id=f"h{i}_{cam}",
                        camera=cam,
                        history_index=i,
                        timestamp_us=times[i],
                        relative_time_s=(times[i] - times[-1]) / 1e6,
                        path=str(path),
                        sha256=digest(path),
                        original_size=original,
                        input_size=list(img.size),
                    )
                )
    return export, evidence, images


def encoder_compatibility(expected, actual):
    """A pinned SQL-only migration is not a change to the feature computation.

    Keep the new producer identity. Permit comparison only for the audited pair;
    every weight, runtime, camera, pooling and pixel preprocessing field matches.
    """
    if expected == actual:
        return dict(compatible=True, reason="exact_profile")
    old = "2276d62b49992c747f1ca8c785bb40c09ca2013521f9e3ac61465fef1b06f684"
    allowed = {
        "d53ea2e357e596715b3ceb53643339fe8d48fe973714afa57992f69e59fe10d5": "audited_sql_scan_order_only_c905df8_to_6459e3f",
        "d63732294e37d2bfa6cd39dde9267ad327b075b299bb65c8ec4a8d9c0a70686f": "audited_sql_scan_and_supplemental_retrieval_provenance_8383861",
    }
    new = actual.get("source", {}).get("vectors.py")
    if expected.get("source", {}).get("vectors.py") != old or new not in allowed:
        raise ValueError("Unreviewed native vector implementation change")
    copy = {**actual, "source": {**actual["source"], "vectors.py": old}}
    if copy != expected:
        raise ValueError("Encoder checkpoint/runtime/preprocessing differs from retrieval profile")
    return dict(
        compatible=True,
        reason=allowed[new],
        old_vectors_sha256=old,
        new_vectors_sha256=new,
        feature_math_changed=False,
        producer_profile_kept_distinct=True,
    )


def encode(
    export_path, source_catalog, profile_id, database, output, *, sample_report=None, _session=None
):
    """Real native encoder, exact old profile; never copies parent embeddings."""
    from driversi.data.catalog.features import VisionEncoder, feature_profile
    from driversi.rsi.experiments.navval_retrieval import training_members

    export_path = Path(export_path).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cat = Catalog(source_catalog, "navtrain", readonly=True)
    try:
        expected = VectorStore(cat).profile_data(profile_id)
        actual = (
            _session.profile(expected)
            if _session is not None
            else feature_profile(expected["config"])
        )
        compatibility = encoder_compatibility(expected, actual)
        retrieval_profile = profile_id
        profile_id = identity(actual)
        export, evidence, images = history(
            export_path, actual["cameras"], actual["config"]["max_side"]
        )
        origin = export["parent"]
        if _session is None:
            allowed, _ = training_members(cat, "navtrain")
        else:
            allowed = _session.member_map(cat)
        membership = export.get("source_train_membership", origin)
        if allowed.get(membership["token"]) != membership["source_log"]:
            raise ValueError("Synthetic source is outside official navtrain")
    finally:
        cat.close()
    source_id = identity(dict(export=digest(export_path), pixels=evidence, profile=profile_id))
    database = Path(database).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database, timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS generated_vectors (id TEXT PRIMARY KEY, profile TEXT NOT NULL, parent_token TEXT NOT NULL, source_log TEXT NOT NULL, evidence TEXT NOT NULL, centroid BLOB NOT NULL, views BLOB NOT NULL, payload_sha TEXT NOT NULL, receipt TEXT NOT NULL, admission TEXT, searchable INTEGER NOT NULL DEFAULT 0)"
        )
        old = db.execute(
            "SELECT centroid,views,payload_sha,receipt,admission,searchable FROM generated_vectors WHERE id=?",
            (source_id,),
        ).fetchone()
        if old:
            import hashlib

            if hashlib.sha256(old[0] + old[1]).hexdigest() != old[2]:
                raise ValueError("Stored generated vectors corrupted")
            report = json.loads(old[3])
            report = dict(report, reused=True, new_encoder_calls=0)
        else:
            start = time.monotonic()
            values, cost = (
                _session.encode(images)
                if _session is not None
                else VisionEncoder(actual["config"])(images)
            )
            values = normalize(values)
            if values.shape != (4 * len(actual["cameras"]), actual["dimension"]):
                raise ValueError("Native output dimensions mismatch")
            centroid = normalize(values.mean(0)).astype("<f4").tobytes()
            packed = values.astype("<f2").tobytes()
            import hashlib

            report = dict(
                kind="fresh_generated_history_features",
                id=source_id,
                profile=profile_id,
                retrieval_profile=retrieval_profile,
                compatibility=compatibility,
                actual_encoder_profile=actual,
                export=str(export_path),
                export_sha256=digest(export_path),
                source=origin,
                images=evidence,
                cost={**cost, "total_seconds": time.monotonic() - start},
                dimension=actual["dimension"],
                views=len(images),
                reused=False,
                new_encoder_calls=1,
                parent_vectors_copied=False,
                future_images_encoded=False,
                trainable=False,
                payload_sha256=hashlib.sha256(centroid + packed).hexdigest(),
            )
            with db:
                db.execute(
                    "INSERT INTO generated_vectors VALUES (?,?,?,?,?,?,?,?,?,?,0)",
                    (
                        source_id,
                        profile_id,
                        origin["token"],
                        origin["source_log"],
                        json.dumps(evidence),
                        centroid,
                        packed,
                        report["payload_sha256"],
                        json.dumps(report),
                        None,
                    ),
                )
        if sample_report is None and old and old[5]:
            stored = json.loads(old[4])
            if digest(stored["path"]) != stored["sha256"]:
                raise ValueError("Stored feature admission changed")
            sample_report = stored["path"]
        if sample_report is not None:
            from driversi.data.datasets.native_admission import validate_sample

            sample = validate_sample(sample_report)
            if sample["source_log"] != origin["source_log"] or any(
                sample["files"].get(e["path"]) != e["sha256"] for e in evidence
            ):
                raise ValueError("Native admitted sample and encoded pixels differ")
            admission = json.dumps(
                dict(path=str(Path(sample_report).resolve()), sha256=digest(sample_report))
            )
            with db:
                db.execute(
                    "UPDATE generated_vectors SET admission=?,searchable=1 WHERE id=?",
                    (admission, source_id),
                )
            report.update(trainable=True, admission=json.loads(admission))
        else:
            report.update(trainable=False, scope="quarantine_not_training_search")
        from driversi.rsi.decisions.scene_index import describe_records
        from driversi.rsi.experiments.full_training_mining import describe

        descriptor = describe_records(
            [
                dict(
                    token=export["token"],
                    source_log=export["native_log_name"],
                    log_path=export["log_path"],
                    anchor_index=export["anchor_index"],
                )
            ]
        )[0]
        text = describe(descriptor)
        db.execute(
            "CREATE TABLE IF NOT EXISTS generated_text(id TEXT PRIMARY KEY REFERENCES generated_vectors(id), document TEXT NOT NULL, descriptor TEXT NOT NULL, scope TEXT NOT NULL)"
        )
        encoded = json.dumps(descriptor, sort_keys=True)
        prior = db.execute(
            "SELECT document,descriptor FROM generated_text WHERE id=?", (source_id,)
        ).fetchone()
        if prior and prior != (text, encoded):
            raise ValueError("Generated history text changed for the same sealed image feature")
        with db:
            db.execute(
                "INSERT OR IGNORE INTO generated_text VALUES (?,?,?,?)",
                (source_id, text, encoded, "executed_four_frame_history_only"),
            )
        report["text_index"] = dict(
            id=source_id,
            database=str(database),
            kind="structured_history_text",
            future_used=False,
            searchable=report["trainable"],
            vlm_caption=False,
        )
        write_json(output / "report.json", report)
        return report
    finally:
        db.close()


def search(database, query_views, profile_id, *, limit=8):
    """Retrieve only currently admitted generated pixels in a verified feature space."""
    import numpy as np

    from driversi.data.datasets.native_admission import validate_sample

    if not 1 <= limit <= 64:
        raise ValueError("Bounded generated retrieval required")
    query = normalize(np.asarray(query_views, dtype=np.float32))
    if query.ndim != 2 or not np.isfinite(query).all():
        raise ValueError("Finite aligned query views required")
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    results = []
    try:
        for key, views, receipt, admission, centroid, payload_sha in db.execute(
            "SELECT id,views,receipt,admission,centroid,payload_sha FROM generated_vectors WHERE searchable=1"
        ):
            import hashlib

            if hashlib.sha256(centroid + views).hexdigest() != payload_sha:
                raise ValueError("Corrupt generated search vectors")
            value = json.loads(receipt)
            if profile_id not in (value["profile"], value.get("retrieval_profile")):
                continue
            proof = json.loads(admission)
            if digest(proof["path"]) != proof["sha256"]:
                raise ValueError("Generated search admission changed")
            validate_sample(proof["path"])
            vectors = normalize(
                np.frombuffer(views, dtype="<f2").reshape(value["views"], value["dimension"])
            )
            if vectors.shape != query.shape:
                raise ValueError("Generated query view layout mismatch")
            results.append(
                dict(
                    id=key,
                    source=value["source"],
                    sample_report=proof["path"],
                    similarity=float((query * vectors).sum(-1).mean()),
                    kind="admitted_generated_history",
                    profile=value["profile"],
                )
            )
        return sorted(results, key=lambda x: (-x["similarity"], x["id"]))[:limit]
    finally:
        db.close()


def search_text(database, descriptor, *, limit=8):
    """Historical condition retrieval in the generated partition; admission still required."""
    import math
    from collections import Counter

    from driversi.data.datasets.native_admission import validate_sample
    from driversi.rsi.experiments.full_training_mining import describe

    if not 1 <= limit <= 64:
        raise ValueError("Bounded generated text retrieval required")
    terms = set(describe(descriptor).split())
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='generated_text'").fetchone():
            return []
        rows = list(
            db.execute(
                "SELECT v.id,t.document,v.receipt,v.admission FROM generated_vectors v JOIN generated_text t ON t.id=v.id WHERE v.searchable=1 AND t.scope='executed_four_frame_history_only'"
            )
        )
        frequency = Counter(word for _, text, _, _ in rows for word in set(text.split()))
        result = []
        for key, text, receipt, admission in rows:
            shared = terms & set(text.split())
            if not shared:
                continue
            proof = json.loads(admission)
            if digest(proof["path"]) != proof["sha256"]:
                raise ValueError("Generated text admission changed")
            validate_sample(proof["path"])
            value = json.loads(receipt)
            if digest(value["export"]) != value["export_sha256"]:
                raise ValueError("Generated text source changed")
            score = sum(math.log(1 + (len(rows) + 1) / (frequency[t] + 1)) for t in shared)
            result.append(
                dict(
                    id=key,
                    source=value["source"],
                    sample_report=proof["path"],
                    text_score=score,
                    shared_terms=sorted(shared),
                    kind="admitted_generated_history_text",
                )
            )
        return sorted(result, key=lambda x: (-x["text_score"], x["id"]))[:limit]
    finally:
        db.close()
