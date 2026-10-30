"""Persistent split-local membership for the main visual retrieval collection.

Only the registration is persistent. Reads compose existing native and recovered
vectors; source rows, profile identities and failed temporal checks are untouched.
"""

import hashlib
import json

import numpy as np

from .store import encode, identity

KEY = "visual_retrieval_collection"


def _validate_profiles(catalog, base_profile, recovery_profile):
    from .history_recovery import KIND, feature_base
    from .vectors import VectorStore

    store = VectorStore(catalog)
    base, recovered = store.profile_data(base_profile), store.profile_data(recovery_profile)
    if (recovered.get("kind") != KIND
            or recovered.get("recovery", {}).get("base_profile") != base_profile
            or feature_base(recovered) != base):
        raise ValueError("Recovery profile is not encoder-compatible with base")
    return base


def register_collection(catalog, base_profile, recovery_profile):
    """Validate existing supplement bytes and bind them to the main reader.

    Idempotent and append-only: changing an existing registration needs an
    explicit new catalog version. No feature extraction or model scoring occurs.
    """
    base = _validate_profiles(catalog, base_profile, recovery_profile)
    expected = [f"h{i}_{camera}" for i in range(4) for camera in base["cameras"]]
    rows = catalog.db.execute(
        "SELECT * FROM main.vectors WHERE profile=? ORDER BY token", (recovery_profile,)
    ).fetchall()
    if not rows:
        raise ValueError("No supplemental vectors to register")
    fingerprints = []
    for row in rows:
        if row["status"] != "ready":
            raise ValueError("Every registered supplemental vector must be ready")
        scene = catalog.scene(row["token"])
        if (scene["record"]["kind"] != "original_" + catalog.split
                or scene["descriptor"]["history"]["continuous"] is not False):
            raise ValueError("Supplement source must be split-local observed irregular history")
        prior = catalog.db.execute(
            "SELECT status,error FROM main.vectors WHERE profile=? AND token=?",
            (base_profile, row["token"]),
        ).fetchone()
        if prior is None or tuple(prior) != (
                "failed", "ValueError: Irregular history is not silently padded"):
            raise ValueError("Supplement must retain its original temporal failure")
        evidence = json.loads(row["evidence"])
        if [e["id"] for e in evidence] != expected or identity(evidence) != row["input_id"]:
            raise ValueError("Supplement evidence identity or camera layout differs")
        times = [e["timestamp_us"] for e in evidence[::len(base["cameras"])]]
        if (any(b <= a for a, b in zip(times, times[1:]))
                or times[-1] != scene["descriptor"]["history"]["anchor_timestamp_us"]):
            raise ValueError("Supplement timestamps do not match the observed anchor")
        for i, e in enumerate(evidence):
            if (e.get("synthetic") is not False
                    or e.get("temporal_training_admitted") is not False
                    or e.get("source_kind") != "original_observed_image"
                    or e.get("history_sampling") != "observed_irregular"
                    or e["timestamp_us"] != times[i // len(base["cameras"])]):
                raise ValueError("Supplement is not original observed history evidence")
        if row["views"] != len(expected) or row["dimension"] != base["dimension"]:
            raise ValueError("Supplement dimensions differ")
        for field, dtype, size in (
            ("centroid", "<f4", base["dimension"]),
            ("image_vectors", "<f2", len(expected) * base["dimension"]),
        ):
            values = np.frombuffer(row[field], dtype=dtype)
            if values.size != size or not np.isfinite(values).all():
                raise ValueError("Supplement tensor bytes are incomplete or nonfinite")
        fingerprints.append({"token": row["token"], "input_id": row["input_id"],
            "bytes_sha256": hashlib.sha256(row["centroid"] + row["image_vectors"]).hexdigest()})
    value = {"schema_version": 1, "split": catalog.split, "base_profile": base_profile,
        "history_recovery_profile": recovery_profile, "supplemental_rows": len(rows),
        "supplement_identity": identity(fingerprints),
        "semantics": "ordered_real_visual_views_not_temporal_alignment",
        "training_history_repaired": False}
    with catalog.db:
        previous = catalog.db.execute("SELECT value FROM metadata WHERE key=?", (KEY,)).fetchone()
        if previous and previous[0] != encode(value):
            raise ValueError("Main retrieval registration changed; use a new catalog version")
        catalog.db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (KEY, encode(value)))
    return value


def activate_registered_collection(catalog):
    """Install the registered main collection on a read-only connection."""
    row = catalog.db.execute("SELECT value FROM metadata WHERE key=?", (KEY,)).fetchone()
    if row is None:
        return None
    value = json.loads(row[0])
    if value.get("schema_version") != 1 or value.get("split") != catalog.split:
        raise ValueError("Main retrieval collection schema/split mismatch")
    from .history_recovery import enable_recovery_view
    enable_recovery_view(catalog, value["base_profile"], value["history_recovery_profile"])
    return value


def active_collection_identity(catalog, profile):
    """Cache namespace for the activated split-local view (no vector scan)."""
    active = getattr(catalog, '_history_recovery_view', None)
    if not active or active[0] != profile:
        return None
    row = catalog.db.execute('SELECT value FROM metadata WHERE key=?', (KEY,)).fetchone()
    if row is None:
        raise ValueError('Active collection has no persistent registration')
    value = json.loads(row[0])
    if (value.get('split') != catalog.split or value.get('base_profile') != profile
            or value.get('history_recovery_profile') != active[1]):
        raise ValueError('Active collection and persistent registration differ')
    return identity(value)
