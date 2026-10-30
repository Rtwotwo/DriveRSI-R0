"""Immutable compact scene centroids; source SQLite files are never modified."""

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from .retrieval_collection import active_collection_identity
from .store import Catalog, identity
from .vectors import VectorStore, normalize


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def signature(path):
    """Conservative cache invalidation, including uncheckpointed SQLite writes."""
    path = Path(path).resolve()
    result = {}
    for p in (path, Path(str(path) + "-wal")):
        result[str(p)] = (
            [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino] if p.exists() and (p == path or p.stat().st_size > 0) else None
        )
    return result


def open_cache(output, *, verify=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Incomplete centroid cache")
    if verify:
        for name, sha in manifest["files"].items():
            if digest(output / name) != sha:
                raise ValueError("Centroid cache content changed")
    rows = json.loads((output / "members.json").read_text())
    values = np.load(output / "centroids.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != (manifest["count"], manifest["dimension"]) or len(rows) != len(values):
        raise ValueError("Cache dimensions differ")
    if len({r["token"] for r in rows}) != len(rows):
        raise ValueError("Duplicate cache member")
    return manifest, rows, values


def build_cache(catalog_path, split, profile, output, *, scratch=None):
    """Backup to node-local storage to avoid random reads of NAS BLOB pages.

    SQLite backup supplies a consistent snapshot. A conservative source signature
    must additionally remain stable throughout the build. Publication is atomic;
    after interruption the same command safely rebuilds the incomplete output.
    """
    import fcntl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    with (output / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_signature = signature(catalog_path)
        request = dict(source=source_signature, split=split, profile=profile, version=1)
        if (output / "manifest.json").exists():
            manifest, _, _ = open_cache(output, verify=True)
            if manifest["request"] != request:
                raise ValueError("Source changed; build a new versioned cache directory")
            return dict(manifest, reused=True)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="driversi-centroids-", dir=scratch) as tmp:
            local = Path(tmp) / "catalog.sqlite"
            source = sqlite3.connect(Path(catalog_path).resolve().as_uri() + "?mode=ro", uri=True)
            destination = sqlite3.connect(local)
            last = 0.0

            def progress(status, remaining, total):
                nonlocal last
                if time.monotonic() - last >= 3 or remaining == 0:
                    write_json(output / "progress.json", dict(
                        phase="snapshot", remaining_pages=remaining, total_pages=total,
                        elapsed_seconds=time.monotonic() - start))
                    last = time.monotonic()

            try:
                source.backup(destination, pages=4096, progress=progress)
            finally:
                source.close()
                destination.close()
            catalog = Catalog(local, split, readonly=True)
            try:
                store = VectorStore(catalog)
                config = store.profile_data(profile)
                provenance_row = catalog.db.execute(
                    "SELECT value FROM metadata WHERE key='source'").fetchone()
                provenance = json.loads(provenance_row[0]) if provenance_row else None
                scenes = {}
                for row in catalog.db.execute(
                    "SELECT token,source_log,source_group,descriptor,record FROM scenes"):
                    desc, record = json.loads(row["descriptor"]), json.loads(row["record"])
                    if record.get("kind") != "original_" + split:
                        raise ValueError("Only original split-local scenes may build this pool")
                    if desc.get("history", {}).get("information_set") != "current_state_history":
                        raise ValueError("Only past/current scene facts accepted")
                    scenes[row["token"]] = dict(
                        token=row["token"], source_log=row["source_log"],
                        source_group=row["source_group"], features=desc.get("features", {}),
                        conditions=desc.get("conditions", []),
                        history_continuous=desc["history"].get("continuous"),
                        information_set="current_state_history")
                rows, vectors = [], []
                for row in catalog.db.execute(
                    "SELECT token,input_id,centroid,dimension FROM vectors "
                    "WHERE profile=? AND status='ready'", (profile,)):
                    if row["token"] not in scenes or row["dimension"] != config["dimension"]:
                        raise ValueError("Vector member/dimension mismatch")
                    v = np.frombuffer(row["centroid"], dtype="<f4")
                    if v.shape != (config["dimension"],):
                        raise ValueError("Incomplete centroid bytes")
                    vectors.append(normalize(v))
                    rows.append(dict(scenes[row["token"]], input_id=row["input_id"]))
                if len(rows) != len(scenes) or len({x["token"] for x in rows}) != len(scenes):
                    raise ValueError("Ready vector coverage must match complete catalog membership")
                order = sorted(range(len(rows)), key=lambda i: rows[i]["token"])
                rows = [rows[i] for i in order]
                matrix = np.stack([vectors[i] for i in order]).astype("<f4")
                with (output / "centroids.npy.partial").open("wb") as f:
                    np.save(f, matrix, allow_pickle=False)
                os.chmod(output / "centroids.npy.partial", 0o600)
                (output / "centroids.npy.partial").replace(output / "centroids.npy")
                write_json(output / "members.json", rows)
                if signature(catalog_path) != source_signature:
                    raise ValueError("Source changed during snapshot; no cache published")
                manifest = dict(
                    status="complete", schema_version=1, request=request, split=split,
                    profile=profile, encoder=config, source_provenance=provenance,
                    collection_id=active_collection_identity(catalog, profile),
                    count=len(rows), dimension=matrix.shape[1],
                    membership_id=identity([r["token"] for r in rows]),
                    files={n: digest(output / n) for n in ["centroids.npy", "members.json"]},
                    elapsed_seconds=time.monotonic() - start, publishable=False,
                    temporal_training_admitted=False)
                manifest["cache_id"] = identity(manifest)
                write_json(output / "manifest.json", manifest)
                write_json(output / "progress.json", dict(phase="complete", count=len(rows)))
                return dict(manifest, reused=False)
            finally:
                catalog.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--split", choices=["navtrain", "navtest"], required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch")
    args = parser.parse_args()
    result = build_cache(args.catalog, args.split, args.profile, args.output, scratch=args.scratch)
    print(json.dumps({k: result[k] for k in ["split", "count", "elapsed_seconds", "reused"]}))


if __name__ == "__main__":
    main()
