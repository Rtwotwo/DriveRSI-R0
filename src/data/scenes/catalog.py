"""Read-only dataset discovery and provenance-aware NAVSIM scene catalogues."""

import json
import pickle
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from driversi.data.sources.navsim import (
    BACKEND_MAX_DURATION_SECONDS,
    SIM_DT_SECONDS,
    compile_frames,
    plan_horizon,
    read_log,
    validate_horizon_request,
    validate_log,
)

PARTITION_USES = {"mini": "engineering", "trainval": "unassigned", "test": "heldout"}
PARTITION_SPLITS = {"mini": "mini", "trainval": "unassigned", "test": "heldout"}
CATALOG_VERSION = 2
DATA_ERRORS = (ValueError, KeyError, TypeError, OSError, EOFError, pickle.UnpicklingError)
SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE logs (
 id INTEGER PRIMARY KEY, relative_path TEXT UNIQUE NOT NULL, partition TEXT NOT NULL,
 size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
 source_log TEXT, log_token TEXT, map_name TEXT, first_us INTEGER, last_us INTEGER,
 frames INTEGER, status TEXT NOT NULL, reason TEXT
);
CREATE TABLE scenes (
 id INTEGER PRIMARY KEY, log_id INTEGER NOT NULL REFERENCES logs(id),
 start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL, token TEXT NOT NULL,
 start_us INTEGER NOT NULL, end_us INTEGER NOT NULL, status TEXT NOT NULL,
 reason TEXT, actors INTEGER, controlled INTEGER, exclusions TEXT,
 planned_duration_seconds REAL, horizon_plan TEXT NOT NULL,
 UNIQUE(log_id, start_frame)
);
CREATE TABLE overlaps (
 log_a INTEGER NOT NULL REFERENCES logs(id), log_b INTEGER NOT NULL REFERENCES logs(id),
 first_us INTEGER NOT NULL, last_us INTEGER NOT NULL, PRIMARY KEY(log_a, log_b)
);
CREATE INDEX scene_status ON scenes(status);
CREATE INDEX log_source ON logs(source_log);
"""


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {r["key"]: json.loads(r["value"])
                    for r in connection.execute("SELECT * FROM metadata")}
        if metadata.get("schema_version") not in {1, CATALOG_VERSION} or metadata.get("state") != "complete":
            raise ValueError("Unsupported or incomplete scene catalogue")
    except Exception:
        connection.close()
        raise
    return connection


def catalog_metadata(connection: sqlite3.Connection) -> dict:
    return {r["key"]: json.loads(r["value"])
            for r in connection.execute("SELECT * FROM metadata")}


def catalog_summary(path: Path) -> dict:
    with closing(_connect(path)) as connection:
        report = catalog_metadata(connection)
        report["log_statuses"] = dict(connection.execute(
            "SELECT status, count(*) FROM logs GROUP BY status"))
        report["scene_statuses"] = dict(connection.execute(
            "SELECT status, count(*) FROM scenes GROUP BY status"))
        report["rejections"] = [dict(r) for r in connection.execute(
            "SELECT reason, count(*) AS count FROM scenes WHERE reason IS NOT NULL GROUP BY reason")]
        report["log_errors"] = [dict(r) for r in connection.execute(
            "SELECT relative_path, reason FROM logs WHERE status='rejected'")]
        report["cross_partition_log_overlaps"] = connection.execute(
            "SELECT count(*) FROM overlaps").fetchone()[0]
        report["overlap_unchecked_logs"] = [dict(r) for r in connection.execute(
            "SELECT relative_path,reason FROM logs WHERE first_us IS NULL OR source_log IS NULL")]
        return report


def _add_log(connection: sqlite3.Connection, path: Path, root: Path, partition: str,
             duration: float, windows_per_log: int, max_actors: int, trusted: bool,
             duration_mode: str, min_duration: float) -> None:
    before = path.stat()
    cursor = connection.execute(
        "INSERT INTO logs(relative_path, partition, size_bytes, mtime_ns, status) VALUES(?,?,?,?,?)",
        (path.relative_to(root).as_posix(), partition, before.st_size, before.st_mtime_ns, "reading"),
    )
    log_id = cursor.lastrowid
    try:
        frames = read_log(path, trusted=trusted)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source file changed during indexing")
        identities = {(str(f["log_name"]), str(f["map_location"])) for f in frames}
        if len(identities) != 1:
            raise ValueError("A source file must contain exactly one log and map")
        log_tokens = {str(f["log_token"]) for f in frames if f.get("log_token") is not None}
        if len(log_tokens) > 1:
            raise ValueError("Source file contains multiple original log tokens")
        source_log, map_name = identities.pop()
        timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.int64)
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("NAVSIM timestamps must be strictly increasing")
        connection.execute(
            "UPDATE logs SET source_log=?,log_token=?,map_name=?,first_us=?,last_us=?,"
            "frames=?,status='indexed' WHERE id=?",
            (source_log, next(iter(log_tokens), None), map_name, int(timestamps[0]),
             int(timestamps[-1]), len(frames), log_id),
        )
        validate_log(frames)
        last_start = (len(frames) - 2 if duration_mode == "fit_source" else int(
            np.searchsorted(timestamps, timestamps[-1] - round(duration * 1e6), side="right")) - 1)
        if last_start < 0:
            raise ValueError("Log does not cover the requested duration")
        starts = np.linspace(0, last_start, min(windows_per_log, last_start + 1), dtype=int)
        for start in starts:
            plan = plan_horizon(frames, start=int(start), duration=duration,
                                mode=duration_mode, min_duration=min_duration)
            end = plan["source_end_frame"] if plan["source_end_frame"] is not None else int(start)
            status, reason, actors, controlled, exclusions = "geometry_pending", None, None, None, {}
            try:
                scene = compile_frames(frames, start=int(start), duration=duration,
                                       split=PARTITION_SPLITS[partition], max_actors=max_actors,
                                       duration_mode=duration_mode, min_duration=min_duration)
                actors, controlled = len(scene.actor_ids), int(scene.controlled.sum())
                exclusions = {k: v for k, v in scene.metadata.items() if k.startswith("excluded_")}
            except DATA_ERRORS as error:
                status, reason = "rejected", str(error)
            connection.execute(
                "INSERT INTO scenes(log_id,start_frame,end_frame,token,start_us,end_us,status,"
                "reason,actors,controlled,exclusions,planned_duration_seconds,horizon_plan) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (log_id, int(start), end, str(frames[start]["token"]), int(timestamps[start]),
                 int(timestamps[end]), status, reason, actors, controlled, _json(exclusions),
                 plan["planned_duration_seconds"], _json(plan)),
            )
    except DATA_ERRORS as error:
        connection.execute("DELETE FROM scenes WHERE log_id=?", (log_id,))
        connection.execute("UPDATE logs SET status='rejected',reason=? WHERE id=?", (str(error), log_id))


def build_catalog(root: Path, output: Path, *, partitions: list[str], max_logs: int = 3,
                  windows_per_log: int = 2, duration: float = 3.0, max_actors: int = 32,
                  trusted: bool = False, duration_mode: str = "exact",
                  min_duration: float = SIM_DT_SECONDS) -> dict:
    """Bounded preflight. Passing windows still require map and backend validation."""
    if not trusted:
        raise ValueError("Indexing requires --trusted-pickle for trusted local data")
    if (max_logs < 1 or windows_per_log < 1 or not partitions
            or len(set(partitions)) != len(partitions) or not 1 <= max_actors <= 32):
        raise ValueError("Require distinct partitions and positive valid indexing limits")
    validate_horizon_request(duration, duration_mode, min_duration)
    duration, min_duration = float(duration), float(min_duration)
    if duration_mode == "exact" and duration > BACKEND_MAX_DURATION_SECONDS:
        raise ValueError("exact duration exceeds backend cap of 9 seconds")
    root = root.resolve()
    available = {p: partition_files(root, p) for p in partitions}
    if any(not files for files in available.values()):
        raise ValueError("A requested partition has no .pkl files")
    metadata = {
        "schema_version": CATALOG_VERSION, "state": "building", "source_format": "navsim_openscene",
        "source_version": "unknown", "dataset_root": str(root), "official_scene_filter": None,
        "partitions": {p: {"available_files": len(files), "selected_files": min(max_logs, len(files)),
                            "use": PARTITION_USES[p]} for p, files in available.items()},
        "selection": ("sorted filenames, evenly spaced non-final start-frame indices per log; failures retained"
                      if duration_mode == "fit_source" else
                      "sorted filenames, evenly spaced eligible start-frame indices per log"),
        "parameters": {"max_logs_per_partition": max_logs, "windows_per_log": windows_per_log,
                       "duration": duration, "max_actors": max_actors,
                       "duration_mode": duration_mode, "min_duration": min_duration},
        "overlap_scope": "indexed log names/tokens and closed source intervals only; not a global split guarantee",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a usable catalogue.
    with output.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(output)) as connection:
            connection.executescript(SCHEMA)
            connection.executemany("INSERT INTO metadata VALUES(?,?)",
                                   [(k, _json(v)) for k, v in metadata.items()])
            for partition, files in available.items():
                for path in files[:max_logs]:
                    _add_log(connection, path, root, partition, duration, windows_per_log,
                             max_actors, trusted, duration_mode, min_duration)
            connection.execute("""
                INSERT INTO overlaps
                SELECT a.id,b.id,max(a.first_us,b.first_us),min(a.last_us,b.last_us)
                FROM logs a JOIN logs b ON a.id < b.id AND a.partition != b.partition
                AND (a.source_log=b.source_log OR
                     (a.log_token IS NOT NULL AND a.log_token=b.log_token))
                WHERE a.first_us <= b.last_us AND b.first_us <= a.last_us
            """)
            connection.execute("""
                UPDATE scenes SET status='conflict',reason='cross_partition_source_overlap'
                WHERE status='geometry_pending' AND EXISTS (
                    SELECT 1 FROM overlaps o WHERE (o.log_a=scenes.log_id OR o.log_b=scenes.log_id)
                    AND scenes.start_us <= o.last_us AND o.first_us <= scenes.end_us)
            """)
            connection.execute("UPDATE metadata SET value=? WHERE key='state'", (_json("complete"),))
            connection.commit()
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return catalog_summary(output)


def list_scenes(path: Path, *, limit: int = 20, offset: int = 0,
                status: str = "geometry_pending", map_name: str | None = None,
                min_controlled: int = 1) -> list[dict]:
    if limit < 1 or offset < 0 or min_controlled < 0:
        raise ValueError("Invalid scene query limits")
    if status not in {"all", "geometry_pending", "rejected", "conflict"}:
        raise ValueError("Unknown scene status")
    where, values = [], []
    if status != "all":
        where.append("s.status=?")
        values.append(status)
    if map_name is not None:
        where.append("l.map_name=?")
        values.append(map_name)
    if min_controlled > 0:
        where.append("s.controlled>=?")
        values.append(min_controlled)
    query = "SELECT s.*,l.relative_path,l.partition,l.source_log,l.log_token,l.map_name," \
            "l.size_bytes,l.mtime_ns FROM scenes s JOIN logs l ON l.id=s.log_id"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY s.id LIMIT ? OFFSET ?"
    with closing(_connect(path)) as connection:
        metadata = catalog_metadata(connection)
        rows = [dict(r) for r in connection.execute(query, [*values, limit, offset])]
        for row in rows:
            if metadata["schema_version"] == 1:
                duration = metadata["parameters"]["duration"]
                row["planned_duration_seconds"] = duration
                row["horizon_plan"] = {
                    "schema_version": 1, "legacy_catalog": True, "mode": "exact",
                    "requested_duration_seconds": duration, "planned_duration_seconds": duration,
                    "source_start_frame": row["start_frame"], "source_end_frame": row["end_frame"],
                    "source_start_timestamp_us": row["start_us"], "source_end_timestamp_us": row["end_us"],
                    "scope": "legacy_fixed_duration_not_previously_audited",
                }
            else:
                row["horizon_plan"] = json.loads(row["horizon_plan"])
        return rows


def partition_files(root: Path, partition: str) -> list[Path]:
    if partition not in PARTITION_USES:
        raise ValueError(f"Unsupported source partition: {partition}")
    directory = root / "navsim_logs" / partition
    if not directory.is_dir():
        raise ValueError(f"Missing NAVSIM partition: {directory}")
    return sorted(directory.glob("*.pkl"))


def inventory(root: Path) -> dict:
    """Inspect public log directories only; never deserialize or traverse sensors."""
    if not root.is_dir():
        raise ValueError(f"Missing dataset root: {root}")
    logs = root / "navsim_logs"
    partitions = {}
    for partition, use in PARTITION_USES.items():
        present = (logs / partition).is_dir()
        files = partition_files(root, partition) if present else []
        partitions[partition] = {"present": present, "files": len(files), "use": use}
    return {
        "schema_version": 1, "dataset_root": str(root.resolve()),
        "source_format": "navsim_openscene", "source_version": "unknown",
        "partitions": partitions, "maps_present": (root / "maps").is_dir(),
        "sensor_directory_present": (root / "sensor_blobs").is_dir(),
        "ignored_log_directories": sorted(
            p.name for p in logs.iterdir() if p.is_dir() and p.name not in PARTITION_USES
        ) if logs.is_dir() else [],
        "scope": "directory_inventory_only; no completeness or sensor validation",
    }
