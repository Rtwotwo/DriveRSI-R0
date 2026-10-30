"""nuPlan source discovery in the platform's existing scene catalogue schema."""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from driversi.data.scenes.catalog import CATALOG_VERSION, SCHEMA, _json, catalog_summary
from driversi.data.sources.navsim import plan_horizon, validate_horizon_request
from driversi.data.sources.nuplan_db import compile_window, log_header, open_db, read_window

ROLES = {"train", "development", "heldout", "unassigned"}
PROTECTED_PARTITIONS = {"val", "test"}


def source_group(name):
    name = Path(name).name
    for suffix in (".db", ".pkl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"_\d+_\d+$", "", name)


def source_files(root):
    base = Path(root).resolve() / "nuplan-v1.1" / "splits"
    if not base.is_dir():
        raise ValueError("Expected nuplan-v1.1/splits beneath dataset root")
    result = {p.name: sorted(p.glob("*.db")) for p in sorted(base.iterdir()) if p.is_dir()}
    for partition, files in result.items():
        if any(
            p.resolve().parent != (base / partition).resolve()
            or not p.resolve().is_relative_to(base)
            for p in files
        ):
            raise ValueError("nuPlan DB symlink escapes its source partition")
    return result


def inventory(root):
    files = source_files(root)
    return {
        "schema_version": 1,
        "source_format": "nuplan_db",
        "source_version": "v1.1",
        "dataset_root": str(Path(root).resolve()),
        "partitions": {
            p: {
                "files": len(paths),
                "bytes": sum(f.stat().st_size for f in paths),
                "default_use": "heldout" if p in PROTECTED_PARTITIONS else "unassigned",
            }
            for p, paths in files.items()
        },
        "maps_present": (Path(root) / "maps/nuplan-maps-v1.0.json").is_file(),
        "scope": "filename inventory only; not full DB integrity, official split or sensor validation",
    }


def split_rules(files, policy=None, protected_logs=()):
    roles, protected, evidence = {}, set(), []
    for partition in PROTECTED_PARTITIONS:
        protected.update(source_group(p.name) for p in files.get(partition, []))
    for path in protected_logs:
        path = Path(path)
        data = path.read_bytes()
        names = json.loads(data)
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or not n for n in names)
        ):
            raise ValueError("Protected logs must be a nonempty JSON list of source names")
        protected.update(source_group(n) for n in names)
        evidence.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
    if policy is not None:
        data = Path(policy).read_bytes()
        value = json.loads(data)
        if value.get("schema_version") != 1 or not isinstance(value.get("source_roles"), dict):
            raise ValueError("Split policy requires schema_version=1 and source_roles")
        for name, role in value["source_roles"].items():
            group = source_group(name)
            if not group or role not in ROLES or (group in roles and roles[group] != role):
                raise ValueError("Invalid or conflicting original-log split roles")
            roles[group] = role
        if any(role in {"train", "development"} for role in roles.values()) and not evidence:
            raise ValueError(
                "Training/development requires an explicit external protected-log registry"
            )
        if any(roles.get(group) in {"train", "development"} for group in protected):
            raise ValueError("Protected source group cannot enter training/development")
        evidence.append(
            {"path": str(Path(policy).resolve()), "sha256": hashlib.sha256(data).hexdigest()}
        )
    return {
        "source_roles": roles,
        "protected_source_groups": sorted(protected),
        "evidence": evidence,
        "scope": "whole original drive isolation across inventoried val/test and supplied external registries; no complete official split claim",
    }


def role_for(source, partition, rules):
    group = source_group(source)
    if partition in PROTECTED_PARTITIONS or group in rules["protected_source_groups"]:
        return "heldout"
    return rules["source_roles"].get(group, "unassigned")


def build_catalog(
    root,
    output,
    *,
    partitions,
    max_logs=1,
    windows_per_log=2,
    duration=9.0,
    history_seconds=1.5,
    max_actors=32,
    split_policy=None,
    protected_logs=(),
):
    validate_horizon_request(duration, "exact", 0.1)
    if duration > 9 or not 0 <= history_seconds <= 5 or not np.isfinite(history_seconds):
        raise ValueError("nuPlan catalogue needs duration <=9s and bounded history")
    if (
        any(type(v) is not int or v < 1 for v in (max_logs, windows_per_log, max_actors))
        or max_actors > 32
    ):
        raise ValueError("Require positive indexing limits, with <=32 policy slots")
    root, output = Path(root).resolve(), Path(output)
    files = source_files(root)
    if (
        not partitions
        or len(set(partitions)) != len(partitions)
        or any(not files.get(p) for p in partitions)
    ):
        raise ValueError("Require distinct nonempty nuPlan source partitions")
    rules = split_rules(files, split_policy, protected_logs)
    meta = {
        "schema_version": CATALOG_VERSION,
        "state": "building",
        "source_format": "nuplan_db",
        "source_version": "v1.1",
        "dataset_root": str(root),
        "official_scene_filter": None,
        "split_rules": rules,
        "parameters": {
            "duration": duration,
            "duration_mode": "exact",
            "min_duration": 0.1,
            "history_seconds": history_seconds,
            "max_actors": max_actors,
            "max_logs_per_partition": max_logs,
            "windows_per_log": windows_per_log,
        },
        "partitions": {
            p: {
                "available_files": len(files[p]),
                "selected_files": min(max_logs, len(files[p])),
                "use": "per_original_log_policy",
            }
            for p in partitions
        },
        "selection": "sorted files, evenly spaced anchors with measured history and full future; not official scenario filtering",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(output)) as c:
            c.executescript(SCHEMA)
            c.executemany(
                "INSERT INTO metadata VALUES(?,?)", [(k, _json(v)) for k, v in meta.items()]
            )
            for part in partitions:
                for path in files[part][:max_logs]:
                    stat = path.stat()
                    ident = c.execute(
                        "INSERT INTO logs(relative_path,partition,size_bytes,mtime_ns,status) VALUES(?,?,?,?,?)",
                        (
                            path.relative_to(root).as_posix(),
                            part,
                            stat.st_size,
                            stat.st_mtime_ns,
                            "reading",
                        ),
                    ).lastrowid
                    try:
                        with open_db(path) as db:
                            h = log_header(db)
                            if h["source_log"] != path.stem:
                                raise ValueError(
                                    "DB logfile differs from filename; cannot trust protected-group inventory"
                                )
                            streams = db.execute(
                                "SELECT DISTINCT lidar_token FROM lidar_pc"
                            ).fetchall()
                            if len(streams) != 1:
                                raise ValueError("Expected a single lidar timestamp stream")
                            ts = db.execute(
                                "SELECT token,timestamp FROM lidar_pc ORDER BY timestamp LIMIT 200001"
                            ).fetchall()
                        if not 2 <= len(ts) <= 200000:
                            raise ValueError("DB frame count outside bounded indexing capacity")
                        times = np.asarray([row["timestamp"] for row in ts], dtype=np.int64)
                        if np.any(np.diff(times) <= 0):
                            raise ValueError("Non-monotonic source timestamps")
                        c.execute(
                            "UPDATE logs SET source_log=?,log_token=?,map_name=?,first_us=?,last_us=?,frames=?,status='indexed' WHERE id=?",
                            (
                                h["source_log"],
                                h["log_token"],
                                h["map_name"],
                                int(times[0]),
                                int(times[-1]),
                                len(ts),
                                ident,
                            ),
                        )
                        eligible = np.flatnonzero(
                            (times >= times[0] + round(history_seconds * 1e6))
                            & (times <= times[-1] - round(duration * 1e6))
                        )
                        if not len(eligible):
                            raise ValueError("Log too short for requested history plus future")
                        indices = eligible[
                            np.linspace(
                                0, len(eligible) - 1, min(windows_per_log, len(eligible)), dtype=int
                            )
                        ]
                        for index in indices:
                            token = ts[index]["token"].hex()
                            start = int(times[index])
                            end_index = int(index)
                            end = start
                            status, reason, actors, controlled, plan = (
                                "rejected",
                                None,
                                None,
                                None,
                                {},
                            )
                            try:
                                w = read_window(
                                    path,
                                    token=token,
                                    duration=duration,
                                    history_seconds=history_seconds,
                                )
                                plan = plan_horizon(
                                    w["frames"], start=0, duration=duration, mode="exact"
                                )
                                if plan["source_end_frame"] is not None:
                                    end_index = int(index) + plan["source_end_frame"]
                                    end = plan["source_end_timestamp_us"]
                                scene = compile_window(
                                    w,
                                    duration=duration,
                                    split=role_for(h["source_log"], part, rules),
                                    duration_mode="exact",
                                    max_actors=max_actors,
                                )
                                actors, controlled = (
                                    len(scene.actor_ids),
                                    int(scene.controlled.sum()),
                                )
                                status = "geometry_pending"
                            except (
                                ValueError,
                                KeyError,
                                TypeError,
                                OSError,
                                sqlite3.Error,
                            ) as error:
                                reason = str(error)
                            c.execute(
                                "INSERT INTO scenes(log_id,start_frame,end_frame,token,start_us,end_us,status,reason,actors,controlled,exclusions,planned_duration_seconds,horizon_plan) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (
                                    ident,
                                    int(index),
                                    end_index,
                                    token,
                                    start,
                                    end,
                                    status,
                                    reason,
                                    actors,
                                    controlled,
                                    "{}",
                                    plan.get("planned_duration_seconds"),
                                    _json(plan),
                                ),
                            )
                        after = path.stat()
                        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                            raise ValueError("Source changed during indexing")
                    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
                        c.execute("DELETE FROM scenes WHERE log_id=?", (ident,))
                        c.execute(
                            "UPDATE logs SET status='rejected',reason=? WHERE id=?",
                            (str(error), ident),
                        )
            c.execute("UPDATE metadata SET value=? WHERE key='state'", (_json("complete"),))
            c.commit()
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return catalog_summary(output)


def materialize_window(root, row, metadata):
    relative = Path(row["relative_path"])
    expected = Path("nuplan-v1.1/splits") / row["partition"]
    path = (Path(root) / relative).resolve(strict=True)
    if relative.parent != expected or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Catalogue DB escapes its source partition")
    p = metadata["parameters"]
    w = read_window(
        path, token=row["token"], duration=p["duration"], history_seconds=p["history_seconds"]
    )
    if (
        any(w[k] != row[k] for k in ("source_log", "log_token", "map_name"))
        or w["anchor_timestamp_us"] != row["start_us"]
    ):
        raise ValueError("Indexed DB identity changed")
    if w["source_log"] != path.stem:
        raise ValueError("Indexed DB filename/log identity differs")
    if (w["source_stat"]["size_bytes"], w["source_stat"]["mtime_ns"]) != (
        row["size_bytes"],
        row["mtime_ns"],
    ):
        raise ValueError("Source file changed since indexing")
    role = role_for(w["source_log"], row["partition"], metadata["split_rules"])
    scene = compile_window(
        w, duration=p["duration"], split=role, duration_mode="exact", max_actors=p["max_actors"]
    )
    if scene.metadata["horizon_plan"] != row["horizon_plan"]:
        raise ValueError("Indexed DB horizon changed")
    scene.metadata["split_verification"] = "original_log_policy_and_protected_registry"
    scene.metadata["source_group"] = source_group(w["source_log"])
    scene.metadata["split_policy_evidence"] = metadata["split_rules"]["evidence"]
    return scene, role
