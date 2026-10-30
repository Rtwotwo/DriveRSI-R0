"""Materialize selected catalogue windows as reusable, provenance-carrying scenes."""

import json
import multiprocessing as mp
import sqlite3
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path

from driversi.data.scenes.catalog import (
    DATA_ERRORS,
    PARTITION_SPLITS,
    PARTITION_USES,
    catalog_summary,
    list_scenes,
)
from driversi.data.road_network.maps import attach_nuplan_map
from driversi.data.sources.navsim import SIM_DT_SECONDS, compile_frames, load_log, plan_horizon


def _source_path(root: Path, row: dict) -> Path:
    path = root / row["relative_path"]
    expected = Path("navsim_logs") / row["partition"]
    if Path(row["relative_path"]).parent != expected:
        raise ValueError("Catalogue source path does not match its public partition")
    return path


def _check_stat(path: Path, row: dict) -> None:
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
        raise ValueError("Source file changed since indexing; build a new catalogue")


def _verify_window(frames: list[dict], row: dict) -> None:
    start, end = row["start_frame"], row["end_frame"]
    if not 0 <= start < end < len(frames):
        raise ValueError("Indexed source window is no longer available")
    if (str(frames[start]["token"]) != row["token"]
            or int(frames[start]["timestamp"]) != row["start_us"]
            or int(frames[end]["timestamp"]) != row["end_us"]):
        raise ValueError("Indexed token or timestamps do not match source")
    for frame in frames[start:end + 1]:
        if str(frame["log_name"]) != row["source_log"] or str(frame["map_location"]) != row["map_name"]:
            raise ValueError("Indexed log or map identity does not match source")
        if (row["log_token"] is not None and frame.get("log_token") is not None
                and str(frame["log_token"]) != row["log_token"]):
            raise ValueError("Original log token does not match source")


def _verify_horizon(frames: list[dict], row: dict) -> dict:
    frozen = row["horizon_plan"]
    legacy = frozen.get("legacy_catalog", False)
    plan = plan_horizon(
        frames, start=row["start_frame"], duration=frozen["requested_duration_seconds"],
        mode=frozen["mode"], min_duration=frozen.get("min_duration_seconds", SIM_DT_SECONDS),
    )
    if (plan["status"] != "supported"
            or plan["planned_duration_seconds"] != row["planned_duration_seconds"]
            or plan["source_end_frame"] != row["end_frame"]
            or plan["source_end_timestamp_us"] != row["end_us"]):
        raise ValueError("Indexed horizon no longer matches the source window")
    if not legacy and plan != frozen:
        raise ValueError("Indexed horizon plan does not match the current source support")
    if legacy:
        plan["provenance"] = "revalidated_legacy_catalog_v1_exact_duration"
    return plan


def _write_manifest(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _compile_group(selected, root, output, metadata, catalog, trusted, maps_root):
    """One source log per task; preserve source-read caching within its windows."""
    parameters = metadata["parameters"]
    cached_path, frames = None, None
    records = []
    for row in selected:
        record = {k: row[k] for k in ("id", "relative_path", "partition", "source_log", "log_token",
                                      "token", "start_us", "end_us", "map_name")}
        record["use"] = PARTITION_USES.get(row["partition"], "unassigned")
        record["planned_duration_seconds"] = row["planned_duration_seconds"]
        record["horizon_plan"] = row["horizon_plan"]
        try:
            if metadata["source_format"] == "nuplan_db":
                from driversi.data.sources.nuplan_catalog import materialize_window
                scene, record["use"] = materialize_window(root, row, metadata)
                record["split"] = scene.metadata["split"]
                record["source_group"] = scene.metadata["source_group"]
                plan = scene.metadata["horizon_plan"]
            else:
                source = _source_path(root, row)
                _check_stat(source, row)
                if source != cached_path:
                    frames = load_log(source, trusted=trusted)
                    _check_stat(source, row)
                    cached_path = source
                _verify_window(frames, row)
                plan = _verify_horizon(frames, row)
                scene = compile_frames(frames, start=row["start_frame"], duration=row["planned_duration_seconds"],
                                       split=PARTITION_SPLITS[row["partition"]],
                                       max_actors=parameters["max_actors"])
            scene.metadata["horizon_plan"] = plan
            record["horizon_plan"] = plan
            scene.metadata.update({
                "source_dataset": metadata["source_format"], "source_version": metadata["source_version"],
                "source_partition": row["partition"], "intended_use": record["use"],
                "official_scene_filter": None, "split_verification": scene.metadata.get("split_verification", "catalog_source_partition_only"),
                "source_log_token": row["log_token"], "source_interval_us": [row["start_us"], row["end_us"]],
                "source_file": row["relative_path"], "source_size_bytes": row["size_bytes"],
                "source_mtime_ns": row["mtime_ns"], "catalog_scene_id": row["id"],
                "catalog_path": str(catalog.resolve()),
            })
            attach_nuplan_map(scene, maps_root)
            filename = f"scene-{row['id']:06d}.npz"
            scene.save(output / filename)
            record.update(status="compiled", scene=filename, actors=len(scene.actor_ids),
                          controlled=int(scene.controlled.sum()), map_version=scene.metadata["map_version"])
        except (*DATA_ERRORS, sqlite3.Error) as error:
            record.update(status="failed", reason=str(error))
        records.append(record)
    return records


def _group_results(groups, kwargs, workers):
    if workers == 0:
        for group in groups:
            yield _compile_group(group, **kwargs)
        return
    # Bounded submission, ordered results, distinct scene filenames, one manifest writer.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        iterator = iter(groups)
        pending = deque()
        for _ in range(workers):
            group = next(iterator, None)
            if group is not None:
                pending.append(pool.submit(_compile_group, group, **kwargs))
        while pending:
            yield pending.popleft().result()
            group = next(iterator, None)
            if group is not None:
                pending.append(pool.submit(_compile_group, group, **kwargs))


def compile_catalog(catalog: Path, output: Path, *, dataset_root: Path | None = None,
                    limit: int = 8, offset: int = 0, map_name: str | None = None,
                    min_controlled: int = 1, trusted: bool = False, workers: int = 0,
                    maps_root: Path | None = None) -> dict:
    """Keep each failure in a manifest. Existing output directories are never replaced."""
    if type(workers) is not int or not 0 <= workers <= 64:
        raise ValueError("Compilation workers must be an integer between 0 and 64")
    if not Path(catalog).exists() and not trusted:
        raise ValueError("Compilation requires --trusted-pickle for trusted source logs or an existing nuPlan catalogue")
    metadata = catalog_summary(catalog)
    if metadata["source_format"] not in {"navsim_openscene", "nuplan_db"}:
        raise ValueError("Unsupported catalogue source format")
    if metadata["source_format"] == "navsim_openscene" and not trusted:
        raise ValueError("Compilation requires --trusted-pickle for trusted source logs")
    root = (dataset_root or Path(metadata["dataset_root"])).resolve()
    maps_root = Path(maps_root).resolve() if maps_root is not None else root / "maps"
    selected = list_scenes(catalog, limit=limit, offset=offset, map_name=map_name,
                           min_controlled=min_controlled)
    if not selected:
        raise ValueError("No geometry_pending scenes match the selection")
    output.mkdir(parents=True, exist_ok=False)
    parameters = metadata["parameters"]
    manifest = {
        "schema_version": 2, "kind": "compiled_scene_library", "state": "building",
        "catalog_schema_version": metadata["schema_version"],
        "catalog": str(catalog.resolve()), "dataset_root": str(root), "maps_root": str(maps_root),
        "source_format": metadata["source_format"], "source_version": metadata["source_version"],
        "official_scene_filter": metadata["official_scene_filter"],
        "catalog_partitions": metadata["partitions"], "parameters": parameters,
        "selection": {"limit": limit, "offset": offset, "map_name": map_name,
                      "min_controlled": min_controlled},
        "selected": len(selected), "compiled": 0, "failed": 0, "workers": workers,
        "validation_scope": "source window and map compilation; driving semantics not certified",
        "records": [],
    }
    manifest_path = output / "manifest.json"
    _write_manifest(manifest_path, manifest)
    groups = [list(rows) for _source, rows in groupby(selected, key=lambda row: row["relative_path"])]
    kwargs = dict(root=root, output=output, metadata=metadata, catalog=catalog, trusted=trusted,
                  maps_root=maps_root)
    try:
        for records in _group_results(groups, kwargs, workers):
            manifest["records"].extend(records)
            manifest["compiled"] += sum(row["status"] == "compiled" for row in records)
            manifest["failed"] += sum(row["status"] == "failed" for row in records)
            _write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest.update(state="failed", infrastructure_error=f"{type(error).__name__}: {error}")
        _write_manifest(manifest_path, manifest)
        raise
    if metadata["source_format"] == "nuplan_db":
        for role in ("train", "development", "heldout", "unassigned"):
            paths = [str((output / row["scene"]).resolve()) for row in manifest["records"]
                     if row["status"] == "compiled" and row["split"] == role]
            (output / (role + ".txt")).write_text("".join(path + "\n" for path in paths), encoding="utf-8")
        manifest["list_scope"] = "compiled paths only; not simulator or driving quality admission"
    manifest["state"] = "complete"
    _write_manifest(manifest_path, manifest)
    return {k: manifest[k] for k in ("selected", "compiled", "failed", "state")} | {
        "manifest": str(manifest_path),
    }
