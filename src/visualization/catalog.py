"""Bounded, read-only projections for the stable visualization contract."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
KINDS = {"text", "number", "bool", "percent"}
SENSITIVE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|/data/|/home/|[A-Z]:[\\/]|https?://)", re.I)


def read_json(path, max_bytes=8 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("JSON resource exceeds configured size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_text(value):
    text = str(value)
    return "内部记录（未导出）" if SENSITIVE.search(text) else text[:1800]


def resolve_receipt(project, relative):
    path = (project / relative).resolve()
    if not path.is_relative_to((project / "exp").resolve()) or path.suffix != ".json":
        raise ValueError("Only registered experiment JSON summaries are accepted")
    return path


def pointer(value, dotted):
    for part in dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def scalar(value):
    if isinstance(value, str):
        return safe_text(value)
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float) and math.isfinite(value):
        return value
    raise ValueError("Only finite aggregate scalar fields may be published")


def typed_scalar(value, kind):
    value = scalar(value)
    if value is None:
        return None
    valid = {
        "bool": type(value) is bool,
        "number": type(value) in (int, float),
        "percent": type(value) in (int, float),
        "text": isinstance(value, str),
    }
    if not valid.get(kind, False) or (kind == "percent" and not 0 <= value <= 1):
        raise ValueError("Aggregate value does not match its declared display type")
    return value


def _experiment(row, project):
    target = resolve_receipt(project, row["receipt"])
    public = {key: safe_text(row[key]) for key in ("id", "title", "model", "scope")}
    public.update(
        receipt=target.name,
        fields=[],
        status="missing",
        raw_status=None,
        note=safe_text(row.get("note", "")),
    )
    if not target.is_file():
        return public
    value = read_json(target)
    public["recorded_at"] = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat()
    try:
        raw = str(pointer(value, row.get("status_path", "status")))
    except (KeyError, IndexError, TypeError):
        raw = "not_recorded"
    public["raw_status"] = safe_text(raw)
    public["status"] = row.get("status_map", {}).get(
        raw,
        raw
        if raw in {"completed", "failed", "running", "cancelled", "blocked", "pending"}
        else ("unknown" if raw == "not_recorded" else "recorded"),
    )
    for field in row.get("fields", []):
        if field["type"] not in KINDS:
            raise ValueError("Unknown display scalar type")
        try:
            v = typed_scalar(pointer(value, field["path"]), field["type"])
        except (KeyError, IndexError, TypeError):
            v = None
        public["fields"].append(
            {
                "label": safe_text(field["label"]),
                "value": v,
                "type": field["type"],
                "section": field.get("section", "experiment"),
                "unit": safe_text(field.get("unit", "")),
            }
        )
    return public


def experiment(row, project):
    resolve_receipt(project, row["receipt"])
    try:
        return _experiment(row, project)
    except (ValueError, OSError) as exc:
        return {key: safe_text(row[key]) for key in ("id", "title", "model", "scope")} | {
            "status": "invalid",
            "raw_status": type(exc).__name__,
            "fields": [],
            "note": "该回执无法安全解析，其他记录仍可显示。",
            "receipt": Path(row["receipt"]).name,
        }


def scene_index(site, configured):
    if not configured:
        return []
    site = Path(site).resolve()
    cards = read_json(site / "data/cases.json")["cases"]
    by_id = {row["id"]: row for row in cards}
    result = []
    for identifier in configured:
        card = by_id[identifier]
        # Registered bundles are explicitly published engineering/training evidence.
        data = (site / card["data"]).resolve()
        if not data.is_relative_to(site / "data") or not data.is_file():
            raise ValueError("Missing registered replay")
        result.append(
            {
                key: card[key]
                for key in (
                    "id",
                    "title",
                    "city",
                    "kind",
                    "duration_seconds",
                    "camera_count",
                    "trainable",
                    "summary",
                    "warnings",
                )
            }
        )
    return result


def whitebox_index(site, configured):
    site = Path(site).resolve()
    result = []
    for row in configured:
        path = (site / row["bundle"]).resolve()
        if not path.is_relative_to(site) or path.is_symlink():
            raise ValueError("Whitebox bundle must be under published site")
        manifest = read_json(path / "manifest.json")
        if manifest.get("kind") != "whitebox_replay":
            raise ValueError("Unsupported whitebox bundle")
        for name in manifest["streams"].values():
            stream = (path / name).resolve()
            if not stream.is_relative_to(path):
                raise ValueError("Whitebox stream escapes its registered bundle")
            seq = read_json(stream, max_bytes=128 * 1024 * 1024)
            if seq.get("source_split") not in {
                "train",
                "training",
                "navtrain",
                "trainval",
            } and not (
                seq.get("source_split") == "unassigned"
                and manifest.get("source_family") == "nuplan"
                and row.get("usage") == "engineering_only"
            ):
                raise ValueError("Only training-source whitebox data can be indexed")
        result.append(
            {
                "id": row["id"],
                "title": row["title"],
                "bundle": row["bundle"],
                "duration_seconds": manifest["timeline_end_seconds"],
                "family": manifest.get("source_family", "unknown"),
                "usage": row.get("usage"),
                "source_frames": manifest["counts"]["source_frames"],
                "branch_frames": manifest["counts"]["interaction_frames"],
                "trainable": False,
            }
        )
    return result


def harness_rows(configured, project):
    from driversi.rsi.execution.registry import ExperimentRegistry
    from driversi.rsi.workflow.store import public_tasks

    result = []
    for row in configured:
        path = (project / row["database"]).resolve()
        if not path.is_relative_to(project / "exp") or not path.is_file():
            result.append({"id": row["id"], "status": "missing", "stages": []})
            continue
        source = ExperimentRegistry(path, read_only=True)
        seen = set()
        for task in public_tasks(source, task_ids=row["tasks"]):
            if task["id"] not in row["tasks"]:
                continue
            seen.add(task["id"])
            result.append(
                {
                    "id": safe_text(task["id"]),
                    "label": safe_text(task["label"]),
                    "status": safe_text(task["status"]),
                    "elapsed_seconds": scalar(task["elapsed_seconds"]),
                    "sim_agent_steps": scalar(task["sim_agent_steps"]),
                    "needs_recovery": bool(task["needs_recovery"]),
                    "stages": [
                        {
                            "id": safe_text(s["id"]),
                            "provider": safe_text(s["provider"]),
                            "status": safe_text(s["status"]),
                            "elapsed_seconds": scalar(s.get("elapsed_seconds")),
                            "charged_agent_steps": scalar(s.get("charged_agent_steps")),
                            "cost_basis": safe_text(s.get("cost_basis") or "not_recorded"),
                        }
                        for s in task["stages"]
                    ],
                }
            )
        for task_id in row["tasks"]:
            if task_id not in seen:
                result.append({"id": safe_text(task_id), "status": "missing", "stages": []})
    return result


def snapshot(config, project=PROJECT):
    from driversi.rsi.services.catalog import describe

    from .current import current_campaign, current_config
    current = current_campaign(config, project)
    config = current_config(config)
    project = Path(project).resolve()
    site = Path(config["site_root"]).resolve()
    runs = [experiment(row, project) for row in config["experiments"]]
    from .operations import training_traces
    from .system import system_overview
    from .clustering import overview as cluster_overview

    return {
        "schema_version": 1,
        "ui_version": "1.3",
        "display_mode": config.get("display_mode", "evidence_snapshot"),
        "current": current,
        "clustering": cluster_overview(config.get("clustering"), project),
        "system": system_overview(config, project),
        "read_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "evidence_snapshot",
        "experiments": runs,
        "scenes": scene_index(site, config["scenes"]),
        "whitebox": whitebox_index(site, config.get("whitebox", [])),
        "harness": harness_rows(config.get("harness", []), project),
        "services": [
            {"name": r["name"], "group": r["group"], "readiness": r["readiness"]}
            for r in describe()
        ],
        "configs": config_guide(project),
        "legacy": config.get("legacy", []),
        "training": training_traces(config.get("training", []), project),
        "production": [experiment(row, project) for row in config.get("production", [])],
    }


def config_guide(project):
    path = project / "src/config/rsi/catalog.json"
    if not path.exists():
        return []
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "role": row["role"],
            "status": row["status"],
            "purpose": row["purpose"],
            "exists": (project / "src/config/rsi" / row["id"]).is_file(),
        }
        for row in read_json(path)["presets"]
    ]
