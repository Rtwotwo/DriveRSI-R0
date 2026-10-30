"""Refresh registered read-only projections when their source receipts change."""

import json
import time
from pathlib import Path

from .build import build
from .catalog import PROJECT


def sources(config_path):
    config = Path(config_path).resolve()
    cfg = json.loads(config.read_text())
    paths = [config, PROJECT / "src/config/rsi/catalog.json"]
    if cfg.get("architecture"):
        paths.append(PROJECT / cfg["architecture"])
    for row in cfg.get("experiments", []) + cfg.get("production", []):
        paths.append(PROJECT / row["receipt"])
    for row in cfg.get("training", []):
        for phase in row["phases"]:
            paths.extend(PROJECT / phase[key] for key in ("receipt", "log", "config"))
    for row in cfg.get("harness", []):
        path = PROJECT / row["database"]
        paths.extend([path, Path(str(path) + "-wal")])
    return paths


def fingerprint(paths):
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            result.append((str(path), None, None))
    return tuple(result)


def watch(config_path, interval=30):
    if not 5 <= interval <= 3600:
        raise ValueError("Refresh interval must be 5..3600 seconds")
    previous = None
    while True:
        try:
            current = fingerprint(sources(config_path))
            if current != previous:
                print(json.dumps(build(config_path), ensure_ascii=False), flush=True)
                previous = current
        except (ValueError, OSError) as exc:
            print(
                json.dumps(
                    {
                        "status": "refresh_failed",
                        "error_type": type(exc).__name__,
                        "previous_snapshot_retained": True,
                    }
                ),
                flush=True,
            )
        time.sleep(interval)
