"""Read-only native progress adapters; never publish log bodies or sample content."""

import math
import re

import yaml

from .catalog import read_json, safe_text, scalar

MAX_BYTES = 16 * 1024 * 1024
PARAMETERS = (
    "learning_rate",
    "min_lr",
    "warmup_steps",
    "gradient_accumulation_steps",
    "per_device_train_batch_size",
    "max_steps",
    "seed",
    "weight_decay",
)
PROGRESS = re.compile(
    r"Epoch (\d+)/(\d+):.*?\|\s*(\d+)/(\d+)\s*\[.*?diff=([\d.eE+\-]+), lm=([\d.eE+\-]+), loss=([\d.eE+\-]+), lr=([\d.eE+\-]+)"
)


def resource(project, relative):
    path = (project / relative).resolve()
    if not path.is_relative_to((project / "exp").resolve()):
        raise ValueError("Progress source must stay under exp")
    return path


def parse_augustus_progress(text, phase):
    latest = {}
    for line in re.split(r"[\r\n]", text):
        match = PROGRESS.search(line)
        if not match:
            continue
        epoch, epochs, batch, batches = map(int, match.groups()[:4])
        diff, lm, loss, lr = map(float, match.groups()[4:])
        if not all(math.isfinite(x) for x in (diff, lm, loss, lr)):
            continue
        latest[(epoch, batch)] = dict(
            phase=phase,
            epoch=epoch,
            batch=batch,
            batches=batches,
            loss=loss,
            diffusion_loss=diff,
            language_loss=lm,
            learning_rate=lr,
        )
    return [latest[key] for key in sorted(latest)]


def training_traces(configured, project):
    result = []
    for row in configured:
        item = {k: safe_text(row[k]) for k in ("id", "title", "model", "scope")}
        item.update(
            status="missing",
            phases=[],
            points=[],
            parameters=[],
            note="原生日志最后一次epoch/batch记录；横轴为记录序号，不表示时间或batch间距，也不是优化器步数。日志数值保留原始舍入。",
        )
        for phase in row["phases"]:
            receipt, log, config = [
                resource(project, phase[k]) for k in ("receipt", "log", "config")
            ]
            entry = dict(
                id=phase["id"],
                status="missing",
                updates=None,
                total_updates=None,
                wall_seconds=None,
            )
            try:
                if receipt.exists():
                    v = read_json(receipt)
                    entry.update(
                        status=safe_text(v.get("status", "unknown")),
                        updates=scalar(v.get("updates")),
                        total_updates=scalar(v.get("total_updates")),
                        wall_seconds=scalar(v.get("consumed", {}).get("wall_seconds")),
                    )
                if log.exists():
                    if log.stat().st_size > MAX_BYTES:
                        raise ValueError("Native log exceeds cap; use scalar JSONL telemetry")
                    item["points"].extend(
                        parse_augustus_progress(log.read_text(errors="replace"), phase["id"])
                    )
                if config.exists():
                    if config.stat().st_size > MAX_BYTES:
                        raise ValueError("Native configuration exceeds cap")
                    training = yaml.safe_load(config.read_text()).get("training", {})
                    item["parameters"].append(
                        {
                            "phase": phase["id"],
                            "values": {key: scalar(training.get(key)) for key in PARAMETERS},
                        }
                    )
            except (ValueError, OSError, yaml.YAMLError):
                entry["status"] = "invalid"
            item["phases"].append(entry)
        statuses = [x["status"] for x in item["phases"]]
        item["status"] = (
            "completed" if statuses and all(x == "completed" for x in statuses) else "incomplete"
        )
        item["planned_updates"] = row.get("planned_updates")
        item["observed_updates"] = item["phases"][-1]["total_updates"] if item["phases"] else None
        result.append(item)
    return result
