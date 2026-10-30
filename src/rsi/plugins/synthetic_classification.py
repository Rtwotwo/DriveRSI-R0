"""Harness provider for same-branch caption and frozen scene-cluster assignment."""

from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.data.catalog.synthetic_classification import enqueue, process
from driversi.rsi.contracts import _keys
from driversi.rsi.services.synthetic_classification import policy
from driversi.rsi.workflow.results import result


def prepare(config, base, registry):
    _keys(config, ("export", "sample_report"), "synthetic scene classification")
    path = (base / config["export"]).resolve(strict=True)
    sample = (
        (base / config["sample_report"]).resolve(strict=True) if config["sample_report"] else None
    )
    job = enqueue(path, sample)
    return dict(job=job["job"], sha256=digest(job["job"]), policy=policy()), 0


def execute(config, inputs, output, registry):
    if digest(config["job"]) != config["sha256"]:
        raise ValueError("Prepared synthetic job changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = process(config["job"], config["policy"])
    return result(
        output,
        report,
        status="completed" if report["status"] == "classified" else "failed",
        reason=None if report["status"] == "classified" else report["status"],
    )


def register(registry, Provider):
    registry.register(
        Provider(
            "synthetic_scene_classification",
            "1",
            (),
            "synthetic_scene_cluster_record",
            prepare,
            execute,
            ("report",),
        )
    )
