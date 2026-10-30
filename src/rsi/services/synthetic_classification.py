"""Supervised generated-caption stage shared by native Harness postprocessors."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.data.catalog.synthetic_classification import enqueue, library_root, process
from driversi.runtime.budget import write_json


def policy():
    return read_mapping(project_root() / "src/config/data/synthetic-scene-classification.json")


def complete(export_path, sample_report=None):
    """Return explicit status; caption failure never changes native admission."""
    queued = enqueue(export_path, sample_report)
    directory = Path(queued["job"]).parent
    root = project_root()
    command = [
        str(root / ".venv/bin/python"),
        str(root / "scripts/data/classify_synthetic_scenes.py"),
        "--job",
        queued["job"],
    ]
    # Independent base runtime contains the frozen sklearn representation; native
    # model environments and their released dependencies remain untouched.
    with (directory / "worker.log").open("a") as log:
        p = subprocess.Popen(
            [
                str(root / ".venv/bin/python"),
                str(root / "src/runtime/parent_exec.py"),
                str(os.getpid()),
                *command,
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = p.wait(timeout=600)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            code = 124
    report = directory / "report.json"
    return dict(
        id=queued["id"],
        status="completed" if code == 0 else "needs_attention",
        returncode=code,
        report=str(report) if report.exists() else None,
        training_admission_granted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", type=Path)
    group.add_argument("--export", type=Path)
    group.add_argument("--drain", action="store_true")
    parser.add_argument("--sample-report", type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.limit <= 128:
        raise ValueError("Bounded queue drain requires 1..128 jobs")
    if args.sample_report and not args.export:
        raise ValueError("Sample report requires --export")
    if args.export:
        jobs = [Path(enqueue(args.export, args.sample_report)["job"])]
    elif args.job:
        jobs = [args.job]
    else:
        jobs = []
        for p in sorted((library_root() / "jobs").glob("*/job.json")):
            report = p.parent / "report.json"
            if not report.exists():
                jobs.append(p)
        jobs = jobs[: args.limit]
    outcomes = []
    for job in jobs:
        try:
            value = process(job, policy())
        except Exception as exc:
            # Never persist provider exception text, which could contain secrets.
            value = dict(
                status="classification_failed",
                error_type=type(exc).__name__,
                training_admission_granted=False,
            )
            write_json(job.parent / "report.json", value)
        outcomes.append(value)
    print(json.dumps(outcomes, ensure_ascii=False))
    if any(v["status"] != "classified" for v in outcomes):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
