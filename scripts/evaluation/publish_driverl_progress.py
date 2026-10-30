"""Bounded result publishing for already-running DriveRL evaluation jobs.

No scheduler, model invocation, notification, scoring retry or simulation restart.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="append", type=Path)
    args = parser.parse_args()
    launches = args.launch or [
        root / "outputs/quality/driverl-multicity-v1/motion-v3-batch-launch.json",
        root / "outputs/quality/driverl-benchmark-v1/launch.json",
    ]
    jobs = {json.loads(p.read_text())["pid"]: None for p in launches}
    jobs = {pid: process_identity(pid) for pid in jobs}
    python = "/data/xinyao/navsim_workspace/conda_env/driverl_navsim/bin/python"
    deadline = time.monotonic() + 86400
    while True:
        active = [
            pid
            for pid, identity in jobs.items()
            if identity is not None and process_identity(pid) == identity
        ]
        for script in ["legacy/reports/export_driverl_multicity.py", "evaluation/export_driverl_benchmark.py", "diagnostics/export_driverl_failure_analysis.py"]:
            try:
                result = subprocess.run(
                    [python, str(root / "scripts" / script)],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(script=script, status="publication_timeout", active_jobs=active)), flush=True)
                continue
            print(
                json.dumps(
                    dict(
                        script=script,
                        exit_code=result.returncode,
                        active_jobs=active,
                        error=result.stderr[-2000:] if result.returncode else None,
                    )
                ),
                flush=True,
            )
        if not active or time.monotonic() >= deadline:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
