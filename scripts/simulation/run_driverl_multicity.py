"""Bounded execution of a fixed corpus. No training or success-only selection."""

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--source-name", default="prepared-final")
    parser.add_argument("--run-name", default="final")
    a = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = root / "outputs/demonstrations/driverl-multicity-v1"
    quality = root / "outputs/quality/driverl-multicity-v1"
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "agents/driverl/native/DriveRL/src")
        + ":"
        + str(root / "agents/driverl/native/DriveRL/nuplan-devkit"),
        DRIVERSI_NATIVE_PATH=str(
            root / ".venvs/driverl-native-py311-v2/_native.cpython-311-x86_64-linux-gnu.so"
        ),
        OMP_NUM_THREADS="4",
    )
    python = "/data/xinyao/navsim_workspace/conda_env/driverl_navsim/bin/python"

    def worker(slot, logs):
        for log in logs:
            parent = base / log
            source = parent / (
                a.source_name
                if a.final
                else "prepared-v2"
                if (parent / "prepared-v2/scene.npz").exists()
                else "prepared"
            )
            jobs = (
                [("simulation/verify_driverl_closed_loop.py", a.run_name)]
                if a.final
                else [
                    ("simulation/verify_driverl_closed_loop.py", "closed-loop"),
                    ("simulation/verify_driverl_official.py", "official"),
                ]
            )
            for script, folder in jobs:
                out = parent / folder
                if out.exists():
                    raise FileExistsError(out)
                logfile = quality / (log + "-" + folder + ".log")
                args = [
                    python,
                    str(root / "scripts" / script),
                    "--source",
                    str(source),
                    "--output",
                    str(out),
                    "--device",
                    f"cuda:{slot}",
                ]
                try:
                    with logfile.open("w") as f:
                        run = subprocess.run(
                            args,
                            cwd=root,
                            env=env,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            timeout=3600,
                        )
                    status = run.returncode
                    result = dict(
                        exit_code=status,
                        log=str(logfile),
                        source=str(source),
                        command=args,
                        trainable=False,
                    )
                except subprocess.TimeoutExpired:
                    result = dict(
                        exit_code=None, error="3600-second limit", log=str(logfile), trainable=False
                    )
                out.mkdir(parents=True, exist_ok=True)
                (out / "job-result.json").write_text(json.dumps(result, indent=2))
                print(log, folder, json.dumps(result), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(worker, g, logs)
            for g, logs in (
                [
                    (0, ["log-02", "log-01"]),
                    (1, ["log-04", "log-03"]),
                    (3, ["log-07", "log-06"]),
                    (7, ["log-08", "log-05"]),
                ]
                if a.final
                else [
                    (2, ["log-01", "log-06"]),
                    (3, ["log-02", "log-07"]),
                    (4, ["log-04", "log-08"]),
                    (5, ["log-05"]),
                ]
            )
        ]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
