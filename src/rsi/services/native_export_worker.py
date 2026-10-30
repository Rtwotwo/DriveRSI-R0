"""Bounded resident native export interpreter; original export validation is retained."""

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    import types

    _root = Path(__file__).resolve().parents[3]
    for _name, _folder in (("driversi", "src"), ("driversi_agents", "agents")):
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_root / _folder)]
        sys.modules[_name] = _module

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


class NativeExportSession:
    """One client, fixed interpreter/GPU, no retry of failed or unsettled jobs."""

    def __init__(self, python, gpu, directory, *, max_jobs=64):
        if type(gpu) is not int or not 0 <= gpu <= 7 or not 1 <= max_jobs <= 64:
            raise ValueError("Explicit GPU and bounded export job count required")
        self.python, self.gpu = str(Path(python).absolute()), gpu
        self.directory, self.max_jobs = Path(directory), max_jobs
        self.process = self.log = None
        self.count = 0

    def __call__(self, action, arguments, directory, *, gpu, timeout=None, python=None):
        from driversi.rsi.experiments.reactive_postprocess import run_native

        timeout = (1800 if action == "chainflow_export" else 3600) if timeout is None else timeout
        if action != "chainflow_export":
            return run_native(action, arguments, directory, gpu=gpu, timeout=timeout, python=python)
        if str(Path(python).absolute()) != self.python or gpu != self.gpu:
            raise ValueError("Resident native interpreter/GPU changed")
        if not 1 <= timeout <= 1800:
            raise ValueError("Bounded export deadline required")
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        job = directory / "job.json"
        frozen_json(job, dict(action=action, arguments=arguments))
        sha = digest(job)
        if (directory / "exit.json").exists():
            saved = read_mapping(directory / "exit.json")
            if saved["job_sha256"] != sha or saved["returncode"] != 0:
                raise ValueError("Prior native export changed or failed")
            if saved.get("result_sha256") != digest(directory / "result.json"):
                raise ValueError("Prior export result changed")
            return
        if (directory / "launch.json").exists():
            raise ValueError("Unsettled export; no automatic retry")
        if self.count >= self.max_jobs:
            raise ValueError("Resident export job budget exhausted")
        if self.process is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.log = (self.directory / "service.log").open("w")
            self.process = subprocess.Popen(
                [
                    self.python,
                    str(Path(__file__).resolve()),
                    "--serve",
                    "--max-jobs",
                    str(self.max_jobs),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                cwd=project_root(),
                stdin=subprocess.PIPE,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"},
            )
        elif self.process.poll() is not None:
            raise ValueError("Resident export exited; no implicit restart")
        write_json(
            directory / "launch.json",
            dict(
                pid=self.process.pid,
                job_sha256=sha,
                resident=True,
                sequence=self.count,
                started=time.time(),
            ),
        )
        self.count += 1
        self.process.stdin.write(json.dumps(dict(job=str(job), sha256=sha, timeout=timeout)) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout + 10
        while not (directory / "exit.json").exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Native export exited without settlement; preserve unknown outcome"
                )
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError("Native export exceeded deadline; no retry")
            time.sleep(0.05)
        saved = read_mapping(directory / "exit.json")
        if saved["job_sha256"] != sha or saved["returncode"] != 0:
            raise RuntimeError("Native export failed; see its worker.log")
        if saved["result_sha256"] != digest(directory / "result.json"):
            raise ValueError("Native export response changed")

    def close(self):
        if self.process is not None:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        if self.log:
            self.log.close()


def serve(max_jobs, parent):
    import contextlib
    import select

    from driversi.rsi.experiments.reactive_postprocess import execute_job

    if (
        os.getppid() != parent
        or ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0
        or os.getppid() != parent
    ):
        raise RuntimeError("Cannot bind native worker lifetime")
    end = time.monotonic() + 14400
    for index in range(max_jobs):
        if time.monotonic() >= end or not select.select([sys.stdin], [], [], 600)[0]:
            break
        line = sys.stdin.readline()
        if not line:
            break
        request = json.loads(line)
        job_path = Path(request["job"]).resolve(strict=True)
        job = read_mapping(job_path)
        if digest(job_path) != request["sha256"] or job["action"] != "chainflow_export":
            raise ValueError("Worker accepts sealed native ChainFlow export only")
        if not job_path.is_relative_to(project_root() / "exp"):
            raise ValueError("Worker job must be under exp")
        started = time.time()
        code, result_sha = 1, None

        def deadline(_signum, _frame):
            raise TimeoutError("Resident native export deadline")

        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(min(int(request["timeout"]), 1800))
        with (job_path.parent / "worker.log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    result = execute_job(job)
                    if digest(job_path) != request["sha256"]:
                        raise ValueError("Native job changed while running")
                    write_json(job_path.parent / "result.json", result)
                    result_sha = digest(job_path.parent / "result.json")
                    code = 0
                except BaseException:
                    traceback.print_exc()
                finally:
                    signal.alarm(0)
        write_json(
            job_path.parent / "exit.json",
            dict(
                returncode=code,
                job_sha256=request["sha256"],
                result_sha256=result_sha,
                resident=True,
                sequence=index,
                started=started,
                finished=time.time(),
            ),
        )
        if code:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--max-jobs", type=int, default=64)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 64:
        raise ValueError("At most64 export jobs per worker")
    serve(args.max_jobs, args.parent_pid)
