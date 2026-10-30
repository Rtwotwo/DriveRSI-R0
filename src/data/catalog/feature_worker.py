"""Private, bounded file-queue worker for native generated-history features.

Activation is explicit: DATABASE.feature-worker.json. Unknown jobs are never
re-executed. The existing caller still owns its launch/exit/result receipts.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    import types

    _root = Path(__file__).resolve().parents[3]
    for _name, _folder in (("driversi", "src"), ("driversi_agents", "agents")):
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_root / _folder)]
        sys.modules[_name] = _module

from driversi.agents.backends.native import digest, signature
from driversi.runtime.budget import write_json


def descriptor(database):
    return Path(str(Path(database).resolve()) + ".feature-worker.json")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("kind") != "resident_generated_features_v1":
        raise ValueError("Unknown feature worker descriptor")
    for name, minimum, maximum in (
        ("gpu", 0, 7),
        ("idle_seconds", 1, 3600),
        ("max_seconds", 1, 14400),
        ("max_jobs", 1, 256),
        ("job_seconds", 1, 1800),
    ):
        if type(config.get(name)) is not int or not minimum <= config[name] <= maximum:
            raise ValueError("Invalid worker limit: " + name)
    for name in ("database", "source_catalog", "directory"):
        if not Path(config[name]).is_absolute():
            raise ValueError("Absolute worker paths required")
    if not isinstance(config.get("profile_id"), str) or not config["profile_id"]:
        raise ValueError("Explicit feature profile required")
    return config


def validate_job(job, config):
    if job.get("action") != "features":
        raise ValueError("Resident worker accepts feature jobs only")
    args = job["arguments"]
    if set(args) != {
        "export_path",
        "source_catalog",
        "profile_id",
        "database",
        "output",
        "sample_report",
    }:
        raise ValueError("Explicit admitted feature job required")
    for key in ("database", "source_catalog"):
        if Path(args[key]).resolve() != Path(config[key]).resolve():
            raise ValueError("Feature worker source/database mismatch")
    if args["profile_id"] != config["profile_id"]:
        raise ValueError("Feature worker profile mismatch")
    if not args["sample_report"]:
        raise ValueError("Native admitted sample required")


def is_running(directory):
    with (directory / "service.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def submit(job_path, config_path):
    config = load_config(config_path)
    job_path = Path(job_path).resolve(strict=True)
    job = json.loads(job_path.read_text())
    validate_job(job, config)
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    queue = directory / "jobs"
    queue.mkdir(exist_ok=True)
    key = signature(dict(path=str(job_path), sha256=digest(job_path)))
    request = queue / (key + ".request.json")
    result = queue / (key + ".result.json")
    running = queue / (key + ".running.json")
    config_sha = digest(config_path)
    started = time.time()
    deadline = started + config["job_seconds"]
    with (directory / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if result.exists():
            raise ValueError(
                "Completed resident request: reuse the native exit receipt; no RPC replay"
            )
        if running.exists():
            raise ValueError("Feature worker exited during job; unknown outcome, no retry")
        if not request.exists():
            write_json(
                request,
                dict(
                    job=str(job_path),
                    job_sha256=digest(job_path),
                    config_sha256=config_sha,
                    submitted_at=started,
                    deadline=deadline,
                ),
            )
        start_worker(directory, config_path, config_sha)
    while not result.exists():
        if time.time() >= deadline:
            write_json(queue / (key + ".cancel.json"), dict(reason="client_deadline"))
            raise TimeoutError("Feature worker job deadline; no automatic retry")
        if not is_running(directory):
            if result.exists():
                continue
            if running.exists():
                raise ValueError("Feature worker exited during job; unknown outcome, no retry")
            # Rollover is safe only for a request never marked as started.
            with (directory / "launch.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                start_worker(directory, config_path, config_sha)
        time.sleep(0.2)
    return read_result(result, config_sha)


def start_worker(directory, config_path, config_sha):
    config = load_config(config_path)
    if is_running(directory):
        return
    with (directory / "worker.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(Path(config_path).resolve()),
            ],
            cwd=Path(__file__).resolve().parents[3],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
                "PYTHONUNBUFFERED": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        directory / "launch.json",
        dict(
            pid=process.pid,
            config_sha256=config_sha,
            command=process.args,
            started_at=time.time(),
        ),
    )
    until = time.monotonic() + 20
    while not is_running(directory):
        if process.poll() is not None:
            status = directory / "status.json"
            if status.exists() and json.loads(status.read_text()).get("pid") == process.pid:
                return  # A very short job may finish before the first lock poll.
            raise ValueError("Feature worker failed to start")
        if time.monotonic() >= until:
            raise ValueError("Feature worker failed to start")
        time.sleep(0.1)


def read_result(path, config_sha):
    value = json.loads(Path(path).read_text())
    if value["config_sha256"] != config_sha:
        raise ValueError("Resident worker configuration changed")
    if not value["ok"]:
        raise ValueError("Resident features failed: " + value["error"])
    return value["result"]


def process_birth(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def watchdog(config_path, pid, birth, deadline):
    directory = Path(load_config(config_path)["directory"])
    while process_birth(pid) == birth:
        reason = "service_deadline" if time.monotonic() >= deadline else None
        active = directory / "active.json"
        if active.exists():
            value = json.loads(active.read_text())
            if value.get("pid") == pid and value.get("birth") == birth:
                if value.get("deadline", float("inf")) <= time.time():
                    reason = "job_deadline"
                if value.get("cancel") and Path(value["cancel"]).exists():
                    reason = "caller_cancelled"
        if reason:
            write_json(
                directory / "watchdog-termination.json",
                dict(pid=pid, reason=reason, timestamp=time.time()),
            )
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(2)
                if process_birth(pid) == birth:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        time.sleep(0.2)


def expire(_signum, _frame):
    raise TimeoutError("Native feature job exceeded bounded deadline")


def serve(config_path):
    config = load_config(config_path)
    config_sha = digest(config_path)
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    queue = directory / "jobs"
    queue.mkdir(exist_ok=True)
    from driversi.data.catalog.feature_session import FeatureSession
    from driversi.rsi.experiments.reactive_postprocess import execute_job

    session = FeatureSession()
    started = last = time.time()
    done = 0
    with (directory / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGALRM, expire)
        birth = process_birth(os.getpid())
        guard = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path),
                "--watchdog-pid",
                str(os.getpid()),
                "--watchdog-birth",
                birth,
                "--watchdog-deadline",
                str(time.monotonic() + config["max_seconds"]),
            ],
            start_new_session=True,
        )
        while done < config["max_jobs"] and time.time() - started < config["max_seconds"]:
            pending = [
                p
                for p in sorted(queue.glob("*.request.json"))
                if not p.with_name(p.name.replace(".request.", ".result.")).exists()
                and not p.with_name(p.name.replace(".request.", ".running.")).exists()
            ]
            if not pending:
                if time.time() - last >= config["idle_seconds"]:
                    break
                time.sleep(0.2)
                continue
            path = min(pending, key=lambda p: p.stat().st_mtime_ns)
            request = json.loads(path.read_text())
            result_path = path.with_name(path.name.replace(".request.", ".result."))
            running_path = path.with_name(path.name.replace(".request.", ".running."))
            began = time.time()
            write_json(
                running_path, dict(pid=os.getpid(), started_at=began, config_sha256=config_sha)
            )
            result = dict(ok=False, config_sha256=config_sha, started_at=began)
            try:
                if digest(config_path) != config_sha or request["config_sha256"] != config_sha:
                    raise ValueError("Feature service configuration changed")
                remaining = min(
                    request["deadline"] - time.time(),
                    config["max_seconds"] - (time.time() - started),
                )
                if remaining <= 0:
                    raise TimeoutError("Feature request expired before execution")
                if digest(request["job"]) != request["job_sha256"]:
                    raise ValueError("Feature request changed")
                job = json.loads(Path(request["job"]).read_text())
                validate_job(job, config)
                cancel = path.with_name(path.name.replace(".request.", ".cancel."))
                if cancel.exists():
                    raise TimeoutError("Feature caller cancelled before execution")
                write_json(
                    directory / "active.json",
                    dict(
                        pid=os.getpid(),
                        birth=birth,
                        deadline=time.time() + remaining,
                        cancel=str(cancel),
                    ),
                )
                signal.setitimer(signal.ITIMER_REAL, remaining)
                value = execute_job(job, feature_session=session)
                session.validate()
                result.update(ok=True, result=value)
            except Exception as error:
                result["error"] = type(error).__name__ + ": " + str(error)[:1000]
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            write_json(directory / "active.json", dict(pid=os.getpid(), birth=birth))
            done += 1
            last = time.time()
            result.update(finished_at=last, elapsed_seconds=last - began)
            write_json(result_path, result)
            write_json(
                directory / "status.json",
                dict(
                    status="waiting",
                    pid=os.getpid(),
                    completed=done,
                    config_sha256=config_sha,
                    updated_at=last,
                ),
            )
        guard.terminate()
        guard.wait(timeout=5)
        write_json(
            directory / "status.json",
            dict(
                status="stopped",
                pid=os.getpid(),
                completed=done,
                config_sha256=config_sha,
                updated_at=time.time(),
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watchdog-pid", type=int)
    parser.add_argument("--watchdog-birth")
    parser.add_argument("--watchdog-deadline", type=float)
    args = parser.parse_args()
    if args.watchdog_pid is not None:
        watchdog(args.config, args.watchdog_pid, args.watchdog_birth, args.watchdog_deadline)
    else:
        serve(args.config)
