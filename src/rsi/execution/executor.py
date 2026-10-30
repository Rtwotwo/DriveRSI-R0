"""Own-process supervisor for bounded training/sampling and separate evaluation."""
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.runtime.budget import BudgetExceeded, write_json
from driversi.rsi.execution.execution_spec import resolve_execution
from driversi.rsi.execution.jobs import JobStore, process_identity
from driversi.runtime.processes import group_members, reap_helpers, stop_group, subreaper


def children_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def recover(store, job_id):
    state = store.get(job_id)
    owner = state["supervisor"]
    import socket
    if owner["host"] != socket.gethostname():
        raise ValueError("Recover on the original host")
    owner_start = process_identity(owner["pid"])
    if owner_start is not None and (owner["start"] is None or owner_start == owner["start"]):
        raise ValueError("Supervisor is still alive; cancel it instead")
    child = state.get("child")
    if child:
        current = process_identity(child["pid"])
        if (current is not None and (child["start"] is None or current == child["start"])
                or any(m["alive"] for m in group_members(child["pid"]).values())):
            raise ValueError("Tool group is still alive; stop owned tool before recovery")
    # Unknown tail consumption cannot be treated as zero.
    state["cost"] = dict(state["job"]["reserved"])
    state["cost"]["trials"] = 1
    state["cost_basis"] = "reserved_caps_unknown_usage"
    return store.settle(state, "failed", "supervisor_lost_usage_unknown_reserved_budget_charged")


def execute(registry_path, spec_path, output_root, *, cancel_requested=None):
    store = JobStore(registry_path)
    raw = read_mapping(Path(spec_path))
    plan = store.plan(raw.get("plan_id"))
    job = resolve_execution(raw, Path(spec_path).resolve().parent, plan)
    state = store.claim(job, output_root)
    job = state["job"]
    directory = Path(state["directory"])
    started = time.monotonic()
    own_start, children_start = time.process_time(), children_cpu()
    gpu_total = 0.0
    prior_subreaper = None
    owns_directory = False
    child = None
    stage_started = None
    gpu_stage = False
    terminal, reason = "failed", "execution_did_not_finish"
    state["cost"]["trials"] = 1

    def refresh():
        nonlocal gpu_total
        progress = directory / "progress.json"
        if owns_directory and progress.exists():
            state["progress"] = json.loads(progress.read_text())
            state["cost"]["sim_agent_steps"] = state["progress"]["admitted_agent_steps"]
        if child is not None:
            reap_helpers(child)
        state["cost"]["cpu_seconds"] = (time.process_time() - own_start +
                                        children_cpu() - children_start +
                                        (sum(m["cpu"] for m in group_members(child.pid).values()) if child else 0))
        state["cost"]["gpu_seconds"] = gpu_total + (
            time.monotonic() - stage_started if gpu_stage and stage_started is not None else 0)
        state["elapsed_seconds"] = time.monotonic() - started

    def check_limits():
        if (cancel_requested is not None and cancel_requested()) or store.get(job["id"])["cancel_requested"]:
            raise InterruptedError("operator_cancelled")
        if state["elapsed_seconds"] > job["wall_seconds"]:
            raise BudgetExceeded("wall_deadline")
        for key in ("cpu_seconds", "gpu_seconds", "sim_agent_steps"):
            if state["cost"][key] > job["reserved"][key]:
                raise BudgetExceeded(key)

    try:
        prior_subreaper = subreaper(True)
        directory.mkdir(parents=True, exist_ok=False)
        owns_directory = True
        write_json(directory / "job.json", job)
        state["status"] = "running"
        store.update(state)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "PYTHONPATH"):
            env.pop(key, None)
        for stage in ("training", "evaluating"):
            refresh()
            check_limits()
            state["phase"] = stage
            state["events"].append({"phase": stage, "elapsed_seconds": state["elapsed_seconds"]})
            store.update(state)
            remaining_cpu = job["reserved"]["cpu_seconds"] - state["cost"]["cpu_seconds"]
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"]) if stage == "training" and job["device"] == "cuda" else ""
            with (directory / f"{stage}.log").open("wb") as log:
                stage_started = time.monotonic()
                gpu_stage = stage == "training" and job["device"] == "cuda"
                child = subprocess.Popen(
                    [sys.executable, "-B", "-m", "driversi.rsi.execution.worker", str(directory / "job.json"),
                     stage, "--cpu-limit", str(remaining_cpu), "--supervisor-pid", str(os.getpid())],
                    stdout=log, stderr=log, env=env, start_new_session=True)
                state["child"] = {"pid": child.pid, "start": process_identity(child.pid)}
                store.update(state)
                while child.poll() is None:
                    refresh()
                    check_limits()
                    store.update(state)
                    time.sleep(0.1)
                refresh()
                check_limits()
                if child.returncode:
                    error_file = directory / "worker-error.json"
                    error = json.loads(error_file.read_text()) if error_file.exists() else {}
                    if error.get("type") == "BudgetExceeded" or child.returncode == -signal.SIGXCPU:
                        raise BudgetExceeded(error.get("message", "cpu_limit"))
                    raise RuntimeError(f"{stage} exited {child.returncode}: {error.get('message', 'see tool log')}")
                # The stage is not finished until all helpers are stopped and reaped.
                stop_group(child)
                refresh()
                check_limits()
                if gpu_stage:
                    gpu_total += time.monotonic() - stage_started
                gpu_stage = False
                stage_started = None
                child = None
                state.pop("child", None)
        result = json.loads((directory / "evaluation-result.json").read_text())
        mining = job.get("tool") == "structured_probe"
        if ((not mining and not result.get("source_separation_checked"))
                or (mining and result.get("purpose") != "development_mining")
                or not result.get("independent_process")
                or result.get("candidate_model") != job["candidate_version"]
                or result.get("evaluator_version") != job["evaluator_version"]
                or plan["protocol"]["metric"] not in result):
            raise ValueError("Independent evaluation evidence mismatch")
        state["evaluation"] = result
        refresh()
        check_limits()
        terminal, reason = "completed", None
    except InterruptedError as exc:
        terminal, reason = "cancelled", str(exc)
    except BudgetExceeded as exc:
        terminal, reason = "budget_exhausted", str(exc)
    except Exception as exc:
        terminal, reason = "failed", f"{type(exc).__name__}: {exc}"
    except BaseException:
        terminal, reason = "cancelled", "supervisor_interrupted"
    finally:
        try:
            stop_group(child)
            refresh()
            # Completion is decided using the last measured sample, after cleanup.
            # No extra ledger read here: settle() handles cancellation atomically.
            if terminal == "completed":
                breach = ("wall_deadline" if state["elapsed_seconds"] > job["wall_seconds"] else
                          next((key for key in ("cpu_seconds", "gpu_seconds", "sim_agent_steps")
                                if state["cost"][key] > job["reserved"][key]), None))
                if breach:
                    terminal, reason = "budget_exhausted", breach
            state = store.settle(state, terminal, reason)
            if owns_directory:
                try:
                    write_json(directory / "result.json", state)
                except OSError as exc:
                    # The atomic database receipt remains authoritative.
                    state["receipt_write_error"] = str(exc)
        finally:
            if prior_subreaper is not None:
                subreaper(prior_subreaper)
    return state
