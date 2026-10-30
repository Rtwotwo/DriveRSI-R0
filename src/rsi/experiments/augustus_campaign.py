"""Owned full evaluation with explicit placement; next stages require exact coverage."""

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi_agents.augustus.full_evaluation import merge, prepare


def execution_settings(execution, plan, root):
    """Host placement is separate from the immutable statistical protocol."""
    value = read_mapping(execution)
    if value["host"] != socket.gethostname():
        raise ValueError("Execution belongs to another host")
    if value["plan_sha256"] != digest(plan):
        raise ValueError("Execution belongs to another evaluation plan")
    gpus = value["gpus"]
    if not gpus or len(gpus) > 8 or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("Declare one to eight GPU indices")
    if len(set(gpus)) != len(gpus):
        raise ValueError("Unique worker GPUs required")
    logs = Path(value["output"]).resolve()
    if not logs.is_relative_to(root / "exp"):
        raise ValueError("Execution records must remain under exp")
    previous = read_mapping(value["predecessor_stop_receipt"])
    if previous.get("plan_sha256") != value["plan_sha256"] or not previous.get("all_stopped"):
        raise ValueError("Prior execution must stop before repartitioning shared chunks")
    return value, logs


def evaluate_full(config, *, execution=None):
    c = read_mapping(config)
    root = project_root()
    out = Path(c["output"]) / "baseline"
    plan = out / "plan.json"
    if not plan.exists():
        prepare(config, out)
    if merge(plan)["complete"]:
        return
    placement = None
    logs = out
    if execution is not None:
        placement, logs = execution_settings(execution, plan, root)
        logs.mkdir(parents=True, exist_ok=True)
        if (logs / "launch.json").exists():
            raise ValueError("Preserve prior launch receipt; choose a new execution directory")
    gpus = placement["gpus"] if placement is not None else c["gpus"]
    if len(set(gpus)) != len(gpus):
        raise ValueError("Unique worker GPUs required")
    processes = []
    streams = []
    try:
        for shard, gpu in enumerate(gpus):
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONHASHSEED=str(c["seed"]),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                PYTHONUNBUFFERED="1",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
            )
            stream = (logs / f"worker-{shard}.log").open("a")
            streams.append(stream)
            p = subprocess.Popen(
                [
                    str(root / ".venvs/augustus/bin/python"),
                    str(root / "scripts/evaluation/run_augustus_benchmark.py"),
                    "worker",
                    str(plan),
                    "--shard",
                    str(shard),
                    "--shards",
                    str(len(gpus)),
                ],
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(p)
        write_json(
            logs / "launch.json",
            dict(
                pids=[p.pid for p in processes],
                gpus=gpus,
                expected=12146,
                host=socket.gethostname(),
                plan_sha256=digest(plan),
                shards=len(gpus),
                training_started=False,
            ),
        )
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Full evaluation worker failed; downstream training stopped")
            merge(plan)
            time.sleep(30)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Full evaluation worker failed")
        result = merge(plan)
        if not result["complete"]:
            raise ValueError("Full official coverage not reached")
        write_json(logs / "exit.json", dict(status="completed", completed=12146))
        return result
    except BaseException as exc:
        write_json(
            logs / "exit.json", dict(status="failed", error=type(exc).__name__ + ": " + str(exc))
        )
        raise
    finally:
        stop_owned_workers(processes)
        for f in streams:
            f.close()


def stop_owned_workers(processes):
    """Reap only sessions created by this dispatcher, including orphaned pools."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        # A failed/reaped leader can still have a scoring or preparation pool.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def interrupt_dispatch(signum, frame):
    raise InterruptedError("Evaluation dispatcher interrupted: " + str(signum))


def process_identity(pid):
    """Bind an already-running worker without owning unrelated process groups."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return dict(
            pid=pid, start_ticks=stat[19], command=(proc / "cmdline").read_bytes().split(b"\0")[:-1]
        )
    except FileNotFoundError:
        return None


def supervise_existing(config):
    """Continue receipts for adopted workers; never enter the superseded relabel route."""
    c = read_mapping(config)
    out = Path(c["output"])
    ownership = read_mapping(out / "evaluation-only-workers.json")
    plan = out / "baseline/plan.json"
    while True:
        summary = merge(plan)
        alive = []
        for expected in ownership["workers"]:
            current = process_identity(expected["pid"])
            if current:
                actual = dict(current, command=[x.decode() for x in current["command"]])
                if actual != expected:
                    raise ValueError("Recorded evaluation worker identity changed")
                alive.append(current)
        if summary["complete"] and not alive:
            write_json(
                out / "production-status.json",
                dict(
                    status="waiting",
                    stage="driveRL_new_history_and_GS",
                    evaluation_completed=12146,
                    relabel_fallback=False,
                    training_started=False,
                    reason="Full evaluation complete; new-history production plan must pass readiness",
                ),
            )
            return summary
        if len(alive) != len(ownership["workers"]) and not summary["complete"]:
            # A worker can legitimately finish its disjoint half first.
            missing = []
            from driversi_agents.augustus.full_evaluation import completed_chunk

            frozen = read_mapping(plan)
            for shard, expected in enumerate(ownership["workers"]):
                if process_identity(expected["pid"]) is None:
                    missing += [
                        i
                        for i in range(len(frozen["chunks"]))
                        if i % len(ownership["workers"]) == shard
                        and completed_chunk(frozen, i) is None
                    ]
            if missing:
                summary.update(status="failed", failed_worker_shards=True)
                write_json(out / "baseline/summary.json", summary)
                write_json(
                    out / "production-status.json",
                    dict(
                        status="failed",
                        stage="full_evaluation",
                        reason="A worker stopped before completing its assigned chunks",
                        training_started=False,
                    ),
                )
                raise RuntimeError("Full evaluation worker stopped incomplete")
        time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["supervise-existing", "evaluate-on-host"])
    parser.add_argument("config")
    parser.add_argument("--execution")
    args = parser.parse_args()
    if args.command == "supervise-existing":
        supervise_existing(args.config)
    else:
        if not args.execution:
            parser.error("evaluate-on-host requires --execution")
        signal.signal(signal.SIGTERM, interrupt_dispatch)
        result = evaluate_full(args.config, execution=args.execution)
        write_json(
            Path(read_mapping(args.config)["output"]) / "production-status.json",
            dict(
                status="waiting",
                stage="driverl_new_history_gs",
                evaluation_completed=12146,
                relabel_fallback=False,
                training_started=False,
                reason="Full evaluation complete; new-history production plan must pass readiness",
            ),
        )
