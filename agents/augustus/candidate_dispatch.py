"""Same-host native candidate chunks, parallel execution only; no scoring changes."""

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.agents.backends.native_process import OwnedTree
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def precompute(plan, previous, frozen):
    placement = plan["data"].get("full_evaluation_execution")
    if not placement:
        return
    if set(placement) != {"hostname", "gpus"} or socket.gethostname() != placement["hostname"]:
        raise ValueError("Candidate benchmark must run on the verified baseline host")
    gpus = placement["gpus"]
    if (
        not gpus
        or len(gpus) > 8
        or len(set(gpus)) != len(gpus)
        or any(type(x) is not int or x < 0 for x in gpus)
    ):
        raise ValueError("Invalid evaluation allocation")
    out = Path(plan["output"])
    plan_path = out / "training-plan.json"
    trees = []
    processes = []
    started = time.monotonic()
    try:
        for shard, gpu in enumerate(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            args = [
                plan["python"],
                "-u",
                "-m",
                "driversi_agents.augustus.candidate_dispatch",
                str(plan_path),
                "--shard",
                str(shard),
                "--shards",
                str(len(gpus)),
            ]
            tree = OwnedTree()
            with (out / f"candidate-worker-{shard}.log").open("ab") as log:
                process = subprocess.Popen(
                    args,
                    cwd=project_root(),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            tree.attach(process)
            trees.append(tree)
            processes.append(process)
        deadline = started + plan["resolved"]["request"]["budget"]["max_wall_seconds"] - 30
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError("Native candidate chunk worker failed; saved outputs preserved")
            if time.monotonic() > deadline:
                raise TimeoutError("Candidate parallel wall budget exhausted")
            completed = len(list((out / "candidate-chunks").glob("*/report.json")))
            write_json(
                out / "dispatch-progress.json",
                dict(
                    hostname=socket.gethostname(),
                    completed_chunks=completed,
                    expected_chunks=len(frozen["chunks"]),
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            time.sleep(10)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Native candidate worker failed")
    finally:
        for tree in trees:
            tree.stop()


def worker(plan_path, shard, shards):
    from driversi.agents.backends.native import digest
    from driversi_agents.augustus.evaluation import evaluate
    from driversi_agents.augustus.full_evaluation import check_plan, completed_chunk

    plan = read_mapping(plan_path)
    previous = read_mapping(plan["previous"])
    placement = plan["data"]["full_evaluation_execution"]
    if socket.gethostname() != placement["hostname"] or not 0 <= shard < shards == len(
        placement["gpus"]
    ):
        raise ValueError("Candidate worker placement mismatch")
    for path, sha in plan["files"].items():
        if digest(path) != sha:
            raise ValueError("Frozen candidate input changed")
    frozen = check_plan(plan["data"]["full_evaluation_plan"])
    for i in range(shard, len(frozen["chunks"]), shards):
        if completed_chunk(frozen, i) is None:
            raise ValueError("Missing parent chunk")
        chunk = Path(frozen["output"]) / "chunks" / f"{i:04d}"
        evaluate(
            chunk / "data/receipt.json",
            previous["checkpoint"],
            read_mapping(chunk / "context/receipt.json")["metric_cache"],
            Path(plan["output"]) / "candidate-chunks" / f"{i:04d}",
            seed=frozen["config"]["seed"] + i,
            batch_size=frozen["config"]["batch_size"],
            reuse_model=True,
        )
        print("completed", i, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    worker(a.plan, a.shard, a.shards)


if __name__ == "__main__":
    main()
