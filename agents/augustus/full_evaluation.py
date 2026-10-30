"""Resumable full official NAVSIM evaluation with resident native AUGUstus workers.

Chunk layout and RNG are fixed before execution. A partial mean is never a
benchmark score. CPU data preparation overlaps model inference without changing
native visual preprocessing, trajectory generation, or PDM mathematics.
"""

import argparse
import json
import math
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi_agents.chainflow.benchmark import read_index

METRICS = ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C")


def prepare(config, output):
    c = read_mapping(config)
    out = Path(output).resolve()
    root = project_root()
    if not out.is_relative_to(root / "exp"):
        raise ValueError("Artifacts must be in exp")
    index = read_index(c["test_index"], "navtest")
    rows = sorted(index["records"], key=lambda x: x["token"])
    if len(rows) != 12146:
        raise ValueError("Exact official 12146 population required")
    if c.get("chunk_size") != 32 or c.get("batch_size") != 2 or type(c.get("seed")) is not int:
        raise ValueError("Declare fixed 32-scene chunks, batch 2 and seed")
    code = [
        __file__,
        root / "agents/augustus/evaluation.py",
        root / "agents/augustus/preparation.py",
        root / "agents/augustus/cache.py",
    ]
    from driversi_agents.augustus.adapter import complete_assets

    inputs = {
        str(Path(p).resolve()): digest(p)
        for p in [
            config,
            c["test_index"],
            c["test_annotations"],
            *complete_assets(root / "weights/augustus/SFT"),
            Path(c["checkpoint"]) / "model.safetensors",
            *code,
        ]
    }
    plan = dict(
        schema_version=1,
        kind="augustus_full_navtest",
        config=c,
        output=str(out),
        records=rows,
        chunks=[rows[i : i + 32] for i in range(0, len(rows), 32)],
        inputs=inputs,
        protocol="native_sft_native_pdm_v1_full_navtest_chunk_seed",
        seed_policy="seed+chunk_index",
        full_population=True,
        test_details_private=True,
        training_permitted_only_after_full_coverage=True,
    )
    if (out / "plan.json").exists():
        old = read_mapping(out / "plan.json")
        if old != plan:
            raise ValueError("Prepared evaluation changed; use another output")
        return old
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "plan.json", plan)
    return plan


def check_plan(path):
    p = read_mapping(path)
    for name, sha in p["inputs"].items():
        if digest(name) != sha:
            raise ValueError("Frozen evaluation input changed: " + name)
    p["_plan_sha256"] = digest(path)
    return p


def prepare_chunk(plan, index):
    from driversi_agents.augustus.preparation import metric_context, rows_cache

    out = Path(plan["output"]) / "chunks" / f"{index:04d}"
    c = plan["config"]
    records = plan["chunks"][index]
    rows_cache(records, c["test_annotations"], out / "data")
    context = metric_context(records, c["maps"], out / "context")
    return dict(cache=str(out / "data/receipt.json"), metric_cache=context["metric_cache"])


def completed_chunk(plan, index):
    path = Path(plan["output"]) / "chunks" / f"{index:04d}" / "completed.json"
    if not path.exists():
        return None
    receipt = read_mapping(path)
    if receipt["plan_sha256"] != (
        plan.get("_plan_sha256") or digest(Path(plan["output"]) / "plan.json")
    ):
        raise ValueError("Chunk belongs to another full evaluation")
    result = read_mapping(receipt["report"])
    if digest(receipt["report"]) != receipt["sha256"]:
        raise ValueError("Chunk report changed")
    expected = {x["token"]: x["source_log"] for x in plan["chunks"][index]}
    actual = {x["token"]: x["source_log"] for x in result["rows"]}
    if expected != actual or len(result["rows"]) != len(expected):
        raise ValueError("Chunk population mismatch")
    for name, sha in result["files"].items():
        if digest(name) != sha:
            raise ValueError("Chunk predictions changed")
    return result


def worker(path, shard, shards, *, max_chunks=None):
    import fcntl
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from driversi_agents.augustus.evaluation import evaluate

    plan = check_plan(path)
    out = Path(plan["output"])
    c = plan["config"]
    if not 0 <= shard < shards <= 8:
        raise ValueError("Invalid shard")
    with (out / f"worker-{shard}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = [
            i
            for i in range(len(plan["chunks"]))
            if i % shards == shard and completed_chunk(plan, i) is None
        ]
        if max_chunks is not None:
            pending = pending[:max_chunks]
        if not pending:
            return dict(status="completed", new_chunks=0)
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            future = pool.submit(prepare_chunk, plan, pending[0])
            for position, index in enumerate(pending):
                chunk = out / "chunks" / f"{index:04d}"
                write_json(
                    out / f"worker-{shard}-progress.json",
                    dict(status="preparing", chunk=index, completed=position),
                )
                try:
                    prepared = future.result()
                    if position + 1 < len(pending):
                        future = pool.submit(prepare_chunk, plan, pending[position + 1])
                    attempts = chunk / "attempts"
                    attempts.mkdir(exist_ok=True)
                    attempt = attempts / f"{len(list(attempts.iterdir())):03d}"
                    write_json(
                        out / f"worker-{shard}-progress.json",
                        dict(status="evaluating", chunk=index, completed=position),
                    )
                    result = evaluate(
                        prepared["cache"],
                        c["checkpoint"],
                        prepared["metric_cache"],
                        attempt,
                        seed=c["seed"] + index,
                        batch_size=c["batch_size"],
                        reuse_model=True,
                    )
                    write_json(
                        chunk / "completed.json",
                        dict(
                            report=str(attempt / "report.json"),
                            sha256=digest(attempt / "report.json"),
                            plan_sha256=digest(path),
                        ),
                    )
                    write_json(
                        out / f"worker-{shard}-progress.json",
                        dict(
                            status="running",
                            chunk=index,
                            completed=position + 1,
                            scenes_completed_this_process=sum(
                                len(plan["chunks"][j]) for j in pending[: position + 1]
                            ),
                        ),
                    )
                    print(
                        json.dumps(
                            dict(
                                shard=shard,
                                chunk=index,
                                samples=len(result["rows"]),
                                status="completed",
                            )
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    write_json(
                        out / f"worker-{shard}-failure.json",
                        dict(chunk=index, error=type(exc).__name__ + ": " + str(exc)),
                    )
                    raise
        return dict(status="completed", new_chunks=len(pending))


def aggregate_rows(rows, records):
    expected = {r["token"]: r["source_log"] for r in records}
    tokens = [r["token"] for r in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate test outcome")
    if any(r["token"] not in expected or expected[r["token"]] != r["source_log"] for r in rows):
        raise ValueError("Unexpected test member")
    for row in rows:
        if any(
            type(row["metrics"].get(m)) not in (int, float)
            or not math.isfinite(row["metrics"][m])
            or not 0 <= row["metrics"][m] <= 1
            for m in METRICS
        ):
            raise ValueError("Invalid native test metric")
    complete = len(rows) == len(expected)
    return dict(
        complete=complete,
        expected=len(expected),
        completed=len(rows),
        missing=len(expected) - len(rows),
        metrics={m: math.fsum(r["metrics"][m] for r in rows) / len(rows) for m in METRICS}
        if complete
        else None,
        zero_count=sum(r["metrics"]["PDMS"] == 0 for r in rows),
        below_95_count=sum(r["metrics"]["PDMS"] < 0.95 for r in rows),
        formal_score_available=complete,
    )


def merge(path):
    plan = read_mapping(path)
    plan["_plan_sha256"] = digest(path)
    rows = []
    reports = []
    for index in range(len(plan["chunks"])):
        value = completed_chunk(plan, index)
        if value:
            rows.extend(value["rows"])
            reports.append(str(Path(plan["output"]) / "chunks" / f"{index:04d}" / "completed.json"))
    summary = aggregate_rows(rows, plan["records"])
    out = Path(plan["output"])
    summary.update(
        status="completed" if summary["complete"] else "running",
        full_benchmark=True,
        protocol=plan["protocol"],
        optimizer_updates=0,
        training_started=False,
    )
    write_json(out / "summary.json", summary)
    if summary["complete"]:
        write_json(
            out / "report.json",
            dict(
                **summary,
                rows=rows,
                plan_sha256=digest(path),
                chunk_receipts={p: digest(p) for p in reports},
            ),
        )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("config")
    q.add_argument("output")
    q = sub.add_parser("worker")
    q.add_argument("plan")
    q.add_argument("--shard", type=int, required=True)
    q.add_argument("--shards", type=int, required=True)
    q.add_argument("--max-chunks", type=int)
    q = sub.add_parser("merge")
    q.add_argument("plan")
    a = p.parse_args()
    if a.cmd == "prepare":
        v = prepare(a.config, a.output)
        print(len(v["records"]))
    elif a.cmd == "worker":
        print(worker(a.plan, a.shard, a.shards, max_chunks=a.max_chunks))
    else:
        print(merge(a.plan))


if __name__ == "__main__":
    main()
