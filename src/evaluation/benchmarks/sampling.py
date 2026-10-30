"""Single-node sampling capacity probe: frozen initialization, no optimizer or renderer."""
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from driversi.agents.interfaces.config import build_agent, read_agent_config
from driversi.data.scenes.scene_store import scene_header
from driversi.training.sampling.collector import Collector
from driversi.training.sampling.environment import source_log_group, validate_splits


def _rss_mib(pids):
    values = []
    for pid in pids:
        path = Path(f"/proc/{pid}/status")
        if not path.exists():
            return None
        row = next((line for line in path.read_text().splitlines() if line.startswith("VmRSS:")), None)
        if row is None:
            return None
        values.append(int(row.split()[1]) / 1024)
    return sum(values)  # Sum of RSS may double-count shared pages; not physical memory attribution.


def benchmark_sampling(suite, output, *, worlds=32, workers=0, steps=16, batches=5,
                       seed=0, scene_cache_size=32, max_rollout_mib=1024,
                       worker_timeout_seconds=30., record=False, min_seconds=0., agent_config=None):
    if agent_config is None:
        raise ValueError("Sampling benchmark requires an explicit agent_config")
    agent = read_agent_config(Path(agent_config))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if any(type(v) is not int or v < 1 for v in (worlds, steps, batches)):
        raise ValueError("Positive world, step and batch counts required")
    if isinstance(min_seconds, bool) or not isinstance(min_seconds, (int, float)) or not np.isfinite(min_seconds) or min_seconds < 0:
        raise ValueError("min_seconds must be finite and nonnegative")
    manifest = json.loads(Path(suite).read_text())
    paths = [row["path"] for row in manifest["train"]]
    validate_splits(paths, [row["path"] for row in manifest["evaluation"]])
    headers = [scene_header(path) for path in paths]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    report = {"schema_version": 2, "kind": "sampling-capacity-probe", "status": "running",
              "worlds": worlds, "workers": workers, "steps_per_batch": steps, "batches": 0, "requested_batches": batches, "min_seconds": min_seconds,
              "seed": seed, "device": "cpu", "model": agent["model_target"],
              "training": False, "rendering": False, "official_test": False,
              "source_scenes": len(paths), "scene_paths": paths, "maps": sorted({h["metadata"]["map_name"] for h in headers}),
              "scene_seconds": [h["max_steps"] * .1 for h in headers],
              "controlled_actor_counts": [len(h["controlled_ids"]) for h in headers],
              "recording": "temporary compressed NPZ arrays" if record else "disabled",
              "limits": ["CPU sampling only; no optimization, checkpoint writes or rendering",
                         "RSS sum may double-count shared pages; does not include filesystem page cache",
                         "Measured scene pool reused; not full-dataset coverage or long-term reliability"],
              "measurements": []}
    visited = Counter()

    def summarize():
        rows = report["measurements"]
        count = sum(row["valid_agent_transitions"] for row in rows)
        seconds = sum(row["sample_seconds"] for row in rows)
        inclusive = sum(row["sample_and_record_seconds"] for row in rows)
        by_map = Counter()
        for index, amount in visited.items():
            by_map[headers[index]["metadata"]["map_name"]] += amount
        report.update(batches=len(rows), valid_agent_transitions=count,
                      sample_seconds=seconds, sample_and_record_seconds=inclusive,
                      valid_transitions_per_second=count / seconds if seconds else None,
                      valid_transitions_per_second_with_recording=count / inclusive if inclusive else None,
                      world_steps=sum(row["world_steps"] for row in rows),
                      resets=sum(row["resets"] for row in rows),
                      observed_scene_indices=sorted(visited), observed_scene_count=len(visited),
                      observed_source_logs=sorted({source_log_group(headers[i]["metadata"]["source_log"])
                                                   for i in visited}),
                      valid_transitions_by_map=dict(sorted(by_map.items())))
        if rows:
            report["batch_sample_latency_seconds"] = dict(zip(
                ("p50", "p95", "p99", "max"),
                np.percentile([row["sample_seconds"] for row in rows], [50, 95, 99, 100]).tolist()))

    def persist():
        summarize()
        pending = output.with_suffix(output.suffix + ".pending")
        pending.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(output)

    collector = None
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        model = build_agent(agent)
        collector = Collector(paths, worlds, seed, workers=workers, scene_cache_size=scene_cache_size,
                              max_rollout_mib=max_rollout_mib, worker_timeout_seconds=worker_timeout_seconds)
        report["startup_seconds"] = time.perf_counter() - start
        pids = [os.getpid(), *[process.pid for _, process, _ in collector.pool.clients]]
        report["rss_sum_before_mib"] = _rss_mib(pids)
        with tempfile.TemporaryDirectory(prefix="sampling-record-", dir=output.parent) as directory:
            probe_start = last_report = time.perf_counter()
            index = 0
            while index < batches or time.perf_counter() - probe_start < min_seconds:
                begin = time.perf_counter()
                data = collector.collect(model, steps, policy_version=f"benchmark-frozen-seed{seed}")
                row = dict(collector.last_metrics, batch=index)
                write_begin = time.perf_counter()
                if record:
                    np.savez_compressed(Path(directory) / "batch.npz", **data)
                row["record_seconds"] = time.perf_counter() - write_begin if record else 0.
                row["sample_and_record_seconds"] = time.perf_counter() - begin
                row["rss_sum_with_rollout_mib"] = _rss_mib(pids)
                indices, counts = np.unique(data["scene_index"][data["valid"]], return_counts=True)
                visited.update({int(i): int(n) for i, n in zip(indices, counts)})
                del data
                row["rss_sum_after_release_mib"] = _rss_mib(pids)
                report["measurements"].append(row)
                index += 1
                report["probe_wall_seconds"] = time.perf_counter() - probe_start
                if time.perf_counter() - last_report >= 30:
                    persist()
                    last_report = time.perf_counter()
        report.update(status="complete", shard_statistics=collector.pool.stats())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if collector is not None:
            collector.close()
        report["total_wall_seconds"] = time.perf_counter() - start
        persist()
    return report
