"""Materialize completed chunk caches with hardlinks, preserving native bytes."""

import os
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi_agents.augustus.full_evaluation import check_plan, merge


def link(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target) and digest(source) != digest(target):
            raise ValueError("Materialized cache differs")
    else:
        os.link(source, target)


def materialize(plan_path, output):
    plan = check_plan(plan_path)
    if not merge(plan_path)["complete"]:
        raise ValueError("Full navtest must finish before retrieval or training")
    out = Path(output)
    blocks, metric_paths = [], []
    for i in range(len(plan["chunks"])):
        chunk = Path(plan["output"]) / "chunks" / f"{i:04d}"
        block = read_mapping(chunk / "data/receipt.json")["block"]
        blocks.extend(block["records"])
        for j, batch in enumerate(sorted(Path(block["cache"]).glob("batch_*"))):
            for file in batch.iterdir():
                if file.is_file():
                    link(file, out / "evaluation-data/cache" / f"batch_{i:04d}_{j}" / file.name)
        context = read_mapping(chunk / "context/receipt.json")
        base = Path(context["metric_cache"])
        for name in context["files"]:
            source = Path(name)
            if source.name == "metric_cache.pkl":
                target = out / "evaluation-context/metric_cache" / source.relative_to(base)
                link(source, target)
                metric_paths.append(str(target))
    if len(blocks) != 12146 or len({x["frame_id"] for x in blocks}) != 12146:
        raise ValueError("Incomplete materialized population")
    if len(metric_paths) != 12146 or len(set(metric_paths)) != 12146:
        raise ValueError("Incomplete native scoring contexts")
    cache = out / "evaluation-context/metric_cache"
    meta = cache / "metadata/metric_cache_metadata_node_0.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    text = "file_name\n" + "\n".join(metric_paths) + "\n"
    if meta.exists() and meta.read_text() != text:
        raise ValueError("Materialized native metadata changed")
    meta.write_text(text)
    write_json(
        out / "evaluation-data/receipt.json",
        dict(
            block=dict(cache=str(out / "evaluation-data/cache"), records=blocks),
            source_plan=str(plan_path),
            source_plan_sha256=digest(plan_path),
        ),
    )
    write_json(
        out / "evaluation-context/receipt.json",
        dict(
            metric_cache=str(cache),
            files={p: digest(p) for p in [str(meta), *metric_paths]},
            source_plan=str(plan_path),
        ),
    )
    return dict(status="prepared", count=len(blocks), metric_cache=str(cache))
