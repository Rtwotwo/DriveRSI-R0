"""Independent evaluator artifacts, separate from a student's training recipe."""

from pathlib import Path

from driversi.agents.backends.native import digest, signature
from driversi.configuration import read_mapping
from driversi.training.sampling.environment import source_log_group

PROTOCOLS = {
    "augustus": "bundled_navsim_pdm_v1",
    "chainflow_stage1": "chainflow_native_navsim_pdm_v1",
}


def resolve_context(path, model, development_groups):
    path = Path(path).resolve(strict=True)
    value = read_mapping(path)
    required = {
        "schema_version",
        "model",
        "protocol",
        "development_source_logs",
        "dataset_manifest_sha256",
        "metric_cache",
        "files",
    }
    if set(value) != required or value["schema_version"] not in (1, 2, 3):
        raise ValueError("Malformed evaluator context")
    if value["model"] != model or value["protocol"] != PROTOCOLS.get(model):
        raise ValueError("Unregistered native evaluator protocol")
    if {source_log_group(x) for x in value["development_source_logs"]} != set(development_groups):
        raise ValueError("Evaluator sources differ from the training holdout")
    cache = Path(value["metric_cache"]).resolve(strict=True)
    files = value["files"]
    if not files or not cache.is_dir():
        raise ValueError("Evaluator cache is missing")
    for name, expected in files.items():
        artifact = Path(name)
        if not artifact.is_absolute() or (value["schema_version"] == 1 and not artifact.resolve().is_relative_to(cache)):
            raise ValueError("Evaluator artifact outside cache")
        if digest(artifact) != expected:
            raise ValueError("Evaluator artifact changed")
    if value["schema_version"] == 2:
        from driversi_agents.chainflow.benchmark import read_index
        if model != "chainflow_stage1" or len(files) != 1:
            raise ValueError("Benchmark evaluator needs one exact navtest index")
        index = read_index(next(iter(files)), "navtest")
        if any(not Path(r["metric_cache"]).resolve().is_relative_to(cache) for r in index["records"]):
            raise ValueError("Official metric cache outside evaluator root")
    if value["schema_version"] == 3:
        from driversi_agents.chainflow.benchmark import read_pilot_subset
        if model != "chainflow_stage1" or len(files) != 2:
            raise ValueError("Pilot evaluator needs exact subset and full navtest index")
        subsets = [p for p in files if read_mapping(p).get("kind") == "chainflow_locked_navtest_pilot"]
        if len(subsets) != 1:
            raise ValueError("Pilot evaluator lacks one locked subset receipt")
        pilot = read_pilot_subset(subsets[0])
        if files.get(pilot["test_index"]) != pilot["test_index_sha256"]:
            raise ValueError("Pilot evaluator does not seal full navtest population")
        if {source_log_group(r["source_log"]) for r in pilot["records"]} != set(development_groups):
            raise ValueError("Pilot evaluator source groups differ")
        if any(not Path(r["metric_cache"]).resolve().is_relative_to(cache) for r in pilot["records"]):
            raise ValueError("Pilot metric cache outside evaluator root")
    return {
        "manifest": str(path),
        "manifest_sha256": digest(path),
        "signature": signature(value),
        "value": value,
    }
