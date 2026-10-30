"""Locked pilot permissions never grant training access to test sources."""
import json

import pytest

from driversi.agents.backends.native import digest
from driversi.training.contracts.evaluation import resolve_context
from driversi.training.orchestration import lifecycle
from driversi_agents.chainflow import benchmark


@pytest.fixture
def context(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    index = tmp_path / "index.json"
    index.write_text("{}")
    row = dict(source_log="test-a_00000_00010", metric_cache=str(cache / "metric_cache.pkl"))
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps(dict(kind="chainflow_locked_navtest_pilot")))
    proof = dict(test_index=str(index), test_index_sha256=digest(index), records=[row])
    monkeypatch.setattr(benchmark, "read_pilot_subset", lambda path: proof)
    monkeypatch.setattr(benchmark, "official_filter", lambda split: (index, dict(log_names=[row["source_log"], "test-b_00000_00010"])))
    monkeypatch.setattr(lifecycle, "read_mapping", lambda path: dict(log_names=[row["source_log"], "test-b_00000_00010"]))
    data = dict(train=dict(source_logs=["train-a_00000_00010"]),
                development=dict(source_logs=[row["source_log"]]), parent_provenance="unknown",
                evaluation_protocol=benchmark.PILOT_PROTOCOL, evaluation_subset=str(subset))
    evaluator = dict(schema_version=3, model="chainflow_stage1", protocol="chainflow_native_navsim_pdm_v1",
                     development_source_logs=[row["source_log"]], dataset_manifest_sha256="a" * 64,
                     metric_cache=str(cache), files={str(index): digest(index), str(subset): digest(subset)})
    return data, evaluator


def test_named_pilot_allows_only_locked_subset_groups(context):
    data, _ = context
    assert lifecycle.sources(data)["development"] == ["test-a"]
    data["development"]["source_logs"] = ["test-b_00000_00010"]
    with pytest.raises(ValueError, match="source population"):
        lifecycle.sources(data)


def test_pilot_does_not_grant_test_training_access(context):
    data, _ = context
    data["train"]["source_logs"] = ["test-b_00000_00010"]
    with pytest.raises(ValueError, match="test logs"):
        lifecycle.sources(data)


def test_formal_protocol_still_requires_complete_test_population(context):
    data, _ = context
    data["evaluation_protocol"] = "official_navtest_v1"
    with pytest.raises(ValueError, match="complete navtest"):
        lifecycle.sources(data)


def test_evaluator_seals_subset_and_full_index(context, tmp_path):
    _, value = context
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(value))
    assert resolve_context(path, "chainflow_stage1", ["test-a"])["value"]["schema_version"] == 3
    value["files"].pop(next(iter(value["files"])))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="subset and full navtest"):
        resolve_context(path, "chainflow_stage1", ["test-a"])


def test_pilot_evaluator_cannot_use_formal_schema(context, tmp_path):
    _, value = context
    value["schema_version"] = 2
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="one exact navtest index"):
        resolve_context(path, "chainflow_stage1", ["test-a"])
