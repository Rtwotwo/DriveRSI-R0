import copy

import pytest

from driversi.rsi.workflow.engine import prepare
from driversi.rsi.workflow.graph import blocked_by
from driversi.rsi.workflow.providers import Provider
from tests.unit.rsi.test_harness import providers, request


def test_unordered_dag_is_compiled_before_execution(tmp_path):
    raw = request()
    raw["schema_version"] = 2
    raw["steps"].reverse()
    before = copy.deepcopy(raw)
    task = prepare(raw, tmp_path, tmp_path / "db", providers())
    assert task["execution_order"] == ["a", "b"]
    assert task["dependencies"] == {"b": ["a"], "a": []}
    assert raw == before


def test_shared_source_is_reserved_once_and_order_is_stable(tmp_path):
    raw = request()
    raw["schema_version"] = 2
    raw["steps"].append({**copy.deepcopy(raw["steps"][1]), "id": "c"})
    task = prepare(raw, tmp_path, tmp_path / "db", providers())
    assert task["reserved_agent_steps"] == 10
    assert task["execution_order"] == ["a", "b", "c"]
    assert task["dependencies"]["b"] == task["dependencies"]["c"] == ["a"]


def test_cycle_rejected_without_preparing_any_provider(tmp_path):
    registry = providers()
    registry._items["consumer"] = Provider("consumer", "1", (("data", "evidence"),), "evidence",
        lambda *args: pytest.fail("Cycle must fail before provider preparation"), None)
    raw = request()
    raw["schema_version"] = 2
    raw["steps"][0].update(provider="consumer", inputs={"data": "b"})
    with pytest.raises(ValueError, match="Cycle"):
        prepare(raw, tmp_path, tmp_path / "db", registry)


def test_blocking_follows_dependencies_only():
    row = {"inputs": {"data": "source"}}
    assert blocked_by(row, [{"id": "other", "status": "failed"}, {"id": "source", "status": "completed"}]) == {}
    assert blocked_by(row, [{"id": "source", "status": "rejected"}]) == {"source": "rejected"}
