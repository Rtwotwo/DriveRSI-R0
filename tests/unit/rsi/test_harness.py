import copy

import pytest

from driversi.rsi.workflow.engine import prepare, recover
from driversi.rsi.workflow.providers import Provider, Providers, builtin_providers
from driversi.rsi.workflow.store import HarnessStore, public_tasks


def request():
    return {"schema_version": 1, "id": "task", "label": "test",
            "budget": {"max_wall_seconds": 20, "max_sim_agent_steps": 10},
            "steps": [{"id": "a", "provider": "source", "inputs": {}, "config": {},
                       "max_wall_seconds": 10},
                      {"id": "b", "provider": "consumer", "inputs": {"data": "a"},
                       "config": {}, "max_wall_seconds": 10}]}


def providers():
    registry = Providers()
    registry.register(Provider("source", "1", (), "evidence", lambda c, b, r: (c, 10), None))
    registry.register(Provider("consumer", "2", (("data", "evidence"),), "diagnostic",
                               lambda c, b, r: (c, 0), None))
    return registry


def test_preflight_freezes_version_and_does_not_mutate_request(tmp_path):
    raw = request()
    before = copy.deepcopy(raw)
    result = prepare(raw, tmp_path, tmp_path / "db", providers())
    assert raw == before
    assert result["steps"][1]["provider_version"] == "2"
    assert result["reserved_agent_steps"] == 10
    assert result["steps"][1]["output_kind"] == "diagnostic"


@pytest.mark.parametrize("fault", [
    "unknown", "missing", "forward", "type", "duplicate", "budget",
    "negative", "nan", "extra", "boolean", "cycle",
])
def test_preflight_rejects_before_execution(tmp_path, fault):
    raw, registry = request(), providers()
    if fault == "unknown":
        raw["steps"][0]["provider"] = "untrusted"
    if fault == "missing":
        raw["steps"][1]["inputs"] = {}
    if fault == "forward":
        raw["steps"].reverse()
    if fault == "type":
        registry._items["consumer"] = Provider(
            "consumer", "1", (("data", "qualified_dataset"),), "model", None, None)
    if fault == "duplicate":
        raw["steps"][1]["id"] = "a"
    if fault == "budget":
        raw["budget"]["max_sim_agent_steps"] = 9
    if fault == "negative":
        raw["steps"][1]["max_wall_seconds"] = -1
    if fault == "nan":
        raw["budget"]["max_wall_seconds"] = float("nan")
    if fault == "extra":
        raw["execute_arbitrary_code"] = True
    if fault == "boolean":
        raw["budget"]["max_sim_agent_steps"] = True
    if fault == "cycle":
        raw["steps"][1]["inputs"]["data"] = "b"
    with pytest.raises(ValueError):
        prepare(raw, tmp_path, tmp_path / "db", registry)


def test_provider_registration_does_not_replace_live_implementation():
    registry = providers()
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(registry.get("source"))


def test_no_teacher_dataset_port_on_online_ppo():
    provider = builtin_providers().get("structured_experiment")
    assert provider.inputs == ()
    assert builtin_providers().get("navsim_export").output == "navsim_inspection_dataset"


def stored(tmp_path):
    store = HarnessStore(tmp_path / "db")
    task = prepare(request(), tmp_path, tmp_path / "db", providers())
    store.register(task, tmp_path / "output")
    return store


def test_atomic_claim_immutable_identity_and_read_only_projection(tmp_path):
    store = stored(tmp_path)
    state = store.claim("task")
    with pytest.raises(ValueError, match="claimed"):
        store.claim("task")
    altered = copy.deepcopy(state)
    altered["task"]["providers"][0]["version"] = "different"
    with pytest.raises(ValueError, match="immutable"):
        store.save(altered, "stage/start", {})
    projection = public_tasks(store.registry)
    projection[0]["providers"][0]["version"] = "mutated"
    assert store.get("task")["task"]["providers"][0]["version"] == "1"
    state["status"] = "completed"
    store.save(state, "task/end", {"status": "completed"})
    with pytest.raises(ValueError, match="not running"):
        store.save(state, "task/end", {})
    assert [e["kind"] for e in store.events("task")] == [
        "task/registered", "task/start", "task/end"]


def test_cancel_before_claim_has_balanced_events(tmp_path):
    store = stored(tmp_path)
    store.cancel("task")
    assert store.claim("task")["status"] == "cancelled"
    assert store.events("task")[-1]["kind"] == "task/end"


def test_recovery_never_reexecutes_unknown_side_effect(tmp_path, monkeypatch):
    store = stored(tmp_path)
    state = store.claim("task")
    state["stages"].append({"id": "a", "provider": "source", "status": "running", "sim_cap": 10})
    store.save(state, "stage/start", {"id": "a"})
    with pytest.raises(ValueError, match="still alive"):
        recover(tmp_path / "db", "task")
    monkeypatch.setattr("driversi.rsi.workflow.engine.process_identity", lambda pid: None)
    result = recover(tmp_path / "db", "task")
    assert result["status"] == "outcome_unknown"
    assert result["sim_agent_steps"] == 10
    assert result["stages"][0]["cost_basis"] == "reserved_cap_unknown_usage"
    with pytest.raises(ValueError):
        recover(tmp_path / "db", "task")


def test_protected_sources_not_exposed(tmp_path):
    store = stored(tmp_path)
    state = store.claim("task")
    # Register another task already assigned a held-out source.
    task = copy.deepcopy(state["task"])
    task["id"], task["source_groups"] = "hidden", ["secret"]
    store.register(task, tmp_path / "hidden")
    with store.registry._connect() as db:
        db.execute("CREATE TABLE repair_gate_groups(log_group TEXT PRIMARY KEY, gate_id TEXT)")
        db.execute("INSERT INTO repair_gate_groups VALUES ('secret','gate')")
    assert [x["id"] for x in public_tasks(store.registry)] == ["task"]


def test_cancellation_wins_atomic_success_settlement(tmp_path):
    store = stored(tmp_path)
    state = store.claim("task")
    store.cancel("task")
    state["status"] = "completed"
    store.save(state, "task/end", {"status": "completed"})
    assert store.get("task")["status"] == "cancelled"
    assert store.events("task")[-1]["data"]["status"] == "cancelled"


def test_harness_exposure_cannot_become_model_promotion_cohort(tmp_path):
    from driversi.rsi.experiments.campaign_store import CampaignStore
    from driversi.rsi.execution.registry import harness_exposed_groups
    store = stored(tmp_path)
    task = prepare(request(), tmp_path, tmp_path / "db", providers())
    task["id"], task["source_groups"] = "exposed", ["seen-log"]
    store.register(task, tmp_path / "exposed")
    spec = {"id": "promotion", "promotion_groups": [["seen-log"]],
            "train_groups": [], "development_groups": [], "initial_training_groups": []}
    with pytest.raises(ValueError, match="DriveHarness"):
        CampaignStore(tmp_path / "db").register(spec, tmp_path / "campaigns")
    with store.registry._connect() as db:
        assert harness_exposed_groups(db) == {"seen-log"}
        db.execute("CREATE TABLE repair_gate_groups(log_group TEXT PRIMARY KEY, gate_id TEXT)")
        db.execute("INSERT INTO repair_gate_groups VALUES ('held-out','gate')")
    task["id"], task["source_groups"] = "forbidden", ["held-out"]
    with pytest.raises(ValueError, match="Protected"):
        store.register(task, tmp_path / "forbidden")
