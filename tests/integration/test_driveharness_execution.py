"""Real native-world and subprocess checks for the task/service boundary."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

pytest.importorskip("driversi.sim._native")
from driversi.rsi.workflow.engine import register, run
from driversi.rsi.workflow.store import HarnessStore
from driversi.rsi.execution.jobs import process_identity
from tests.integration.test_repair_recipes import make_request


def task_case(straight_scene, tmp_path, *, requirement="engineering_diagnostic", cap=60, schema=1):
    spec = make_request(straight_scene, tmp_path / "inputs")
    repair_file = tmp_path / "repair.json"
    repair_file.write_text(json.dumps(spec))
    raw = {"schema_version": schema, "id": "workflow", "label": "integration",
           "budget": {"max_wall_seconds": 90, "max_sim_agent_steps": 200},
           "steps": [{"id": "repair", "provider": "same_state_repair", "inputs": {},
                      "config": {"spec": str(repair_file)}, "max_wall_seconds": cap},
                     {"id": "admit", "provider": "repair_admission",
                      "inputs": {"evidence": "repair"}, "config": {"requirement": requirement},
                      "max_wall_seconds": 10}]}
    if schema == 2:
        raw["steps"].reverse()
    path = tmp_path / "task.json"
    path.write_text(json.dumps(raw))
    registry = tmp_path / "registry.sqlite"
    register(registry, path, tmp_path / "tasks")
    return registry


@pytest.mark.parametrize("requirement,status", [
    ("engineering_diagnostic", "completed"), ("qualified_supervision", "rejected")])
def test_actual_repair_and_independent_admission(straight_scene, tmp_path, requirement, status):
    registry = task_case(straight_scene, tmp_path, requirement=requirement)
    result = run(registry, "workflow")
    assert result["status"] == status, result
    assert result["sim_agent_steps"] == 140
    assert result["stages"][0]["result"]["summary"]["anchor_verified"]
    assert not result["stages"][-1]["result"]["admission"]["trainable"]
    events = HarnessStore(registry).events("workflow")
    assert sum(e["kind"] == "stage/start" for e in events) == 2
    assert sum(e["kind"] == "stage/end" for e in events) == 2
    assert events[-1]["kind"] == "task/end"
    with pytest.raises(ValueError, match="claimed"):
        run(registry, "workflow")


def test_provider_timeout_retains_reservation(straight_scene, tmp_path):
    registry = task_case(straight_scene, tmp_path, cap=0.001)
    result = run(registry, "workflow")
    assert result["status"] == "budget_exhausted"
    assert result["sim_agent_steps"] == 200
    assert result["stages"][0]["status"] == "outcome_unknown"
    assert len(result["stages"]) == 1


@pytest.mark.parametrize("interruption", ["cancel", "kill_supervisor"])
def test_provider_dies_with_task(straight_scene, tmp_path, interruption):
    registry = task_case(straight_scene, tmp_path)
    store = HarnessStore(registry)
    with (tmp_path / "task.log").open("wb") as log:
        proc = subprocess.Popen([sys.executable, "-B", "-m", "driversi.rsi.workflow.engine",
                                 "--registry", str(registry), "run", "workflow"],
                                stdout=log, stderr=log, start_new_session=True)
        child = None
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = store.get("workflow")
                child = state.get("child")
                if child:
                    break
                if proc.poll() is not None:
                    pytest.fail((tmp_path / "task.log").read_text())
                time.sleep(0.02)
            assert child
            if interruption == "cancel":
                store.cancel("workflow")
            else:
                os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            deadline = time.monotonic() + 8
            while process_identity(child["pid"]) == child["start"] and time.monotonic() < deadline:
                time.sleep(0.02)
            assert process_identity(child["pid"]) != child["start"]
            if interruption == "kill_supervisor":
                from driversi.rsi.workflow.engine import recover
                state = recover(registry, "workflow")
                assert state["status"] == "outcome_unknown"
            else:
                state = store.get("workflow")
                assert state["status"] == "cancelled"
            assert state["sim_agent_steps"] == 200
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            if child and process_identity(child["pid"]) == child["start"]:
                os.killpg(child["pid"], signal.SIGKILL)


def test_recovery_settles_only_owned_child(straight_scene, tmp_path, monkeypatch):
    from driversi.rsi.workflow.engine import recover
    from driversi.rsi.workflow.providers import settle_interrupted_child
    from driversi.rsi.correction.store import RepairStore
    registry = task_case(straight_scene, tmp_path)
    store = HarnessStore(registry)
    state = store.claim("workflow")
    row = state["task"]["steps"][0]
    output = tmp_path / "tasks" / "workflow" / "repair"
    output.mkdir(parents=True)
    child_store = RepairStore(registry)
    child_store.register_run(row["config"]["spec"], output, role="development", gate_id=None)
    # Another output root with the same child identity must never recover it.
    assert settle_interrupted_child(row, tmp_path / "foreign", registry)["status"] == "foreign_child_identity"
    assert child_store.get(row["config"]["spec"]["id"])["status"] == "registered"
    state["stages"].append({"id": "repair", "provider": "same_state_repair", "status": "running", "sim_cap": 200})
    store.save(state, "stage/start", {"id": "repair"})
    monkeypatch.setattr("driversi.rsi.workflow.engine.process_identity", lambda pid: None)
    monkeypatch.setattr("driversi.rsi.correction.store.process_identity", lambda pid: None)
    recovered = recover(registry, "workflow")
    assert recovered["stages"][0]["child_settlement"]["status"] == "settled"
    assert child_store.get(row["config"]["spec"]["id"])["status"] == "interrupted"
    assert recovered["sim_agent_steps"] == 200


def test_harness_exposure_cannot_enter_recipe_gate(straight_scene, tmp_path):
    from driversi.rsi.correction.gate import run_gate, validate_gate
    from driversi.rsi.correction.store import RepairStore
    from tests.integration.test_repair_recipes import gate_request
    registry = task_case(straight_scene, tmp_path)
    harness_store = HarnessStore(registry)
    repair = harness_store.get("workflow")["task"]["steps"][0]["config"]["spec"]
    store = RepairStore(registry)
    gate = gate_request(store, straight_scene, tmp_path / "gate-inputs", repair["baseline"], repair["recipes"][0])
    validated = validate_gate(gate, tmp_path)
    task = dict(harness_store.get("workflow")["task"])
    task.update(id="export-exposure", source_groups=[validated["cases"][0]["source_log"]])
    harness_store.register(task, tmp_path / "exported")
    with pytest.raises(ValueError, match="already exposed"):
        run_gate(store, gate, tmp_path / "gate")
    assert not (tmp_path / "gate").exists()


def test_unordered_v2_native_spine(straight_scene, tmp_path):
    registry = task_case(straight_scene, tmp_path, schema=2)
    result = run(registry, "workflow")
    assert result["status"] == "completed", result
    assert result["task"]["execution_order"] == ["repair", "admit"]
    assert result["stages"][-1]["result"]["summary"]["input_references"]["evidence"]
    assert (tmp_path / "tasks/workflow/report.json").is_file()


def test_rejected_branch_does_not_abort_independent_dataset(straight_scene, tmp_path):
    registry = task_case(straight_scene, tmp_path, schema=2)
    raw = json.loads((tmp_path / "task.json").read_text())
    raw["id"] = "branches"
    raw["budget"]["max_sim_agent_steps"] = 400
    repair = next(s for s in raw["steps"] if s["id"] == "repair")
    admit = next(s for s in raw["steps"] if s["id"] == "admit")
    deny = {**admit, "id": "deny", "config": {"requirement": "qualified_supervision"}}
    def dataset(identity, gate):
        return {"id": identity, "provider": "correction_dataset",
                "inputs": {"repair": "repair", "admission": gate},
                "config": {"purpose": "structured_imitation_diagnostic", "max_sim_agent_steps": 100},
                "max_wall_seconds": 15}
    raw["steps"] = [dataset("blocked_data", "deny"), deny, repair, dataset("data", "admit"), admit]
    file = tmp_path / "branches.json"
    file.write_text(json.dumps(raw))
    register(registry, file, tmp_path / "tasks")
    state = run(registry, "branches")
    assert state["status"] == "failed", state
    rows = {s["id"]: s for s in state["stages"]}
    assert rows["deny"]["status"] == "rejected"
    assert rows["blocked_data"]["status"] == "blocked"
    assert rows["blocked_data"]["charged_agent_steps"] == 0
    assert rows["blocked_data"]["blocked_by"] == {"deny": "rejected"}
    assert rows["data"]["status"] == "completed"
    assert state["sim_agent_steps"] == 200
    assert not (tmp_path / "tasks/branches/blocked_data").exists()
    experience = HarnessStore(registry).experience()
    assert next(e for e in experience if e["task_id"] == "branches" and e["stage_id"] == "blocked_data")["agent_steps"] == 0


def test_graph_timeout_blocks_remaining_nodes_without_charging(straight_scene, tmp_path):
    registry = task_case(straight_scene, tmp_path, schema=2, cap=.001)
    state = run(registry, "workflow")
    assert state["status"] == "budget_exhausted"
    assert state["sim_agent_steps"] == 200
    assert [s["status"] for s in state["stages"]] == ["outcome_unknown", "blocked"]
    assert state["stages"][-1]["charged_agent_steps"] == 0


def test_changed_provider_contract_fails_before_execution(straight_scene, tmp_path):
    registry = task_case(straight_scene, tmp_path, schema=2)
    store = HarnessStore(registry)
    # Persist a deliberate declaration mismatch to exercise the real worker.
    with store.registry._connect() as db:
        row = db.execute("SELECT payload FROM harness_tasks WHERE id='workflow'").fetchone()
        state = json.loads(row[0])
        state["task"]["steps"][0]["provider_contract"]["required_artifacts"].append("not_declared_by_provider")
        db.execute("UPDATE harness_tasks SET payload=? WHERE id='workflow'", (json.dumps(state),))
    result = run(registry, "workflow")
    assert result["status"] == "failed"
    assert result["sim_agent_steps"] == 0
    stage = result["stages"][0]
    assert stage["failure"]["category"] == "precondition_failed"
    assert stage["cost_basis"] == "verified_not_executed"
    assert result["stages"][1]["status"] == "blocked"
    assert not (tmp_path / "tasks/workflow/repair").exists()


@pytest.mark.parametrize("last,corrupt,expected", [
    ("repair", False, "failed"), ("admit", False, "completed"), ("repair", True, "outcome_unknown")])
def test_reconcile_published_result_without_reexecution(straight_scene, tmp_path, monkeypatch, last, corrupt, expected):
    from pathlib import Path

    from driversi.rsi.workflow.engine import recover
    registry = task_case(straight_scene, tmp_path, schema=2)
    complete = run(registry, "workflow")
    assert complete["status"] == "completed"
    store = HarnessStore(registry)
    stage_index = 0 if last == "repair" else 1
    state = store.get("workflow")
    state["status"] = "running"
    active = state["stages"][stage_index]
    frozen_path = Path(active["result"]["artifacts"]["report"])
    original_mtime = frozen_path.stat().st_mtime_ns
    state["stages"] = state["stages"][:stage_index] + [{
        "id": active["id"], "provider": active["provider"], "status": "running", "sim_cap": active["sim_cap"]}]
    state["sim_agent_steps"] = 0 if last == "repair" else 140
    # Fault injection: supervisor died after result publication, before ledger commit.
    with store.registry._connect() as db:
        db.execute("UPDATE harness_tasks SET status='running',payload=? WHERE id='workflow'", (json.dumps(state),))
        if last == "repair":
            db.execute("DELETE FROM harness_experience WHERE task_id='workflow'")
        else:
            db.execute("DELETE FROM harness_experience WHERE task_id='workflow' AND stage_id='admit'")
    if corrupt:
        with frozen_path.open("ab") as f:
            f.write(b"changed")
    monkeypatch.setattr("driversi.rsi.workflow.engine.process_identity", lambda pid: None)
    recovered = recover(registry, "workflow")
    assert recovered["status"] == expected, recovered
    if corrupt:
        assert recovered["sim_agent_steps"] == 200
        assert recovered["stages"][0]["cost_basis"] == "reserved_cap_unknown_usage"
    else:
        assert recovered["sim_agent_steps"] == 140
        assert recovered["stages"][stage_index]["cost_basis"] == "reconciled_published_result"
        assert frozen_path.stat().st_mtime_ns == original_mtime
        assert recovered["recovery"]["pending_not_reexecuted"] == (["admit"] if last == "repair" else [])
    with pytest.raises(ValueError, match="Only interrupted"):
        recover(registry, "workflow")


@pytest.mark.parametrize("mode", ["teardown", "nested"])
@pytest.mark.parametrize("cause", ["budget_exhausted", "cancelled"])
def test_cleanup_uncertainty_retains_cost_and_recovery_handle(straight_scene, tmp_path, monkeypatch, mode, cause):
    import driversi.rsi.workflow.engine as harness
    registry = task_case(straight_scene, tmp_path, schema=2, cap=.001)
    original_stop = harness.stop_group
    original_settle = harness._settle_child

    def stopped_but_confirmation_failed(child):
        original_stop(child)  # Never strand a real test process.
        if cause == "cancelled":
            HarnessStore(registry).cancel("workflow")
        raise RuntimeError("injected unconfirmed teardown")

    def unresolved_nested(*args):
        if cause == "cancelled":
            HarnessStore(registry).cancel("workflow")
        return {"status": "requires_explicit_child_recovery", "reason": "injected nested uncertainty"}

    if mode == "teardown":
        monkeypatch.setattr(harness, "stop_group", stopped_but_confirmation_failed)
    else:
        monkeypatch.setattr(harness, "_settle_child", unresolved_nested)
    state = run(registry, "workflow")
    assert state["status"] == "running" and state["needs_recovery"]
    assert state["interruption"]["status"] == cause
    assert state["child"]["pid"] > 0
    assert state["sim_agent_steps"] == 200
    assert [s["status"] for s in state["stages"]] == ["outcome_unknown", "blocked"]
    assert state["stages"][0]["charged_agent_steps"] == 200
    store = HarnessStore(registry)
    assert store.events("workflow")[-1]["kind"] == "task/recovery_required"
    with pytest.raises(ValueError, match="owner still alive"):
        harness.recover(registry, "workflow")
    monkeypatch.setattr(harness, "process_identity", lambda pid: None)
    monkeypatch.setattr(harness, "group_members", lambda pid: {pid: {"alive": True}})
    with pytest.raises(ValueError, match="group still alive"):
        harness.recover(registry, "workflow")
    monkeypatch.setattr(harness, "group_members", lambda pid: {})
    monkeypatch.setattr(harness, "_settle_child", lambda *args: {
        "status": "requires_explicit_child_recovery", "reason": "still unresolved"})
    with pytest.raises(ValueError, match="settlement still unconfirmed"):
        harness.recover(registry, "workflow")
    assert store.get("workflow")["sim_agent_steps"] == 200
    monkeypatch.setattr(harness, "_settle_child", original_settle)
    monkeypatch.setattr(harness, "stop_group", original_stop)
    recovered = harness.recover(registry, "workflow")
    assert recovered["status"] == cause
    assert recovered["failure_reason"] == state["interruption"]["reason"]
    assert not recovered["needs_recovery"] and "child" not in recovered
    assert recovered["sim_agent_steps"] == 200
    assert sum(e["agent_steps"] for e in store.experience()) == 200
    assert sum(e["kind"] == "stage/start" for e in store.events("workflow")) == 1
    with pytest.raises(ValueError, match="Only interrupted"):
        harness.recover(registry, "workflow")


def test_graph_unknown_status_matches_recovery_policy(straight_scene, tmp_path, monkeypatch):
    import driversi.rsi.workflow.engine as harness
    registry = task_case(straight_scene, tmp_path, schema=2)
    def unreadable(*args):
        raise ValueError("injected invalid published output")
    monkeypatch.setattr(harness, "_published_result", unreadable)
    state = harness.run(registry, "workflow")
    assert state["status"] == "outcome_unknown"
    assert state["sim_agent_steps"] == 200
    assert [s["status"] for s in state["stages"]] == ["outcome_unknown", "blocked"]
