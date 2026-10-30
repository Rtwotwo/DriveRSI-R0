import os
import subprocess
import sys
import time

import pytest

from driversi.runtime.processes import group_members, reap_helpers, stop_group, subreaper


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")
def test_adopted_helper_cpu_and_successful_leader_cleanup(tmp_path):
    previous = subreaper(True)
    path = tmp_path / "helper.pid"
    code = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"
            "'while True: pass']); open(sys.argv[1],'w').write(str(p.pid))")
    leader = subprocess.Popen([sys.executable, "-c", code, str(path)], start_new_session=True)
    try:
        leader.wait(timeout=5)
        helper = int(path.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            members = group_members(leader.pid)
            if members.get(helper, {}).get("cpu", 0) > .03:
                break
            time.sleep(.02)
        assert members[helper]["cpu"] > .03
        assert members[helper]["parent"] == os.getpid()
        stop_group(leader)
        assert not group_members(leader.pid)
    finally:
        stop_group(leader)
        reap_helpers(leader)
        subreaper(previous)


def test_recovery_accepts_absent_pid_with_unknown_start(monkeypatch):
    from driversi.rsi.execution.executor import recover
    class Store:
        def get(self, _):
            import socket
            return {"supervisor": {"pid": 2**30, "start": "1", "host": socket.gethostname()},
                    "child": {"pid": 2**30-1, "start": None},
                    "job": {"reserved": {"trials": 3, "cpu_seconds": 10}}}
        def settle(self, state, status, reason):
            assert status == "failed"
            return state
    result = recover(Store(), "missing")
    assert result["cost"]["trials"] == 1


def test_recovery_refuses_live_helper_without_leader(monkeypatch):
    import driversi.rsi.execution.executor as executor
    class Store:
        def get(self, _):
            import socket
            return {"supervisor": {"pid": 2**30, "start": "1", "host": socket.gethostname()},
                    "child": {"pid": 2**30-1, "start": None}}
    monkeypatch.setattr(executor, "group_members", lambda _: {123: {"alive": True}})
    with pytest.raises(ValueError, match="group is still alive"):
        executor.recover(Store(), "missing")


def test_output_directory_race_cannot_touch_other_owner(monkeypatch, tmp_path):
    import json

    import driversi.rsi.execution.executor as executor
    job = {"id": "j", "reserved": {"cpu_seconds": 10, "gpu_seconds": 0, "sim_agent_steps": 10},
           "wall_seconds": 10}
    directory = tmp_path / "race"
    class Store:
        def __init__(self, _):
            pass
        def plan(self, _):
            return {}
        def claim(self, _, __):
            directory.mkdir()
            (directory / "result.json").write_text('{"other_owner": true}')
            (directory / "progress.json").write_text('{"admitted_agent_steps": 999}')
            return {"job": job, "directory": str(directory), "cost": {}, "progress": {}}
        def settle(self, state, status, reason):
            return {**state, "status": status, "failure_reason": reason}
    monkeypatch.setattr(executor, "JobStore", Store)
    monkeypatch.setattr(executor, "read_mapping", lambda _: {})
    monkeypatch.setattr(executor, "resolve_execution", lambda *_: job)
    result = executor.execute(tmp_path / "ledger", tmp_path / "spec", tmp_path)
    assert result["status"] == "failed"
    assert result["progress"] == {}
    assert json.loads((directory / "result.json").read_text()) == {"other_owner": True}


def test_final_measured_overspend_cannot_complete(monkeypatch, tmp_path):
    import json

    import driversi.rsi.execution.executor as executor
    cpu = [0.0]
    job = {"id": "j", "reserved": {"cpu_seconds": .05, "gpu_seconds": 0, "sim_agent_steps": 10},
           "wall_seconds": 10, "device": "cpu", "gpu_id": 0,
           "candidate_version": "model", "evaluator_version": "eval"}
    directory = tmp_path / "run"
    state = {"job": job, "directory": str(directory), "cost": {"sim_agent_steps": 0}, "progress": {},
             "evaluation": None, "events": []}
    class Store:
        def __init__(self, _):
            pass
        def plan(self, _):
            return {"protocol": {"metric": "goal_per_actor"}}
        def claim(self, _, __):
            return state
        def update(self, _):
            pass
        def get(self, _):
            if state["evaluation"] is not None:
                cpu[0] = .08  # Work during the last cancellation read crosses the cap.
            return {"cancel_requested": False}
        def settle(self, value, status, reason):
            return {**value, "status": status, "failure_reason": reason}
    class Worker:
        pid = 2**30
        returncode = 0
        def __init__(self, args, **kwargs):
            if "evaluating" in args:
                (directory / "evaluation-result.json").write_text(json.dumps({
                    "source_separation_checked": True, "independent_process": True,
                    "candidate_model": "model", "evaluator_version": "eval", "goal_per_actor": .2}))
        def poll(self):
            return 0
    monkeypatch.setattr(executor, "JobStore", Store)
    monkeypatch.setattr(executor, "read_mapping", lambda _: {})
    monkeypatch.setattr(executor, "resolve_execution", lambda *_: job)
    monkeypatch.setattr(executor.subprocess, "Popen", Worker)
    monkeypatch.setattr(executor.time, "process_time", lambda: cpu[0])
    monkeypatch.setattr(executor, "children_cpu", lambda: 0)
    monkeypatch.setattr(executor, "reap_helpers", lambda _: None)
    monkeypatch.setattr(executor, "group_members", lambda _: {})
    monkeypatch.setattr(executor, "stop_group", lambda _: None)
    monkeypatch.setattr(executor, "subreaper", lambda _: False)
    result = executor.execute(tmp_path / "ledger", tmp_path / "spec", tmp_path)
    assert result["status"] == "budget_exhausted"
    assert result["failure_reason"] == "cpu_seconds"
    assert result["cost"]["cpu_seconds"] == .08
