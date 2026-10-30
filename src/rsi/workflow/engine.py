"""Bounded DriveHarness workflow: frozen services, durable evidence, no implicit retries.

Reference designs: DSH workflow lifecycle and session/repair at 86a4022.
Driving validation and dataset admission stay in platform-owned providers.
"""
import argparse
import copy
import ctypes
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from driversi.agents.backends.native_process import OwnedTree
from driversi.configuration import read_mapping
from driversi.runtime.budget import BudgetExceeded, write_json
from driversi.rsi.contracts import _identity, _keys
from driversi.rsi.workflow.artifacts import seal, verify
from driversi.rsi.workflow.graph import blocked_by, compile_steps
from driversi.rsi.workflow.outcomes import failure, receipt
from driversi.rsi.workflow.providers import builtin_providers, settle_interrupted_child
from driversi.rsi.workflow.store import TERMINAL, HarnessStore
from driversi.rsi.execution.jobs import process_identity
from driversi.runtime.processes import group_members, reap_helpers, stop_group, subreaper


def prepare(raw, base, registry, providers=None):
    providers = providers or builtin_providers()
    _keys(raw, ("schema_version", "id", "label", "budget", "steps"), "Harness task")
    if type(raw["schema_version"]) is not int or raw["schema_version"] not in (1, 2):
        raise ValueError("Unknown Harness task schema")
    _identity(raw["id"], "Harness task ID")
    if not isinstance(raw["label"], str) or not raw["label"].strip():
        raise ValueError("Task label required")
    _keys(raw["budget"], ("max_wall_seconds", "max_sim_agent_steps"), "Harness budget")
    wall, steps = raw["budget"]["max_wall_seconds"], raw["budget"]["max_sim_agent_steps"]
    if (type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0
            or type(steps) is not int or steps < 0):
        raise ValueError("Invalid task budget")
    if not isinstance(raw["steps"], list) or not 1 <= len(raw["steps"]) <= 32:
        raise ValueError("One to 32 explicit task stages required")
    task = copy.deepcopy(raw)
    task["steps"], task["providers"], task["source_groups"] = [], [], []
    ordered, dependencies = compile_steps(raw["steps"], providers, allow_forward=raw["schema_version"] == 2)
    task["execution_order"] = [row["id"] for row in ordered]
    task["dependencies"] = dependencies
    outputs, reservation = {}, 0
    for row in ordered:
        _keys(row, ("id", "provider", "inputs", "config", "max_wall_seconds"), "Harness stage")
        _identity(row["id"], "stage ID")
        if row["id"] in outputs:
            raise ValueError("Duplicate stage identity")
        cap = row["max_wall_seconds"]
        if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= wall:
            raise ValueError("Stage wall cap must fit task deadline")
        provider = providers.get(row["provider"])
        if not isinstance(row["inputs"], dict) or set(row["inputs"]) != dict(provider.inputs).keys():
            raise ValueError("Provider input ports do not match")
        for port, kind in provider.inputs:
            dependency = row["inputs"][port]
            if not isinstance(dependency, str) or outputs.get(dependency) != kind:
                raise ValueError("Missing, forward or incompatible stage dependency: " + port)
        config, sim_cap = provider.prepare(copy.deepcopy(row["config"]), Path(base), Path(registry))
        if type(sim_cap) is not int or sim_cap < 0:
            raise ValueError("Provider returned invalid reservation")
        reservation += sim_cap
        task["steps"].append({**row, "config": config, "provider_version": provider.version,
                              "output_kind": provider.output, "sim_cap": sim_cap, "provider_contract": provider.contract()})
        task["providers"].append({"name": provider.name, "version": provider.version})
        outputs[row["id"]] = provider.output
        task["source_groups"].extend(config.get("source_groups", []))
        task["source_groups"].extend(config.get("parent_source_groups", []))
        if provider.name == "same_state_repair":
            task["source_groups"].append(config["spec"]["source_log"])
        if provider.name == "structured_experiment":
            from driversi.training.sampling.environment import validate_development_paths
            task["source_groups"].extend(validate_development_paths(
                config["resolved"]["train_paths"] + config["resolved"]["eval_paths"]))
        if provider.name == "navsim_export":
            from driversi.contracts.scene import Scene
            from driversi.training.sampling.environment import source_log_group
            task["source_groups"].append(source_log_group(
                Scene.load(Path(config["scene"])).metadata["source_log"]))
    if reservation > steps:
        raise ValueError("Task must reserve all stage simulation caps before starting")
    task["reserved_agent_steps"] = reservation
    task["source_groups"] = sorted(set(task["source_groups"]))
    return task


def register(registry, spec, output_root):
    spec = Path(spec).resolve()
    task = prepare(read_mapping(spec), spec.parent, registry)
    directory = Path(output_root).resolve() / task["id"]
    if directory.exists():
        raise ValueError("Task output must be new")
    return HarnessStore(Path(registry)).register(task, directory)


def _result(value, cap):
    keys = ("status", "reason", "sim_agent_steps", "artifacts", "summary", "admission")
    _keys(value, keys, "provider result")
    if value["status"] not in TERMINAL - {"outcome_unknown"}:
        raise ValueError("Invalid provider terminal state")
    if type(value["sim_agent_steps"]) is not int or not 0 <= value["sim_agent_steps"] <= cap:
        raise ValueError("Provider cost outside reservation")
    if not isinstance(value["artifacts"], dict) or not isinstance(value["summary"], dict):
        raise ValueError("Invalid provider result payload")
    if not isinstance(value["admission"], dict):
        raise ValueError("Explicit admission state required")
    json.dumps(value, allow_nan=False)
    return value


def worker(registry, task_id, stage_id):
    # Bind this provider to the owning supervisor before any work. Nested
    # experiment workers already have the same parent-death contract.
    parent = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
        raise RuntimeError("Cannot bind provider lifetime to task")
    state = HarnessStore(Path(registry)).get(task_id)
    if (os.getppid() != parent or parent != state["owner"]["pid"]
            or state["status"] != "running" or state["stages"][-1]["id"] != stage_id):
        raise ValueError("Stale task/stage identity")
    row = next(r for r in state["task"]["steps"] if r["id"] == stage_id)
    directory = Path(state["directory"])
    phase = "validating"
    receipt(directory, task_id, stage_id, phase)
    try:
        provider = builtin_providers().get(row["provider"])
        if (provider.version != row["provider_version"] or provider.output != row["output_kind"]
                or set(dict(provider.inputs)) != set(row["inputs"])
                or ("provider_contract" in row and provider.contract() != row["provider_contract"])):
            raise ValueError("Provider changed after declaration")
        results = {s["id"]: s["result"] for s in state["stages"] if s["status"] == "completed"}
        for source in row["inputs"].values():
            producer = next(s for s in state["task"]["steps"] if s["id"] == source)
            verify(results[source], task_id, producer)
        inputs = {port: copy.deepcopy(results[source]) for port, source in row["inputs"].items()}
        output = directory / stage_id
        phase = "executing"
        receipt(directory, task_id, stage_id, phase)
        value = provider.execute(row["config"], inputs, output, Path(registry))
        phase = "publishing"
        receipt(directory, task_id, stage_id, phase)
        result = _result(value, row["sim_cap"])
        provider.validate_output(result)
        result["summary"]["artifact_reference"] = seal(result, output, task_id, row)
        result["summary"]["input_references"] = {
            port: value["summary"]["artifact_reference"]["identity"] for port, value in inputs.items()}
        write_json(directory / (stage_id + ".result.json"), result)
        receipt(directory, task_id, stage_id, "published")
    except Exception as exc:
        receipt(directory, task_id, stage_id, "failed", error=exc, failed_phase=phase)
        raise


def _blocked(store, state, row, reason, dependencies=None):
    stage = {"id": row["id"], "provider": row["provider"], "status": "blocked",
             "sim_cap": row["sim_cap"], "charged_agent_steps": 0, "reason": reason,
             "blocked_by": dependencies or {}, "cost_basis": "not_executed", "elapsed_seconds": 0.0}
    state["stages"].append(stage)
    store.save(state, "stage/end", {"id": row["id"], "status": "blocked", "reason": reason})


def _published_result(state, row):
    path = Path(state["directory"]) / (row["id"] + ".result.json")
    result = _result(json.loads(path.read_text()), row["sim_cap"])
    verify(result, state["id"], row)
    required = row.get("provider_contract", {}).get("required_artifacts", [])
    if result["status"] == "completed" and any(name not in result["artifacts"] for name in required):
        raise ValueError("Published result lacks required artifacts")
    completed = {s["id"]: s["result"] for s in state["stages"] if s["status"] == "completed"}
    expected = {}
    for port, source in row["inputs"].items():
        producer = next(s for s in state["task"]["steps"] if s["id"] == source)
        ref = verify(completed[source], state["id"], producer)
        expected[port] = ref["identity"]
    if result["summary"].get("input_references") != expected:
        raise ValueError("Published result dependency lineage differs")
    return result


def _adopt(active, result, basis):
    active.update(status=result["status"], reason=result["reason"], result=result,
                  charged_agent_steps=result["sim_agent_steps"], cost_basis=basis)


def _stop_owned(child):
    """Cleanup failure is evidence to retain, never a reason to drop accounting."""
    try:
        if child is not None:
            tree = getattr(child, "_native_tree", None)
            if tree is None:
                stop_group(child)
                reap_helpers(child)
            else:
                tree.stop()
                write_json(child._native_receipt, {
                    "status": "stopped", "owned": tree.owned,
                    "scope": "native_stage_descendants", "time": time.time()})
        return {"status": "stopped"}
    except (Exception, KeyboardInterrupt) as exc:
        return {"status": "teardown_unconfirmed", "reason": str(exc), "error_type": type(exc).__name__}


def _execute_stage(store, state, row, registry, directory, task_started):
    active = {"id": row["id"], "provider": row["provider"], "status": "running", "sim_cap": row["sim_cap"]}
    state["stages"].append(active)
    store.save(state, "stage/start", {"id": row["id"], "provider": row["provider"]})
    child, fatal = None, None
    native_tree = OwnedTree() if row["provider"].startswith("native_model_") else None
    native_receipt = directory / (row["id"] + ".processes.json")
    tracked = None
    started = time.monotonic()
    remaining = state["task"]["budget"]["max_wall_seconds"] - (started - task_started)
    deadline = started + min(row["max_wall_seconds"], remaining)
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "PYTHONPATH"):
            env.pop(key, None)
        with (directory / (row["id"] + ".log")).open("wb") as log:
            child = subprocess.Popen(
                [sys.executable, "-B", "-m", "driversi.rsi.workflow.engine", "--registry",
                 str(Path(registry).resolve()), "worker", state["id"], row["id"]],
                stdout=log, stderr=log, env=env, start_new_session=True)
            if native_tree is not None:
                child._native_tree = native_tree
                child._native_receipt = native_receipt
                native_tree.attach(child)
                write_json(native_receipt, {"status": "running", "owned": native_tree.owned,
                                           "scope": "native_stage_descendants"})
                tracked = dict(native_tree.owned)
            state["child"] = {"pid": child.pid, "start": process_identity(child.pid)}
            store.save(state, "stage/spawn", {"id": row["id"], "pid": child.pid})
            while child.poll() is None:
                if native_tree is not None:
                    native_tree.capture()
                    if tracked != native_tree.owned:
                        write_json(native_receipt, {"status": "running", "owned": native_tree.owned,
                                                   "scope": "native_stage_descendants"})
                        tracked = dict(native_tree.owned)
                reap_helpers(child)
                if store.get(state["id"])["cancel_requested"]:
                    raise InterruptedError("operator_cancelled")
                if time.monotonic() >= deadline:
                    raise BudgetExceeded("stage_or_task_wall_deadline")
                time.sleep(0.05)
            reap_helpers(child)
        if child.returncode != 0:
            raise RuntimeError("provider_process_failed; inspect stage log")
        teardown = _stop_owned(child)
        if teardown["status"] != "stopped":
            raise RuntimeError("Owned child teardown unconfirmed")
        if time.monotonic() >= deadline:
            raise BudgetExceeded("stage_or_task_wall_deadline")
        _adopt(active, _published_result(state, row), "provider_reported")
    except (Exception, KeyboardInterrupt) as exc:
        teardown = _stop_owned(child)
        try:
            settlement = (_settle_child(state, active, registry) if teardown["status"] == "stopped"
                          else {"status": "requires_explicit_child_recovery", "reason": teardown["reason"]})
        except (Exception, KeyboardInterrupt) as nested:
            settlement = {"status": "requires_explicit_child_recovery", "reason": str(nested)}
        unresolved = settlement["status"] in ("foreign_child_identity", "requires_explicit_child_recovery")
        active["teardown"] = teardown
        if unresolved:
            state.update(needs_recovery=True, recovery_required_stage=row["id"])
        detail = failure(directory, state["id"], row["id"])
        active["failure"] = detail
        active["child_settlement"] = settlement
        try:
            if settlement["status"] in ("foreign_child_identity", "requires_explicit_child_recovery"):
                raise ValueError("Child settlement is unconfirmed")
            _adopt(active, _published_result(state, row), "reconciled_published_result")
        except (OSError, ValueError, KeyError, TypeError) as invalid:
            known = detail["execution_not_entered"] and not unresolved
            active.update(status="failed" if known else "outcome_unknown", reason=detail["reason"] or str(exc),
                          charged_agent_steps=0 if known else active["sim_cap"],
                          cost_basis="verified_not_executed" if known else "reserved_cap_unknown_usage",
                          reconciliation_reason=str(invalid))
        if isinstance(exc, (InterruptedError, KeyboardInterrupt)):
            fatal = "cancelled"
        elif isinstance(exc, BudgetExceeded):
            fatal = "budget_exhausted"
        elif settlement["status"] in ("foreign_child_identity", "requires_explicit_child_recovery"):
            fatal = "outcome_unknown"
    if active["status"] in ("cancelled", "budget_exhausted"):
        fatal = active["status"]
    active["elapsed_seconds"] = time.monotonic() - started
    state["sim_agent_steps"] += active["charged_agent_steps"]
    state["elapsed_seconds"] = time.monotonic() - task_started
    if not state.get("needs_recovery"):
        state.pop("child", None)
    store.save(state, "stage/end", {"id": row["id"], "status": active["status"],
                                   "reason": active.get("reason"), "charged_agent_steps": active["charged_agent_steps"]})
    return active, fatal


def run(registry, task_id, *, evidence_guard=None):
    store = HarnessStore(Path(registry))
    state = store.claim(task_id)
    if state["status"] == "cancelled":
        return state
    directory, started = Path(state["directory"]), time.monotonic()
    graph_mode = state["task"]["schema_version"] == 2
    old_subreaper = None
    try:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "task.json", state["task"])
        old_subreaper = subreaper(True)
        for row in state["task"]["steps"]:
            if evidence_guard is not None:
                evidence_guard()
            if store.get(task_id)["cancel_requested"]:
                raise InterruptedError("operator_cancelled")
            if time.monotonic() - started >= state["task"]["budget"]["max_wall_seconds"]:
                raise BudgetExceeded("task_wall_deadline")
            dependencies = blocked_by(row, state["stages"])
            if dependencies:
                _blocked(store, state, row, "dependency_unavailable", dependencies)
                continue
            active, fatal = _execute_stage(store, state, row, registry, directory, started)
            if evidence_guard is not None:
                evidence_guard()
            if fatal or (not graph_mode and active["status"] != "completed"):
                state.update(status=fatal or ("failed" if active["status"] == "outcome_unknown" else active["status"]),
                             failure_reason=active.get("reason"))
                break
        else:
            failures = [s for s in state["stages"] if s["status"] != "completed"]
            status = ("outcome_unknown" if any(s["status"] == "outcome_unknown" for s in failures) else
                      "failed" if failures else "completed")
            state.update(status=status, failure_reason="one_or_more_branches_unsuccessful" if failures else None)
        if store.get(task_id)["cancel_requested"]:
            state.update(status="cancelled", failure_reason="operator_cancelled_before_settlement")
        elif time.monotonic() - started > state["task"]["budget"]["max_wall_seconds"]:
            state.update(status="budget_exhausted", failure_reason="task_wall_deadline")
    except (Exception, KeyboardInterrupt) as exc:
        status = ("cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else
                  "budget_exhausted" if isinstance(exc, BudgetExceeded) else "failed")
        state.update(status=status, failure_reason=str(exc))
    finally:
        if old_subreaper is not None:
            subreaper(old_subreaper)
    if graph_mode:
        final_status = state["status"]
        state["status"] = "running"  # Publish all unstarted nodes before terminal settlement.
        attempted = {s["id"] for s in state["stages"]}
        for row in state["task"]["steps"]:
            if row["id"] not in attempted:
                _blocked(store, state, row, "task_stopped:" + str(state["failure_reason"]))
        state["status"] = final_status
    state["elapsed_seconds"] = time.monotonic() - started
    if state.get("needs_recovery"):
        if state["status"] in ("cancelled", "budget_exhausted"):
            state["interruption"] = {"status": state["status"], "reason": state["failure_reason"]}
        state.update(status="running", failure_reason="owned_child_settlement_unconfirmed")
        store.save(state, "task/recovery_required", {"stage": state["recovery_required_stage"],
                                                    "reason": state["failure_reason"]})
    else:
        state["finished_at"] = time.time()
        state.pop("child", None)
        store.save(state, "task/end", {"status": state["status"], "reason": state["failure_reason"]})
    if directory.exists():
        write_json(directory / "report.json", state)
    return state


def _settle_child(state, stage, registry):
    row = next(r for r in state["task"]["steps"] if r["id"] == stage["id"])
    for attempt in range(10):
        try:
            return settle_interrupted_child(row, Path(state["directory"]) / row["id"], Path(registry))
        except ValueError as exc:
            if attempt == 9:
                return {"status": "requires_explicit_child_recovery", "reason": str(exc)}
            time.sleep(0.05)


def recover(registry, task_id):
    store = HarnessStore(Path(registry))
    state = store.get(task_id)
    if state["status"] != "running":
        raise ValueError("Only interrupted running tasks need recovery")
    owner = state["owner"]
    if owner["host"] != socket.gethostname():
        raise ValueError("Recover on original execution host")
    current = process_identity(owner["pid"])
    if current is not None and (owner["start"] is None or current == owner["start"]):
        raise ValueError("Task owner still alive; cancel instead")
    child = state.get("child")
    if child and any(m["alive"] for m in group_members(child["pid"]).values()):
        raise ValueError("Task process group still alive; stop owned task before recovery")
    # A dead supervisor does not prove detached native ranks are gone. Only a
    # confirmed descendant teardown releases the reservation automatically.
    for stage in state["stages"]:
        if stage["provider"].startswith("native_model_") and stage["status"] != "blocked":
            process_file = Path(state["directory"]) / (stage["id"] + ".processes.json")
            process_record = json.loads(process_file.read_text()) if process_file.is_file() else None
            if process_record is None or process_record.get("status") != "stopped":
                raise ValueError("Native descendants require explicit resource recovery; teardown unconfirmed")
            for pid, start in process_record.get("owned", {}).items():
                if process_identity(int(pid)) == str(start):
                    raise ValueError("Native descendant still alive; reservation retained")
    if state.get("needs_recovery"):
        unresolved = next(s for s in state["stages"] if s["id"] == state["recovery_required_stage"])
        settlement = _settle_child(state, unresolved, registry)
        if settlement["status"] in ("foreign_child_identity", "requires_explicit_child_recovery"):
            raise ValueError("Nested child settlement still unconfirmed; reservation retained")
        unresolved["child_settlement"] = settlement
        state["needs_recovery"] = False
    if state["stages"] and state["stages"][-1]["status"] == "running":
        stage = state["stages"][-1]
        row = next(r for r in state["task"]["steps"] if r["id"] == stage["id"])
        settlement = _settle_child(state, stage, registry)
        stage["child_settlement"] = settlement
        try:
            if settlement["status"] in ("foreign_child_identity", "requires_explicit_child_recovery"):
                raise ValueError("Child settlement is unconfirmed")
            _adopt(stage, _published_result(state, row), "reconciled_published_result")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if settlement["status"] in ("foreign_child_identity", "requires_explicit_child_recovery"):
                raise ValueError("Nested child settlement still unconfirmed; reservation retained") from exc
            detail = failure(Path(state["directory"]), task_id, row["id"])
            known = detail["execution_not_entered"]
            stage.update(status="failed" if known else "outcome_unknown", reason=detail["reason"] or "interrupted_without_valid_result",
                         failure=detail, charged_agent_steps=0 if known else stage["sim_cap"],
                         cost_basis="verified_not_executed" if known else "reserved_cap_unknown_usage",
                         reconciliation_reason=str(exc))
        state["sim_agent_steps"] += stage["charged_agent_steps"]
        state.pop("child", None)
        store.save(state, "stage/end", {"id": stage["id"], "status": stage["status"], "recovery": True})
    attempted = {s["id"] for s in state["stages"]}
    pending = [r for r in state["task"]["steps"] if r["id"] not in attempted]
    if state["task"]["schema_version"] == 2:
        for row in pending:
            _blocked(store, state, row, "interrupted_not_executed")
    statuses = [s["status"] for s in state["stages"]]
    status = ("outcome_unknown" if "outcome_unknown" in statuses else
              "failed" if pending or any(s != "completed" for s in statuses) else "completed")
    interruption = state.get("interruption", {})
    if interruption.get("status") in ("cancelled", "budget_exhausted"):
        status = interruption["status"]
    if state.get("cancel_requested"):
        status = "cancelled"
    reason = (interruption.get("reason") if interruption.get("status") == status else
              "operator_cancelled" if status == "cancelled" else "recovered_without_automatic_retry")
    state.update(status=status, failure_reason=None if status == "completed" else reason)
    state["finished_at"] = time.time()
    state.pop("child", None)
    state["recovery"] = {"pending_not_reexecuted": [r["id"] for r in pending], "time": time.time()}
    store.save(state, "task/end", {"status": status, "recovery": True})
    if Path(state["directory"]).exists():
        write_json(Path(state["directory"]) / "report.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("providers")
    sub.add_parser("experience")
    reg = sub.add_parser("register")
    reg.add_argument("file", type=Path)
    reg.add_argument("--output-root", type=Path, required=True)
    for name in ("run", "status", "events", "cancel", "recover"):
        sub.add_parser(name).add_argument("task_id")
    w = sub.add_parser("worker")
    w.add_argument("task_id")
    w.add_argument("stage_id")
    args = parser.parse_args(argv)
    if args.command == "experience":
        result = HarnessStore(args.registry).experience()
    elif args.command == "providers":
        result = builtin_providers().describe()
    elif args.command == "register":
        result = register(args.registry, args.file, args.output_root)
    elif args.command == "worker":
        worker(args.registry, args.task_id, args.stage_id)
        return 0
    elif args.command == "run":
        result = run(args.registry, args.task_id)
    elif args.command == "recover":
        result = recover(args.registry, args.task_id)
    else:
        store = HarnessStore(args.registry)
        if args.command == "cancel":
            store.cancel(args.task_id)
        result = store.events(args.task_id) if args.command == "events" else store.get(args.task_id)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if isinstance(result, dict) and (
        result.get("needs_recovery") or result.get("status") in TERMINAL - {"completed"}) else 0


if __name__ == "__main__":
    raise SystemExit(main())
