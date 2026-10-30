"""Durable task-scoped state and observation-only events in the RSI ledger."""
import copy
import json
import os
import socket
import time

from driversi.rsi.execution.jobs import process_identity
from driversi.rsi.execution.registry import ExperimentRegistry

TERMINAL = {"completed", "rejected", "failed", "cancelled", "budget_exhausted", "outcome_unknown"}


class HarnessStore:
    def __init__(self, path):
        self.registry = ExperimentRegistry(path)
        with self.registry._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS harness_tasks "
                       "(id TEXT PRIMARY KEY, status TEXT NOT NULL, cancel INTEGER NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS harness_events "
                       "(task_id TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL, "
                       "PRIMARY KEY(task_id,seq), FOREIGN KEY(task_id) REFERENCES harness_tasks(id))")

            db.execute("CREATE TABLE IF NOT EXISTS harness_experience "
                       "(task_id TEXT NOT NULL, stage_id TEXT NOT NULL, payload TEXT NOT NULL, "
                       "PRIMARY KEY(task_id,stage_id))")

    def experience(self, exclude_groups=()):
        """Recent 500 stage outcomes, then exclude whole-task source groups."""
        excluded = set(exclude_groups) | self.registry.protected_groups()
        with self.registry._connect() as db:
            rows = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM harness_experience ORDER BY rowid DESC LIMIT 500")]
        return [r for r in rows if not excluded.intersection(r["source_groups"])]

    def _event(self, db, task_id, kind, data):
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM harness_events WHERE task_id=?",
                         (task_id,)).fetchone()[0]
        event = {"seq": seq, "task_id": task_id, "kind": kind, "time": time.time(), "data": data}
        db.execute("INSERT INTO harness_events VALUES (?,?,?)",
                   (task_id, seq, json.dumps(event, allow_nan=False)))

    def register(self, task, directory):
        state = {"schema_version": 1, "id": task["id"], "task": copy.deepcopy(task),
                 "directory": str(directory), "status": "registered", "stages": [],
                 "sim_agent_steps": 0, "failure_reason": None, "owner": None}
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if set(task.get("source_groups", [])) & self.registry.protected_groups():
                raise ValueError("Protected logs cannot enter DriveHarness development")
            if db.execute("SELECT 1 FROM harness_tasks WHERE id=?", (task["id"],)).fetchone():
                raise ValueError("Task ID already exists; retries require a new identity")
            db.execute("INSERT INTO harness_tasks VALUES (?,'registered',0,?)",
                       (task["id"], json.dumps(state, allow_nan=False)))
            self._event(db, task["id"], "task/registered", {"providers": task["providers"]})
        return state

    def get(self, task_id):
        with self.registry._connect() as db:
            row = db.execute("SELECT payload,status,cancel FROM harness_tasks WHERE id=?",
                             (task_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown Harness task")
        state = json.loads(row[0])
        state["status"], state["cancel_requested"] = row[1], bool(row[2])
        return state

    def claim(self, task_id):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload,status,cancel FROM harness_tasks WHERE id=?",
                             (task_id,)).fetchone()
            if row is None or row[1] != "registered":
                raise ValueError("Task is already claimed or terminal")
            state = json.loads(row[0])
            state["status"] = "cancelled" if row[2] else "running"
            state["owner"] = {"host": socket.gethostname(), "pid": os.getpid(),
                              "start": process_identity(os.getpid())}
            state["started_at"] = time.time()
            db.execute("UPDATE harness_tasks SET status=?,payload=? WHERE id=?",
                       (state["status"], json.dumps(state), task_id))
            self._event(db, task_id, "task/start", {"status": state["status"]})
            if row[2]:
                self._event(db, task_id, "task/end", {"status": "cancelled"})
        return state

    def save(self, state, event, data):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT payload,status,cancel FROM harness_tasks WHERE id=?",
                             (state["id"],)).fetchone()
            if old is None or old[1] != "running":
                raise ValueError("Task is not running")
            previous = json.loads(old[0])
            for key in ("task", "directory", "owner"):
                if state[key] != previous[key]:
                    raise ValueError("Task identity, provider bindings and owner are immutable")
            if event == "task/end" and old[2] and state["status"] == "completed":
                state.update(status="cancelled", failure_reason="operator_cancelled_before_settlement")
                data = {"status": "cancelled", "reason": state["failure_reason"]}
            db.execute("UPDATE harness_tasks SET status=?,payload=? WHERE id=?",
                       (state["status"], json.dumps(state, allow_nan=False), state["id"]))
            self._event(db, state["id"], event, data)
            if event == "stage/end" and state["stages"]:
                stage = state["stages"][-1]
                experience = {"task_id": state["id"], "stage_id": stage["id"],
                    "provider": stage["provider"], "status": stage["status"],
                    "reason": stage.get("reason"), "source_groups": state["task"].get("source_groups", []),
                    "cost_basis": stage.get("cost_basis", "reserved_cap_unknown_usage"),
                    "agent_steps": stage.get("charged_agent_steps", stage.get("result", {}).get("sim_agent_steps", stage["sim_cap"])),
                    "blocked_by": stage.get("blocked_by", {}),
                    "failure": stage.get("failure"),
                    "summary": stage.get("result", {}).get("summary", {}),
                    "admission": stage.get("result", {}).get("admission", {}),
                    "reuse": "diagnostic_only_no_automatic_promotion"}
                db.execute("INSERT INTO harness_experience VALUES (?,?,?)",
                    (state["id"], stage["id"], json.dumps(experience, allow_nan=False)))


    def cancel(self, task_id):
        with self.registry._connect() as db:
            cursor = db.execute("UPDATE harness_tasks SET cancel=1 WHERE id=? "
                                "AND status IN ('registered','running')", (task_id,))
            if cursor.rowcount != 1:
                raise ValueError("Task absent or terminal")

    def events(self, task_id):
        self.get(task_id)
        with self.registry._connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM harness_events WHERE task_id=? ORDER BY seq", (task_id,))]


def public_tasks(registry, task_ids=None):
    """Read-only projection; UI cannot acquire a live task or change provider state."""
    with registry._connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='harness_tasks'").fetchone():
            return []
        if task_ids is None:
            rows = db.execute("SELECT payload FROM harness_tasks ORDER BY rowid DESC LIMIT 100")
        elif task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            rows = db.execute("SELECT payload FROM harness_tasks WHERE id IN (" + placeholders + ")", tuple(task_ids))
        else:
            rows = []
        states = [json.loads(row[0]) for row in rows]
    protected = registry.protected_groups()
    result = []
    for state in states:
        if protected & set(state["task"].get("source_groups", [])):
            continue
        result.append({
            "id": state["id"], "label": state["task"]["label"], "status": state["status"],
            "needs_recovery": state.get("needs_recovery", False),
            "interruption": state.get("interruption"),
            "failure_reason": state["failure_reason"], "sim_agent_steps": state["sim_agent_steps"],
            "elapsed_seconds": state.get("elapsed_seconds", 0),
            "providers": state["task"]["providers"],
            "schema_version": state["task"]["schema_version"],
            "plan": [{"id": row["id"], "provider": row["provider"],
                      "dependencies": sorted(set(row["inputs"].values())),
                      "status": next((s["status"] for s in state["stages"] if s["id"] == row["id"]), "pending")}
                     for row in state["task"]["steps"]],
            "stages": [{"id": s["id"], "provider": s["provider"], "status": s["status"],
                        "reason": s.get("reason"), "blocked_by": s.get("blocked_by", {}),
                        "failure": s.get("failure"), "cost_basis": s.get("cost_basis"),
                        "charged_agent_steps": s.get("charged_agent_steps", s.get("result", {}).get("sim_agent_steps")),
                        "elapsed_seconds": s.get("elapsed_seconds"), "child_settlement": s.get("child_settlement"), "summary": s.get("result", {}).get("summary"),
                        "admission": s.get("result", {}).get("admission")}
                       for s in state["stages"]],
        })
    return result
