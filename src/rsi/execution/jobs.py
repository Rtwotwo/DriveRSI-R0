"""Atomic arm reservation, progress and settlement; no implicit retries."""
import copy
import json
import os
import socket
import time
from pathlib import Path

from driversi.rsi.contracts import BUDGET_FIELDS, validate_attempt
from driversi.rsi.execution.registry import ExperimentRegistry

ACTIVE = ("queued", "running")


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        parts = text[text.rfind(")") + 2:].split()
        return None if parts[0] == "Z" else parts[19]
    except OSError:
        return None


class JobStore:
    def __init__(self, path):
        self.registry = ExperimentRegistry(Path(path))
        with self.registry._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL,"
                       " arm TEXT NOT NULL, status TEXT NOT NULL, cancel INTEGER NOT NULL,"
                       " payload TEXT NOT NULL, FOREIGN KEY(plan_id) REFERENCES plans(id))")
            db.execute("CREATE TABLE IF NOT EXISTS execution_protocols "
                       "(plan_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def plan(self, plan_id):
        with self.registry._connect() as db:
            row = db.execute("SELECT payload FROM plans WHERE id=?", (plan_id,)).fetchone()
        if not row:
            raise ValueError("Unknown registered plan")
        return json.loads(row[0])

    def claim(self, job, output_root):
        job = copy.deepcopy(job)
        directory = Path(output_root).resolve() / job["id"]
        if directory.exists():
            raise ValueError("Job output directory must be new")
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM jobs WHERE plan_id=? AND arm=? "
                          "AND status IN ('queued','running')", (job["plan_id"], job["arm"])).fetchone():
                raise ValueError("Arm already has an active reservation")
            if db.execute("SELECT 1 FROM jobs WHERE id=?", (job["id"],)).fetchone():
                raise ValueError("Job ID already used")
            if db.execute("SELECT 1 FROM attempts WHERE plan_id=? AND id=?",
                          (job["plan_id"], job["id"])).fetchone():
                raise ValueError("Attempt ID already used")
            plan = self.plan(job["plan_id"])
            attempts = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM attempts WHERE plan_id=?", (job["plan_id"],))]
            remaining = {key: plan["budget_per_arm"][key] - sum(
                a["cost"][key] for a in attempts if a["arm"] == job["arm"]) for key in BUDGET_FIELDS}
            if any(value < 0 for value in remaining.values()):
                raise ValueError("An arm resource budget is already exhausted")
            if remaining["trials"] < 1 or remaining["cpu_seconds"] <= 0:
                raise ValueError("No trial/CPU budget remaining")
            if job["device"] == "cuda" and remaining["gpu_seconds"] <= 0:
                raise ValueError("No GPU budget remaining")
            if job["max_train_agent_steps"] + job["evaluation_max_agent_steps"] > remaining["sim_agent_steps"]:
                raise ValueError("Simulation budget must reserve both training cap and evaluation upper bound")
            job["reserved"] = remaining
            binding = {"eval_sources": [s for s in job["sources"] if s["path"] in job["eval_paths"]],
                       "evaluator": job["evaluator_version"], "task_set": job["task_set"]}
            existing = db.execute("SELECT payload FROM execution_protocols WHERE plan_id=?",
                                  (job["plan_id"],)).fetchone()
            if existing and json.loads(existing[0]) != binding:
                raise ValueError("Independent evaluation protocol already bound to different inputs")
            if not existing:
                db.execute("INSERT INTO execution_protocols VALUES (?,?)",
                           (job["plan_id"], json.dumps(binding)))
            for row in db.execute("SELECT payload FROM jobs WHERE plan_id=?", (job["plan_id"],)):
                prior = json.loads(row[0])["job"]
                if prior["initial_model"] == job["initial_model"]:
                    def weights_binding(item):
                        return [s for s in item["sources"] if s["path"] == item["initial_weights"]]
                    if (any(prior[k] != job[k] for k in ("agent", "seed", "initial_weights"))
                            or weights_binding(prior) != weights_binding(job)):
                        raise ValueError("Same initial-model reference requires same initialization")
            state = {"id": job["id"], "job": job, "directory": str(directory),
                     "status": "queued", "phase": "admitted", "started_at": time.time(),
                     "supervisor": {"pid": os.getpid(), "start": process_identity(os.getpid()),
                                    "host": socket.gethostname()},
                     "cost": {k: 0 for k in BUDGET_FIELDS}, "progress": {},
                     "cost_basis": "process_accounting_and_conservative_step_admission",
                     "events": [], "evaluation": None, "failure_reason": None}
            db.execute("INSERT INTO jobs VALUES (?,?,?,'queued',0,?)",
                       (job["id"], job["plan_id"], job["arm"], json.dumps(state)))
        return state

    def get(self, job_id):
        with self.registry._connect() as db:
            row = db.execute("SELECT payload,status,cancel FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError("Unknown job")
        state = json.loads(row[0])
        state["status"], state["cancel_requested"] = row[1], bool(row[2])
        return state

    def update(self, state):
        with self.registry._connect() as db:
            result = db.execute("UPDATE jobs SET status=?,payload=? WHERE id=? AND status IN ('queued','running')",
                                (state["status"], json.dumps(state, allow_nan=False), state["id"]))
            if result.rowcount != 1:
                raise ValueError("Job is already terminal")

    def cancel(self, job_id):
        with self.registry._connect() as db:
            result = db.execute("UPDATE jobs SET cancel=1 WHERE id=? AND status IN ('queued','running')",
                                (job_id,))
        if not result.rowcount:
            raise ValueError("Job absent or already terminal")

    def settle(self, state, status, reason=None):
        if status not in ("completed", "failed", "cancelled", "budget_exhausted"):
            raise ValueError("Unknown terminal status")
        state = copy.deepcopy(state)
        job = state["job"]
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT status,cancel FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if not active or active[0] not in ACTIVE:
                raise ValueError("Job already settled")
            if active[1] and status == "completed":
                status, reason = "cancelled", "operator_cancelled_before_settlement"
            attempt_status = "completed" if status == "completed" else (
                "cancelled" if status == "cancelled" else "failed")
            evaluation = state["evaluation"] or {}
            metrics = {k: v for k, v in evaluation.items() if k.endswith("_per_actor")}
            if status == "completed":
                metric = self.plan(job["plan_id"])["protocol"]["metric"]
                if metric not in metrics:
                    raise ValueError("Completed job lacks independent protocol metric")
            attempt = validate_attempt({
                "id": job["id"], "plan_id": job["plan_id"], "arm": job["arm"],
                "status": attempt_status, "cost": state["cost"], "metrics": metrics,
                "artifacts": {"directory": state["directory"]},
                "candidate_model": job["candidate_version"] if status == "completed" else None,
                "failure_reason": reason,
            })
            state.update(status=status, phase=status, failure_reason=reason, finished_at=time.time())
            db.execute("INSERT INTO attempts VALUES (?,?,?)",
                       (job["plan_id"], job["id"], json.dumps(attempt, allow_nan=False)))
            db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?",
                       (status, json.dumps(state, allow_nan=False), job["id"]))
        return state
