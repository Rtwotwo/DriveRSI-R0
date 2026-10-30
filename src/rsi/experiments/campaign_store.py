"""Campaign lease, protected log roles and atomic research-incumbent promotion."""

import copy
import json
import os
import socket
import time
from pathlib import Path

from driversi.rsi.contracts import BUDGET_FIELDS
from driversi.rsi.execution.jobs import JobStore, process_identity
from driversi.rsi.decisions.promotion import decide
from driversi.rsi.execution.registry import harness_exposed_groups


class CampaignStore:
    def __init__(self, path):
        self.jobs = JobStore(path)
        self.registry = self.jobs.registry
        with self.registry._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY, cancel INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS campaign_groups (campaign TEXT NOT NULL, log_group TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(campaign,log_group,role))"
            )

    def register(self, spec, output_root):
        directory = str(Path(output_root).resolve() / spec["id"])
        if Path(directory).exists():
            raise ValueError("New campaign output directory required")
        gates = set().union(*map(set, spec["promotion_groups"]))
        develop = set(
            spec["train_groups"] + spec["development_groups"] + spec["initial_training_groups"]
        )
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM campaigns WHERE id=?", (spec["id"],)).fetchone():
                raise ValueError("Campaign is already registered")
            for group, role in db.execute("SELECT log_group,role FROM campaign_groups"):
                if group in gates or group in develop and role == "promotion":
                    raise ValueError(
                        "Protected or already exposed campaign logs cannot change roles"
                    )
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='repair_gate_groups'").fetchone():
                repair_holdout = {r[0] for r in db.execute("SELECT log_group FROM repair_gate_groups")}
                exposed = {json.loads(r[0])["spec"]["source_log"] for r in db.execute("SELECT payload FROM repair_runs")}
                if (develop | gates) & repair_holdout or gates & exposed:
                    raise ValueError("Recipe evidence logs cannot change evaluation roles")
            if gates & harness_exposed_groups(db):
                raise ValueError("Gate logs were already exposed by DriveHarness")
            for row in db.execute("SELECT payload FROM plans"):
                plan = json.loads(row[0])
                if gates & set(
                    plan["data"]["development_groups"] + plan["data"]["evaluation_groups"]
                ):
                    raise ValueError("Gate logs were already exposed in this experiment ledger")
            state = {
                "id": spec["id"],
                "spec": spec,
                "directory": directory,
                "status": "registered",
                "phase": "registered",
                "history": [],
                "current": None,
                "job_ids": [],
                "incumbent": {"version": spec["initial_model"], "weights": spec["initial_weights"]},
                "cost": dict.fromkeys(BUDGET_FIELDS, 0),
                "elapsed_seconds": 0,
                "owner": None,
                "failure_reason": None,
            }
            db.execute("INSERT INTO campaigns VALUES (?,0,?)", (spec["id"], json.dumps(state)))
            for role, values in (("development", develop), ("promotion", gates)):
                db.executemany(
                    "INSERT INTO campaign_groups VALUES (?,?,?)",
                    [(spec["id"], g, role) for g in values],
                )
        return state

    def get(self, campaign_id):
        with self.registry._connect() as db:
            row = db.execute(
                "SELECT payload,cancel FROM campaigns WHERE id=?", (campaign_id,)
            ).fetchone()
        if not row:
            raise ValueError("Unknown campaign")
        state = json.loads(row[0])
        state["cancel_requested"] = bool(row[1])
        return state

    def acquire(self, campaign_id):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self.get(campaign_id)
            if state["status"] != "registered" or state["cancel_requested"]:
                raise ValueError("Campaign cannot start twice; no implicit retry")
            state.update(
                status="running",
                phase="starting",
                started_at=time.time(),
                owner={
                    "pid": os.getpid(),
                    "start": process_identity(os.getpid()),
                    "host": socket.gethostname(),
                },
            )
            db.execute(
                "UPDATE campaigns SET payload=? WHERE id=?", (json.dumps(state), campaign_id)
            )
        return state

    def update(self, state):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = self.get(state["id"])
            if old["status"] != "running" or old["spec"] != state["spec"]:
                raise ValueError("Frozen/terminal campaign cannot be overwritten")
            db.execute(
                "UPDATE campaigns SET payload=? WHERE id=?",
                (json.dumps(state, allow_nan=False), state["id"]),
            )

    def cancel(self, campaign_id):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self.get(campaign_id)
            if state["status"] not in ("registered", "running"):
                return
            if state["status"] == "registered":
                state.update(
                    status="cancelled", phase="cancelled", failure_reason="operator_cancelled"
                )
            db.execute(
                "UPDATE campaigns SET cancel=1,payload=? WHERE id=?",
                (json.dumps(state), campaign_id),
            )

    def close_round(self, campaign_id):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self.get(campaign_id)
            if state["status"] != "running" or not state["current"]:
                raise ValueError("No pending round")
            if state["cancel_requested"]:
                raise InterruptedError("operator_cancelled_before_promotion")
            else:
                spec, current = state["spec"], state["current"]
                if (
                    any(state["cost"][k] > spec["budget"][k] for k in BUDGET_FIELDS)
                    or state["elapsed_seconds"] > spec["wall_seconds"]
                ):
                    raise ValueError("Campaign budget does not permit promotion")
                gate = self.jobs.get(current["gate_job"])
                train = self.jobs.get(current["training_job"])
                if gate["status"] != "completed" or train["status"] != "completed":
                    raise ValueError("Both training and gate must be completed")
                if (
                    train["job"]["tool"] != "structured_ppo"
                    or train["job"]["agent"] != spec["agent"]
                    or gate["job"]["agent"] != spec["agent"]
                    or current["candidate"]["weights"]
                    != str(Path(train["directory"]) / "model-weights.pt")
                    or train["job"]["initial_weights"] != state["incumbent"]["weights"]
                    or gate["job"]["tool"] != "structured_gate"
                    or gate["job"]["initial_weights"] != current["candidate"]["weights"]
                    or gate["job"]["reference_weights"] != state["incumbent"]["weights"]
                    or current["candidate"]["version"] != train["job"]["candidate_version"]
                ):
                    raise ValueError("Promotion artifact lineage mismatch")
                i = current["index"]
                result = decide(
                    spec["gate"],
                    gate["evaluation"],
                    spec["promotion_slots"][i],
                    spec["promotion_groups"][i],
                    state["incumbent"]["version"],
                )
                current["decision"] = result
                if result["outcome"] == "promoted":
                    state["incumbent"] = copy.deepcopy(current["candidate"])
                state["history"].append(current)
                state["current"] = None
                state["phase"] = "round_finished"
            db.execute(
                "UPDATE campaigns SET payload=? WHERE id=?",
                (json.dumps(state, allow_nan=False), campaign_id),
            )
        return state


def public_campaigns(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='campaigns'").fetchone():
        return []
    output = []
    for row in db.execute("SELECT payload FROM campaigns ORDER BY id"):
        state = json.loads(row[0])
        spec = state["spec"]

        def visible_round(item):
            if not item:
                return None
            value = {
                k: item[k]
                for k in (
                    "index",
                    "parent",
                    "mining_job",
                    "proposal",
                    "training_job",
                    "candidate",
                    "gate_job",
                    "decision",
                )
                if k in item
            }
            value["stage_progress"] = {}
            for stage in ("mining", "training", "gate"):
                jid = item.get(stage + "_job")
                record = db.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()
                if record:
                    job = json.loads(record[0])
                    value["stage_progress"][stage] = {
                        "status": job["status"], "phase": job["phase"],
                        "cost": job["cost"], "progress": {k: v for k, v in job["progress"].items()
                            if k in ("valid_transitions", "updates", "batches", "evaluated_scenes",
                                     "admitted_agent_steps", "phase")},
                        "failure_reason": ("gate_failed; inspect private campaign records" if stage == "gate"
                            and job["failure_reason"] else job["failure_reason"]),
                    }
            return value

        job_cost = dict.fromkeys(BUDGET_FIELDS, 0)
        for jid in state["job_ids"]:
            record = db.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()
            if record:
                for key, val in json.loads(record[0])["cost"].items():
                    job_cost[key] += val
        visible_cost = {key: max(state["cost"][key], job_cost[key]) for key in BUDGET_FIELDS}
        output.append(
            {
                "id": state["id"],
                "status": state["status"],
                "phase": state["phase"],
                "rounds_limit": spec["rounds"],
                "incumbent": state["incumbent"]["version"],
                "improver": spec["improver"]["name"],
                "gate_policy": spec["gate"],
                "cost": visible_cost,
                "budget": spec["budget"],
                "elapsed_seconds": state["elapsed_seconds"],
                "failure_reason": (state["status"] + "; inspect private campaign-status for diagnostics"
                    if state["failure_reason"] else None),
                "current": visible_round(state["current"]),
                "history": [visible_round(x) for x in state["history"]],
                "scene_ids": [
                    Path(p).stem for p in spec["train_paths"] + spec["development_paths"]
                ],
                "scope": "research_incumbent_only",
                "approved_teacher": False,
                "recursive_improver_learning": False,
            }
        )
    return output
