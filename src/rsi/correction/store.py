"""Recipe lineage and development-only evidence in the existing RSI SQLite ledger."""

import copy
import json
import os
import socket
from pathlib import Path

from driversi.rsi.execution.jobs import process_identity
from driversi.rsi.execution.registry import ExperimentRegistry


class RepairStore:
    def __init__(self, registry):
        self.registry = ExperimentRegistry(Path(registry))
        with self.registry._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS repair_recipes (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS repair_runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS repair_gates (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS repair_gate_groups (log_group TEXT PRIMARY KEY, gate_id TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS repair_libraries (name TEXT PRIMARY KEY, recipe_id TEXT NOT NULL)"
            )

    def _recipe(self, db, value):
        old = db.execute("SELECT payload FROM repair_recipes WHERE id=?", (value["id"],)).fetchone()
        if old and json.loads(old[0]) != value:
            raise ValueError(
                "Recipe version already names a different implementation/configuration"
            )
        db.execute(
            "INSERT OR IGNORE INTO repair_recipes VALUES (?,?)",
            (value["id"], json.dumps(value, allow_nan=False)),
        )

    def runs(self):
        with self.registry._connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute("SELECT payload FROM repair_runs ORDER BY rowid")
            ]

    def get(self, run_id):
        with self.registry._connect() as db:
            row = db.execute("SELECT payload FROM repair_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise ValueError("Unknown repair run")
        return json.loads(row[0])

    def register_run(self, spec, directory, *, role, gate_id):
        if role not in ("development", "promotion"):
            raise ValueError("Unknown repair evidence role")
        state = {
            "schema_version": 1,
            "kind": "same-state-repair-v1",
            "spec": copy.deepcopy(spec),
            "role": role,
            "gate_id": gate_id,
            "directory": str(directory),
            "status": "registered",
            "attempts": [],
            "selection_evidence": [],
            "trainable": False,
            "failure_reason": None,
            "owner": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "start": process_identity(os.getpid()),
            },
        }
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            reserved = db.execute(
                "SELECT gate_id FROM repair_gate_groups WHERE log_group=?", (spec["source_log"],)
            ).fetchone()
            if role == "development":
                if reserved or spec["source_log"] in self.registry.protected_groups():
                    raise ValueError("Protected evaluation log cannot enter repair development")
            else:
                if not reserved or reserved[0] != gate_id:
                    raise ValueError("Promotion run requires a predeclared gate cohort")
                gate = db.execute(
                    "SELECT payload FROM repair_gates WHERE id=?", (gate_id,)
                ).fetchone()
                declared = json.loads(gate[0]) if gate else {}
                if declared.get("status") != "running" or spec not in declared.get("spec", {}).get(
                    "cases", []
                ):
                    raise ValueError("Promotion run differs from its frozen gate specification")
            for r in [spec["baseline"], *spec["recipes"]]:
                self._recipe(db, r)
            db.execute(
                "INSERT INTO repair_runs VALUES (?,?)",
                (spec["id"], json.dumps(state, allow_nan=False)),
            )
        return state

    def update_run(self, state):
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = json.loads(
                db.execute(
                    "SELECT payload FROM repair_runs WHERE id=?", (state["spec"]["id"],)
                ).fetchone()[0]
            )
            if any(old[k] != state[k] for k in ("spec", "role", "gate_id", "directory", "owner")):
                raise ValueError("Run inputs and role are immutable")
            if old["status"] not in ("registered", "running"):
                raise ValueError("Terminal repair evidence cannot be overwritten")
            db.execute(
                "UPDATE repair_runs SET payload=? WHERE id=?",
                (json.dumps(state, allow_nan=False), state["spec"]["id"]),
            )

    def order(self, spec):
        protected = self.registry.protected_groups()
        excluded = protected | {spec["source_log"]}
        evidence = []
        ids = {r["id"]: r for r in spec["recipes"]}
        # Latest terminal attempt per source group/recipe prevents duplicate retries
        # from dominating cross-log evidence.
        cells = {}
        for run in self.runs():
            if (
                run["role"] != "development"
                or run["status"] != "completed"
                or run["spec"]["source_log"] in excluded
            ):
                continue
            if (
                run["spec"]["verifier"] != spec["verifier"]
                or run["spec"]["case"]["failure_type"] != spec["case"]["failure_type"]
                or run["spec"]["case"]["future_steps"] != spec["case"]["future_steps"]
            ):
                continue
            for a in run["attempts"]:
                rid = a["recipe"]["id"]
                if rid not in ids or a["recipe"] != ids[rid]:
                    continue
                cells[(run["spec"]["source_log"], rid)] = (run, a)
        stats = {}
        for (group, rid), (run, a) in cells.items():
            passed = bool(
                a.get("metrics") and a["metrics"]["engineering_pass"] and a["status"] == "completed"
            )
            stats.setdefault(rid, []).append(int(passed))
            evidence.append(
                {
                    "run_id": run["spec"]["id"],
                    "source_log": group,
                    "recipe_id": rid,
                    "engineering_pass": passed,
                    "status": a["status"],
                }
            )
        position = {r["id"]: i for i, r in enumerate(spec["recipes"])}
        ordered = sorted(
            spec["recipes"],
            key=lambda r: (
                -(sum(stats.get(r["id"], [])) + 1) / (len(stats.get(r["id"], [])) + 2),
                position[r["id"]],
            ),
        )
        return ordered, evidence

    def recover(self, run_id):
        from driversi.runtime.budget import write_json

        state = self.get(run_id)
        if state["status"] not in ("registered", "running"):
            return state
        owner = state["owner"]
        if (
            owner["host"] != socket.gethostname()
            or process_identity(owner["pid"]) == owner["start"]
        ):
            raise ValueError("Cannot recover an active or remote process")
        budget = Path(state["directory"]) / "budget.json"
        if budget.exists():
            state["admitted_agent_steps"] = json.loads(budget.read_text())["admitted_agent_steps"]
        for a in state["attempts"]:
            if a["status"] == "running":
                a.update(
                    status="interrupted", failure_reason="Executor ended before terminal settlement"
                )
        state.update(status="interrupted", failure_reason="Executor no longer running")
        self.update_run(state)
        write_json(Path(state["directory"]) / "report.json", state)
        return state

    def get_gate(self, gate_id):
        with self.registry._connect() as db:
            row = db.execute("SELECT payload FROM repair_gates WHERE id=?", (gate_id,)).fetchone()
        if not row:
            raise ValueError("Unknown repair gate")
        return json.loads(row[0])

    def recover_gate(self, gate_id):
        """Settle a dead gate conservatively; never resume or release its cohort."""
        from driversi.runtime.budget import write_json

        state = self.get_gate(gate_id)
        Path(state["directory"]).mkdir(parents=True, exist_ok=True)
        if state["status"] != "running":
            write_json(Path(state["directory"]) / "gate.json", state)
            return state
        owner = state["owner"]
        if (
            owner["host"] != socket.gethostname()
            or process_identity(owner["pid"]) == owner["start"]
        ):
            raise ValueError("Cannot recover an active or remote gate process")
        children = {r["spec"]["id"]: r for r in self.runs() if r["gate_id"] == gate_id}
        for child in children.values():
            if child["status"] in ("registered", "running"):
                self.recover(child["spec"]["id"])
        state["runs"] = [
            self.get(case["id"]) for case in state["spec"]["cases"] if case["id"] in children
        ]
        state["unstarted_cases"] = [
            c["id"] for c in state["spec"]["cases"] if c["id"] not in children
        ]
        state.update(
            status="interrupted",
            failure_reason="Gate executor no longer running",
            decision={
                "outcome": "inconclusive",
                "reasons": ["executor_interrupted"],
                "model_promoted": False,
                "teacher_qualified": False,
                "trainable": False,
            },
        )
        with self.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = json.loads(
                db.execute("SELECT payload FROM repair_gates WHERE id=?", (gate_id,)).fetchone()[0]
            )
            if old["status"] != "running":
                state = old
            else:
                if old["owner"] != owner:
                    raise ValueError("Gate owner changed during recovery")
                db.execute(
                    "UPDATE repair_gates SET payload=? WHERE id=?",
                    (json.dumps(state, allow_nan=False), gate_id),
                )
        write_json(Path(state["directory"]) / "gate.json", state)
        return state
