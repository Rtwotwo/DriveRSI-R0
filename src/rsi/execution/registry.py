"""Immutable plans and terminal attempts, with costs retained even for failed runs."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from driversi.rsi.contracts import BUDGET_FIELDS, validate_attempt, validate_plan


def harness_exposed_groups(db):
    """Declarations count as exposure, including failed/cancelled attempts."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='harness_tasks'").fetchone():
        return set()
    return {group for row in db.execute("SELECT payload FROM harness_tasks")
            for group in json.loads(row[0])["task"].get("source_groups", [])}


class ExperimentRegistry:
    def __init__(self, path: Path, read_only=False):
        self.path = path.resolve()
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise ValueError("Ledger does not exist")
            with self._connect() as db:
                db.execute("SELECT payload FROM plans LIMIT 1")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as db:
                db.execute("CREATE TABLE IF NOT EXISTS plans "
                           "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS attempts "
                           "(plan_id TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,"
                           " PRIMARY KEY(plan_id,id), FOREIGN KEY(plan_id) REFERENCES plans(id))")

    @contextmanager
    def _connect(self):
        path = self.path.as_uri() + "?mode=ro" if self.read_only else str(self.path)
        connection = sqlite3.connect(path, uri=self.read_only, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register(self, plan):
        if self.read_only:
            raise ValueError("Read-only ledger")
        plan = validate_plan(plan)
        with self._connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='repair_gate_groups'").fetchone():
                protected = {r[0] for r in db.execute("SELECT log_group FROM repair_gate_groups")}
                if protected & set(plan["data"]["development_groups"] + plan["data"]["evaluation_groups"]):
                    raise ValueError("Recipe promotion logs cannot enter generic experiments")
            try:
                db.execute("INSERT INTO plans VALUES (?,?)",
                           (plan["id"], json.dumps(plan, allow_nan=False)))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Plan ID already exists; use a new versioned plan") from exc
        return plan["id"]

    def record(self, attempt):
        if self.read_only:
            raise ValueError("Read-only ledger")
        attempt = validate_attempt(attempt)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='jobs'").fetchone():
                if db.execute("SELECT 1 FROM jobs WHERE plan_id=? AND arm=? "
                              "AND status IN ('queued','running')",
                              (attempt["plan_id"], attempt["arm"])).fetchone():
                    raise ValueError("Arm budget reserved by an active executor job")
            plan_row = db.execute("SELECT payload FROM plans WHERE id=?",
                                  (attempt["plan_id"],)).fetchone()
            if plan_row is None:
                raise ValueError("Unknown plan")
            plan = json.loads(plan_row[0])
            metric = plan["protocol"]["metric"]
            if attempt["status"] == "completed" and metric not in attempt["metrics"]:
                raise ValueError("Completed attempt lacks protocol metric")
            try:
                db.execute("INSERT INTO attempts VALUES (?,?,?)",
                           (attempt["plan_id"], attempt["id"],
                            json.dumps(attempt, allow_nan=False)))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Attempt already recorded; retries require a new ID") from exc
        return attempt["id"]

    def protected_groups(self):
        """Model and recipe promotion cohorts stay out of development evidence."""
        with self._connect() as db:
            groups = set()
            for table, where in (("campaign_groups", " WHERE role='promotion'"),
                                 ("repair_gate_groups", "")):
                if db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                    groups.update(r[0] for r in db.execute("SELECT log_group FROM " + table + where))
            return groups

    def public_summary(self):
        with self._connect() as db:
            plans = [json.loads(row[0]) for row in db.execute("SELECT payload FROM plans ORDER BY id")]
            output = []
            protected = self.protected_groups()
            for plan in plans:
                if (plan["protocol"]["role"] != "development" or protected & set(
                        plan["data"]["development_groups"] + plan["data"]["evaluation_groups"])):
                    continue
                attempts = [json.loads(row[0]) for row in db.execute(
                    "SELECT payload FROM attempts WHERE plan_id=? ORDER BY id", (plan["id"],))]
                costs = {arm: {key: sum(a["cost"][key] for a in attempts if a["arm"] == arm)
                               for key in BUDGET_FIELDS} for arm in ("baseline", "candidate")}
                over = {arm: [k for k in BUDGET_FIELDS if costs[arm][k] > plan["budget_per_arm"][k]]
                        for arm in costs}
                jobs = []
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='jobs'").fetchone():
                    jobs = [json.loads(r[0]) for r in db.execute(
                        "SELECT payload FROM jobs WHERE plan_id=? ORDER BY id", (plan["id"],))]
                output.append({"plan": plan, "jobs": jobs, "attempts": attempts, "cost_by_arm": costs,
                               "over_budget": over, "decision": "not_adjudicated",
                               "evidence_status": "operator_reported_not_independently_verified"})
            from driversi.rsi.experiments.campaign_store import public_campaigns
            campaigns = public_campaigns(db)
        return {"experiments": output, "campaigns": campaigns, "status": "development_only",
                "rsi_claim": "not_evaluated", "lineage": "declared_groups_only"}
