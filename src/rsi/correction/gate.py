"""Predeclared held-out recipe comparison; independent from model promotion."""

import copy
import json
import math
import os
import socket
from pathlib import Path

from driversi.runtime.budget import write_json
from driversi.rsi.contracts import _identity, _keys
from driversi.rsi.execution.jobs import process_identity
from driversi.rsi.correction.recipes import prepare, recipe, run_prepared, verifier


def validate_gate(raw, base):
    _keys(
        raw,
        (
            "schema_version",
            "id",
            "library",
            "incumbent",
            "candidate",
            "verifier",
            "policy",
            "cases",
        ),
        "recipe gate",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported recipe gate")
    for k in ("id", "library"):
        _identity(raw[k], k)
    incumbent, candidate = recipe(raw["incumbent"]), recipe(raw["candidate"])
    if incumbent["id"] == candidate["id"]:
        raise ValueError("Gate requires two distinct recipe versions")
    verifier(raw["verifier"])
    p = raw["policy"]
    _keys(
        p,
        (
            "min_cases",
            "min_groups",
            "min_pass_gain",
            "min_progress_gain_m",
            "max_progress_regression_m",
        ),
        "recipe promotion policy",
    )
    for k in ("min_cases", "min_groups"):
        if type(p[k]) is not int or p[k] < 1:
            raise ValueError("Positive evidence minima required")
    for k in ("min_pass_gain", "min_progress_gain_m", "max_progress_regression_m"):
        if type(p[k]) not in (int, float) or not math.isfinite(p[k]) or p[k] < 0:
            raise ValueError("Finite nonnegative gate thresholds required")
    if p["min_pass_gain"] > 1 or p["min_pass_gain"] == p["min_progress_gain_m"] == 0:
        raise ValueError("Recipe promotion requires a positive improvement")
    if not isinstance(raw["cases"], list) or not raw["cases"]:
        raise ValueError("Predeclare a nonempty cohort")
    prepared = []
    for case in raw["cases"]:
        _keys(case, ("id", "case", "budget"), "gate case")
        prepared.append(
            prepare(
                {
                    "schema_version": 1,
                    **case,
                    "baseline": incumbent,
                    "recipes": [candidate],
                    "verifier": raw["verifier"],
                },
                base,
            )
        )
    keys = [
        (r["inputs"]["scene"], r["inputs"]["rollout"], r["case"]["actor"], r["case"]["anchor_tick"])
        for r in prepared
    ]
    if len({r["id"] for r in prepared}) != len(prepared) or len(set(keys)) != len(keys):
        raise ValueError("Duplicate gate case identity")
    return {**copy.deepcopy(raw), "cases": prepared}


def decide(spec, runs):
    """Pure fixed gate: all declared cases remain in the denominator."""
    p = spec["policy"]
    checks = {
        "complete_cohort": len(runs) == len(spec["cases"]),
        "minimum_cases": len(spec["cases"]) >= p["min_cases"],
        "minimum_groups": len({s["source_log"] for s in spec["cases"]}) >= p["min_groups"],
    }
    pairs = []
    for case, run in zip(spec["cases"], runs):
        by_id = {a["recipe"]["id"]: a for a in run["attempts"]}
        a, b = by_id.get(spec["incumbent"]["id"]), by_id.get(spec["candidate"]["id"])
        valid = (
            run["status"] == "completed"
            and run["spec"] == case
            and all(x and x["status"] == "completed" and x.get("metrics") for x in (a, b))
        )
        if not valid:
            checks["complete_cohort"] = False
            continue
        pairs.append((a["metrics"], b["metrics"]))
    base_pass = sum(a["engineering_pass"] for a, _ in pairs)
    new_pass = sum(b["engineering_pass"] for _, b in pairs)
    gain = (new_pass - base_pass) / len(spec["cases"])
    progress_gain = sum(b["route_progress_m"] - a["route_progress_m"] for a, b in pairs) / len(
        spec["cases"]
    )
    checks.update(
        minimum_pass_gain=gain >= p["min_pass_gain"],
        minimum_progress_gain=progress_gain >= p["min_progress_gain_m"],
        candidate_passes_all=bool(pairs) and all(b["engineering_pass"] for _, b in pairs),
        no_pass_regression=all(
            not a["engineering_pass"] or b["engineering_pass"] for a, b in pairs
        ),
        no_safety_regression=all(
            b[k] <= a[k]
            for a, b in pairs
            for k in ("collision_ticks", "contact_ticks", "offroad_ticks")
        ),
        progress_non_regression=all(
            b["route_progress_m"] >= a["route_progress_m"] - p["max_progress_regression_m"]
            for a, b in pairs
        ),
    )
    enough = all(checks[k] for k in ("complete_cohort", "minimum_cases", "minimum_groups"))
    return {
        "outcome": "promoted" if all(checks.values()) else "rejected" if enough else "inconclusive",
        "checks": checks,
        "pass_rate_gain": gain,
        "mean_route_progress_gain_m": progress_gain,
        "reasons": [k for k, v in checks.items() if not v],
        "model_promoted": False,
        "teacher_qualified": False,
        "trainable": False,
        "claim": "engineering_recipe_comparison_only",
    }


def run_gate(store, raw, output, *, base=Path(".")):
    spec = validate_gate(raw, base)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Gate output must be new")
    groups = {c["source_log"] for c in spec["cases"]}
    state = {
        "schema_version": 1,
        "kind": "repair-recipe-gate-v1",
        "spec": spec,
        "status": "running",
        "runs": [],
        "decision": None,
        "directory": str(output),
        "owner": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "start": process_identity(os.getpid()),
        },
    }
    with store.registry._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        prior = [json.loads(r[0]) for r in db.execute("SELECT payload FROM repair_runs")]
        development = [r for r in prior if r["role"] == "development"]
        if (
            groups & {r["spec"]["source_log"] for r in development}
            or groups & store.registry.protected_groups()
        ):
            raise ValueError("Gate logs overlap development or reserved evaluation evidence")
        from driversi.rsi.execution.registry import harness_exposed_groups
        exposed = harness_exposed_groups(db)
        for row in db.execute("SELECT payload FROM plans"):
            data = json.loads(row[0])["data"]
            exposed.update(data["development_groups"] + data["evaluation_groups"])
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='campaign_groups'").fetchone():
            exposed.update(r[0] for r in db.execute("SELECT log_group FROM campaign_groups"))
        if groups & exposed:
            raise ValueError("Gate logs already exposed to a model experiment")
        if not any(
            a["recipe"] == spec["candidate"]
            for r in development
            if r["status"] == "completed"
            for a in r["attempts"]
        ):
            raise ValueError("Candidate must first have development execution evidence")
        for r in (spec["incumbent"], spec["candidate"]):
            store._recipe(db, r)
        old = db.execute(
            "SELECT recipe_id FROM repair_libraries WHERE name=?", (spec["library"],)
        ).fetchone()
        if old and old[0] != spec["incumbent"]["id"]:
            raise ValueError("Stale recipe library incumbent")
        db.execute(
            "INSERT OR IGNORE INTO repair_libraries VALUES (?,?)",
            (spec["library"], spec["incumbent"]["id"]),
        )
        db.execute(
            "INSERT INTO repair_gates VALUES (?,?)",
            (spec["id"], json.dumps(state, allow_nan=False)),
        )
        for group in groups:
            db.execute("INSERT INTO repair_gate_groups VALUES (?,?)", (group, spec["id"]))
    output.mkdir(parents=True)

    def persist():
        write_json(output / "gate.json", state)
        with store.registry._connect() as db:
            db.execute(
                "UPDATE repair_gates SET payload=? WHERE id=?",
                (json.dumps(state, allow_nan=False), spec["id"]),
            )

    try:
        persist()
        for case in spec["cases"]:
            state["runs"].append(
                run_prepared(
                    store,
                    case,
                    output / case["id"],
                    role="promotion",
                    gate_id=spec["id"],
                    use_memory=False,
                )
            )
            persist()
        state["decision"] = decide(spec, state["runs"])
        with store.registry._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            incumbent = db.execute(
                "SELECT recipe_id FROM repair_libraries WHERE name=?", (spec["library"],)
            ).fetchone()[0]
            if incumbent != spec["incumbent"]["id"]:
                state["decision"].update(
                    outcome="inconclusive", reasons=["concurrent_incumbent_changed"]
                )
            elif state["decision"]["outcome"] == "promoted":
                db.execute(
                    "UPDATE repair_libraries SET recipe_id=? WHERE name=?",
                    (spec["candidate"]["id"], spec["library"]),
                )
            state["status"] = "completed"
            db.execute(
                "UPDATE repair_gates SET payload=? WHERE id=?",
                (json.dumps(state, allow_nan=False), spec["id"]),
            )
    except Exception as exc:
        state.update(status="failed", failure_reason=str(exc))
    persist()
    from html import escape

    links = "".join(
        '<li><a href="' + escape(c["id"]) + '/index.html">' + escape(c["id"]) + "</a></li>"
        for c in spec["cases"]
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>纠正流程独立验收</h1><p>仅晋级规则配方，不晋级模型或训练数据。</p><ul>'
        + links
        + "</ul><pre>"
        + escape(json.dumps(state["decision"], ensure_ascii=False, indent=2))
        + '</pre><a href="gate.json">完整验收记录</a>'
    )
    return state
