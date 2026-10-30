"""Evidence-directed RSI decisions, not a learned causal diagnosis model.

References: RoboRSI scoped attribution; GEPA cheap/full validation;
SEAL downstream learning feedback; CoCoEvolve unsuccessful-attempt backoff.
Rules below are DriveRSI choices, not reproductions of those algorithms.
"""
import math
from copy import deepcopy

from driversi.training.sampling.environment import source_log_group

COSTS = ("wall_seconds", "gpu_seconds", "score_queries")
CHECKS = ("runtime_parity", "world_validity", "task", "history_validity",
          "trajectory_admission", "visual_admission")


def _cost(value):
    if set(value) != set(COSTS) or any(type(value[k]) not in (int, float)
            or not math.isfinite(value[k]) or value[k] < 0 for k in COSTS):
        raise ValueError("Explicit finite resource budget required")
    return value


def validate_case(case):
    required = {"id", "node", "failure_class", "source_log", "split", "parent_version",
                "evaluator_version", "input_kind", "checks", "evidence"}
    if set(case) != required or any(not isinstance(case[k], str) or not case[k] for k in required - {"checks", "evidence"}):
        raise ValueError("Malformed diagnostic case")
    if case["split"] not in ("train", "development", "test") or case["input_kind"] not in ("original", "counterfactual"):
        raise ValueError("Unknown source role or input context")
    if set(case["checks"]) != set(CHECKS) or any(v not in ("pass", "fail", "unknown") for v in case["checks"].values()):
        raise ValueError("Explicit tri-state checks required")
    if not isinstance(case["evidence"], list) or not case["evidence"] or any(not isinstance(p, str) or not p for p in case["evidence"]):
        raise ValueError("Diagnostic evidence references required")
    return case


def diagnose(case):
    validate_case(case)
    c = case["checks"]
    if c["runtime_parity"] == "fail":
        return "repair_adapter", "native_computation_differs"
    if c["world_validity"] == "fail":
        return "repair_world", "platform_evidence_failed"
    if c["runtime_parity"] == "unknown" or c["world_validity"] == "unknown":
        return "probe", "failure_cause_unresolved"
    if c["task"] == "pass":
        return "retain", "no_confirmed_driving_failure"
    if c["task"] == "unknown":
        return "probe", "task_outcome_unresolved"
    if c["trajectory_admission"] != "pass":
        return ("relabel" if case["input_kind"] == "original" else "recover"), "correction_not_admitted"
    if c["history_validity"] != "pass":
        return ("repair_data" if case["input_kind"] == "original" else "render_history"), "input_not_valid"
    if c["visual_admission"] != "pass":
        return "visual_review", "visual_student_input_not_admitted"
    return "train", "admitted_correction_ready"


def route(case, protected_logs):
    action, reason = diagnose(case)
    if case["split"] != "train" or source_log_group(case["source_log"]) in {source_log_group(x) for x in protected_logs}:
        return "evaluation_only", "source_is_not_training"
    return action, reason


def validate_memory(records):
    seen = set()
    keys = {"id", "case_id", "node", "recipe", "parent_version", "evaluator_version",
            "source_logs", "training_source_logs", "evaluation_role", "failure_class", "parent_improved",
            "outcome", "paired", "matched_budget", "safety_passed", "delta", "cost", "execution"}
    for record in records:
        if set(record) != keys or any(not isinstance(record[k], str) or not record[k]
                for k in ("id", "case_id", "node", "recipe", "parent_version", "evaluator_version")):
            raise ValueError("Malformed learning memory")
        if record["id"] in seen:
            raise ValueError("Duplicate memory attempt")
        seen.add(record["id"])
        if record["outcome"] not in ("completed", "failed", "unknown"):
            raise ValueError("Unknown attempt outcome")
        if any(type(record[k]) is not bool for k in ("paired", "matched_budget", "safety_passed", "parent_improved")):
            raise ValueError("Explicit learning comparison flags required")
        if not isinstance(record["source_logs"], list) or not record["source_logs"] or any(not isinstance(x, str) or not x for x in record["source_logs"]):
            raise ValueError("Memory provenance required")
        if record["evaluation_role"] not in ("development", "train", "test", "unknown") or not isinstance(record["failure_class"], str) or not record["failure_class"]:
            raise ValueError("Memory evaluation role and failure class required")
        if not isinstance(record["training_source_logs"], list) or any(not isinstance(x, str) or not x for x in record["training_source_logs"]):
            raise ValueError("Memory training provenance required")
        _execution(record["execution"], required=False)
        delta = record["delta"]
        if delta is not None and (type(delta) not in (float, int) or not math.isfinite(delta)):
            raise ValueError("Finite measured learning delta required")
        _cost(record["cost"])
    return records


def _execution(value, *, required):
    if value is None and not required:
        return
    if (not isinstance(value, dict) or set(value) != {"mode", "model_config_sha256", "max_optimizer_steps"}
            or value["mode"] not in ("supervised", "offline_reward", "online_rl")
            or not isinstance(value["model_config_sha256"], str) or len(value["model_config_sha256"]) != 64
            or type(value["max_optimizer_steps"]) is not int or value["max_optimizer_steps"] <= 0):
        raise ValueError("Explicit native recipe execution contract required")


def allocate(cases, recipes, memory, budget, *, protected_logs, max_failed_attempts=3):
    """Reserve recommendations in supplied case order. Costs are declared caps,
    not measurements. No automatic experiment execution or promotion occurs.
    """
    _cost(budget); validate_memory(memory)
    if type(max_failed_attempts) is not int or max_failed_attempts < 1:
        raise ValueError("Positive failure-attempt limit required")
    ids = set()
    for r in recipes:
        if set(r) != {"id", "action", "node", "cost", "execution"} or any(not isinstance(r[k], str) or not r[k] for k in ("id", "action", "node")):
            raise ValueError("Malformed executable recipe choice")
        if r["id"] in ids:
            raise ValueError("Duplicate recipe")
        ids.add(r["id"]); _cost(r["cost"])
        _execution(r["execution"], required=r["action"] == "train")
    for case in cases:
        validate_case(case)
    training_pool = {source_log_group(c["source_log"]) for c in cases if c["split"] == "train"}
    remaining, decisions, case_ids = deepcopy(budget), [], set()
    protected = {source_log_group(x) for x in protected_logs}
    for case in cases:
        action, reason = route(case, protected_logs)
        if case["id"] in case_ids:
            raise ValueError("Duplicate diagnostic case")
        case_ids.add(case["id"])
        options = []
        excluded = []
        for r in recipes:
            if r["action"] != action or r["node"] != case["node"]:
                continue
            related = [m for m in memory if m["recipe"] == r["id"] and m["node"] == case["node"]
                       and m["parent_version"] == case["parent_version"] and m["evaluator_version"] == case["evaluator_version"]
                       and m["failure_class"] == case["failure_class"] and m["execution"] == r["execution"]]
            failed = [m for m in related if m["case_id"] == case["id"] and
                      (m["outcome"] != "completed" or not m["safety_passed"] or
                       (m["delta"] is not None and m["delta"] <= 0))]
            if len(failed) >= max_failed_attempts:
                excluded.append({"recipe": r["id"], "reason": "unsuccessful_attempt_cap"}); continue
            if any(r["cost"][k] > remaining[k] for k in COSTS):
                excluded.append({"recipe": r["id"], "reason": "resource_budget"}); continue
            gains = [m for m in related if m["evaluation_role"] == "development" and m["training_source_logs"]
                     and not ({source_log_group(x) for x in m["source_logs"]} & {source_log_group(x) for x in m["training_source_logs"]})
                     and m["paired"] and m["matched_budget"] and m["outcome"] == "completed"
                     and m["delta"] is not None and not
                     ({source_log_group(x) for x in m["source_logs"]} & (protected | training_pool | {source_log_group(case["source_log"])}))]
            # Safety regressions never earn positive credit, even with a high mean score.
            utility = sum(m["delta"] if m["safety_passed"] and m["parent_improved"] else min(0., m["delta"]) for m in gains) / len(gains) if gains else 0.
            fraction = sum(r["cost"][k] / budget[k] if budget[k] else 0 for k in COSTS)
            options.append((-utility, len(failed), fraction, r["id"], r, [m["id"] for m in gains]))
        options.sort(key=lambda x: x[:4])
        chosen = options[0] if options else None
        if chosen:
            r = chosen[4]
            for k in COSTS: remaining[k] -= r["cost"][k]
        decisions.append({"case_id": case["id"], "source_log": case["source_log"], "node": case["node"], "parent_version": case["parent_version"],
                          "evaluator_version": case["evaluator_version"], "action": action, "reason": reason,
                          "diagnostic_action": diagnose(case)[0], "diagnostic_reason": diagnose(case)[1],
                          "recipe": chosen[4]["id"] if chosen else None,
                          "execution": chosen[4]["execution"] if chosen else None,
                          "status": "reserved" if chosen else "no_execution_reserved",
                          "reserved_cost": chosen[4]["cost"] if chosen else {k: 0 for k in COSTS},
                          "learning_evidence": chosen[5] if chosen else [], "excluded": excluded,
                          "evidence": case["evidence"]})
    return {"schema_version": 1, "decision_rule": "evidence-routing-v1", "decisions": decisions,
            "remaining_budget": remaining, "cost_semantics": "planned_caps_not_actual_cost",
            "automatic_execution": False, "learned_improver": False, "promoted": False}
