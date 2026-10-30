"""NAVSIM augmentation selection; recommendations never confer data admission.

Development evidence identifies capability slices. Only separately observed
training-source cases may become synthesis tasks. Native metrics stay unchanged.
"""

from copy import deepcopy

from driversi.rsi.decisions.learning_decisions import COSTS, _cost, route, validate_case
from driversi.training.sampling.environment import source_log_group

GENERATORS = {
    "drivetto_logged": {"mode": "relabel", "available": True},
    "simscale_rule": {"mode": "new_state", "available": True},
    "drivezero_navsim": {
        "mode": "new_state",
        "available": False,
        "blocker": "native_navsim_history_and_branch_execution_not_validated",
    },
    "learned_self_play": {
        "mode": "new_state",
        "available": False,
        "blocker": "multi_policy_native_rollout_not_connected",
    },
}
NEW_STATE_FAILURES = {"interaction", "recovery", "distribution_shift", "occlusion"}


def source_roles(train_logs, development_logs, protected_logs):
    def groups(values):
        if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
            raise ValueError("Explicit source log lists required")
        return {source_log_group(x) for x in values}

    train, dev, protected = map(groups, (train_logs, development_logs, protected_logs))
    if not train or not dev or train & (dev | protected) or dev & protected:
        raise ValueError("NAVSIM training, development and protected logs must be disjoint")
    return train, dev, protected


def choose(
    cases,
    *,
    train_logs,
    development_logs,
    protected_logs,
    budget,
    costs,
    advice=None,
    capability_feedback=None,
):
    """Budget a finite intervention queue. Checks are evidence-bound declarations,
    not inferred causal truth. LLM advice can reorder eligible cases only.
    """
    train, dev, protected = source_roles(train_logs, development_logs, protected_logs)
    _cost(budget)
    if set(costs) != {"relabel", "new_state"}:
        raise ValueError("Declare both generation budgets")
    for c in costs.values():
        _cost(c)
        if not c["wall_seconds"] or not c["score_queries"]:
            raise ValueError("Generation must reserve execution and scoring cost")
    from driversi.rsi.decisions.capability_feedback import priorities, public_summary

    capability_priority = priorities(capability_feedback) if capability_feedback is not None else {}
    ids, rows = set(), []
    for case in cases:
        validate_case(case)
        if (
            capability_feedback is not None
            and case["parent_version"] != capability_feedback["checkpoint"]
        ):
            raise ValueError("Evaluation feedback and candidate student differ")
        if case["id"] in ids:
            raise ValueError("Duplicate augmentation case")
        ids.add(case["id"])
        group = source_log_group(case["source_log"])
        if case["split"] == "test" or group in protected:
            raise ValueError("Protected/test outcomes cannot inform development selection")
        if case["split"] == "train" and group not in train:
            raise ValueError("Mining case is outside the declared training pool")
        if case["split"] == "development" and group not in dev:
            raise ValueError("Development evidence is outside its declared pool")
        action, reason = route(case, list(dev | protected))
        curriculum = (
            case["split"] == "train"
            and action == "retain"
            and case["failure_class"] in capability_priority
        )
        if curriculum:
            action, reason = "recover", "capability_matched_active_curriculum_hypothesis"
        mode = None
        if action in ("relabel", "recover"):
            # A shifted state requires a continuous arrival prefix and new images.
            mode = (
                "new_state"
                if (
                    curriculum
                    or case["input_kind"] == "counterfactual"
                    or case["failure_class"] in NEW_STATE_FAILURES
                )
                else "relabel"
            )
        generator = {"new_state": "simscale_rule", "relabel": "drivetto_logged"}.get(mode)
        rows.append(
            {
                "case_id": case["id"],
                "source_log": case["source_log"],
                "source_group": group,
                "parent_version": case["parent_version"],
                "evaluator_version": case["evaluator_version"],
                "evidence": case["evidence"],
                "failure_class": case["failure_class"],
                "active_curriculum_seed": curriculum,
                "capability_priority": capability_priority.get(case["failure_class"], 0.0),
                "action": action,
                "reason": reason,
                "mode": mode,
                "generator": generator,
                "reserved_cost": {k: 0 for k in COSTS},
                "status": "not_selected",
                "trainable": False,
                "required": (
                    [
                        "continuous_arrival",
                        "same_anchor_teacher",
                        "counterfactual_score",
                        "history_and_future_gs",
                        "visual_admission",
                    ]
                    if mode == "new_state"
                    else ["same_original_anchor", "independent_logged_score"]
                    if mode
                    else []
                ),
                "model_cause_established": False,
            }
        )
    advice_report = (
        validate_advice(advice, rows)
        if advice is not None
        else {"status": "disabled", "accepted_order": [], "rejected": []}
    )
    rank = {x: i for i, x in enumerate(advice_report["accepted_order"])}
    remaining = deepcopy(budget)
    # Keep returned evidence order stable; advice affects reservation priority only.
    for row in sorted(
        rows,
        key=lambda x: (rank.get(x["case_id"], len(rank)), -x["capability_priority"], x["case_id"]),
    ):
        if row["mode"] is None:
            continue
        cost = costs[row["mode"]]
        if any(cost[k] > remaining[k] for k in COSTS):
            row["status"] = "budget_deferred"
            continue
        row["reserved_cost"] = deepcopy(cost)
        row["status"] = "reserved"
        for k in COSTS:
            remaining[k] -= cost[k]
    return {
        "capability_feedback": public_summary(capability_feedback)
        if capability_feedback is not None
        else None,
        "schema_version": 1,
        "kind": "navsim_augmentation_plan",
        "decisions": rows,
        "remaining_budget": remaining,
        "advice": advice_report,
        "generators": deepcopy(GENERATORS),
        "trainable": False,
        "promoted": False,
        "data_source": "navsim",
        "nuplan_role": "structured_analysis_only",
        "future_rgb_role": "supervision_not_current_policy_input",
        "cost_semantics": "reserved_caps_not_actual_spend",
        "automatic_training": False,
        "evidence_scope": "bound_checks_not_automatic_causal_diagnosis",
    }


def validate_advice(advice, rows):
    if (
        not isinstance(advice, dict)
        or set(advice) != {"suggestions"}
        or not isinstance(advice["suggestions"], list)
    ):
        raise ValueError("LLM must return only structured suggestions")
    if len(advice["suggestions"]) > len(rows):
        raise ValueError("Too many LLM suggestions")
    known = {x["case_id"]: x for x in rows}
    accepted, rejected, seen = [], [], set()
    for s in advice["suggestions"]:
        if not isinstance(s, dict) or set(s) != {
            "case_id",
            "mode",
            "generator",
            "reason",
            "evidence",
        }:
            raise ValueError("No executable code, thresholds or source changes in LLM advice")
        if not isinstance(s["reason"], str) or not 1 <= len(s["reason"]) <= 2000:
            raise ValueError("Bounded LLM rationale required")
        row = known.get(s["case_id"])
        why = None
        if row is None or s["case_id"] in seen:
            why = "unknown_or_duplicate_case"
        elif row["mode"] is None:
            why = "case_requires_diagnosis_or_is_not_training"
        elif s["mode"] != row["mode"] or s["generator"] != row["generator"]:
            why = "unsupported_or_incompatible_intervention"
        elif (
            not isinstance(s["evidence"], list)
            or not s["evidence"]
            or any(x not in row["evidence"] for x in s["evidence"])
        ):
            why = "unbound_evidence"
        if why:
            rejected.append({"case_id": s.get("case_id"), "reason": why})
        else:
            accepted.append(s["case_id"])
        seen.add(s["case_id"])
    return {
        "status": "validated_untrusted_suggestions",
        "accepted_order": accepted,
        "rejected": rejected,
        "suggestions": advice["suggestions"],
    }
