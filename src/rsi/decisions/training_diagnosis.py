"""Route measured training attempts without attributing infrastructure to model ability."""


def diagnose_training_attempt(plan, result):
    status = result["status"]
    backend = plan["resolved"]["backend"]
    updates = result.get("consumed", {}).get("successful_optimizer_steps")
    if plan.get("schema_version") == 2 and result.get("schema_version") == 2:
        updates = result.get("updates")
    if backend is None:
        category, action, reason = (
            "capability_gap",
            "integrate_native_learner",
            "native_learning_backend_not_connected",
        )
    elif status in ("not_started", "running"):
        category, action, reason = "pending", "wait_for_evidence", "attempt_not_settled"
    elif status == "rejected" and result.get("reason") == "native_prerequisites_missing":
        category, action, reason = (
            "prerequisites",
            "prepare_prerequisites",
            "native_inputs_or_runtime_missing",
        )
    elif status in ("budget_exhausted", "timeout", "timed_out"):
        category, action, reason = (
            "budget",
            "inspect_cost_before_retry",
            "execution_budget_exhausted",
        )
    elif status != "completed":
        category, action, reason = (
            "execution_unresolved",
            "inspect_native_failure",
            "runtime_failure_is_not_model_weakness",
        )
    elif not result.get("checkpoint"):
        category, action, reason = "artifact_gap", "audit_artifacts", "candidate_checkpoint_missing"
    elif type(updates) is not int or updates <= 0:
        category, action, reason = (
            "update_unverified",
            "verify_optimizer_updates",
            "global_steps_do_not_prove_gradient_updates",
        )
    else:
        category, action, reason = (
            "update_observed",
            "independent_evaluation",
            "parameter_update_is_not_generalization_gain",
        )
    return {
        "schema_version": 1,
        "category": category,
        "action": action,
        "reason": reason,
        "model_deficiency_established": False,
        "benefit_established": False,
        "automatic_retry": False,
        "promoted": False,
    }
