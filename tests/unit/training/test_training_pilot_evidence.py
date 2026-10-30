
import pytest

from driversi.rsi.decisions.training_diagnosis import diagnose_training_attempt






@pytest.mark.parametrize(
    "backend,status,updates,checkpoint,action",
    [
        (None, "rejected", None, None, "integrate_native_learner"),
        ({}, "rejected", None, None, "prepare_prerequisites"),
        ({}, "running", None, None, "wait_for_evidence"),
        ({}, "failed", None, None, "inspect_native_failure"),
        ({}, "budget_exhausted", None, None, "inspect_cost_before_retry"),
        ({}, "completed", None, {}, "audit_artifacts"),
        ({}, "completed", None, {"path": "p"}, "verify_optimizer_updates"),
        ({}, "completed", 0, {"path": "p"}, "verify_optimizer_updates"),
        ({}, "completed", True, {"path": "p"}, "verify_optimizer_updates"),
        ({}, "completed", 2, {"path": "p"}, "independent_evaluation"),
    ],
)
def test_training_triage_does_not_diagnose_model_weakness(
    backend, status, updates, checkpoint, action
):
    d = diagnose_training_attempt(
        {"resolved": {"backend": backend}},
        {
            "status": status,
            "reason": "native_prerequisites_missing",
            "consumed": {"successful_optimizer_steps": updates},
            "checkpoint": checkpoint,
        },
    )
    assert d["action"] == action
    assert not d["model_deficiency_established"] and not d["benefit_established"]
    assert not d["automatic_retry"] and not d["promoted"]
