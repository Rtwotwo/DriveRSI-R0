import copy

import pytest

from driversi.rsi.workflow.engine import prepare
from driversi.rsi.workflow.outcomes import failure, receipt
from driversi.rsi.workflow.providers import builtin_providers
from tests.unit.rsi.test_harness import providers, request


def test_preparation_binds_complete_provider_contract(tmp_path):
    task = prepare(request(), tmp_path, tmp_path / "db", providers())
    assert task["steps"][1]["provider_contract"]["inputs"] == {"data": "evidence"}
    assert task["steps"][1]["provider_contract"]["output"] == "diagnostic"


def test_student_output_requires_actual_checkpoint_files():
    provider = builtin_providers().get("student_imitation")
    result = {"status": "completed", "artifacts": {"report": "report.json"}}
    with pytest.raises(ValueError, match="required artifacts"):
        provider.validate_output(result)
    result["artifacts"].update(parent="p.npz", candidate="c.npz", agent="agent.json")
    provider.validate_output(result)
    before = copy.deepcopy(provider.contract())
    provider.contract()["inputs"]["dataset"] = "mutated"
    assert provider.contract() == before


def test_only_failed_validation_receipt_proves_zero_execution(tmp_path):
    receipt(tmp_path, "task", "stage", "validating")
    assert not failure(tmp_path, "task", "stage")["execution_not_entered"]
    receipt(tmp_path, "task", "stage", "failed", ValueError("changed input"), "validating")
    detail = failure(tmp_path, "task", "stage")
    assert detail["execution_not_entered"] and detail["category"] == "precondition_failed"
    assert not failure(tmp_path, "other", "stage")["execution_not_entered"]
    receipt(tmp_path, "task", "stage", "failed", ValueError("bad output"), "publishing")
    detail = failure(tmp_path, "task", "stage")
    assert not detail["execution_not_entered"] and detail["category"] == "output_validation"


def test_malformed_receipt_is_unknown(tmp_path):
    from driversi.rsi.workflow.outcomes import failure
    (tmp_path / "s.receipt.json").write_text("[]")
    assert not failure(tmp_path, "task", "s")["execution_not_entered"]
