import json

import pytest

from driversi.runtime.budget import BudgetExceeded, StepBudget


def test_admission_precedes_operation_and_survives_stage_switch(tmp_path):
    path = tmp_path / "progress.json"
    training = StepBudget(path, 5, "training")
    training.before_step(3)
    training.progress(valid_transitions=3, updates=1)
    with pytest.raises(BudgetExceeded):
        training.before_step(3)
    assert json.loads(path.read_text())["admitted_agent_steps"] == 3
    evaluation = StepBudget(path, 8, "evaluating")
    evaluation.before_step(4)
    assert evaluation.state["admitted_agent_steps"] == 7
    assert evaluation.state["updates"] == 1


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_admission(tmp_path, value):
    guard = StepBudget(tmp_path / "p.json", 5, "training")
    with pytest.raises(ValueError):
        guard.before_step(value)
