"""Persisted, conservative admission accounting shared by training and evaluation."""
import json
import os
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def write_json(path: Path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class StepBudget:
    def __init__(self, path: Path, limit: int, phase: str):
        self.path = Path(path)
        self.limit = limit
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "admitted_agent_steps": 0, "valid_transitions": 0, "updates": 0}
        self.state["phase"] = phase
        write_json(self.path, self.state)

    def before_step(self, count: int):
        if type(count) is not int or count < 0:
            raise ValueError("Active agent count must be a nonnegative integer")
        total = self.state["admitted_agent_steps"] + count
        if total > self.limit:
            raise BudgetExceeded("Simulation admission budget exhausted before world advance")
        self.state["admitted_agent_steps"] = total
        write_json(self.path, self.state)

    def progress(self, **values):
        self.state.update(values)
        write_json(self.path, self.state)
