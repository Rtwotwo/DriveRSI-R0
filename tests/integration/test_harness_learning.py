import json

import numpy as np

from driversi.rsi.workflow.engine import register, run
from driversi.rsi.workflow.store import HarnessStore
from tests.integration.test_repair_recipes import make_request


def dataset_task(straight_scene, tmp_path):
    spec = make_request(straight_scene, tmp_path / "inputs")
    repair = tmp_path / "repair.json"
    repair.write_text(json.dumps(spec))
    return {"schema_version": 1, "id": "learning", "label": "correction learning",
        "budget": {"max_wall_seconds": 120, "max_sim_agent_steps": 400}, "steps": [
        {"id": "repair", "provider": "same_state_repair", "inputs": {}, "config": {"spec": str(repair)}, "max_wall_seconds": 40},
        {"id": "admit", "provider": "repair_admission", "inputs": {"evidence": "repair"},
         "config": {"requirement": "engineering_diagnostic"}, "max_wall_seconds": 10},
        {"id": "data", "provider": "correction_dataset", "inputs": {"repair": "repair", "admission": "admit"},
         "config": {"purpose": "structured_imitation_diagnostic", "max_sim_agent_steps": 200}, "max_wall_seconds": 20}]}


def test_actual_executed_labels_and_experience(straight_scene, tmp_path):
    raw = dataset_task(straight_scene, tmp_path)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(raw))
    registry = tmp_path / "registry.sqlite"
    register(registry, path, tmp_path / "tasks")
    state = run(registry, "learning")
    assert state["status"] == "completed", state
    data = state["stages"][-1]["result"]
    assert data["summary"]["samples"] == 20
    assert data["summary"]["observation_timing"] == "before_executed_action"
    assert not data["admission"]["trainable"]
    assert data["summary"]["input_references"]
    with np.load(data["artifacts"]["samples"]) as samples:
        assert samples["observations"].shape == (20, 1125)
        assert np.array_equal(samples["ticks"], np.arange(10, 30))
        repair = state["stages"][0]["result"]
        with np.load(repair["artifacts"]["rollout:" + data["summary"]["recipe"]]) as branch:
            assert np.array_equal(samples["actions"], branch["actions"][10:30, 0])
    memory = HarnessStore(registry).experience()
    assert len(memory) == 3
    assert HarnessStore(registry).experience(memory[0]["source_groups"]) == []
