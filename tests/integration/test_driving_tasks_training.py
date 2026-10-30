# ruff: noqa: E402
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import Route
from driversi.data.scenes.scene_store import source_inventory
from driversi.data.datasets.world_model import (
    STATE_VERSION,
    export_sequences,
    iter_joint_sequences,
    load_sequence,
)
from driversi.training.sampling.collector import Collector, rollout_nbytes


class ConstantPolicy(torch.nn.Module):
    def forward(self, x):
        logits = torch.full((len(x), 12), -100.0, device=x.device)
        logits[:, 7] = 0.0
        return logits, torch.zeros(len(x), device=x.device)


@pytest.mark.parametrize("workers", [0, 2])
def test_partial_support_end_preserved_through_collector_and_sequences(
    straight_scene, tmp_path, workers
):
    scene = bind_driving_tasks(
        straight_scene, {"a": Route([[0, 0], [5, 0]]), "b": Route([[0, 8], [50, 8]])}
    )
    path = tmp_path / "scene.npz"
    scene.save(path)
    collector = Collector([path], num_worlds=2, workers=workers)
    try:
        data = collector.collect(ConstantPolicy(), 35)
        if isinstance(data, tuple):
            data = data[0]
        assert data["route_support_end"].any()
        assert data["valid"][15, 1] and not data["valid"][15, 0]
        assert data["episode"][15, 1] == 0
        assert data["episode"][30, 1] == 1
        expected = rollout_nbytes(35, 2, route_tasks=True)
        assert sum(a.nbytes for a in data.values() if isinstance(a, np.ndarray)) == expected
        run = tmp_path / "run"
        (run / "rollouts").mkdir(parents=True)
        meta = {
            **source_inventory([str(path)], 0.02),
            "joint_state_version": STATE_VERSION,
            "dt_seconds": 0.1,
            "policy_version": "constant-v1",
        }
        np.savez_compressed(run / "rollouts/batch-000001.npz", metadata=json.dumps(meta), **data)
        sequences = list(iter_joint_sequences(run, context_steps=2, prediction_steps=4, stride=1))
        assert any(s.route_support_end.any() for s in sequences)
        assert any(s.metadata["start_tick"] > 10 and s.metadata["episode"] == 0 for s in sequences)
        out = tmp_path / "dataset"
        export_sequences(run, out, context_steps=2, prediction_steps=4, stride=1)
        restored = load_sequence(out, 0)
        assert "route_support_end" in restored.training_targets()
        assert "route_support_end" not in restored.model_inputs()
        assert "driving_tasks" in restored.model_inputs()["scene"]["metadata"]
    finally:
        collector.close()
