"""Public aliases, portable package imports and optional-dependency boundaries."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from driversi.agents.interfaces.catalog import list_agents
from tests.support.diagnostic_config import build_agent, load_inference, read_agent_config

ROOT = Path(__file__).resolve().parents[3]


def test_model_inventory_and_external_descriptor_do_not_import_optional_runtimes(tmp_path):
    code = ("import sys; import driversi_agents; from driversi.agents.interfaces.catalog import list_agents; "
            "from driversi_agents.driverl.runtime import inspect_runtime; "
            "list_agents(); "
            "assert not {'torch', 'driverl', 'nuplan'}.intersection(sys.modules)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "driverl" not in {row["name"] for row in list_agents()}




@pytest.mark.parametrize("name", ["mlp", "linear"])
def test_canonical_model_config_weights_rng_export_and_numpy_inference(name, tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    old = read_agent_config(ROOT / f"tests/fixtures/legacy/agents/{name}.yaml")
    new = {**old, "model_target": old["model_target"].replace("driversi.agents.", "driversi_agents."),
           "inference_target": old["inference_target"].replace("driversi.agents.", "driversi_agents.")}
    models = []
    for config in [old, new]:
        torch.manual_seed(73)
        model = build_agent(config)
        models.append((model, torch.get_rng_state()))
    assert torch.equal(models[0][1], models[1][1])
    for k, v in models[0][0].state_dict().items():
        assert torch.equal(v, models[1][0].state_dict()[k])
    path = tmp_path / "frozen.npz"
    models[0][0].export(path, "isolation-check")
    x = np.random.default_rng(19).normal(size=(4, 1125)).astype(np.float32)
    expected = load_inference(path, old).predict(x)
    actual = load_inference(path, new).predict(x)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    with np.load(path) as data:
        assert json.loads(str(data["metadata"]))["policy_version"] == "isolation-check"
