import copy
import json
from pathlib import Path

import pytest

from tests.support.diagnostic_config import DEFAULT_AGENT, build_agent, normalize_agent
from driversi.agents.visual_control.image_moment import ImageMomentPolicy
from driversi.configuration import read_mapping
from driversi.contracts.model_capability import (
    STRUCTURED_POLICY,
    VISUAL_POLICY,
    compatibility,
    inspect_capability,
    validate_capability,
)
from driversi.contracts.sampling import policy_description

ROOT = Path(__file__).resolve().parents[3]


def test_old_agent_and_visual_descriptions_remain_identical():
    assert normalize_agent() == DEFAULT_AGENT
    policy = ImageMomentPolicy()
    assert policy_description(policy) == policy.describe()
    assert compatibility(STRUCTURED_POLICY, "structured_ppo")["compatible"]
    assert compatibility(VISUAL_POLICY, "visual_sampling")["compatible"]


def test_explicit_compatible_capabilities_are_preserved_without_mutating_config():
    config = {**copy.deepcopy(DEFAULT_AGENT), "capabilities": copy.deepcopy(STRUCTURED_POLICY)}
    assert normalize_agent(config) == config
    result = validate_capability(config["capabilities"])
    result["output_schema"] = "changed"
    assert config["capabilities"]["output_schema"] != "changed"


@pytest.mark.parametrize("name,role", [("chainflow_vla", "trajectory_policy"),
                                       ("trajectory_video", "observation_generator")])
def test_external_model_examples_are_declarations_not_usable_policy_adapters(name, role):
    value = read_mapping(ROOT / f"src/config/model_capabilities/{name}.example.yaml")
    report = inspect_capability(value)
    assert report["capability"]["role"] == role
    assert all(not row["compatible"] for row in report["runtimes"])
    assert not report["model_loaded"] and not report["training_or_rendering_qualified"]


def test_world_predictor_cannot_take_authority_over_physics():
    predictor = {**VISUAL_POLICY, "role": "dynamics_predictor", "input_schema": "joint-history-actions-v1",
                 "output_schema": "joint-state-distribution-v1", "authority": "prediction_only",
                 "state_scope": "explicit_world", "output_clock": "future_sequence", "horizon_seconds": 2.}
    assert validate_capability(predictor) == predictor
    assert not compatibility(predictor, "structured_ppo")["compatible"]
    with pytest.raises(ValueError, match="privileges"):
        validate_capability({**predictor, "authority": "authoritative_world"})
    with pytest.raises(ValueError, match="action likelihood"):
        validate_capability({**predictor, "likelihood": "categorical_action"})


@pytest.mark.parametrize("change", [{"output_dt_seconds": .2, "horizon_seconds": .2},
    {"likelihood": "none"}, {"state_scope": "explicit_actor"}, {"input_schema": "rgb"}])
def test_incompatible_ppo_rejected_before_model_loading(change, monkeypatch):
    from driversi.agents.interfaces import config as module

    def never_import(*args):
        raise AssertionError("Incompatible model was imported")
    monkeypatch.setattr(module, "_target", never_import)
    value = {**DEFAULT_AGENT, "capabilities": {**STRUCTURED_POLICY, **change}}
    with pytest.raises(ValueError, match="incompatible"):
        build_agent(value)


def test_visual_runtime_rejects_video_generator_even_if_action_name_is_copied():
    class WrongRole(ImageMomentPolicy):
        def describe(self):
            return {**super().describe(), "capabilities": read_mapping(
                ROOT / "src/config/model_capabilities/trajectory_video.example.yaml")}
    with pytest.raises(ValueError, match="incompatible"):
        policy_description(WrongRole())


@pytest.mark.parametrize("change", [{"output_dt_seconds": 0}, {"horizon_seconds": float("nan")},
    {"horizon_seconds": .05}, {"horizon_seconds": .2}, {"schema_version": True}, {"extra": 1}, {"role": []}])
def test_invalid_clock_or_schema_is_rejected(change):
    with pytest.raises(ValueError):
        validate_capability({**STRUCTURED_POLICY, **change})


def test_capability_cli_does_not_import_torch_and_returns_incompatible_exit_code():
    import os
    import subprocess
    import sys

    source = (
        "import sys; from driversi.cli import main; "
        f"code=main(['legacy-agents','--manifest',{str(ROOT / 'src/config/model_capabilities/trajectory_video.example.yaml')!r},"
        "'--for-runtime','structured_ppo']); assert 'torch' not in sys.modules; sys.exit(code)")
    result = subprocess.run([sys.executable, "-B", "-c", source], text=True, capture_output=True,
                            env={k:v for k,v in os.environ.items() if k != "PYTHONPATH"})
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["capability"]["role"] == "observation_generator"
    assert not report["model_loaded"]
