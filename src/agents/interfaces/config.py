"""Explicit trusted-code agent configuration; importing this module needs no Torch."""
import importlib
import json
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.contracts.learning import (
    ACTION_SIZE,
    ACTION_SPACE,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
)
from driversi.contracts.model_capability import STRUCTURED_POLICY, require_capability

DEFAULT_AGENT = {
    "schema_version": 1, "task": "structured_actor_critic",
    "model_target": None,
    "inference_target": None,
    "observation_version": OBSERVATION_VERSION, "action_space": ACTION_SPACE,
    "input_mode": "privileged_structured", "parameters": {},
}


def normalize_agent(config=None):
    if config is None:
        raise ValueError("Explicit agent configuration required; legacy default models were retired")
    value = json.loads(json.dumps(config, allow_nan=False))
    if isinstance(value, dict) and "_target_" in value:
        if "model_target" in value:
            raise ValueError("Use only one of _target_ or model_target")
        value["model_target"] = value.pop("_target_")
    if not isinstance(value, dict) or set(value) - {"capabilities"} != set(DEFAULT_AGENT):
        raise ValueError("Agent config fields must match schema version 1")
    require_capability(value.get("capabilities", STRUCTURED_POLICY), "structured_ppo")
    for key in ("schema_version", "task", "observation_version", "action_space", "input_mode"):
        if type(value[key]) is not type(DEFAULT_AGENT[key]) or value[key] != DEFAULT_AGENT[key]:
            raise ValueError(f"Unsupported agent {key}: {value[key]!r}")
    if not isinstance(value["parameters"], dict):
        raise ValueError("Agent parameters must be a mapping")
    def plain_parameters(item):
        if isinstance(item, dict):
            if "_target_" in item:
                raise ValueError("Nested _target_ construction is not supported")
            for child in item.values():
                plain_parameters(child)
        elif isinstance(item, list):
            for child in item:
                plain_parameters(child)
    plain_parameters(value["parameters"])
    for key in ("model_target", "inference_target"):
        target = value[key]
        if isinstance(target, str) and ":" not in target and "." in target:
            module, name = target.rsplit(".", 1)
            target = value[key] = module + ":" + name
        if not isinstance(target, str) or target.count(":") != 1:
            raise ValueError(f"{key} must be module:class")
        module, name = target.split(":")
        if not name.isidentifier() or not all(p.isidentifier() for p in module.split(".")):
            raise ValueError(f"Invalid {key}")
    return value


def read_agent_config(path: Path):
    return normalize_agent(read_mapping(path))


def _target(value):
    module, name = value.split(":")
    return getattr(importlib.import_module(module), name)


def build_agent(config=None):
    config = normalize_agent(config)  # Reject an incompatible role before loading model code.
    import torch

    from driversi.agents.interfaces.base import ActorCriticAgent

    cls = _target(config["model_target"])
    if not isinstance(cls, type) or not issubclass(cls, ActorCriticAgent):
        raise TypeError("Model must implement ActorCriticAgent")
    inference_cls = _target(config["inference_target"])
    if (not isinstance(inference_cls, type) or not callable(getattr(inference_cls, "predict", None))
            or not callable(getattr(inference_cls, "actions", None))):
        raise TypeError("Inference class must expose predict and actions")
    model = cls(**config["parameters"])
    for key in ("observation_version", "action_space", "input_mode"):
        if getattr(model, key) != config[key]:
            raise ValueError(f"Model {key} differs from agent config")
    # This is shape/contract validation, not evidence of driving quality. Preserve
    # sampling RNG and mode so default construction remains numerically compatible.
    mode = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            logits, values = model(torch.zeros(2, OBSERVATION_SIZE))
        if (logits.shape != (2, ACTION_SIZE) or values.shape != (2,)
                or not torch.isfinite(logits).all() or not torch.isfinite(values).all()):
            raise ValueError("Agent forward must return finite [batch,12] logits and [batch] values")
    finally:
        model.train(mode)
    return model


def load_inference(path: Path, config=None):
    """Only explicit trusted config selects code; NPZ metadata never selects imports."""
    config = normalize_agent(config)
    policy = _target(config["inference_target"])(Path(path))
    if not callable(getattr(policy, "predict", None)) or not callable(getattr(policy, "actions", None)):
        raise TypeError("Frozen inference requires predict and actions")
    for key in ("observation_version", "action_space", "input_mode"):
        if policy.metadata.get(key) != config[key]:
            raise ValueError(f"Frozen model {key} differs from agent config")
    if not isinstance(policy.metadata.get("policy_version"), str) or not policy.metadata["policy_version"]:
        raise ValueError("Frozen inference requires a policy version")
    return policy
