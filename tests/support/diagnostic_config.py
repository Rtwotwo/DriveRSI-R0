"""Explicit test-only model configuration for generic framework coverage."""
from driversi.agents.interfaces import config as _native
from driversi.agents.interfaces.config import read_agent_config
DEFAULT_AGENT = {**_native.DEFAULT_AGENT, "model_target": "tests.support.diagnostic_agents:ActorCritic",
    "inference_target": "driversi.training.legacy.frozen:NpzPolicy", "parameters": {"hidden_sizes": [128,128]}}
def normalize_agent(config=None):
    return _native.normalize_agent(DEFAULT_AGENT if config is None else config)
def build_agent(config=None):
    return _native.build_agent(normalize_agent(config))
def load_inference(path, config=None):
    return _native.load_inference(path, normalize_agent(config))
