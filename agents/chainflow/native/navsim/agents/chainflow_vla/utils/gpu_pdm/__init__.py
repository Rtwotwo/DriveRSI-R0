"""Isolated Torch backend for ChainFlow's training-time PDM scorer."""

from .ops import V1ScoreConfig, VehicleGeometry, proposals_to_state_array
from .scene import load_metric_cache, prepare_pdm_scene, prepare_pdm_scenes
from .scorer import MetricConfig, PreparedPDMScene, score_v1_gpu
from .simulator import SimulatorConfig, TorchPDMSimulator

__all__ = [
    "MetricConfig",
    "PreparedPDMScene",
    "SimulatorConfig",
    "TorchPDMSimulator",
    "V1ScoreConfig",
    "VehicleGeometry",
    "load_metric_cache",
    "prepare_pdm_scene",
    "prepare_pdm_scenes",
    "proposals_to_state_array",
    "score_v1_gpu",
]
