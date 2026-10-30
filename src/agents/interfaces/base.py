"""Trainable stateless structured-policy contract, independent of PPO's loss."""
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from driversi.contracts.learning import ACTION_SPACE, OBSERVATION_VERSION


class ActorCriticAgent(nn.Module, ABC):
    """One frozen set of weights serves all actors; episode state is not stored here.

    Recurrent/visual/trajectory agents need their own compatible collector contract.
    Frozen exports must expose metadata, predict(observations) and actions through
    the inference class selected explicitly by the operator's agent configuration.
    """

    observation_version = OBSERVATION_VERSION
    action_space = ACTION_SPACE
    input_mode = "privileged_structured"

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return categorical logits [batch,12] and values [batch], with gradients."""

    @abstractmethod
    def export(self, path: Path, version: str) -> None:
        """Save non-pickled frozen parameters and explicit policy/observation metadata."""
