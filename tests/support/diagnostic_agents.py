"""TEST ONLY: tiny policies for generic sampler/optimizer regressions, not supported models."""
import json

import numpy as np
from torch import nn

from driversi.agents.interfaces.base import ActorCriticAgent
from driversi.contracts.learning import ACTION_SIZE, OBSERVATION_SIZE, OBSERVATION_VERSION


class ActorCritic(ActorCriticAgent):
    def __init__(self, hidden_sizes=(128, 128)):
        super().__init__()
        if (not isinstance(hidden_sizes, (list, tuple))
                or any(type(n) is not int or n < 1 for n in hidden_sizes)):
            raise ValueError("hidden_sizes must contain positive integers")
        self.hidden_sizes = tuple(hidden_sizes)
        layers = []
        previous = OBSERVATION_SIZE
        for width in self.hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.actor = nn.Linear(previous, ACTION_SIZE)
        self.critic = nn.Linear(previous, 1)
        for layer in [*self.trunk[::2], self.actor, self.critic]:
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1.)

    def forward(self, observations):
        hidden = self.trunk(observations)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def export(self, path, version):
        metadata = {"kind": "ppo-teacher-candidate-v1", "policy_version": version,
                    "observation_version": OBSERVATION_VERSION, "action_space": self.action_space,
                    "approved_teacher": False, "input_mode": self.input_mode,
                    "architecture": {"version": 1, "hidden_sizes": list(self.hidden_sizes),
                                     "activation": "tanh"}}
        np.savez_compressed(path, metadata=json.dumps(metadata),
                            **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})


class LinearActorCritic(ActorCritic):
    """Small linear control/value baseline selected through the same trainer."""

    def __init__(self):
        super().__init__(hidden_sizes=())
