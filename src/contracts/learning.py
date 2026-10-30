"""Versioned structured learner boundary over the authoritative joint world.

No renderer, training framework, log future, or automatic reset belongs here.
"""
from dataclasses import asdict, dataclass

import numpy as np

from driversi.contracts.transition import Transition
from driversi.contracts.visual import frozen_array

OBSERVATION_VERSION = "puffer-current-1124-time-v1"
OBSERVATION_SIZE = 1125
ACTION_SPACE = "jerk_discrete_12"
ACTION_SIZE = 12
END_REASONS = {"collision": 1, "offroad": 2, "goal": 4, "time_budget": 8, "other_absorbing": 16, "route_support_end": 32}


@dataclass(frozen=True)
class DecisionKey:
    session_id: str
    generation: int
    tick: int

    def __post_init__(self):
        if (not isinstance(self.session_id, str) or not self.session_id
                or any(type(x) is not int or x < 0 for x in (self.generation, self.tick))):
            raise ValueError("Invalid decision identity")


@dataclass(frozen=True)
class ClosedLoopSpec:
    max_steps: int
    progress_weight: float
    schema_version: int = 1
    input_mode: str = "privileged_structured"
    observation_version: str = OBSERVATION_VERSION
    observation_size: int = OBSERVATION_SIZE
    action_space: str = ACTION_SPACE
    action_size: int = ACTION_SIZE
    dt_seconds: float = .1
    controlled_capacity: int = 32
    auto_reset: bool = False
    dynamic_population: bool = False  # Logged replay membership; policy ownership stays fixed.
    goal_semantics: str = "endpoint_absorbing"
    horizon_semantics: str = "finite_task_zero_terminal_value"
    reward_version: str = "event-plus-euclidean-endpoint-progress-v1"
    visual_policy_supported: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LearningObservation:
    key: DecisionKey
    actor_ids: tuple[str, ...]
    features: np.ndarray
    alive: np.ndarray
    episode_done: bool
    observation_version: str = OBSERVATION_VERSION
    input_mode: str = "privileged_structured"

    def __post_init__(self):
        n = len(self.actor_ids)
        if (not n or len(set(self.actor_ids)) != n
                or np.shape(self.features) != (n, OBSERVATION_SIZE)
                or not np.isfinite(self.features).all()
                or np.shape(self.alive) != (n,) or np.asarray(self.alive).dtype != bool
                or self.observation_version != OBSERVATION_VERSION
                or self.input_mode != "privileged_structured"):
            raise ValueError("Invalid structured learning observation")
        object.__setattr__(self, "features", frozen_array(self.features, np.float32))
        object.__setattr__(self, "alive", frozen_array(self.alive, bool))


@dataclass(frozen=True)
class JointDecision:
    key: DecisionKey
    actor_ids: tuple[str, ...]
    actions: np.ndarray
    policy_version: str
    log_probs: np.ndarray | None = None
    observation_version: str = OBSERVATION_VERSION
    action_space: str = ACTION_SPACE
    actor_policy_versions: tuple[str, ...] | None = None

    def __post_init__(self):
        actions = np.asarray(self.actions)
        if (not self.actor_ids or len(set(self.actor_ids)) != len(self.actor_ids)
                or actions.shape != (len(self.actor_ids),) or actions.dtype.kind not in "iu"
                or (actions < 0).any() or (actions >= ACTION_SIZE).any()
                or not isinstance(self.policy_version, str) or not self.policy_version.strip()
                or self.observation_version != OBSERVATION_VERSION or self.action_space != ACTION_SPACE):
            raise ValueError("Invalid joint decision or unsupported version/action space")
        if self.actor_policy_versions is not None:
            if (not isinstance(self.actor_policy_versions, tuple)
                    or len(self.actor_policy_versions) != len(self.actor_ids)
                    or any(not isinstance(v, str) or not v.strip() for v in self.actor_policy_versions)):
                raise ValueError("One explicit behavior policy version per actor required")
        object.__setattr__(self, "actions", frozen_array(actions, np.int64))
        if self.log_probs is not None:
            p = np.asarray(self.log_probs)
            if p.shape != actions.shape or not np.isfinite(p).all() or (p > 0).any():
                raise ValueError("Behavior log probabilities must be finite and nonpositive")
            object.__setattr__(self, "log_probs", frozen_array(p, np.float32))


@dataclass(frozen=True)
class LearningStep:
    decision: JointDecision
    observation: LearningObservation
    executed_actions: np.ndarray
    rewards: np.ndarray
    reward_terms: np.ndarray  # [actor, event / weighted declared progress]
    end_reasons: np.ndarray  # END_REASONS bitset; multiple events can coexist
    bootstrap_allowed: np.ndarray
    on_policy_eligible: np.ndarray  # provided likelihood, valid, no actual override
    transition: Transition

    def __post_init__(self):
        for name in ("executed_actions", "rewards", "reward_terms", "end_reasons",
                     "bootstrap_allowed", "on_policy_eligible"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
