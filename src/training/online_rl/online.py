"""Synchronous, native-action actor/learner boundary; no built-in RL algorithm.

An actor supplies native observations, actions, likelihoods and recurrent state.
Transport actions/likelihoods are detached NumPy-compatible numeric arrays;
observations and recurrent memory must support deepcopy. A native adapter owns
GPU/autograd detachment without changing probability semantics.
A learner implements its own loss/value targets. Collection does not reset the
physical world at chunk boundaries or activate an unreviewed candidate.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol

import numpy as np

from driversi.training.sampling.environment import source_log_group


@dataclass(frozen=True)
class Decision:
    action: Any
    log_probability: Any
    policy_version: str
    memory: Any
    learner_context: Any = None  # native value/distribution context, never synthesized


@dataclass(frozen=True)
class Step:
    observation: Any  # actual terminal observation, NOT an auto-reset observation
    executed_action: Any
    reward: float
    terminated: bool
    truncated: bool
    platform_valid: bool
    intervention: str | None = None


@dataclass(frozen=True)
class Transition:
    observation: Any
    memory_before: Any
    decision: Decision
    step: Step


@dataclass(frozen=True)
class Batch:
    policy_version: str
    likelihood_contract: str
    source_log: str
    transitions: tuple[Transition, ...]
    boundary_observation: Any
    boundary_memory: Any
    reasons: tuple[str, ...]
    elapsed_seconds: float

    @property
    def admissible(self):
        return bool(self.transitions) and not self.reasons

    def receipt(self):
        last = self.transitions[-1].step if self.transitions else None
        return {"schema_version": 1, "training_mode": "online_rl",
                "policy_version": self.policy_version, "likelihood_contract": self.likelihood_contract,
                "source_log": self.source_log, "environment_steps": len(self.transitions),
                "elapsed_seconds": self.elapsed_seconds, "admissible": self.admissible,
                "reasons": list(self.reasons), "terminated": last.terminated if last else False,
                "truncated": last.truncated if last else False,
                "chunk_boundary": bool(last and not last.terminated and not last.truncated),
                "bootstrap_required": bool(last and not last.terminated),
                "promoted": False}


class Actor(Protocol):
    policy_version: str
    likelihood_contract: str
    def act(self, observation: Any, memory: Any) -> Decision: ...


class Environment(Protocol):
    source_log: str
    split: str
    def observation(self) -> Any: ...
    def step(self, action: Any) -> Step: ...


class Learner(Protocol):
    parent_version: str
    likelihood_contract: str
    def update(self, batch: Batch, *, max_optimizer_steps: int) -> dict: ...


def _finite(value):
    try:
        x = np.asarray(value)
        return x.size > 0 and x.dtype.kind in "fiu" and bool(np.isfinite(x).all())
    except (TypeError, ValueError, RuntimeError):
        return False


def _same_action(a, b):
    return _finite(a) and _finite(b) and np.array_equal(np.asarray(a), np.asarray(b))


class RolloutSession:
    """Persistent world/history/memory for one frozen actor; caller owns reset.

    The environment adapter MUST disable auto-reset and label every external
    intervention. This protocol only admits synchronous on-policy batches.
    """
    def __init__(self, environment: Environment, actor: Actor, *, memory,
                 protected_logs):
        if environment.split != "train" or not environment.source_log:
            raise ValueError("Online learning requires an explicit training source")
        if source_log_group(environment.source_log) in {source_log_group(x) for x in protected_logs}:
            raise ValueError("Protected source cannot enter online learning")
        if not actor.policy_version or not actor.likelihood_contract:
            raise ValueError("Native policy version and likelihood semantics required")
        self.environment, self.actor = environment, actor
        self.version, self.contract = actor.policy_version, actor.likelihood_contract
        self.source_log = environment.source_log
        self.observation = deepcopy(environment.observation())
        self.memory, self.ended = deepcopy(memory), False
        self.environment_step_calls = 0
        self.confirmed_transitions = 0
        self.unresolved_step = False

    def collect(self, *, max_environment_steps: int, max_wall_seconds: float) -> Batch:
        if type(max_environment_steps) is not int or max_environment_steps <= 0:
            raise ValueError("Positive bounded environment steps required")
        if type(max_wall_seconds) not in (int, float) or not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
            raise ValueError("Positive finite wall budget required")
        if self.unresolved_step:
            raise ValueError("Previous environment step outcome unknown; explicit reconciliation required")
        if self.ended:
            raise ValueError("Episode ended; explicitly create a new session")
        start, records, reasons = monotonic(), [], []
        for _ in range(max_environment_steps):
            if monotonic() - start >= max_wall_seconds:
                break
            if (self.actor.policy_version != self.version or self.actor.likelihood_contract != self.contract
                    or self.environment.source_log != self.source_log or self.environment.split != "train"):
                raise ValueError("Frozen rollout context changed")
            before, memory = deepcopy(self.observation), deepcopy(self.memory)
            decision = deepcopy(self.actor.act(deepcopy(before), deepcopy(memory)))
            if not isinstance(decision, Decision) or decision.policy_version != self.version:
                raise ValueError("Decision policy differs from frozen actor")
            if self.actor.policy_version != self.version or self.actor.likelihood_contract != self.contract:
                raise ValueError("Frozen actor changed during inference")
            if not _finite(decision.action) or not _finite(decision.log_probability):
                raise ValueError("Native action and finite sampling likelihood required")
            self.environment_step_calls += 1
            self.unresolved_step = True
            step = deepcopy(self.environment.step(deepcopy(decision.action)))
            if not isinstance(step, Step) or any(type(x) is not bool for x in (step.terminated, step.truncated, step.platform_valid)):
                raise ValueError("Explicit transition flags required")
            if step.terminated and step.truncated:
                raise ValueError("Terminal and truncated are distinct")
            if step.observation is None or not _finite(step.reward) or np.asarray(step.reward).ndim != 0:
                raise ValueError("Reward and actual next observation required for bootstrap")
            if not step.platform_valid:
                reasons.append("platform_invalid")
            if step.intervention is not None:
                reasons.append("external_intervention:" + str(step.intervention))
            if not _same_action(decision.action, step.executed_action):
                reasons.append("sampled_executed_action_mismatch")
            records.append(Transition(before, memory, decision, step))
            self.observation, self.memory = deepcopy(step.observation), deepcopy(decision.memory)
            self.ended = step.terminated or step.truncated
            self.confirmed_transitions += 1
            self.unresolved_step = False
            if self.ended or reasons:
                break
        return Batch(self.version, self.contract, self.source_log, tuple(records),
                     deepcopy(self.observation), deepcopy(self.memory), tuple(sorted(set(reasons))), monotonic() - start)


def update_from_batch(learner: Learner, batch: Batch, *, max_optimizer_steps: int):
    """Return an unpromoted candidate; never reinterpret offline labels as RL."""
    if type(max_optimizer_steps) is not int or max_optimizer_steps <= 0:
        raise ValueError("Positive optimizer-step cap required")
    batch = deepcopy(batch)
    if not batch.admissible:
        return {**batch.receipt(), "status": "rejected", "optimizer_steps": 0}
    for i, tr in enumerate(batch.transitions):
        if (tr.decision.policy_version != batch.policy_version or not _finite(tr.decision.log_probability)
                or not tr.step.platform_valid or tr.step.intervention is not None
                or not _same_action(tr.decision.action, tr.step.executed_action)
                or tr.step.observation is None
                or (i < len(batch.transitions) - 1 and (tr.step.terminated or tr.step.truncated))):
            raise ValueError("Batch transition violates on-policy provenance")
    if learner.parent_version != batch.policy_version or learner.likelihood_contract != batch.likelihood_contract:
        raise ValueError("Stale policy or incompatible native likelihood contract")
    result = learner.update(batch, max_optimizer_steps=max_optimizer_steps)
    if (not isinstance(result, dict) or not isinstance(result.get("candidate"), str)
            or not result["candidate"] or type(result.get("optimizer_steps")) is not int
            or not 0 < result["optimizer_steps"] <= max_optimizer_steps):
        raise ValueError("Native learner returned an invalid update receipt")
    return {**batch.receipt(), "status": "completed", "candidate": result["candidate"],
            "optimizer_steps": result["optimizer_steps"], "promoted": False,
            "policy_activated": False}
