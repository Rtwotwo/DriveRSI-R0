"""Exact single-worker recovery at collection/update boundaries, never mid-step."""

import copy
from pathlib import Path

import torch

from driversi_agents.driverl.learning.learner import restore_rng, rng_state

COLLECTOR_FIELDS = (
    "version",
    "contract",
    "source_log",
    "observation",
    "memory",
    "ended",
    "environment_step_calls",
    "confirmed_transitions",
    "unresolved_step",
)


def save(path, collector, learner, *, seal):
    if collector.unresolved_step:
        raise ValueError("unresolved world step cannot be checkpointed")
    if not (collector.actor.policy_version == learner.parent_version == collector.version):
        raise ValueError("actor/learner versions are not synchronized")
    actor_state, learner_state = collector.actor.agent.state_dict(), learner.agent.state_dict()
    if actor_state.keys() != learner_state.keys() or any(
        not torch.equal(actor_state[k], learner_state[k]) for k in actor_state
    ):
        raise ValueError("actor/learner weights are not synchronized")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = {
        "format": "driversi-driverl-training-v1",
        "seal": copy.deepcopy(seal),
        "actor_version": collector.actor.policy_version,
        "actor": collector.actor.agent.state_dict(),
        "learner": learner.state_dict(),
        "environment": collector.environment.snapshot(),
        "collector": {k: copy.deepcopy(getattr(collector, k)) for k in COLLECTOR_FIELDS},
        "rng": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def restore(path, collector, learner, *, seal):
    # Checkpoints are task-owned local pickle artifacts, not arbitrary external uploads.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["format"] != "driversi-driverl-training-v1" or state["seal"] != seal:
        raise ValueError("recovery context changed")
    if (
        state["collector"]["unresolved_step"]
        or state["actor_version"] != state["learner"]["parent_version"]
        or state["actor_version"] != state["collector"]["version"]
        or state["collector"]["source_log"] != collector.source_log
    ):
        raise ValueError("recovery actor/source mismatch")
    collector.environment.restore(state["environment"])
    collector.actor.agent.load_state_dict(state["actor"], strict=True)
    learner.parent_version = state["learner"]["parent_version"]
    collector.actor.policy_version = state["actor_version"]
    learner.load_state_dict(state["learner"])
    for key, value in state["collector"].items():
        setattr(collector, key, copy.deepcopy(value))
    restore_rng(state["rng"])
    return state
