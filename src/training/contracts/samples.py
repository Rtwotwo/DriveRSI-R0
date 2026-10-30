"""Native action-conditioned sequences, independent of the legacy 12-action format.

Future observations are targets. Only a world model's action-conditioned input
may include recorded future EXECUTED actions; policy input never includes them.
Adapters supply actual state timestamps and branch identity, not nominal timing.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from driversi.training.online_rl.online import Batch, _finite


@dataclass(frozen=True)
class NativeWorldSequence:
    source_log: str
    branch_id: str
    action_contract: str
    timestamps_ns: np.ndarray
    observations: tuple
    executed_actions: tuple
    interventions: tuple
    context_frames: int

    def __post_init__(self):
        n = len(self.observations)
        if any(
            not isinstance(x, str) or not x
            for x in (self.source_log, self.branch_id, self.action_contract)
        ):
            raise ValueError("Explicit sequence provenance and native action units required")
        times = np.asarray(self.timestamps_ns)
        if (
            times.shape != (n,)
            or times.dtype.kind not in "iu"
            or not np.all(times[1:] > times[:-1])
        ):
            raise ValueError("Actual monotonic state timestamps required")
        if type(self.context_frames) is not int or not 1 <= self.context_frames < n:
            raise ValueError("Both history and future required")
        if len(self.executed_actions) != n - 1 or len(self.interventions) != n - 1:
            raise ValueError("Actions must align state t to state t+1")
        if any(o is None for o in self.observations) or any(
            not _finite(a) for a in self.executed_actions
        ):
            raise ValueError("Missing native observation or executed action")
        if len({np.asarray(a).shape for a in self.executed_actions}) != 1:
            raise ValueError("Native action shape changed inside sequence")
        object.__setattr__(self, "timestamps_ns", times.copy())
        object.__setattr__(self, "observations", deepcopy(self.observations))
        object.__setattr__(self, "executed_actions", deepcopy(self.executed_actions))

    def policy_inputs(self):
        c = self.context_frames
        return deepcopy(
            {
                "observations": self.observations[:c],
                "timestamps_ns": self.timestamps_ns[:c],
                "executed_action_history": self.executed_actions[: c - 1],
            }
        )

    def world_inputs(self):
        c = self.context_frames
        return {
            **self.policy_inputs(),
            "future_executed_actions": deepcopy(self.executed_actions[c - 1 :]),
            "action_contract": self.action_contract,
        }

    def targets(self):
        c = self.context_frames
        return deepcopy(
            {
                "observations": self.observations[c:],
                "timestamps_ns": self.timestamps_ns[c:],
                "interventions": self.interventions[c - 1 :],
            }
        )


def sequence_from_batch(batch: Batch, *, timestamps_ns, branch_id, action_contract, context_frames):
    if not batch.transitions or any(not tr.step.platform_valid for tr in batch.transitions):
        raise ValueError("Physically invalid batch cannot become a world prediction target")
    if any(tr.step.terminated or tr.step.truncated for tr in batch.transitions[:-1]):
        raise ValueError("Sequence crosses an episode reset")
    observations = (batch.transitions[0].observation,) + tuple(
        t.step.observation for t in batch.transitions
    )
    return NativeWorldSequence(
        batch.source_log,
        branch_id,
        action_contract,
        timestamps_ns,
        observations,
        tuple(t.step.executed_action for t in batch.transitions),
        tuple(t.step.intervention for t in batch.transitions),
        context_frames,
    )
