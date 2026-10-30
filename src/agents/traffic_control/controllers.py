"""Engineering controllers and one-observation-time joint action dispatch."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from driversi.contracts.transition import Observation


class BatchPolicy(Protocol):
    def act(self, observation: Observation) -> np.ndarray: ...


@dataclass(frozen=True)
class ConstantJerk:
    """A diagnostic action source, not a driving baseline or learned policy.

    7 holds jerk at zero (does not cancel existing acceleration); 1 applies
    longitudinal braking jerk with zero lateral jerk.
    """

    action: int = 7

    def act(self, observation: Observation) -> np.ndarray:
        return np.full(len(observation.actor_ids), self.action, dtype=np.int32)


class JointController:
    def __init__(self, assignments: dict[str, BatchPolicy]):
        self.assignments = dict(assignments)

    def act(self, observation: Observation) -> dict[str, int]:
        if set(self.assignments) != set(observation.actor_ids):
            raise ValueError("Assign exactly one controller to every controlled actor")
        groups: dict[int, tuple[BatchPolicy, list[int]]] = {}
        for i, actor_id in enumerate(observation.actor_ids):
            policy = self.assignments[actor_id]
            groups.setdefault(id(policy), (policy, []))[1].append(i)
        joint = {}
        for policy, indices in groups.values():
            part = Observation(tuple(observation.actor_ids[i] for i in indices),
                               observation.features[indices].copy(),
                               observation.alive[indices].copy(), observation.step,
                               None if observation.metric is None else observation.metric.select(indices))
            actions = np.asarray(policy.act(part))
            if (actions.shape != (len(indices),) or actions.dtype.kind not in "iu"
                    or np.any(actions < 0) or np.any(actions >= 12)):
                raise ValueError("Controller must return a batch of discrete jerk actions")
            joint.update(zip(part.actor_ids, (int(a) for a in actions), strict=True))
        return joint
