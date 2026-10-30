"""Separate physical sample validity from current-policy learning eligibility."""
import numpy as np


def learning_mask(rollout):
    valid = np.asarray(rollout["valid"])
    if valid.dtype != np.bool_:
        raise ValueError("Physical validity mask must be boolean")
    mask = valid.copy()
    mixed = "behavior_policy_index" in rollout
    if mixed and not {"learner_mask", "on_policy_eligible"} <= set(rollout):
        raise ValueError("Mixed rollout requires role and on-policy eligibility masks")
    for name in ("learner_mask", "on_policy_eligible"):
        if name in rollout:
            value = np.asarray(rollout[name])
            if value.shape != valid.shape or value.dtype != np.bool_:
                raise ValueError(f"Invalid {name} shape or dtype")
            mask &= value
    if mixed:
        owners = np.asarray(rollout["behavior_policy_index"])
        if owners.shape != valid.shape or owners.dtype.kind not in "iu":
            raise ValueError("Invalid behavior policy indices")
        if np.any(valid & ((owners < 0) | (rollout["learner_mask"] != (owners == 0)))):
            raise ValueError("Learner ownership disagrees with behavior policy")
    return mask
