"""Per-actor finite-task GAE; collection cuts and episode ends are distinct."""
import numpy as np


def generalized_advantage(rewards, values, next_values, valid, terminated, truncated,
                          gamma=.99, lam=.95):
    arrays = [np.asarray(a) for a in (rewards, values, next_values, valid, terminated, truncated)]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Expected matching [time, actor_slot] arrays")
    if not all(np.isfinite(a).all() for a in arrays[:3]):
        raise ValueError("Non-finite rollout")
    rewards, values, next_values, valid, terminated, truncated = arrays
    if any(a.dtype != np.bool_ for a in arrays[3:]):
        raise ValueError("Masks must be boolean")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("Invalid discount/GAE lambda")
    continuation = valid & ~terminated & ~truncated
    advantage = np.zeros_like(rewards, dtype=np.float32)
    carried = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * continuation[t] * next_values[t] - values[t]
        carried = (delta + gamma * lam * continuation[t] * carried) * valid[t]
        advantage[t] = carried
    return advantage, (advantage + values) * valid
