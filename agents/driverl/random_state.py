"""RNG ownership for native stochastic policy branches, without changing policy."""
import copy
import random

import numpy as np


def validate_sampling(method, *, search_candidates=0, parity=False):
    if method not in ("argmax", "sample"):
        raise ValueError("Unknown native DriveRL sampling method")
    if method == "sample" and (search_candidates or parity):
        raise ValueError("Native sampling cannot mix TTS or continuous parity")
    return method


def capture():
    import torch

    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else None,
    }


def restore(state):
    import torch

    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Native branch RNG state is incomplete")
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch"].clone())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.clone() for s in state["cuda"]])
