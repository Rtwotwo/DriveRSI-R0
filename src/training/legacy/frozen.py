"""Read historical diagnostic NPZ policies; no training model or default selection."""
import json
from pathlib import Path

import numpy as np

from driversi.contracts.learning import OBSERVATION_SIZE, OBSERVATION_VERSION


class NpzPolicy:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.weights = {k: data[k].copy() for k in data.files if k != "metadata"}
        if self.metadata.get("observation_version") != OBSERVATION_VERSION:
            raise ValueError("Teacher observation version mismatch")
        architecture = self.metadata.get("architecture", {"version": 1, "hidden_sizes": [128, 128],
                                                           "activation": "tanh"})
        if (not isinstance(architecture, dict) or set(architecture) != {"version", "hidden_sizes", "activation"}
                or architecture["version"] != 1 or architecture["activation"] != "tanh"
                or not isinstance(architecture["hidden_sizes"], list)
                or any(type(n) is not int or n < 1 for n in architecture["hidden_sizes"])):
            raise ValueError("Unsupported frozen architecture")
        self.layers = tuple(f"trunk.{2*i}" for i in range(len(architecture["hidden_sizes"])))
        sizes = {}
        previous = OBSERVATION_SIZE
        for layer, width in zip(self.layers, architecture["hidden_sizes"]):
            sizes.update({layer + ".weight": (width, previous), layer + ".bias": (width,)})
            previous = width
        sizes.update({"actor.weight": (12, previous), "actor.bias": (12,),
                      "critic.weight": (1, previous), "critic.bias": (1,)})
        if set(self.weights) != set(sizes) or any(
            self.weights[k].shape != shape or not np.isfinite(self.weights[k]).all()
            for k, shape in sizes.items()
        ):
            raise ValueError("Invalid teacher weights")

    def predict(self, observations):
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_SIZE or not np.isfinite(x).all():
            raise ValueError("Invalid privileged teacher input")
        for layer in self.layers:
            x = np.tanh(x @ self.weights[layer + ".weight"].T + self.weights[layer + ".bias"])
        logits = x @ self.weights["actor.weight"].T + self.weights["actor.bias"]
        values = (x @ self.weights["critic.weight"].T + self.weights["critic.bias"]).reshape(-1)
        return logits, values

    def actions(self, observations):
        return self.predict(observations)[0].argmax(axis=1)
