"""Parent normalization is immutable across RSI rounds."""

import json
from pathlib import Path

import numpy as np


def read_stats(path, action_dim=3, action_repr="absolute", horizon=8):
    value = json.loads(Path(path).read_text())
    if value.get("action_repr", "absolute") != action_repr:
        raise ValueError("Parent normalization action representation mismatch")
    for key, shape in (
        ("q01", (action_dim,)),
        ("q99", (action_dim,)),
        ("q01_per_t", (horizon, action_dim)),
        ("q99_per_t", (horizon, action_dim)),
    ):
        array = np.asarray(value.get(key), dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Parent normalization incomplete: " + key)
    if np.any(np.asarray(value["q99"]) < value["q01"]) or np.any(
        np.asarray(value["q99_per_t"]) < value["q01_per_t"]
    ):
        raise ValueError("Invalid normalization quantile ordering")
    return value


def frozen_loader(path, action_dim=3, action_repr="absolute"):
    path = Path(path).resolve(strict=True)
    original = path.read_bytes()
    stats = read_stats(path, action_dim, action_repr)

    def load(stats_file=None, train_data_path=None, action_dim=3, action_repr="absolute"):
        if (
            Path(stats_file).resolve() != path
            or path.read_bytes() != original
            or action_dim != 3
            or action_repr != "absolute"
        ):
            raise ValueError("Parent normalization changed")
        return json.loads(json.dumps(stats))

    return load
