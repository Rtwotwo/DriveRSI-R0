import json

import numpy as np
import pytest

from driversi_agents.augustus.cache import original_window


def sample():
    frames = [
        dict(
            token=str(i),
            log_name="source",
            timestamp=1000000 + i * 500000,
            ego2global_translation=np.array([i * 0.5, 0.0, 0.0]),
            ego2global_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        for i in range(12)
    ]

    def block(values):
        return "```json\n" + json.dumps(values) + "\n```"

    entry = dict(
        frame_id="3",
        scene_name="source",
        historical_trajectory=block([{"x_y_radian": [(i - 3) * 0.5, 0, 0]} for i in range(4)]),
        solution=dict(
            planning=block([{"x_y_radian": [(i - 3) * 0.5, 0, 0]} for i in range(4, 12)]),
            prediction=block([{"bbox_2d": [], "future_position": []}]),
        ),
    )
    return frames, entry


def test_native_annotation_matches_full_original_window():
    f, e = sample()
    w, h, t, p = original_window(f, e)
    assert h.shape == (4, 3) and t.shape == (8, 3) and p == [1.5, 0.0, 0.0]


def test_changed_future_or_source_rejected():
    f, e = sample()
    f[10]["ego2global_translation"][0] += 1
    with pytest.raises(ValueError, match="anchor"):
        original_window(f, e)
    f, e = sample()
    f[2]["log_name"] = "other"
    with pytest.raises(ValueError, match="source"):
        original_window(f, e)


def test_missing_prediction_and_missing_future_rejected():
    f, e = sample()
    e["solution"]["prediction"] = ""
    with pytest.raises(ValueError, match="prediction"):
        original_window(f, e)
    f, e = sample()
    with pytest.raises(ValueError, match="full"):
        original_window(f[:-1], e)


def test_large_timestamp_gap_rejected():
    f, e = sample()
    f[4]["timestamp"] += 300000
    with pytest.raises(ValueError, match="sampling"):
        original_window(f, e)


def test_smoke_cache_cannot_enter_independent_training_protocol(tmp_path):
    from driversi_agents.augustus.data import manifest

    file = tmp_path / "dataset.json"
    file.write_text(json.dumps({"kind": "augustus_cache_smoke"}))
    with pytest.raises(ValueError, match="parent-disjoint"):
        manifest(file)
