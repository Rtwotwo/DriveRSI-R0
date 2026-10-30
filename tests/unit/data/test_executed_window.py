import numpy as np
import pytest

from driversi.data.datasets.executed_window import trajectory_window


def test_window_preserves_executed_history_and_rear_axle_future():
    states = np.zeros((71, 2, 13))
    states[:, 0, 0] = np.arange(71) * 0.1
    states[:, 1, 0] = 100
    states[:, 0, 3] = np.pi / 2
    presence = np.ones((71, 2), bool)
    presence[25:, 1] = False
    result = trajectory_window(states, presence, ("ego", "neighbor"), anchor_tick=20, dt=0.1)
    assert [r["tick"] for r in result["history"]] == [5, 10, 15, 20]
    assert [r["tick"] for r in result["future"]] == list(range(25, 61, 5))
    assert len(result["history"][-1]["objects"]) == 2
    assert len(result["future"][0]["objects"]) == 1
    np.testing.assert_allclose(result["future_trajectory"]["poses"][0], [0, -0.5, 0], atol=1e-12)
    assert result["input_scope"] == "history_only" and not result["trainable"]


@pytest.mark.parametrize("fault", ["short", "clock", "ego_missing", "duplicates"])
def test_window_rejects_ambiguous_or_incomplete_execution(fault):
    states = np.zeros((71, 2, 13))
    presence = np.ones((71, 2), bool)
    ids = ("ego", "other")
    dt = 0.1
    if fault == "short":
        states = states[:60]
        presence = presence[:60]
    elif fault == "clock":
        dt = 0.5
    elif fault == "ego_missing":
        presence[19, 0] = False
    elif fault == "duplicates":
        ids = ("ego", "ego")
    with pytest.raises(ValueError):
        trajectory_window(states, presence, ids, anchor_tick=20, dt=dt)


def test_ten_point_window_uses_actual_last_second_and_requires_it():
    states = np.zeros((71, 2, 13))
    states[:, 0, 0] = np.arange(71) * 0.1
    states[65:, 0, 1] = np.arange(6) * 0.2
    states[:, 1, 0] = 20.0
    presence = np.ones((71, 2), bool)
    presence[66:, 1] = False
    result = trajectory_window(
        states, presence, ('ego', 'other'), anchor_tick=20, dt=0.1, future_frames=10
    )
    assert result['kind'] == 'executed_4history_10future_v1'
    assert [x['tick'] for x in result['future']] == list(range(25, 71, 5))
    assert result['future'][-1]['objects'][0]['center_state'] == states[70, 0].tolist()
    assert len(result['future'][-1]['objects']) == 1
    np.testing.assert_allclose(result['future_trajectory']['poses'][-1], [5, 1, 0])
    with pytest.raises(ValueError, match='Complete requested'):
        trajectory_window(
            states[:70], presence[:70], ('ego', 'other'), anchor_tick=20, dt=0.1, future_frames=10
        )


@pytest.mark.parametrize('count', [True, 9, 10.0, 0])
def test_window_rejects_ambiguous_future_count(count):
    with pytest.raises(ValueError, match='Future frame count'):
        trajectory_window(
            np.zeros((71, 1, 13)), np.ones((71, 1), bool), ('ego',),
            anchor_tick=20, dt=0.1, future_frames=count,
        )
