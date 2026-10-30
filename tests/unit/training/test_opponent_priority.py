import numpy as np
import pytest

from driversi.training.self_play.opponent_priority import failure_probabilities


def test_priority_monotonic_exploration_and_small_sample_smoothing():
    probs = failure_probabilities([(100, 0), (100, 20), (100, 90)])
    assert np.isfinite(probs).all() and probs.sum() == pytest.approx(1)
    assert np.all(probs >= 0.2 / 3) and np.all(np.diff(probs) > 0)
    np.testing.assert_allclose(failure_probabilities([(20, 0), (20, 0)]), [0.5, 0.5])
    np.testing.assert_allclose(
        failure_probabilities([(20, 5), (20, 19)], uniform_mix=1), [0.5, 0.5]
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"uniform_mix": 0},
        {"uniform_mix": float("nan")},
        {"power": True},
        {"power": 9},
        {"min_actor_episodes": False},
        {"min_actor_episodes": 21},
    ],
)
def test_bad_prioritization_rejected(kw):
    with pytest.raises(ValueError):
        failure_probabilities([(20, 5)], **kw)


@pytest.mark.parametrize("counts", [[], [(20, 21)], [(20, -1)], [(True, 0)], [(0, 0)]])
def test_bad_counts_rejected(counts):
    with pytest.raises(ValueError):
        failure_probabilities(counts)
