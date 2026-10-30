import numpy as np
import pytest

from driversi_agents.driverl.repair import bounded_tracking_control
from driversi_agents.driverl.search import repair_window


def test_opt_in_and_native_identity_separate():
    assert repair_window() is None
    v = repair_window("pdm_route_v1", 280, 600)
    assert not v["baseline_comparable"] and not v["native_model_modified"]
    assert not v["future_observations"] and not v["training"]


@pytest.mark.parametrize("args", [(None, 1, 3), ("arbitrary", 1, 3),
    ("pdm_route_v1", True, 3), ("pdm_route_v1", 3, 3),
    ("pdm_route_v1", -1, 4), ("pdm_route_v1", 1, 6001)])
def test_invalid_repair_window(args):
    with pytest.raises(ValueError):
        repair_window(*args)


@pytest.mark.parametrize("previous", [-6.5, -3.0, 0.0, 3.8, 4.0])
@pytest.mark.parametrize("desired", [-100, 0, 100])
def test_limits_are_physical_not_score_edits(previous, desired):
    acceleration, steering = bounded_tracking_control(previous, desired, 20, 0.1)
    assert -6.5 <= acceleration <= 4.0
    assert -8.00000001 <= (acceleration - previous) / 0.1 <= 5.00000001
    assert steering == 0.8
    assert bounded_tracking_control(previous, desired, -20, 0.1)[1] == -0.8


@pytest.mark.parametrize("args", [(np.nan, 0, 0, .1), (0, np.inf, 0, .1),
    (0, 0, np.nan, .1), (0, 0, 0, .2), (-8, 0, 0, .1), (5, 0, 0, .1)])
def test_unsupported_state_is_rejected(args):
    with pytest.raises(ValueError):
        bounded_tracking_control(*args)


def test_unsafe_high_score_does_not_override_safe_lower_score():
    from driversi_agents.driverl.repair import select_admissible_proposal
    selected, valid = select_admissible_proposal([1.0, 0.8, 0.9], np.ones((2, 3)), [0.0, 1.0, 1.0], [1.0, 1.0, 0.0])
    assert selected == 1
    assert valid.tolist() == [False, True, False]


def test_no_feasible_proposal_is_not_silent_argmax_zero():
    from driversi_agents.driverl.repair import select_admissible_proposal
    selected, valid = select_admissible_proposal([0.0, float("nan")], np.zeros((2, 2)), [1.0, 1.0], [1.0, 1.0])
    assert selected is None and not valid.any()


def test_stop_fallback_cannot_create_reverse_velocity():
    from driversi_agents.driverl.repair import nonreversing_tracking_control
    speed, previous = 0.5, 0.0
    for _ in range(120):
        command, _ = nonreversing_tracking_control(previous, -3.0, 0.0, speed, 0.1)
        previous += (command - previous) / 3.0
        speed += 0.1 * previous
        assert speed >= -1e-12
        speed = max(0.0, speed)  # numeric round-off in the reference integration
    command, _ = nonreversing_tracking_control(0.0, -3.0, 0.0, 0.0, 0.1)
    assert command == 0.0


def test_unsupported_stop_rejects_before_world_advance():
    from driversi_agents.driverl.repair import nonreversing_tracking_control
    with pytest.raises(ValueError, match="anchor unsupported"):
        nonreversing_tracking_control(-3.0, -3.0, 0.0, 0.0, 0.1)


def test_equal_valid_scores_use_progress_but_never_override_safety():
    from driversi_agents.driverl.repair import select_admissible_proposal
    chosen, _ = select_admissible_proposal([1., 1., 1.], np.ones((2, 3)), [1., 1., 0.], [1., 1., 1.], [0., 2., 50.])
    assert chosen == 1
