import numpy as np
import pytest
from driversi.rsi.plugins.recovery import prepare_correction
from driversi_agents.driverl.control_lattice import schedules, propagate_control
from driversi_agents.driverl.repair import select_admissible_proposal


def test_backend_is_explicit_and_frozen(tmp_path):
    default, _ = prepare_correction({"max_wall_seconds": 60}, tmp_path, None)
    assert default["backend"] == "pdm_route_v1" and "teacher_files" not in default
    new, _ = prepare_correction(
        {"max_wall_seconds": 60, "backend": "native_recovery_search_v1"}, tmp_path, None
    )
    assert len(new["teacher_files"]) == 1
    from driversi.rsi.plugins.synthesis import check_files

    check_files(new["teacher_files"])
    with pytest.raises(ValueError):
        prepare_correction({"max_wall_seconds": 60, "backend": "untrusted"}, tmp_path, None)
    with pytest.raises(ValueError):
        prepare_correction({"max_wall_seconds": 60, "direction_threshold": 0}, tmp_path, None)


def test_search_bank_is_finite_unique_and_has_both_constructions():
    bank = schedules()
    assert len(bank) == 78 and len(set(bank)) == 78
    assert {s[0] for s in bank} == {"feedback", "reference"}


def test_expanded_search_keeps_every_safety_constraint():
    scores = np.array([1.0, 0.99, 0.98, 0.97])
    multi = np.ones((2, 4))
    direction = np.array([0.0, 1.0, 1.0, 1.0])
    ttc = np.array([1.0, 0.0, 1.0, 1.0])
    multi[1, 2] = 0
    chosen, valid = select_admissible_proposal(
        scores, multi, direction, ttc, [40.0, 30.0, 20.0, 10.0]
    )
    assert chosen == 3 and valid.tolist() == [False, False, False, True]
    multi[0, 3] = 0
    assert select_admissible_proposal(scores, multi, direction, ttc)[0] is None
