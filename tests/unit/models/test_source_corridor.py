import numpy as np
import pytest

from driversi_agents.driverl.source_corridor import corridor_evidence, recovery_offsets


def test_quintic_preserves_anchor_and_recovers_exactly_without_endpoint_extension():
    t = np.array([0.0, 0.001, 1.0, 2.0, 3.0, 5.0])
    p, v, a = np.array([1.0, -0.5]), np.array([0.2, 0.3]), np.array([0.1, -0.2])
    x = recovery_offsets(t, p, v, a, 2.0)
    np.testing.assert_allclose(x[0], p)
    np.testing.assert_allclose((x[1] - x[0]) / 0.001, v, atol=0.001)
    np.testing.assert_allclose(x[3:], 0.0, atol=1e-13)


def test_corridor_rejects_greater_than_fixed_two_metres():
    assert corridor_evidence([[0, 0], [2, 0]], [[0, 0], [0, 0]])["passed"]
    assert not corridor_evidence([[0, 0], [2.001, 0]], [[0, 0], [0, 0]])["passed"]
    with pytest.raises(ValueError):
        corridor_evidence([[np.nan, 0]], [[0, 0]])


def test_unplanned_recovery_duration_rejected():
    with pytest.raises(ValueError, match="declared"):
        recovery_offsets([0, 1], [0, 0], [0, 0], [0, 0], 4.0)


def test_asset_corridor_limit_cannot_be_widened():
    with pytest.raises(ValueError, match="fixed"):
        corridor_evidence([[0, 0]], [[0, 0]], limit_m=10.0)
