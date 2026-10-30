import pytest

from driversi_agents.driverl.navsim_context import require_dense_history


def history(times):
    return [{"ego": [0.0, float(t)]} for t in times]


def test_teacher_requires_distinct_dense_physical_history():
    with pytest.raises(ValueError, match="Dense"):
        require_dense_history(history([0, 500000, 1000000, 1500000]), 1500000)
    with pytest.raises(ValueError, match="Dense"):
        require_dense_history(history([2000000] * 5), 2000000)
    assert require_dense_history(history(range(0, 2000001, 100000)), 2000000)["max_error_us"] == 0


def test_teacher_history_rejects_missing_native_sample():
    h = history([0, 200000, 400000, 800000])
    with pytest.raises(ValueError, match="Dense"):
        require_dense_history(h, 800000)


def test_causal_bootstrap_grid_never_uses_after_zero():
    from driversi_agents.driverl.history_bootstrap import causal_grid

    rows = causal_grid([-1500000, -1000000, -500000, 0], 0)
    assert [r[0] for r in rows] == list(range(-1000000, 0, 100000))
    assert all(0 <= r[3] <= 1 and r[1] <= r[2] <= 3 for r in rows)
    with pytest.raises(ValueError):
        causal_grid([-1000000, -500000, 100000], 0)
    with pytest.raises(ValueError):
        causal_grid([-500000, -250000, 0], 0)


def test_history_interpolation_preserves_track_presence_and_angle():
    import math

    from driversi_agents.driverl.history_bootstrap import interpolate_objects

    def obj(token, yaw):
        return dict(
            track_token=token,
            token=token,
            kind="VEHICLE",
            pose=[0, 0, yaw],
            size=[4, 2, 1.5],
            velocity=[1, 0],
            angular_velocity=None,
        )

    rows = interpolate_objects(
        [obj("a", math.pi - 0.1), obj("leaves", 0)],
        [obj("a", -math.pi + 0.1), obj("enters", 0)],
        0.5,
    )
    assert [x["track_token"] for x in rows] == ["a"]
    assert abs(abs(rows[0]["pose"][2]) - math.pi) < 1e-9
    assert rows[0]["angular_velocity"] is None
    with pytest.raises(ValueError):
        interpolate_objects([], [], 1.1)


def test_all_window_native_mode_has_no_rule_control():
    from driversi.evaluation.search.navsim_teacher import arrival_control, step_arguments

    for tick in range(20):
        action, native = arrival_control(tick, "driverl_native_causal_history", [0, 0])
        assert action is None and native is True
        assert step_arguments(action, force_native=native) == {}
