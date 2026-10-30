import pytest
from driversi.rendering.execution.native_replay import recorded_generator_provenance


def test_control_search_never_inherits_model_identity():
    metadata = {
        "policy": "bounded_search",
        "protocol": {
            "arrival_controller": "bounded_continuous_control",
            "future_controller": "bounded_native_control_rollout_search",
            "future_ticks": [20, 70],
        },
        "controller_configuration": {
            "planner": {"model": "DriveRL"},
            "executed_teacher_backend": "bounded_native_search_v1",
        },
    }
    result = recorded_generator_provenance(metadata)
    assert (
        result["controller_protocol"]["future_controller"]
        == "bounded_native_control_rollout_search"
    )
    assert "planner" not in result
    result["controller_protocol"]["future_ticks"].append(71)
    assert metadata["protocol"]["future_ticks"] == [20, 70]


def test_missing_provenance_is_unknown_not_driverl():
    result = recorded_generator_provenance({})
    assert result["policy"] == "unknown"
    assert result["controller_protocol"] == {}
    with pytest.raises(ValueError):
        recorded_generator_provenance({"protocol": "invalid"})
