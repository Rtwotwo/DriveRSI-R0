import copy
import pytest

from driversi.rsi.correction.trajectory_admission import (
    trajectory_quality_pass,
    validate_trajectory_evidence,
)
from driversi.rsi.experiments.skill_execution import execution_evidence


def evidence():
    return dict(
        branch="teacher",
        completed_steps=50,
        failure=None,
        producer_identity="sealed",
        history_changed=True,
        restore_exact=True,
        arrival_contact_records=0,
        future_contact_records=0,
        events=dict(collision=0, offroad=0, route_support_end=0),
        score=dict(
            score=0.96,
            driving_direction_compliance=1.0,
            actual_world_seconds=5.0,
            protocol="navsim-v1-native-math-executed-counterfactual-v1",
        ),
    )


def test_quality_policy_is_explicit_and_strictly_greater_than_95():
    e = evidence()
    validate_trajectory_evidence(e)
    e.update(arrival_contact_records=30, future_contact_records=50)
    e["score"]["driving_direction_compliance"] = 0.5
    with pytest.raises(ValueError):
        validate_trajectory_evidence(e)
    e["admission_profile"] = "pdms_gt95_v1"
    validate_trajectory_evidence(e)
    for score in [0.95, 0.949, float("nan"), float("inf"), True, None]:
        e["score"]["score"] = score
        with pytest.raises(ValueError):
            validate_trajectory_evidence(e)


@pytest.mark.parametrize(
    "changes",
    [
        dict(completed_steps=49),
        dict(failure="route_end"),
        dict(restore_exact=False),
        dict(history_changed=False),
        dict(producer_identity=""),
        dict(score=None),
    ],
)
def test_pdms_only_does_not_admit_incomplete_or_unbound_transport(changes):
    e = {**evidence(), "admission_profile": "pdms_gt95_v1", **changes}
    with pytest.raises(ValueError):
        validate_trajectory_evidence(e)


def test_unknown_policy_is_not_a_bypass():
    with pytest.raises(ValueError, match="Unknown"):
        trajectory_quality_pass(dict(score=1.0), profile="unregistered")


def test_incomplete_teacher_diagnostic_preserves_actual_cause():
    t = dict(
        branches=dict(
            teacher=dict(
                score=None,
                object_contacts=[],
                completed_steps=45,
                failure="ego_terminated_before_complete_score_window",
            )
        )
    )
    result = execution_evidence(t)
    assert result["teacher_failure"] == "ego_terminated_before_complete_score_window"
    assert "future_execution_complete" in result["failed_checks"]
    assert execution_evidence(dict(branches=dict(teacher=None)))
