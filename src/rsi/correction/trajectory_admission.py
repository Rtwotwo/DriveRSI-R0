"""Versioned executed-trajectory quality and transport integrity contracts."""

import math

STRICT_PROFILE = "strict_world95_v1"
PDMS_PROFILE = "pdms_gt95_v1"


def trajectory_policy(name=STRICT_PROFILE):
    if name not in (STRICT_PROFILE, PDMS_PROFILE):
        raise ValueError("Unknown trajectory admission profile")
    return dict(
        name=name,
        pdms_min=0.95,
        strict_gt=name == PDMS_PROFILE,
        extra_world_vetoes=name == STRICT_PROFILE,
    )


def trajectory_quality_pass(
    score, *, profile=STRICT_PROFILE, arrival_contacts=0, future_contacts=0, events=None
):
    policy = trajectory_policy(profile)
    score = score or {}
    value = score.get("score")
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    if not (value > 0.95 if policy["strict_gt"] else value >= 0.95):
        return False
    if policy["extra_world_vetoes"]:
        return (
            arrival_contacts == 0
            and future_contacts == 0
            and score.get("driving_direction_compliance") == 1.0
            and not any(
                (events or {}).get(k, 1) for k in ("collision", "offroad", "route_support_end")
            )
        )
    return True


def validate_trajectory_evidence(evidence):
    """ADE remains a separate cohort check; this never certifies RGB quality."""
    score = evidence.get("score") or {}
    if (
        evidence.get("branch") != "teacher"
        or evidence.get("completed_steps") != 50
        or evidence.get("failure") is not None
        or not evidence.get("producer_identity")
        or evidence.get("history_changed") is not True
        or evidence.get("restore_exact") is not True
        or score.get("actual_world_seconds") != 5.0
        or score.get("protocol") != "navsim-v1-native-math-executed-counterfactual-v1"
        or not trajectory_quality_pass(
            score,
            profile=evidence.get("admission_profile", STRICT_PROFILE),
            arrival_contacts=evidence.get("arrival_contact_records"),
            future_contacts=evidence.get("future_contact_records"),
            events=evidence.get("events"),
        )
    ):
        raise ValueError("Executed teacher evidence does not meet trajectory admission")
