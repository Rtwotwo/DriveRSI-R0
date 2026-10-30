"""Deterministic risk selection and executed-branch admission, independent of search."""
import math
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping

SAFETY = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "time_to_collision_within_bound",
)


def select_anchor(rows):
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise ValueError("No live anchor with continuous history and future support")
    def key(r):
        return (r["closing_ttc_s"] if r["closing_ttc_s"] is not None else math.inf,
                r["clearance_m"] if r["clearance_m"] is not None else math.inf, -r["tick"])
    return min(eligible, key=key)


def admit(execution, score):
    reasons = []
    for key in ("same_anchor", "complete", "valid", "teacher_no_fallback"):
        if execution.get(key) is not True:
            reasons.append(key + "_failed")
    if execution.get("contact_states") != 0:
        reasons.append("world_contact")
    if score.get("status") != "scored" or score.get("official_benchmark") is not False:
        reasons.append("wrong_score_protocol")
    metrics = score.get("metric_scores", {})
    for key in SAFETY:
        if metrics.get(key) != 1.0:
            reasons.append(key + "_not_passed")
    value = score.get("score")
    if type(value) not in (int, float) or not math.isfinite(value) or value < .95:
        reasons.append("official_score_below_0.95")
    return {"accepted": not reasons, "reasons": reasons,
            "purpose": "feasible_online_reset_curriculum",
            "teacher_actions_are_on_policy": False, "visual_trainable": False,
            "demonstrated_correction_benefit": False, "promoted": False}


def validate_curriculum(path, *, scene, source_log, checkpoint, arm):
    value = read_mapping(path)
    if value.get("schema_version") != 1 or value.get("purpose") != "feasible_online_reset_curriculum":
        raise ValueError("Wrong curriculum schema or use")
    if not value["admission"]["accepted"] or arm not in ("control", "treatment"):
        raise ValueError("Rejected curriculum or unknown arm")
    for filename, sha in value["files"].items():
        if digest(filename) != sha:
            raise ValueError("Curriculum evidence changed: " + filename)
    if value["scene_sha256"] != digest(scene) or value["source_log"] != source_log:
        raise ValueError("Curriculum source mismatch")
    if value["parent_checkpoint_sha256"] != digest(checkpoint):
        raise ValueError("Curriculum parent mismatch")
    # Recompute admission from immutable execution and official score evidence.
    correction = read_mapping(value["correction_report"])
    if admit(correction["execution"], correction["score"]) != value["admission"]:
        raise ValueError("Curriculum admission mismatch")
    if value.get("hard_failure_required"):
        joint=read_mapping(value["paired_correction_report"])
        if not correction_verdict(joint["branches"],True)["accepted"]:
            raise ValueError("Genuine hard correction not proven")
    selected = value["starts"][arm]
    if not Path(selected["snapshot"]).is_file():
        raise ValueError("Missing native reset snapshot")
    return value, selected


def failed_driving(score):
    metrics=score.get("metric_scores", {})
    values=[metrics.get(k) for k in SAFETY]
    return (score.get("status")=="scored" and score.get("official_benchmark") is False
        and all(type(v) in (float,int) and math.isfinite(v) for v in values)
        and any(v<1.0 for v in values))


def correction_verdict(reports, required):
    a,b=reports["student"],reports["teacher"]
    actual=failed_driving(a["score"])
    complete=a["execution"].get("complete") is True and a["execution"].get("same_anchor") is True
    admission=admit(b["execution"],b["score"])
    accepted=admission["accepted"] and (not required or (actual and complete))
    return dict(required=required,actual_student_failure=actual,accepted=bool(accepted),
        reason=("teacher_not_admitted" if not admission["accepted"] else
                "not_a_complete_failed_student" if required and not(actual and complete) else None))


def valid_recovery_transition(transition, *, final_step):
    """A scheduled final time limit is not an exhausted route or driving failure."""
    return bool(transition.valid.all() and not transition.terminated.any()
        and not any(transition.events.get("route_support_end", (False,)))
        and (not transition.truncated.any() or (final_step and transition.episode_done)))


def route_support_failure(flags):
    """Source/map support exhaustion is not a learnable driving failure."""
    import numpy as np
    values = np.asarray(flags, dtype=bool)
    if values.size == 0:
        values = values.reshape(0, 0)
    if values.ndim != 2:
        raise ValueError("Expected step-by-policy-actor route support flags")
    ticks = (np.flatnonzero(values.any(axis=1)) + 1).tolist()
    return dict(blocked=bool(ticks), ticks=ticks,
        reason="route_support_exhausted" if ticks else None,
        attribution="platform_or_source_support_not_model_failure")
