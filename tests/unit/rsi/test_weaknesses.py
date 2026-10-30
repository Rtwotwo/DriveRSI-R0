import pytest

from driversi.rsi.decisions.training_diagnosis import diagnose_training_attempt
from driversi.rsi.decisions.weaknesses import review, validate_samples


def evidence():
    student = dict(status="completed", context="c", checkpoint="exp/model", model="augustus")
    evaluation = dict(
        status="completed",
        context="c",
        evaluated_checkpoint="exp/model",
        source_groups={"train": ["train-log"], "development": ["dev-log"]},
        directions={"NC": 1, "EP": 1},
        metrics={"parent": {"NC": 0.5, "EP": 0.8}, "candidate": {"NC": 0.5, "EP": 0.7}},
        evaluated_sample_ids=["a", "b"],
        paired_samples=[
            dict(
                sample_id="a",
                source_log="dev-log",
                parent={"NC": 1.0, "EP": 0.8},
                candidate={"NC": 0.0, "EP": 0.7},
            ),
            dict(
                sample_id="b",
                source_log="dev-log",
                parent={"NC": 0.0, "EP": 0.8},
                candidate={"NC": 1.0, "EP": 0.7},
            ),
        ],
    )
    return student, evaluation, dict(pilot_metric_improved=False)


def test_sample_safety_regression_not_hidden_by_unchanged_average():
    result = review(*evidence())
    row = next(x for x in result["signals"] if x["metric"] == "NC")
    assert row["delta"] == 0 and row["regressed_sample_ids"] == ["a"]
    assert not row["model_cause_established"]
    assert result["support"]["original_log_groups"] == 1
    assert not result["experiment_proposal"]["targeted_trainable"]
    assert not result["automatic_execution"] and not result["promoted"]


@pytest.mark.parametrize("fault", ["duplicates", "coverage", "leakage", "aggregate", "nonfinite"])
def test_untrusted_or_incomplete_diagnostic_rows_cannot_be_used(fault):
    _, data, _ = evidence()
    if fault == "duplicates":
        data["paired_samples"][1]["sample_id"] = "a"
    if fault == "coverage":
        data["evaluated_sample_ids"].append("missing")
    if fault == "leakage":
        data["paired_samples"][0]["source_log"] = "train-log"
    if fault == "aggregate":
        data["metrics"]["candidate"]["NC"] = 1.0
    if fault == "nonfinite":
        data["paired_samples"][0]["candidate"]["EP"] = float("nan")
    with pytest.raises(ValueError):
        validate_samples(data)


def test_aggregate_only_loss_is_not_driving_failure_diagnosis():
    student, data, feedback = evidence()
    student["model"] = "unregistered_world_model"
    data.pop("paired_samples")
    data.pop("evaluated_sample_ids")
    data["directions"] = {"val_loss_jepa_pred": -1}
    data["metrics"] = {
        "parent": {"val_loss_jepa_pred": 0.01},
        "candidate": {"val_loss_jepa_pred": 0.02},
    }
    result = review(student, data, feedback)
    assert result["signals"] == []
    assert not result["support"]["per_sample_available"]
    assert result["causal_attribution"] == "unresolved"


def test_recipe_preserves_native_loss_and_requires_control_and_source_isolation():
    proposal = review(*evidence())["experiment_proposal"]
    assert "native_loss" in proposal["frozen"]
    assert "equal_update_budget" in proposal["control"]
    assert "never_copy_development_cases" in proposal["target_source"]
    assert "prediction_language" in proposal["data_contract"]


def test_wrong_checkpoint_cannot_generate_weakness_review():
    student, data, feedback = evidence()
    data["evaluated_checkpoint"] = "other"
    with pytest.raises(ValueError):
        review(student, data, feedback)


@pytest.mark.parametrize(
    "updates,category",
    [(2, "update_observed"), (0, "update_unverified"), (True, "update_unverified")],
)
def test_schema2_confirmed_updates_recognized_without_legacy_field(updates, category):
    result = diagnose_training_attempt(
        {"schema_version": 2, "resolved": {"backend": {"id": "native"}}},
        {
            "schema_version": 2,
            "status": "completed",
            "checkpoint": "exp/checkpoint",
            "updates": updates,
        },
    )
    assert result["category"] == category
    assert not result["model_deficiency_established"]


def test_legacy_update_evidence_still_supported():
    result = diagnose_training_attempt(
        {"resolved": {"backend": {"id": "native"}}},
        {
            "status": "completed",
            "checkpoint": "exp/checkpoint",
            "consumed": {"successful_optimizer_steps": 2},
        },
    )
    assert result["category"] == "update_observed"


def test_declared_logs_do_not_inflate_observed_sample_support():
    student, data, feedback = evidence()
    data["source_groups"]["development"].append("unused-log")
    report = review(student, data, feedback)
    assert report["support"]["original_log_groups"] == 1
    assert report["support"]["declared_original_log_groups"] == 2


def test_local_safety_regression_blocks_positive_credit():
    student, data, feedback = evidence()
    feedback["pilot_metric_improved"] = True
    data["safety_higher"] = ["NC"]
    result = review(student, data, feedback)
    assert not result["pilot_metric_improved"]
    assert result["local_safety_regressions"] == [{"sample_id": "a", "metric": "NC"}]


def test_persistent_native_compliance_failure_is_not_lost_when_delta_is_zero():
    student, data, feedback = evidence()
    for side in ("parent", "candidate"):
        data["metrics"][side] = {"NC": 0.0, "EP": 0.8}
        for row in data["paired_samples"]:
            row[side] = {"NC": 0.0, "EP": 0.8}
    result = review(student, data, feedback)
    assert result["next_action"] == "diagnose_observed_weakness"
    assert len(result["signals"]) == 1
    signal = result["signals"][0]
    assert signal["metric"] == "NC" and signal["delta"] == 0
    assert signal["regressed_sample_ids"] == []
    assert signal["persistently_deficient_sample_ids"] == ["a", "b"]
    assert signal["candidate_compliance_shortfall"]
    assert not signal["model_cause_established"]


def test_unchanged_loss_does_not_get_an_invented_failure_threshold():
    student, data, feedback = evidence()
    student["model"] = "unregistered_world_model"
    data.pop("paired_samples")
    data.pop("evaluated_sample_ids")
    data["directions"] = {"val_loss_jepa_pred": -1}
    data["metrics"] = {side: {"val_loss_jepa_pred": 0.01} for side in ("parent", "candidate")}
    assert review(student, data, feedback)["signals"] == []


def control_evidence():
    from copy import deepcopy

    anchor = {
        "source_log": "train-log",
        "token": "anchor-token",
        "timestamp_us": 123,
        "metric_cache_sha256": "a" * 64,
    }
    protocol = {
        "id": "navsim-v1-native-original-source-4s",
        "scorer_sha256": "b" * 64,
        "horizon_s": 4.0,
        "interval_s": 0.5,
        "traffic_mode": "log_replay",
        "world_kind": "original_source",
    }
    values = {
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "driving_direction_compliance": 1.0,
        "ego_progress": 0.68,
        "time_to_collision_within_bound": 1.0,
        "comfort": 1.0,
        "score": 0.86,
    }
    controls = []
    for role in ("student", "source_human", "reference"):
        v = dict(values)
        if role == "reference":
            v.update(score=0.99, ego_progress=0.99)
        controls.append(
            {
                "id": role,
                "role": role,
                "status": "scored",
                "anchor": deepcopy(anchor),
                "protocol": deepcopy(protocol),
                "trajectory_sha256": "c" * 64,
                "checkpoint_sha256": "d" * 64,
                "completed_horizon_s": 4.0,
                "trajectory_points": 8,
                "metrics": v,
            }
        )
    return {
        "kind": "same_anchor_native_controls",
        "status": "completed",
        "model": "chainflow_stage1",
        "source_split": "navtrain",
        "checkpoint": "weights/model.ckpt",
        "checkpoint_sha256": "d" * 64,
        "anchor": anchor,
        "protocol": protocol,
        "controls": controls,
    }


def control_review(data):
    from driversi.rsi.decisions.weaknesses import review_same_anchor_controls

    return review_same_anchor_controls(
        data, train_logs=["train-log"], protected_logs=["dev-log"], target_score=0.95
    )


def test_same_anchor_reference_routes_to_label_study_not_new_history_or_model_blame():
    report = control_review(control_evidence())
    assert report["intervention_class"] == "original_state_label_correction"
    assert [x["id"] for x in report["qualified_same_context_references"]] == ["reference"]
    assert report["observed_shortfalls"][0]["capability_hypothesis"] == "route_progress"
    assert not report["trainable"] and not report["synthesis_authorized"]
    assert not report["model_cause_established"] and not report["automatic_execution"]
    assert "equal_budget_original_label_training_control" in report["missing_evidence"]


@pytest.mark.parametrize(
    "fault",
    [
        "anchor",
        "cache",
        "protocol",
        "split",
        "protected",
        "checkpoint",
        "partial",
        "metrics",
        "nan",
        "duplicate",
        "missing_student",
    ],
)
def test_controls_reject_mixed_worlds_protocols_and_incomplete_evidence(fault):
    data = control_evidence()
    ref = data["controls"][-1]
    if fault == "anchor":
        ref["anchor"]["token"] = "other"
    if fault == "cache":
        ref["anchor"]["metric_cache_sha256"] = "e" * 64
    if fault == "protocol":
        ref["protocol"]["horizon_s"] = 2.0
    if fault == "split":
        data["source_split"] = "navtest"
    if fault == "protected":
        data["anchor"]["source_log"] = "dev-log"
        for row in data["controls"]:
            row["anchor"]["source_log"] = "dev-log"
    if fault == "checkpoint":
        data["controls"][0]["checkpoint_sha256"] = "f" * 64
    if fault == "partial":
        ref["completed_horizon_s"] = 2.0
    if fault == "metrics":
        del ref["metrics"]["drivable_area_compliance"]
    if fault == "nan":
        ref["metrics"]["score"] = float("nan")
    if fault == "duplicate":
        ref["id"] = data["controls"][0]["id"]
    if fault == "missing_student":
        data["controls"][0]["role"] = "reference"
    with pytest.raises(ValueError):
        control_review(data)


def test_unknown_student_is_evaluation_problem_not_driving_zero():
    data = control_evidence()
    row = data["controls"][0]
    row.update(status="outcome_unknown", reason="worker_exited_without_receipt")
    row.pop("metrics")
    report = control_review(data)
    assert report["student_score"] is None
    assert report["intervention_class"] == "evaluation_repair"
    assert len(report["control_failures"]) == 1


def test_reference_with_safety_shortfall_cannot_justify_relabel_plan():
    data = control_evidence()
    data["controls"][-1]["metrics"]["driving_direction_compliance"] = 0.5
    report = control_review(data)
    assert report["intervention_class"] == "diagnosis_pending"
    assert report["qualified_same_context_references"] == []


def test_already_qualified_student_does_not_trigger_targeted_synthesis():
    data = control_evidence()
    data["controls"][0]["metrics"].update(score=0.96, ego_progress=0.91)
    assert control_review(data)["intervention_class"] == "no_targeted_intervention"


def test_chainflow_persistent_safety_shortfall_is_visible_without_delta():
    student, data, feedback = evidence()
    student["model"] = "chainflow_stage1"
    data["directions"] = {"driving_direction_compliance": 1}
    for side in ("parent", "candidate"):
        data["metrics"][side] = {"driving_direction_compliance": 0.5}
        for row in data["paired_samples"]:
            row[side] = {"driving_direction_compliance": 0.5}
    report = review(student, data, feedback)
    assert len(report["signals"]) == 1
    assert report["signals"][0]["persistently_deficient_sample_ids"] == ["a", "b"]
    assert not report["signals"][0]["model_cause_established"]


def test_high_total_does_not_hide_direction_shortfall_in_zero_weight_native_metric():
    data = control_evidence()
    data["controls"][0]["metrics"].update(score=0.99, driving_direction_compliance=0.5)
    report = control_review(data)
    assert report["intervention_class"] != "no_targeted_intervention"
    assert any(x["metric"] == "driving_direction_compliance" for x in report["observed_shortfalls"])


def test_tiny_progress_change_retained_without_synthetic_curriculum_trigger():
    student, data, feedback = evidence()
    for side in ("parent", "candidate"):
        data["metrics"][side]["NC"] = 1.0
        for row in data["paired_samples"]:
            row[side]["NC"] = 1.0
    for row in data["paired_samples"]:
        row["candidate"]["EP"] = row["parent"]["EP"] - 1e-9
    data["metrics"]["candidate"]["EP"] = data["metrics"]["parent"]["EP"] - 1e-9
    result = review(student, data, feedback)
    signal = result["signals"][0]
    assert signal["delta"] < 0 and signal["regressed_sample_ids"] == ["a", "b"]
    assert signal["change_scale"] == "below_reporting_resolution"
    assert not signal["actionable_from_this_change"]
    assert result["next_action"] == "expand_stratified_evaluation"
    # A declared safety metric never receives the progress-reporting exclusion.
    data["safety_higher"] = ["EP"]
    result = review(student, data, feedback)
    assert result["signals"][0]["actionable_from_this_change"]
    assert len(result["local_safety_regressions"]) == 2
