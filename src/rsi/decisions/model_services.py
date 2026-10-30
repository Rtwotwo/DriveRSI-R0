"""Model-specific service boundaries; these are not learned driving skills."""
import copy
import math

STUDENTS = {
    "chainflow_stage1": {"history_frames": 4, "future_points": 10, "interval_seconds": .5,
        "paired_skill_feedback": True, "new_history_training": True,
        "language_supervision": None},
    "augustus": {"history_frames": 4, "future_points": 8, "interval_seconds": .5,
        "cameras": ["CAM_L0", "CAM_F0", "CAM_R0"],
        "native_video_size": [2048, 512], "language_supervision": "prediction",
        "paired_skill_feedback": False, "new_history_training": True,
        "paired_training_service": True,
        "native_sft_scope": "original_or_independently_admitted_counterfactual_prediction",
        "grpo_bridge": False},
}


def student_contract(model):
    if model not in STUDENTS:
        raise ValueError("No declared model-service skill contract for this student")
    return copy.deepcopy(STUDENTS[model])


def diagnose_augustus_branch(branch):
    """Audit an existing physical branch without evaluating the student.

    Teacher score is a label gate, never an estimate of student ability. A
    diagnosis can run on rejected branches; it cannot authorize their training.
    """
    if branch.get("kind") != "selected_native_branch":
        raise ValueError("A selected physical branch is required")
    if branch.get("branch") != "teacher":
        raise ValueError("Teacher diagnosis requires the selected teacher branch, not baseline")
    evidence = branch.get("trajectory_evidence", {})
    score = evidence.get("score", {})
    value = score.get("score")
    numeric = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    score_pass = (numeric and value >= .95 and score.get("native_score_threshold") == .95
                  and score.get("protocol") == "navsim-v1-native-math-executed-counterfactual-v1")
    checks = {
        "arrival_contacts_absent": type(evidence.get("arrival_contact_records")) is int and evidence["arrival_contact_records"] == 0,
        "future_contacts_absent": type(evidence.get("future_contact_records")) is int and evidence["future_contact_records"] == 0,
        "producer_changed_history_evidence": evidence.get("history_changed") is True,
        "restore_exact": evidence.get("restore_exact") is True,
        "native_teacher_pdms_ge_95": score_pass,
        "future_support": type(branch.get("future_frames")) is int and branch["future_frames"] >= 8,
        "producer_trajectory_admitted": branch.get("teacher_qualified") is True,
    }
    failed = [key for key, passed in checks.items() if not passed]
    contacts = not checks["arrival_contacts_absent"] or not checks["future_contacts_absent"]
    action = ("audit_world_contacts_before_synthesis" if contacts else
              "repair_physical_branch_or_teacher_before_synthesis" if failed else
              "build_and_verify_branch_consistent_augustus_supervision")
    return {
        "kind": "augustus_model_service_diagnosis", "schema_version": 1,
        "skills": [
            {"id": "physical_validity_audit", "version": 1,
             "checks": {k: checks[k] for k in ("arrival_contacts_absent", "future_contacts_absent", "restore_exact")}},
            {"id": "same_anchor_teacher_audit", "version": 1,
             "checks": {k: checks[k] for k in ("native_teacher_pdms_ge_95", "producer_changed_history_evidence", "future_support", "producer_trajectory_admitted")}},
            {"id": "augustus_supervision_audit", "version": 1, "checks": {
                "counterfactual_joint_cache_connected": True, "skill_paired_feedback_connected": False,
                "this_branch_joint_supervision_verified": False}},
        ],
        "teacher_score100": value * 100 if numeric else None,
        "score_scope": "native_4s_math_on_executed_counterfactual_world_not_student_or_benchmark_score",
        "failed_physical_checks": failed,
        "causal_attribution": "unresolved", "student_failure_established": False,
        "student_evaluated": False, "selection_provenance_rebound": False,
        "next_action": action, "student_contract": student_contract("augustus"),
        "joint_supervision_requirements": [
            "four_actual_history_frames_three_cameras_same_physical_branch",
            "eight_future_poses_at_native_0p5s_no_invented_or_resampled_points",
            "prediction_targets_from_same_branch_objects_and_future",
            "visual_and_trajectory_admission_for_same_branch",
            "native_action_normalization_and_losses_unchanged",
        ],
        "learning_blockers": ["same_branch_visual_and_native_label_admission_required",
                              "prospective_skill_attribution_not_verified"],
        "teacher_samples_are_on_policy": False, "trainable": False,
        "training_executed": False, "promoted": False,
    }
