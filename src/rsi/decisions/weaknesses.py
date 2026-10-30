"""Observed weakness evidence and bounded experiment advice, not causal diagnosis.

RSIGym separates experiment choice from fixed services. This module specializes
that boundary for existing DriveHarness feedback; it neither trains nor changes
an evaluator. Development examples must never become correction-training inputs.
"""

import math
from collections import defaultdict

from driversi.training.sampling.environment import source_log_group

# Native quantities remain separate. Their scale cannot rank different families.
PROFILES = {
    "augustus": {
        "NC": (1, "collision_avoidance", "same_anchor_collision_attribution"),
        "DAC": (1, "road_compliance", "map_and_tracking_parity"),
        "DDC": (1, "direction_compliance", "route_direction_and_connector_check"),
        "EP": (1, "route_progress", "timing_and_feasible_route_probe"),
        "TTC": (1, "anticipatory_interaction", "responsive_actor_and_timing_probe"),
        "C": (1, "comfort", "tracking_and_trajectory_smoothness_probe"),
    },
    "driverl": {
        "contact_free_fraction": (
            1,
            "world_contact_diagnostic",
            "ego_background_and_platform_contact_attribution",
        ),
        "travelled_m": (
            1,
            "short_horizon_motion_diagnostic",
            "legal_route_progress_over_complete_horizon",
        ),
    },
}


PROFILES["chainflow_stage1"] = {
    "no_at_fault_collisions": (1, "collision_avoidance", "same_anchor_collision_attribution"),
    "drivable_area_compliance": (1, "road_compliance", "map_and_tracking_parity"),
    "driving_direction_compliance": (
        1,
        "direction_compliance",
        "route_direction_and_connector_check",
    ),
    "ego_progress": (1, "route_progress", "timing_and_feasible_route_probe"),
    "time_to_collision_within_bound": (
        1,
        "anticipatory_interaction",
        "responsive_actor_and_timing_probe",
    ),
    "comfort": (1, "comfort", "tracking_and_trajectory_smoothness_probe"),
}


# Reference the native compliance scale, not an invented loss/progress threshold.
# Below one is an observed constraint shortfall; its cause is still unresolved.
COMPLIANCE_REFERENCE = {
    "augustus": {key: 1.0 for key in ("NC", "DAC", "DDC", "TTC", "C")},
    "driverl": {"contact_free_fraction": 1.0},
    "chainflow_stage1": {
        key: 1.0
        for key in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "time_to_collision_within_bound",
            "comfort",
        )
    },
}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite native evaluation metric required")
    return float(value)


def validate_samples(evaluation):
    """Validate paired records before they can become weakness evidence."""
    rows = evaluation.get("paired_samples", [])
    if not isinstance(rows, list):
        raise ValueError("Paired samples must be a list")
    if not rows:
        return []
    declared = {
        source_log_group(x) for x in evaluation.get("source_groups", {}).get("development", [])
    }
    train = {source_log_group(x) for x in evaluation.get("source_groups", {}).get("train", [])}
    identities = set()
    keys = set(evaluation["directions"])
    for row in rows:
        if set(row) != {"sample_id", "source_log", "parent", "candidate"}:
            raise ValueError("Malformed paired weakness sample")
        if (
            not isinstance(row["sample_id"], str)
            or not row["sample_id"]
            or row["sample_id"] in identities
        ):
            raise ValueError("Missing or duplicate paired sample identity")
        identities.add(row["sample_id"])
        group = source_log_group(row["source_log"])
        if group not in declared or group in train:
            raise ValueError("Paired sample source outside heldout logs")
        if (
            not keys <= set(row["parent"])
            or set(row["parent"]) != set(row["candidate"])
            or set(row["parent"]) != set(rows[0]["parent"])
        ):
            raise ValueError("Paired sample metric coverage differs")
        for side in ("parent", "candidate"):
            for value in row[side].values():
                _number(value)
    expected = evaluation.get("evaluated_sample_ids")
    if not isinstance(expected, list) or len(expected) != len(rows) or set(expected) != identities:
        raise ValueError("Paired sample coverage differs from executed evaluation")
    for side in ("parent", "candidate"):
        for key in rows[0][side]:
            if not all(key in row[side] for row in rows) or key not in evaluation["metrics"][side]:
                raise ValueError("Paired metric missing from native aggregate")
            average = sum(row[side][key] for row in rows) / len(rows)
            if not math.isclose(
                average, _number(evaluation["metrics"][side][key]), abs_tol=1e-8, rel_tol=1e-7
            ):
                raise ValueError("Paired metrics disagree with native aggregate")
    return rows


def review(student, evaluation, feedback):
    if student["status"] != "completed" or evaluation["status"] != "completed":
        raise ValueError("Weakness review requires completed evaluation")
    if (
        student["context"] != evaluation["context"]
        or evaluation["evaluated_checkpoint"] != student["checkpoint"]
    ):
        raise ValueError("Weakness review checkpoint/context mismatch")
    rows = validate_samples(evaluation)
    groups = evaluation.get("source_groups", {}).get("development", [])
    by_log = defaultdict(list)
    for row in rows:
        by_log[source_log_group(row["source_log"])].append(row)
    signals = []
    for metric, (direction, capability, probe) in PROFILES.get(student["model"], {}).items():
        values = evaluation["metrics"]
        if metric not in values["parent"] or metric not in values["candidate"]:
            continue
        parent, candidate = (_number(values[side][metric]) for side in ("parent", "candidate"))
        delta = candidate - parent
        regressions = [
            row["sample_id"]
            for row in rows
            if metric in row["parent"]
            and (row["candidate"][metric] - row["parent"][metric]) * direction < 0
        ]
        reference = COMPLIANCE_REFERENCE.get(student["model"], {}).get(metric)
        shortfall = reference is not None and candidate < reference
        deficient = [
            row["sample_id"]
            for row in rows
            if reference is not None and row["candidate"][metric] < reference
        ]
        persistent = [
            row["sample_id"]
            for row in rows
            if reference is not None
            and row["parent"][metric] < reference
            and row["candidate"][metric] < reference
        ]
        if delta * direction >= 0 and not regressions and not shortfall and not deficient:
            continue
        log_changes = []
        for log, examples in sorted(by_log.items()):
            if metric in examples[0]["parent"]:
                log_changes.append(
                    {
                        "source_group": log,
                        "samples": len(examples),
                        "mean_delta": sum(
                            x["candidate"][metric] - x["parent"][metric] for x in examples
                        )
                        / len(examples),
                    }
                )
        per_sample_deltas = [
            row["candidate"][metric] - row["parent"][metric]
            for row in rows
            if metric in row["parent"]
        ]
        below_reporting_resolution = (
            metric in ("ego_progress", "EP")
            and metric not in evaluation.get("safety_higher", [])
            and max([abs(delta), *map(abs, per_sample_deltas)]) <= 1e-7
            and not shortfall
            and not deficient
        )
        signals.append(
            {
                "metric": metric,
                "capability_hypothesis": capability,
                "parent": parent,
                "candidate": candidate,
                "delta": delta,
                "change_scale": "below_reporting_resolution"
                if below_reporting_resolution
                else "observed_change_or_shortfall",
                "reporting_resolution": 1e-7 if metric in ("ego_progress", "EP") else None,
                "actionable_from_this_change": not below_reporting_resolution,
                "regressed_sample_ids": regressions,
                "native_full_compliance_reference": reference,
                "candidate_compliance_shortfall": shortfall,
                "deficient_sample_ids": deficient,
                "persistently_deficient_sample_ids": persistent,
                "per_log": log_changes,
                "causal_probe": probe,
                "model_cause_established": False,
            }
        )
    local_safety_regressions = [
        {"sample_id": row["sample_id"], "metric": key}
        for row in rows
        for key in evaluation.get("safety_higher", [])
        if key in row["parent"] and row["candidate"][key] < row["parent"][key]
    ]
    track = {
        "name": "data",
        "allowed_changes": ["training_source_selection", "admitted_relabels", "data_mixture"],
        "frozen": [
            "starting_parent_checkpoint",
            "native_architecture",
            "native_loss",
            "native_optimizer",
            "update_budget",
            "evaluation_protocol",
            "development_sources",
        ],
        "control": "same_parent_original_labels_equal_update_budget",
        "target_source": "matching_training_logs_only_never_copy_development_cases",
        "targeted_trainable": False,
        "required_before_training": [
            "verified_weakness_slice_on_separate_original_logs",
            "runtime_and_world_validity",
            "admitted_model_specific_supervision",
            "frozen_target_and_unaffected_evaluation_slices",
        ],
    }
    if student["model"] == "driverl":
        track["data_contract"] = (
            "live_current_policy_rollouts_with_sampling_likelihoods_no_teacher_as_on_policy"
        )
    elif student["model"] == "augustus":
        track["data_contract"] = "history_trajectory_and_branch_consistent_prediction_language"
    return {
        "schema_version": 1,
        "rule": "weakness-evidence-v1",
        "model": student["model"],
        "checkpoint": student["checkpoint"],
        "context": student["context"],
        "evaluator_context": evaluation.get("evaluation_context"),
        "metric_scope": evaluation.get("metric_scope", "unspecified"),
        "signals": signals,
        "local_safety_regressions": local_safety_regressions,
        "support": {
            "paired_samples": len(rows),
            "original_log_groups": len(by_log) if rows else None,
            "declared_original_log_groups": len({source_log_group(x) for x in groups}),
            "per_sample_available": bool(rows),
            "counts_are_not_independent_trials": True,
        },
        "next_action": "diagnose_observed_weakness"
        if any(x["actionable_from_this_change"] for x in signals)
        else "expand_stratified_evaluation",
        "experiment_proposal": track,
        "other_tracks": {
            "harness": "freeze_model_weights_modify_execution_only",
            "joint": "separate_data_and_harness_ablation_before_attributing_gain",
        },
        "budget_order": [
            "reuse_existing_evidence",
            "same_anchor_short_probe",
            "bounded_relabel_search",
            "small_matched_training",
            "expanded_development",
            "milestone_external_test",
        ],
        "feedback_credit": "measured_student_gain_vs_equal_budget_control_with_safety_and_cost",
        "causal_attribution": "unresolved",
        "statistical_significance": "not_claimed",
        "automatic_execution": False,
        "promoted": False,
        "pilot_metric_improved": feedback["pilot_metric_improved"] and not local_safety_regressions,
    }


def review_mining(scores, *, train_logs, protected_logs, model, checkpoint, target_score):
    """Rank original-source probes, never confuse them with trained-pair gains.

    A low score identifies a diagnostic candidate. It does not establish model
    causality, qualify a different branch, or authorize training.
    """
    if model not in PROFILES or not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Known model and explicit checkpoint required")
    target = _number(target_score)
    if not 0 <= target <= 1:
        raise ValueError("Native mining target must lie in [0,1]")
    train = {source_log_group(x) for x in train_logs}
    protected = {source_log_group(x) for x in protected_logs}
    if not train or train & protected:
        raise ValueError("Disjoint declared training and protected sources required")
    seen, candidates, failures = set(), [], []
    for row in scores:
        key = (row["source_log"], row["token"])
        group = source_log_group(key[0])
        if key in seen or group not in train or group in protected:
            raise ValueError("Duplicate or non-training mining source")
        seen.add(key)
        if row["status"] != "scored":
            failures.append(
                {
                    "source_log": key[0],
                    "token": key[1],
                    "reason": row.get("reason", "unscored"),
                    "next_action": "repair_evaluation_input",
                }
            )
            continue
        metrics = row["metrics"]
        if not {"score", *PROFILES[model]} <= set(metrics):
            raise ValueError("Incomplete native submetrics cannot diagnose a weakness")
        if any(not 0 <= _number(v) <= 1 for v in metrics.values()):
            raise ValueError("Invalid native metric range")
        if metrics["score"] >= target:
            continue
        candidates.append(
            {
                "source_log": key[0],
                "token": key[1],
                "score": metrics["score"],
                "capability_hypotheses": [
                    {
                        "metric": metric,
                        "value": metrics[metric],
                        "capability": capability,
                        "required_probe": probe,
                    }
                    for metric, (_, capability, probe) in PROFILES[model].items()
                    if metrics[metric] < 1
                ],
                "next_action": "same_anchor_runtime_world_and_task_diagnosis",
                "model_cause_established": False,
                "synthesis_authorized": False,
                "trainable": False,
            }
        )
    return {
        "kind": "training_source_mining_review",
        "model": model,
        "checkpoint": checkpoint,
        "target_score": target,
        "candidates": sorted(candidates, key=lambda x: (x["score"], x["source_log"], x["token"])),
        "evaluation_failures": failures,
        "scored_count": len(scores) - len(failures),
        "scope": "small_training_probe_not_independent_benefit_or_causal_attribution",
        "trainable": False,
    }


def review_same_anchor_controls(evidence, *, train_logs, protected_logs, target_score):
    """Route a native NAVSIM control experiment without claiming model causality.

    Callers must first verify the sealed artifact containing this receipt. Hashes
    here bind controls to one source world and scoring implementation; this pure
    validator does not authenticate files or qualify training supervision.
    """
    model = evidence.get("model")
    if model != "chainflow_stage1" or evidence.get("source_split") != "navtrain":
        raise ValueError("This native control protocol requires ChainFlow navtrain evidence")
    if (
        evidence.get("kind") != "same_anchor_native_controls"
        or evidence.get("status") != "completed"
    ):
        raise ValueError("Completed same-anchor control receipt required")
    checkpoint = evidence.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Explicit evaluated checkpoint required")

    def sha(value):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Explicit lowercase SHA256 required")
        return value

    sha(evidence.get("checkpoint_sha256"))
    anchor = evidence.get("anchor", {})
    if set(anchor) != {"source_log", "token", "timestamp_us", "metric_cache_sha256"}:
        raise ValueError("Complete source anchor and metric cache identity required")
    if any(not isinstance(anchor[k], str) or not anchor[k] for k in ("source_log", "token")):
        raise ValueError("Nonempty source log and token required")
    if type(anchor["timestamp_us"]) is not int or anchor["timestamp_us"] <= 0:
        raise ValueError("Source anchor timestamp required")
    sha(anchor["metric_cache_sha256"])
    train = {source_log_group(x) for x in train_logs}
    protected = {source_log_group(x) for x in protected_logs}
    group = source_log_group(anchor["source_log"])
    if not train or train & protected or group not in train or group in protected:
        raise ValueError("Same-anchor intervention must use isolated training sources")
    protocol = evidence.get("protocol", {})
    if set(protocol) != {
        "id",
        "scorer_sha256",
        "horizon_s",
        "interval_s",
        "traffic_mode",
        "world_kind",
    }:
        raise ValueError("Complete native scoring protocol required")
    if (
        protocol["id"] != "navsim-v1-native-original-source-4s"
        or protocol["traffic_mode"] != "log_replay"
        or protocol["world_kind"] != "original_source"
    ):
        raise ValueError("Counterfactual and source protocols cannot be mixed")
    sha(protocol["scorer_sha256"])
    if _number(protocol["horizon_s"]) != 4.0 or _number(protocol["interval_s"]) != 0.5:
        raise ValueError("Native eight-point four-second control window required")
    target = _number(target_score)
    if not 0 <= target <= 1:
        raise ValueError("Native target must lie in [0,1]")
    controls = evidence.get("controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("Executed controls required")
    identities, student, references, failures = set(), [], [], []
    metrics_required = {"score", *PROFILES[model]}
    for control in controls:
        ident = control.get("id")
        if not isinstance(ident, str) or not ident or ident in identities:
            raise ValueError("Unique control identity required")
        identities.add(ident)
        if control.get("anchor") != anchor or control.get("protocol") != protocol:
            raise ValueError("Controls differ in anchor, source world, or scoring protocol")
        role = control.get("role")
        if role not in {"student", "source_human", "reference", "timing_probe"}:
            raise ValueError("Unknown control role")
        if role == "student":
            student.append(control)
            if control.get("checkpoint_sha256") != evidence["checkpoint_sha256"]:
                raise ValueError("Student checkpoint identity differs")
        sha(control.get("trajectory_sha256"))
        if control.get("status") != "scored":
            if control.get("status") not in {"failed", "outcome_unknown"} or not control.get(
                "reason"
            ):
                raise ValueError("Unscored control must preserve its failure or unknown outcome")
            failures.append({"id": ident, "status": control["status"], "reason": control["reason"]})
            continue
        if _number(control.get("completed_horizon_s")) != protocol["horizon_s"]:
            raise ValueError("Partial windows cannot be compared as complete native scores")
        if type(control.get("trajectory_points")) is not int or control["trajectory_points"] != 8:
            raise ValueError("Eight actual future trajectory points required")
        values = control.get("metrics", {})
        if set(values) != metrics_required or any(
            not 0 <= _number(v) <= 1 for v in values.values()
        ):
            raise ValueError("Complete finite native submetrics required")
        if (
            role in {"source_human", "reference"}
            and values["score"] >= target
            and all(values[k] == v for k, v in COMPLIANCE_REFERENCE[model].items())
        ):
            references.append({"id": ident, "role": role, "score": values["score"]})
    if len(student) != 1:
        raise ValueError("Exactly one checkpoint-bound student control required")
    current = student[0]
    deficits = []
    if current["status"] != "scored":
        intervention, next_action = (
            "evaluation_repair",
            "resolve_unscored_student_before_weakness_selection",
        )
    elif current["metrics"]["score"] >= target and all(
        current["metrics"][k] == value for k, value in COMPLIANCE_REFERENCE[model].items()
    ):
        intervention, next_action = (
            "no_targeted_intervention",
            "retain_baseline_coverage_and_expand_stratified_mining",
        )
    else:
        deficits = [
            {
                "metric": metric,
                "value": current["metrics"][metric],
                "capability_hypothesis": profile[1],
                "required_probe": profile[2],
            }
            for metric, profile in PROFILES[model].items()
            if current["metrics"][metric] < 1
        ]
        if references:
            intervention, next_action = (
                "original_state_label_correction",
                "validate_reference_execution_then_matched_label_experiment",
            )
        else:
            intervention, next_action = (
                "diagnosis_pending",
                "bounded_same_anchor_reference_search_or_platform_attribution",
            )
    return {
        "schema_version": 1,
        "kind": "same_anchor_intervention_review",
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "source_split": "navtrain",
        "source_group": group,
        "anchor": dict(anchor),
        "protocol": dict(protocol),
        "target_score": target,
        "student_control_id": current["id"],
        "student_score": current.get("metrics", {}).get("score")
        if current["status"] == "scored"
        else None,
        "observed_shortfalls": deficits,
        "qualified_same_context_references": references,
        "control_failures": failures,
        "intervention_class": intervention,
        "next_action": next_action,
        "plan_validity": "diagnostic_recommendation_only",
        "synthesis_decision": "blocked_no_evidence_new_observations_are_required",
        "synthesis_authorized": False,
        "model_cause_established": False,
        "causal_attribution": "unresolved",
        "missing_evidence": [
            "independent_source_log_weakness_replication",
            "world_and_route_validity_attribution",
            "complete_intervention_specific_trajectory_and_visual_admission",
            "equal_budget_original_label_training_control",
            "heldout_student_gain_and_unaffected_slice_regression_checks",
        ],
        "counts_are_not_independent_trials": True,
        "trainable": False,
        "automatic_execution": False,
        "promoted": False,
    }
