from copy import deepcopy

import pytest

from driversi.rsi.plugins.chainflow import compare
from driversi.rsi.workflow.providers import builtin_providers


def arm():
    metrics = {
        "score": 0.9,
        "ego_progress": 0.8,
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "driving_direction_compliance": 1.0,
        "time_to_collision_within_bound": 1.0,
        "comfort": 1.0,
    }
    return {
        "student": {
            "model": "chainflow_stage1",
            "status": "completed",
            "context": "one",
            "checkpoint": "/exp/model",
            "total_updates": 2,
            "sample_trace": [
                {"optimizer_step": 1, "sample_ids": ["original"]},
                {"optimizer_step": 2, "sample_ids": ["original"]},
            ],
            "consumed": {},
        },
        "evaluation": {
            "status": "completed",
            "context": "one",
            "evaluated_checkpoint": "/exp/model",
            "metrics": {"parent": deepcopy(metrics), "candidate": deepcopy(metrics)},
            "directions": {k: 1 for k in metrics},
            "primary_metric": "score",
            "safety_higher": [
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "driving_direction_compliance",
                "time_to_collision_within_bound",
                "comfort",
            ],
            "source_groups": {"train": ["train"], "development": ["dev"]},
            "paired_samples": [
                {
                    "sample_id": "d",
                    "source_log": "dev",
                    "parent": deepcopy(metrics),
                    "candidate": deepcopy(metrics),
                }
            ],
            "evaluated_sample_ids": ["d"],
            "consumed": {},
        },
        "plan": {
            "phase": "resume",
            "model_config": {"parent": "frozen"},
            "resolved": {
                "request": {
                    "pilot": {"seed": 42, "batch_size": 1},
                    "budget": {"max_optimizer_steps": 1},
                }
            },
        },
        "dataset": {
            "splits": {
                "train": [{"sample_id": "original", "kind": "original_navtrain"}],
                "development": [{"sample_id": "d"}],
            },
            "files": {"f": "sha"},
        },
    }


def improve(value):
    for where in (
        value["evaluation"]["metrics"]["candidate"],
        value["evaluation"]["paired_samples"][0]["candidate"],
    ):
        where["score"] = 0.95
        where["ego_progress"] = 0.9


def test_provider_consumes_existing_unified_artifacts():
    provider = builtin_providers().get("chainflow_paired_feedback")
    assert dict(provider.inputs)["control_student"] == "unified_training_result"


def test_same_data_cannot_claim_augmentation_gain():
    a, b = arm(), arm()
    improve(b)
    report = compare(a, b)
    assert report["scope"] == "identical_data_plumbing_only"
    assert not report["observed_pilot_benefit"]


def test_equal_budget_original_pool_augmentation_can_report_only_pilot_gain():
    a, b = arm(), arm()
    b["dataset"]["splits"]["train"].append({"sample_id": "new", "kind": "original_navtrain"})
    b["student"]["sample_trace"][1]["sample_ids"] = ["new"]
    improve(b)
    report = compare(a, b)
    assert report["observed_pilot_benefit"]
    assert not report["synthetic_gain_demonstrated"] and not report["promoted"]


def test_local_safety_loss_blocks_score_gain():
    a, b = arm(), arm()
    b["dataset"]["splits"]["train"].append({"sample_id": "new", "kind": "original_navtrain"})
    improve(b)
    for where in (
        b["evaluation"]["metrics"]["candidate"],
        b["evaluation"]["paired_samples"][0]["candidate"],
    ):
        where["no_at_fault_collisions"] = 0
    report = compare(a, b)
    assert not report["observed_pilot_benefit"]
    assert report["regressions_against_control"]


def test_different_completed_updates_are_not_a_matched_pair():
    a, b = arm(), arm()
    b["student"]["total_updates"] = 3
    with pytest.raises(ValueError, match="optimizer updates"):
        compare(a, b)


def test_replacing_heldout_records_cannot_improve_pair():
    a, b = arm(), arm()
    b["dataset"]["splits"]["development"] = [{"sample_id": "easier"}]
    with pytest.raises(ValueError, match="development coverage"):
        compare(a, b)


def test_unconsumed_new_sample_cannot_earn_augmentation_credit():
    a, b = arm(), arm()
    b["dataset"]["splits"]["train"].append({"sample_id": "new", "kind": "original_navtrain"})
    improve(b)
    report = compare(a, b)
    assert not report["observed_pilot_benefit"]
    assert report["exposed_added_sample_ids"] == []


def test_reordered_original_pool_is_not_pure_augmentation():
    a, b = arm(), arm()
    a["dataset"]["splits"]["train"].append({"sample_id": "second", "kind": "original_navtrain"})
    b["dataset"]["splits"]["train"].insert(0, {"sample_id": "second", "kind": "original_navtrain"})
    with pytest.raises(ValueError, match="sample order"):
        compare(a, b)


def test_effective_recipe_difference_rejected():
    a, b = arm(), arm()
    a["student"]["effective_training_recipe"] = {"warmup": 20}
    b["student"]["effective_training_recipe"] = {"warmup": 100}
    with pytest.raises(ValueError, match="effective training recipes"):
        compare(a, b)


def test_one_log_gain_never_becomes_stable_improvement():
    a, b = arm(), arm()
    b["dataset"]["splits"]["train"].append({"sample_id": "new", "kind": "original_navtrain"})
    b["student"]["sample_trace"][1]["sample_ids"] = ["new"]
    improve(b)
    result = compare(a, b)
    assert result["observed_pilot_benefit"]
    assert result["paired_evidence"]["status"] == "insufficient_independent_logs"
    assert not result["stable_improvement_demonstrated"]
    assert result["next_action"] == "expand_independent_log_evaluation"


def test_partial_batch_exposure_cannot_claim_full_intervention_gain():
    a, b = arm(), arm()
    b["dataset"]["splits"]["train"].extend(
        [
            {"sample_id": "seen", "kind": "synthetic_navtrain"},
            {"sample_id": "unseen", "kind": "synthetic_navtrain"},
        ]
    )
    b["student"]["sample_trace"][1]["sample_ids"] = ["seen"]
    improve(b)
    report = compare(a, b)
    assert report["exposed_added_sample_ids"] == ["seen"]
    assert report["unconsumed_added_sample_ids"] == ["unseen"]
    assert not report["observed_pilot_benefit"]
    assert not report["synthetic_gain_demonstrated"]
    assert report["next_action"] == "verify_real_data_intervention"
