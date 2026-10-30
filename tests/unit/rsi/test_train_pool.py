import json

import pytest

from driversi.rsi.decisions.evaluation_aggregate import METRICS
from driversi.rsi.decisions.train_pool import build_plan, read_scored_rows


def inputs():
    index = dict(
        split="navtrain",
        complete=True,
        expected_tokens=["train-a", "train-b"],
        records=[
            dict(token=t, source_log="train-log", kind="original_navtrain", anchor_index=3 + i)
            for i, t in enumerate(["train-a", "train-b"])
        ],
    )
    descriptors = dict(
        kind="navtrain_history_scene_index",
        records=[
            dict(
                token=t,
                source_log="train-log",
                conditions=["nearby_vehicle"],
                history=dict(continuous=i == 0),
            )
            for i, t in enumerate(["train-a", "train-b"])
        ],
    )
    rows = [
        dict(
            token="train-a",
            log_name="train-log",
            valid=True,
            score=0.0,
            **dict.fromkeys(METRICS, 1.0),
        )
    ]
    rows[0]["no_at_fault_collisions"] = 0.0
    return index, descriptors, rows


def test_training_zero_is_actionable_evidence_but_not_automatic_admission():
    plan = build_plan(*inputs(), {"checkpoint_sha256": "parent"})
    assert plan["test_feedback_used"] is False
    assert plan["partial"] is True
    assert plan["coverage"]["missing"] == 1
    row = plan["augmentation_candidates"][0]
    assert row["observed_shortfalls"] == ["collision"]
    assert row["next_action"] == "same_anchor_teacher_and_platform_controls"
    assert not row["model_cause_established"] and not row["trainable"]


def test_runtime_failure_is_never_zero_score_or_augmentation_label():
    index, desc, rows = inputs()
    rows.append(dict(token="train-b", log_name="train-log", valid=False, error="cache_error"))
    plan = build_plan(index, desc, rows, {})
    assert plan["coverage"]["all_recorded"] and not plan["coverage"]["all_scored"]
    assert plan["bins"]["zero"] == 1 and len(plan["augmentation_candidates"]) == 1
    assert plan["coverage"]["runtime_failures"] == 1


def test_irregular_history_is_retained_but_requires_repair_before_generation():
    index, desc, rows = inputs()
    rows[0].update(token="train-b")
    plan = build_plan(index, desc, rows, {})
    assert plan["augmentation_candidates"][0]["next_action"] == "repair_history_support"


def test_high_total_score_cannot_hide_direction_or_other_safety_shortfalls():
    index, desc, rows = inputs()
    rows[0].update(score=0.99, no_at_fault_collisions=1.0, driving_direction_compliance=0.5)
    plan = build_plan(index, desc, rows, {})
    assert plan["augmentation_candidates"][0]["selection_triggers"] == [
        "safety_submetric_shortfall"
    ]
    rows[0].update(driving_direction_compliance=1.0, ego_progress=0.98)
    assert not build_plan(index, desc, rows, {})["augmentation_candidates"]


@pytest.mark.parametrize("mutation", ["split", "token", "source", "duplicate"])
def test_test_sources_and_bad_membership_never_enter_train_plan(mutation):
    index, desc, rows = inputs()
    if mutation == "split":
        index["split"] = "navtest"
    if mutation == "token":
        rows[0]["token"] = "test-token"
    if mutation == "source":
        rows[0]["log_name"] = "test-log"
    if mutation == "duplicate":
        rows = rows * 2
    with pytest.raises(ValueError):
        build_plan(index, desc, rows, {})


def test_shard_reader_refuses_test_feedback_and_incomplete_json_lines(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(dict(split="navtest")))
    with pytest.raises(ValueError, match="Test evaluation"):
        read_scored_rows([tmp_path])
    path.write_text(json.dumps(dict(split="navtrain", purpose="training_pool_mining", shard=0)))
    (tmp_path / "scores.jsonl").write_text('{"token":"a"}\n{"token":')
    rows, _ = read_scored_rows([tmp_path])
    assert rows == [dict(token="a")]


def test_completed_old_scores_cannot_be_used_under_a_new_checkpoint(tmp_path):
    identity = dict(
        split="navtrain", purpose="training_pool_mining", checkpoint_sha256="old", shard=0
    )
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    (tmp_path / "scores.jsonl").write_text('{"token":"train-a"}\n')
    requested = {k: v for k, v in identity.items() if k != "shard"}
    requested["checkpoint_sha256"] = "new"
    with pytest.raises(ValueError, match="requested native identity"):
        read_scored_rows([tmp_path], requested)
