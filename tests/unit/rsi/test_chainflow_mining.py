import copy

import pytest

from driversi.rsi.decisions.weaknesses import PROFILES, review_mining


def row():
    return dict(
        source_log="training-log",
        token="scene-a",
        status="scored",
        metrics={
            "score": 0.86,
            **{k: (0.67 if k == "ego_progress" else 1.0) for k in PROFILES["chainflow_stage1"]},
        },
    )


def review(rows):
    return review_mining(
        rows,
        train_logs=["training-log"],
        protected_logs=["heldout"],
        model="chainflow_stage1",
        checkpoint="fixed-parent",
        target_score=0.95,
    )


def test_low_progress_requests_diagnosis_not_training():
    result = review([row()])
    c = result["candidates"][0]
    assert c["capability_hypotheses"][0]["capability"] == "route_progress"
    assert not c["synthesis_authorized"] and not c["model_cause_established"] and not c["trainable"]


def test_scoring_failure_is_not_zero_model_score():
    r = row()
    r.update(status="failed", reason="missing cache")
    r.pop("metrics")
    result = review([r])
    assert not result["candidates"] and result["scored_count"] == 0


def test_heldout_duplicate_and_missing_metrics_are_rejected():
    r = row()
    r["source_log"] = "heldout"
    with pytest.raises(ValueError, match="non-training"):
        review([r])
    with pytest.raises(ValueError, match="Duplicate"):
        review([row(), row()])
    r = copy.deepcopy(row())
    r["metrics"].pop("comfort")
    with pytest.raises(ValueError, match="Incomplete"):
        review([r])
