import json
from pathlib import Path

import pytest

from driversi.data.catalog.store import identity
from driversi.rsi.decisions.intervention_policy import select_intervention
from driversi.rsi.experiments import intervention_campaign as m


def candidates():
    return [
        dict(id="first", priority=1, hypothesis="test a lower data dose"),
        dict(id="second", priority=2, hypothesis="test a longer fixed update budget"),
    ]


def test_policy_uses_feedback_without_equating_skill_and_model_gain():
    assert select_intervention(candidates(), [], 2)["intervention_id"] == "first"
    history = [dict(node="learning_review", status="no_benefit", intervention_id="first")]
    decision = select_intervention(candidates(), history, 1)
    assert decision["intervention_id"] == "second"
    assert not decision["policy_effectiveness_established"]


@pytest.mark.parametrize(
    "node",
    [
        "source_selection",
        "visual_admission",
        "paired_training",
        "independent_evaluation",
        "resume_verification",
    ],
)
def test_execution_problems_stop_learning_trials(node):
    assert (
        select_intervention(candidates(), [dict(node=node, status="blocked")], 2)["action"]
        == "repair"
    )


def test_positive_pilot_requests_validation_not_automatic_promotion():
    assert (
        select_intervention(candidates(), [dict(node="learning_review", status="pilot_signal")], 2)[
            "action"
        ]
        == "validate"
    )
    assert select_intervention(candidates(), [], 0)["reason"] == "trial_budget_exhausted"


def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(m.augustus_round, "_exp", lambda p: Path(p))
    program = tmp_path / "program"
    program.mkdir()
    rows = [
        dict(candidates()[0], program=str(program), program_id="p", optimizer_update_reservation=12)
    ]
    plan = dict(
        id="c",
        candidates=rows,
        seed_history=[],
        files={},
        max_trials=1,
        max_optimizer_updates_including_replays=12,
    )
    plan["campaign_id"] = identity(plan)
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "campaign.json").write_text(json.dumps(plan))
    return out, program


def test_preview_never_runs_training_and_assignment_is_sealed(tmp_path, monkeypatch):
    out, _ = campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(m.augustus_round, "run", lambda *a, **k: pytest.fail("preview trained"))
    assert m.advance(out)["status"] == "prepared"
    assert m.advance(out)["status"] == "prepared"
    p = out / "trial-000/assignment.json"
    value = json.loads(p.read_text())
    value["candidate"]["program_id"] = "other"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="assignment differs"):
        m.advance(out, execute=True)


def test_native_failure_is_preserved_and_resume_does_not_repeat_it(tmp_path, monkeypatch):
    out, program = campaign(tmp_path, monkeypatch)
    calls = []

    def run(*a, **k):
        calls.append(1)
        result = dict(node="joint_supervision", status="blocked", observed_gain=None)
        (program / "review.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(m.augustus_round, "run", run)
    assert m.advance(out, execute=True)["status"] == "trial_completed"
    assert m.advance(out, execute=True)["status"] == "repair"
    assert len(calls) == 1
    (program / "review.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        m.advance(out, execute=True)


def test_budget_includes_both_arms_and_replay(tmp_path, monkeypatch):
    out, _ = campaign(tmp_path, monkeypatch)
    p = out / "campaign.json"
    value = json.loads(p.read_text())
    value["max_optimizer_updates_including_replays"] = 11
    del value["campaign_id"]
    value["campaign_id"] = identity(value)
    p.write_text(json.dumps(value))
    monkeypatch.setattr(m.augustus_round, "run", lambda *a, **k: pytest.fail("over budget"))
    assert m.advance(out, execute=True)["decision"]["reason"] == "optimizer_budget_exhausted"


def test_campaign_tampering_is_rejected(tmp_path, monkeypatch):
    out, _ = campaign(tmp_path, monkeypatch)
    p = out / "campaign.json"
    value = json.loads(p.read_text())
    value["max_trials"] = 40
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="configuration changed"):
        m.advance(out)


@pytest.mark.parametrize("status,action", [("pilot_signal", "validate"), ("blocked", "repair")])
def test_final_trial_preserves_required_next_action(status, action):
    node = "learning_review" if status == "pilot_signal" else "independent_evaluation"
    assert (
        select_intervention(candidates(), [dict(node=node, status=status)], 0)["action"] == action
    )


@pytest.mark.parametrize(
    "field,value",
    [("optimizer_update_reservation", 0), ("status", "pilot_signal"), ("intervention_id", "other")],
)
def test_receipt_cannot_rewrite_budget_or_observation(tmp_path, monkeypatch, field, value):
    out, program = campaign(tmp_path, monkeypatch)

    def run(*a, **k):
        result = dict(node="joint_supervision", status="blocked", observed_gain=None)
        (program / "review.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(m.augustus_round, "run", run)
    m.advance(out, execute=True)
    p = out / "trial-000/result.json"
    receipt = json.loads(p.read_text())
    if field == "optimizer_update_reservation":
        receipt[field] = value
    else:
        receipt["observation"][field] = value
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from sealed evidence"):
        m.advance(out, execute=True)


@pytest.mark.parametrize("field", ["context", "program_id"])
def test_seed_review_context_cannot_be_substituted(tmp_path, field):
    p = tmp_path / "review.json"
    review = dict(
        context={"parent_assets_id": "p"},
        program_id="real",
        promoted=False,
        skill_gain_credited=False,
    )
    (tmp_path / "program.json").write_text(json.dumps(review))
    review[field] = {} if field == "context" else "forged"
    p.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="context/program"):
        m.checked_review(p)


def test_receipt_cannot_drop_raw_execution_evidence(tmp_path, monkeypatch):
    out, program = campaign(tmp_path, monkeypatch)
    raw = program / "worker.json"
    raw.write_text("{}")

    def run(*a, **k):
        result = dict(
            node="joint_supervision",
            status="blocked",
            observed_gain=None,
            execution_evidence=m.augustus_round._files([raw]),
        )
        (program / "review.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(m.augustus_round, "run", run)
    m.advance(out, execute=True)
    p = out / "trial-000/result.json"
    receipt = json.loads(p.read_text())
    del receipt["evidence"][str(raw)]
    p.write_text(json.dumps(receipt))
    raw.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="evidence closure"):
        m.advance(out, execute=True)


def test_prepared_seed_seals_transitive_historical_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(m.augustus_round, "_exp", lambda p: Path(p))
    program = tmp_path / "program"
    program.mkdir()
    (program / "program.json").write_text("{}")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"budget": {"max_optimizer_steps": 1}}))
    seed = tmp_path / "old-review.json"
    seed.write_text("{}")
    old_code = tmp_path / "old-scorer.py"
    old_code.write_text("version = 1")
    monkeypatch.setattr(
        m.augustus_round,
        "verify_program",
        lambda p: {"program_id": "p", "request_template": str(request)},
    )
    monkeypatch.setattr(m, "_comparison_key", lambda p: ("same",))
    monkeypatch.setattr(
        m,
        "checked_review",
        lambda p: dict(
            node="learning_review",
            status="no_benefit",
            observed_gain=-1,
            comparison_key=["same"],
            validated_dependency_files=m.augustus_round._files([old_code]),
        ),
    )
    out = tmp_path / "prepared"
    config = dict(
        schema_version=1,
        id="test",
        output=str(out),
        candidates=[dict(candidates()[0], program=str(program))],
        seed_reviews=[str(seed)],
        max_trials=1,
        max_optimizer_updates_including_replays=6,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    plan = m.prepare(config_path)
    assert str(old_code) in plan["files"]
    old_code.write_text("version = 2")
    with pytest.raises(ValueError, match="changed"):
        m.advance(out, execute=True)


def test_parent_relocation_requires_identical_complete_relative_contents():
    a = {"/original/weights/model.bin": "abc", "/original/weights/config.json": "def"}
    b = {"/frozen/weights/model.bin": "abc", "/frozen/weights/config.json": "def"}
    assert m._canonical_assets(a) == m._canonical_assets(b)
    b["/frozen/weights/config.json"] = "changed"
    assert m._canonical_assets(a) != m._canonical_assets(b)
    del b["/frozen/weights/config.json"]
    assert m._canonical_assets(a) != m._canonical_assets(b)
