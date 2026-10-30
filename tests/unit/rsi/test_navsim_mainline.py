import json
import sqlite3
from pathlib import Path

import pytest

from driversi.rsi.decisions.augmentation import choose, validate_advice
from driversi.rsi.decisions.learning_decisions import CHECKS
from driversi.rsi.decisions.llm_advisor import validate_config
from driversi.rsi.plugins import navsim
from driversi.rsi.workflow.providers import builtin_providers


def case(identity="a", **overrides):
    return {
        "id": identity,
        "node": "planning",
        "failure_class": "interaction",
        "source_log": "train_001_002",
        "split": "train",
        "parent_version": "model-v1",
        "evaluator_version": "native-v1",
        "input_kind": "original",
        "checks": {
            **dict.fromkeys(CHECKS, "unknown"),
            "runtime_parity": "pass",
            "world_validity": "pass",
            "task": "fail",
        },
        "evidence": ["evidence.json"],
        **overrides,
    }


def plan(rows, **overrides):
    cost = {"wall_seconds": 10, "gpu_seconds": 5, "score_queries": 2}
    return choose(
        rows,
        **{
            "train_logs": ["train_001_002"],
            "development_logs": ["dev_001_002"],
            "protected_logs": ["test"],
            "budget": cost,
            "costs": {"new_state": cost, "relabel": cost},
            **overrides,
        },
    )


def test_shifted_state_requires_history_future_and_counterfactual_score():
    row = plan([case()])["decisions"][0]
    assert row["status"] == "reserved" and row["mode"] == "new_state"
    assert "same_anchor_teacher" in row["required"] and "counterfactual_score" in row["required"]
    assert "history_and_future_gs" in row["required"] and not row["trainable"]


def test_original_label_is_cheaper_route_for_non_interaction_failure():
    row = plan([case(failure_class="route_progress")])["decisions"][0]
    assert row["mode"] == "relabel" and row["generator"] == "drivetto_logged"


@pytest.mark.parametrize(
    "check,action", [("runtime_parity", "repair_adapter"), ("world_validity", "repair_world")]
)
def test_platform_fault_is_not_synthesis(check, action):
    c = case()
    c["checks"][check] = "fail"
    row = plan([c])["decisions"][0]
    assert row["action"] == action and row["status"] == "not_selected"


def test_unknown_evidence_needs_probe_not_more_images():
    c = case()
    c["checks"]["runtime_parity"] = "unknown"
    assert plan([c])["decisions"][0]["action"] == "probe"


def test_development_case_and_adjacent_shards_never_enter_training():
    row = plan([case(source_log="dev_050_060", split="development")])["decisions"][0]
    assert row["action"] == "evaluation_only" and row["mode"] is None
    with pytest.raises(ValueError, match="disjoint"):
        plan([case()], development_logs=["train_100_200"])
    with pytest.raises(ValueError, match="outside"):
        plan([case(source_log="foreign")])


def test_llm_cannot_change_action_generator_evidence_or_admit_development():
    rows = plan([case(), case("dev", source_log="dev", split="development")])["decisions"]
    proposal = {
        "suggestions": [
            {
                "case_id": "dev",
                "mode": "new_state",
                "generator": "simscale_rule",
                "reason": "hard",
                "evidence": ["evidence.json"],
            }
        ]
    }
    assert validate_advice(proposal, rows)["rejected"]
    s = proposal["suggestions"][0]
    s["case_id"] = "a"
    s["generator"] = "drivezero_navsim"
    assert validate_advice(proposal, rows)["rejected"]
    s["generator"] = "simscale_rule"
    s["evidence"] = ["invented"]
    assert validate_advice(proposal, rows)["rejected"]
    s["threshold"] = 0.5
    with pytest.raises(ValueError, match="thresholds"):
        validate_advice(proposal, rows)


def test_llm_order_can_allocate_only_fixed_budget():
    advice = {
        "suggestions": [
            {
                "case_id": "b",
                "mode": "new_state",
                "generator": "simscale_rule",
                "reason": "priority",
                "evidence": ["evidence.json"],
            }
        ]
    }
    result = plan([case(), case("b")], advice=advice)
    assert [x["status"] for x in result["decisions"]] == ["budget_deferred", "reserved"]
    assert all(x == 0 for x in result["remaining_budget"].values())
    assert not result["generators"]["learned_self_play"]["available"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("endpoint", "http://remote/v1"),
        ("endpoint", "https://user:secret@remote/v1"),
        ("timeout_seconds", 61),
        ("max_tokens", 4097),
    ],
)
def test_llm_transport_is_explicit_and_bounded(key, value):
    cfg = dict(
        endpoint="https://provider.example/v1/chat/completions",
        model="chosen",
        api_key_env="DRIVERSI_LLM_KEY",
        timeout_seconds=30,
        max_tokens=1000,
    )
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_mainline_has_feedback_port_and_cannot_train_from_inspection():
    p = builtin_providers()
    assert p.get("navsim_feedback_augmentation").inputs == (
        ("feedback", "unified_learning_feedback"),
    )
    assert p.get("navsim_history_future_export").output != "native_training_dataset"
    cfg, _ = navsim.prepare_export({"cameras": ["CAM_F0", "CAM_L0", "CAM_R0"]}, None, None)
    assert cfg["future_images"] is True
    with pytest.raises(ValueError):
        navsim.prepare_export({"cameras": ["CAM_F0"], "future_images": False}, None, None)


def test_evidence_changed_after_registration_stops_decision(tmp_path):
    p = tmp_path / "evidence.json"
    p.write_text("original")
    from driversi.agents.backends.native import digest

    cfg = {"files": {str(p): digest(p)}}
    p.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        navsim.run_decision(cfg, {}, tmp_path / "out", tmp_path / "registry")
    assert not (tmp_path / "out").exists()


def test_llm_payload_is_anonymized(monkeypatch):
    from driversi.rsi.decisions import llm_advisor as mod

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            proposal = {
                "suggestions": [
                    {
                        "case_id": "case-0",
                        "mode": "new_state",
                        "generator": "simscale_rule",
                        "reason": "test",
                        "evidence": ["evidence-0"],
                    }
                ]
            }
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(proposal)}}]}
            ).encode()

    class Opener:
        def open(self, req, timeout):
            captured["body"] = req.data.decode()
            return Response()

    monkeypatch.setenv("DRIVERSI_TEST_KEY", "do-not-store")
    monkeypatch.setattr(mod.urllib.request, "build_opener", lambda *a: Opener())
    c = case("secret-token", source_log="private-log", failure_class="/private/file")
    c["evidence"] = ["/private/evidence.json"]
    cfg = dict(
        endpoint="https://provider.example/v1/chat/completions",
        model="chosen",
        api_key_env="DRIVERSI_TEST_KEY",
        timeout_seconds=30,
        max_tokens=100,
    )
    proposal, receipt = mod.advise(cfg, [c])
    assert all(x not in captured["body"] for x in ("private", "secret-token", "do-not-store"))
    assert proposal["suggestions"][0]["case_id"] == "secret-token"
    assert receipt["requests"] == 1


def test_external_llm_disabled_without_explicit_credential(monkeypatch):
    from driversi.rsi.decisions.llm_advisor import advise

    monkeypatch.delenv("DRIVERSI_NO_SUCH_KEY", raising=False)
    cfg = dict(
        endpoint="https://provider.example/v1/chat/completions",
        model="chosen",
        api_key_env="DRIVERSI_NO_SUCH_KEY",
        timeout_seconds=30,
        max_tokens=100,
    )
    with pytest.raises(ValueError, match="credential"):
        advise(cfg, [case()])


def test_no_generation_with_successful_task():
    c = case()
    c["checks"]["task"] = "pass"
    r = plan([c])
    assert r["decisions"][0]["action"] == "retain"
    assert r["decisions"][0]["mode"] is None


def test_navsim_preparation_registers_original_groups_for_promotion(monkeypatch, tmp_path):
    from driversi.rsi.correction import gate
    from driversi.rsi.correction.store import RepairStore
    from driversi.rsi.workflow.store import HarnessStore

    doc = {"train_logs": ["train_001_002"], "development_logs": ["dev_001_002"], "cases": []}
    p = tmp_path / "evidence.json"
    p.write_text("{}")
    monkeypatch.setattr(navsim, "_load", lambda *a: (doc, [], {}))
    prepared, _ = navsim.prepare_decision(
        {"evidence": str(p), "llm": None}, tmp_path, tmp_path / "registry"
    )
    assert prepared["source_groups"] == ["dev", "train"]
    store = HarnessStore(tmp_path / "registry")
    store.register(
        {"id": "selection", "providers": [], "source_groups": prepared["source_groups"]},
        tmp_path / "selection",
    )
    monkeypatch.setattr(gate, "validate_gate", lambda *a: {"cases": [{"source_log": "train"}]})
    with pytest.raises(ValueError, match="already exposed"):
        gate.run_gate(RepairStore(tmp_path / "registry"), {}, tmp_path / "gate")


def test_protected_outcomes_rejected_before_evidence_access(monkeypatch, tmp_path):
    c = case("test", source_log="test", split="test")
    c["evidence"] = ["missing-secret-test-results.json"]
    cost = dict(wall_seconds=120, gpu_seconds=120, score_queries=16)
    doc = dict(
        schema_version=1,
        train_logs=["train"],
        development_logs=["dev"],
        protected_logs=["test"],
        cases=[c],
        budget=cost,
        costs={"relabel": cost, "new_state": cost},
        student_checkpoint="model-v1",
        score_contract=dict(
            protocol="fixed", metric="PDMS", threshold=0.95, context="executed_counterfactual"
        ),
    )
    p = tmp_path / "evidence.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(navsim, "_protected", lambda *a: [])
    with pytest.raises(ValueError, match="Protected/test"):
        navsim.prepare_decision({"evidence": str(p), "llm": None}, tmp_path, tmp_path / "registry")


def test_atomic_reservation_survives_failure_and_concurrent_attempts(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    registry = tmp_path / "registry.sqlite"
    with sqlite3.connect(registry) as connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('preserved')")

    def attempt(i):
        try:
            return str(navsim.claim_reservation(registry, "sealed", "case", tmp_path / str(i), {}))
        except ValueError:
            return None

    with ThreadPoolExecutor(4) as workers:
        results = list(workers.map(attempt, range(4)))
    assert sum(x is not None for x in results) == 1
    with pytest.raises(ValueError, match="already consumed"):
        navsim.claim_reservation(registry, "sealed", "case", tmp_path / "retry", {})
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT value FROM marker").fetchall() == [("preserved",)]


@pytest.mark.parametrize("generator_fails", [False, True])
def test_duplicate_selected_execution_does_not_start_generator(
    monkeypatch, tmp_path, generator_fails
):
    from types import SimpleNamespace

    row = plan([case()])["decisions"][0]
    row["reserved_cost"]["wall_seconds"] = 120
    doc = {
        "source_files": {},
        "decisions": [row],
        "train_logs": ["train_001_002"],
        "development_logs": ["dev"],
        "score_contract": {"metric": "PDMS"},
    }
    p = tmp_path / "decision.json"
    p.write_text(json.dumps(doc))
    inputs = {
        "decision": {
            "artifacts": {"report": str(p)},
            "summary": {"artifact_reference": {"identity": "sealed"}},
        }
    }
    monkeypatch.setattr(
        navsim.Scene, "load", lambda *a: SimpleNamespace(metadata={"source_log": "train_001_002"})
    )
    monkeypatch.setattr(navsim, "protect", lambda *a: None)
    calls = []

    def execute(config, inputs, output, registry):
        calls.append(output)
        if generator_fails:
            raise RuntimeError("forced physical generator failure")
        output.mkdir()
        report = output / "report.json"
        report.write_text("{}")
        return {"summary": {}, "artifacts": {"report": str(report), "bundle": str(report)}}

    monkeypatch.setattr(navsim.reactive, "run_world", execute)
    config = {"case_id": "a", "world": {"scene": "scene.npz"}}
    registry = tmp_path / "registry.sqlite"
    with sqlite3.connect(registry) as connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('preserved')")
    if generator_fails:
        with pytest.raises(RuntimeError, match="forced physical"):
            navsim.run_world(config, inputs, tmp_path / "first", registry)
    else:
        from driversi.rsi.workflow.artifacts import seal, verify

        first = navsim.run_world(config, inputs, tmp_path / "first", registry)
        assert Path(first["artifacts"]["reservation"]).is_relative_to(tmp_path / "first")
        assert json.loads(Path(first["artifacts"]["reservation"]).read_text())["case_id"] == "a"
        stage = dict(
            id="generate",
            output_kind="reactive_execution",
            provider="navsim_selected_synthesis",
            provider_version="1",
        )
        first["summary"]["artifact_reference"] = seal(first, tmp_path / "first", "task", stage)
        assert verify(first, "task", stage)["files"]["reservation"]["sha256"]
    with pytest.raises(ValueError, match="already consumed"):
        navsim.run_world(config, inputs, tmp_path / "second", registry)
    assert len(calls) == 1 and not (tmp_path / "second").exists()
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT value FROM marker").fetchall() == [("preserved",)]


def test_mainline_rejects_development_render_before_export(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(
        navsim.Scene, "load", lambda *a: SimpleNamespace(metadata={"split": "development"})
    )
    with pytest.raises(ValueError, match="training-source"):
        navsim.run_export(
            {},
            {"rendered": {"artifacts": {"scene": "scene.npz"}}},
            tmp_path / "out",
            tmp_path / "registry",
        )
    assert not (tmp_path / "out").exists()


def feedback():
    return dict(
        schema_version=1,
        benchmark="navtest",
        checkpoint="model-v1",
        evaluator_version="native-v1",
        slices=[dict(capability="interaction", count=10, failures=4)],
    )


def test_aggregate_feedback_selects_train_curriculum_without_claiming_model_fault():
    c = case()
    c["checks"]["task"] = "pass"
    r = plan([c], capability_feedback=feedback())
    row = r["decisions"][0]
    assert row["active_curriculum_seed"] and row["mode"] == "new_state"
    assert not row["model_cause_established"] and not row["trainable"]
    assert r["capability_feedback"]["evaluation_role"] == "adaptive_evaluation"
    assert "checkpoint" not in r["capability_feedback"]


def test_aggregate_feedback_never_overrides_platform_or_missing_evidence():
    for check in ("runtime_parity", "world_validity"):
        for state in ("unknown", "fail"):
            c = case()
            c["checks"]["task"] = "pass"
            c["checks"][check] = state
            row = plan([c], capability_feedback=feedback())["decisions"][0]
            assert not row["active_curriculum_seed"] and row["mode"] is None


def test_aggregate_feedback_rejects_foreign_model_and_test_content():
    from driversi.rsi.decisions.capability_feedback import validate_feedback

    f = feedback()
    f["checkpoint"] = "other-model"
    with pytest.raises(ValueError, match="student"):
        plan([case()], capability_feedback=f)
    f = feedback()
    f["slices"][0]["token"] = "test-scene"
    with pytest.raises(ValueError, match="tokens"):
        validate_feedback(f)
    with pytest.raises(ValueError, match="Protected/test"):
        plan([case(split="test", source_log="test")], capability_feedback=feedback())


@pytest.mark.parametrize("count,failures", [(True, 1), (2, 3), (0, 0), (2, -1), (2, 1.5)])
def test_feedback_counts_are_observed_not_unbounded_claims(count, failures):
    from driversi.rsi.decisions.capability_feedback import validate_feedback

    f = feedback()
    f["slices"][0].update(count=count, failures=failures)
    with pytest.raises(ValueError):
        validate_feedback(f)
