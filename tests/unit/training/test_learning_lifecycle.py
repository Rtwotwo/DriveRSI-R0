"""Checks for actual integration boundaries, independent of optional model runtimes."""

import pytest

from driversi.rsi.workflow.providers import builtin_providers
from driversi.training.contracts.job import validate_request
from driversi.training.orchestration import lifecycle


def request():
    return dict(
        schema_version=2,
        id="pilot",
        model_config="model.yaml",
        mode="supervised",
        initialization="model_weights",
        data="data.json",
        pilot=dict(seed=42, batch_size=2, evaluation_steps=20),
        budget=dict(
            max_wall_seconds=10,
            max_optimizer_steps=2,
            max_environment_steps=0,
            max_scoring_agent_steps=0,
        ),
    )


def test_four_typed_lifecycle_stages_exist():
    providers = builtin_providers()
    train = providers.get("native_model_unified_training")
    resume = providers.get("native_model_unified_resume")
    evaluate = providers.get("native_model_unified_evaluation")
    feedback = providers.get("native_model_unified_feedback")
    assert dict(resume.inputs)["student"] == train.output == resume.output
    assert dict(evaluate.inputs)["student"] == resume.output
    assert dict(feedback.inputs)["evaluation"] == evaluate.output


@pytest.mark.parametrize(
    "key,value", [("batch_size", 0), ("batch_size", 32), ("seed", -1), ("evaluation_steps", 10000)]
)
def test_pilot_resource_settings_fail_before_execution(key, value):
    r = request()
    r["pilot"][key] = value
    with pytest.raises(ValueError):
        validate_request(r)


def test_schema2_does_not_claim_weight_loading_is_resume():
    r = request()
    assert validate_request(r) is r
    r["initialization"] = "resume"
    with pytest.raises(ValueError):
        validate_request(r)


def test_source_log_shards_cannot_cross_holdout(monkeypatch, tmp_path):
    f = (
        tmp_path
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    f.parent.mkdir(parents=True)
    f.write_text("log_names: []\n")
    monkeypatch.setattr(lifecycle, "project_root", lambda: tmp_path)
    data = {
        "parent_provenance": "unknown",
        "train": {"source_logs": ["2021.08.17.13.10.50_veh-08_00122_00295"]},
        "development": {"source_logs": ["2021.08.17.13.10.50_veh-08_00500_00600"]},
    }
    with pytest.raises(ValueError, match="separate original logs"):
        lifecycle.sources(data)


def evaluation():
    student = {
        "status": "completed",
        "context": "same",
        "model": "augustus",
        "checkpoint": "/exp/student",
        "consumed": {"wall_seconds": 5},
    }
    report = {
        "status": "completed",
        "context": "same",
        "evaluated_checkpoint": "/exp/student",
        "metrics": {"parent": {"PDMS": 90.0, "NC": 1.0}, "candidate": {"PDMS": 95.0, "NC": 1.0}},
        "directions": {"PDMS": 1, "NC": 1},
        "primary_metric": "PDMS",
        "safety_higher": ["NC"],
        "consumed": {"wall_seconds": 2},
    }
    return student, report


def test_improved_small_holdout_cannot_promote_model():
    result = lifecycle.feedback(*evaluation())
    assert result["pilot_metric_improved"]
    assert not result["promoted"] and not result["observed_generalization_benefit"]


def test_score_gain_with_safety_loss_is_not_benefit():
    student, report = evaluation()
    report["metrics"]["candidate"]["NC"] = 0.8
    assert not lifecycle.feedback(student, report)["pilot_metric_improved"]


def test_evaluation_of_another_checkpoint_cannot_feed_student():
    student, report = evaluation()
    report["evaluated_checkpoint"] = "/exp/other"
    with pytest.raises(ValueError, match="another student"):
        lifecycle.feedback(student, report)


def test_failed_attempt_cannot_generate_positive_feedback():
    student, report = evaluation()
    student["status"] = "failed"
    with pytest.raises(ValueError):
        lifecycle.feedback(student, report)


def test_missing_fpd_training_is_not_a_registered_worker():
    assert "fpd_drive" not in lifecycle.WORKERS


def test_new_schema1_training_cannot_write_legacy_outputs(tmp_path, monkeypatch):
    from driversi.training.orchestration import engine
    from tests.unit.training.test_unified_training import request as legacy_request

    p = legacy_request(tmp_path, "driverl", "online_rl")
    monkeypatch.setattr(engine.native, "project_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="exp"):
        engine.prepare(p, tmp_path / "outputs/legacy")
    assert not (tmp_path / "outputs/legacy").exists()


def test_feedback_includes_both_training_phases():
    student, report = evaluation()
    student["training_lineage_cost"] = {"wall_seconds": 12}
    assert lifecycle.feedback(student, report)["training_cost"]["wall_seconds"] == 12


def test_inventory_distinguishes_paper_learner_from_official_release():
    from driversi.training.contracts.protocols import inventory

    rows = {r["model"]: r for r in inventory()["models"]}
    assert rows["driverl"]["lifecycle_modes"]["online_rl"]
    assert not rows["driverl"]["training_modes"]["online_rl"]
    assert not any(rows["fpd_drive"]["lifecycle_modes"].values())


def test_lineage_reconstructs_old_receipts_without_double_counting(tmp_path):
    import json

    def save(name, previous, cost):
        folder = tmp_path / name
        folder.mkdir()
        report = folder / "training-result.json"
        report.write_text(
            json.dumps({"status": "completed", "context": "c", "consumed": {"wall_seconds": cost}})
        )
        (folder / "training-plan.json").write_text(
            json.dumps(
                {
                    "phase": "train" if previous is None else "resume",
                    "previous": str(previous) if previous else None,
                }
            )
        )
        return report

    parent = save("train", None, 7)
    child = save("resume", parent, 5)
    derived = lifecycle.training_lineage(child)
    assert derived["cost"]["wall_seconds"] == 12
    assert len(derived["receipts"]) == 2


@pytest.fixture
def reusable_training(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from driversi.rsi.plugins import training
    from driversi.rsi.workflow import artifacts, store

    root = tmp_path / "exp/original/train"
    root.mkdir(parents=True)
    report = root / "training-result.json"
    report.write_text(json.dumps({"status": "completed", "phase": "train"}))
    plan = root / "training-plan.json"
    plan.write_text("{}")
    declaration = {
        "id": "train",
        "output_kind": "unified_training_result",
        "provider": "native_model_unified_training",
        "provider_version": "1",
    }
    result = {"artifacts": {"report": str(report), "plan": str(plan)}, "summary": {}}
    result["summary"]["artifact_reference"] = artifacts.seal(result, root, "original", declaration)
    state = {
        "id": "original",
        "task": {"steps": [declaration], "source_groups": ["train-log"]},
        "stages": [{"id": "train", "status": "completed", "result": result}],
    }
    protected = set()
    fake = SimpleNamespace(
        get=lambda task: state, registry=SimpleNamespace(protected_groups=lambda: protected)
    )
    monkeypatch.setattr(store, "HarnessStore", lambda path: fake)
    monkeypatch.setattr(training.engine.native, "project_root", lambda: tmp_path)
    return training, state, report, protected


def test_reuse_completed_training_without_reexecuting_optimizer(tmp_path, reusable_training):
    training, _, report, _ = reusable_training
    config, cost = training.prepare_reference(
        {"source_task": "original", "source_stage": "train"}, tmp_path, tmp_path
    )
    result = training.run_reference(config, {}, tmp_path / "exp/reuse", tmp_path)
    assert cost == result["sim_agent_steps"] == 0
    assert result["summary"]["reused_training"]
    assert report.read_bytes() == (tmp_path / "exp/reuse/training-result.json").read_bytes()


@pytest.mark.parametrize("fault", ["tampered", "incomplete", "protected"])
def test_reuse_rejects_invalid_or_changed_prior_result(tmp_path, reusable_training, fault):
    training, state, report, protected = reusable_training
    config, _ = training.prepare_reference(
        {"source_task": "original", "source_stage": "train"}, tmp_path, tmp_path
    )
    if fault == "tampered":
        report.write_text("{}")
    elif fault == "incomplete":
        state["stages"][0]["status"] = "failed"
    else:
        protected.add("train-log")
    with pytest.raises(ValueError):
        training.run_reference(config, {}, tmp_path / "exp/reuse", tmp_path)
    assert not (tmp_path / "exp/reuse").exists()


@pytest.mark.parametrize("fault", [None, "source", "protocol", "tampered", "escape"])
def test_evaluator_context_pins_protocol_sources_and_cache(tmp_path, fault):
    import json

    from driversi.agents.backends.native import digest
    from driversi.training.contracts.evaluation import resolve_context

    cache = tmp_path / "metric-cache"
    cache.mkdir()
    artifact = cache / "metric_cache.pkl"
    artifact.write_bytes(b"sealed")
    value = dict(
        schema_version=1,
        model="augustus",
        protocol="bundled_navsim_pdm_v1",
        development_source_logs=["development-log"],
        dataset_manifest_sha256="abc",
        metric_cache=str(cache),
        files={str(artifact): digest(artifact)},
    )
    if fault == "source":
        value["development_source_logs"] = ["training-log"]
    if fault == "protocol":
        value["protocol"] = "relaxed-scorer"
    if fault == "tampered":
        artifact.write_bytes(b"changed")
    if fault == "escape":
        value["files"] = {str(tmp_path / "outside.pkl"): "abc"}
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps(value))
    if fault:
        with pytest.raises(ValueError):
            resolve_context(manifest, "augustus", ["development-log"])
    else:
        result = resolve_context(manifest, "augustus", ["development-log"])
        assert result["manifest_sha256"] == digest(manifest)
        assert result["value"] == value


def test_training_provider_cannot_accept_evaluator_override(tmp_path):
    from driversi.rsi.plugins.training import prepare_job

    with pytest.raises(ValueError):
        prepare_job(
            {"request": "training.json", "evaluator": "evaluation.json"}, tmp_path, tmp_path
        )
