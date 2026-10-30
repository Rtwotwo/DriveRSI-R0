import copy
import json

import numpy as np
import pytest

from driversi.evaluation.search.tto_skill import scoring_coordinates
from driversi.rsi.plugins import paired, reactive, synthesis
from driversi.rsi.workflow.providers import builtin_providers


def test_scoring_rotation_preserves_world_pose_and_input():
    source_yaw, scoring_yaw = 2.12, 2.12023
    traj = np.array([[2.0, 1.0, 0.5], [7.0, -2.0, -3.14]])
    original = traj.copy()
    converted = scoring_coordinates(traj, source_yaw - scoring_yaw)

    def world(points, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return points @ np.array([[c, s], [-s, c]])

    np.testing.assert_allclose(
        world(converted[:, :2], scoring_yaw), world(traj[:, :2], source_yaw), atol=1e-14
    )
    np.testing.assert_allclose(
        np.sin(converted[:, 2] + scoring_yaw), np.sin(traj[:, 2] + source_yaw)
    )
    np.testing.assert_array_equal(traj, original)


def test_logged_and_reactive_admission_ports_are_distinct():
    providers = builtin_providers()
    assert providers.get("drivetto_independent_admission").output == "admitted_original_branch"
    assert dict(providers.get("selected_branch_gs").inputs)["branch"] == "selected_reactive_branch"
    assert providers.get("selected_branch_navsim_export").output != "native_training_dataset"
    assert (
        dict(providers.get("native_dataset_training").inputs)["dataset"]
        == "native_training_dataset"
    )


@pytest.mark.parametrize(
    "model,mode", [("driverl", "online_rl"), ("wa_jepa", "world_model"), ("fpd", "supervised")]
)
def test_offline_trajectory_cannot_become_an_on_policy_or_video_sample(
    monkeypatch, tmp_path, model, mode
):
    monkeypatch.setattr(paired, "prepare_job", lambda *a: ({"request": "dummy"}, 0))
    monkeypatch.setattr(
        paired,
        "resolve_request",
        lambda *a: {"model": model, "request": {"schema_version": 2, "mode": mode}},
    )
    with pytest.raises(ValueError, match="not RL transitions"):
        paired.prepare_training({}, tmp_path, "registry")


def test_original_history_rejects_counterfactual_admission(monkeypatch, tmp_path):
    import driversi_agents.augustus.data as data

    source = tmp_path / "source.json"
    source.write_text("{}")
    report = tmp_path / "admission.json"
    report.write_text(json.dumps({"admitted_scope": "reactive"}))
    monkeypatch.setattr(data, "manifest", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="counterfactual"):
        synthesis.run_history(
            {"manifest": str(source), "sha256": synthesis.digest(source)},
            {"admission": {"artifacts": {"report": str(report)}}},
            tmp_path / "out",
            "registry",
        )
    assert not (tmp_path / "out").exists()


def test_changed_inner_cache_is_not_hidden_by_unchanged_receipt(tmp_path):
    p = tmp_path / "cache"
    p.write_bytes(b"original")
    files = {str(p): synthesis.digest(p)}
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        synthesis.check_files(files)


@pytest.mark.parametrize(
    "key",
    ["comparison_identity", "sensor_identity", "independent_evaluator_sha256", "search_identity"],
)
def test_paired_feedback_rejects_cross_task_substitution(key):
    a = {
        "binding": dict(
            comparison_identity="x",
            sensor_identity="y",
            independent_evaluator_sha256="z",
            search_identity="s",
        )
    }
    b = copy.deepcopy(a)
    b["binding"][key] = "different"
    with pytest.raises(ValueError, match="Unmatched"):
        paired.compare(a, b)


def test_unbudgeted_resume_rejected(tmp_path):
    for value in (-1, True, 100001):
        with pytest.raises(ValueError):
            paired.prepare_resume({"max_sim_agent_steps": value}, tmp_path, "r")


def test_wrong_reactive_branch_rejected(tmp_path):
    with pytest.raises(ValueError):
        reactive.prepare_select({"branch": "best_unverified"}, tmp_path, "r")


def test_local_control_safety_failure_cannot_be_cancelled_by_another_improvement():
    base = dict(PDMS=0.5, NC=1.0, DAC=1.0, DDC=1.0, TTC=0.0, C=1.0, EP=0.5)
    arms = []
    for name, ttc, pdms in [("original", [1.0, 0.0], 0.6), ("drivetto", [0.0, 1.0], 0.8)]:
        rows = [
            dict(
                sample_id=str(i),
                source_log="dev",
                parent=base.copy(),
                candidate={**base, "TTC": t, "PDMS": pdms},
            )
            for i, t in enumerate(ttc)
        ]
        avg = {**base, "TTC": 0.5, "PDMS": pdms}
        student = dict(
            status="completed",
            context=name,
            checkpoint=name,
            model="augustus",
            total_updates=2,
            consumed={},
        )
        evaluation = dict(
            status="completed",
            context=name,
            evaluated_checkpoint=name,
            consumed={},
            source_groups={"train": ["train"], "development": ["dev"]},
            evaluated_sample_ids=["0", "1"],
            paired_samples=rows,
            metrics={"parent": base, "candidate": avg},
            directions={k: 1 for k in base},
            primary_metric="PDMS",
            safety_higher=["NC", "DAC", "DDC", "TTC", "C"],
        )
        binding = dict(
            comparison_identity="same",
            sensor_identity="same",
            independent_evaluator_sha256="same",
            search_identity="same",
            label_source=name,
        )
        arms.append(dict(student=student, evaluation=evaluation, binding=binding))
    report = paired.compare(*arms)
    assert report["treatment_minus_continued_training"]["TTC"] == 0
    assert not report["observed_pilot_benefit"]
    assert report["regressions_against_control"] == [
        dict(sample_id="0", source_log="dev", metric="TTC", control=1.0, treatment=0.0)
    ]


def test_resume_rejects_changed_manifest_before_starting_native_worker(tmp_path, monkeypatch):
    manifest = tmp_path / "dataset.json"
    manifest.write_text('{"cache":"original"}')
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"manifest": str(manifest), "manifest_sha256": synthesis.digest(manifest)})
    )
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "files": {},
                "dataset_receipt": str(receipt),
                "dataset_receipt_sha256": synthesis.digest(receipt),
            }
        )
    )
    manifest.write_text('{"cache":"changed"}')

    def unexpected(*a, **k):
        pytest.fail("native execution entered before identity verification")

    monkeypatch.setattr(paired.lifecycle, "prepare", unexpected)
    with pytest.raises(ValueError, match="changed"):
        paired.phase(
            {},
            {"student": {"artifacts": {"binding": str(binding)}}},
            tmp_path / "out",
            None,
            "resume",
        )


@pytest.mark.parametrize("payload_name", ["frame.png", "valid-mask.png", "episode.npz"])
def test_export_rejects_changed_render_payload(tmp_path, monkeypatch, payload_name):
    import driversi.data.datasets.navsim_synthetic as exporter

    payload = tmp_path / payload_name
    payload.write_bytes(b"original")
    receipt = tmp_path / "render.json"
    receipt.write_text(json.dumps({"payload_files": {str(payload): synthesis.digest(payload)}}))
    payload.write_bytes(b"changed")

    def unexpected(*a, **k):
        pytest.fail("export entered before render identity verification")

    monkeypatch.setattr(exporter, "export_navsim_sample", unexpected)
    with pytest.raises(ValueError, match="changed"):
        reactive.run_export(
            {}, {"rendered": {"artifacts": {"report": str(receipt)}}}, tmp_path / "out", None
        )


def test_fixed_parent_changed_under_same_configuration_rejected(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text("{}")
    config = tmp_path / "model.yaml"
    config.write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"original")
    declaration = dict(
        request=str(request),
        request_sha256=synthesis.digest(request),
        model_config=str(config),
        model_config_sha256=synthesis.digest(config),
        parent_files={str(weight): synthesis.digest(weight)},
    )
    weight.write_bytes(b"same-shapes-different-values")
    with pytest.raises(ValueError, match="changed"):
        paired.run_training(declaration, {}, tmp_path / "out", None)
