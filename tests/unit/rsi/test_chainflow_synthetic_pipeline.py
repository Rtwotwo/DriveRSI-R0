"""Contract tests only; fixtures do not constitute visual/model evidence."""

import json
from pathlib import Path

import pytest

from driversi.rsi.plugins import chainflow_synthetic as plugin
from driversi.rsi.workflow.providers import builtin_providers


def test_synthetic_pipeline_requires_both_separate_evidence_artifacts():
    provider = builtin_providers().get("chainflow_synthetic_dataset")
    assert provider.inputs == (
        ("branch", "selected_native_branch"),
        ("rendered", "worldengine_inspection_window"),
        ("visual", "worldengine_visual_admission"),
    )
    assert provider.output == "chainflow_admitted_synthetic_sample"


def test_visual_rejection_stops_native_worker_before_export(tmp_path, monkeypatch):
    from driversi.rendering.diagnostics import worldengine_admission
    from driversi.rsi.plugins import native_branch

    monkeypatch.setattr(native_branch, "validate_trajectory_evidence", lambda evidence: None)
    monkeypatch.setattr(
        worldengine_admission, "validate_visual_admission", lambda path: {"admitted": False}
    )
    monkeypatch.setattr(
        plugin, "run_process", lambda *a, **kw: pytest.fail("Rejected branch started a worker")
    )
    branch = tmp_path / "branch.json"
    branch.write_text(json.dumps({"trajectory_evidence": {}}))
    render = tmp_path / "render.json"
    render.write_text("{}")
    with pytest.raises(ValueError, match="visual admission rejected"):
        plugin.execute(
            {},
            {
                "branch": {"artifacts": {"report": str(branch)}},
                "rendered": {"artifacts": {"report": str(render)}},
                "visual": {"artifacts": {"report": "rejection.json"}},
            },
            tmp_path / "out",
            None,
        )
    assert not (tmp_path / "out").exists()


def test_pair_compiler_retains_control_and_binds_real_intervention(tmp_path, monkeypatch):
    from driversi_agents.chainflow import data, synthetic

    control = {"sample_id": "original", "source_log": "train-log", "kind": "original_navtrain"}
    development = {
        "sample_id": "dev",
        "source_log": "dev-log",
        "kind": "original_navtrain",
        "metric_cache": str(tmp_path / "cache/a/cache.pkl"),
    }
    base = {
        "schema_version": 1,
        "kind": "chainflow_native_scene_dataset",
        "splits": {
            "train": [control],
            "development": [
                development,
                dict(
                    development,
                    sample_id="dev2",
                    source_log="dev2-log",
                    metric_cache=str(tmp_path / "cache/b/cache.pkl"),
                ),
            ],
        },
        "files": {},
    }
    for item in base["splits"]["development"]:
        cache = Path(item["metric_cache"])
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"sealed independent cache")
        base["files"][str(cache)] = plugin.digest(cache)
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    row = {
        "sample_id": "synthetic_branch",
        "source_log": "synthesis-log",
        "kind": "synthetic_navtrain",
    }
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(
        json.dumps({"kind": "chainflow_admitted_synthetic_sample", "row": row, "files": {}})
    )
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "model": "chainflow_stage1",
                "settings": {
                    "training_recipe": {"scheduler_dataset_size": 2, "scheduler_num_epochs": 10}
                },
            }
        )
    )
    monkeypatch.setattr(plugin, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        data, "validate_manifest", lambda path, **kw: json.loads(Path(path).read_text())
    )
    checked = []
    monkeypatch.setattr(
        synthetic, "validate_admission", lambda row, files: checked.append(row["sample_id"])
    )
    path = plugin.compile_paired_task(base_path, sample_path, model, tmp_path / "exp/pair")
    task = json.loads(path.read_text())
    left = json.loads((path.parent / "control/manifest.json").read_text())
    right = json.loads((path.parent / "treatment/manifest.json").read_text())
    assert left == base and right["splits"]["train"] == [control, row]
    assert right["splits"]["development"] == left["splits"]["development"]
    requests = [
        json.loads((path.parent / side / "request.json").read_text())
        for side in ("control", "treatment")
    ]
    assert requests[0]["model_config"] == requests[1]["model_config"]
    assert requests[0]["budget"] == requests[1]["budget"]
    assert requests[0]["pilot"] == requests[1]["pilot"]
    assert checked == ["synthetic_branch"]
    from driversi.training.contracts.evaluation import resolve_context

    context = resolve_context(
        path.parent / "control/evaluator.json", "chainflow_stage1", {"dev-log", "dev2-log"}
    )
    assert context["value"]["metric_cache"] == str(tmp_path / "cache")
    Path(development["metric_cache"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        resolve_context(
            path.parent / "control/evaluator.json", "chainflow_stage1", {"dev-log", "dev2-log"}
        )
    Path(development["metric_cache"]).write_bytes(b"sealed independent cache")
    assert next(s for s in task["steps"] if s["id"] == "treatment_replay")["inputs"] == {
        "student": "treatment_train"
    }
    assert task["steps"][-1]["provider"] == "chainflow_paired_feedback"
    second = dict(row, sample_id="second-synthetic", source_log="another-train-log")
    second_path = tmp_path / "sample2.json"
    second_path.write_text(
        json.dumps({"kind": "chainflow_admitted_synthetic_sample", "row": second, "files": {}})
    )
    batch_path = plugin.compile_paired_task(
        base_path, [sample_path, second_path], model, tmp_path / "exp/batch", updates_per_arm=12
    )
    intervention = json.loads((batch_path.parent / "intervention.json").read_text())
    assert intervention["optimizer_updates_per_arm"] == 12
    batch_task = json.loads(batch_path.read_text())
    request = json.loads((batch_path.parent / "treatment/request.json").read_text())
    assert (
        batch_task["budget"]["max_sim_agent_steps"]
        == 7 * request["budget"]["max_scoring_agent_steps"]
    )
    assert intervention["expected_sample_exposures"]["treatment"] == {
        "original": 4,
        "synthetic_branch": 4,
        "second-synthetic": 4,
    }
    for name, receipts, updates, match in [
        ("duplicate", [sample_path, sample_path], 8, "duplicates"),
        ("unconsumed", [sample_path, second_path], 2, "consume every"),
        ("odd", [sample_path], 9, "Even pilot"),
        ("over-horizon", [sample_path], 22, "horizon"),
    ]:
        with pytest.raises(ValueError, match=match):
            plugin.compile_paired_task(
                base_path, receipts, model, tmp_path / "exp" / name, updates_per_arm=updates
            )
        assert not (tmp_path / "exp" / name).exists()
    row["source_log"] = "dev-log"
    sample_path.write_text(
        json.dumps({"kind": "chainflow_admitted_synthetic_sample", "row": row, "files": {}})
    )
    with pytest.raises(ValueError, match="separate original logs"):
        plugin.compile_paired_task(base_path, sample_path, model, tmp_path / "exp/leaking-pair")
    assert not (tmp_path / "exp/leaking-pair").exists()


def test_accepted_visual_gate_dispatches_sealed_native_export(tmp_path, monkeypatch):
    from driversi.rendering.diagnostics import worldengine_admission
    from driversi.rsi.plugins import native_branch
    from driversi_agents.chainflow import synthetic

    monkeypatch.setattr(native_branch, "validate_trajectory_evidence", lambda evidence: None)
    monkeypatch.setattr(
        worldengine_admission, "validate_visual_admission", lambda path: {"admitted": True}
    )
    monkeypatch.setattr(plugin, "protect", lambda *args: None)
    execution = tmp_path / "execution"
    execution.mkdir()
    context = execution / "teacher-counterfactual-context.pkl"
    context.write_bytes(b"unit fixture: no pickle is deserialized")
    window, visual, render = [
        tmp_path / name for name in ("window.json", "visual.json", "pixels.json")
    ]
    window.write_text(json.dumps({"future": list(range(10))}))
    visual.write_text("{}")
    render.write_text("{}")
    branch, receipt = tmp_path / "branch.json", tmp_path / "render.json"
    branch.write_text(
        json.dumps(
            {
                "trajectory_evidence": {},
                "source_log": "train-log",
                "execution": str(execution),
                "branch": "teacher",
                "upstream_source_files": {},
                "files": {str(context): plugin.digest(context)},
            }
        )
    )
    receipt.write_text(json.dumps({"branch_identity": "original-identity", "payload_files": {}}))
    calls = []

    def dispatch(argv, output, timeout):
        plan = json.loads((output / "export-plan.json").read_text())
        assert plan["context"] == str(context)
        assert plan["files"][str(context)] == plugin.digest(context)
        assert plan["visual_admission"] == str(visual)
        assert plan["branch"]["trajectory_window"] == str(window)
        calls.append(plan)
        (output / "dataset.json").write_text(
            json.dumps(
                {"row": {"sample_id": "new-history"}, "files": {}, "scored_agent_steps": 160}
            )
        )

    monkeypatch.setattr(plugin, "run_process", dispatch)
    verified = []
    monkeypatch.setattr(
        synthetic, "validate_admission", lambda row, files: verified.append(row["sample_id"])
    )
    result = plugin.execute(
        {"files": {}, "python": "/native/python", "maps_root": "/maps", "max_wall_seconds": 600},
        {
            "branch": {
                "artifacts": {"report": str(branch), "trajectory_window": str(window)},
                "summary": {"artifact_reference": {"identity": "original-identity"}},
            },
            "rendered": {"artifacts": {"report": str(receipt), "render_report": str(render)}},
            "visual": {"artifacts": {"report": str(visual)}},
        },
        tmp_path / "out",
        None,
    )
    assert len(calls) == 1 and verified == ["new-history"]
    assert result["status"] == "completed" and result["sim_agent_steps"] == 160


@pytest.mark.parametrize(
    "epoch,count,accepted", [(True, 70, True), (True, 129, False), (False, 70, False)]
)
def test_pair_compiler_large_epoch_cohort_boundary(tmp_path, monkeypatch, epoch, count, accepted):
    from driversi_agents.chainflow import data, synthetic

    base = dict(
        schema_version=2,
        benchmark={"protocol": "fixture"},
        splits={
            "train": [dict(sample_id="original", kind="original_navtrain", source_log="train")],
            "development": [],
        },
    )
    monkeypatch.setattr(data, "validate_manifest", lambda p: base)
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            dict(
                settings={
                    "distributed_training": dict(
                        devices=[6, 7], checkpoint_interval=2, synthetic_every=8
                    )
                }
            )
        )
    )
    paths = []
    for i in range(count):
        p = tmp_path / f"sample-{i}.json"
        p.write_text(
            json.dumps(
                dict(
                    kind="chainflow_admitted_synthetic_sample",
                    row=dict(sample_id=f"s{i}", kind="synthetic_navtrain", source_log=f"log{i}"),
                    files={},
                )
            )
        )
        paths.append(p)
    checked = []

    class ReachedAllNativeAdmissions(Exception):
        pass

    def validate(row, files):
        checked.append(row["sample_id"])
        if len(checked) == count:
            raise ReachedAllNativeAdmissions

    monkeypatch.setattr(synthetic, "validate_admission", validate)
    if accepted:
        with pytest.raises(ReachedAllNativeAdmissions):
            plugin.compile_paired_task(
                "base",
                paths,
                model,
                tmp_path / "out",
                updates_per_arm=140,
                epoch_training=True,
                batch_size=2,
            )
        assert len(checked) == 70
    else:
        with pytest.raises(ValueError, match="admitted synthetic receipts"):
            plugin.compile_paired_task(
                "base",
                paths,
                model,
                tmp_path / "out",
                updates_per_arm=128,
                epoch_training=epoch,
                batch_size=2 if epoch else 1,
            )
        assert checked == []
