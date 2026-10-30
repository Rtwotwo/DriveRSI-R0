import json
from types import SimpleNamespace

import pytest

from driversi.rsi.plugins import native_branch
from driversi.rsi.workflow.providers import builtin_providers


def test_native_render_export_cannot_be_used_as_training_dataset():
    p = builtin_providers()
    assert p.get("native_branch_selection").inputs == (("world", "navsim_teacher_execution"),)
    assert p.get("native_branch_gs").inputs == (("branch", "selected_native_branch"),)
    assert p.get("native_branch_history_future_export").output == "navsim_inspection_dataset"
    assert p.get("native_branch_history_future_export").output != "native_training_dataset"


def test_select_consumes_real_nested_producer_artifact_names(tmp_path, monkeypatch):
    execution = tmp_path / "producer/execution"
    execution.mkdir(parents=True)
    for name in ("scene.npz", "road-graph.json", "teacher.npz", "teacher-targets.json"):
        (execution / name).write_text(name)
    report = dict(
        status="completed",
        restore_exact=True,
        history_changed=True,
        source_files={},
        arrival_tick=20,
        branches={
            "teacher": dict(
                completed_steps=50,
                failure=None,
                trajectory_admitted=False,
                score=None,
                events={},
                object_contacts=[],
            )
        },
    )
    path = tmp_path / "producer/report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        native_branch.Scene,
        "load",
        lambda p: SimpleNamespace(metadata=dict(source_log="train", split="train")),
    )
    monkeypatch.setattr(native_branch, "protect", lambda *a: None)
    from driversi.data.datasets import executed_window

    monkeypatch.setattr(
        executed_window,
        "export_executed_window",
        lambda scene, rollout, out, **kw: out.write_text("{}"),
    )
    result = native_branch.run_select(
        {"branch": "teacher"},
        {
            "world": {
                "artifacts": {
                    "report": str(path),
                    "execution/scene.npz": str(execution / "scene.npz"),
                },
                "summary": {"artifact_reference": {"identity": "sealed-parent"}},
            }
        },
        tmp_path / "selected",
        tmp_path / "registry",
    )
    assert result["summary"]["execution"] == str(execution)
    assert result["summary"]["teacher_qualified"] is False
    assert result["admission"]["trainable"] is False


def test_selection_rejects_incomplete_recording_before_render(tmp_path):
    p = tmp_path / "report.json"
    p.write_text(json.dumps(dict(status="running", restore_exact=False)))
    with pytest.raises(ValueError, match="Complete"):
        native_branch.run_select(
            {"branch": "teacher"},
            {"world": {"artifacts": {"report": str(p)}}},
            tmp_path / "out",
            tmp_path / "registry",
        )
    assert not (tmp_path / "out").exists()


def test_trajectory_admission_rejects_baseline_contacts_and_low_score():
    import copy

    evidence = dict(
        branch="teacher",
        completed_steps=50,
        producer_identity="sealed",
        history_changed=True,
        restore_exact=True,
        arrival_contact_records=0,
        future_contact_records=0,
        events=dict(collision=0, offroad=0, route_support_end=0),
        score=dict(
            score=0.95,
            driving_direction_compliance=1.0,
            actual_world_seconds=5.0,
            protocol="navsim-v1-native-math-executed-counterfactual-v1",
        ),
    )
    native_branch.validate_trajectory_evidence(evidence)
    for field, value in [
        ("branch", "baseline"),
        ("arrival_contact_records", 1),
        ("history_changed", False),
    ]:
        bad = copy.deepcopy(evidence)
        bad[field] = value
        with pytest.raises(ValueError, match="admission"):
            native_branch.validate_trajectory_evidence(bad)
    for value in [0.949999, float("nan")]:
        bad = copy.deepcopy(evidence)
        bad["score"]["score"] = value
        with pytest.raises(ValueError, match="admission"):
            native_branch.validate_trajectory_evidence(bad)


def test_selection_ten_future_is_opt_in_without_changing_old_callers():
    assert native_branch.prepare_select({'branch': 'teacher'}, None, None)[0]['future_frames'] == 8
    assert native_branch.prepare_select(
        {'branch': 'teacher', 'future_frames': 10}, None, None
    )[0]['future_frames'] == 10
    for count in (9, True, 10.0):
        with pytest.raises(ValueError, match='actual future'):
            native_branch.prepare_select({'branch': 'teacher', 'future_frames': count}, None, None)
