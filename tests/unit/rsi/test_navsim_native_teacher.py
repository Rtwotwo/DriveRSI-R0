from pathlib import Path

import pytest

from driversi.rsi.plugins.navsim_teacher import prepare


def config():
    return dict(
        scene="missing",
        road_graph="missing",
        python="missing",
        checkpoint="missing",
        maps_root="missing",
        arrival_action=[0.0, 0.0],
        search_candidates=0,
        device="cpu",
        max_wall_seconds=300,
    )


def test_unknown_teacher_rejected_before_file_access():
    with pytest.raises(ValueError, match="Unknown same-anchor"):
        prepare({**config(), "teacher_backend": "pretend_teacher"}, Path("/"), None)


def test_unknown_arrival_rejected_before_file_access():
    with pytest.raises(ValueError, match="Unknown arrival"):
        prepare({**config(), "arrival_backend": "teleport"}, Path("/"), None)


def test_native_arrival_and_baseline_cannot_be_taken_over_by_teacher():
    from driversi.evaluation.search.navsim_teacher import step_arguments
    teacher = object()
    assert step_arguments(None, teacher, force_native=True) == {}
    assert step_arguments(None, teacher) == {"teacher": teacher}
    assert step_arguments([-0.1, 0.0], teacher) == {
        "native_action": [-0.1, 0.0], "action_origin": "bounded_perturbation"}
    with pytest.raises(ValueError, match="silently ignore"):
        step_arguments([0.0, 0.0], teacher, force_native=True)


def test_native_arrival_waits_for_five_actual_200ms_observations():
    from driversi.evaluation.search.navsim_teacher import arrival_control
    for tick in range(8):
        assert arrival_control(tick, "driverl_native", [0.0, 0.0]) == ([0.0, 0.0], False)
    for tick in range(8, 20):
        assert arrival_control(tick, "driverl_native", [0.0, 0.0]) == (None, True)
    for tick in range(20):
        assert arrival_control(tick, "bounded_control", [-0.1, 0.0]) == ([-0.1, 0.0], False)


def test_cannot_mix_pdm_teacher_and_driverl_search():
    with pytest.raises(ValueError, match="cannot mix"):
        prepare(
            {**config(), "teacher_backend": "pdm_route_v1", "search_candidates": 8}, Path("/"), None
        )


def test_completed_teacher_reference_rechecks_hashes_and_copies_without_execution(
    tmp_path, monkeypatch
):
    import json

    from driversi.rsi.plugins import navsim_teacher
    from driversi.rsi.workflow.artifacts import seal
    from driversi.rsi.workflow.store import HarnessStore

    source = tmp_path / "source"
    (source / "execution").mkdir(parents=True)
    (source / "execution/scene.npz").write_bytes(b"fixture-scene")
    (source / "report.json").write_text(
        json.dumps(dict(status="completed", restore_exact=True, source_files={}))
    )
    declaration = dict(
        id="teacher",
        provider="navsim_native_teacher_probe",
        provider_version="2",
        output_kind="navsim_teacher_execution",
    )
    result = dict(
        artifacts={
            "report": str(source / "report.json"),
            "execution/scene.npz": str(source / "execution/scene.npz"),
        },
        sim_agent_steps=42,
        summary={},
    )
    result["summary"]["artifact_reference"] = seal(result, source, "origin", declaration)
    state = dict(
        id="origin",
        task=dict(steps=[declaration], source_groups=["train-log"]),
        stages=[dict(id="teacher", status="completed", result=result)],
    )
    monkeypatch.setattr(HarnessStore, "get", lambda self, task: state)
    monkeypatch.setattr(navsim_teacher, "protect", lambda *args: None)
    config, cap = navsim_teacher.prepare_reference(
        dict(source_task="origin", source_stage="teacher"), tmp_path, tmp_path / "h.sqlite"
    )
    assert cap == 0
    copied = navsim_teacher.run_reference(config, {}, tmp_path / "dest", tmp_path / "h.sqlite")
    assert copied["sim_agent_steps"] == 0
    assert copied["summary"]["original_sim_agent_steps"] == 42
    assert Path(copied["artifacts"]["execution/scene.npz"]).read_bytes() == b"fixture-scene"
    (source / "execution/scene.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        navsim_teacher.run_reference(config, {}, tmp_path / "bad", tmp_path / "h.sqlite")


def test_cannot_silently_apply_corridor_duration_to_native_teacher():
    with pytest.raises(ValueError, match="Recovery duration"):
        prepare({**config(), "recovery_seconds": 3.0}, Path("/"), None)


def test_source_corridor_cannot_mix_native_search():
    with pytest.raises(ValueError, match="cannot mix"):
        prepare(
            {**config(), "teacher_backend": "source_corridor_lqr_v1", "search_candidates": 8},
            Path("/"),
            None,
        )


def test_bool_cannot_masquerade_as_source_corridor_gain():
    with pytest.raises(ValueError, match="Position gain"):
        prepare(
            {**config(), "teacher_backend": "source_corridor_lqr_v1", "position_gain": True},
            Path("/"),
            None,
        )
