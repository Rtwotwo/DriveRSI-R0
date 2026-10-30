import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.evaluation.benchmarks import gallery
from driversi.evaluation.benchmarks.gallery import build_interaction_gallery, interaction_duration


def test_duration_uses_valid_control_intervals_not_video_length(tmp_path):
    path = tmp_path / "record.npz"
    np.savez(path, metadata=json.dumps({"dt_seconds": .1}), controlled_ids=["first", "second"],
             valid=np.array([[True, True], [True, True], [False, True], [False, False]]))
    result = interaction_duration(path)
    assert result["actual_seconds"] == pytest.approx(.4)
    assert result["any_controlled_seconds"] == pytest.approx(.3)
    assert result["two_controlled_seconds"] == pytest.approx(.2)
    assert result["per_actor_active_seconds"] == pytest.approx({"first": .2, "second": .3})


def test_gallery_requires_explicit_pickle_trust_before_creating_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="trusted"):
        build_interaction_gallery(tmp_path / "missing.toml", tmp_path, tmp_path, output)
    assert not output.exists()


def test_gallery_finalizes_unexpected_error_without_overwriting_previous_run(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('schema_version=1\nduration_seconds=9.0\nmax_actors=32\n'
                      'selection_file="selection.json"\nfollowing_config_file="following.toml"\n'
                      'selected_following_cases=[0]\n')
    (tmp_path / "selection.json").write_text(json.dumps({"schema_version": 1, "cases": [
        {"scene": "scene.npz", "scenario_id": "frozen"}]}))
    (tmp_path / "following.toml").write_text('duration_seconds=9.0\nspeeds_mps=[6]\nbumper_gaps_m=[20]\n')
    def fail(*args):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(gallery, "evaluate_following", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="injected"):
        build_interaction_gallery(config, tmp_path, tmp_path, output, trusted=True)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "error" and "injected failure" in report["error"]
    assert "injected failure" in (output / "index.html").read_text(encoding="utf-8")
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_interaction_gallery(config, tmp_path, tmp_path, output, trusted=True)
    assert (output / "report.json").read_bytes() == before


@pytest.mark.parametrize("error", [EOFError(), pickle.UnpicklingError("bad pickle")])
def test_corrupt_source_is_retained_as_a_rejected_case(tmp_path, monkeypatch, error):
    previous = SimpleNamespace(metadata={"scenario_id": "frozen", "source_log": "log",
                                         "map_name": "map"})
    monkeypatch.setattr(gallery.Scene, "load", lambda path: previous)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(gallery, "load_log", fail)
    result = gallery._real_case({"scene": "scene.npz", "scenario_id": "frozen"},
                                tmp_path, tmp_path, tmp_path / "case", 9., 32)
    assert result["status"] == "compile_rejected"
    assert type(error).__name__ in result["error"]
    assert (tmp_path / "case/case.json").exists()


@pytest.mark.parametrize("field", ["scenario_id", "source_log", "map_name", "source_start_timestamp_us"])
def test_changed_source_identity_cannot_replace_frozen_case(tmp_path, monkeypatch, field):
    metadata = {"scenario_id": "frozen", "source_log": "log", "map_name": "map",
                "source_start_timestamp_us": 100, "source_start_frame": 0, "split": "mini"}
    previous = SimpleNamespace(metadata=metadata, times=np.array([0., 3.]))
    changed = SimpleNamespace(metadata={**metadata, field: "changed"})
    monkeypatch.setattr(gallery.Scene, "load", lambda path: previous)
    monkeypatch.setattr(gallery, "load_log", lambda *a, **k: [{}])
    monkeypatch.setattr(gallery, "audit_horizons", lambda *a, **k: {})
    monkeypatch.setattr(gallery, "compile_frames", lambda *a, **k: changed)
    result = gallery._real_case({"scene": "scene.npz", "scenario_id": "frozen"},
                                tmp_path, tmp_path, tmp_path / "case", 9., 32)
    assert result["status"] == "compile_rejected"
    assert "frozen scene identity" in result["error"]
    assert not (tmp_path / "case/scene.npz").exists()


def _selection():
    return {"schema_version": 1, "cases": [
        {"scene": "first.npz", "scenario_id": "a"},
        {"scene": "second.npz", "scenario_id": "b"}]}


def test_adaptive_gallery_keeps_frozen_order_and_uses_independent_real_budgets():
    selection = _selection()
    config = {"schema_version": 2, "duration_seconds": 22., "duration_mode": "fit_source",
              "min_duration_seconds": 2.5,
              "cases": [{"scene": "second.npz", "duration_seconds": 6.}]}
    result = gallery._gallery_cases(config, selection, {"duration_seconds": 9.})
    assert [(case["scene"], case["scenario_id"], case["duration_seconds"]) for case in result] == [
        ("first.npz", "a", 22.), ("second.npz", "b", 6.)]
    assert all("duration_seconds" not in case for case in selection["cases"])


@pytest.mark.parametrize("override", [
    [{"scene": "missing.npz", "duration_seconds": 6.}],
    [{"scene": "first.npz", "duration_seconds": 6.},
     {"scene": "first.npz", "duration_seconds": 3.}],
    [{"scene": "first.npz", "duration_seconds": 6., "start": 100}],
])
def test_adaptive_overrides_cannot_replace_sources_or_change_start(override):
    config = {"schema_version": 2, "duration_seconds": 9., "cases": override}
    with pytest.raises(ValueError, match="unique frozen"):
        gallery._gallery_cases(config, _selection(), {"duration_seconds": 9.})


@pytest.mark.parametrize("duration", [False, -1, 0, float("nan"), float("inf"), "6"])
def test_adaptive_gallery_rejects_invalid_budgets(duration):
    with pytest.raises(ValueError, match="finite positive"):
        gallery._gallery_cases({"schema_version": 2, "duration_seconds": duration},
                               _selection(), {"duration_seconds": 9.})


def test_legacy_gallery_still_requires_nine_second_matched_protocol():
    with pytest.raises(ValueError, match="same horizon"):
        gallery._gallery_cases({"schema_version": 1, "duration_seconds": 9.},
                               _selection(), {"duration_seconds": 6.})
    with pytest.raises(ValueError, match="schema v2"):
        gallery._gallery_cases({"schema_version": 1, "duration_seconds": 9., "cases": [
            {"scene": "first.npz", "duration_seconds": 6.}]}, _selection(), {"duration_seconds": 9.})


def test_budget_below_minimum_is_kept_for_per_source_rejection():
    config = {"schema_version": 2, "duration_seconds": 2., "duration_mode": "fit_source",
              "min_duration_seconds": 2.5}
    cases = gallery._gallery_cases(config, _selection(), {"duration_seconds": 9.})
    assert len(cases) == 2 and all(case["duration_seconds"] == 2. for case in cases)


def test_rejected_horizon_is_saved_without_trying_a_shorter_successful_window(tmp_path, monkeypatch):
    previous = SimpleNamespace(times=np.array([0., 3.]), metadata={
        "scenario_id": "frozen", "source_log": "log", "map_name": "map", "source_start_frame": 12})
    plan = {"status": "rejected", "requested_duration_seconds": 22.,
            "planned_duration_seconds": None, "candidate_duration_seconds": .7,
            "available_future_seconds": .76, "backend_max_duration_seconds": 9.,
            "limiting_factors": ["source_exhaustion"], "quantization_loss_seconds": .06,
            "mode": "fit_source", "rejection_reasons": ["below_minimum_duration"]}
    calls = []
    monkeypatch.setattr(gallery.Scene, "load", lambda path: previous)
    monkeypatch.setattr(gallery, "load_log", lambda *a, **k: [{}])
    def select(*args, **kwargs):
        calls.append(kwargs)
        return plan
    monkeypatch.setattr(gallery, "plan_horizon", select)
    def no_compile(*args, **kwargs):
        pytest.fail("A rejected plan cannot invoke compilation or outcome-based retry")
    monkeypatch.setattr(gallery, "compile_frames", no_compile)
    result = gallery._real_case({"scene": "scene.npz", "scenario_id": "frozen"},
                                tmp_path, tmp_path, tmp_path / "case", 22., 32,
                                duration_mode="fit_source", min_duration=2.5)
    assert calls == [{"start": 12, "duration": 22., "mode": "fit_source", "min_duration": 2.5}]
    assert result["status"] == "compile_rejected" and result["planned_seconds"] is None
    assert result["interaction_probe"]["status"] == "not_run"
    assert json.loads((tmp_path / "case/horizon_plan.json").read_text()) == plan
    assert "below_minimum_duration" in result["error"]


def test_builder_dispatches_distinct_real_budgets_without_matching_synthetic_horizon(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('schema_version=2\nduration_seconds=22.0\nduration_mode="fit_source"\n'
                      'min_duration_seconds=2.5\nmax_actors=32\nselection_file="selection.json"\n'
                      'following_config_file="following.toml"\nselected_following_cases=[0]\n'
                      '[[cases]]\nscene="first.npz"\nduration_seconds=3.0\n')
    (tmp_path / "selection.json").write_text(json.dumps(_selection()))
    (tmp_path / "following.toml").write_text('duration_seconds=6.0\nspeeds_mps=[6]\nbumper_gaps_m=[20]\n')
    def following(path, output):
        output.mkdir()
        (output / "report.json").write_text(json.dumps({"cases": [{"checks": {"check": True},
            "speed_mps": 6., "initial_bumper_gap_m": 20., "passed": True, "modes": {}}],
            "passed": True, "independent_templates": 1, "backend_build": "test"}))
    requests = []
    def real(case, scene_dir, dataset_root, folder, duration, max_actors, **kwargs):
        requests.append((case["scene"], duration, kwargs))
        return {"id": case["scene"], "source_scene": case["scene"],
                "status": "compile_rejected", "requested_seconds": duration}
    monkeypatch.setattr(gallery, "evaluate_following", following)
    monkeypatch.setattr(gallery, "_real_case", real)
    monkeypatch.setattr(gallery.Scene, "load", lambda path: None)
    monkeypatch.setattr(gallery, "export_replay", lambda *a, **k: None)
    output = tmp_path / "output"
    build_interaction_gallery(config, tmp_path, tmp_path, output, trusted=True)
    assert requests == [("first.npz", 3., {"duration_mode": "fit_source", "min_duration": 2.5}),
                        ("second.npz", 22., {"duration_mode": "fit_source", "min_duration": 2.5})]
    report = json.loads((output / "report.json").read_text())
    assert report["schema_version"] == 2 and report["following"]["duration_seconds"] == 6.
    assert [case["duration_seconds"] for case in report["requested_cases"]] == [3., 22.]


@pytest.mark.parametrize("status", ["verified", "evidence_insufficient", "not_applicable", "failed"])
def test_probe_status_does_not_reclassify_successful_base_rollout(tmp_path, monkeypatch, status):
    metadata = {"scenario_id": "frozen", "source_log": "log", "map_name": "map",
                "source_start_timestamp_us": 100, "source_start_frame": 0, "split": "mini"}
    scene = SimpleNamespace(metadata=metadata, times=np.array([0., 3.]),
                            actor_ids=("a", "b"), controlled=np.array([True, True]), save=lambda path: None)
    graph = SimpleNamespace(save=lambda path: None)
    context = SimpleNamespace(save=lambda path: None)
    monkeypatch.setattr(gallery.Scene, "load", lambda path: scene)
    monkeypatch.setattr(gallery, "load_log", lambda *a, **k: [{}])
    monkeypatch.setattr(gallery, "plan_horizon", lambda *a, **k: {
        "status": "supported", "planned_duration_seconds": 3.})
    monkeypatch.setattr(gallery, "audit_horizons", lambda *a, **k: {})
    monkeypatch.setattr(gallery, "compile_frames", lambda *a, **k: scene)
    monkeypatch.setattr(gallery, "attach_nuplan_map", lambda *a: None)
    monkeypatch.setattr(gallery, "compile_lane_graph", lambda *a: graph)
    monkeypatch.setattr(gallery, "inspect_routes", lambda *a: {"all_resolved": True, "actors": {}})
    monkeypatch.setattr(gallery, "rollout", lambda *a, **k: None)
    monkeypatch.setattr(gallery, "summarize_rollout", lambda *a: {})
    monkeypatch.setattr(gallery, "interaction_duration", lambda *a: {"actual_seconds": 3., "two_controlled_seconds": 3.})
    monkeypatch.setattr(gallery, "verify_replay", lambda *a: {"joint_action_replay_exact": True})
    monkeypatch.setattr(gallery, "compile_context", lambda *a, **k: context)
    monkeypatch.setattr(gallery, "export_replay", lambda *a, **k: None)
    calls = []
    def probe(*args):
        calls.append(args)
        return {"status": status, "metrics": {"actual_response_delay_s": .3}}
    monkeypatch.setattr(gallery, "_probe_with_views", probe)
    arguments = ({"scene": "scene.npz", "scenario_id": "frozen"}, tmp_path, tmp_path)
    result = gallery._real_case(*arguments, tmp_path / "adaptive", 22., 32, duration_mode="fit_source")
    assert result["status"] == "complete"
    assert result["interaction_probe"]["status"] == status
    assert result["qualification"] == {"source_time": "supported", "route_task": "resolved",
                                       "joint_control": "available", "interaction": status}
    assert len(calls) == 1 and calls[0][:3] == (scene, graph, context)
    calls.clear()
    legacy = gallery._real_case(*arguments, tmp_path / "legacy", 9., 32)
    assert legacy["status"] == "complete" and legacy["interaction_probe"]["status"] == "not_run"
    assert not calls
    monkeypatch.setattr(gallery, "verify_replay", lambda *a: {"joint_action_replay_exact": False})
    invalid = gallery._real_case(*arguments, tmp_path / "invalid", 22., 32, duration_mode="fit_source")
    assert invalid["status"] == "contract_failure" and invalid["interaction_probe"]["status"] == "not_run"
    assert not calls


@pytest.mark.parametrize("unequal", [False, True])
def test_probe_views_compare_only_equal_recorded_horizons_without_padding(tmp_path, monkeypatch, unequal):
    folder = tmp_path / "scene-a"
    folder.mkdir()
    branches = ("baseline", "lead_brake", "action_replay_keep", "action_replay_brake")
    def probe(scene, graph, output):
        output.mkdir()
        for branch in branches:
            steps = 20 if unequal and "brake" in branch else 30
            np.savez(output / (branch + ".npz"), metadata=json.dumps({"dt_seconds": .1}),
                     valid=np.ones((steps, 2), dtype=bool))
        return {"status": "verified", "recordings": {branch: branch + ".npz" for branch in branches}}
    monkeypatch.setattr(gallery, "probe_interaction", probe)
    exports = []
    def export(scene, recording, page, **kwargs):
        exports.append((recording, page, kwargs))
    monkeypatch.setattr(gallery, "export_replay", export)
    result = gallery._probe_with_views(None, None, None, folder)
    assert result["status"] == "verified" and len(exports) == len(result["views"]) == 4
    comparisons = [options["compare_path"] for _, _, options in exports if options["compare_path"]]
    assert len(comparisons) == (0 if unequal else 2)
    assert len(result["comparison_notes"]) == (2 if unequal else 0)
    assert result["recording_seconds"]["baseline"] == 3.
    assert result["recording_seconds"]["lead_brake"] == (2. if unequal else 3.)
    for view in result["views"]:
        assert view["path"].startswith("real/scene-a/probe/")


def test_probe_exception_preserves_its_partial_checks_and_returns_failed(tmp_path, monkeypatch):
    folder = tmp_path / "scene-a"
    folder.mkdir()
    def fail(scene, graph, output):
        output.mkdir()
        (output / "report.json").write_text(json.dumps({
            "status": "failed", "checks": {"parent_unchanged": False}, "recordings": {}}))
        raise RuntimeError("injected probe failure")
    monkeypatch.setattr(gallery, "probe_interaction", fail)
    result = gallery._probe_with_views(None, None, None, folder)
    assert result["status"] == "failed" and result["checks"]["parent_unchanged"] is False
    assert "injected probe failure" in result["reason"]
    assert result["report_path"] == "real/scene-a/probe/report.json"
