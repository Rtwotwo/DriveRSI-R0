import json
import pickle

import numpy as np
import pytest

from driversi.cli import main
from driversi.contracts.scene import Scene
from driversi.data.sources.navsim import (
    HorizonPlanError,
    audit_horizons,
    compile_frames,
    plan_horizon,
    transform_box,
    yaw_from_quaternion,
)


def frames():
    result = []
    for i in range(5):
        result.append({
            "timestamp": 1_000_000 + i * 500_000, "token": f"{i:016d}", "log_name": "log",
            "map_location": "test-map", "ego2global_translation": [100 + i * 2, 200, 0],
            "ego2global_rotation": [1, 0, 0, 0], "traffic_lights": [],
            "anns": {"gt_boxes": np.asarray([[10, 0, 0, 4, 2, 1.5, 0]]),
                     "gt_names": ["vehicle"], "track_tokens": ["car-b"]},
        })
    return result


def test_box_transform_rotates_translation_and_heading():
    actual = transform_box([2, 1, 0, 4, 2, 1, 0], [10, 20, 3], np.pi / 2)
    np.testing.assert_allclose(actual, [9, 22, 3, np.pi / 2])
    assert yaw_from_quaternion([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]) == pytest.approx(np.pi / 2)


def test_compile_keeps_stable_ids_and_metric_motion(tmp_path):
    scene = compile_frames(frames(), start=0, duration=2, split="mini")
    assert scene.actor_ids == ("ego", "car-b")
    np.testing.assert_allclose(scene.poses[0, :, 0], [1.461, 10])
    np.testing.assert_allclose(scene.velocities[:, :, 0], 4)
    assert scene.controlled.tolist() == [True, True]
    target = tmp_path / "scene.npz"
    scene.save(target)
    restored = Scene.load(target)
    np.testing.assert_array_equal(restored.poses, scene.poses)
    assert restored.metadata == scene.metadata


def test_dropped_track_is_not_interpolated_through_absence():
    source = frames()
    source[2]["anns"] = {"gt_boxes": np.empty((0, 7)), "gt_names": [], "track_tokens": []}
    scene = compile_frames(source, start=0, duration=2, split="mini")
    assert scene.actor_ids == ("ego", "car-b")
    assert not scene.presence[10:15, 1].any()
    assert scene.presence[:10, 1].all() and scene.presence[15:, 1].all()
    assert not scene.poses[10:15, 1].any()
    assert scene.metadata["excluded_incomplete_tracks"] == 0
    assert "no_spawn_despawn" not in scene.metadata["limitations"]


def test_gap_and_out_of_range_duration_fail():
    with pytest.raises(ValueError, match="cover"):
        compile_frames(frames(), start=0, duration=3, split="mini")
    source = frames()
    source[2]["timestamp"] += 400_000
    with pytest.raises(ValueError, match="gap"):
        compile_frames(source, start=0, duration=2, split="mini")


def test_nonfinite_scene_cannot_be_saved(tmp_path):
    scene = compile_frames(frames(), start=0, duration=2, split="mini")
    scene.poses[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="poses"):
        scene.save(tmp_path / "invalid.npz")


def test_missing_signals_remain_distinct_from_empty_signals():
    source = frames()
    del source[1]["traffic_lights"]
    scene = compile_frames(source, start=0, duration=2, split="mini")
    signals = scene.metadata["traffic_lights"]
    assert signals[0]["records"] == signals[1]["records"] == []
    assert signals[0]["available"] is True
    assert signals[1]["available"] is False


def test_horizon_endpoint_matches_compiler_bracketing_and_counts_raw_tracks():
    source = frames()
    # An unsupported object remains part of raw population accounting. The
    # disappearing car returns, but must not count as persistent in this window.
    for i, tokens in enumerate([["car-b", "cone"], ["car-b", "cone", "new-car"],
                                ["cone", "new-car"], ["car-b", "cone"], ["car-b", "cone"]]):
        source[i]["anns"] = {"gt_boxes": np.tile([10, 0, 0, 4, 2, 1.5, 0], (len(tokens), 1)),
                             "gt_names": ["traffic_cone" if t == "cone" else "vehicle" for t in tokens],
                             "track_tokens": tokens}
    audit = audit_horizons(source, start=0, durations=[.5, .6, 1.5])
    exact, bracketed, returned = audit["horizons"]
    assert exact["source_frames"] == 2
    assert exact["population"]["persistent_tracks"] == 2
    assert bracketed["source_end_offset_seconds"] == 1
    assert bracketed["source_frames"] == 3
    assert bracketed["population"] == {"initial_tracks": 2, "persistent_tracks": 1,
                                        "initial_excluded_incomplete_tracks": 1, "new_arriving_tracks": 1}
    assert returned["population"]["persistent_tracks"] == 1
    scene = compile_frames(source, start=0, duration=.6, split="mini")
    assert scene.metadata["source_frames"] == bracketed["source_frames"]
    assert "car-b" in scene.actor_ids
    assert scene.presence.shape == scene.poses.shape[:2]


def test_horizon_continuity_includes_bracketing_frame_and_reports_past_support():
    source = frames()
    for frame, timestamp in zip(source, [0, 750_000, 1_250_000, 2_250_001, 2_750_001]):
        frame["timestamp"] = timestamp
    audit = audit_horizons(source, start=1, durations=[.5, .6, 3])
    assert audit["source"]["continuous_segment_count"] == 2
    assert audit["source"]["longest_continuous_seconds"] == 1.25
    assert audit["start"]["available_history_seconds"] == .75
    assert audit["start"]["available_future_seconds"] == .5
    assert audit["start"]["remaining_file_seconds"] == pytest.approx(2.000001)
    valid, gap, insufficient = audit["horizons"]
    assert valid["source_continuous"] is True
    assert gap["file_covers_duration"] is True
    assert gap["source_continuous"] is False
    assert gap["population"] is None
    assert gap["unavailable_reasons"] == ["source_gap_exceeds_0.75_seconds"]
    assert insufficient["file_covers_duration"] is False
    assert insufficient["population"] is None
    assert insufficient["source_end_frame"] is None
    assert insufficient["unavailable_reasons"] == ["file_does_not_cover_requested_duration"]


@pytest.mark.parametrize(("field", "reason"), [("log_name", "source_log_boundary"),
                                                ("map_location", "source_map_boundary")])
def test_horizon_does_not_cross_source_identity_or_hide_a_return(field, reason):
    source = frames()
    source[2][field] = "other"
    audit = audit_horizons(source, start=1, durations=[.5, 1])
    assert audit["source"]["continuous_segment_count"] == 3
    assert audit["source"]["continuous_segments"][1]["duration_seconds"] == 0
    assert audit["start"]["available_future_seconds"] == 0
    for horizon in audit["horizons"]:
        assert horizon["file_covers_duration"] is True
        assert horizon["source_continuous"] is False
        assert horizon["population"] is None
        assert horizon["unavailable_reasons"] == [reason]
    after = audit_horizons(source, start=3, durations=[.5])
    assert after["start"]["available_history_seconds"] == 0
    assert after["horizons"][0]["source_continuous"] is True


def test_source_support_and_compiler_duration_support_are_independent():
    source = [dict(frames()[0], timestamp=i * 500_000) for i in range(25)]
    audit = audit_horizons(source, start=0, durations=[9, 10, 13])
    assert [h["compiler_duration_supported"] for h in audit["horizons"]] == [True, True, True]
    assert [h["source_continuous"] for h in audit["horizons"]] == [True, True, False]
    assert audit["horizons"][1]["population"]["persistent_tracks"] == 1
    # One-frame segments have zero support, including at a file's final frame.
    single = audit_horizons([source[0]], start=0, durations=[.1])
    assert single["source"]["interval_seconds"] is None
    assert single["start"]["available_history_seconds"] == 0
    assert single["start"]["available_future_seconds"] == 0
    assert single["horizons"][0]["population"] is None


@pytest.mark.parametrize("start", [-1, 5, 1.5, True, "1"])
def test_horizon_rejects_invalid_start(start):
    with pytest.raises(ValueError, match="start"):
        audit_horizons(frames(), start=start, durations=[.5])


@pytest.mark.parametrize("durations", [[], [0], [-.1], [.15], [1e-12], [np.nan], [np.inf],
                                         [True], ["1"], [.5, None], "1"])
def test_horizon_rejects_invalid_durations(durations):
    with pytest.raises(ValueError, match="[Dd]uration"):
        audit_horizons(frames(), start=0, durations=durations)


def test_horizon_cli_reports_unsupported_windows_and_never_overwrites(tmp_path, capsys):
    source = tmp_path / "source.pkl"
    original = pickle.dumps(frames())
    source.write_bytes(original)
    arguments = ["audit-horizons", "--log-file", str(source), "--trusted-pickle"]
    # Zero means the diagnostic completed, including unavailable default horizons.
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["source_file"] == str(source.resolve())
    assert [item["duration_seconds"] for item in report["horizons"]] == [3, 9, 30, 60]
    assert all(item["population"] is None for item in report["horizons"])
    output = tmp_path / "report.json"
    assert main([*arguments, "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert json.loads(saved) == report
    for existing in [output, source]:
        with pytest.raises(SystemExit) as error:
            main([*arguments, "--output", str(existing)])
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main([*arguments, "--start-frame", "-1"])
    assert error.value.code == 2
    assert source.read_bytes() == original
    assert output.read_bytes() == saved


def test_horizon_cli_reports_single_frame_support_without_weakening_compiler_reader(tmp_path, capsys):
    from driversi.data.sources.navsim import load_log

    source = tmp_path / "single.pkl"
    source.write_bytes(pickle.dumps(frames()[:1]))
    arguments = ["audit-horizons", "--log-file", str(source), "--trusted-pickle"]
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"]["longest_continuous_seconds"] == 0
    assert report["source"]["continuous_segment_count"] == 1
    assert all(item["population"] is None for item in report["horizons"])
    with pytest.raises(ValueError, match="at least 2"):
        load_log(source, trusted=True)
    source.write_bytes(pickle.dumps([]))
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_fit_source_selects_each_start_independently_and_keeps_population_accounting():
    source = frames()
    source[-1]["anns"] = {"gt_boxes": np.empty((0, 7)), "gt_names": [], "track_tokens": []}
    early = plan_horizon(source, start=0, duration=3, mode="fit_source")
    late = plan_horizon(source, start=3, duration=3, mode="fit_source")
    assert (early["planned_duration_seconds"], late["planned_duration_seconds"]) == (2, .5)
    assert early["limiting_factors"] == ["source_exhaustion"]
    assert late["source_end_frame"] == 4
    assert early["population"]["initial_excluded_incomplete_tracks"] == 1
    assert early["scope"].startswith("time_support_only")
    scene = compile_frames(source, start=0, duration=3, duration_mode="fit_source", split="mini")
    assert scene.times[-1] == 2
    assert scene.metadata["horizon_plan"] == early
    assert "car-b" in scene.actor_ids
    assert scene.presence.shape == scene.poses.shape[:2]


def test_fit_source_floors_non_grid_budget_and_source_with_same_segment_bracket():
    source = frames()[:2]
    source[0]["timestamp"], source[1]["timestamp"] = 0, 740_000
    plan = plan_horizon(source, start=0, duration=3, mode="fit_source")
    assert plan["planned_duration_seconds"] == .7
    assert plan["quantization_loss_seconds"] == pytest.approx(.04)
    assert plan["source_end_offset_seconds"] == .74
    assert plan["source_end_frame"] == plan["source_segment_end_frame"] == 1
    budget = plan_horizon(frames(), start=0, duration=.69, mode="fit_source")
    assert budget["planned_duration_seconds"] == .6
    assert budget["quantization_loss_seconds"] == pytest.approx(.09)
    assert budget["limiting_factors"] == ["step_quantization"]


@pytest.mark.parametrize(("change", "reason"), [
    ("gap", "source_gap_exceeds_0.75_seconds"),
    ("log_name", "source_log_boundary"), ("map_location", "source_map_boundary"),
])
def test_fit_source_cannot_borrow_bracketing_frame_across_boundary(change, reason):
    source = frames()
    if change == "gap":
        for frame in source[2:]:
            frame["timestamp"] += 300_000
    else:
        source[2][change] = "other"
    plan = plan_horizon(source, start=0, duration=.6, mode="fit_source")
    assert plan["planned_duration_seconds"] == .5
    assert plan["source_end_frame"] == 1
    assert plan["limiting_factors"] == [reason]
    exact = plan_horizon(source, start=0, duration=.6)
    assert exact["status"] == "rejected"
    assert exact["planned_duration_seconds"] is None
    assert reason in exact["rejection_reasons"]


def test_backend_cap_is_reported_separately_from_source_support():
    source = [dict(frames()[0], timestamp=i * 500_000) for i in range(1205)]
    plan = plan_horizon(source, start=0, duration=610, mode="fit_source")
    assert plan["available_future_seconds"] == 602
    assert plan["planned_duration_seconds"] == plan["backend_max_duration_seconds"] == 600
    assert set(plan["limiting_factors"]) == {"source_exhaustion", "backend_cap"}
    with pytest.raises(HorizonPlanError) as error:
        compile_frames(source, start=0, duration=601, split="mini")
    assert "requested_duration_exceeds_backend_cap" in error.value.plan["rejection_reasons"]
    assert error.value.plan["candidate_duration_seconds"] == 600


def test_exact_never_shortens_and_unusable_tail_retains_rejection_plan():
    with pytest.raises(HorizonPlanError) as error:
        compile_frames(frames(), start=0, duration=3, split="mini")
    assert error.value.plan["planned_duration_seconds"] is None
    assert error.value.plan["candidate_duration_seconds"] == 2
    short = plan_horizon(frames(), start=3, duration=3, mode="fit_source", min_duration=.6)
    assert short["status"] == "rejected"
    assert short["rejection_reasons"] == ["candidate_below_min_duration"]
    assert short["candidate_duration_seconds"] == .5
    tail = plan_horizon(frames(), start=4, duration=3, mode="fit_source")
    assert tail["available_future_seconds"] == tail["candidate_duration_seconds"] == 0
    assert tail["source_end_frame"] is None


@pytest.mark.parametrize("options", [
    {"duration": True}, {"duration": np.inf}, {"duration": np.nan}, {"duration": 0},
    {"duration": .15}, {"min_duration": False}, {"min_duration": np.inf},
    {"min_duration": .01}, {"start": True}, {"mode": "unknown"},
])
def test_planner_rejects_malformed_parameters(options):
    parameters = {"start": 0, "duration": 1}
    parameters.update(options)
    with pytest.raises(ValueError):
        plan_horizon(frames(), **parameters)


@pytest.mark.parametrize("timestamp", [True, 1.2, np.nan, np.inf, "1000000"])
def test_planner_does_not_coerce_invalid_source_timestamps(timestamp):
    source = frames()
    source[0]["timestamp"] = timestamp
    with pytest.raises(ValueError, match="timestamp"):
        plan_horizon(source, start=0, duration=1)


def test_compiler_keeps_stationary_box_centre_and_audits_half_turn_without_fake_spin():
    source = frames()
    for index, frame in enumerate(source):
        # Ego moves, but this world object remains at global x=110. The source
        # annotation changes its rectangular box axis without centre motion.
        frame["anns"]["gt_boxes"][0, 0] = 10 - index * 2
        frame["anns"]["gt_boxes"][0, 6] = np.pi if index > 0 else 0
    scene = compile_frames(source, start=0, duration=2, split="mini")
    np.testing.assert_allclose(scene.poses[:, 1, 0], 10)
    np.testing.assert_allclose(scene.poses[:, 1, 3], 0, atol=1e-12)
    assert scene.controlled.tolist() == [True, False]
    corrections = scene.metadata["heading_audit"]["corrections"]
    assert len(corrections) == 1
    assert corrections[0]["actor_id"] == "car-b"
    assert source[1]["anns"]["gt_boxes"][0, 6] == np.pi
    timing = scene.metadata["source_timing"]
    assert timing["sample_count"] == 5
    assert timing["interval_seconds"] == {"min": .5, "median": .5, "max": .5}
    assert timing["target_dt_seconds"] == .1


def test_late_vehicle_heading_audit_has_source_window_indices():
    source = frames()
    for i in [1, 2]:
        ann = source[i]["anns"]
        ann["gt_boxes"] = np.concatenate((ann["gt_boxes"], [[30-2*i, 5, 0, 4, 2, 1.5, 0 if i == 1 else np.pi]]))
        ann["gt_names"].append("vehicle")
        ann["track_tokens"].append("late")
    scene = compile_frames(source, start=0, duration=2, split="mini")
    corrections = [x for x in scene.metadata["heading_audit"]["corrections"] if x["actor_id"] == "late"]
    assert len(corrections) == 1
    item = corrections[0]
    assert (item["source_start_index"], item["source_end_index"]) == (1, 2)
    assert (item["start_time_seconds"], item["end_time_seconds"]) == (.5, 1.)
    actor = scene.actor_ids.index("late")
    assert scene.poses[10, actor, 3] == pytest.approx(0.)


def test_first_observation_retains_velocity_in_world_axes():
    source = frames()
    for frame in source:
        frame["ego2global_rotation"] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        frame["anns"]["gt_velocity_3d"] = [[2., 0., .1]]
    scene = compile_frames(source, start=0, duration=2, split="mini")
    value = scene.metadata["first_observations"]["car-b"]
    np.testing.assert_allclose(value["velocity_world"], [0., 2., .1], atol=1e-10)
    assert value["velocity_source"] == "gt_velocity_3d_rotated_by_ego_yaw"
