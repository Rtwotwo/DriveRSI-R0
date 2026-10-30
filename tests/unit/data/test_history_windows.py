import numpy as np
import pytest

from driversi.data.datasets.history_windows import executed_label, four_frame_ticks


def test_four_distinct_frames_and_no_future():
    assert four_frame_ticks(20, 5) == (5, 10, 15, 20)
    for a, s in [(14, 5), (20, 0), (True, 1), (20, False)]:
        with pytest.raises(ValueError):
            four_frame_ticks(a, s)


def test_executed_label_rotation_and_actual_clock():
    state = np.zeros((8, 2, 13))
    state[:, 1, 0] = 4
    state[:, 1, 1] = np.arange(8) * 2
    state[:, 1, 3] = np.pi / 2
    result = executed_label(state, 1, 3, [5, 7], 0.1)
    np.testing.assert_allclose(result["poses"], [[4, 0, 0], [8, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(result["times"], [0.2, 0.4])
    for ticks in [[3, 4], [5, 5], [7, 8], [4.5, 6.0]]:
        with pytest.raises(ValueError):
            executed_label(state, 1, 3, ticks, 0.1)


@pytest.fixture
def rendered_source(tmp_path):
    import copy
    import json

    from tests.unit.data.test_mtgs_episode import source as existing

    path, doc = existing.__wrapped__(tmp_path)
    doc["initial_world_by_generation"] = {"0": doc["initial_world_by_generation"]["0"]}
    doc["steps"] = [row for row in doc["steps"] if row["generation"] == 0]
    doc["frames"] = [row for row in doc["frames"] if row["generation"] == 0]
    for tick in (1, 3):
        row = copy.deepcopy(doc["frames"][0])
        row.update(
            tick=tick,
            render_tick=tick,
            time_seconds=tick * 0.1,
            simulation_timestamp_us=1000000 + tick * 100000,
        )
        for c in row["cameras"]:
            c.update(
                source_timestamp_us=1000000 + tick * 100000,
                camera_timestamp_us=999000 + tick * 100000,
            )
        doc["frames"].append(row)
    doc["frames"].sort(key=lambda r: r["tick"])
    path.write_text(json.dumps(doc))
    return path, doc


def test_sample_input_label_boundary(rendered_source, tmp_path):
    import json

    from driversi.data.datasets.history_windows import export_four_frame_sample

    path, doc = rendered_source
    s = export_four_frame_sample(
        path,
        tmp_path / "sample",
        actor_id="ego",
        anchor_tick=3,
        history_stride=1,
        future_steps=1,
        target_stride=1,
        include_future_images=True,
    )
    assert s["history_ticks"] == [0, 1, 2, 3]
    assert [f["tick"] for f in s["target"]["future_frames"]] == [4]
    assert all(f["relative_seconds"] <= 0 for f in s["input"]["frames"])
    assert "episode" in s["permissions"] and not s["trainable"]
    assert not any(
        "source" in c or "state" in c or "camera_to_world" in c["calibration"]
        for f in s["input"]["frames"]
        for c in f["cameras"]
    )
    target = json.loads((tmp_path / "sample/trajectory.json").read_text())
    assert target["times"] == [0.1]
    np.testing.assert_allclose(target["poses"], [[1, 0, 0]])


@pytest.mark.parametrize("change", ["missing_capture", "short_future", "no_owner_camera", "reset"])
def test_sample_rejects_invalid_windows(rendered_source, tmp_path, change):
    import json

    from driversi.data.datasets.history_windows import export_four_frame_sample

    path, doc = rendered_source
    future = 1
    actor = "ego"
    if change == "missing_capture":
        doc["frames"] = [f for f in doc["frames"] if f["tick"] != 1]
    elif change == "short_future":
        future = 2
    elif change == "no_owner_camera":
        actor = "car"
    else:
        doc["frames"][1]["generation"] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        export_four_frame_sample(
            path,
            tmp_path / "bad",
            actor_id=actor,
            anchor_tick=3,
            history_stride=1,
            future_steps=future,
            target_stride=1,
        )
    assert not (tmp_path / "bad/sample.json").exists()


@pytest.fixture
def source_history_case(tmp_path, straight_scene):
    from tests.unit.data.test_sensors import png_bytes, source_frame

    frames = [source_frame() for _ in range(4)]
    for i, f in enumerate(frames):
        f.update(timestamp=1000000 + i * 499997, token=f"frame-{i}")
        f["cams"]["CAM_F0"]["data_path"] = f"camera-{i}.png"
        (tmp_path / f"camera-{i}.png").write_bytes(png_bytes(32, 24))
    straight_scene.metadata.update(
        source_start_frame=3,
        source_start_timestamp_us=frames[-1]["timestamp"],
        scenario_id="frame-3",
        source_log="log-a",
        map_name="map-a",
    )
    return frames, straight_scene, tmp_path


def test_real_history_keeps_actual_clock(source_history_case):
    from driversi.data.datasets.history_windows import source_history

    frames, scene, root = source_history_case
    h = source_history(frames, scene, root, camera_ids=["CAM_F0"])
    assert h["span_seconds"] == 1.499991
    assert len(h["frames"]) == 4 and h["frames"][-1]["relative_seconds"] == 0
    assert not h["rgb_used_by_generator"]


@pytest.mark.parametrize("fault", ["gap", "log", "token", "file", "short"])
def test_real_history_rejects_missing_or_wrong_source(source_history_case, fault):
    from driversi.data.datasets.history_windows import source_history

    frames, scene, root = source_history_case
    if fault == "gap":
        frames[0]["timestamp"] -= 1000000
    if fault == "log":
        frames[0]["log_name"] = "wrong-log"
    if fault == "token":
        frames[-1]["token"] = "wrong-anchor"
    if fault == "file":
        (root / "camera-2.png").unlink()
    if fault == "short":
        scene.metadata["source_start_frame"] = 2
    with pytest.raises(ValueError):
        source_history(frames, scene, root, camera_ids=["CAM_F0"])


def test_owner_termination_on_last_target_is_not_valid(rendered_source, tmp_path):
    import json

    from driversi.data.datasets.history_windows import export_four_frame_sample

    path, doc = rendered_source
    last = doc["steps"][-1]
    last["events"]["route_support_end"] = [True, False]
    last["valid"][0] = True
    last["alive"][0] = False
    last["truncated"][0] = True
    last["state"][0][9] = 1.0
    path.write_text(json.dumps(doc))
    sample = export_four_frame_sample(
        path,
        tmp_path / "last-state",
        actor_id="ego",
        anchor_tick=3,
        history_stride=1,
        future_steps=1,
        target_stride=1,
    )
    assert not sample["safety"]["owner_valid_throughout"]
    assert "owner_inactive_within_sample" in sample["rejection_reasons"]
