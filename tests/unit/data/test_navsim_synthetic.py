import numpy as np
import pytest

from driversi.data.datasets.navsim_synthetic import frame_geometry, rear_axle_states


def test_rear_axle_offset_rotates_with_vehicle():
    state = np.zeros((3, 1, 13))
    state[:, 0, :2] = [10.0, 20.0]
    state[:, 0, 3] = [0.0, np.pi / 2, np.pi]
    pose = rear_axle_states(state, 0, offset=2.0)
    np.testing.assert_allclose(pose[:, :2], [[8, 20], [10, 18], [12, 20]])
    assert np.array_equal(state[:, 0, :2], np.tile([10.0, 20.0], (3, 1)))


def test_geometry_in_rear_axle_frame_retains_present_inactive_object(straight_scene):
    scene = straight_scene
    scene.actor_ids = ("ego", *scene.actor_ids[1:])
    state = np.zeros((len(scene.actor_ids), 13))
    state[:, 0:3] = [10, 25, 1]
    state[:, 3] = np.pi
    state[:, 4:7] = [0, 2, 0]
    state[:, 9] = 1  # termination does not erase physical presence
    world = {"state": state.tolist(), "presence": [True] * len(state)}
    pose = np.array([10.0, 20.0, 0.0, np.pi / 2])
    anns = frame_geometry(scene, world, pose)
    assert len(anns["gt_boxes"]) == len(state) - 1
    np.testing.assert_allclose(
        anns["gt_boxes"][:, :3], np.tile([5, 0, 1], (len(state) - 1, 1)), atol=1e-6
    )
    np.testing.assert_allclose(
        anns["gt_velocity_3d"], np.tile([2, 0, 0], (len(state) - 1, 1)), atol=1e-6
    )
    np.testing.assert_allclose(anns["gt_boxes"][:, 6], np.pi / 2)
    world["presence"] = [False] * len(state)
    empty = frame_geometry(scene, world, pose)
    assert empty["gt_boxes"].shape == (0, 7)
    assert empty["gt_velocity_3d"].shape == (0, 3)


def test_nonfinite_world_rejected():
    with pytest.raises(ValueError):
        rear_axle_states(np.full((4, 2, 13), np.nan), 0)


def test_subset_requires_actual_right_camera(tmp_path):
    from driversi.data.datasets.navsim_synthetic import export_navsim_sample

    with pytest.raises(ValueError, match="CAM_R0"):
        export_navsim_sample(
            tmp_path / "missing.json",
            tmp_path / "missing.npz",
            tmp_path / "new",
            anchor_tick=20,
            cameras=["CAM_F0"],
        )
    assert not (tmp_path / "new").exists()


@pytest.fixture
def navsim_export_source(monkeypatch, tmp_path, straight_scene):
    """Renderer boundary fixture: executed states are independent of RGB availability."""
    import hashlib
    import json

    from driversi.data.assets.mtgs_assets import file_sha256

    scene = straight_scene
    scene.actor_ids = ("ego", *scene.actor_ids[1:])
    scene.metadata.update(
        pose_reference="vehicle_center",
        coordinate_frame="map_axes_translated_from_initial_ego",
        split="development",
        origin_xyz=[0, 0, 0],
        map_name="map",
        source_log="log",
        scenario_id="token",
    )
    scene_path = tmp_path / "scene.npz"
    scene.save(scene_path)
    image = tmp_path / "pixels.png"
    from tests.unit.data.test_sensors import png_bytes

    image.write_bytes(png_bytes(32, 24))
    artifact = {"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    worlds = []
    for tick in range(61):
        state = np.zeros((len(scene.actor_ids), 13))
        state[:, 0] = tick
        state[:, 7:9] = 2
        worlds.append({"state": state.tolist(), "presence": [True] * len(state)})
    cal = {
        "camera_model": "opencv_distorted_source_lens",
        "camera_to_body": np.eye(4).tolist(),
        "K": np.eye(3).tolist(),
        "distortion": [0, 0, 0, 0, 0],
    }

    def capture(tick):
        return {
            "tick": tick,
            "timestamp_us": 1000000 + tick * 100000,
            "cameras": [
                {"camera_id": "CAM_R0", "calibration": cal, "image": artifact, "valid": artifact}
            ],
        }

    def export(manifest, output, **options):
        output.mkdir(parents=True)
        episode = output / "episode"
        episode.mkdir()
        (episode / "manifest.json").write_text(
            json.dumps(
                {
                    "dt_seconds": 0.1,
                    "actor_ids": list(scene.actor_ids),
                    "inputs": {"scene": {"sha256": file_sha256(scene_path)}},
                }
            )
        )
        (episode / "generation-000-physics.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"privileged": {"world_before": worlds[t], "world_after": worlds[t + 1]}}
                )
                for t in range(60)
            )
        )
        return {
            "input": {"frames": [capture(t) for t in (5, 10, 15, 20)]},
            "target": {
                "future_frames": [capture(t) for t in range(25, 61, 5)]
                if options["include_future_images"]
                else []
            },
            "source": {"manifest_sha256": "fixture"},
            "rejection_reasons": ["unqualified"],
        }

    monkeypatch.setattr("driversi.data.datasets.navsim_synthetic.export_four_frame_sample", export)
    return scene_path


def test_history_only_preserves_all_future_states(navsim_export_source, tmp_path):
    import pickle

    from driversi.data.datasets.navsim_synthetic import export_navsim_sample

    reports, frames = [], []
    for flag in (True, False):
        output = tmp_path / str(flag)
        result = export_navsim_sample(
            tmp_path / "render.json",
            navsim_export_source,
            output,
            anchor_tick=20,
            cameras=["CAM_R0"],
            include_future_images=flag,
        )
        reports.append(result)
        frames.append(pickle.loads((output / result["log_file"]).read_bytes()))
    assert reports[0]["image_counts"] == {"history": 4, "future": 8}
    assert reports[1]["image_counts"] == {"history": 4, "future": 0}
    assert reports[1]["num_future_frames"] == 8
    assert not (tmp_path / "False/sensor_blobs_fut").exists()
    assert len(frames[1]) == 12
    for a, b in zip(*frames):
        assert a["timestamp"] == b["timestamp"]
        for key in ("ego2global_translation", "ego_dynamic_state"):
            np.testing.assert_array_equal(a[key], b[key])
    assert all(f["cams"]["CAM_R0"]["data_path"] is None for f in frames[1][4:])
    assert all(f["cams"]["CAM_R0"]["data_path"] for f in frames[1][:4])
    assert not reports[1]["trainable"]


def test_image_option_must_be_explicit_boolean(tmp_path):
    from driversi.data.datasets.navsim_synthetic import export_navsim_sample

    with pytest.raises(ValueError, match="boolean"):
        export_navsim_sample("x", "y", tmp_path, anchor_tick=20, include_future_images=1)


def test_signal_lookup_never_uses_future_or_treats_missing_as_green():
    from driversi.data.datasets.navsim_synthetic import causal_signal_records

    timeline = [
        {"timestamp_us": 100, "records": [["12", True]], "available": True},
        {"timestamp_us": 200, "records": [["12", False]], "available": True},
    ]
    assert causal_signal_records(timeline, 99) == ([], False)
    assert causal_signal_records(timeline, 150) == ([["12", True]], True)
    assert causal_signal_records(timeline, 200) == ([["12", False]], True)
    assert causal_signal_records(None, 200) == ([], False)
    with pytest.raises(ValueError):
        causal_signal_records(timeline[::-1], 200)


def test_missing_signal_sample_invalidates_last_known_observation():
    from driversi.data.datasets.navsim_synthetic import causal_signal_records

    rows = [
        dict(timestamp_us=100, records=[["12", False]], available=True),
        dict(timestamp_us=200, records=[], available=False),
        dict(timestamp_us=300, records=[], available=True),
    ]
    assert causal_signal_records(rows, 250) == ([], False)
    assert causal_signal_records(rows, 300) == ([], True)
