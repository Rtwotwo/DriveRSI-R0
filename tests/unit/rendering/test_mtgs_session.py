import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_session import (
    MTGSWorldSession,
    apply_camera_delta,
    apply_rigid_delta,
    planar_delta,
)


def test_camera_rotation_is_about_vehicle_centre_not_map_origin():
    pose = np.column_stack((np.eye(3), [101.0, 200.0, 3.0]))
    moved = apply_camera_delta(pose, [100.0, 200.0, 2.0], [2.0, -4.0], np.pi / 2)
    np.testing.assert_allclose(moved[:, 3], [102.0, 197.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(moved[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_array_equal(pose[:, 3], [101.0, 200.0, 3.0])


def test_zero_motion_preserves_reconstruction_and_wraps_heading():
    delta, yaw = planar_delta([1.0, 2.0, 100.0, np.pi], [1.0, 2.0, -5.0, -np.pi])
    np.testing.assert_array_equal(delta, [0.0, 0.0])
    assert abs(yaw) < 1e-12  # Source/refined altitude difference is not injected.
    xyz, quat = apply_rigid_delta([3.0, 4.0, 5.0], [1.0, 0.0, 0.0, 0.0], delta, 0.0)
    np.testing.assert_array_equal(xyz, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(quat, [1.0, 0.0, 0.0, 0.0])


def test_rigid_yaw_left_composition_preserves_height_and_unit_quaternion():
    half = np.sqrt(0.5)
    xyz, q = apply_rigid_delta([2.0, 3.0, 4.0], [half, half, 0.0, 0.0], [1.0, -1.0], np.pi)
    np.testing.assert_allclose(xyz, [3.0, 2.0, 4.0])
    np.testing.assert_allclose(q, [0.0, 0.0, half, half], atol=1e-12)
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_nonfinite_pose_and_nonunit_quaternion_fail():
    with pytest.raises(ValueError):
        planar_delta([0, 0, 0, np.nan], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        apply_rigid_delta([0, 0, 0], [2, 0, 0, 0], [0, 0], 0.0)


def reference_session():
    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.metadata = {"source_start_timestamp_us": 1000000}
    session.times = np.array([0.0, 0.1, 0.2])
    session.reference = np.zeros((3, 1, 4))
    session.reference[:, 0, 0] = [0, 1, 2]
    session.reference[:, 0, 3] = np.radians([179, -179, -177])
    return session


def test_delta_uses_seed_time_reference_and_shortest_heading_arc():
    session = reference_session()
    reference = session._reference_at(1050000)[0]
    assert reference[0] == pytest.approx(0.5)
    assert abs(reference[3]) == pytest.approx(np.pi)
    # Source seed is 50 ms behind the 100 ms world. Even a log-following
    # world needs a 0.5 m seed-to-world correction, not the old zero delta.
    delta, _ = planar_delta(session.reference[1, 0], reference)
    np.testing.assert_allclose(delta, [0.5, 0])
    pose, _ = apply_rigid_delta([0.5, 0, 0], [1, 0, 0, 0], delta, 0)
    assert pose[0] == pytest.approx(1.0)


def test_reference_endpoint_jitter_clamp_is_bounded_to_one_millisecond():
    session = reference_session()
    assert session._reference_at(1200200)[0, 0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="bounded"):
        session._reference_at(1201001)


def test_camera_metadata_comparison_handles_array_fields_and_rejects_changes():
    from driversi.rendering.gaussian.mtgs_session import camera_metadata_equal

    original = {"frame_token": "a", "extrinsic": np.eye(4), "nested": {"values": [1, 2]}}
    duplicate = copy.deepcopy(original)
    assert camera_metadata_equal(original, duplicate)
    duplicate["extrinsic"][0, 3] = 1
    assert not camera_metadata_equal(original, duplicate)


def test_source_inventory_preserves_unbound_and_missing_pedestrians():
    session = reference_session()
    session.ids = ("ego", "car")
    session.node_tokens = {"car", "parked", "absent"}
    session._node_presence = {
        "car": np.array([True, True]),
        "parked": np.array([True, True]),
        "absent": np.array([True, False]),
    }
    source = {
        "token": "frame",
        "timestamp": 1000000,
        "track_tokens": ["car", "parked", "person"],
        "gt_names": ["vehicle", "vehicle", "pedestrian"],
    }
    inventory = session._source_inventory(source, 1)
    assert inventory["source_category_counts"] == {"vehicle": 2, "pedestrian": 1}
    assert inventory["source_without_reconstruction_node"] == [
        {"track_token": "person", "kind": "pedestrian"}
    ]
    assert len(inventory["source_without_physics_actor"]) == 2
    assert inventory["reconstruction_only_current_nodes"] == ["parked"]


def binding_session():
    session = reference_session()
    session.travel_id = 7
    session.metadata.update(
        source_log="log", map_name="map", scenario_id="a", source_frames=2, source_start_frame=3
    )
    frames, sources = [], {}
    for i, token in enumerate(("a", "b")):
        stamp = 1000000 + i * 200000
        camera = {"data_path": f"log/CAM_F0/{token}.jpg", "timestamp": stamp - 1000}
        source = {
            "token": token,
            "timestamp": stamp,
            "log_name": "log",
            "ego2global_original": np.eye(4),
            "ego2global": np.eye(4),
            "cams": {"CAM_F0": camera},
        }
        sources[(7, token)] = source
        frames.append(
            {
                "navsim_index": 3 + i,
                "token": token,
                "timestamp_us": stamp,
                "camera_image_path": camera["data_path"],
                "camera_token": token,
                "camera_timestamp_us": stamp - 1000,
                "original_pose_local": np.eye(4).tolist(),
                "refined_pose_local": np.eye(4).tolist(),
            }
        )
    session.binding = {
        "kind": "navsim_mtgs_exact_source_binding_audit",
        "match_count": 2,
        "video_token": "block-7",
        "camera_name": "CAM_F0",
        "frames": frames,
        "source_start_timestamp_us": 1000000,
        "source_end_timestamp_us": 1200000,
    }
    session.runtime = SimpleNamespace(source_frames=sources)
    return session


def test_binding_rechecks_each_frame_against_loaded_runtime():
    session = binding_session()
    session._validate_binding()
    session.runtime.source_frames[(7, "b")]["timestamp"] += 1
    with pytest.raises(ValueError, match="identity differs"):
        session._validate_binding()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.binding["frames"].pop(),
        lambda s: s.binding["frames"][1].update(navsim_index=6),
        lambda s: s.binding["frames"][1].update(token="unknown"),
        lambda s: setattr(s, "travel_id", 8),
        lambda s: s.binding["frames"][1].update(camera_timestamp_us=1200000),
        lambda s: s.binding["frames"][1]["refined_pose_local"][0].__setitem__(3, 1.0),
    ],
)
def test_binding_rejects_incomplete_wrong_traversal_or_changed_refinement(change):
    session = binding_session()
    change(session)
    with pytest.raises(ValueError):
        session._validate_binding()


def session_without_gpu(tmp_path, stride=1):
    """Exercise session commits using a deterministic, already-validated worker."""
    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.output = tmp_path
    session._source_images = {}
    session.generation, session.sensor_stride = 0, stride
    session.metadata = {"scenario_id": "fixture"}
    session.map = {}
    session.manifest = {
        "status": "running",
        "frames": [],
        "steps": [],
        "resets": [],
        "limits": [],
    }
    initial = {"tick": 0, "done": False, "controlled_ids": ["ego", "car"]}
    session.current = copy.deepcopy(initial)

    class Worker:
        calls = []

        def request(self, operation, **values):
            self.calls.append((operation, values))
            if operation == "reset":
                return copy.deepcopy(initial)
            return {
                **initial,
                "tick": session.current["tick"] + 1,
                "policy_actions": {"ego": 4, "car": 3},
                "executed_actions": {"ego": 4, "car": 1},
                "diagnostics": [{"actor": "car", "action": 3}],
                "valid": [True, True],
                "terminated": [False, False],
                "truncated": [False, False],
                "events": {"collision": [False, False]},
            }

    session.worker = Worker()

    def capture():
        frame = {
            "generation": session.generation,
            "tick": session.current["tick"],
            "cameras": [],
            "capture_seconds": 0.01,
            "transition": session._transition(),
        }
        session.manifest["frames"].append(frame)
        session.latest = frame
        session._publish()
        return copy.deepcopy(frame)

    session.capture = capture
    capture()
    return session


def test_render_failure_preserves_committed_step_and_requires_reset(tmp_path):
    session = session_without_gpu(tmp_path)
    capture = session.capture

    def fail_capture():
        persisted = json.loads((tmp_path / "transitions.jsonl").read_text().splitlines()[-1])
        assert persisted["executed_actions"]["car"] == 1
        assert persisted["capture_status"] == "pending"
        raise RuntimeError("GPU render failed")

    session.capture = fail_capture
    with pytest.raises(RuntimeError, match="GPU render failed"):
        session.advance()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["status"] == "failed"
    assert saved["steps"][0]["capture_status"] == "failed"
    assert session.current["tick"] == 1 and session.latest["tick"] == 0
    count = len(session.worker.calls)
    with pytest.raises(RuntimeError, match="Reset"):
        session.advance()
    assert len(session.worker.calls) == count
    session.capture = capture
    state = session.reset()
    assert state["world_tick"] == state["tick"] == 0
    assert state["generation"] == state["world_generation"] == 1
    assert state["transition"] is None and state["latest_transition"] is None
    assert state["status"] == "running" and state["failure"] is None
    assert saved["steps"][0]["events"] == session.manifest["steps"][0]["events"]


def test_reset_render_failure_keeps_failed_state_and_distinct_world_generation(tmp_path):
    session = session_without_gpu(tmp_path)
    session.advance()
    session.capture = lambda: (_ for _ in ()).throw(RuntimeError("reset image failed"))
    with pytest.raises(RuntimeError, match="reset image failed"):
        session.reset()
    state = session.state()
    assert state["status"] == "failed"
    assert state["world_tick"] == 0 and state["tick"] == 1
    assert state["world_generation"] == 1 and state["generation"] == 0
    assert session.manifest["resets"][-1]["capture_status"] == "failed"
    with pytest.raises(RuntimeError, match="Reset"):
        session.advance()


def test_nonrender_tick_does_not_attach_newer_action_to_previous_image(tmp_path):
    session = session_without_gpu(tmp_path, stride=2)
    state = session.advance()
    assert state["tick"] == 0 and state["world_tick"] == 1
    assert state["transition"] is None
    assert state["latest_transition"]["observation_tick"] == 0
    state = session.advance()
    assert state["tick"] == state["world_tick"] == 2
    assert state["transition"]["observation_tick"] == 1
    assert state["transition"]["successor_tick"] == 2
    assert state["transition"]["policy_actions"]["car"] == 3
    assert state["transition"]["executed_actions"]["car"] == 1
    state["transition"]["executed_actions"]["car"] = 7
    assert session.latest["transition"]["executed_actions"]["car"] == 1


def test_hot_manifest_preserves_transitions_and_scopes_partial_performance(tmp_path):
    session = session_without_gpu(tmp_path)
    session.advance()
    text = (tmp_path / "manifest.json").read_text()
    saved = json.loads(text)
    assert text.count("\n") == 1
    assert saved["steps"] == session.manifest["steps"]
    assert "real_time_factor" not in saved["performance"]
    assert "excludes_manifest" in saved["performance"]["scope"]


def test_capture_preserves_uncontrolled_checkpoint_nodes_without_normalizing_them(
    tmp_path, monkeypatch
):
    """A static checkpoint node may be non-unit; it must stay byte-untouched."""
    # This test stops before image I/O; core CI need not install Pillow.
    import sys

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=None))
    session = session_without_gpu(tmp_path)
    del session.capture  # Exercise real capture, stop only at the GPU boundary.
    session.metadata.update(source_start_timestamp_us=1000000, source_log="log")
    session.ids = ("ego", "parked", "car")
    session.controlled = np.array([True, False, True])
    session.node_tokens = {"parked", "car"}
    session.travel_id = 7
    session.times = np.array([0.0, 0.1])
    session.reference = np.zeros((2, 3, 4))
    state = np.zeros((3, 13))
    state[2, 0] = 1
    session.current["state"] = state.tolist()
    session.camera_names = ("CAM_F0",)
    session._actors = lambda: []
    session._source_inventory = lambda *_: {}

    class Tensor(np.ndarray):
        def new_tensor(self, value):
            return np.asarray(value).view(Tensor)

    camera = SimpleNamespace(
        metadata={"frame_token": "frame", "frame_idx": 0},
        camera_to_worlds=np.eye(4)[None, :3].view(Tensor),
    )
    session._sample = lambda *_: (1000000, 0, {}, camera, {})

    def untouched(*_):
        pytest.fail("An uncontrolled reconstruction node must not be edited or normalized")

    def render(_camera, *, rigid_poses):
        assert set(rigid_poses) == {"car"}
        np.testing.assert_array_equal(rigid_poses["car"][0], [3.0, 0.0, 0.0])
        raise RuntimeError("reached GPU boundary")

    session.runtime = SimpleNamespace(
        model=SimpleNamespace(
            gaussian_models={
                "rigid_object_parked": SimpleNamespace(get_object_pose=untouched),
                "rigid_object_car": SimpleNamespace(
                    get_object_pose=lambda *_: (
                        np.array([1.0, 0.0, 0.0, 0.0]),
                        np.array([2.0, 0.0, 0.0]),
                    )
                ),
            }
        ),
        source_frames={(7, "frame"): {"ego2global": np.eye(4)}},
        render_camera=render,
        validate_camera=lambda camera: (7, int(camera.metadata["frame_idx"])),
    )
    with pytest.raises(RuntimeError, match="reached GPU boundary"):
        session.capture()


def test_reset_records_complete_initial_world_and_owns_its_nested_buffers(tmp_path):
    session = session_without_gpu(tmp_path)
    request = session.worker.request

    def reset_with_extra(operation, **values):
        result = request(operation, **values)
        result.update(
            state=[[1, 2, 3]],
            motion={"extension": [7]},
            build_id="native-id",
            all_actor_ids=["ego", "car"],
            alive=[True, True],
            dt_seconds=0.1,
        )
        return result

    session.worker.request = reset_with_extra
    session.reset()
    saved = session.manifest["initial_world_by_generation"]["1"]
    assert saved == session.current
    session.current["motion"]["extension"][0] = 99
    assert saved["motion"]["extension"] == [7]
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["initial_world_by_generation"]["1"]
        == saved
    )


@pytest.mark.parametrize("height_mode", ["source-time", "reference-track"])
@pytest.mark.parametrize("write_workers", [1, 4])
@pytest.mark.parametrize("write_failure", [False, True])
def test_virtual_capture_has_own_pose_and_no_false_source_photo(
    tmp_path, monkeypatch, height_mode, write_workers, write_failure
):
    import sys

    class Image:
        @staticmethod
        def fromarray(_array):
            return SimpleNamespace(save=lambda path, **_kwargs: path.write_bytes(b"fixture"))

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=Image))
    monkeypatch.setattr(
        "driversi.rendering.gaussian.mtgs_runtime.resample_source_lens",
        lambda rgb, *_: (rgb, np.ones(rgb.shape[:2], dtype=bool)),
    )
    session = session_without_gpu(tmp_path)
    del session.capture
    session.metadata.update(source_start_timestamp_us=1000000, source_log="log")
    session.ids = ("ego", "car")
    session.controlled = np.array([True, True])
    session.dimensions = np.array([[5, 2, 2], [4, 2, 2]])
    session.node_tokens, session.travel_id = {"car"}, 7
    session.times, session.reference = np.array([0.0, 0.1]), np.zeros((2, 2, 4))
    session.current["state"] = np.zeros((2, 13)).tolist()
    session.current["state"][1][0] = 1.0
    session.height_mode = height_mode
    from driversi.rendering.gaussian.mtgs_height import ReferenceTrackHeight

    session._height_profiles = {
        actor: ReferenceTrackHeight(
            np.array([[-2.0, 0], [0.0, 0], [2.0, 0]]), np.array([0.8, 1.0, 1.2])
        )
        for actor in session.ids
    }
    session._height_profile_reasons = {}
    expected_height = 2.3 if height_mode == "reference-track" else 2.2
    session.camera_names, session.virtual_camera_actors = (), ("car",)
    session._actors = lambda: []
    session._source_inventory = lambda *_: {
        "source_without_reconstruction_node": [{"kind": "pedestrian"}]
    }
    session._source_images = {}
    camera = SimpleNamespace(
        metadata={"frame_token": "frame", "frame_idx": 0, "cam_token": "seed-camera"},
        camera_to_worlds=np.eye(4)[None, :3],
    )
    projection = {
        "map_x": None,
        "map_y": None,
        "camera_model": "source_lens",
        "K": np.eye(3).tolist(),
        "distortion": [0] * 5,
    }
    session._sample = lambda *_: (1000000, 0, {}, copy.deepcopy(camera), projection)

    def render(seed, *, rigid_poses, target_camera_to_world, exclude_tokens):
        assert exclude_tokens == ("car",) and set(rigid_poses) == {"car"}
        np.testing.assert_allclose(target_camera_to_world[:, 3], [4.4, 0, expected_height])
        np.testing.assert_array_equal(seed.camera_to_worlds, camera.camera_to_worlds)
        return {
            "rgb": np.zeros((2, 3, 3), np.uint8),
            "render_seconds": 0.1,
            "override_tokens": ["car"],
            "excluded_tokens": ["car"],
            "camera_to_world": target_camera_to_world.tolist(),
            "camera_to_world_semantics": "effective_after_world_motion_and_checkpoint_camera_optimizer",
            "camera_coordinate_frame": "mtgs_reconstruction_metres",
            "camera_pose_axes": "nerfstudio_right_up_back",
        }

    session.runtime = SimpleNamespace(
        model=SimpleNamespace(
            gaussian_models={
                "rigid_object_car": SimpleNamespace(
                    get_object_pose=lambda *_: (np.array([1.0, 0, 0, 0]), np.array([2.0, 0, 1.0]))
                )
            }
        ),
        source_frames={(7, "frame"): {"cams": {"CAM_F0": {"timestamp": 999000}}}},
        render_camera=render,
        validate_camera=lambda camera: (7, int(camera.metadata["frame_idx"])),
    )
    session.image_write_workers = write_workers
    if write_failure:
        def fail_save(*_args, **_kwargs):
            raise OSError("injected artifact write failure")

        monkeypatch.setattr("driversi.rendering.gaussian.mtgs_artifacts._save_png", fail_save)
        previous_frames = copy.deepcopy(session.manifest["frames"])
        previous_latest = copy.deepcopy(session.latest)
        with pytest.raises(OSError, match="injected artifact write failure"):
            session.capture()
        assert session.manifest["frames"] == previous_frames
        assert session.latest == previous_latest
        assert session._source_images == {}
        return
    frame = session.capture()
    assert frame["artifact_write"]["workers"] == write_workers
    assert frame["artifact_write"]["peak_pending_jobs"] <= 2 * write_workers
    assert all((tmp_path / frame["cameras"][0][key]).exists() for key in ("image", "pinhole", "valid"))
    view = frame["cameras"][0]
    assert view["camera_id"] == "car__virtual_front" and view["owner_actor_id"] == "car"
    assert view["virtual"] and not view["trainable"]
    assert view["excluded_tokens"] == ["car"]
    assert view["camera_timestamp_us"] == 1000000 and view["source_camera_timestamp_us"] == 999000
    assert view["source"] is None and view["support"]["ego_without_reconstruction_asset"]
    assert view["rig"]["body_reference"] == "reconstructed_3d_box_center"
    np.testing.assert_allclose(np.asarray(view["camera_to_world"])[:, 3], [4.4, 0, expected_height])
    assert view["camera_coordinate_frame"] == "mtgs_reconstruction_metres"
    assert session.state()["cameras"][0]["source_url"] is None
    assert session._source_images == {}
    assert view["height_binding"]["supported"] == (height_mode == "reference-track")
    assert view["height_binding"]["correction_m"] == pytest.approx(expected_height - 2.2)
    assert view["scene_height_supported"] == (height_mode == "reference-track")
    assert not frame["render_quality"]["high_quality_supervision_eligible"]
    assert view["quality"]["checks"]["reconstruction_quality"] == "unknown"


def test_ego_camera_and_vehicle_nodes_receive_separate_position_height_queries(
    tmp_path, monkeypatch
):
    import sys

    from driversi.rendering.gaussian.mtgs_height import ReferenceTrackHeight

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=None))
    session = session_without_gpu(tmp_path)
    del session.capture
    session.metadata.update(source_start_timestamp_us=1000000, source_log="log")
    session.ids, session.controlled = ("ego", "car"), np.array([True, True])
    session.node_tokens, session.travel_id = {"car"}, 7
    session.times, session.reference = np.array([0.0, 0.1]), np.zeros((2, 2, 4))
    state = np.zeros((2, 13))
    state[:, 0] = [2.0, 1.0]
    session.current["state"] = state.tolist()
    session.camera_names = ("CAM_F0",)
    session.height_mode = "reference-track"
    session._height_profiles = {
        actor: ReferenceTrackHeight(np.array([[0.0, 0], [4.0, 0]]), np.array([0.0, 0.4]))
        for actor in session.ids
    }
    session._height_profile_reasons = {}
    session._actors = lambda: []
    session._source_inventory = lambda *_: {}

    class Tensor(np.ndarray):
        def new_tensor(self, value):
            return np.asarray(value).view(Tensor)

    pose = np.eye(4)[None, :3].copy()
    pose[0, 2, 3] = 2.0
    camera = SimpleNamespace(
        metadata={"frame_token": "frame", "frame_idx": 0}, camera_to_worlds=pose.view(Tensor)
    )
    session._sample = lambda *_: (1000000, 0, {}, camera, {})

    def render(moved_camera, *, rigid_poses):
        np.testing.assert_allclose(moved_camera.camera_to_worlds[0, :, 3], [2.0, 0, 2.2])
        np.testing.assert_allclose(rigid_poses["car"][0], [1.0, 0, 1.1])
        np.testing.assert_array_equal(session.current["state"], state)
        raise RuntimeError("height checked at GPU boundary")

    session.runtime = SimpleNamespace(
        model=SimpleNamespace(
            gaussian_models={
                "rigid_object_car": SimpleNamespace(
                    get_object_pose=lambda *_: (np.array([1.0, 0, 0, 0]), np.array([0.0, 0, 1.0]))
                )
            }
        ),
        source_frames={(7, "frame"): {"ego2global": np.eye(4)}},
        render_camera=render,
        validate_camera=lambda camera: (7, int(camera.metadata["frame_idx"])),
    )
    with pytest.raises(RuntimeError, match="height checked"):
        session.capture()


@pytest.mark.parametrize("mode", ["source-time", "reference-track"])
@pytest.mark.parametrize("ego_shift", [0.0, 2.0])
def test_uncontrolled_ego_camera_keeps_source_height_without_claiming_support(
    tmp_path, monkeypatch, mode, ego_shift
):
    import sys

    from driversi.rendering.gaussian.mtgs_height import ReferenceTrackHeight

    class Image:
        @staticmethod
        def fromarray(_array):
            return SimpleNamespace(save=lambda path, **_kwargs: path.write_bytes(b"fixture"))

    class Tensor(np.ndarray):
        def new_tensor(self, value):
            return np.asarray(value).view(Tensor)

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=Image))
    monkeypatch.setattr(
        "driversi.rendering.gaussian.mtgs_runtime.resample_source_lens",
        lambda rgb, *_: (rgb, np.ones(rgb.shape[:2], dtype=bool)),
    )
    session = session_without_gpu(tmp_path)
    del session.capture
    session.metadata.update(source_start_timestamp_us=1000000, source_log="log")
    session.ids, session.controlled = ("ego", "car", "car2"), np.array([False, True, True])
    session.node_tokens, session.travel_id = {"car", "car2"}, 7
    session.times, session.reference = np.array([0.0, 0.1]), np.zeros((2, 3, 4))
    session.current["state"] = np.zeros((3, 13)).tolist()
    session.current["state"][0][0] = ego_shift
    session.current["controlled_ids"] = ["car", "car2"]
    session.camera_names, session.height_mode = ("CAM_F0",), mode
    session._height_profiles = {
        actor: ReferenceTrackHeight(np.array([[0.0, 0], [4.0, 0]]), np.array([0.0, 0.4]))
        for actor in ("car", "car2")
    }
    session._height_profile_reasons = {}
    session._actors, session._source_inventory = lambda: [], lambda *_: {}
    session._source_images = {0: tmp_path / "source.png"}
    pose = np.eye(4)[None, :3].copy()
    pose[0, 2, 3] = 2.0
    camera = SimpleNamespace(
        metadata={"frame_token": "frame", "frame_idx": 0}, camera_to_worlds=pose.view(Tensor)
    )
    projection = {
        "map_x": None,
        "map_y": None,
        "camera_model": "source_lens",
        "K": np.eye(3).tolist(),
        "distortion": [0] * 5,
    }
    session._sample = lambda *_: (1000000, 0, {}, camera, projection)

    def render(moved_camera, *, rigid_poses):
        expected_pose = pose.copy()
        expected_pose[0, 0, 3] += ego_shift
        np.testing.assert_array_equal(moved_camera.camera_to_worlds, expected_pose)
        assert set(rigid_poses) == {"car", "car2"}
        return {
            "rgb": np.zeros((2, 3, 3), np.uint8),
            "render_seconds": 0.1,
            "override_tokens": list(rigid_poses),
            "camera_to_world": expected_pose[0].tolist(),
            "camera_to_world_semantics": "effective",
            "camera_coordinate_frame": "mtgs",
            "camera_pose_axes": "nerfstudio_right_up_back",
        }

    node = SimpleNamespace(get_object_pose=lambda *_: (np.array([1.0, 0, 0, 0]), np.zeros(3)))
    session.runtime = SimpleNamespace(
        model=SimpleNamespace(
            gaussian_models={"rigid_object_car": node, "rigid_object_car2": node}
        ),
        source_frames={
            (7, "frame"): {"ego2global": np.eye(4), "cams": {"CAM_F0": {"timestamp": 1000000}}}
        },
        render_camera=render,
        validate_camera=lambda camera: (7, int(camera.metadata["frame_idx"])),
    )
    frame = session.capture()
    view = frame["cameras"][0]
    assert view["height_binding"]["reason"] == "uncontrolled_source_camera"
    assert view["height_binding"]["correction_m"] == 0.0
    assert not view["height_binding"]["supported"]
    assert view["scene_height_supported"] == (mode == "reference-track")
    assert view["support"]["distance_from_seed_camera_m"] == ego_shift
    assert view["support"]["pose_comparison"] == "pre_optimizer_moved_vs_source_seed"
    assert "virtual_pose_not_real_sensor_calibration" not in view["support"]["limitations"]
    assert view["quality"]["checks"]["novel_view_domain"] == "unknown"
    assert {a["actor_id"] for a in frame["height_binding"]["per_camera"][0]["actors"]} == {
        "car",
        "car2",
    }


def test_height_profiles_use_all_matching_frames_independent_of_camera_sampling_order():
    session = reference_session()
    session.ids, session.controlled, session.travel_id = ("ego",), np.array([True]), 7
    session.metadata["source_log"] = "log"
    session.binding = {"source_start_timestamp_us": 1000000, "source_end_timestamp_us": 1200000}
    tokens = ["start", "end", "start", "middle", "end"]
    frames = {}
    for token, timestamp, height in (
        ("start", 1000000, 0.0),
        ("middle", 1100000, 0.02),
        ("end", 1200000, 0.0),
    ):
        pose = np.eye(4)
        pose[2, 3] = height
        frames[7, token] = {"timestamp": timestamp, "log_name": "log", "ego2global": pose}
    session.runtime = SimpleNamespace(
        sources=SimpleNamespace(frame_tokens=tokens, travel_ids=[7] * 5, frame_ids=[0, 2, 0, 1, 2]),
        source_frames=frames,
    )
    results = []
    for indices in (
        {"front": [(1000000, 0), (1200000, 1)], "rear": [(1000000, 2), (1100000, 3), (1200000, 4)]},
        {"rear": [(1000000, 2), (1100000, 3), (1200000, 4)], "front": [(1000000, 0), (1200000, 1)]},
        {"front": [(1000000, 0), (1200000, 1)]},
    ):
        session._indices = indices
        session._height_profiles, session._height_profile_reasons = {}, {}
        session._build_height_profiles()
        results.append(session._height_profiles["ego"].correction([1.0, 0], [0.0, 0]))
    assert results[0] == results[1] == results[2]
    assert results[0]["supported"] and results[0]["correction_m"] == pytest.approx(0.02)
    session.runtime.sources.frame_ids[-1] = 99
    with pytest.raises(ValueError, match="Ambiguous"):
        session._build_height_profiles()


def test_external_actions_need_explicit_current_tick_before_any_mutation(tmp_path):
    session = session_without_gpu(tmp_path)
    session.session_id = "external-fixture"
    session.current["behavior_policies"] = {
        "ego": {"controller": "route-idm-v2", "policy_id": "rule", "version": "v1"},
        "car": {"controller": "external-joint-v1", "policy_id": "teacher", "version": "v1"},
    }
    before = copy.deepcopy(session.current)
    for actions, tick in (
        (None, None),
        ({"car": 7}, None),
        ({"car": 7}, 1),
        ({}, 0),
        ({"car": True}, 0),
        ({"car": 7, "ego": 7}, 0),
    ):
        with pytest.raises(ValueError):
            session.advance(
                external_actions=actions,
                expected_tick=tick,
                decision_identity=session.decision_identity(),
            )
        assert session.current == before and session.manifest["status"] == "running"
        assert not session.manifest["steps"]
    original = session.worker.request
    captured = []

    def request(operation, **payload):
        captured.append(payload)
        return original(operation, **payload)

    session.worker.request = request
    session.advance(
        external_actions={"car": 7}, expected_tick=0, decision_identity=session.decision_identity()
    )
    assert captured[0]["external_actions"] == {"car": 7} and session.current["tick"] == 1


def test_worker_population_is_copied_to_session_manifest_for_export(tmp_path):
    from driversi.contracts.population import EpisodePopulation

    session = session_without_gpu(tmp_path)
    population = EpisodePopulation.from_dict(
        {
            "schema_version": 1,
            "policies": [
                {
                    "policy_id": "baseline",
                    "version": "v1",
                    "controller": "route-idm-v2",
                    "parameters": {},
                }
            ],
            "assignments": [
                {"actor_id": actor, "policy_id": "baseline", "role": "participant"}
                for actor in ("ego", "car")
            ],
        }
    )
    ready = {
        "population_configuration": population.to_dict(),
        "behavior_policies": population.behavior_policies,
        "controlled_ids": ["ego", "car"],
    }
    session._record_population(ready)
    assert session.manifest["protocol"]["rgb_is_policy_input"] is False
    assert session.manifest["population_configuration"] == population.to_dict()
    assert session.manifest["behavior_policies"] == population.behavior_policies
    ready["behavior_policies"]["ego"]["version"] = "changed"
    assert session.manifest["behavior_policies"]["ego"]["version"] == "v1"
    with pytest.raises(ValueError, match="identity differ"):
        session._record_population(ready)


@pytest.mark.parametrize(
    "invalid_identity", ["missing", "previous_reset", "foreign_session", "foreign_policy"]
)
def test_unmanaged_external_actions_reject_missing_or_stale_identity_before_worker(
    tmp_path, invalid_identity
):
    session = session_without_gpu(tmp_path)
    session.session_id = "external-fixture"
    session.current["behavior_policies"] = {
        "car": {"controller": "external-joint-v1", "policy_id": "teacher", "version": "v1"}
    }
    identity = session.decision_identity()
    if invalid_identity == "missing":
        identity = None
    elif invalid_identity == "previous_reset":
        session.generation += 1
    elif invalid_identity == "foreign_session":
        identity["session_id"] = "another-session"
    else:
        identity["policy_versions"]["car"]["version"] = "v0"
    before = copy.deepcopy(session.current)
    calls = len(session.worker.calls)
    with pytest.raises(ValueError, match="identity"):
        session.advance(external_actions={"car": 7}, expected_tick=0, decision_identity=identity)
    assert session.current == before
    assert len(session.worker.calls) == calls and not session.manifest["steps"]
    assert session.manifest["status"] == "running"


def test_seed_lifetime_boundaries_never_interpolate_absent_origin():
    session = reference_session()
    session.reference[:] = 0
    session.reference[1, 0, 0] = 100
    session.lifecycle_presence = np.array([[False], [True], [False]])
    before, valid_before = session._reference_sample(1090000)
    assert not valid_before[0]
    assert before[0, 0] == 0
    at, valid_at = session._reference_sample(1100000)
    assert valid_at[0] and at[0, 0] == 100
    held, valid_held = session._reference_sample(1190000)
    assert valid_held[0] and held[0, 0] == 100
    absent, valid_absent = session._reference_sample(1200000)
    assert not valid_absent[0] and absent[0, 0] == 0


def test_retained_asset_anchor_requires_past_same_actor_supported_reference():
    from types import SimpleNamespace

    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.ids = ("a", "b")
    session.kinds = [1, 1]
    session.controlled = [True, True]
    session.metadata = {}
    session._indices = {"CAM_F0": [(100, 0), (200, 1), (300, 2)]}
    session.runtime = SimpleNamespace(sources=SimpleNamespace(frame_ids=[0, 1, 2]))
    session._node_presence = {"a": [True, False, True], "b": [False, True, True]}
    session._reference_sample = lambda stamp: (
        np.full((2, 4), stamp),
        np.array([True, stamp != 200]),
    )
    stamp, frame, ref = session._earlier_asset_anchor("a", 250, 1)
    assert (stamp, frame) == (100, 0) and ref[0, 0] == 100
    assert session._earlier_asset_anchor("a", 99, 2) is None
    assert session._earlier_asset_anchor("b", 250, 1) is None
    assert session._earlier_asset_anchor("b", 300, 2)[1] == 2


def test_retained_asset_never_creates_nonvehicle_or_unowned_replay_node():
    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.ids = ("pedestrian", "replay")
    session.kinds = [2, 1]
    session.controlled = [True, False]
    session.metadata = {}
    assert session._earlier_asset_anchor("pedestrian", 200, 1) is None
    assert session._earlier_asset_anchor("replay", 200, 1) is None


def test_height_profile_excludes_compile_bracketing_frame_outside_world_horizon():
    session = reference_session()
    session.ids, session.controlled, session.travel_id = ("ego",), np.array([True]), 7
    session.metadata["source_log"] = "log"
    session.binding = {"source_start_timestamp_us": 1000000, "source_end_timestamp_us": 1500000}
    frames = {}
    for token, stamp, height in [
        ("start", 1000000, 0.0),
        ("middle", 1100000, 0.02),
        ("end", 1200000, 0.0),
        ("bracket", 1500000, 100.0),
    ]:
        pose = np.eye(4)
        pose[2, 3] = height
        frames[7, token] = {"timestamp": stamp, "log_name": "log", "ego2global": pose}
    session.runtime = SimpleNamespace(
        sources=SimpleNamespace(
            frame_tokens=["start", "middle", "end", "bracket"],
            travel_ids=[7] * 4,
            frame_ids=[0, 1, 2, 3],
        ),
        source_frames=frames,
    )
    session._height_profiles, session._height_profile_reasons = {}, {}
    session._build_height_profiles()
    assert session._height_profiles["ego"].correction([1.0, 0], [0.0, 0])[
        "correction_m"
    ] == pytest.approx(0.02)
    assert session._reference_seed_supported(1200500)
    assert not session._reference_seed_supported(1202000)
    assert not session._reference_seed_supported(1500000)
    with pytest.raises(ValueError, match="bounded reference"):
        session._reference_at(1500000)


def test_explicit_capture_window_does_not_require_future_ttc_rgb():
    session = binding_session()
    session.metadata["source_frames"] = 3
    session.times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="complete declared"):
        session._validate_binding()
    session.binding["render_end_tick"] = 2
    session.capture_end_tick = 2
    session._validate_binding()
    session.capture_end_tick = 3
    with pytest.raises(ValueError, match="capture horizon"):
        session._validate_binding()
