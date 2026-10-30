from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_bindings import (
    compiled_observed_pose,
    exact_float32_identity,
    validate_binding,
)
from driversi.rendering.gaussian.mtgs_training_window import NativeTrainingRenderer


def fixture():
    scene = SimpleNamespace(
        actor_ids=("car",),
        metadata={"source_start_timestamp_us": 1000000},
        times=np.arange(11) * 0.1,
        poses=np.zeros((11, 1, 4)),
        presence_at=lambda tick: np.asarray([tick != 5]),
    )
    source = {"token": "first", "timestamp": 1000000, "track_tokens": ["car"]}
    state = np.asarray([1.0, 0, 0, 0])
    binding = dict(
        actor_id="car",
        source_token="first",
        source_timestamp_us=1000000,
        native_frame_index=0,
        target_timestamp_us=2000000,
        target_tick=10,
        reference_tick=0,
        reference_contract="sealed_scene_planar_observation_v1",
        reference_physical_pose=[0.0, 0, 0, 0],
        target_physical_pose=state.tolist(),
        planar_delta=[1.0, 0, 0],
        policy="latest_observed_asset_v1",
        checkpoint_presence_unchanged=True,
    )
    return scene, source, state, binding


def test_compiled_anchor_rejects_absent_source_and_future_identity():
    scene, source, state, binding = fixture()
    validate_binding(binding, source, "car", state, 10, 2000000, scene)
    for patch in (
        {"target_tick": 9},
        {"source_timestamp_us": 3000000},
        {"reference_tick": 5},
        {"reference_physical_pose": [0.1, 0, 0, 0]},
        {"planar_delta": [0, 0, 0]},
        {"native_frame_index": False},
    ):
        bad = {**binding, **patch}
        with pytest.raises(ValueError):
            validate_binding(bad, source, "car", state, 10, 2000000, scene)
    source["timestamp"] = 1500000
    with pytest.raises(ValueError, match="actual compiled"):
        compiled_observed_pose(source, "car", scene)


def test_identity_is_exact_float32_roundtrip_not_free_movement_tolerance():
    source = np.array([35.0, 1.0, 0.0, 0.5], dtype=np.float32).astype(float)
    assert exact_float32_identity(source, source)["equal"]
    shifted = source.copy()
    shifted[0] = float(np.nextafter(np.float32(35), np.float32(36)))
    assert not exact_float32_identity(source, shifted)["equal"]
    with pytest.raises(ValueError):
        exact_float32_identity(source, [np.nan] * 4)


def renderer(static=False):
    obj = NativeTrainingRenderer.__new__(NativeTrainingRenderer)
    frames = [
        {"token": str(i), "timestamp": 1000000 + i * 500000, "track_tokens": ["car"]}
        for i in range(3)
    ]
    obj.frames = {x["token"]: x for x in frames}
    obj.by_timestamp = {x["timestamp"]: x for x in frames}
    obj.indices = {(str(i), "CAM_F0"): i for i in range(3)}
    obj.runtime = SimpleNamespace(sources=SimpleNamespace(frame_ids=[0, 1, 2]))
    node = SimpleNamespace(num_frames=3, in_frame_mask=[True, False, True])
    model = SimpleNamespace(node=node, static_in_log=static)
    obj.models = {"rigid_object_car": model}
    obj._physics = None
    obj._static_anchors = {}
    scene = SimpleNamespace(
        actor_ids=("car",),
        metadata={"source_start_timestamp_us": 1000000},
        times=np.arange(11) * 0.1,
        presence_at=lambda tick: np.asarray([True]),
    )
    obj.bind_physics(scene, np.zeros((11, 1, 7)), np.ones((11, 1), bool))
    return obj, model


def test_retained_asset_uses_past_observation_not_future_presence():
    obj, model = renderer()
    assert obj.asset_anchor(model, 1500000)[2] == 0
    assert obj.asset_anchor(model, 2000000)[2] == 2
    obj._physics = None
    assert obj.asset_anchor(model, 1500000) is None  # canonical source unchanged


def test_static_anchor_does_not_follow_source_annotation_drift():
    obj, model = renderer(static=True)
    assert obj.asset_anchor(model, 1000000)[2] == 0
    assert obj.asset_anchor(model, 2000000)[2] == 0


def test_source_obb_preserves_full_observed_pitch_without_moving_planar_physics():
    from driversi.rendering.gaussian.mtgs_bindings import observed_3d_box_corners

    angle = 0.05
    pose = np.eye(4)
    pose[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    source = {
        "token": "frame",
        "timestamp": 123,
        "track_tokens": ["car"],
        "gt_boxes": [[40, 0, 1, 4, 2, 2, 0]],
        "ego2global": pose,
    }
    before = pose.copy()
    corners, evidence = observed_3d_box_corners(source, "car", [4, 2, 2])
    expected = (pose @ [40, 0, 1, 1])[:3]
    assert np.allclose(corners.mean(0), expected)
    assert abs(expected[2] - 1) > 1.9
    assert evidence["contract"] == "same_actor_observed_3d_obb_v1"
    assert np.array_equal(pose, before)
    assert np.allclose(np.linalg.norm(corners[4] - corners[0]), 4)
    assert np.allclose(np.linalg.norm(corners[2] - corners[0]), 2)


def test_source_obb_rejects_missing_actor_and_nonrigid_pose():
    from driversi.rendering.gaussian.mtgs_bindings import observed_3d_box_corners

    source = {
        "token": "frame",
        "timestamp": 123,
        "track_tokens": ["car"],
        "gt_boxes": [[40, 0, 1, 4, 2, 2, 0]],
        "ego2global": np.eye(4),
    }
    with pytest.raises(ValueError):
        observed_3d_box_corners(source, "other", [4, 2, 2])
    source["ego2global"][0, 0] = 2
    with pytest.raises(ValueError):
        observed_3d_box_corners(source, "car", [4, 2, 2])


def test_native_centered_reference_matches_representation_without_motion_tolerance():
    from driversi.data.scenes.pufferdrive import native_initial_center, native_observed_pose

    scene, source, state, binding = fixture()
    scene.poses[0, 0] = [-3.0914228805340827, -43.882993013598025, 0.5, 1.6]
    scene.road_points = np.array([[64.8, -36.5, 0], [65.1, -35.9, 0]])
    expected = native_observed_pose(scene, scene.poses[0, 0])
    ref, tick = compiled_observed_pose(source, "car", scene, native_centering=True)
    assert np.array_equal(ref, expected)
    before = native_initial_center(scene)
    scene.poses[1:] += 100  # future cannot change the native origin
    assert np.array_equal(native_initial_center(scene), before)
    binding.update(
        reference_contract="native_centered_scene_observation_v2",
        reference_physical_pose=ref.tolist(),
        target_physical_pose=ref.tolist(),
        planar_delta=[0.0, 0.0, 0.0],
    )
    validate_binding(binding, source, "car", ref, 10, 2000000, scene)
    moved = ref.astype(np.float32)
    moved[0] = np.nextafter(moved[0], np.float32(np.inf))
    assert not exact_float32_identity(ref, moved)["equal"]
    bad = dict(binding, reference_physical_pose=moved.tolist())
    with pytest.raises(ValueError, match="source observation|physical state"):
        validate_binding(bad, source, "car", ref, 10, 2000000, scene)
