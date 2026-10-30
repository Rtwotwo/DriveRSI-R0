"""Persistent renderer transactions and source provenance without optional GPU deps."""

import threading
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime, temporary_rigid_poses


class TensorView:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def __getitem__(self, index):
        return TensorView(self.array[index])

    def detach(self):
        return self

    def clone(self):
        return TensorView(self.array.copy())

    def new_tensor(self, value):
        return TensorView(value)

    def copy_(self, value):
        self.array[:] = value.array


def node(static=False, travel=4):
    return SimpleNamespace(
        instance_trans=TensorView([1, 2, 3] if static else [[1, 2, 3], [4, 5, 6]]),
        instance_quats=TensorView([1, 0, 0, 0] if static else [[1, 0, 0, 0]] * 2),
        is_static=static,
        travel_id=travel,
        num_frames=2,
        in_frame_mask=[True, True],
        cached_gaussians={"old": True},
        last_step=30000,
        last_frame_idx=0,
    )


def runtime():
    instance = MTGSRuntime.__new__(MTGSRuntime)
    instance._operation = threading.Lock()
    instance._pinhole_cache = OrderedDict()
    instance._lens_cache = OrderedDict()
    instance.sources = SimpleNamespace(
        image_filenames=["source.png"],
        cam_tokens=["camera"],
        frame_tokens=["frame"],
        travel_ids=[4],
        frame_ids=[1],
        cameras=SimpleNamespace(times=np.array([0.1])),
    )
    instance.torch = SimpleNamespace(
        no_grad=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None)
    )
    instance.model = SimpleNamespace(
        camera_optimizer=SimpleNamespace(
            apply_to_camera=lambda value: value.camera_to_worlds.copy()
        ),
        gaussian_models={
            "rigid_object_car": node(),
            "rigid_object_parked": node(True),
        },
    )
    camera = SimpleNamespace(
        metadata={
            "cam_idx": 0,
            "cam_token": "camera",
            "frame_token": "frame",
            "travel_id": 4,
            "frame_idx": 1,
        },
        times=np.array([0.1]),
        camera_to_worlds=np.eye(4)[None, :3, :],
        width=np.array([3]),
        height=np.array([2]),
        get_intrinsics_matrices=lambda: np.eye(3)[None],
    )
    return instance, camera


def test_joint_pose_changes_restore_translation_orientation_and_all_caches_on_error():
    moving, parked = node(), node(True)
    model = SimpleNamespace(gaussian_models={"rigid_object_a": moving, "rigid_object_b": parked})
    identities = moving.instance_trans, moving.instance_quats
    with pytest.raises(RuntimeError, match="GPU failure"):
        with temporary_rigid_poses(model, 4, 1, {"a": ([9, 8, 7], [0, 0, 0, 1])}):
            np.testing.assert_array_equal(moving.instance_trans.array[1], [9, 8, 7])
            np.testing.assert_array_equal(moving.instance_quats.array[1], [0, 0, 0, 1])
            assert parked.cached_gaussians is None
            moving.cached_gaussians = parked.cached_gaussians = {"edited": True}
            raise RuntimeError("GPU failure")
    assert (moving.instance_trans, moving.instance_quats) == identities
    np.testing.assert_array_equal(moving.instance_trans.array, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(moving.instance_quats.array, [[1, 0, 0, 0]] * 2)
    assert moving.cached_gaussians is parked.cached_gaussians is None


def test_bad_later_binding_does_not_partially_mutate_earlier_binding():
    a, wrong = node(), node(travel=9)
    model = SimpleNamespace(gaussian_models={"rigid_object_a": a, "rigid_object_b": wrong})
    with pytest.raises(ValueError, match="another traversal"):
        with temporary_rigid_poses(
            model,
            4,
            1,
            {
                "a": ([9, 8, 7], [1, 0, 0, 0]),
                "b": ([6, 5, 4], [1, 0, 0, 0]),
            },
        ):
            pytest.fail("Mismatched traversal reached renderer")
    np.testing.assert_array_equal(a.instance_trans.array[1], [4, 5, 6])
    assert a.cached_gaussians == {"old": True}


@pytest.mark.parametrize(
    "token,pose,message",
    [
        ("missing", ([0, 0, 0], [1, 0, 0, 0]), "No reconstructed"),
        ("a", ([np.nan, 0, 0], [1, 0, 0, 0]), "finite"),
        ("a", ([0, 0, 0], [2, 0, 0, 0]), "unit norm"),
    ],
)
def test_invalid_pose_or_binding_is_rejected(token, pose, message):
    model = SimpleNamespace(gaussian_models={"rigid_object_a": node()})
    with pytest.raises(ValueError, match=message):
        with temporary_rigid_poses(model, 4, 1, {token: pose}):
            pytest.fail("Invalid override accepted")


def test_static_pose_and_absent_dynamic_lifetime_are_distinct():
    parked = node(True)
    model = SimpleNamespace(gaussian_models={"rigid_object_a": parked})
    with temporary_rigid_poses(model, 4, 1, {"a": ([0, 1, 2], [0, 0, 0, 1])}):
        np.testing.assert_array_equal(parked.instance_trans.array, [0, 1, 2])
    np.testing.assert_array_equal(parked.instance_trans.array, [1, 2, 3])
    parked.in_frame_mask[1] = False
    with pytest.raises(ValueError, match="absent"):
        with temporary_rigid_poses(model, 4, 1, {"a": ([0, 1, 2], [1, 0, 0, 0])}):
            pytest.fail("Absent node spawned")


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_idx", 0),
        ("frame_token", "another"),
        ("travel_id", 7),
        ("driversi_camera_correction_applied", True),
    ],
)
def test_camera_rejects_mixed_provenance_or_double_correction(field, value):
    instance, camera = runtime()
    camera.metadata[field] = value
    with pytest.raises(ValueError):
        instance.validate_camera(camera)


def test_camera_allows_world_pose_delta_but_not_time_drift_or_reflection():
    instance, camera = runtime()
    camera.camera_to_worlds[0, :3, 3] = [100, 200, 3]
    assert instance.validate_camera(camera) == (4, 1)
    camera.times[0] = 0.2
    with pytest.raises(ValueError, match="exposure time"):
        instance.validate_camera(camera)
    camera.times[0] = 0.1
    camera.camera_to_worlds[0, 0, 0] = -1
    with pytest.raises(ValueError, match="proper rotation"):
        instance.validate_camera(camera)


def test_camera_invalid_intrinsics_fail_before_gpu_rasterization():
    instance, camera = runtime()
    camera.get_intrinsics_matrices = lambda: np.full((1, 3, 3), np.nan)
    with pytest.raises(ValueError, match="calibration"):
        instance.render_camera(camera)


def test_render_uses_frozen_pose_then_restores_even_when_renderer_mutates_camera():
    instance, camera = runtime()

    def render(copy):
        assert copy is not camera
        copy.metadata["frame_idx"] = 999
        np.testing.assert_array_equal(
            instance.model.gaussian_models["rigid_object_car"].instance_trans.array[1],
            [9, 8, 7],
        )
        return {"rgb_appearance": np.full((2, 3, 3), 0.5)}

    instance.model.get_outputs_for_camera = render
    result = instance.render_camera(camera, rigid_poses={"car": ([9, 8, 7], [1, 0, 0, 0])})
    assert result["rgb"].dtype == np.uint8 and result["rgb"].min() == 128
    assert result["override_tokens"] == ["car"] and result["render_seconds"] >= 0
    assert camera.metadata["frame_idx"] == 1
    np.testing.assert_array_equal(
        instance.model.gaussian_models["rigid_object_car"].instance_trans.array[1],
        [4, 5, 6],
    )


def test_concurrent_render_is_rejected_without_touching_model():
    instance, camera = runtime()
    with instance._exclusive():
        with pytest.raises(RuntimeError, match="concurrent"):
            instance.render_camera(camera)


def test_runtime_trust_and_scale_validation_happen_before_dependency_import(tmp_path):
    args = dict(config_path=tmp_path, checkpoint_path=tmp_path, road_block_config=tmp_path)
    with pytest.raises(ValueError, match="trusted"):
        MTGSRuntime(**args)
    with pytest.raises(ValueError, match="scale"):
        MTGSRuntime(**args, trusted_assets=True, scale=2)


class SeedCamera:
    """Only the public camera operations required by the camera-only path."""

    def __init__(self):
        self.fx, self.fy = np.array([100.0]), np.array([120.0])
        self.cx, self.cy = np.array([4.0]), np.array([3.0])
        self.width, self.height = np.array([8]), np.array([6])
        self.times = np.array([0.1])
        self.camera_to_worlds = np.eye(4)[None, :3, :]
        self.metadata = {}

    def to(self, device):
        assert device == "test"
        return self

    def rescale_output_resolution(self, scaling_factor):
        for attribute in ("fx", "fy", "cx", "cy"):
            getattr(self, attribute)[:] *= scaling_factor
        self.width[:] = (self.width * scaling_factor).astype(int)
        self.height[:] = (self.height * scaling_factor).astype(int)

    def get_intrinsics_matrices(self):
        return np.array([[[self.fx[0], 0, self.cx[0]], [0, self.fy[0], self.cy[0]], [0, 0, 1]]])


class SeedCameras:
    def __init__(self, camera):
        self.camera = camera
        self.times = camera.times

    def __getitem__(self, _index):
        return self.camera


def camera_only_runtime():
    instance, _ = runtime()
    seed = SeedCamera()
    cameras = SeedCameras(seed)
    instance.sources.cameras = cameras
    instance.sources.undistort_params = [(seed.get_intrinsics_matrices()[0], np.zeros(5))]
    instance.device, instance.scale = "test", 0.5

    def push(camera, sources, index):
        camera.metadata.update(
            cam_idx=index,
            cam_token=sources.cam_tokens[index],
            frame_token=sources.frame_tokens[index],
            travel_id=sources.travel_ids[index],
            frame_idx=sources.frame_ids[index],
        )
        return camera

    def forbid(*_args, **_kwargs):
        pytest.fail("Camera-only path accessed source images or annotations")

    instance.dataset = SimpleNamespace(
        cameras=cameras,
        crop_size=None,
        undistort_images="optimal",
        push_cam_details_to_metadata=push,
        get_data_and_camera=forbid,
        get_numpy_image=forbid,
        _get_panoptic_map=forbid,
        _get_depth_from_lidar=forbid,
    )
    return instance, seed


def test_camera_only_does_not_read_images_and_preserves_seed_pose_time_and_metadata(monkeypatch):
    instance, seed = camera_only_runtime()
    calls = []

    def calibration(intrinsic, distortion, size, mode):
        calls.append((size, mode))
        updated = intrinsic.copy()
        updated[0, 0] = 90.0
        return updated

    monkeypatch.setattr(
        "driversi.rendering.gaussian.mtgs_runtime.source_pinhole_intrinsics", calibration
    )
    first = instance.source_camera(0)
    assert first.fx.item() == 45.0 and first.width.item() == 4
    assert first.height.item() == 3 and first.metadata["frame_idx"] == 1
    np.testing.assert_array_equal(first.camera_to_worlds, seed.camera_to_worlds)
    np.testing.assert_array_equal(first.times, seed.times)
    first.fx[:] = -1
    first.metadata["frame_token"] = "modified"
    second = instance.source_camera(0)
    assert second.fx.item() == 45.0 and second.metadata["frame_token"] == "frame"
    assert seed.fx.item() == 100.0 and seed.metadata == {}
    assert calls == [((8, 6), "optimal")]


def test_camera_calibration_cache_is_bounded_and_keys_distortion(monkeypatch):
    instance, _ = camera_only_runtime()
    calls = []

    def calibration(intrinsic, distortion, *_args):
        calls.append(float(distortion[0]))
        return intrinsic.copy()

    monkeypatch.setattr(
        "driversi.rendering.gaussian.mtgs_runtime.source_pinhole_intrinsics", calibration
    )
    intrinsic = instance.sources.undistort_params[0][0]
    for i in range(17):
        instance.sources.undistort_params[0] = (intrinsic, np.array([i * 0.001, 0, 0, 0, 0]))
        instance.source_camera(0)
    assert len(instance._pinhole_cache) == 16 and len(calls) == 17
    instance.sources.undistort_params[0] = (intrinsic, np.zeros(5))
    instance.source_camera(0)
    assert len(calls) == 18  # The oldest calibration was evicted.


def test_camera_only_rejects_crop_and_keeps_nonundistorted_calibration(monkeypatch):
    instance, _ = camera_only_runtime()
    instance.dataset.crop_size = (0, 0, 8, 6)
    with pytest.raises(ValueError, match="cropped"):
        instance.source_camera(0)
    instance.dataset.crop_size = None
    instance.dataset.undistort_images = False
    monkeypatch.setattr(
        "driversi.rendering.gaussian.mtgs_runtime.source_pinhole_intrinsics",
        lambda *_args: pytest.fail("Undistortion was disabled"),
    )
    camera = instance.source_camera(0)
    assert camera.fx.item() == 50.0 and len(instance._pinhole_cache) == 0


def test_lens_cache_reuses_calibration_but_not_mutable_metadata_and_is_bounded(monkeypatch):
    instance, camera = camera_only_runtime()
    # Use a calibrated full-resolution seed, independent of source_camera.
    camera = instance.dataset.push_cam_details_to_metadata(camera, instance.sources, 0)
    calls = []

    def lens(**values):
        calls.append(values)
        width, height = values["output_size"]
        return np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)

    monkeypatch.setattr("driversi.rendering.gaussian.mtgs_runtime.source_lens_map", lens)
    first = instance.source_lens_projection(camera)
    first["K"][0][0] = -999
    first["distortion"][0] = -999
    second = instance.source_lens_projection(camera)
    assert len(calls) == 1 and second["K"][0][0] == 100.0
    assert second["distortion"][0] == 0
    assert first["map_x"] is second["map_x"]
    with pytest.raises(ValueError, match="read-only"):
        first["map_x"][0, 0] = 1
    for i in range(16):
        camera.fx[:] = 101.0 + i
        instance.source_lens_projection(camera)
    assert len(instance._lens_cache) == 16 and len(calls) == 17
    camera.fx[:] = 100.0
    instance.source_lens_projection(camera)
    assert len(calls) == 18


class CameraTensor(np.ndarray):
    def new_tensor(self, value):
        return np.asarray(value).view(CameraTensor)


def test_virtual_target_is_exact_after_single_optimizer_and_self_exclusion_is_camera_local():
    instance, camera = runtime()
    camera.camera_to_worlds = camera.camera_to_worlds.view(CameraTensor)
    adjustment = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])

    def optimize(value):
        result = value.camera_to_worlds.copy()
        result[0, :, :3] = adjustment @ result[0, :, :3]
        result[0, :, 3] += [1, 2, 3]
        return result

    instance.model.camera_optimizer = SimpleNamespace(apply_to_camera=optimize)
    target = np.eye(4)[:3]
    target[:, 3] = [10, 20, 30]
    calls = []

    def render(value):
        calls.append(value)
        if len(calls) == 1:
            np.testing.assert_allclose(optimize(value)[0], target)
            assert value.metadata["visible_tokens"] == ["rigid_object_parked"]
        else:
            assert "visible_tokens" not in value.metadata
        return {"rgb_appearance": np.zeros((2, 3, 3))}

    instance.model.get_outputs_for_camera = render
    result = instance.render_camera(camera, target_camera_to_world=target, exclude_tokens=("car",))
    assert result["excluded_tokens"] == ["car"]
    np.testing.assert_array_equal(result["target_camera_to_world"], target)
    assert "visible_tokens" not in camera.metadata
    np.testing.assert_array_equal(camera.camera_to_worlds, np.eye(4)[None, :3])
    instance.render_camera(camera)
    assert len(calls) == 2


def test_virtual_exclusion_rejects_missing_or_foreign_owner_before_rasterization():
    instance, camera = runtime()
    instance.model.get_outputs_for_camera = lambda *_: pytest.fail("Invalid owner reached renderer")
    with pytest.raises(ValueError, match="unique reconstructed"):
        instance.render_camera(camera, exclude_tokens=("missing",))
    instance.model.gaussian_models["rigid_object_car"].travel_id = 99
    with pytest.raises(ValueError, match="another traversal"):
        instance.render_camera(camera, exclude_tokens=("car",))


def test_source_camera_record_uses_effective_optimizer_pose_without_changing_rgb_input():
    instance, camera = runtime()
    input_pose = camera.camera_to_worlds.copy()
    input_pose[0, :, 3] = [10.0, 20.0, 30.0]
    camera.camera_to_worlds = input_pose.copy()
    effective = input_pose.copy()
    effective[0, :, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    effective[0, :, 3] += [0.1, 0.2, 0.3]
    instance.model.camera_optimizer = SimpleNamespace(
        apply_to_camera=lambda _camera: effective.copy()
    )
    expected_rgb = np.full((2, 3, 3), 0.5)

    def render(value):
        np.testing.assert_array_equal(value.camera_to_worlds, input_pose)
        return {"rgb_appearance": expected_rgb}

    instance.model.get_outputs_for_camera = render
    record = instance.render_camera(camera)
    np.testing.assert_array_equal(record["camera_to_world"], effective[0])
    assert not np.array_equal(record["camera_to_world"], input_pose[0])
    assert record["camera_coordinate_frame"] == "mtgs_reconstruction_metres"
    assert record["camera_pose_axes"] == "nerfstudio_right_up_back"
    np.testing.assert_array_equal(record["rgb"], np.full((2, 3, 3), 128, np.uint8))
    np.testing.assert_array_equal(camera.camera_to_worlds, input_pose)


def test_surface_evidence_is_copied_without_changing_rgb():
    instance, camera = runtime()
    alpha = np.full((2, 3, 1), 0.7, dtype=np.float32)
    depth = np.full((2, 3, 1), 12.0, dtype=np.float32)
    instance.model.get_outputs_for_camera = lambda _: {
        "rgb_appearance": np.full((2, 3, 3), 0.5),
        "accumulation": alpha,
        "depth": depth,
    }
    result = instance.render_camera(camera)
    alpha[:] = 0
    depth[:] = 1
    assert result["rgb"].min() == 128
    np.testing.assert_array_equal(result["accumulation"], np.full((2, 3), 0.7, dtype=np.float32))
    np.testing.assert_array_equal(result["depth"], np.full((2, 3), 12.0, dtype=np.float32))


@pytest.mark.parametrize(
    "alpha,depth",
    [
        (None, np.ones((2, 3))),
        (np.ones((2, 3)), None),
        (np.full((2, 3), np.nan), np.ones((2, 3))),
        (np.ones((2, 3)), np.full((2, 3), -1)),
        (np.full((2, 3), 1.1), np.ones((2, 3))),
        (np.ones((2, 2)), np.ones((2, 3))),
    ],
)
def test_bad_surface_evidence_fails_and_restores_pose(alpha, depth):
    instance, camera = runtime()
    instance.model.get_outputs_for_camera = lambda _: {
        "rgb_appearance": np.zeros((2, 3, 3)),
        "accumulation": alpha,
        "depth": depth,
    }
    with pytest.raises(ValueError, match="Renderer"):
        instance.render_camera(camera, rigid_poses={"car": ([8, 7, 6], [1, 0, 0, 0])})
    np.testing.assert_array_equal(
        instance.model.gaussian_models["rigid_object_car"].instance_trans.array[1], [4, 5, 6]
    )


def test_retained_asset_activation_restores_presence_and_pose_after_render_failure():
    instance, camera = runtime()
    car = instance.model.gaussian_models["rigid_object_car"]
    car.in_frame_mask[1] = False
    camera.metadata["visible_tokens"] = ["rigid_object_parked"]
    before = car.instance_trans.array.copy()

    def fail(value):
        assert car.in_frame_mask[1]
        assert "rigid_object_car" in value.metadata["visible_tokens"]
        np.testing.assert_array_equal(car.instance_trans.array[1], [9, 8, 7])
        raise RuntimeError("deliberate render failure")

    instance.model.get_outputs_for_camera = fail
    with pytest.raises(RuntimeError, match="deliberate"):
        instance.render_camera(
            camera, rigid_poses={"car": ([9, 8, 7], [1, 0, 0, 0])}, rigid_source_frames={"car": 0}
        )
    assert car.in_frame_mask == [True, False]
    np.testing.assert_array_equal(car.instance_trans.array, before)
    assert camera.metadata["visible_tokens"] == ["rigid_object_parked"]


@pytest.mark.parametrize("anchors", [{"car": 2}, {"car": True}, {"car": -1}, {"missing": 0}])
def test_invalid_asset_anchor_does_not_activate_or_mutate(anchors):
    instance, _ = runtime()
    car = instance.model.gaussian_models["rigid_object_car"]
    car.in_frame_mask[1] = False
    with pytest.raises(ValueError, match="anchor"):
        with temporary_rigid_poses(
            instance.model, 4, 1, {"car": ([9, 8, 7], [1, 0, 0, 0])}, source_frames=anchors
        ):
            pytest.fail("invalid anchor activated")
    assert not car.in_frame_mask[1]
    np.testing.assert_array_equal(car.instance_trans.array[1], [4, 5, 6])


def test_moved_asset_enters_visibility_but_camera_owner_stays_hidden():
    instance, camera = runtime()
    camera.metadata["visible_tokens"] = ["rigid_object_parked"]
    seen = []

    def render(value):
        seen.append(value.metadata["visible_tokens"])
        return {"rgb_appearance": np.zeros((2, 3, 3))}

    instance.model.get_outputs_for_camera = render
    poses = {"car": ([9, 8, 7], [1, 0, 0, 0])}
    instance.render_camera(camera, rigid_poses=poses)
    instance.render_camera(camera, rigid_poses=poses, exclude_tokens=("car",))
    assert seen == [["rigid_object_parked", "rigid_object_car"], ["rigid_object_parked"]]
    assert camera.metadata["visible_tokens"] == ["rigid_object_parked"]


def test_photometric_context_never_treats_normalized_training_rgb_as_raw_photo():
    inst, _ = runtime()
    inst.sources.v_adjust_factors = [1.7]
    context = inst.photometric_context(0)
    assert context["render_domain"] == "checkpoint_exposure_aligned"
    assert context["source_reference_domain"] == "source_rgb"
    assert context["v_adjust_factor"] == 1.7
    assert not context["raw_recovery_verified"]
    assert not context["quality_certificate"]
    inst.sources.v_adjust_factors = None
    assert inst.photometric_context(0)["render_domain"] == "source_rgb"
    for invalid in [True, -1, 1]:
        with pytest.raises(ValueError):
            inst.photometric_context(invalid)
    inst.sources.v_adjust_factors = [float("nan")]
    with pytest.raises(ValueError, match="exposure factor"):
        inst.photometric_context(0)


def test_native_eval_camera_none_indices_are_preserved():
    instance, camera = runtime()
    instance.source_partition = "eval"
    instance.source_travel_id = 4
    camera.metadata.update(cam_idx=None, frame_idx=None, driversi_source_index=0)
    assert instance.validate_camera(camera) == (4, 1)
    assert camera.metadata["cam_idx"] is None and camera.metadata["frame_idx"] is None
    camera.metadata["cam_idx"] = 0
    with pytest.raises(ValueError, match="mismatch"):
        instance.validate_camera(camera)
