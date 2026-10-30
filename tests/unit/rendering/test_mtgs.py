"""MTGS probe contracts without torch or a GPU in the core environment."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs import (
    _index_source_frames,
    _probe_evidence,
    _rgb8,
    _source_inventory,
    _validate_checkpoint_topology,
    compare_probe_images,
    render_mtgs_probe,
    source_camera_valid_mask,
    temporary_translation,
)


class TensorView:
    """Minimal mutable tensor view used to test restoration, including exceptions."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def __len__(self):
        return len(self.array)

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


def test_pose_probe_replaces_one_frame_and_restores_after_failure():
    node = SimpleNamespace(
        instance_trans=TensorView([[1, 2, 3], [4, 5, 6]]), is_static=False,
        in_frame_mask=[True, True], last_step=30000, last_frame_idx=1,
        cached_gaussians={"means": "stale world-space values"},
    )
    with pytest.raises(RuntimeError, match="render failed"):
        with temporary_translation(node, 1, [9, 8, 7]):
            np.testing.assert_array_equal(node.instance_trans.array, [[1, 2, 3], [9, 8, 7]])
            assert node.cached_gaussians is None and node.last_step is None
            node.cached_gaussians = {"means": "intervention-space values"}
            node.last_step = 30000
            raise RuntimeError("render failed")
    np.testing.assert_array_equal(node.instance_trans.array, [[1, 2, 3], [4, 5, 6]])
    assert node.cached_gaussians is None and node.last_frame_idx is None
    assert node.last_step is None


def test_pose_probe_rejects_absent_entity_and_nonfinite_without_mutation():
    node = SimpleNamespace(
        instance_trans=TensorView([[1, 2, 3]]), is_static=False,
        in_frame_mask=[False], cached_gaussians="untouched",
    )
    for index, translation, message in (
        (0, [1, 2, 3], "absent"), (2, [1, 2, 3], "outside"),
        (0, [np.nan, 0, 0], "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            with temporary_translation(node, index, translation):
                pytest.fail("Invalid pose reached renderer")
    assert node.cached_gaussians == "untouched"
    np.testing.assert_array_equal(node.instance_trans.array, [[1, 2, 3]])


def test_static_pose_probe_uses_single_pose_and_preserves_parameter_identity():
    parameter = TensorView([1, 2, 3])
    node = SimpleNamespace(instance_trans=parameter, is_static=True)
    with temporary_translation(node, 47, [4, 5, 6]):
        assert node.instance_trans is parameter
        np.testing.assert_array_equal(parameter.array, [4, 5, 6])
    np.testing.assert_array_equal(parameter.array, [1, 2, 3])


def test_image_comparison_distinguishes_no_intervention_and_failed_restore():
    source = np.zeros((2, 3, 3), dtype=np.uint8)
    changed = source.copy()
    changed[0, 0] = [255, 2, 1]
    result = compare_probe_images(source, changed, source.copy())
    assert result["changed_pixel_count"] == 1
    assert result["changed_pixel_fraction"] == pytest.approx(1 / 6)
    assert result["restore_equal"] and result["max_restore_pixel_error"] == 0
    assert result["baseline_rgb_sha256"] == result["restored_rgb_sha256"]
    result = compare_probe_images(source, source, changed)
    assert result["changed_pixel_count"] == 0
    assert not result["restore_equal"] and result["max_restore_pixel_error"] == 255
    json.dumps(result, allow_nan=False)


def test_images_with_different_domains_are_not_compared():
    source = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        compare_probe_images(source, source.astype(float), source)
    with pytest.raises(ValueError, match="shape"):
        compare_probe_images(source, source[:1], source)
    with pytest.raises(ValueError, match="invalid RGB"):
        _rgb8(np.full((2, 3, 3), np.nan))
    with pytest.raises(ValueError, match="within"):
        _rgb8(np.full((2, 3, 3), 1.1))


def test_lens_mask_does_not_hide_semantically_excluded_objects():
    class Dataset:
        undistort_images = "optimal"
        RESAMPLE_MAP = {"mask": "nearest sentinel", "image": "linear sentinel"}
        # A training mask might exclude every pedestrian; this must never be read.
        mask = np.zeros((4, 6, 1), dtype=bool)

        def _undistort_image(self, image, index, *, return_mask, interpolation):
            assert index == 7 and return_mask and interpolation == "linear"
            assert image.shape == (4, 6, 3)
            lens = np.ones((4, 6, 1), dtype=bool)
            lens[:, :2] = False
            return {"valid_mask": lens}

        def _resize_numpy_image(self, image, newsize, resample):
            assert newsize == (3, 2) and resample == "nearest sentinel"
            assert image.shape == (4, 6) and image[:, 2:].all()
            return np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)

    valid = source_camera_valid_mask(Dataset(), 7, (6, 4), (3, 2))
    np.testing.assert_array_equal(valid, [[False, True, True], [False, True, True]])
    original = np.zeros((2, 3, 3), dtype=np.uint8)
    edited = original.copy()
    edited[0, :2] = 255
    metrics = compare_probe_images(original, edited, original, valid)
    assert metrics["changed_pixel_count"] == 2
    assert metrics["valid_changed_pixel_count"] == 1
    assert metrics["changed_outside_camera_valid_count"] == 1
    assert metrics["valid_changed_pixel_fraction"] == 0.25
    assert metrics["camera_valid_pixel_count"] == 4
    with pytest.raises(ValueError, match="nonempty bool"):
        compare_probe_images(original, edited, original, np.zeros((2, 3), dtype=bool))


def test_no_lens_distortion_has_full_geometric_support():
    dataset = SimpleNamespace(undistort_images=False, mask=np.zeros((2, 3)))
    assert source_camera_valid_mask(dataset, 0, (6, 4), (3, 2)).all()


def test_checkpoint_mismatch_never_silently_turns_vehicle_static():
    node = SimpleNamespace(instance_trans=np.zeros((3, 3)), instance_quats=np.zeros((3, 4)))
    model = SimpleNamespace(gaussian_models={"rigid_object_a": node}, num_train_data=8)
    state = {"_model.gaussian_models.rigid_object_a.instance_trans": np.zeros((3, 3)),
             "_model.gaussian_models.rigid_object_a.instance_quats": np.zeros((3, 4)),
             "_model.camera_optimizer.pose_adjustment": np.zeros((8, 6))}
    _validate_checkpoint_topology(model, state)
    state["_model.gaussian_models.rigid_object_a.instance_trans"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="pose shape"):
        _validate_checkpoint_topology(model, state)
    with pytest.raises(ValueError, match="node mismatch"):
        _validate_checkpoint_topology(model, {})


def test_deserialization_is_explicit_and_budget_checked_before_gpu_import(tmp_path):
    args = dict(config_path=tmp_path / "none", checkpoint_path=tmp_path / "none",
                road_block_config=tmp_path / "none", output=tmp_path / "out")
    with pytest.raises(ValueError, match="trusted"):
        render_mtgs_probe(**args)
    with pytest.raises(ValueError, match="budget"):
        render_mtgs_probe(**args, trusted_assets=True, frame_count=100)
    with pytest.raises(ValueError, match="nonzero"):
        render_mtgs_probe(**args, trusted_assets=True, displacement=(0, 0, 0))


def test_import_does_not_load_optional_gpu_libraries():
    code = (
        "import sys; "
        "import driversi.rendering.gaussian.mtgs; "
        "assert not {'torch', 'mtgs', 'gsplat', 'nerfstudio'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_inventory_uses_stable_tokens_despite_different_train_eval_frame_indices():
    frames = [
        {"token": "skipped", "timestamp": 1, "gt_names": ["vehicle"], "track_tokens": ["a"]},
        {"token": "train", "timestamp": 2, "gt_names": ["pedestrian"], "track_tokens": ["p"]},
        {"token": "eval", "timestamp": 3, "gt_names": ["vehicle"], "track_tokens": ["b"]},
    ]
    indexed = _index_source_frames({"log-video-5": {"frame_infos": frames}})
    # Both loaders could call their first filtered image frame zero. Tokens remain distinct.
    selected = indexed[(5, "train")]
    assert selected["timestamp"] == 2
    assert _source_inventory(selected, {"b"}) == [
        {"token": "p", "class": "pedestrian", "reconstructed_rigid_node": False,
         "physics_bound": False, "appearance_coverage": "unverified"},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        _index_source_frames({"a-5": {"frame_infos": frames}, "b-5": {"frame_infos": frames}})


def test_failed_render_retains_explicit_attempt_and_failure_manifest(tmp_path):
    manifest = {"selection": {"frame_count": 2}, "frames": [], "attempts": []}
    with pytest.raises(RuntimeError, match="GPU failed"):
        with _probe_evidence(tmp_path, manifest):
            manifest["attempts"].append({"dataset_index": 7, "status": "running"})
            raise RuntimeError("GPU failed")
    evidence = json.loads((tmp_path / "manifest.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["attempts"][0]["status"] == "failed"
    assert evidence["selection"]["candidates_examined"] == 1
    assert not (tmp_path / ".manifest.json.tmp").exists()


def test_insufficient_visible_candidates_are_failed_not_success(tmp_path):
    manifest = {"selection": {"frame_count": 2}, "frames": [{"baseline": "one.png"}],
                "attempts": [{"status": "skipped", "reason": "no_changed_pixels"}]}
    with pytest.raises(RuntimeError, match="1/2"):
        with _probe_evidence(tmp_path, manifest):
            pass
    evidence = json.loads((tmp_path / "manifest.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["attempts"][0]["reason"] == "no_changed_pixels"
