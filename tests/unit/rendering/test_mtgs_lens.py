"""Actual OpenCV lens math; run in the isolated MTGS environment, not core-only CI."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="Lens math requires the isolated renderer's OpenCV")

from driversi.rendering.gaussian.mtgs_runtime import (  # noqa: E402
    project_source_lens,
    resample_source_lens,
    source_lens_map,
    source_pinhole_intrinsics,
)


def test_identity_lens_preserves_all_pixels_and_half_pixel_convention():
    intrinsic = np.array([[100.0, 0, 4.1], [0, 110, 3.2], [0, 0, 1]])
    x, y = source_lens_map(
        raw_intrinsics=intrinsic, distortion=np.zeros(5), pinhole_intrinsics=intrinsic,
        raw_size=(8, 6), output_size=(8, 6),
    )
    yy, xx = np.mgrid[:6, :8]
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    image = np.arange(144, dtype=np.uint8).reshape(6, 8, 3)
    sampled, valid = resample_source_lens(image, x, y)
    assert valid.all()
    np.testing.assert_array_equal(image, sampled)


def test_distorted_projection_roundtrip_with_actual_opencv():
    raw_k = np.array([[500.0, 0, 319.5], [0, 510.0, 240.5], [0, 0, 1]])
    pinhole_k = np.array([[280.0, 0, 160], [0, 285.0, 120], [0, 0, 1]])
    distortion = np.array([-0.12, 0.03, 0.002, -0.001, 0.001])
    x, y = source_lens_map(
        raw_intrinsics=raw_k, distortion=distortion, pinhole_intrinsics=pinhole_k,
        raw_size=(640, 480), output_size=(320, 240),
    )
    normalized = np.stack(((x + 0.5 - pinhole_k[0, 2]) / pinhole_k[0, 0],
                           (y + 0.5 - pinhole_k[1, 2]) / pinhole_k[1, 1],
                           np.ones_like(x)), axis=-1)
    redistorted, _ = cv2.projectPoints(
        normalized.reshape(-1, 3).astype(np.float64), np.zeros(3), np.zeros(3), raw_k, distortion,
    )
    yy, xx = np.mgrid[:240, :320]
    expected = np.stack(((xx + 0.5) * 2, (yy + 0.5) * 2), axis=-1)
    np.testing.assert_allclose(redistorted.reshape(240, 320, 2), expected, atol=0.002)


def test_lens_support_requires_all_bilinear_neighbors_and_does_not_edit_input():
    image = np.full((3, 4, 3), 100, dtype=np.uint8)
    support = np.ones((3, 4), dtype=bool)
    support[:, 0] = False
    x = np.array([[0.5, 1.5, -1, 3]], dtype=np.float32)
    y = np.ones_like(x)
    sampled, valid = resample_source_lens(image, x, y, support)
    np.testing.assert_array_equal(valid, [[False, True, False, True]])
    np.testing.assert_array_equal(sampled[0, 0], [0, 0, 0])
    assert image.min() == 100 and sampled[0, 1].min() == 100


def test_resolution_scaling_preserves_physical_lens_rays():
    raw_k = np.array([[100.0, 0, 4], [0, 120, 3], [0, 0, 1]])
    pinhole_k = raw_k.copy()
    pinhole_k[:2] *= 0.5
    x, y = source_lens_map(
        raw_intrinsics=raw_k, distortion=np.zeros(5), pinhole_intrinsics=pinhole_k,
        raw_size=(8, 6), output_size=(4, 3),
    )
    yy, xx = np.mgrid[:3, :4]
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)


def test_strong_distortion_corners_converge_beyond_opencv_default_iterations():
    raw_k = np.array([[770.0, 0, 480], [0, 770.0, 280], [0, 0, 1]])
    distortion = np.array([-0.35, 0.17, -0.002, 0.0004, -0.052])
    cv_k = raw_k.copy()
    cv_k[:2, 2] -= 0.5
    pinhole, _ = cv2.getOptimalNewCameraMatrix(cv_k, distortion, (960, 540), 1)
    pinhole[:2, 2] += 0.5
    x, y = source_lens_map(
        raw_intrinsics=raw_k, distortion=distortion, pinhole_intrinsics=pinhole,
        raw_size=(960, 540), output_size=(960, 540),
    )
    normalized = np.stack(((x + 0.5 - pinhole[0, 2]) / pinhole[0, 0],
                           (y + 0.5 - pinhole[1, 2]) / pinhole[1, 1],
                           np.ones_like(x)), axis=-1)[::30, ::30]
    redistorted, _ = cv2.projectPoints(
        normalized.reshape(-1, 3).astype(np.float64), np.zeros(3), np.zeros(3), raw_k, distortion,
    )
    yy, xx = np.mgrid[:540, :960]
    expected = np.stack((xx + 0.5, yy + 0.5), axis=-1)[::30, ::30]
    np.testing.assert_allclose(redistorted.reshape(expected.shape), expected, atol=0.0002)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_camera_only_intrinsics_match_optimal_full_canvas_and_preserve_inputs(dtype):
    raw = np.array([[770., 0, 480.1], [0, 760., 270.4], [0, 0, 1]], dtype=dtype)
    distortion = np.array([-.3, .12, -.001, .003, -.02], dtype=dtype)
    saved = raw.copy()
    result = source_pinhole_intrinsics(raw, distortion, (960, 540), "optimal")
    # Validate the calibration through the independently constructed OpenCV
    # projection, using full alpha=1 canvas (never an ROI crop).
    cv_raw = raw.copy()
    cv_raw[:2, 2] -= .5
    expected, _ = cv2.getOptimalNewCameraMatrix(cv_raw, distortion, (960, 540), 1)
    expected[:2, 2] += .5
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(raw, saved)
    assert result.dtype == expected.dtype


def test_camera_only_keep_focal_length_preserves_pixel_centres():
    raw = np.array([[100., 0, 4.125], [0, 105., 3.25], [0, 0, 1]])
    result = source_pinhole_intrinsics(raw, np.zeros(5), (8, 6), "keep_focal_length")
    np.testing.assert_array_equal(result, raw)
    assert result is not raw
    with pytest.raises(ValueError, match="Unsupported"):
        source_pinhole_intrinsics(raw, np.zeros(5), (8, 6), "crop_roi")


def test_shared_projection_keeps_rgb_alpha_and_depth_on_identical_support():
    from driversi.rendering.gaussian.surface_quality import remap_surface

    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    alpha = np.ones((3, 4), dtype=np.float32)
    alpha[:, 0] = 0
    depth = np.full((3, 4), 5., dtype=np.float32)
    depth[:, 0] = 1000.  # Transparent upstream placeholder must not bleed.
    x = np.array([[.5, 1.5, -1, 3]], dtype=np.float32)
    y = np.ones_like(x)
    result = {"rgb": rgb, "accumulation": alpha, "depth": depth}
    saved = {k: v.copy() for k, v in result.items()}
    actual = project_source_lens(result, {"map_x": x, "map_y": y})
    expected_rgb, expected_valid = resample_source_lens(rgb, x, y)
    expected_alpha, expected_depth = remap_surface(alpha, depth, x, y, expected_valid)
    for k, expected in [("rgb", expected_rgb), ("valid", expected_valid),
                        ("accumulation", expected_alpha), ("depth", expected_depth)]:
        np.testing.assert_array_equal(actual[k], expected)
    assert actual["depth"][0, 0] == 5.
    assert not actual["valid"][0, 2]
    for k, value in saved.items():
        np.testing.assert_array_equal(result[k], value)


def test_shared_projection_retains_unknown_surface_and_rejects_wrong_canvas():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    y, x = np.mgrid[:2, :3].astype(np.float32)
    projection = {"map_x": x, "map_y": y}
    actual = project_source_lens({"rgb": rgb}, projection)
    assert actual["depth"] is None and actual["accumulation"] is None
    with pytest.raises(ValueError, match="match pinhole"):
        project_source_lens({"rgb": rgb, "accumulation": np.ones((1, 1))}, projection)


def test_official_reference_keeps_raw_and_checkpoint_targets_separate(tmp_path):
    from collections import OrderedDict
    from threading import Lock
    from types import SimpleNamespace

    from PIL import Image

    from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime

    raw = np.full((4, 6, 3), 80, dtype=np.uint8)
    target = np.full_like(raw, 120)
    path = tmp_path / "raw.png"
    Image.fromarray(raw).save(path)
    instance = MTGSRuntime.__new__(MTGSRuntime)
    instance._operation = Lock()
    instance._lens_cache = OrderedDict()
    instance.sources = SimpleNamespace(image_filenames=[path],v_adjust_factors=[1.5])
    calls = []
    def get_target(index):
        calls.append(index)
        return target.copy()
    instance.dataset = SimpleNamespace(get_numpy_image=get_target)
    result = instance.source_reference(0,(3,2))
    assert calls == [0]
    assert result["raw"].shape == result["checkpoint_target"].shape == (2,3,3)
    assert (result["raw"]==80).all() and (result["checkpoint_target"]==120).all()
    assert result["photometric"]["v_adjust_factor"] == 1.5
    np.testing.assert_array_equal(np.asarray(Image.open(path)),raw)
