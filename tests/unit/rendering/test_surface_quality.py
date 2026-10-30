"""Measured surface cues must not turn dark objects into invented hole labels."""

import json

import numpy as np
import pytest

from driversi.rendering.gaussian.surface_quality import remap_surface, summarize_surface, surface_previews


def arrays():
    return (np.zeros((2, 2, 3), np.uint8), np.array([[0, .25], [.96, 1]], np.float32),
            np.array([[99, 4], [.2, 10]], np.float32), np.ones((2, 2), bool))


def test_dark_opaque_surface_and_low_alpha_are_separate_not_quality_labels():
    rgb, alpha, depth, valid = arrays()
    report = summarize_surface(rgb, alpha, depth, valid)
    assert report["low_opacity_fraction"] == .5
    assert report["below_opaque_fraction"] == .5
    assert report["black_opaque_fraction"] == .5
    assert report["opaque_near_surface_fraction"] == .25
    assert report["positive_depth_pixels"] == 3
    assert report["expected_depth_m_quantiles"]["p99"] < 10
    assert report["quality_certificate"] is False and report["trainable"] is False
    assert "not_automatically_holes" in report["interpretation"]
    np.testing.assert_array_equal(depth, arrays()[2])


def test_zero_lens_coverage_has_no_invented_percentages():
    rgb, alpha, depth, valid = arrays()
    valid[:] = False
    report = summarize_surface(rgb, alpha, depth, valid)
    assert report["status"] == "unsupported"
    assert report["low_opacity_fraction"] is None
    assert report["black_opaque_fraction"] is None
    assert report["expected_depth_m_quantiles"] is None


def test_legacy_missing_channels_are_explicitly_unknown():
    report = summarize_surface(None, None, None, None)
    assert report["status"] == "unavailable"
    assert report["reason"] == "renderer_channels_not_provided"
    assert remap_surface(None, None, None, None, None) == (None, None)


@pytest.mark.parametrize("change", [
    lambda a, d: (None, d),
    lambda a, d: (a, None),
    lambda a, d: (a[:, :1], d),
    lambda a, d: (a.astype(np.int32), d),
    lambda a, d: (a + 1, d),
    lambda a, d: (a - 1, d),
    lambda a, d: (a * np.nan, d),
    lambda a, d: (a, d * np.inf),
    lambda a, d: (a, -d),
    lambda a, d: (a, np.full(d.shape, 1e300, dtype=np.float64)),
])
def test_invalid_renderer_channels_are_not_admitted(change):
    rgb, alpha, depth, valid = arrays()
    alpha, depth = change(alpha, depth)
    with pytest.raises(ValueError):
        summarize_surface(rgb, alpha, depth, valid)


def test_lens_validity_has_strict_boolean_shape():
    rgb, alpha, depth, valid = arrays()
    for invalid in (valid.astype(float), valid[:, :1]):
        with pytest.raises(ValueError, match="Lens validity"):
            summarize_surface(rgb, alpha, depth, invalid)


def test_previews_are_separate_with_transparent_depth_placeholder_masked():
    _, alpha, depth, valid = arrays()
    alpha_view, depth_view = surface_previews(alpha, depth, valid)
    assert alpha_view.dtype == depth_view.dtype == np.uint8
    assert depth_view[0, 0] == 0 and depth[0, 0] == 99
    assert depth_view[1, 1] > depth_view[1, 0]
    assert alpha_view[1, 1] == 255


def test_calibrated_bilinear_map_and_invalid_lens_mask_apply_to_both_channels():
    pytest.importorskip("cv2")
    alpha = np.array([[0., 1.], [.4, .8]], np.float32)
    depth = np.array([[2., 4.], [6., 8.]], np.float32)
    x = np.array([[.5, .5]], np.float32)
    y = np.array([[0., 1.]], np.float32)
    valid = np.array([[True, False]])
    mapped_alpha, mapped_depth = remap_surface(alpha, depth, x, y, valid)
    np.testing.assert_array_equal(mapped_alpha, [[.5, 0]])
    np.testing.assert_array_equal(mapped_depth, [[4., 0]])
    assert mapped_alpha.dtype == mapped_depth.dtype == np.float32
    with pytest.raises(ValueError, match="lens maps"):
        remap_surface(alpha, depth, x * np.nan, y, valid)


def test_recorded_viewer_checks_surface_assets_and_keeps_rgb_primary(tmp_path):
    from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer

    for name in ("image.png", "alpha.png", "depth.png", "surface.npz"):
        (tmp_path / name).write_bytes(b"fixture")
    camera = {"camera_id": "front", "image": "image.png", "alpha": "alpha.png",
              "depth": "depth.png", "surface": "surface.npz",
              "surface_diagnostics": {"status": "measured", "quality_certificate": False}}
    manifest = {"schema_version": 1, "kind": "mtgs_world_session", "trainable": False,
                "status": "completed", "frames": [{"tick": 0, "time_seconds": 0.,
                                                     "cameras": [camera]}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    page = write_session_viewer(path).read_text()
    config = json.loads(page.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    view = config["manifest"]["frames"][0]["cameras"][0]
    assert view["url"] == "image.png"
    assert view["alpha_url"] == "alpha.png" and view["depth_url"] == "depth.png"
    assert view["surface_url"] == "surface.npz"
    assert config["manifest"]["trainable"] is False
    (tmp_path / "surface.npz").unlink()
    with pytest.raises(ValueError, match="missing"):
        write_session_viewer(path)


@pytest.mark.parametrize("placeholder", [.01, 1., 99., 10000.])
def test_transparent_source_depth_cannot_pollute_supported_near_surface(placeholder):
    pytest.importorskip("cv2")
    alpha = np.array([[1., 0.], [1., 0.]], np.float32)
    depth = np.array([[.2, placeholder], [.2, placeholder]], np.float32)
    map_x = np.array([[1/32, .75, 1.]], np.float32)
    map_y = np.zeros_like(map_x)
    lens = np.array([[True, True, False]])
    mapped_alpha, mapped_depth = remap_surface(alpha, depth, map_x, map_y, lens)
    np.testing.assert_array_equal(mapped_alpha, [[31/32, .25, 0]])
    np.testing.assert_allclose(mapped_depth, [[.2, .2, 0]], rtol=0, atol=2e-8)
    report = summarize_surface(np.full((1, 3, 3), 255, np.uint8), mapped_alpha, mapped_depth, lens)
    assert report["kind"] == "mtgs-surface-diagnostics-v2"
    assert report["opaque_near_surface_fraction"] == .5
    assert report["low_opacity_fraction"] == .5
    assert report["positive_depth_pixels"] == 2
    assert report["expected_depth_m_quantiles"]["p99"] == pytest.approx(.2)
    np.testing.assert_array_equal(depth[:, 1], np.full(2, placeholder, np.float32))


def test_fully_transparent_remapped_depth_is_zero_and_excluded():
    pytest.importorskip("cv2")
    zero = np.zeros((2, 2), np.float32)
    mapped_alpha, mapped_depth = remap_surface(
        zero, np.full((2, 2), 99., np.float32), zero, zero, np.ones((2, 2), bool))
    np.testing.assert_array_equal(mapped_alpha, zero)
    np.testing.assert_array_equal(mapped_depth, zero)
    report = summarize_surface(np.zeros((2, 2, 3), np.uint8), mapped_alpha, mapped_depth,
                               np.ones((2, 2), bool))
    assert report["positive_depth_pixels"] == 0
    assert report["expected_depth_m_quantiles"] is None
    assert report["opaque_near_surface_fraction"] == 0


@pytest.mark.parametrize("low_pixels, expected", [
    (0, "no_low_coverage_warning"),
    (1, "no_low_coverage_warning"),
    (2, "low_coverage_warning"),
])
def test_low_coverage_warning_uses_alpha_and_strict_threshold_not_black_rgb(low_pixels, expected):
    alpha = np.ones((1, 10), np.float32)
    alpha[0, :low_pixels] = .49
    # Entire image is black: genuinely opaque dark objects must not trigger it.
    report = summarize_surface(np.zeros((1, 10, 3), np.uint8), alpha,
                               np.ones((1, 10), np.float32), np.ones((1, 10), bool))
    assert report["coverage_status"] == expected
    assert report["thresholds"]["low_coverage_warning_fraction"] == .1
    assert report["low_opacity_fraction"] == low_pixels / 10
    assert report["quality_certificate"] is False
