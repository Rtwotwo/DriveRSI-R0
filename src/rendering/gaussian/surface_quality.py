"""Measured splat support diagnostics, separate from visual quality certification."""

from __future__ import annotations

import numpy as np

VERSION = "mtgs-surface-diagnostics-v2"
THRESHOLDS = {"low_opacity": .5, "opaque": .95, "near_depth_m": .5,
              "dark_rgb_max_uint8": 10, "depth_preview_max_m": 80.,
              "low_coverage_warning_fraction": .1}


def _channels(accumulation, depth):
    if accumulation is None and depth is None:
        return None
    if accumulation is None or depth is None:
        raise ValueError("Accumulation and depth must be jointly present or unavailable")
    alpha, depth = np.asarray(accumulation), np.asarray(depth)
    if (alpha.ndim != 2 or alpha.shape != depth.shape or not alpha.size
            or alpha.dtype.kind != "f" or depth.dtype.kind != "f"
            or not np.isfinite(alpha).all() or not np.isfinite(depth).all()
            or np.any((alpha < 0) | (alpha > 1)) or np.any(depth < 0)):
        raise ValueError("Surface channels require equal finite HxW floats, alpha [0,1], depth >=0")
    # Input transparent pixels may carry MTGS's max-depth placeholder. The
    # lens remap weights depths by alpha before interpolation to remove it.
    with np.errstate(over="ignore", invalid="ignore"):
        converted = alpha.astype(np.float32, copy=True), depth.astype(np.float32, copy=True)
    if any(not np.isfinite(value).all() for value in converted):
        raise ValueError("Surface channels exceed finite float32 storage range")
    return converted


def _valid(lens_valid, shape):
    valid = np.asarray(lens_valid)
    if valid.dtype != bool or valid.shape != shape:
        raise ValueError("Lens validity must be bool and match the diagnostic canvas")
    return valid


def remap_surface(accumulation, depth, map_x, map_y, lens_valid):
    """Use the RGB source-lens bilinear coordinates; never repair image pixels."""
    channels = _channels(accumulation, depth)
    if channels is None:
        return None, None
    import cv2

    maps = [np.asarray(item) for item in (map_x, map_y)]
    if (maps[0].ndim != 2 or maps[0].shape != maps[1].shape
            or any(item.dtype.kind != "f" or not np.isfinite(item).all() for item in maps)):
        raise ValueError("Surface lens maps require equal finite floating HxW coordinates")
    valid = _valid(lens_valid, maps[0].shape)
    source_alpha, source_depth = channels
    # Expected depth is conditional on accumulated opacity. Interpolating it
    # directly would blend alpha-zero max-depth placeholders into real surfaces.
    # Reconstruct the weighted depth numerator, remap both fields, then normalize.
    arrays = []
    for field in (source_alpha, source_alpha * source_depth):
        mapped = cv2.remap(field, maps[0].astype(np.float32), maps[1].astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        mapped[~valid] = 0
        arrays.append(mapped)
    mapped_alpha, weighted_depth = arrays
    mapped_depth = np.divide(weighted_depth, mapped_alpha, out=np.zeros_like(weighted_depth),
                             where=mapped_alpha > 0)
    return mapped_alpha, mapped_depth


def summarize_surface(rgb, accumulation, depth, lens_valid):
    """Count support cues without inferring holes, visibility truth or training quality."""
    channels = _channels(accumulation, depth)
    common = {"schema_version": 1, "kind": VERSION, "quality_certificate": False,
              "trainable": False, "thresholds": dict(THRESHOLDS),
              "depth_semantics": "alpha_weighted_lens_remapped_expected_splat_depth_not_first_hit_geometry",
              "depth_resampling": "remap(alpha * expected_depth) / remap(alpha); zero if alpha=0",
              "transparent_depth_semantics": "upstream_placeholder_excluded_before_interpolation; remapped_alpha_zero_depth_is_zero",
              "interpretation": "dark_opaque_may_be_real_objects_or_artifacts_not_automatically_holes"}
    if channels is None:
        return {**common, "status": "unavailable", "coverage_status": "unavailable",
                "reason": "renderer_channels_not_provided"}
    alpha, expected_depth = channels
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.shape != (*alpha.shape, 3):
        raise ValueError("Surface diagnostics require matching uint8 RGB")
    valid = _valid(lens_valid, alpha.shape)
    count = int(valid.sum())
    opaque = valid & (alpha >= THRESHOLDS["opaque"])
    observed = valid & (alpha > 0) & (expected_depth > 0)
    dark = (image.max(axis=2) <= THRESHOLDS["dark_rgb_max_uint8"])
    near = opaque & (expected_depth < THRESHOLDS["near_depth_m"])
    def fraction(mask):
        return float(np.count_nonzero(mask & valid) / count) if count else None
    low_fraction = fraction(alpha < THRESHOLDS["low_opacity"])
    coverage = ("unavailable" if low_fraction is None else
                "low_coverage_warning" if low_fraction > THRESHOLDS["low_coverage_warning_fraction"]
                else "no_low_coverage_warning")
    depth_values = expected_depth[observed]
    quantiles = None if not depth_values.size else {
        key: float(value) for key, value in zip(("p01", "p50", "p99"),
                                               np.quantile(depth_values, [.01, .5, .99]))}
    return {**common, "status": "measured" if count else "unsupported",
            "coverage_status": coverage,
            "coverage_warning_semantics": "engineering_low_alpha_warning_not_quality_or_authenticity_gate",
            "pixel_count": int(valid.size), "lens_valid_pixels": count,
            "fraction_denominator": "lens_valid_pixels",
            "low_opacity_fraction": low_fraction,
            "below_opaque_fraction": fraction(alpha < THRESHOLDS["opaque"]),
            "opaque_near_surface_fraction": fraction(near),
            "black_opaque_fraction": fraction(dark & opaque),
            "positive_depth_pixels": int(observed.sum()),
            "expected_depth_m_quantiles": quantiles,
            "depth_statistics_mask": "lens_valid & alpha > 0 & expected_depth > 0",
            "near_surface_mask": "lens_valid & alpha >= 0.95 & expected_depth < 0.5m",
            "preview": {"alpha": "linear_grayscale_0_to_1",
                        "depth": "log1p_grayscale_0_to_80m; clipped_preview_only",
                        "invalid_or_transparent_depth": "black; consult raw mask and channels"}}


def surface_previews(accumulation, depth, lens_valid):
    """Separate uint8 diagnostic views; lossless float channels remain authoritative."""
    channels = _channels(accumulation, depth)
    if channels is None:
        raise ValueError("Cannot preview unavailable surface channels")
    alpha, expected_depth = channels
    valid = _valid(lens_valid, alpha.shape)
    alpha_view = np.rint(alpha * 255).astype(np.uint8)
    scaled = np.log1p(np.minimum(expected_depth, THRESHOLDS["depth_preview_max_m"]))
    depth_view = np.rint(scaled / np.log1p(THRESHOLDS["depth_preview_max_m"]) * 255).astype(np.uint8)
    alpha_view[~valid] = 0
    depth_view[~valid | (alpha == 0)] = 0
    return alpha_view, depth_view
