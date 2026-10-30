"""Offline, bounded interpolation of holes in released road-height assets.

The live sampler retains its strict finite bilinear rule. This utility derives a
new asset and never overwrites measured values or uses earlier repairs as data.
A locally planar surface must have support on all sides and pass fixed spatial
holdouts. These tests are geometry evidence, not image or training admission.
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from .worldengine_geometry import RoadHeightMap


@dataclass(frozen=True)
class TerrainRepairPolicy:
    radius_m: float = 2.0
    min_support: int = 16
    max_slope: float = 0.2
    max_rmse_m: float = 0.03
    max_residual_m: float = 0.08
    max_holdout_residual_m: float = 0.08
    max_condition: float = 20.0

    def __post_init__(self):
        numeric = [self.radius_m, self.max_slope, self.max_rmse_m,
                   self.max_residual_m, self.max_holdout_residual_m,
                   self.max_condition]
        if not np.isfinite(numeric).all() or min(numeric) <= 0:
            raise ValueError("Positive finite terrain repair limits required")
        if isinstance(self.min_support, bool) or not isinstance(self.min_support, int) or self.min_support < 12:
            raise ValueError("At least 12 measured support cells required")


def _inside_hull(points):
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        return False
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for sequence, hull in ((points, lower), (reversed(points), upper)):
        for point in sequence:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
    hull = lower[:-1] + upper[:-1]
    return len(hull) >= 3 and all(
        cross(hull[i], hull[(i+1) % len(hull)], (0., 0.)) >= -1e-12
        for i in range(len(hull))
    )


def repair_road_height_holes(grid, transform, requested_cells, policy=None):
    """Return a derived grid and complete per-cell provenance; no extrapolation.

    Coordinates are row/column indices. Only NaN holes may be filled. Infinite
    values remain errors rather than becoming inferred terrain. Every local fit
    uses all original finite samples inside the metric radius; no outliers are
    discarded. Four deterministic row/column-parity folds test interpolation.
    """
    policy = policy or TerrainRepairPolicy()
    source = np.asarray(grid)
    if source.dtype.kind != "f":
        raise ValueError("Floating point road height asset required")
    terrain = RoadHeightMap(source, transform)
    cells = []
    for value in requested_cells:
        if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in value)):
            raise ValueError("Integer row/column pairs required")
        cells.append(tuple(map(int, value)))
    cells = sorted(set(cells))
    repaired = source.copy()
    records = []
    radius = int(np.ceil(policy.radius_m * terrain.scale))
    for row, col in cells:
        record = {"cell": [row, col], "accepted": False, "reasons": []}
        records.append(record)
        if not (0 <= row < source.shape[0] and 0 <= col < source.shape[1]):
            record["reasons"].append("outside_asset")
            continue
        if not np.isnan(source[row, col]):
            record["reasons"].append("not_nan_hole")
            continue
        top, left = max(0, row-radius), max(0, col-radius)
        patch = source[top:min(source.shape[0], row+radius+1), left:min(source.shape[1], col+radius+1)]
        yy, xx = np.where(np.isfinite(patch))
        yy, xx = yy+top, xx+left
        xy = np.column_stack((xx-col, yy-row)) / terrain.scale
        within = np.einsum("ij,ij->i", xy, xy) <= policy.radius_m**2 + 1e-12
        xy, yy, xx = xy[within], yy[within], xx[within]
        z = source[yy, xx].astype(float)
        record["support_count"] = len(z)
        record["source_cells"] = np.column_stack((yy, xx)).tolist()
        record["source_heights_m"] = z.tolist()
        if len(z) < policy.min_support:
            record["reasons"].append("insufficient_measured_support")
            continue
        if not _inside_hull(xy):
            record["reasons"].append("outside_measured_convex_hull")
            continue
        design = np.column_stack((xy, np.ones(len(z))))
        condition = float(np.linalg.cond(design))
        record["condition"] = condition if np.isfinite(condition) else None
        if not np.isfinite(condition) or condition > policy.max_condition:
            record["reasons"].append("ill_conditioned_surface")
            continue
        coef = np.linalg.lstsq(design, z, rcond=None)[0]
        residual = z - design @ coef
        rmse, maximum = float(np.sqrt(np.mean(residual**2))), float(abs(residual).max())
        slope = float(np.linalg.norm(coef[:2]))
        record.update(plane_centered_grid_basis=coef.tolist(), slope=slope,
                      rmse_m=rmse, max_residual_m=maximum)
        if slope > policy.max_slope:
            record["reasons"].append("excessive_surface_slope")
        if rmse > policy.max_rmse_m or maximum > policy.max_residual_m:
            record["reasons"].append("inconsistent_measured_surface")
        folds = (yy % 2)*2 + (xx % 2)
        fold_records = []
        for fold in range(4):
            test = folds == fold
            train = ~test
            fold_record = {"fold": fold, "held_out_count": int(test.sum()),
                           "training_count": int(train.sum())}
            fold_records.append(fold_record)
            if test.sum() < 2 or train.sum() < 6:
                fold_record["error"] = "insufficient_spatial_holdout"
                continue
            fold_condition = float(np.linalg.cond(design[train]))
            if not np.isfinite(fold_condition) or fold_condition > policy.max_condition:
                fold_record["error"] = "ill_conditioned_holdout_fit"
                continue
            fit = np.linalg.lstsq(design[train], z[train], rcond=None)[0]
            error = z[test] - design[test] @ fit
            fold_record["max_residual_m"] = float(abs(error).max())
            fold_record["rmse_m"] = float(np.sqrt(np.mean(error**2)))
        record["spatial_holdouts"] = fold_records
        if any("error" in f or f["max_residual_m"] > policy.max_holdout_residual_m for f in fold_records):
            record["reasons"].append("spatial_holdout_failed")
        if record["reasons"]:
            continue
        height = source.dtype.type(coef[2])
        if not np.isfinite(height):
            record["reasons"].append("nonfinite_derived_height")
            continue
        repaired[row, col] = height
        record.update(accepted=True, height_m=float(height))
    finite = np.isfinite(source)
    if repaired[finite].tobytes() != source[finite].tobytes():
        raise AssertionError("Measured height values changed")
    return repaired, {
        "method": "bounded_original_surface_with_spatial_holdouts_v1",
        "policy": asdict(policy), "input_dtype": str(source.dtype),
        "input_shape": list(source.shape),
        "transform": {"R": terrain.rotation.tolist(), "t": terrain.translation.tolist(), "s": terrain.scale},
        "input_array_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "output_array_sha256": hashlib.sha256(repaired.tobytes()).hexdigest(),
        "requested_count": len(records), "accepted_count": sum(x["accepted"] for x in records),
        "original_finite_values_byte_identical": True,
        "repaired_values_used_as_fit_support": False,
        "quality_admitted": False,
        "scope": "local_vertical_interpolation_not_semantic_surface_or_visual_admission",
        "cells": records,
    }
