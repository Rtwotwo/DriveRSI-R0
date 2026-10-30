import numpy as np
import pytest

from driversi.rendering.diagnostics.terrain_repair import repair_queries

TRANSFORM = {"R": [[1.0, 0.0], [0.0, 1.0]], "t": [0.0, 0.0], "s": 4.0}


def test_queries_repair_actual_target_and_preserve_supported_source():
    y, x = np.indices((41, 41))
    grid = (1.0 + 0.03 * x / 4.0 + 0.02 * y / 4.0).astype(np.float32)
    grid[20, 20] = np.nan
    queries = [
        dict(tick=5, actor="moving", center=[4.0, 5.0, 2.0], translation=[1.05, 0.05, 0.0]),
        dict(tick=0, actor="ego", center=[2.0, 2.0, 2.0], translation=[0.01, 0.01, 0.0]),
    ]
    fixed, report = repair_queries(grid, TRANSFORM, queries)
    assert report["failures_before"] == 1 and report["failures_after"] == 0
    assert report["all_queries_supported"]
    assert not report["trainable"] and not report["quality_admitted"]
    assert report["repair"]["requested_count"] == 1
    finite = np.isfinite(grid)
    assert grid[finite].tobytes() == fixed[finite].tobytes()
    assert report["queries"][1]["supported_before"]


def test_unobserved_area_is_not_flattened_or_extrapolated():
    grid = np.ones((41, 41), dtype=np.float32)
    grid[:, 20:] = np.nan
    fixed, report = repair_queries(
        grid,
        TRANSFORM,
        [dict(tick=10, actor="car", center=[4.0, 4.0, 2.0], translation=[2.0, 0.0, 0.0])],
    )
    assert report["failures_after"] == 1
    assert not report["all_queries_supported"]
    assert np.isnan(fixed[:, 20:]).all()


def test_previous_source_query_also_checked_and_empty_not_certified():
    grid = np.full((41, 41), np.nan, dtype=np.float32)
    _, report = repair_queries(
        grid,
        TRANSFORM,
        [dict(tick=0, actor="car", center=[5.0, 5.0, 2.0], translation=[0.0, 0.0, 0.0])],
    )
    assert report["failures_before"] == report["failures_after"] == 1
    assert not repair_queries(grid, TRANSFORM, [])[1]["all_queries_supported"]
    with pytest.raises(ValueError, match="Finite"):
        repair_queries(
            grid, TRANSFORM, [dict(center=[np.nan, 2.0, 2.0], translation=[0.0, 0.0, 0.0])]
        )
