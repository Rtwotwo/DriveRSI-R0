import numpy as np
import pytest

from driversi.rendering.gaussian.worldengine_geometry import RoadHeightMap, UnsupportedTerrain
from driversi.rendering.gaussian.worldengine_terrain_repair import repair_road_height_holes

TRANSFORM = {"R": [[1., 0.], [0., 1.]], "t": [0., 0.], "s": 4.}


def plane():
    y, x = np.indices((41, 41))
    return (5. + .08*x/4. - .03*y/4.).astype(np.float32)


def test_repair_plane_preserves_all_measured_and_unrequested_cells():
    grid = plane()
    expected = float(grid[20, 20])
    grid[20, 20] = grid[19, 21] = np.nan
    with pytest.raises(UnsupportedTerrain):
        RoadHeightMap(grid, TRANSFORM).sample([5.01, 5.01])
    fixed, report = repair_road_height_holes(grid, TRANSFORM, [[20, 20]])
    assert fixed[20, 20] == pytest.approx(expected, abs=1e-6)
    assert np.isnan(fixed[19, 21])
    assert report["accepted_count"] == 1
    assert not report["quality_admitted"]
    assert np.isfinite(RoadHeightMap(fixed, TRANSFORM).sample([5.01, 5.01])["height_m"])
    finite = np.isfinite(grid)
    assert fixed[finite].tobytes() == grid[finite].tobytes()
    assert all(f["max_residual_m"] < 1e-5 for f in report["cells"][0]["spatial_holdouts"])


def test_nearby_discontinuity_is_not_discarded_as_an_outlier():
    grid = plane()
    grid[20, 20] = np.nan
    grid[20, 21] += 1.
    fixed, report = repair_road_height_holes(grid, TRANSFORM, [[20, 20]])
    assert np.isnan(fixed[20, 20])
    assert "inconsistent_measured_surface" in report["cells"][0]["reasons"]
    assert "spatial_holdout_failed" in report["cells"][0]["reasons"]


def test_no_extrapolation_or_chained_support():
    grid = plane()
    grid[:, 20:] = np.nan
    a, ra = repair_road_height_holes(grid, TRANSFORM, [[20, 20], [20, 21]])
    b, rb = repair_road_height_holes(grid, TRANSFORM, [[20, 21], [20, 20]])
    assert ra == rb
    assert np.isnan(a[:, 20:]).all() and np.isnan(b[:, 20:]).all()
    assert all("outside_measured_convex_hull" in x["reasons"] for x in ra["cells"])
    assert all(all(c[1] < 20 for c in x["source_cells"]) for x in ra["cells"])


def test_existing_values_infinite_corruption_and_bounds_are_not_repaired():
    grid = plane()
    grid[20, 21] = np.inf
    fixed, report = repair_road_height_holes(grid, TRANSFORM, [[20, 20], [20, 21], [-1, 1]])
    assert fixed.tobytes() == grid.tobytes()
    assert report["accepted_count"] == 0
    assert [r["reasons"] for r in report["cells"]] == [["outside_asset"], ["not_nan_hole"], ["not_nan_hole"]]


@pytest.mark.parametrize("cells", [[[1.2, 2]], [[True, 1]], [[1]], ["a"]])
def test_reject_malformed_cell_coordinates(cells):
    with pytest.raises(ValueError, match="Integer row/column"):
        repair_road_height_holes(plane(), TRANSFORM, cells)


def test_steep_or_sparse_surface_is_not_accepted():
    grid = plane()
    grid *= 10.
    grid[20, 20] = np.nan
    _, report = repair_road_height_holes(grid, TRANSFORM, [[20, 20]])
    assert "excessive_surface_slope" in report["cells"][0]["reasons"]
    grid[:] = np.nan
    grid[19, 20] = 5.
    _, report = repair_road_height_holes(grid, TRANSFORM, [[20, 20]])
    assert report["cells"][0]["reasons"] == ["insufficient_measured_support"]


def test_empty_input_is_stable_and_does_not_modify_asset():
    grid = plane()
    fixed, report = repair_road_height_holes(grid, TRANSFORM, [])
    assert fixed.tobytes() == grid.tobytes()
    assert report["requested_count"] == report["accepted_count"] == 0
