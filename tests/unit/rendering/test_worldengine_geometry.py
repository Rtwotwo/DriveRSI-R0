import numpy as np
import pytest

from driversi.rendering.gaussian.worldengine_geometry import (
    RoadHeightMap,
    UnsupportedTerrain,
    background_box_support,
)


def test_sim2_and_bilinear_lift_preserve_clearance_not_planar_z():
    grid = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    terrain = RoadHeightMap(grid, {"R": [[1, 0], [0, 1]], "t": [1, 1], "s": 2})
    np.testing.assert_allclose(terrain.sample([-0.75, -0.75])["height_m"], 1.5)
    delta, evidence = terrain.lift([-0.75, -0.75, 2.5], [0.25, 0, 99.0])
    np.testing.assert_allclose(delta, [0.25, 0, 0.5])
    assert evidence["source_clearance_m"] == 1.0
    assert evidence["vertical_correction_m"] == -98.5
    import json

    json.dumps(evidence)


def test_terrain_rejects_missing_support_and_boundaries_instead_of_clamping():
    grid = np.zeros((3, 3))
    grid[1, 1] = np.nan
    terrain = RoadHeightMap(grid, {"R": [[1, 0], [0, 1]], "t": [0, 0], "s": 1})
    for xy in [[0.2, 0.2], [-0.01, 0], [2, 0], [1, 2]]:
        with pytest.raises(UnsupportedTerrain):
            terrain.sample(xy)
    with pytest.raises(ValueError):
        terrain.sample([np.nan, 0])


def test_height_rotation_is_checked_and_zero_displacement_is_identity():
    terrain = RoadHeightMap(np.zeros((5, 5)), {"R": [[0, -1], [1, 0]], "t": [2, 2], "s": 1})
    np.testing.assert_allclose(terrain.sample([1, 0])["grid_xy"], [2, 3])
    delta, _ = terrain.lift([0, 0, 1], [0, 0, 0])
    np.testing.assert_array_equal(delta, [0, 0, 0])
    with pytest.raises(ValueError):
        RoadHeightMap(np.zeros((5, 5)), {"R": [[2, 0], [0, 1]], "t": [0, 0], "s": 1})


def test_background_geometry_is_not_an_appearance_certificate():
    means = np.array([[0.0, 0, 0], [0, 0, 10], [0.4, 0, 0]])
    report = background_box_support(means, [0.9, 1, 0.01], [0, 0, 0, 1, 1, 1, 0], np.eye(4))
    assert report["gaussian_count"] == 1
    assert report["scope"] == "background_geometry_support_not_semantic_or_visual_certificate"
    assert "quality_admitted" not in report


def test_frustum_exclusion_is_whole_box_not_center_or_size():
    from itertools import product

    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support

    camera = np.c_[np.eye(3), np.zeros(3)]
    cube = np.asarray(list(product((-1, 1), repeat=3)), dtype=float)
    assert box_frustum_support(cube + [20, 0, 5], camera, [-1, 1, -1, 1])["outside_frustum"]
    assert not box_frustum_support(cube + [5, 0, 5], camera, [-1, 1, -1, 1])["outside_frustum"]
    assert not box_frustum_support(cube * 1e-8 + [0, 0, 1000], camera, [-1, 1, -1, 1])[
        "outside_frustum"
    ]
    assert box_frustum_support(cube + [0, 0, -5], camera, [-1, 1, -1, 1])["outside_frustum"]
    assert not box_frustum_support(cube, camera, [-1, 1, -1, 1])["outside_frustum"]


def test_physical_box_registration_uses_source_origin_without_object_removal():
    from driversi.rendering.gaussian.worldengine_geometry import physical_box_corners

    source = {"ego2global_original": np.eye(4), "ego2global": np.eye(4)}
    corners = physical_box_corners([1, 2, 3, 0], [2, 4, 6], [10, 20, 30], [10, 20, 30], source)
    np.testing.assert_allclose(corners.mean(0), [1, 2, 3])
    np.testing.assert_allclose(np.ptp(corners, axis=0), [2, 4, 6])
