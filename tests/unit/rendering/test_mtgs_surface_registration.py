import json

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_surface_registration import (
    buffered_spatial_holdouts,
    derive_rigid_registration,
    load_registered_road_surface,
    require_derived_transform,
)


def test_derived_registration_is_rigid_and_arbitrary_transform_rejected():
    x, y = np.meshgrid(np.arange(12.0), np.arange(10.0))
    xy = np.column_stack((x.ravel(), y.ravel()))
    delta = 0.001 * xy[:, 0] - 0.002 * xy[:, 1] + 0.02
    transform, coefficients, _ = derive_rigid_registration(xy, delta)
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    assert coefficients[:2] == pytest.approx([0.001, -0.002])
    require_derived_transform(transform, transform)
    tampered = transform.copy()
    tampered[2, 3] += 0.01
    with pytest.raises(ValueError, match="not the transform derived"):
        require_derived_transform(tampered, transform)


def test_buffered_folds_have_disjoint_measurement_neighborhoods():
    rows = []
    for x in np.arange(24.0):
        for y in np.arange(12.0):
            h = 1 + 0.001 * x - 0.002 * y + 0.02
            rows.append(
                {
                    "xy": [x, y],
                    "moving": {"height_m": h, "plane_centered_reconstruction": [0.001, -0.002, h]},
                    "reference": {"height_m": 1},
                }
            )
    records = buffered_spatial_holdouts(rows)
    assert len(records) == 4
    assert all(x["train_test_center_gap_m"] > 4 for x in records)
    assert all(x["max_residual_m"] < 1e-4 for x in records)
    for row in rows:
        row["moving"]["height_m"] += 0.5 * np.sin(row["xy"][0])
    with pytest.raises(ValueError, match="failed unchanged spatial holdout"):
        buffered_spatial_holdouts(rows)


def test_no_arbitrary_policy_or_unlabelled_source_paths(tmp_path):
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps({"rigid_transform": np.eye(4).tolist()}))
    with pytest.raises(ValueError, match="Native traversal-labelled"):
        load_registered_road_surface(
            "fit.pcd", "f.npy", "ref.pcd", "r.npy", "moving.pcd", "m.npy", path
        )
    with pytest.raises(ValueError, match="Explicit frozen"):
        load_registered_road_surface(
            "block-12.pcd", "f.npy", "block-4.pcd", "r.npy", "block-35.pcd", "m.npy", path
        )


def test_actual_queries_reject_calibration_neighborhood_and_lift_target(monkeypatch):
    from types import SimpleNamespace

    from driversi.rendering.gaussian.mtgs_road_surface import IndependentlyValidatedRoadSurface
    from driversi.rendering.gaussian.mtgs_surface_registration import RegisteredRoadSurface
    from driversi.rendering.gaussian.worldengine_geometry import UnsupportedTerrain

    fit = SimpleNamespace(array_sha256="fit", policy="frozen", provenance={})
    other = SimpleNamespace(array_sha256="independent", policy="frozen", provenance={})
    centers = np.array([[0.0, 0.0], [20.0, 20.0]])
    surface = RegisteredRoadSurface(fit, other, centers)
    centers[:] = 1000.0  # A caller cannot move the certified exclusion zones.
    seen = []

    def measured_sample(self, xy):
        seen.append(np.asarray(xy).tolist())
        return {"height_m": 1.0}

    monkeypatch.setattr(IndependentlyValidatedRoadSurface, "sample", measured_sample)
    for query in ([0.0, 0.0], [4.999, 0.0], [20.0, 20.0]):
        with pytest.raises(UnsupportedTerrain, match="actual_query_overlaps") as caught:
            surface.sample(query)
        assert caught.value.evidence["actual_center_distance_m"] < 5.0
    assert not seen
    assert (
        surface.sample([5.0, 0.0])["calibration_query_exclusion"]["actual_center_distance_m"] == 5.0
    )
    with pytest.raises(UnsupportedTerrain, match="actual_query_overlaps"):
        surface.lift([6.0, 0.0, 1.5], [-2.0, 0.0, 0.0])
    assert seen == [[5.0, 0.0], [6.0, 0.0]]
    with pytest.raises(UnsupportedTerrain, match="actual_query_overlaps"):
        surface.lift([0.0, 0.0, 1.5], [6.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Finite reconstruction"):
        surface.sample([float("nan"), 0.0])


def test_registered_surface_requires_recomputed_finite_centers():
    from types import SimpleNamespace

    from driversi.rendering.gaussian.mtgs_surface_registration import RegisteredRoadSurface

    fit = SimpleNamespace(array_sha256="fit", policy="frozen", provenance={})
    other = SimpleNamespace(array_sha256="independent", policy="frozen", provenance={})
    for centers in ([], [[float("nan"), 0.0]], [[1.0, 2.0, 3.0]]):
        with pytest.raises(ValueError, match="Finite recomputed"):
            RegisteredRoadSurface(fit, other, centers)
