import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_road_surface import MeasuredRoadPolicy, MeasuredRoadSurface
from driversi.rendering.gaussian.worldengine_geometry import UnsupportedTerrain


def cloud():
    x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    return np.column_stack((x.ravel(), y.ravel(), (1 + 0.03 * x - 0.04 * y).ravel()))


def test_measured_surface_interpolates_and_preserves_clearance():
    road = MeasuredRoadSurface(cloud(), {"source": "test_measured"})
    source = road.sample([0.0, 0.0])
    assert source["height_m"] == pytest.approx(1)
    assert len(source["spatial_holdouts"]) == 4
    delta, record = road.lift([0, 0, 2], [0.3, -0.2, 0])
    assert delta[2] == pytest.approx(0.017)
    assert record["source_clearance_m"] == pytest.approx(1)
    assert record["target"]["provenance"]["source"] == "test_measured"


def test_outlier_not_silently_dropped_and_extrapolation_rejected():
    points = cloud()
    points[len(points) // 2, 2] += 1
    road = MeasuredRoadSurface(points, {"source": "test_measured"})
    with pytest.raises(UnsupportedTerrain, match="inconsistent_measured") as caught:
        road.sample([0, 0])
    assert caught.value.evidence["outlier_pruning"] is False
    road = MeasuredRoadSurface(cloud(), {"source": "test_measured"})
    with pytest.raises(UnsupportedTerrain):
        road.sample([3, 0])


def test_mutable_caller_cannot_change_measurements():
    points = cloud()
    road = MeasuredRoadSurface(points, {"source": "test_measured"})
    points[:] = 0
    assert road.sample([0, 0])["height_m"] == pytest.approx(1)
    with pytest.raises(ValueError):
        road.points[0] = 0


def test_nonfinite_bad_policy_and_invalid_queries_rejected():
    with pytest.raises(ValueError):
        MeasuredRoadPolicy(radius_m=float("inf"))
    with pytest.raises(ValueError):
        MeasuredRoadPolicy(min_support=True)
    points = cloud()
    points[0, 2] = float("nan")
    with pytest.raises(ValueError, match="Finite measured"):
        MeasuredRoadSurface(points, {"source": "test_measured"})
    road = MeasuredRoadSurface(cloud(), {"source": "test_measured"})
    with pytest.raises(ValueError):
        road.sample([0, 0, 0])
    with pytest.raises(ValueError):
        road.lift([0, 0, 0], [0, 0, float("nan")])


def test_one_dimensional_support_rejected():
    x = np.linspace(-2, 2, 40)
    road = MeasuredRoadSurface(np.column_stack((x, x * 0, x * 0.02)), {"source": "test_measured"})
    with pytest.raises(UnsupportedTerrain, match="convex_hull"):
        road.sample([0, 0])


def test_layer_separation_records_exclusions_and_checks_distinct_cloud():
    from driversi.rendering.gaussian.mtgs_road_surface import IndependentlyValidatedRoadSurface

    train = cloud()
    train[len(train) // 2, 2] += 1
    independent = cloud()
    independent[:, 2] += 0.005
    fit = MeasuredRoadSurface(train, {"source": "fit_traversal"})
    check = MeasuredRoadSurface(independent, {"source": "independent_traversal"})
    road = IndependentlyValidatedRoadSurface(fit, check)
    result = road.sample([0, 0])
    assert result["height_m"] == pytest.approx(1)
    separated = result["provenance"]["geometric_layer"]
    assert len(separated["excluded_points_xyz"]) == 1
    assert separated["excluded_points_xyz"][0][2] == pytest.approx(2)
    assert result["independent_surface_validation"][
        "independent_center_difference_m"
    ] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="Independent measured cloud"):
        IndependentlyValidatedRoadSurface(fit, fit)


def test_independent_disagreement_and_ambiguous_layers_remain_rejected():
    from driversi.rendering.gaussian.mtgs_road_surface import IndependentlyValidatedRoadSurface

    fit = MeasuredRoadSurface(cloud(), {"source": "fit"})
    different = cloud()
    different[:, 2] += 0.1
    road = IndependentlyValidatedRoadSurface(
        fit, MeasuredRoadSurface(different, {"source": "different"})
    )
    with pytest.raises(UnsupportedTerrain, match="independent_measured_road_surface_disagrees"):
        road.sample([0, 0])
    ambiguous = cloud()
    ambiguous[::5, 2] += 1
    road = IndependentlyValidatedRoadSurface(
        MeasuredRoadSurface(ambiguous, {"source": "ambiguous"}), fit
    )
    with pytest.raises(UnsupportedTerrain, match="ambiguous_measured_road_layers"):
        road.sample([0, 0])


def test_support_labels_do_not_admit_buildings_and_rigid_transform_is_required(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    points = cloud()
    path = tmp_path / "cloud.pcd"
    path.write_bytes(b"test measured points via mocked reader")
    labels = np.zeros(len(points), dtype=np.uint8)
    labels[::2] = 1
    semantic = tmp_path / "labels.npy"
    np.save(semantic, labels)
    monkeypatch.setitem(
        sys.modules,
        "open3d",
        SimpleNamespace(
            io=SimpleNamespace(read_point_cloud=lambda _: SimpleNamespace(points=points))
        ),
    )
    road = MeasuredRoadSurface.from_pcd(path, semantic, surface_labels=(0, 1))
    assert road.provenance["surface_count"] == len(points)
    assert (
        road.provenance["support_scope"]
        == "physical_vertical_surface_not_drivable_area_or_right_of_way"
    )
    with pytest.raises(ValueError, match="road, sidewalk or terrain"):
        MeasuredRoadSurface.from_pcd(path, semantic, surface_labels=(0, 2))
    with pytest.raises(ValueError, match="road, sidewalk or terrain"):
        MeasuredRoadSurface.from_pcd(path, semantic, surface_labels=(0, 0))
    invalid = np.eye(4)
    invalid[0, 1] = 0.01
    with pytest.raises(ValueError, match="Rigid metric"):
        MeasuredRoadSurface.from_pcd(path, semantic, coordinate_transform=invalid)
    rigid = np.eye(4)
    rigid[2, 3] = 0.02
    shifted = MeasuredRoadSurface.from_pcd(
        path, semantic, surface_labels=(0, 1), coordinate_transform=rigid
    )
    assert shifted.sample([0, 0])["height_m"] == pytest.approx(1.02)
