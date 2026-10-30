"""Road surfaces must follow the SDK union without moving boundary vertices."""

import sys
import types

import numpy as np
import pytest

from driversi.contracts.scene import ROAD_EDGE
from driversi.data.road_network.maps import extract_map


def test_official_surface_keeps_lane_gap_and_concave_boundary(tmp_path, monkeypatch):
    shapely = pytest.importorskip("shapely")
    pytest.importorskip("pyproj")
    path = tmp_path / "city" / "v1" / "map.gpkg"
    path.parent.mkdir(parents=True)
    path.touch()
    # A narrow indentation is deliberate: simplification must not erase it.
    surface = shapely.Polygon(
        [(-10, -5), (10, -5), (10, 5), (0.04, 5), (0, 4.96), (-0.04, 5), (-10, 5)]
    )
    seen = []

    def read(path, *, layer, **kwargs):
        seen.append(layer)
        if layer == "meta":
            return {}, None, None, [["projectedCoordSystem"], ["EPSG:3857"]]
        shapes = {
            "baseline_paths": [shapely.LineString([(-9, 0), (9, 0)])],
            "road_segments": [surface],
            "intersections": [],
            "generic_drivable_areas": [],
            "carpark_areas": [],
        }
        assert layer in shapes, "lane polygons must not define drivable boundaries"
        return {}, None, np.asarray([shapely.to_wkb(g) for g in shapes[layer]], object), []

    package = types.ModuleType("pyogrio")
    raw = types.ModuleType("pyogrio.raw")
    raw.read = read
    package.raw = raw
    package.read_info = lambda *a, **k: {"crs": "EPSG:3857"}
    monkeypatch.setitem(sys.modules, "pyogrio", package)
    monkeypatch.setitem(sys.modules, "pyogrio.raw", raw)
    points, offsets, kinds, meta = extract_map("city", [[0, 0]], [0, 0, 0], tmp_path)
    edges = [
        shapely.LineString(points[a:b, :2])
        for a, b, k in zip(offsets[:-1], offsets[1:], kinds)
        if k == ROAD_EDGE
    ]
    boundary = shapely.union_all(edges)
    assert boundary.hausdorff_distance(surface.boundary) < 1e-10
    assert boundary.distance(shapely.Point(0, 4.96)) < 1e-10
    assert "road_segments" in seen and "lanes_polygons" not in seen
    assert meta["road_edge_model"] == "clipped_official_nuplan_drivable_union_v2"
