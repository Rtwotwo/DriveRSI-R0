from pathlib import Path

import pytest

shapely = pytest.importorskip("shapely")
pyogrio = pytest.importorskip("pyogrio.raw")
from driversi.data.catalog import scene_topology  # noqa: E402


def test_lane_graph_and_connector_membership_are_separate(monkeypatch):
    layers = {
        "lanes_polygons": ([shapely.box(0, 0, 10, 3), shapely.box(0, 4, 10, 7)], [11, 12],
                           {"lane_fid": [11, 12], "lane_group_fid": [7, 7]}),
        "lane_connectors": ([shapely.LineString([(10, 1), (20, 1)]), shapely.LineString([(10, 1), (15, 7)])], [31, 32],
                            {"entry_lane_fid": [21, 22], "exit_lane_fid": [11, 11]}),
        "intersections": ([shapely.box(10, -2, 20, 8)], [1], {}),
        "crosswalks": ([shapely.box(8, 0, 9, 7)], [1], {}),
        "gen_lane_connectors_scaled_width_polygons": ([shapely.box(10, 0, 20, 3)], [31], {}),
    }
    monkeypatch.setattr(scene_topology, "read_layer", lambda path, layer, cols, proj: layers[layer])
    monkeypatch.setattr(pyogrio, "read", lambda *a, **k: ({}, None, None, [["projectedCoordSystem"], ["EPSG:3857"]]))
    result = scene_topology.map_facts(Path("/maps/city/v1/map.gpkg"), [
        {"token": "lane", "map": "city", "xy": [5, 1]},
        {"token": "connector", "map": "city", "xy": [12, 1]}])
    assert result[0]["lane_association"] == "unique_polygon"
    assert result[0]["features"]["lane_outgoing_edges"] == 2
    assert result[0]["features"]["lane_group_lanes"] == 2
    assert result[0]["features"]["inside_intersection"] == 0
    assert result[1]["features"]["inside_intersection"] == 1
    assert result[1]["features"]["inside_lane_connector"] == 1
    assert result[1]["features"]["lane_outgoing_edges"] is None
    assert result[1]["lane_association"] == "not_on_lane_polygon"
