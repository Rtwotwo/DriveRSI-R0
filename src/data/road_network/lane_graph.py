"""Read original nuPlan lane IDs and directed topology without future-pose crops."""

from pathlib import Path

import numpy as np

from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene


def _fid(value) -> str | None:
    if value is None or str(value).lower() in {"nan", "none", ""}:
        return None
    return str(int(value))


def compile_lane_graph(
    scene: Scene, maps_root: Path, *, margin: float = 120, full_map_coverage: bool = False
) -> RoadGraph:
    import pyogrio.raw
    import shapely
    from pyproj import Transformer

    if not np.isfinite(margin) or not 40 <= margin <= 500:
        raise ValueError("Initial-position map margin must be between 40 and 500 m")
    candidates = sorted((maps_root / scene.metadata["map_name"]).glob("*/map.gpkg"))
    if len(candidates) != 1:
        raise ValueError("Require exactly one GPKG version per map")
    path = candidates[0]
    if path.parent.name != scene.metadata.get("map_version"):
        raise ValueError("Graph and scene must use the same map version")
    _, _, _, fields = pyogrio.raw.read(path, layer="meta", columns=["key", "value"])
    projection = dict(zip(fields[0], fields[1], strict=True))["projectedCoordSystem"]
    origin = np.asarray(scene.metadata["origin_xyz"], dtype=float)
    positions = scene.poses[0, :, :2]
    low, high = positions.min(axis=0) - margin, positions.max(axis=0) + margin
    if full_map_coverage:
        loaded = np.asarray(scene.metadata["map_bounds_global"], float) - np.tile(origin[:2], 2)
        low, high = loaded[:2], loaded[2:]
    bounds = [*low, *high]
    if np.any(high - low > 2000):
        raise ValueError("Initial map extent exceeds 2 km")

    def read(layer, columns, *, geometry=True, bounded=True):
        info = pyogrio.read_info(path, layer=layer)
        to_source = Transformer.from_crs(projection, info["crs"], always_xy=True)
        bbox = to_source.transform_bounds(
            *(np.asarray(bounds) + np.tile(origin[:2], 2)), densify_pts=21
        )
        meta, fids, wkb, arrays = pyogrio.raw.read(
            path,
            layer=layer,
            columns=columns,
            return_fids=True,
            read_geometry=geometry,
            bbox=bbox if bounded else None,
        )
        rows = [
            dict(zip(meta["fields"], values, strict=True)) for values in zip(*arrays, strict=True)
        ]
        geometries = None
        if geometry:
            transform = Transformer.from_crs(info["crs"], projection, always_xy=True)
            geometries = shapely.transform(
                shapely.from_wkb(wkb), transform.transform, interleaved=False
            )
        return fids, rows, geometries

    # Reading all connector attributes preserves knowledge of exits outside ROI.
    fids, rows, _ = read(
        "lane_connectors",
        [
            "exit_lane_fid",
            "entry_lane_fid",
            "speed_limit_mps",
            "traffic_light_stop_line_fids",
            "lane_group_connector_fid",
        ],
        geometry=False,
        bounded=False,
    )
    outgoing, attributes = {}, {}
    for fid, row in zip(fids, rows, strict=True):
        upstream, downstream = _fid(row["exit_lane_fid"]), _fid(row["entry_lane_fid"])
        connector = "connector:" + _fid(fid)
        if upstream:
            outgoing.setdefault("lane:" + upstream, []).append(connector)
        outgoing[connector] = ["lane:" + downstream] if downstream else []
        attributes[connector] = row
    _, rows, _ = read(
        "lanes_polygons", ["lane_fid", "speed_limit_mps", "lane_group_fid"], geometry=False
    )
    for row in rows:
        attributes["lane:" + _fid(row["lane_fid"])] = row
    _, rows, geometries = read("baseline_paths", ["lane_fid", "lane_connector_fid"])
    nodes = {}
    region = shapely.box(*bounds)
    for row, geometry in zip(rows, geometries, strict=True):
        lane, connector = _fid(row["lane_fid"]), _fid(row["lane_connector_fid"])
        kind, fid = ("lane", lane) if lane else ("connector", connector)
        if fid is None or geometry is None or geometry.geom_type != "LineString":
            continue
        points = np.asarray(geometry.coords)[:, :2] - origin[:2]
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
        if len(points) < 2 or not shapely.LineString(points).intersects(region):
            continue
        key = kind + ":" + fid
        if key in nodes:
            raise ValueError(f"Multiple baselines for {key}")
        row = attributes.get(key, {})
        limit = row.get("speed_limit_mps")
        if limit is None or not np.isfinite(limit) or limit <= 0:
            limit = None
        signals = row.get("traffic_light_stop_line_fids")
        nodes[key] = {
            "kind": kind,
            "points": points.tolist(),
            "outgoing": sorted(set(outgoing.get(key, []))),
            "speed_limit_mps": float(limit) if limit is not None else None,
            "signal_stop_line_ids": None
            if signals is None
            else str(signals).split(",")
            if str(signals)
            else [],
            "signal_state": "unknown",
            "priority": "unknown",
            "roadblock_id": _fid(
                row.get("lane_group_fid" if kind == "lane" else "lane_group_connector_fid")
            ),
        }
    graph = RoadGraph(
        {
            "schema_version": 1,
            "map_name": scene.metadata["map_name"],
            "map_version": path.parent.name,
            "origin_xyz": origin.tolist(),
            "bounds_local": bounds,
            "crop_source": "scene_map_bounds" if full_map_coverage else "initial_actor_positions",
            "margin_m": margin,
            "map_file": str(path),
            "projected_crs": projection,
        },
        nodes,
    )
    graph.validate_scene(scene)
    return graph
