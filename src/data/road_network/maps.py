"""Extract a bounded metric map from nuPlan GPKG layers; no sensor dependencies."""

from pathlib import Path

import numpy as np

from driversi.contracts.scene import LANE, ROAD_EDGE, Scene


def attach_nuplan_map(scene: Scene, maps_root: Path, margin: float = 40.0) -> None:
    present = np.stack([scene.presence_at(t) for t in range(len(scene.times))])
    points, offsets, types, metadata = extract_map(
        scene.metadata["map_name"],
        scene.poses[..., :2][present],
        scene.metadata["origin_xyz"],
        maps_root,
        margin=margin,
    )
    scene.road_points, scene.road_offsets, scene.road_types = points, offsets, types
    scene.metadata.update(metadata)
    scene.metadata.setdefault("limitations", []).extend(
        ["planar_map_no_overpass_separation", "artificial_crop_boundary"]
    )
    scene.validate()


def extract_map(map_name, positions, origin, maps_root, *, margin=40.0, max_extent_m=2000):
    """Shared metric extraction; render-only callers may request a larger bounded region."""
    import pyogrio.raw
    import shapely
    from pyproj import Transformer

    if not np.isfinite(margin) or margin < 20:
        raise ValueError("Map margin must be at least 20 metres")
    candidates = sorted((maps_root / map_name).glob("*/map.gpkg"))
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one map.gpkg version for this map; use an unambiguous maps root"
        )
    path = candidates[0]
    _, _, _, fields = pyogrio.raw.read(path, layer="meta", columns=["key", "value"])
    projection = dict(zip(fields[0], fields[1], strict=True)).get("projectedCoordSystem")
    if not projection:
        raise ValueError("Map is missing projectedCoordSystem metadata")
    origin = np.asarray(origin)
    positions = np.asarray(positions).reshape(-1, 2) + origin[:2]
    low, high = positions.min(axis=0) - margin, positions.max(axis=0) + margin
    if not np.isfinite(positions).all() or np.any(high - low > max_extent_m):
        raise ValueError("Map extent exceeds configured bound or is nonfinite")
    bbox = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
    region = shapely.box(*bbox)

    def read(layer: str) -> list:
        source_crs = pyogrio.read_info(path, layer=layer)["crs"]
        to_source = Transformer.from_crs(projection, source_crs, always_xy=True)
        to_metric = Transformer.from_crs(source_crs, projection, always_xy=True)
        source_bbox = to_source.transform_bounds(*bbox, densify_pts=21)
        _, _, wkb, _ = pyogrio.raw.read(path, layer=layer, bbox=source_bbox, columns=[])
        geometries = shapely.from_wkb(wkb)
        geometries = shapely.transform(geometries, to_metric.transform, interleaved=False)
        return [
            shapely.intersection(shapely.make_valid(g), region)
            for g in geometries
            if g is not None and not g.is_empty
        ]

    lanes = read("baseline_paths")
    # Match NuPlanMap._initialize_drivable_area. Lane polygons leave gaps
    # inside legal road surfaces and must not become physical road edges.
    surface_layers = ["road_segments", "intersections", "generic_drivable_areas", "carpark_areas"]
    surfaces = [geometry for layer in surface_layers for geometry in read(layer)]
    if not lanes or not surfaces:
        raise ValueError("No lane baselines or drivable polygons in the requested region")
    surface = shapely.union_all([shapely.make_valid(g) for g in surfaces])
    lines, types = [], []

    def add(geometry, kind: int) -> None:
        if geometry.is_empty:
            return
        if geometry.geom_type in {"LineString", "LinearRing"}:
            # Simplifying a physical boundary can move it across a legal footprint.
            # Baselines are visualization/navigation lines; edges retain source vertices.
            if kind != ROAD_EDGE:
                geometry = shapely.simplify(geometry, 0.1)
            geometry = shapely.segmentize(geometry, 2.0)
            xy = np.asarray(geometry.coords)[:, :2] - origin[:2]
            if len(xy) >= 2:
                lines.append(np.column_stack((xy, np.zeros(len(xy)))))
                types.append(kind)
        elif hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                add(part, kind)

    for lane in lanes:
        add(lane, LANE)
    add(surface.boundary, ROAD_EDGE)
    if not lines or ROAD_EDGE not in types:
        raise ValueError("Map extraction produced no usable road boundary")
    metadata = {
        "map_complete": True,
        "map_file": str(path),
        "map_version": path.parent.name,
        "map_projected_crs": projection,
        "map_bounds_global": list(bbox),
        "map_layers": ["baseline_paths", *surface_layers],
        "road_edge_model": "clipped_official_nuplan_drivable_union_v2",
        "map_elevation_model": "planar_zero",
    }
    return (
        np.concatenate(lines),
        np.asarray([0, *np.cumsum([len(x) for x in lines])], dtype=np.int64),
        np.asarray(types, dtype=np.int32),
        metadata,
    )
