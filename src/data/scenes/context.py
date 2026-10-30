"""Compile display context independently of the limited interactive population.

Source annotations retain their original sampling and presence. Map features are
diagnostic geometry; importing them never changes collision or control semantics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from driversi.contracts.context import SceneContext, scene_fingerprint
from driversi.contracts.scene import OTHER_OBJECT, Scene
from driversi.data.sources.navsim import KINDS, load_log, transform_box, yaw_from_quaternion

# Source-layer names for the NAVSIM/nuPlan map adapter, not inferred traffic rules.
# Additional spatial layers are still discovered and retained below.
SUPPORTED_MAP_LAYERS = (
    "baseline_paths", "boundaries", "carpark_areas", "crosswalks", "generic_drivable_areas",
    "dubins_nodes", "gen_lane_connectors_scaled_width_polygons", "intersections",
    "lane_connectors", "lane_group_connectors", "lane_groups_polygons", "lanes_polygons",
    "road_segments", "stop_polygons", "traffic_lights", "traffic_signs", "walkways",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_frames(scene: Scene, frames: list[dict]) -> list[dict]:
    """Bind an annotation window to the scene and preserve every source object."""
    start = scene.metadata.get("source_start_frame")
    if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(frames):
        raise ValueError("Scene requires a valid source_start_frame")
    initial = frames[start]
    origin = np.asarray(scene.metadata["origin_xyz"], dtype=float)
    if (initial["log_name"] != scene.metadata["source_log"]
            or initial["map_location"] != scene.metadata["map_name"]
            or str(initial["token"]) != scene.metadata["scenario_id"]
            or initial["timestamp"] != scene.metadata.get("source_start_timestamp_us")
            or not np.allclose(initial["ego2global_translation"], origin, rtol=0, atol=1e-7)):
        raise ValueError("Context source identity, time or origin does not match the scene")
    times = (np.asarray([f["timestamp"] for f in frames], dtype=np.int64)
             - initial["timestamp"]) / 1e6
    end = int(np.searchsorted(times, scene.times[-1], side="left"))
    if end >= len(frames):
        raise ValueError("Source does not cover the scene endpoint")
    if np.any(np.diff(times[start:end + 1]) > .75):
        raise ValueError("Source gap exceeds 0.75 seconds")
    result = []
    for time, frame in zip(times[start:end + 1], frames[start:end + 1], strict=True):
        if (frame["log_name"], frame["map_location"]) != (
                scene.metadata["source_log"], scene.metadata["map_name"]):
            raise ValueError("Context window crosses a source log or map boundary")
        translation = np.asarray(frame["ego2global_translation"], dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Source ego translation must be a finite xyz vector")
        yaw = yaw_from_quaternion(frame["ego2global_rotation"])
        objects = []
        annotations = frame["anns"]
        for token, kind, box in zip(annotations["track_tokens"], annotations["gt_names"],
                                    annotations["gt_boxes"], strict=True):
            box = np.asarray(box, dtype=float)
            if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                raise ValueError(f"Invalid source annotation geometry for {token}")
            if not isinstance(token, str) or not token or token == "ego":
                raise ValueError("Source track IDs must be nonempty strings distinct from ego")
            if not isinstance(kind, str) or not kind:
                raise ValueError("Source object kinds must be nonempty strings")
            pose = transform_box(box, translation, yaw)
            pose[:3] -= origin
            objects.append({"id": token, "kind": str(kind), "pose": pose.tolist(),
                            "dimensions": box[3:6].tolist()})
        signals = []
        records = frame.get("traffic_lights", [])
        if not isinstance(records, (list, tuple)):
            raise ValueError("Source traffic_lights must be a list of connector/red pairs")
        signal_ids = set()
        for record in records:
            if not isinstance(record, (list, tuple)) or len(record) != 2:
                raise ValueError("Source signal record must be a connector/red pair")
            connector, is_red = record
            if isinstance(connector, (bool, np.bool_)) or not isinstance(
                    connector, (str, int, np.integer)) or not str(connector):
                raise ValueError("Signal connector ID must be a nonempty string or integer")
            if not isinstance(is_red, (bool, np.bool_)):
                raise ValueError("Source signal state must be a literal red boolean")
            if str(connector) in signal_ids:
                raise ValueError("Duplicate connector signal in source frame")
            signal_ids.add(str(connector))
            signals.append({"connector_id": str(connector), "is_red": bool(is_red)})
        result.append({"time_seconds": float(time), "timestamp_us": int(frame["timestamp"]),
                       "objects": objects,
                       "signals": {"available": "traffic_lights" in frame, "records": signals}})

    # Source samples are independent of simulated trajectories, but must originate
    # from the same initial actor references rather than a coincidentally named log.
    initial_objects = {obj["id"]: obj for obj in result[0]["objects"]}
    initial_yaw = yaw_from_quaternion(initial["ego2global_rotation"])
    ego_pose = np.array([1.461 * np.cos(initial_yaw), 1.461 * np.sin(initial_yaw), 0,
                         initial_yaw])
    for index, actor_id in enumerate(scene.actor_ids):
        if not scene.presence_at(0)[index]:
            matches = [obj for frame in result for obj in frame["objects"] if obj["id"] == actor_id]
            if (not matches or KINDS.get(matches[0]["kind"], OTHER_OBJECT) != scene.kinds[index]
                    or not np.allclose(matches[0]["dimensions"], scene.dimensions[index], rtol=0, atol=1e-7)):
                raise ValueError(f"Late scene actor differs from source context: {actor_id}")
            continue
        if actor_id == "ego":
            reference = ego_pose
            if scene.kinds[index] != KINDS["vehicle"]:
                raise ValueError("Source ego kind differs from the scene")
        elif actor_id in initial_objects:
            reference = np.asarray(initial_objects[actor_id]["pose"])
            if KINDS.get(initial_objects[actor_id]["kind"], OTHER_OBJECT) != scene.kinds[index]:
                raise ValueError(f"Source actor kind differs from the scene: {actor_id}")
            if not np.allclose(initial_objects[actor_id]["dimensions"], scene.dimensions[index],
                               rtol=0, atol=1e-7):
                raise ValueError(f"Source actor dimensions differ from the scene: {actor_id}")
        else:
            raise ValueError(f"Scene actor is absent from source context: {actor_id}")
        difference = reference - scene.poses[0, index]
        difference[3] = np.arctan2(np.sin(difference[3]), np.cos(difference[3]))
        if not np.allclose(difference, 0, rtol=0, atol=1e-7):
            raise ValueError(f"Source actor reference differs from the scene: {actor_id}")
    return result


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported map property type: {type(value).__name__}")


def _read_map_context(scene: Scene, maps_root: Path, bounds: np.ndarray) -> tuple:
    return read_map_context(scene.metadata["map_name"], maps_root,
                            scene.metadata["origin_xyz"], bounds,
                            expected_version=scene.metadata.get("map_version"))


def read_map_context(map_name: str, maps_root: Path, origin_xyz: object,
                     bounds: np.ndarray, *, expected_version: str | None = None) -> tuple:
    """Read source map geometry without requiring a controllable simulation scene."""
    import pyogrio.raw
    import shapely
    from pyproj import Transformer

    if not isinstance(map_name, str) or Path(map_name).name != map_name or map_name in {".", ".."}:
        raise ValueError("Map name must be a single directory name")
    origin = np.asarray(origin_xyz, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if (origin.shape != (3,) or not np.isfinite(origin).all() or bounds.shape != (4,)
            or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2])
            or np.any(bounds[2:] - bounds[:2] > 2000)):
        raise ValueError("Map context requires a finite origin and bounds within 2 km")
    candidates = sorted((maps_root / map_name).glob("*/map.gpkg"))
    if len(candidates) != 1:
        raise ValueError("Require exactly one GPKG version for the context map")
    path = candidates[0]
    if expected_version is not None and path.parent.name != expected_version:
        raise ValueError("Scene and display context require the same map version")
    _, _, _, fields = pyogrio.raw.read(path, layer="meta", columns=["key", "value"])
    projection = dict(zip(fields[0], fields[1], strict=True)).get("projectedCoordSystem")
    if not projection:
        raise ValueError("Map is missing projectedCoordSystem metadata")
    global_bounds = bounds + np.tile(origin[:2], 2)
    region = shapely.box(*global_bounds)
    features, layers = [], {}
    for layer, geometry_type in pyogrio.list_layers(path):
        if geometry_type is None:
            continue
        info = pyogrio.read_info(path, layer=layer)
        to_source = Transformer.from_crs(projection, info["crs"], always_xy=True)
        to_metric = Transformer.from_crs(info["crs"], projection, always_xy=True)
        bbox = to_source.transform_bounds(*global_bounds, densify_pts=21)
        meta, fids, wkb, arrays = pyogrio.raw.read(path, layer=layer, bbox=bbox, return_fids=True)
        geometries = shapely.transform(shapely.from_wkb(wkb), to_metric.transform, interleaved=False)
        count = 0
        for index, (fid, geometry) in enumerate(zip(fids, geometries, strict=True)):
            if geometry is None or geometry.is_empty:
                continue
            geometry = shapely.intersection(shapely.make_valid(geometry), region)
            if geometry.is_empty:
                continue
            local = shapely.transform(geometry, lambda xy: xy - origin[:2])
            properties = {name: _json_value(values[index])
                          for name, values in zip(meta["fields"], arrays, strict=True)}
            features.append({"id": f"{layer}:{int(fid)}", "layer": str(layer),
                             "geometry": json.loads(shapely.to_geojson(local)),
                             "properties": properties})
            count += 1
        layers[str(layer)] = {"availability": "available" if count else "loaded_empty",
                              "feature_count": count}
    # Missing geometry is distinct from a loaded layer empty within this ROI.
    # In particular, stop-sign-related stop areas are not a physical sign inventory.
    for layer in SUPPORTED_MAP_LAYERS:
        layers.setdefault(layer, {"availability": "unavailable", "feature_count": 0})
    stat = path.stat()
    provenance = {"path": str(path), "version": path.parent.name, "projected_crs": projection,
                  "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                  "geometry_model": "planar_xy_clipped_to_diagnostic_roi"}
    return features, layers, provenance


def compile_context(
    scene: Scene, log_file: Path, maps_root: Path, *, trusted: bool = False, margin: float = 40,
) -> SceneContext:
    """Attach complete source display context without modifying the simulation scene."""
    scene.validate()
    if isinstance(margin, bool) or not np.isfinite(margin) or not 20 <= margin <= 500:
        raise ValueError("Context margin must be between 20 and 500 metres")
    source = load_log(log_file, trusted=trusted)
    frames = _source_frames(scene, source)
    points = [scene.poses[..., :2].reshape(-1, 2)]
    points.extend(np.asarray([obj["pose"][:2] for obj in frame["objects"]], dtype=float)
                  for frame in frames if frame["objects"])
    positions = np.concatenate(points)
    lower, upper = positions.min(axis=0) - margin, positions.max(axis=0) + margin
    if np.any(upper - lower > 2000):
        raise ValueError("Display context map extent exceeds 2 km")
    bounds = np.r_[lower, upper]
    features, layers, map_provenance = _read_map_context(scene, maps_root, bounds)
    metadata = {
        "schema_version": 1, "scene_fingerprint": scene_fingerprint(scene),
        "scenario_id": scene.metadata["scenario_id"], "source_log": scene.metadata["source_log"],
        "map_name": scene.metadata["map_name"], "map_version": scene.metadata["map_version"],
        "origin_xyz": scene.metadata["origin_xyz"], "scene_times": scene.times.tolist(),
        "reference_policy": "previous_sample", "max_sample_age_seconds": .75,
        "map_bounds_local": bounds.tolist(), "source_start_frame": scene.metadata["source_start_frame"],
        "source_file": str(log_file), "source_sha256": _sha256(log_file),
        "map_provenance": map_provenance,
        "map_role": "diagnostic_only", "map_extent_source": "all_source_objects_and_scene_positions",
        "signal_semantics": "source_connector_red_boolean_only",
    }
    context = SceneContext(metadata, frames, features, layers)
    context.validate_scene(scene)
    return context
