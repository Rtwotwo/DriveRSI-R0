"""Source-only scene context; never part of policy observations or dynamics."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene


def scene_fingerprint(scene: Scene) -> str:
    """Bind context to the complete metric scene, independent of file container."""
    scene.validate()
    digest = hashlib.sha256()
    digest.update(json.dumps(scene.metadata, sort_keys=True, allow_nan=False).encode())
    digest.update(json.dumps(scene.actor_ids).encode())
    for name in Scene.__dataclass_fields__:
        value = getattr(scene, name)
        if isinstance(value, np.ndarray):
            digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"Invalid context {name}")
    return array


def _geometry(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("Context geometry must be a GeoJSON object")
    kind = value.get("type")
    if kind == "GeometryCollection":
        children = value.get("geometries")
        if not isinstance(children, list) or not children:
            raise ValueError("Empty or invalid context geometry collection")
        for child in children:
            _geometry(child)
        return
    if kind == "Point":
        _vector(value.get("coordinates"), 2, "point")
        return
    coordinates = value.get("coordinates")
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise ValueError("Empty context geometry")
    if kind == "LineString":
        points = np.asarray(coordinates)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or points.dtype.kind not in "iuf" or not np.isfinite(points).all()):
            raise ValueError("Invalid context line")
    elif kind == "Polygon":
        for ring in coordinates:
            _geometry({"type": "LineString", "coordinates": ring})
            if len(ring) < 4 or not np.array_equal(ring[0], ring[-1]):
                raise ValueError("Context polygon rings must be closed")
    elif kind in {"MultiPoint", "MultiLineString", "MultiPolygon"}:
        child_kind = {"MultiPoint": "Point", "MultiLineString": "LineString",
                      "MultiPolygon": "Polygon"}[kind]
        for child in coordinates:
            _geometry({"type": child_kind, "coordinates": child})
    else:
        raise ValueError("Unsupported context geometry type")


@dataclass
class SceneContext:
    metadata: dict
    frames: list[dict]
    map_features: list[dict]
    map_layers: dict

    def validate(self) -> None:
        meta = self.metadata
        if (not isinstance(meta, dict) or type(meta.get("schema_version")) is not int
                or meta["schema_version"] != 1):
            raise ValueError("Unsupported context schema_version")
        fingerprint = meta.get("scene_fingerprint", "")
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("Invalid context scene_fingerprint")
        for name in ("scenario_id", "source_log", "map_name"):
            if not isinstance(meta.get(name), str) or not meta[name]:
                raise ValueError(f"Missing context identity: {name}")
        _vector(meta.get("origin_xyz"), 3, "origin")
        times = np.asarray(meta.get("scene_times"))
        if (times.ndim != 1 or len(times) < 2 or times.dtype.kind not in "iuf"
                or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)):
            raise ValueError("Invalid context scene_times")
        if meta.get("reference_policy") != "previous_sample" or meta.get("max_sample_age_seconds") != .75:
            raise ValueError("Unsupported context sampling policy")
        if "map_bounds_local" in meta:
            bounds = _vector(meta["map_bounds_local"], 4, "map bounds")
            if np.any(bounds[2:] <= bounds[:2]):
                raise ValueError("Invalid context map bounds")
        if not isinstance(self.frames, list) or not self.frames:
            raise ValueError("Context requires source frames")
        previous_time, previous_stamp, first_stamp = -1., None, None
        for frame in self.frames:
            if not isinstance(frame, dict):
                raise ValueError("Invalid context frame")
            time = frame.get("time_seconds")
            stamp = frame.get("timestamp_us")
            if (not isinstance(time, (int, float)) or isinstance(time, bool)
                    or not np.isfinite(time) or time < 0 or time <= previous_time
                    or type(stamp) is not int or (previous_stamp is not None and stamp <= previous_stamp)):
                raise ValueError("Invalid context source time")
            first_stamp = stamp if first_stamp is None else first_stamp
            if not np.isclose(time, (stamp-first_stamp)/1e6, atol=1e-9, rtol=0):
                raise ValueError("Context source times disagree with timestamps")
            previous_time, previous_stamp = time, stamp
            objects = frame.get("objects")
            if not isinstance(objects, list):
                raise ValueError("Context objects must be a list")
            ids = set()
            for obj in objects:
                if (not isinstance(obj, dict) or not isinstance(obj.get("id"), str)
                        or not obj["id"] or obj["id"] in ids
                        or not isinstance(obj.get("kind"), str) or not obj["kind"]):
                    raise ValueError("Invalid or duplicate context object identity")
                ids.add(obj["id"])
                _vector(obj.get("pose"), 4, "object pose")
                if np.any(_vector(obj.get("dimensions"), 3, "object dimensions") <= 0):
                    raise ValueError("Context dimensions must be positive")
            signals = frame.get("signals")
            if (not isinstance(signals, dict) or type(signals.get("available")) is not bool
                    or not isinstance(signals.get("records"), list)):
                raise ValueError("Invalid context signals")
            if not signals["available"] and signals["records"]:
                raise ValueError("Unavailable signals cannot contain records")
            signal_ids = set()
            for record in signals["records"]:
                if (not isinstance(record, dict) or not isinstance(record.get("connector_id"), str)
                        or not record["connector_id"] or record["connector_id"] in signal_ids
                        or type(record.get("is_red")) is not bool):
                    raise ValueError("Invalid or duplicate context signal record")
                signal_ids.add(record["connector_id"])
        if self.frames[0]["time_seconds"] != 0 or previous_time < times[-1]:
            raise ValueError("Context source frames do not bracket scene times")
        if not isinstance(self.map_features, list) or not isinstance(self.map_layers, dict):
            raise ValueError("Invalid context map")
        feature_ids, counts = set(), {}
        for feature in self.map_features:
            if (not isinstance(feature, dict) or not isinstance(feature.get("id"), str)
                    or not feature["id"] or feature["id"] in feature_ids
                    or not isinstance(feature.get("layer"), str)
                    or feature.get("layer") not in self.map_layers
                    or not isinstance(feature.get("properties"), dict)):
                raise ValueError("Invalid or duplicate context map feature")
            feature_ids.add(feature["id"])
            counts[feature["layer"]] = counts.get(feature["layer"], 0) + 1
            _geometry(feature.get("geometry"))
        for layer, status in self.map_layers.items():
            if (not isinstance(layer, str) or not layer or not isinstance(status, dict)
                    or type(status.get("feature_count")) is not int
                    or status["feature_count"] != counts.get(layer, 0)
                    or status.get("availability") not in {"available", "loaded_empty", "unavailable"}):
                raise ValueError("Invalid context map layer status/count")
            if (status["availability"] == "available") != (status["feature_count"] > 0):
                raise ValueError("Context layer availability contradicts its features")
        try:
            json.dumps(vars(self), allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("Context must contain finite JSON values") from error

    def validate_scene(self, scene: Scene) -> None:
        self.validate()
        if self.metadata["scene_fingerprint"] != scene_fingerprint(scene):
            raise ValueError("Context scene_fingerprint does not match scene")
        for key in ("scenario_id", "source_log", "map_name", "map_version", "origin_xyz"):
            if self.metadata.get(key) != scene.metadata.get(key):
                raise ValueError(f"Context {key} does not match scene")
        if not np.array_equal(self.metadata["scene_times"], scene.times):
            raise ValueError("Context time axis does not match scene")
        if ("source_start_timestamp_us" in scene.metadata
                and self.frames[0]["timestamp_us"] != scene.metadata["source_start_timestamp_us"]):
            raise ValueError("Context source start timestamp does not match scene")

    def with_driving_tasks(self, before: Scene, after: Scene) -> "SceneContext":
        """Rebind source context only across a verified task-only scene change."""
        self.validate_scene(before)
        after.validate()
        if "driving_tasks" not in after.metadata:
            raise ValueError("Rebinding requires declared driving_tasks")
        comparable = copy.deepcopy(after)
        for key in ("driving_tasks", "task_goals_xy"):
            if key in before.metadata:
                comparable.metadata[key] = copy.deepcopy(before.metadata[key])
            else:
                comparable.metadata.pop(key, None)
        if scene_fingerprint(comparable) != scene_fingerprint(before):
            raise ValueError("Context rebinding permits only driving task changes")
        result = copy.deepcopy(self)
        result.metadata["scene_fingerprint"] = scene_fingerprint(after)
        result.validate_scene(after)
        return result

    def to_dict(self) -> dict:
        self.validate()
        return copy.deepcopy(vars(self))

    def save(self, path: Path) -> None:
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")

    @classmethod
    def load(cls, path: Path) -> "SceneContext":
        def reject_constant(value):
            raise ValueError(f"Non-finite context JSON: {value}")

        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if not isinstance(data, dict) or set(data) != {"metadata", "frames", "map_features", "map_layers"}:
            raise ValueError("Invalid SceneContext fields")
        context = cls(**data)
        context.validate()
        return context
