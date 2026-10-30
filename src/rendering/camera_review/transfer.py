"""Display-only map preparation and lossless static transfer compression."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

from .gallery import dump


def compressed_file(path):
    path = Path(path)
    raw = path.read_bytes()
    packed = path.with_name(path.name + ".gz")
    packed.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {
        "path": packed.name,
        "bytes": packed.stat().st_size,
        "sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
    }


def display_map(data, tolerance=0.02):
    """Keep every feature and all attributes; only simplify its display geometry."""
    from shapely.geometry import mapping, shape

    if not 0 <= tolerance <= 0.02:
        raise ValueError("Display tolerance must be between 0 and 2 cm")
    result = copy.deepcopy(data)
    maximum = 0.0
    for feature in result["features"]:
        source = shape(feature["geometry"])
        simplified = source.simplify(tolerance, preserve_topology=True)
        if simplified.is_empty or simplified.geom_type != source.geom_type:
            simplified = source
        error = source.hausdorff_distance(simplified)
        if error > tolerance + 1e-8:
            simplified = source
            error = 0.0
        maximum = max(maximum, error)
        feature["geometry"] = mapping(simplified)
    result["display_only"] = True
    result["maximum_hausdorff_error_m"] = maximum
    result["source_feature_count"] = len(data["features"])
    result["description"] = "Display geometry only; original map remains in map.json."
    return result


def optimize(root):
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    for camera in scene["cameras"].values():
        for chunk in camera["chunks"]:
            packed = compressed_file(root / chunk["path"])
            packed["path"] = chunk["path"] + ".gz"
            chunk["gzip"] = packed
    scene["map"]["gzip"] = compressed_file(root / scene["map"]["path"])
    data = json.loads((root / scene["map"]["path"]).read_text())
    scene["display_map"] = dump(root / "map-display.json", display_map(data))
    scene["display_map"]["gzip"] = compressed_file(root / "map-display.json")
    scene_ref = dump(root / "scene.json", scene)
    scene_ref["gzip"] = compressed_file(root / "scene.json")
    dump(root / "bootstrap.json", {"schema_version": 1, "scene": scene_ref})
    return {
        "structured_initial_bytes": (root / "bootstrap.json").stat().st_size
        + scene_ref["gzip"]["bytes"]
        + scene["display_map"]["gzip"]["bytes"],
        "feature_count": len(data["features"]),
    }
