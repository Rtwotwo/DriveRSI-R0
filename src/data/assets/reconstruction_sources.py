"""Resolve measured reconstruction inputs from separately declared sensor roots.

Finding cameras in one root does not prove that LiDAR is absent in another.
This resolver records absence and conflicting replicas; it never fabricates data.
"""
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.data.assets.sensors import _asset_path


def resolve_reconstruction_sensor(relative_path, *, modality, sensor_roots):
    if modality not in {"camera", "lidar"}:
        raise ValueError("Explicit camera or lidar modality required")
    roots = sensor_roots.get(modality)
    if not isinstance(roots, (list, tuple)) or not roots:
        raise ValueError("Declare at least one root for the requested modality")
    checked, found = [], []
    seen = set()
    for value in roots:
        root = Path(value).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Sensor root must be a directory")
        relative, path = _asset_path(root, relative_path, modality)
        if path in seen:
            continue
        seen.add(path)
        checked.append(str(path))
        if path.is_file():
            found.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
    if len({row["sha256"] for row in found}) > 1:
        raise ValueError("Conflicting sensor replicas; choose an explicit verified source")
    return {
        "modality": modality,
        "relative_path": relative,
        "status": "resolved" if found else "missing",
        "selected": found[0] if found else None,
        "verified_replicas": found,
        "checked_paths": checked,
        "imputed": False,
    }
