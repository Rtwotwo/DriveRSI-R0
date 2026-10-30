"""Bounded decoded scene cache; paths and small headers do not preload scene arrays."""
import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from driversi.contracts.driving_task import spec_options, tasks_from_metadata
from driversi.contracts.scene import MAX_WORLD_ACTORS, Scene


class SceneCache:
    def __init__(self, capacity=32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Positive scene cache capacity required")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.loads = 0

    def get(self, path):
        key = str(Path(path).resolve())
        if key not in self.entries:
            value = Scene.load(Path(key))
            self.loads += 1
            self.entries[key] = value
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return self.entries[key]


class LazyScenes(Sequence):
    def __init__(self, paths, capacity=32):
        self.paths = tuple(paths)
        self.cache = SceneCache(capacity)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        return self.cache.get(self.paths[index])


def scene_header(path):
    """Read source and actor/time metadata; full validation still occurs at world creation."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        ids = archive["actor_ids"].tolist()
        controlled = archive["controlled"]
        times = archive["times"]
        if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
                or any(not isinstance(metadata.get(k), str) or not metadata[k]
                       for k in ("source_log", "split", "scenario_id", "map_name"))
                or not 1 <= len(ids) <= MAX_WORLD_ACTORS or controlled.sum() > 32 or len(set(ids)) != len(ids)
                or controlled.dtype != bool or controlled.shape != (len(ids),)
                or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or times[0] != 0 or not np.allclose(np.diff(times), .1, atol=1e-7, rtol=0)):
            raise ValueError(f"Invalid compiled scene header: {path}")
        tasks_from_metadata(metadata, [a for a, c in zip(ids, controlled) if c])
        return {"spec_options": spec_options(metadata), "metadata": {k: metadata[k] for k in ("source_log", "split", "scenario_id", "map_name")}, "actor_ids": ids,
                "controlled_ids": [a for a, c in zip(ids, controlled) if c], "max_steps": len(times) - 1,
                "dynamic_population": "presence" in archive.files}


def source_inventory(paths, progress_weight):
    from driversi.contracts.learning import ClosedLoopSpec

    headers = [scene_header(path) for path in paths]
    return {"schema_version": 1, "kind": "rollout-scene-inventory", "scene_paths": list(paths),
            "actor_ids": [h["actor_ids"] for h in headers],
            "controlled_ids": [h["controlled_ids"] for h in headers],
            "environment_specs": [ClosedLoopSpec(h["max_steps"], progress_weight, dynamic_population=h["dynamic_population"], **h["spec_options"]).to_dict() for h in headers]}


def rollout_sources(metadata, run, cache):
    """New runs reference one in-root inventory; old embedded metadata still loads."""
    reference = metadata.get("source_manifest")
    if reference is None:
        return metadata
    if not isinstance(reference, str) or not reference:
        raise ValueError("Invalid rollout source_manifest")
    root = Path(run).resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Rollout source manifest outside run")
    if path not in cache:
        value = json.loads(path.read_text(encoding="utf-8"))
        fields = ("scene_paths", "actor_ids", "controlled_ids", "environment_specs")
        if (not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("kind") != "rollout-scene-inventory"
                or any(not isinstance(value.get(k), list) for k in fields)
                or not value["scene_paths"]
                or any(len(value[k]) != len(value["scene_paths"]) for k in fields)):
            raise ValueError("Invalid rollout scene inventory")
        cache[path] = {k: value[k] for k in fields}
    source = cache[path]
    for key in source:
        if key in metadata and metadata[key] != source[key]:
            raise ValueError("Embedded and referenced rollout sources disagree")
    return {**metadata, **source}
