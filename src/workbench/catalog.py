"""Lazy views of compiled reference scenes; no policy or training execution."""
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np

from driversi.contracts.scene import Scene


class SceneCatalog:
    def __init__(self, suite: Path, cache_size: int = 4):
        self.suite = suite.resolve(strict=True)
        self.root = self.suite.parent
        if not 1 <= cache_size <= 32:
            raise ValueError("cache_size must be between 1 and 32")
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._lock = RLock()
        self._rows = {}
        self._paths = {}
        data = json.loads(self.suite.read_text())
        if data.get("official_test") is not False:
            raise ValueError("Workbench requires an explicitly non-official engineering suite")
        groups = {}
        for split in ("train", "evaluation"):
            groups[split] = set()
            for item in data[split]:
                path = Path(item["path"])
                if not path.is_absolute():
                    path = self.root / path
                path = path.resolve()
                if not path.is_relative_to(self.root) or path.suffix != ".npz":
                    raise ValueError("Scene paths must be NPZ files within the suite directory")
                sid = path.stem
                if sid in self._rows:
                    raise ValueError("Duplicate scene identity")
                group = item["source_group"]
                if not isinstance(group, str) or not group:
                    raise ValueError("Missing source-log group")
                groups[split].add(group)
                self._paths[sid] = path
                self._rows[sid] = {
                    "id": sid, "split": split, "city": item["city"], "source_group": group,
                    "controlled_count": int(item["actors"]), "duration": float(item["seconds"]),
                }
        if groups["train"] & groups["evaluation"]:
            raise ValueError("Declared source-log groups overlap")
        self.limits = data.get("limits", {})

    def index(self, excluded_groups=()):
        return {
            "schema_version": 1, "role": "engineering_exposed",
            "replay_kind": "compiled_source_reference", "official_score": None,
            "not_rsi_holdout": True, "scenes": [r for r in self._rows.values()
                if r["source_group"] not in excluded_groups],
            "limits": self.limits,
        }

    def scene(self, sid, excluded_groups=()):
        with self._lock:
            if sid not in self._rows or self._rows[sid]["source_group"] in excluded_groups:
                raise KeyError(sid)
            if sid in self._cache:
                self._cache.move_to_end(sid)
                return self._cache[sid]
            scene = Scene.load(self._paths[sid])
            value = self._serialize(scene, self._rows[sid])
            self._cache[sid] = value
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return value

    @staticmethod
    def _serialize(scene, row):
        meta = scene.metadata
        metadata_keys = (
            "scenario_id", "source_log", "source_dataset", "source_partition",
            "source_interval_us", "source_timing", "horizon_plan", "population_policy",
            "limitations", "excluded_incomplete_tracks", "excluded_over_capacity",
            "excluded_unsupported_initial", "map_layers", "map_complete",
            "road_edge_model", "map_elevation_model", "heading_audit",
        )
        types = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
        roads = []
        for i, kind in enumerate(scene.road_types):
            a, b = int(scene.road_offsets[i]), int(scene.road_offsets[i + 1])
            roads.append({"kind": int(kind), "points": scene.road_points[a:b, :2].tolist()})
        return {
            **row, "replay_kind": "compiled_source_reference", "trainable": False,
            "metadata": {k: meta[k] for k in metadata_keys if k in meta},
            "times": scene.times.tolist(),
            "actors": [
                {"id": aid, "kind": types.get(int(scene.kinds[i]), "unknown"),
                 "dimensions": scene.dimensions[i].tolist(),
                 "control_eligible": bool(scene.controlled[i])}
                for i, aid in enumerate(scene.actor_ids)
            ],
            "poses": scene.poses.tolist(),
            "speeds": np.linalg.norm(scene.velocities[..., :2], axis=-1).tolist(),
            "roads": roads,
            "evidence": {
                "rgb": "not_bound", "policy_rollout": "not_bound", "paired_branch": "not_bound",
                "rsi_improvement": "not_evaluated",
                "population": "compiled_subset_not_all_source_objects",
                "signals": "source_metadata_only_not_bound_to_display_geometry",
                "road_edges": "clipped_support_boundary_not_certified_physical_curb",
            },
        }
