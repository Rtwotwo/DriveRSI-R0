"""Metric scene format. NPZ files contain arrays and JSON, never pickled objects."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCENE_VERSION = 1
VEHICLE, PEDESTRIAN, CYCLIST, OTHER_OBJECT = 1, 2, 3, 10
MAX_WORLD_ACTORS = 1024
MAX_CONTROLLED_ACTORS = 32
LANE, ROAD_EDGE = 4, 6


@dataclass
class Scene:
    metadata: dict
    actor_ids: tuple[str, ...]
    kinds: np.ndarray
    dimensions: np.ndarray  # [actor, length/width/height], metres
    times: np.ndarray  # seconds from the initial frame
    poses: np.ndarray  # [time, actor, x/y/z/yaw], vehicle centres
    velocities: np.ndarray  # [time, actor, vx/vy/vz]
    controlled: np.ndarray
    road_points: np.ndarray  # local metric map axes, translated by metadata.origin_xyz
    road_offsets: np.ndarray
    road_types: np.ndarray

    presence: np.ndarray | None = None  # [time, actor]; None means legacy all-present

    def presence_at(self, tick: int) -> np.ndarray:
        if not 0 <= tick < len(self.times):
            raise ValueError("Presence tick outside scene")
        return (np.ones(len(self.actor_ids), dtype=bool) if self.presence is None
                else self.presence[tick].copy())

    def validate(self) -> None:
        n, t = len(self.actor_ids), len(self.times)
        if self.metadata.get("schema_version") != SCENE_VERSION:
            raise ValueError("Unsupported scene schema_version")
        for key in ("scenario_id", "source_log", "split", "map_name", "origin_xyz"):
            if key not in self.metadata:
                raise ValueError(f"Missing scene metadata: {key}")
        if not 1 <= n <= MAX_WORLD_ACTORS or t < 2 or len(set(self.actor_ids)) != n:
            raise ValueError(f"Require 1–{MAX_WORLD_ACTORS} unique actor IDs and at least two time samples")
        expected = {
            "kinds": (n,), "dimensions": (n, 3), "poses": (t, n, 4),
            "velocities": (t, n, 3), "controlled": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite {shape}")
        if not np.isfinite(self.times).all() or self.times[0] != 0:
            raise ValueError("Scene times must be finite and start at zero")
        if np.any(np.diff(self.times) <= 0) or np.any(self.dimensions <= 0):
            raise ValueError("Timestamps must increase and dimensions must be positive")
        if not np.isin(self.kinds, [VEHICLE, PEDESTRIAN, CYCLIST, OTHER_OBJECT]).all():
            raise ValueError("Unsupported actor kind")
        if self.controlled.dtype != np.bool_ or not self.controlled.any():
            raise ValueError("controlled must be a nonempty boolean control mask")
        if np.any(self.controlled & (self.kinds != VEHICLE)):
            raise ValueError("This scene version supports vehicle control only")
        if self.controlled.sum() > MAX_CONTROLLED_ACTORS:
            raise ValueError("At most32 controlled actors")
        if self.presence is not None:
            if self.presence.shape != (t, n) or self.presence.dtype != np.bool_:
                raise ValueError("Invalid presence mask")
            if not self.presence[:, self.controlled].all():
                raise ValueError("Controlled actors remain present throughout the episode")
        from driversi.data.participants.background import validate_background

        validate_background(self)
        from driversi.data.participants.fleet import validate_vehicle_fleet

        validate_vehicle_fleet(self)
        goals = self.metadata.get("task_goals_xy")
        if goals is not None:
            controlled_ids = {actor for actor, active in zip(self.actor_ids, self.controlled) if active}
            if set(goals) != controlled_ids:
                raise ValueError("Explicit task goals must match every controlled actor")
            if any(np.asarray(goal).shape != (2,) or not np.isfinite(goal).all() for goal in goals.values()):
                raise ValueError("Task goals must contain finite xy pairs")
        from driversi.contracts.driving_task import tasks_from_metadata

        tasks_from_metadata(self.metadata, tuple(a for a, c in zip(self.actor_ids, self.controlled) if c))
        if self.road_points.ndim != 2 or self.road_points.shape[1] != 3:
            raise ValueError("road_points must have shape [point, 3]")
        if not np.isfinite(self.road_points).all():
            raise ValueError("Non-finite map geometry")
        if self.road_offsets.shape != (len(self.road_types) + 1,):
            raise ValueError("Invalid road offsets shape")
        if self.road_offsets[0] != 0 or self.road_offsets[-1] != len(self.road_points):
            raise ValueError("Road offsets must cover exactly the stored points")
        if np.any(np.diff(self.road_offsets) < 2):
            raise ValueError("Every polyline needs at least two points")
        if not np.isin(self.road_types, [LANE, ROAD_EDGE]).all():
            raise ValueError("Unsupported road type")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: value for name, value in vars(self).items() if isinstance(value, np.ndarray)}
        with path.open("wb") as stream:
            np.savez_compressed(
                stream, metadata=json.dumps(self.metadata, allow_nan=False),
                actor_ids=np.asarray(self.actor_ids), **arrays,
            )

    @classmethod
    def load(cls, path: Path) -> "Scene":
        with np.load(path, allow_pickle=False) as bundle:
            scene = cls(
                metadata=json.loads(str(bundle["metadata"])),
                actor_ids=tuple(str(x) for x in bundle["actor_ids"]),
                **{name: bundle[name].copy() for name in cls.__dataclass_fields__
                   if name not in {"metadata", "actor_ids"} and name in bundle.files},
            )
        scene.validate()
        return scene
