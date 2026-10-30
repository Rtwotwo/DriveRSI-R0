"""Explicit time, actor order, transition validity and episode-boundary contracts."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MetricObservation:
    """metric-v1, current scene axes in SI units; no future/reference trajectory.

    ego: x,y,yaw,vx,vy,length,width,a_long,a_lat,steering.
    neighbors: x,y,yaw,vx,vy,length,width. Invisible entries are zero/blank.
    No occlusion; visibility is centre distance <= 50 m, excluding self.
    """

    ego: np.ndarray
    neighbors: np.ndarray
    visible: np.ndarray
    neighbor_ids: tuple[tuple[str, ...], ...]

    def select(self, indices: list[int]) -> "MetricObservation":
        return MetricObservation(self.ego[indices].copy(), self.neighbors[indices].copy(),
                                 self.visible[indices].copy(),
                                 tuple(self.neighbor_ids[i] for i in indices))


@dataclass(frozen=True)
class Observation:
    actor_ids: tuple[str, ...]  # controlled actors, stable for the whole episode
    features: np.ndarray  # [controlled_actor, 1124], pinned PufferDrive encoding
    alive: np.ndarray
    step: int
    metric: MetricObservation | None = None


@dataclass(frozen=True)
class Transition:
    observation: Observation  # next / final observation, never an auto-reset state
    rewards: np.ndarray
    valid: np.ndarray  # alive before this joint action
    terminated: np.ndarray  # newly reached an absorbing actor state
    truncated: np.ndarray  # still alive when the time budget expires
    episode_done: bool
    events: dict[str, np.ndarray]
    traffic_controls: tuple[dict, ...] = ()  # Actual nonlearning traffic proposals for this transition


@dataclass(frozen=True)
class Snapshot:
    scene_key: str
    build_id: str
    native_state: bytes
    alive: tuple[bool, ...]
    episode_done: bool
    runtime: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema_version": 1, "scene_key": self.scene_key, "build_id": self.build_id,
                    "alive": self.alive, "episode_done": self.episode_done, "runtime": self.runtime}
        with path.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata),
                                native_state=np.frombuffer(self.native_state, np.uint8))

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.pop("schema_version") != 1:
                raise ValueError("Unsupported snapshot schema")
            metadata["alive"] = tuple(metadata["alive"])
            return cls(native_state=data["native_state"].tobytes(), **metadata)
