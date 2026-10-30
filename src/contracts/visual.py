"""Versioned primitive-camera contracts; actor observations exclude world truth."""

from dataclasses import dataclass

import numpy as np

VISUAL_OBSERVATION_VERSION = "visual-rgb-history-v2"


def frozen_array(value, dtype=None) -> np.ndarray:
    """Own the buffer; callers cannot change a frame through their source array."""
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole optical axes right/down/forward -> vehicle center forward/left/up.

    Primitive cameras have zero distortion. Real calibration is audited separately.
    """

    camera_id: str
    width: int
    height: int
    intrinsic: np.ndarray
    camera_to_body: np.ndarray
    near: float = .2
    far: float = 80.

    def __post_init__(self):
        k, transform = np.asarray(self.intrinsic), np.asarray(self.camera_to_body)
        if (not self.camera_id or not isinstance(self.width, int) or not isinstance(self.height, int)
                or not 16 <= self.width <= 1024 or not 16 <= self.height <= 1024):
            raise ValueError("Camera requires an ID and integer dimensions in [16,1024]")
        if (k.shape != (3, 3) or transform.shape != (4, 4)
                or not np.isfinite(k).all() or not np.isfinite(transform).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
                or not np.isfinite([self.near, self.far]).all() or not 0 < self.near < self.far):
            raise ValueError("Invalid pinhole calibration or rigid camera transform")
        object.__setattr__(self, "intrinsic", frozen_array(k, float))
        object.__setattr__(self, "camera_to_body", frozen_array(transform, float))

    @classmethod
    def front(cls, width: int = 160, height: int = 96) -> "CameraCalibration":
        fx = width / (2 * np.tan(np.deg2rad(80) / 2))
        transform = np.eye(4)
        transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        transform[:3, 3] = [1.5, 0, 1.6]
        return cls("front", width, height,
                   np.asarray([[fx, 0, width/2], [0, fx, height/2], [0, 0, 1]]), transform)

    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "width": self.width, "height": self.height,
                "intrinsic": self.intrinsic.tolist(), "camera_to_body": self.camera_to_body.tolist(),
                "near": self.near, "far": self.far}


@dataclass(frozen=True)
class CameraRig:
    cameras: tuple[CameraCalibration, ...]
    version: str = "virtual-planar-rig-v1"

    def __post_init__(self):
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if (not self.cameras or not self.version
                or len({c.camera_id for c in self.cameras}) != len(self.cameras)
                or len({(c.width, c.height) for c in self.cameras}) != 1):
            raise ValueError("Rig requires unique cameras of one image size")

    def to_dict(self) -> dict:
        return {"version": self.version, "cameras": [c.to_dict() for c in self.cameras]}


@dataclass(frozen=True)
class WorldFrame:
    """Frozen physical actors only, scene order; poses are xyz/yaw in map axes."""

    actor_ids: tuple[str, ...]
    poses: np.ndarray
    tick: int
    episode_id: str
    branch_id: str
    generation: int
    dt: float = .1

    def __post_init__(self):
        if (not self.actor_ids or len(set(self.actor_ids)) != len(self.actor_ids)
                or np.shape(self.poses) != (len(self.actor_ids), 4)
                or not np.isfinite(self.poses).all()
                or not isinstance(self.tick, int) or self.tick < 0
                or self.generation < 0 or not self.episode_id or not self.branch_id
                or not np.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("Invalid frozen world frame")
        object.__setattr__(self, "poses", frozen_array(self.poses, float))

    @property
    def sim_time(self) -> float:
        return self.tick * self.dt


@dataclass(frozen=True)
class SensorBatch:
    """Renderer output [actor,camera,height,width,...]; depth/instance are QA only.

    Depth is optical-axis metres, +inf for sky. Instance 0 is static/background;
    a physical actor's instance is its scene index + 1. Actor input is RGB only.
    """

    actor_ids: tuple[str, ...]
    camera_ids: tuple[str, ...]
    rgb: np.ndarray
    depth: np.ndarray
    instances: np.ndarray
    valid: np.ndarray
    tick: int
    episode_id: str
    branch_id: str
    generation: int
    renderer_version: str = "primitive-cpu-v1"
    modality: str = "primitive_rgb"

    def __post_init__(self):
        shape = np.shape(self.rgb)
        if (len(shape) != 5 or shape[:2] != (len(self.actor_ids), len(self.camera_ids))
                or shape[-1] != 3 or np.asarray(self.rgb).dtype != np.uint8
                or np.shape(self.depth) != shape[:-1] or np.shape(self.instances) != shape[:-1]
                or np.shape(self.valid) != shape[:2]
                or np.isnan(self.depth).any() or np.any(np.asarray(self.depth) <= 0)):
            raise ValueError("Invalid rendered sensor batch")
        for name, dtype in (("rgb", np.uint8), ("depth", np.float32),
                            ("instances", np.int32), ("valid", bool)):
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))


@dataclass(frozen=True)
class VisualObservation:
    """Actor-only input; no instance labels, depth, neighbors or reference futures.

    rgb [actor,history,camera,height,width,3], history_valid [actor,history,camera],
    history_ticks [actor,history] (-1 means absent). Telemetry is own speed,
    longitudinal/lateral acceleration and steering. Navigation is assigned goal
    in own forward/left axes; it is task input, never a neighbor's hidden goal.
    Camera IDs align the image camera dimension with K and optical-to-body SE(3).
    """

    actor_ids: tuple[str, ...]
    rgb: np.ndarray
    history_valid: np.ndarray
    history_ticks: np.ndarray
    telemetry: np.ndarray
    navigation: np.ndarray
    alive: np.ndarray
    intrinsic: np.ndarray
    camera_ids: tuple[str, ...]
    camera_to_body: np.ndarray
    tick: int
    dt: float
    version: str = VISUAL_OBSERVATION_VERSION

    def __post_init__(self):
        object.__setattr__(self, "camera_ids", tuple(self.camera_ids))
        n = len(self.actor_ids)
        shape = np.shape(self.rgb)
        if (len(shape) != 6 or shape[0] != n or shape[-1] != 3
                or np.asarray(self.rgb).dtype != np.uint8
                or np.shape(self.history_valid) != shape[:3]
                or np.shape(self.history_ticks) != shape[:2]
                or np.shape(self.telemetry) != (n, 4) or np.shape(self.navigation) != (n, 2)
                or np.shape(self.alive) != (n,) or np.shape(self.intrinsic) != (shape[2], 3, 3)
                or not self.camera_ids or len(self.camera_ids) != shape[2]
                or len(set(self.camera_ids)) != len(self.camera_ids)
                or not all(isinstance(camera, str) and camera for camera in self.camera_ids)
                or np.shape(self.camera_to_body) != (shape[2], 4, 4)
                or not np.isfinite(self.intrinsic).all() or not np.isfinite(self.camera_to_body).all()
                or not np.isfinite(self.telemetry).all() or not np.isfinite(self.navigation).all()):
            raise ValueError("Invalid actor visual observation")
        for index, camera in enumerate(self.camera_ids):
            CameraCalibration(camera, shape[4], shape[3], self.intrinsic[index], self.camera_to_body[index])
        for name in ("rgb", "history_valid", "history_ticks", "telemetry", "navigation", "alive", "intrinsic",
                     "camera_to_body"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))


@dataclass(frozen=True)
class VisualPolicyResult:
    actions: np.ndarray  # one jerk_discrete_12 action per actor
    memories: np.ndarray  # [actor,memory]; all persistent model state is explicit
    diagnostics: tuple[dict, ...] = ()

    def validate(self, count: int) -> None:
        if (np.shape(self.actions) != (count,) or np.asarray(self.actions).dtype.kind not in "iu"
                or np.any(self.actions < 0) or np.any(self.actions >= 12)
                or np.ndim(self.memories) != 2 or np.shape(self.memories)[0] != count
                or not np.isfinite(self.memories).all()
                or len(self.diagnostics) not in (0, count)):
            raise ValueError("Invalid visual policy actions or explicit memories")
