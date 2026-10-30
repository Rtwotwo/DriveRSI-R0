"""Heterogeneous actor-owned RGB inputs, independent of any renderer backend."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from driversi.contracts.model_capability import VISUAL_POLICY, require_capability
from driversi.contracts.visual import VisualPolicyResult, frozen_array


@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    rgb: np.ndarray
    pixel_valid: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    camera_to_body: np.ndarray
    camera_model: str
    timestamp_us: int
    image_path: str
    valid_path: str

    def __post_init__(self):
        rgb, mask = np.asarray(self.rgb), np.asarray(self.pixel_valid)
        k, pose = np.asarray(self.intrinsic), np.asarray(self.camera_to_body)
        if (not self.camera_id or not self.camera_model or rgb.ndim != 3 or rgb.shape[-1] != 3
                or rgb.dtype != np.uint8 or mask.shape != rgb.shape[:2] or mask.dtype != bool
                or not mask.any() or k.shape != (3, 3) or pose.shape != (4, 4)
                or not np.isfinite(k).all() or not np.isfinite(pose).all()
                or not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
                or np.asarray(self.distortion).ndim != 1 or not np.isfinite(self.distortion).all()
                or type(self.timestamp_us) is not int or not self.image_path or not self.valid_path):
            raise ValueError("Invalid actor-owned camera input")
        for name in ("rgb", "pixel_valid", "intrinsic", "distortion", "camera_to_body"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))


@dataclass(frozen=True)
class ActorVisualInput:
    actor_id: str
    history: tuple[tuple[CameraInput, ...], ...]
    history_ticks: tuple[int, ...]
    telemetry: np.ndarray
    navigation: np.ndarray
    alive: bool

    def __post_init__(self):
        if (not self.actor_id or not self.history or len(self.history) != len(self.history_ticks)
                or any(type(t) is not int or t < 0 for t in self.history_ticks)
                or any(a >= b for a, b in zip(self.history_ticks, self.history_ticks[1:]))
                or np.shape(self.telemetry) != (4,) or np.shape(self.navigation) != (2,)
                or not np.isfinite(self.telemetry).all() or not np.isfinite(self.navigation).all()
                or type(self.alive) is not bool):
            raise ValueError("Invalid actor visual history or own telemetry/navigation")
        signature = None
        for views in self.history:
            ids = tuple(v.camera_id for v in views)
            if not ids or len(set(ids)) != len(ids) or (signature is not None and ids != signature):
                raise ValueError("Camera identity/order must be stable within actor history")
            signature = ids
        object.__setattr__(self, "telemetry", frozen_array(self.telemetry, float))
        object.__setattr__(self, "navigation", frozen_array(self.navigation, float))


# Model weights/configuration are shared and frozen during collection. Every
# persistent episode state must be returned through memories and actor RNGs.
# Hidden mutable caches that affect decisions violate this interface.
class SamplingPolicy(Protocol):
    def describe(self) -> dict: ...
    def initial_memories(self, count: int) -> np.ndarray: ...
    def infer(self, observations: tuple[ActorVisualInput, ...], memories: np.ndarray,
              rngs: tuple[np.random.Generator, ...]) -> VisualPolicyResult: ...


def policy_description(policy: SamplingPolicy) -> dict:
    import json

    descriptor = json.loads(json.dumps(policy.describe(), allow_nan=False))
    if (not isinstance(descriptor, dict)
            or any(not isinstance(descriptor.get(k), str) or not descriptor[k].strip()
                   for k in ("version", "kind", "preprocessing"))
            or descriptor.get("action_space") != "jerk_discrete_12"):
        raise ValueError("Policy must declare version, kind, preprocessing and jerk_discrete_12")
    require_capability(descriptor.get("capabilities", VISUAL_POLICY), "visual_sampling")
    return descriptor
