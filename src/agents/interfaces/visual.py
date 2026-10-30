"""Actor-only visual policy protocol and a deliberately limited pixel diagnostic.

The diagnostic recognizes the primitive renderer's blue vehicles. It is not a
learned driver, a general detector, a route planner, or a safe teaching policy.
"""

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from driversi.agents.traffic_control.following import FollowingConfig, idm_acceleration, quantize_jerk
from driversi.contracts.visual import CameraCalibration, VisualObservation, VisualPolicyResult


class VisualPolicy(Protocol):
    """Persistent inference state belongs in memories/RNG, never hidden on self."""

    def describe(self) -> dict: ...

    def initial_memories(self, count: int) -> np.ndarray: ...

    def infer(self, observation: VisualObservation, memories: np.ndarray,
              rngs: tuple[np.random.Generator, ...]) -> VisualPolicyResult: ...


@dataclass(frozen=True)
class PixelBrakePolicy:
    """Straight-road RGB size/looming diagnostic with explicit per-actor memory.

    Size gives approximate optical distance, not true bumper gap. Only a blue
    connected column run near the image centre is used; partially occluded,
    turning and differently sized objects invalidate the geometric assumption.
    """

    desired_speed: float = 4.
    headway: float = 2.
    assumed_vehicle_width: float = 2.

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Pixel diagnostic parameters must be finite and positive")

    def describe(self) -> dict:
        return {"version": "pixel-brake-diagnostic-v2", "parameters": asdict(self),
                "input": "RGB, own telemetry, own navigation and identified camera calibration only",
                "camera_requirement": "camera_id=front with CameraCalibration.front canonical extrinsic",
                "scope": "primitive blue vehicles; straight longitudinal diagnostic; not a teacher",
                "memory": ["previous_optical_distance", "previous_tick", "previous_visible"],
                "blue_mask": "B-R>80 and B-G>35",
                "distance": "fx * assumed_vehicle_width / visible_column_run_width",
                "closing_speed_clip_mps": [-12., 12.], "action_space": "jerk_discrete_12"}

    def initial_memories(self, count: int) -> np.ndarray:
        return np.zeros((count, 3), dtype=float)

    @staticmethod
    def _width(rgb: np.ndarray, principal_x: float, principal_y: float) -> int:
        image = rgb.astype(np.int16)
        mask = ((image[..., 2]-image[..., 0] > 80)
                & (image[..., 2]-image[..., 1] > 35))
        height, width = mask.shape
        # Source camera projects vehicle bodies below the horizon. Require two
        # coloured pixels per column to avoid isolated edge rasterization noise.
        first_row = max(0, min(height-1, int(principal_y-height/6)))
        columns = np.flatnonzero(mask[first_row:height-1].sum(axis=0) >= 2)
        if not len(columns):
            return 0
        runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1)+1)
        candidates = [run for run in runs
                      if run[0] <= principal_x+.1*width and run[-1] >= principal_x-.1*width]
        return max((len(run) for run in candidates), default=0)

    def infer(self, observation: VisualObservation, memories: np.ndarray,
              rngs: tuple[np.random.Generator, ...]) -> VisualPolicyResult:
        count = len(observation.actor_ids)
        if (memories.shape != (count, 3) or len(rngs) != count
                or not np.isfinite(memories).all()):
            raise ValueError("Pixel policy requires independent [actor,3] memories and RNGs")
        if "front" not in observation.camera_ids:
            raise ValueError("Pixel policy requires a camera named front")
        camera_index = observation.camera_ids.index("front")
        if not np.allclose(observation.camera_to_body[camera_index],
                           CameraCalibration.front().camera_to_body, atol=1e-8, rtol=0):
            raise ValueError("Pixel policy requires the canonical front camera extrinsic")
        intrinsic = observation.intrinsic[camera_index]
        updated = np.array(memories, dtype=float, copy=True)
        actions = np.full(count, 7, dtype=np.int32)
        diagnostics = []
        config = FollowingConfig(desired_speed=self.desired_speed, headway=self.headway)
        for index, actor in enumerate(observation.actor_ids):
            if observation.alive[index] and not observation.history_valid[index, -1, camera_index]:
                raise ValueError("Pixel policy needs a valid current front image")
            width = self._width(observation.rgb[index, -1, camera_index], intrinsic[0, 2], intrinsic[1, 2])
            gap = float(intrinsic[0, 0]*self.assumed_vehicle_width/width) if width else None
            closing = 0.
            elapsed = (observation.tick-memories[index, 1])*observation.dt
            if gap is not None and memories[index, 2] and elapsed > 0:
                closing = float(np.clip((memories[index, 0]-gap)/elapsed, -12, 12))
            speed, longitudinal, lateral, _ = observation.telemetry[index]
            target = idm_acceleration(float(speed), self.desired_speed, gap, closing, config)
            if speed < .2 and target < 0:
                target = 0.
            if observation.alive[index]:
                actions[index] = quantize_jerk(target, 0., np.asarray([longitudinal, lateral]), float(speed))
            updated[index] = [gap or 0., observation.tick, gap is not None]
            diagnostics.append({"actor_id": actor, "visible_width_pixels": width,
                                "camera_id": "front",
                                "estimated_optical_distance_m": gap,
                                "estimated_closing_speed_mps": closing,
                                "desired_acceleration_mps2": target,
                                "input_tick": observation.tick})
        return VisualPolicyResult(actions, updated, tuple(diagnostics))
