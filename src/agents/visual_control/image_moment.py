"""A pixel-dependence diagnostic, never a driver or teacher."""

import numpy as np

from driversi.contracts.sampling import ActorVisualInput
from driversi.contracts.visual import VisualPolicyResult


class ImageMomentPolicy:
    version = "rgb-moment-diagnostic-v1"

    def describe(self):
        return {"version": self.version, "kind": "diagnostic_not_learned_or_teacher",
                "action_space": "jerk_discrete_12", "threshold": .5,
                "input": "actor-owned RGB/history; own telemetry/navigation permitted",
                "rule": "valid front RGB mean/255 < .5: action 1; otherwise action 7",
                "memory": ["last_brightness", "last_tick", "decision_count"],
                "preprocessing": "identity_uint8_rgb_v1"}

    def initial_memories(self, count):
        return np.zeros((count, 3), float)

    def infer(self, observations: tuple[ActorVisualInput, ...], memories, rngs):
        if memories.shape != (len(observations), 3) or len(rngs) != len(observations):
            raise ValueError("Independent actor memory/RNG required")
        updated = memories.copy()
        actions, diagnostics = [], []
        for i, obs in enumerate(observations):
            # Source front or explicitly named virtual front; do not rely on color codes.
            front = next((v for v in obs.history[-1]
                          if v.camera_id == "CAM_F0" or v.camera_id.endswith("__virtual_front")), None)
            if front is None:
                raise ValueError("Diagnostic requires an identified front camera")
            brightness = float(front.rgb[front.pixel_valid].mean()/255)
            action = 1 if brightness < .5 and obs.alive else 7
            if obs.alive:
                updated[i] = [brightness, obs.history_ticks[-1], memories[i, 2]+1]
            actions.append(action)
            diagnostics.append({"actor_id": obs.actor_id, "input_camera": front.camera_id,
                                "brightness": brightness, "input_tick": obs.history_ticks[-1],
                                "scope": "pixel_dependency_only_not_driving_quality"})
        return VisualPolicyResult(np.asarray(actions, np.int32), updated, tuple(diagnostics))
