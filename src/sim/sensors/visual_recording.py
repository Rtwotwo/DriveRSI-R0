"""Auditable visual-engineering records; these are not on-policy learner batches."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from driversi.contracts.scene import Scene
from driversi.sim.persistence.recording import RolloutRecord, write_rollout

if TYPE_CHECKING:
    from driversi.sim.sensors.visual import VisualSession, VisualStep


@dataclass
class VisualRecord:
    physical: RolloutRecord
    steps: list["VisualStep"] = field(default_factory=list)
    memories: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def start(cls, session: "VisualSession") -> "VisualRecord":
        return cls(RolloutRecord.start(session.world), memories=[session.memories.copy()])

    def append(self, session: "VisualSession", step: "VisualStep") -> None:
        actions = dict(zip(session.actor_ids, (int(x) for x in step.executed_actions), strict=True))
        diagnostics = []
        for index, actor in enumerate(session.actor_ids):
            entry = dict(step.diagnostics[index]) if step.diagnostics else {}
            entry.update(actor=actor, action=actions[actor], controller="pixel-brake-diagnostic")
            if step.overrides[index]:
                entry["controller"] = "explicit-action-intervention"
            diagnostics.append(entry)
        self.physical.append(session.world, actions, step.transition, diagnostics)
        self.steps.append(step)
        self.memories.append(session.memories.copy())

    def copy_prefix(self) -> "VisualRecord":
        """Records/frames are immutable at the API boundary; copy the list owners."""
        physical = RolloutRecord(
            list(self.physical.states), list(self.physical.motion), list(self.physical.transitions),
            list(self.physical.actions), list(self.physical.diagnostics))
        return VisualRecord(physical, list(self.steps), list(self.memories))


def write_visual_record(scene: Scene, session: "VisualSession", record: VisualRecord,
                        directory: Path, *, initial_snapshot, elapsed_seconds: float,
                        protocol: dict | None = None) -> dict:
    """Write into an owned new run directory; caller commits manifest last.

    Each physical step references the actual visual input used, including stale
    observations for held actions. Input history is stored explicitly, losslessly.
    Depth and instance buffers are deliberately absent from the actor trajectory.
    """
    steps = len(record.steps)
    if not steps:
        raise ValueError("Visual recording requires at least one valid joint transition")
    if len(record.memories) != steps + 1:
        raise ValueError("Visual memory boundaries must cover every transition")
    if any((directory / name).exists() for name in
           ("scene.npz", "initial-visual-snapshot.npz", "rollout.npz", "visual.npz")):
        raise FileExistsError("Visual run artifacts must not overwrite an existing run")
    directory.mkdir(parents=True, exist_ok=True)
    ids = session.actor_ids
    scene.save(directory / "scene.npz")
    initial_snapshot.save(directory / "initial-visual-snapshot.npz")
    # Reuse the verified full-prefix physical schema and its BEV inspector.
    world_snapshot = session.world.snapshot()  # identity/build only for the physical writer
    write_rollout(scene, directory / "rollout.npz", record=record.physical,
                  snapshot=world_snapshot, controlled_ids=ids, policy="pixel-brake-diagnostic",
                  controller_configuration=session.describe(), protocol=protocol)
    arrays = {
        "actor_ids": np.asarray(ids),
        "physical_actor_ids": np.asarray(scene.actor_ids),
        "camera_ids": np.asarray(record.steps[0].observation.camera_ids),
        "intrinsic": record.steps[0].observation.intrinsic,
        "camera_to_body": record.steps[0].observation.camera_to_body,
        "states": np.stack(record.physical.states),
        "motion": np.stack(record.physical.motion),
        "rgb": np.stack([s.observation.rgb for s in record.steps]),
        "history_valid": np.stack([s.observation.history_valid for s in record.steps]),
        "history_ticks": np.stack([s.observation.history_ticks for s in record.steps]),
        "telemetry": np.stack([s.observation.telemetry for s in record.steps]),
        "navigation": np.stack([s.observation.navigation for s in record.steps]),
        "observation_alive": np.stack([s.observation.alive for s in record.steps]),
        "observation_tick": np.asarray([s.observation_tick for s in record.steps]),
        "newly_decided": np.asarray([s.newly_decided for s in record.steps], bool),
        "policy_actions": np.stack([s.policy_actions for s in record.steps]),
        "executed_actions": np.stack([s.executed_actions for s in record.steps]),
        "overrides": np.stack([s.overrides for s in record.steps]),
        "memories": np.stack(record.memories),
        "trainable": np.zeros((steps, len(ids)), bool),
    }
    for name in ("rewards", "valid", "terminated", "truncated"):
        arrays[name] = np.stack([getattr(s.transition, name) for s in record.steps])
    for name in ("collision", "offroad", "goal"):
        arrays[name] = np.stack([s.transition.events[name] for s in record.steps])
    final_observation = session.observe()
    arrays["final_rgb"] = final_observation.rgb
    arrays["final_history_valid"] = final_observation.history_valid
    arrays["final_history_ticks"] = final_observation.history_ticks
    arrays["final_telemetry"] = final_observation.telemetry
    arrays["final_navigation"] = final_observation.navigation
    arrays["final_alive"] = final_observation.alive
    config = session.describe()
    metadata = {"schema_version": 1, "kind": "visual_engineering_rollout",
                "modality": "primitive_rgb", "actor_observation_version": record.steps[0].observation.version,
                "action_space": "jerk_discrete_12", "dt_seconds": .1, "steps": steps,
                "scene_key": world_snapshot.scene_key, "backend_build": world_snapshot.build_id,
                "scene_metadata": scene.metadata, "configuration": config,
                "learning_status": "diagnostic_policy_no_model_update",
                "image_semantics": "actual pre-action actor input; held samples retain source ticks",
                "observation_alive_semantics": "actor alive mask in the actual input at observation_tick; "
                                               "held observations retain this mask",
                "physical_valid_semantics": "actor active before each physical transition; "
                                            "may differ from observation_alive during action holding",
                "world_successor_semantics": "states[k+1] follows executed_actions[k]",
                "final_observation_tick": final_observation.tick,
                "protocol": protocol or {}}
    metadata["frame_identity"] = [
        {"episode": s.sensor.episode_id, "branch": s.sensor.branch_id,
         "generation": s.sensor.generation, "tick": s.sensor.tick}
        for s in record.steps]
    with (directory / "visual.npz").open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False), **arrays)
    decisions = int(arrays["newly_decided"].sum())
    camera_count = len(session.rig.cameras)
    return {"schema_version": 1, "kind": "visual_engineering_run", "status": "complete",
            "scenario_id": scene.metadata["scenario_id"], "controlled_actors": len(ids),
            "planned_seconds": float(scene.times[-1]), "actual_seconds": steps * .1,
            "joint_steps": steps, "effective_actor_steps": int(arrays["valid"].sum()),
            "at_least_two_active_seconds": float(np.sum(arrays["valid"].sum(axis=1) >= 2) * .1),
            "decisions": decisions, "consumed_camera_frames": decisions * len(ids) * camera_count,
            "elapsed_sampling_seconds": elapsed_seconds,
            "events": {name: int(np.any(arrays[name], axis=0).sum())
                       for name in ("collision", "offroad", "goal")},
            "configuration": config, "protocol": protocol or {},
            "limitations": ["primitive planar geometry", "straight pixel diagnostic, not learned driving",
                            "fixed physical population", "no PPO/DAgger/RSI update"]}
