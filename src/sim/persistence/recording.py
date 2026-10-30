"""One engineering rollout schema for ordinary episodes and paired branches."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from driversi.contracts.scene import Scene
from driversi.contracts.transition import Snapshot, Transition

if TYPE_CHECKING:
    from driversi.sim.backends.pufferdrive import PufferDriveWorld


@dataclass
class RolloutRecord:
    """Actual state/action sequence, with no interpolation or time relabeling."""

    states: list[np.ndarray]
    motion: list[np.ndarray]
    transitions: list[Transition] = field(default_factory=list)
    actions: list[list[int]] = field(default_factory=list)
    diagnostics: list[list[dict]] = field(default_factory=list)

    presence: list[np.ndarray] = field(default_factory=list)
    lifecycle: list[dict] = field(default_factory=list)

    @classmethod
    def start(cls, world: "PufferDriveWorld") -> "RolloutRecord":
        return cls([world.state()], [world.motion()], presence=[world.presence()], lifecycle=[world.lifecycle()])

    def append(self, world: "PufferDriveWorld", actions: dict[str, int], transition: Transition,
               diagnostics: list[dict] | None = None) -> None:
        self.transitions.append(transition)
        self.actions.append([actions[actor] for actor in world.actor_ids])
        self.states.append(world.state())
        self.motion.append(world.motion())
        self.presence.append(world.presence())
        if transition.traffic_controls:
            self.lifecycle[-1]["executed_traffic_controls"] = list(transition.traffic_controls)
        self.lifecycle.append(world.lifecycle())
        if diagnostics is not None:
            self.diagnostics.append(diagnostics)

    def followed_by(self, branch: "RolloutRecord") -> "RolloutRecord":
        """Join a real prefix to a branch only at an exactly shared world state."""
        if (not np.array_equal(self.states[-1], branch.states[0])
                or not np.array_equal(self.motion[-1], branch.motion[0])
                or bool(self.presence) != bool(branch.presence)
                or self.presence and not np.array_equal(self.presence[-1], branch.presence[0])):
            raise ValueError("Rollout prefix and branch do not share their boundary state")
        return RolloutRecord(self.states[:-1] + branch.states, self.motion[:-1] + branch.motion,
                             self.transitions + branch.transitions, self.actions + branch.actions,
                             self.diagnostics + branch.diagnostics,
                             self.presence[:-1] + branch.presence, self.lifecycle[:-1] + branch.lifecycle)


def write_rollout(scene: Scene, output: Path, *, record: RolloutRecord, snapshot: Snapshot,
                  controlled_ids: tuple[str, ...], policy: str, controller_configuration: dict,
                  protocol: dict | None = None) -> dict:
    """Write the same full-prefix schema consumed by the strict replay exporter."""
    steps, count = len(record.transitions), len(controlled_ids)
    if len(record.states) != steps + 1 or len(record.motion) != steps + 1 or len(record.actions) != steps:
        raise ValueError("Rollout must pair each executed action with its actual successor")
    if len(record.diagnostics) not in (0, steps):
        raise ValueError("Diagnostics must cover every transition or be absent")
    expected_ids = tuple(actor for actor, controlled in zip(scene.actor_ids, scene.controlled) if controlled)
    if controlled_ids != expected_ids:
        raise ValueError("Rollout control order must match the scene")
    if any(transition.observation.step != index + 1 or transition.observation.actor_ids != controlled_ids
           for index, transition in enumerate(record.transitions)):
        raise ValueError("Engineering rollouts require the real contiguous prefix from step zero")
    expected_presence = np.stack([scene.presence_at(i) for i in range(steps+1)])
    if "traffic_continuity" in scene.metadata:
        from driversi.sim.execution.traffic_lifecycle import validate_recorded_lifecycle
        validate_recorded_lifecycle(scene, record.states, record.presence, record.lifecycle)
        expected_presence = np.asarray(record.presence, bool)
    if scene.presence is not None and (len(record.presence) != steps+1 or not np.array_equal(record.presence, expected_presence)):
        raise ValueError("Rollout must record actual participant presence")
    native = controller_configuration.get("action_space") == "driverl_native_jerk_steering_rate"
    action_shape = (steps, count, 2) if native else (steps, count)
    action_array = np.asarray(record.actions, dtype=np.float64 if native else np.int32).reshape(action_shape)
    if not np.isfinite(action_array).all():
        raise ValueError("Nonfinite action record")
    summary = {"schema_version": 1, "kind": "engineering_rollout", "policy": policy,
               "scenario_id": scene.metadata["scenario_id"], "split": scene.metadata["split"],
               "steps": steps, "controlled_actors": count, "backend_build": snapshot.build_id,
               "scene_key": snapshot.scene_key, "action_space": "driverl_native_jerk_steering_rate" if native else "jerk_discrete_12", "dt_seconds": 0.1,
               "observation_version": "driverl-nuplan-native-v1" if native else "pufferdrive-jerk-1124-v1",
                "world_observation_version": "pufferdrive-jerk-1124-v1",
               "reward_version": "event-only-failure-first-v1",
               "controller_configuration": controller_configuration,
               "metric_observation_version": "metric-v1", "scene_metadata": scene.metadata}
    if protocol is not None:
        summary["protocol"] = protocol
    arrays = {name: np.asarray([getattr(t, name) for t in record.transitions], dtype=dtype).reshape(-1, count)
              for name, dtype in (("rewards", np.float32), ("valid", bool),
                                  ("terminated", bool), ("truncated", bool))}
    arrays.update({name: np.asarray([t.events[name] for t in record.transitions], dtype=bool).reshape(-1, count)
                   for name in ("collision", "offroad", "goal")})
    if all("road_boundary_contact" in t.events for t in record.transitions) and record.transitions:
        arrays["road_boundary_contact"] = np.asarray(
            [t.events["road_boundary_contact"] for t in record.transitions], dtype=bool
        ).reshape(-1, count)
    if "driving_tasks" in scene.metadata:
        arrays["route_support_end"] = np.asarray([t.events["route_support_end"] for t in record.transitions], dtype=bool).reshape(-1, count)
        summary["goal_semantics"] = "explicit-destination-or-route-support-truncation-v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(summary, ensure_ascii=False, allow_nan=False),
                            actor_ids=np.asarray(scene.actor_ids), controlled_ids=np.asarray(controlled_ids),
                            states=np.stack(record.states), motion=np.stack(record.motion),
                            presence=expected_presence, lifecycle=json.dumps(record.lifecycle, allow_nan=False),
                            controller_diagnostics=json.dumps(record.diagnostics, allow_nan=False),
                            actions=action_array, **arrays)
    return {key: summary[key] for key in ("kind", "scenario_id", "steps", "controlled_actors")}
