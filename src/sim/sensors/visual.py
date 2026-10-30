"""Synchronous visual sessions: one physical world, explicit actor state and clocks.

All actor inference is batched before the joint physical step. Render/policy
failures roll back owned state. No wall-clock scheduling, learner, or RPC lives
here. A renderer supplies QA labels, but actor observations carry RGB only.
"""

import copy
import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from driversi.agents.interfaces.visual import VisualPolicy
from driversi.contracts.scene import Scene
from driversi.contracts.transition import Snapshot, Transition
from driversi.contracts.visual import (
    VISUAL_OBSERVATION_VERSION,
    CameraRig,
    SensorBatch,
    VisualObservation,
    WorldFrame,
    frozen_array,
)
from driversi.rendering.visual.primitive import PrimitiveRenderer
from driversi.sim.backends.pufferdrive import PufferDriveWorld

_OBSERVATION_ARRAYS = ("rgb", "history_valid", "history_ticks", "telemetry", "navigation", "alive", "intrinsic",
                       "camera_to_body")
_SENSOR_ARRAYS = ("rgb", "depth", "instances", "valid")


@dataclass(frozen=True)
class VisualStep:
    observation: VisualObservation  # exact policy input; held on unsampled ticks
    sensor: SensorBatch  # QA output for the same source observation tick
    policy_actions: np.ndarray
    executed_actions: np.ndarray
    observation_tick: int
    diagnostics: tuple[dict, ...]
    newly_decided: bool
    overrides: np.ndarray
    transition: Transition

    def __post_init__(self):
        for name in ("policy_actions", "executed_actions", "overrides"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        object.__setattr__(self, "diagnostics", copy.deepcopy(self.diagnostics))


@dataclass(frozen=True)
class VisualSnapshot:
    """Portable JSON metadata + numeric arrays, with no Python object pickle."""

    world: Snapshot
    metadata: dict
    arrays: dict[str, np.ndarray]

    def __post_init__(self):
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
        object.__setattr__(self, "arrays", {key: frozen_array(value) for key, value in self.arrays.items()})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema_version": 1, "session": self.metadata,
                    "world": {"scene_key": self.world.scene_key, "build_id": self.world.build_id,
                              "alive": self.world.alive, "episode_done": self.world.episode_done}}
        with path.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata, allow_nan=False),
                                native_state=np.frombuffer(self.world.native_state, np.uint8),
                                **{f"payload_{key}": value for key, value in self.arrays.items()})

    @classmethod
    def load(cls, path: Path) -> "VisualSnapshot":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("schema_version") != 1:
                raise ValueError("Unsupported visual snapshot schema")
            world = metadata["world"]
            world["alive"] = tuple(world["alive"])
            snapshot = Snapshot(native_state=data["native_state"].tobytes(), **world)
            arrays = {key.removeprefix("payload_"): data[key].copy()
                      for key in data.files if key.startswith("payload_")}
            return cls(snapshot, metadata["session"], arrays)


class VisualSession:
    """One owner for world, rendering history, held decisions and actor memory.

    Policy objects must keep no mutable hidden state; memories/RNGs are supplied
    explicitly. This is an interface contract, not a sandbox for untrusted code.
    Access to ``world`` is for operator/recorder QA, never passed into ``infer``.
    """

    version = "visual-session-v1"

    def __init__(self, scene: Scene, policy: VisualPolicy, rig: CameraRig, *,
                 history: int = 2, sensor_stride: int = 1, seed: int = 0,
                 episode_id: str | None = None, branch_id: str = "main",
                 allow_unverified_heading: bool = False):
        if (type(history) is not int or not 1 <= history <= 16
                or type(sensor_stride) is not int or sensor_stride < 1
                or type(seed) is not int or seed < 0 or not branch_id):
            raise ValueError("Invalid visual history, clock, seed or branch")
        self._scene, self.policy, self.rig = copy.deepcopy((scene, policy, rig))
        self.history, self.sensor_stride, self.seed = history, sensor_stride, seed
        self.actor_ids = tuple(actor for actor, active in zip(scene.actor_ids, scene.controlled) if active)
        if type(allow_unverified_heading) is not bool:
            raise ValueError("allow_unverified_heading must be an explicit boolean")
        qualifications = scene.metadata.get("heading_qualification", {})
        if not isinstance(qualifications, dict):
            raise ValueError("heading_qualification must map actor IDs to qualification strings")
        self.heading_qualification = {
            actor: qualifications.get(actor, "unverified_box_axis") for actor in self.actor_ids}
        if any(not isinstance(status, str) for status in self.heading_qualification.values()):
            raise ValueError("Heading qualifications must be strings")
        unverified = [actor for actor, status in self.heading_qualification.items()
                      if status not in {"verified_front", "declared_synthetic"}]
        if unverified and not allow_unverified_heading:
            raise ValueError("Unverified camera heading: explicitly allow virtual box-axis experiments "
                             f"or supply verified heading qualifications for {unverified}")
        self.allow_unverified_heading = allow_unverified_heading
        self._indices = np.flatnonzero(self._scene.controlled)
        self.dt = .1  # pinned native action/integration contract
        self.renderer = PrimitiveRenderer(self._scene, self.rig)
        self._description = {"version": self.version, "policy": self.policy.describe(),
                             "rig": self.rig.to_dict(), "history": history,
                             "seed": seed,
                             "sensor_stride": sensor_stride, "physics_dt": self.dt,
                             "renderer_version": self.renderer.version,
                             "heading_qualification": self.heading_qualification,
                             "allow_unverified_heading": allow_unverified_heading,
                             "action_space": "jerk_discrete_12",
                             "actor_observation_version": VISUAL_OBSERVATION_VERSION,
                             "scope": "primitive planar RGB diagnostic; no learner or real-time guarantee"}
        if not isinstance(self._description["policy"].get("version"), str) or not self._description["policy"]["version"]:
            raise ValueError("Visual policy must declare a nonempty version")
        self._config_key = hashlib.sha256(json.dumps(self._description, sort_keys=True,
                                                    allow_nan=False).encode()).hexdigest()
        self.world = PufferDriveWorld(self._scene)
        self.episode_id = episode_id or uuid.uuid4().hex
        self.branch_id, self.generation = branch_id, 0
        try:
            self._initialize_actor_state()
        except Exception:
            self.close()
            raise

    def _initialize_actor_state(self):
        self._memories = np.array(self.policy.initial_memories(len(self.actor_ids)), dtype=float, copy=True)
        if (self._memories.ndim != 2 or self._memories.shape[0] != len(self.actor_ids)
                or not np.isfinite(self._memories).all()):
            raise ValueError("Initial model memories must be finite [actor,memory]")
        self._rngs = tuple(np.random.default_rng(np.random.SeedSequence(
            [self.seed, int.from_bytes(hashlib.sha256(actor.encode()).digest()[:8], "little")]))
            for actor in self.actor_ids)
        self._history: list[SensorBatch] = []
        self._observation: VisualObservation | None = None
        self._held_actions = np.full(len(self.actor_ids), 7, np.int32)
        self._held_observation_tick = -1
        self._diagnostics: tuple[dict, ...] = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.world.close()

    @property
    def tick(self) -> int:
        return self.world.observe().step

    @property
    def episode_done(self) -> bool:
        return self.world.episode_done

    @property
    def memories(self) -> np.ndarray:
        return self._memories.copy()

    @property
    def last_sensor(self) -> SensorBatch | None:
        return copy.deepcopy(self._history[-1]) if self._history else None

    def describe(self) -> dict:
        return copy.deepcopy({**self._description, "actor_ids": self.actor_ids,
                              "bindings": [{"actor_id": actor,
                                            "policy_version": self._description["policy"]["version"],
                                            "heading_qualification": self.heading_qualification[actor],
                                            "trainable": False} for actor in self.actor_ids],
                              "seed": self.seed, "configuration_hash": self._config_key})

    def world_frame(self) -> WorldFrame:
        return WorldFrame(self._scene.actor_ids, self.world.state()[:, :4], self.tick,
                          self.episode_id, self.branch_id, self.generation, self.dt)

    def reset(self) -> VisualObservation:
        self.world.reset()
        self.generation += 1
        self.episode_id = uuid.uuid4().hex
        self._initialize_actor_state()
        return self.observe()

    def observe(self) -> VisualObservation:
        tick = self.tick
        if self._observation is not None and (tick % self.sensor_stride or self._observation.tick == tick):
            return copy.deepcopy(self._observation)
        frame = self.world_frame()
        sensor = self.renderer.render_batch(frame, self.actor_ids)
        alive = self.world.observe().alive
        if (sensor.actor_ids != self.actor_ids
                or sensor.camera_ids != tuple(camera.camera_id for camera in self.rig.cameras)
                or sensor.rgb.shape != (len(self.actor_ids), len(self.rig.cameras),
                                        self.rig.cameras[0].height, self.rig.cameras[0].width, 3)
                or sensor.renderer_version != self._description["renderer_version"]
                or sensor.modality != "primitive_rgb"
                or (sensor.tick, sensor.episode_id, sensor.branch_id, sensor.generation)
                != (frame.tick, frame.episode_id, frame.branch_id, frame.generation)
                or not sensor.valid[alive].all()):
            raise ValueError("Renderer returned stale, misbound or invalid actor cameras")
        history = (self._history + [sensor])[-self.history:]
        shape = (len(self.actor_ids), self.history, *sensor.rgb.shape[1:])
        rgb = np.zeros(shape, np.uint8)
        valid = np.zeros(shape[:3], bool)
        ticks = np.full(shape[:2], -1, np.int64)
        for index, sample in enumerate(history, self.history-len(history)):
            rgb[:, index], valid[:, index], ticks[:, index] = sample.rgb, sample.valid, sample.tick
        state = self.world.state()[self._indices]
        cosine, sine = np.cos(state[:, 3]), np.sin(state[:, 3])
        speed = state[:, 4]*cosine + state[:, 5]*sine
        telemetry = np.column_stack((speed, self.world.motion()[self._indices]))
        target = state[:, 7:9]-state[:, :2]
        navigation = np.column_stack((target[:, 0]*cosine + target[:, 1]*sine,
                                      -target[:, 0]*sine + target[:, 1]*cosine))
        observation = VisualObservation(self.actor_ids, rgb, valid, ticks, telemetry, navigation,
                                        alive, np.stack([camera.intrinsic for camera in self.rig.cameras]),
                                        tuple(camera.camera_id for camera in self.rig.cameras),
                                        np.stack([camera.camera_to_body for camera in self.rig.cameras]),
                                        tick, self.dt)
        self._history, self._observation = history, observation
        return copy.deepcopy(observation)

    def step(self, action_overrides: Mapping[str, int] | None = None) -> VisualStep:
        if self.episode_done:
            raise RuntimeError("Episode is done; reset explicitly before stepping")
        before = self.snapshot()
        try:
            observation = self.observe()
            newly_decided = self._held_observation_tick != observation.tick
            memories = self._memories.copy()
            rngs = copy.deepcopy(self._rngs)
            actions = self._held_actions.copy()
            diagnostics = copy.deepcopy(self._diagnostics)
            if newly_decided:
                result = self.policy.infer(copy.deepcopy(observation), memories, rngs)
                result.validate(len(self.actor_ids))
                if result.memories.shape != self._memories.shape:
                    raise ValueError("Policy changed memory shape during the episode")
                actions, memories = np.array(result.actions, copy=True), np.array(result.memories, copy=True)
                diagnostics = copy.deepcopy(result.diagnostics)
                json.dumps(diagnostics, allow_nan=False)
            executed, overridden = actions.copy(), np.zeros(len(actions), bool)
            if action_overrides is not None:
                if (not isinstance(action_overrides, Mapping)
                        or not set(action_overrides).issubset(self.actor_ids)):
                    raise ValueError("Overrides must name controlled actors at the current tick boundary")
                for actor, action in action_overrides.items():
                    if not isinstance(action, (int, np.integer)) or isinstance(action, bool) or not 0 <= action < 12:
                        raise ValueError("Override requires an integer action in [0,11]")
                    index = self.actor_ids.index(actor)
                    executed[index], overridden[index] = action, True
            transition = self.world.step(executed)
            self._memories, self._rngs = memories, rngs
            self._held_actions, self._held_observation_tick = actions.copy(), observation.tick
            self._diagnostics = diagnostics
            return VisualStep(observation, self.last_sensor, actions, executed, observation.tick,
                              diagnostics, newly_decided, overridden, transition)
        except Exception:
            self.restore(before)
            raise

    def snapshot(self) -> VisualSnapshot:
        arrays = {"memories": self._memories, "held_actions": self._held_actions}
        history = []
        for index, sample in enumerate(self._history):
            history.append({name: copy.deepcopy(getattr(sample, name)) for name in
                            ("actor_ids", "camera_ids", "tick", "episode_id", "branch_id",
                             "generation", "renderer_version", "modality")})
            arrays.update({f"history_{index}_{name}": getattr(sample, name) for name in _SENSOR_ARRAYS})
        observation = None
        if self._observation is not None:
            observation = {"tick": self._observation.tick, "version": self._observation.version,
                           "camera_ids": self._observation.camera_ids}
            arrays.update({f"observation_{name}": getattr(self._observation, name) for name in _OBSERVATION_ARRAYS})
        metadata = {"configuration_hash": self._config_key, "actor_ids": self.actor_ids,
                    "episode_id": self.episode_id, "branch_id": self.branch_id,
                    "generation": self.generation, "tick": self.tick, "history": history,
                    "observation": observation, "held_observation_tick": self._held_observation_tick,
                    "diagnostics": self._diagnostics,
                    "rng_states": [copy.deepcopy(rng.bit_generator.state) for rng in self._rngs]}
        return VisualSnapshot(self.world.snapshot(), metadata, arrays)

    def restore(self, snapshot: VisualSnapshot) -> VisualObservation | None:
        metadata, arrays = snapshot.metadata, snapshot.arrays
        json.dumps(metadata, allow_nan=False)
        if (metadata.get("configuration_hash") != self._config_key
                or tuple(metadata.get("actor_ids", ())) != self.actor_ids):
            raise ValueError("Visual snapshot configuration or actors mismatch")
        if (type(metadata["tick"]) is not int or not 0 <= metadata["tick"] <= self.world.max_steps
                or type(metadata["generation"]) is not int or metadata["generation"] < 0
                or not isinstance(metadata["episode_id"], str) or not metadata["episode_id"]
                or not isinstance(metadata["branch_id"], str) or not metadata["branch_id"]
                or type(metadata["held_observation_tick"]) is not int
                or not -1 <= metadata["held_observation_tick"] <= metadata["tick"]):
            raise ValueError("Invalid visual snapshot identity or clock")
        memories, actions = arrays["memories"].copy(), arrays["held_actions"].copy()
        if (memories.shape != self._memories.shape or not np.isfinite(memories).all()
                or actions.shape != (len(self.actor_ids),) or actions.dtype.kind not in "iu"
                or np.any(actions < 0) or np.any(actions >= 12)
                or not 0 <= len(metadata["history"]) <= self.history
                or len(metadata["rng_states"]) != len(self.actor_ids)
                or len(metadata["diagnostics"]) not in (0, len(self.actor_ids))):
            raise ValueError("Invalid visual snapshot actor state")
        history = []
        for index, item in enumerate(metadata["history"]):
            values = {name: arrays[f"history_{index}_{name}"] for name in _SENSOR_ARRAYS}
            values.update(item)
            values["actor_ids"], values["camera_ids"] = tuple(values["actor_ids"]), tuple(values["camera_ids"])
            sample = SensorBatch(**values)
            if (sample.actor_ids != self.actor_ids or not 0 <= sample.tick <= metadata["tick"]
                    or sample.tick % self.sensor_stride
                    or sample.camera_ids != tuple(camera.camera_id for camera in self.rig.cameras)
                    or sample.rgb.shape != (len(self.actor_ids), len(self.rig.cameras),
                                            self.rig.cameras[0].height, self.rig.cameras[0].width, 3)
                    or sample.renderer_version != self._description["renderer_version"]
                    or sample.modality != "primitive_rgb"
                    or (sample.episode_id, sample.branch_id, sample.generation)
                    != (metadata["episode_id"], metadata["branch_id"], metadata["generation"])
                    or (history and sample.tick <= history[-1].tick)):
                raise ValueError("Invalid visual snapshot history")
            history.append(sample)
        cached = metadata["observation"]
        observation = None if cached is None else VisualObservation(
            self.actor_ids, **{name: arrays[f"observation_{name}"] for name in _OBSERVATION_ARRAYS},
            camera_ids=tuple(cached["camera_ids"]),
            tick=cached["tick"], dt=self.dt, version=cached["version"])
        if ((observation is None) != (not history)
                or (observation is not None and observation.tick != history[-1].tick)
                or (observation is not None and metadata["tick"]-observation.tick > self.sensor_stride)
                or (metadata["held_observation_tick"] >= 0
                    and (observation is None or metadata["held_observation_tick"] > observation.tick
                         or metadata["held_observation_tick"] % self.sensor_stride))):
            raise ValueError("Invalid visual snapshot clock")
        if observation is not None:
            expected_shape = (len(self.actor_ids), self.history, *history[-1].rgb.shape[1:])
            if (observation.version != self._description["actor_observation_version"]
                    or observation.rgb.shape != expected_shape
                    or observation.camera_ids != tuple(camera.camera_id for camera in self.rig.cameras)
                    or not np.array_equal(observation.camera_to_body,
                                          np.stack([camera.camera_to_body for camera in self.rig.cameras]))
                    or not np.array_equal(observation.intrinsic,
                                          np.stack([camera.intrinsic for camera in self.rig.cameras]))):
                raise ValueError("Visual snapshot observation version or camera geometry mismatch")
            padding = self.history-len(history)
            if (observation.rgb[:, :padding].any() or observation.history_valid[:, :padding].any()
                    or np.any(observation.history_ticks[:, :padding] != -1)):
                raise ValueError("Visual snapshot warmup contains nonexistent history")
            for index, sample in enumerate(history, padding):
                if (not np.array_equal(observation.rgb[:, index], sample.rgb)
                        or not np.array_equal(observation.history_valid[:, index], sample.valid)
                        or np.any(observation.history_ticks[:, index] != sample.tick)):
                    raise ValueError("Visual snapshot image/history alignment mismatch")
        rngs = tuple(np.random.default_rng() for _ in self.actor_ids)
        for rng, state in zip(rngs, metadata["rng_states"]):
            rng.bit_generator.state = copy.deepcopy(state)
        previous = self.world.snapshot()
        try:
            self.world.restore(snapshot.world)
            if self.tick != metadata["tick"]:
                raise ValueError("Visual snapshot clock differs from physical world")
        except Exception:
            self.world.restore(previous)
            raise
        self._memories, self._held_actions, self._rngs = memories, actions, rngs
        self._history, self._observation = history, observation
        self.episode_id, self.branch_id, self.generation = (metadata[name] for name in
                                                          ("episode_id", "branch_id", "generation"))
        self._held_observation_tick = metadata["held_observation_tick"]
        self._diagnostics = copy.deepcopy(tuple(metadata["diagnostics"]))
        return copy.deepcopy(observation)

    def fork(self, branch_id: str | None = None) -> "VisualSession":
        branch_id = branch_id or uuid.uuid4().hex
        if not branch_id or branch_id == self.branch_id:
            raise ValueError("A fork needs a distinct nonempty branch ID")
        child = type(self)(self._scene, self.policy, self.rig, history=self.history,
                           sensor_stride=self.sensor_stride, seed=self.seed,
                           allow_unverified_heading=self.allow_unverified_heading)
        try:
            snapshot = self.snapshot()
            metadata = copy.deepcopy(snapshot.metadata)
            metadata["branch_id"] = branch_id
            for item in metadata["history"]:
                item["branch_id"] = branch_id
            child.restore(VisualSnapshot(snapshot.world, metadata, snapshot.arrays))
        except Exception:
            child.close()
            raise
        return child
