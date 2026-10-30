"""Owned, synchronous worlds over a pinned PufferDrive build; no implicit resets."""

import copy
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.background import ReactiveBackground
from driversi.contracts.driving_task import task_identity, tasks_from_metadata
from driversi.contracts.scene import Scene
from driversi.contracts.transition import MetricObservation, Observation, Snapshot, Transition
from driversi.data.participants.background import boundary_identity, identity_suffix
from driversi.data.scenes.pufferdrive import write_binary
from driversi.sim.backends.native_loader import load_native
from driversi.sim.execution.traffic_lifecycle import TrafficLifecycle


def physical_scene_key(scene: Scene, binary=None):
    """The native world's existing identity, available in render-only runtimes."""
    if binary is None:
        with tempfile.TemporaryDirectory(prefix="driversi-scene-key-") as directory:
            path = Path(directory) / "scene.bin"
            write_binary(scene, path)
            binary = path.read_bytes()
    return hashlib.sha256(
            binary + boundary_identity(scene) + json.dumps(scene.actor_ids).encode() + identity_suffix(scene) + task_identity(scene)
            + (json.dumps(scene.metadata["native_initial_motion"],sort_keys=True).encode() if "native_initial_motion" in scene.metadata else b"")
        ).hexdigest()


class PufferDriveWorld:
    def __init__(self, scene: Scene):
        self._handle = None
        self._native = load_native()
        if len(scene.times) > 91 and not getattr(self._native, "variable_horizon", 0):
            raise RuntimeError("Rebuild native adapter for variable-length horizons")
        native_capacity = getattr(self._native, "max_world_actors", 256)
        if len(scene.actor_ids) > native_capacity:
            raise RuntimeError(f"Scene has {len(scene.actor_ids)} actors; native backend capacity is "
                               f"{native_capacity}. Rebuild with scripts/build/build_pufferdrive.py")

        if not hasattr(self._native, "contacts"):
            raise RuntimeError("Native backend predates participant lifecycle; rebuild with scripts/build/build_pufferdrive.py")
        self._corner_boundary = scene.metadata.get("road_boundary_policy") == "nuplan_strict_corners_v1"
        if "road_boundary_policy" in scene.metadata and not self._corner_boundary:
            raise ValueError("Unsupported road boundary policy")
        if self._corner_boundary:
            if getattr(self._native, "corner_boundary_version", 0) != 1:
                raise RuntimeError("Rebuild native adapter for explicit nuPlan boundary profile")
            if scene.metadata.get("road_edge_model") != "clipped_official_nuplan_drivable_union_v2":
                raise ValueError("Corner profile requires the official nuPlan drivable union")
            from driversi.contracts.scene import ROAD_EDGE
            edges = [(a, b) for a, b, k in zip(scene.road_offsets[:-1], scene.road_offsets[1:], scene.road_types) if k == ROAD_EDGE]
            if not edges or any(b-a < 4 or not np.array_equal(scene.road_points[a], scene.road_points[b-1]) for a, b in edges):
                raise ValueError("Corner profile requires closed boundary rings")
        self._scene = copy.deepcopy(scene)
        self._scene.validate()
        self._tasks = tasks_from_metadata(self._scene.metadata, tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c))
        self._background = ReactiveBackground(self._scene)
        self._lifecycle = TrafficLifecycle(self._scene) if "traffic_continuity" in self._scene.metadata else None
        if self._lifecycle and getattr(self._native, "continuity_version", 0) != 1:
            raise RuntimeError("Rebuild native adapter for traffic continuity")
        if self._background.tasks and getattr(self._native, "background_state_version", 0) != 1:
            raise RuntimeError("Rebuild the native adapter for reactive background support")
        self._handle = None
        with tempfile.TemporaryDirectory(prefix="driversi-scene-") as directory:
            path = Path(directory) / "scene.bin"
            write_binary(self._scene, path)
            self._binary = path.read_bytes()
        self._scene_key = physical_scene_key(self._scene, self._binary)
        self.actor_ids = tuple(x for x, active in zip(scene.actor_ids, scene.controlled) if active)
        self.max_steps = len(scene.times) - 1
        self.reset()

    def __enter__(self) -> "PufferDriveWorld":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._native.close(self._handle)
            self._handle = None

    def _read(self):
        if self._handle is None:
            raise RuntimeError("World is closed; call reset to create a new episode")
        return self._native.read(self._handle)

    def reset(self) -> Observation:
        self.close()
        with tempfile.TemporaryDirectory(prefix="driversi-world-") as directory:
            path = Path(directory) / "scene.bin"
            path.write_bytes(self._binary)
            self._handle = (self._native.create(str(path), bool(self._lifecycle), True)
                            if self._corner_boundary else self._native.create(str(path), bool(self._lifecycle)))
        if self._lifecycle:
            self._lifecycle.reset()
        _, state, indices, _, _ = self._read()
        expected = np.flatnonzero(self._scene.controlled)
        if not np.array_equal(indices, expected):
            self.close()
            raise ValueError("Backend rejected the scene's explicit control mask")
        if "native_initial_motion" in self._scene.metadata:
            motion = self._scene.metadata["native_initial_motion"]
            if not isinstance(motion, dict) or set(motion) != set(self.actor_ids):
                self.close()
                raise ValueError("Initial model motion must cover policy actors")
            try:
                self._native.initialize_motion(self._handle, np.asarray([motion[a] for a in self.actor_ids],float))
            except BaseException:
                self.close()
                raise
        self._alive = state[indices, 9] == 0
        self.episode_done = not self._alive.any() and not self._background.fleet
        return self.observe()

    def observe(self) -> Observation:
        features, state, indices, _, step = self._read()
        motion = self.motion()
        current = np.column_stack((state[:, :2], state[:, 3:6], self._scene.dimensions[:, :2]))
        ego = np.column_stack((current[indices], motion[indices]))
        neighbors = np.broadcast_to(current, (len(indices), *current.shape)).copy()
        visible = np.linalg.norm(current[None, :, :2] - current[indices, None, :2], axis=-1) <= 50
        visible &= self.presence()[None, :]
        visible[np.arange(len(indices)), indices] = False
        neighbors[~visible] = 0
        ids = tuple(tuple(actor if visible[i, j] else "" for j, actor in enumerate(self._scene.actor_ids))
                    for i in range(len(indices)))
        return Observation(self.actor_ids, features, self._alive.copy(), step,
                           MetricObservation(ego, neighbors, visible, ids))

    def motion(self) -> np.ndarray:
        """All actors: actual a_long, a_lat (m/s²), steering (rad).

        Replay bypasses dynamics; its acceleration columns are not measured
        accelerations. Differentiate replay velocities for evaluation instead.
        """
        if self._handle is None:
            raise RuntimeError("World is closed; call reset to create a new episode")
        return self._native.motion(self._handle)

    def state(self) -> np.ndarray:
        """Copy ALL actors in scene order: xyz/yaw, velocity xyz, goal xy,
        stopped, collision, offroad, reached_goal. Coordinates use scene map axes.
        """
        state = self._read()[1]
        for i, actor in enumerate(self._scene.actor_ids):
            if actor in self._tasks and self._tasks[actor][1]["endpoint"] == "support_boundary":
                state[i, 12] = 0
        state[~self.presence()] = 0
        state[~self.presence(), 9] = 1
        return state

    def presence(self) -> np.ndarray:
        return (np.asarray(self._lifecycle.state["present"], bool) if self._lifecycle
                else self._scene.presence_at(self._read()[4]))

    def contacts(self):
        return tuple((self._scene.actor_ids[i], self._scene.actor_ids[j])
                     for i, j in self._native.contacts(self._handle))

    def lifecycle(self):
        tick=self._read()[4]
        now=self.presence()
        previous=(np.asarray(self._lifecycle.state["previous"],bool) if self._lifecycle else
                  self._scene.presence_at(tick-1) if tick else np.zeros_like(now))
        return {"tick":tick,"present":now.tolist(),
                "entered":[a for a,m in zip(self._scene.actor_ids, now & ~previous) if m],
                "exited":[a for a,m in zip(self._scene.actor_ids, previous & ~now) if m],
                "contacts":[list(pair) for pair in self.contacts()],
                **({"continuity": copy.deepcopy(self._lifecycle.state)} if self._lifecycle else {}),
                **self.background_diagnostics()}

    def background_diagnostics(self):
        if not self._background.tasks:
            return {}
        tick = self._read()[4]
        config = self._scene.metadata["reactive_background"]
        _, decisions = self._background.propose(self.state(), self.presence(), self.presence())
        return {"reactive_ids": list(self._background.tasks),
                "source_supported": config["source_presence"][tick],
                "prediction_tick": tick, "background_decisions": decisions,
                "lifetime": config["lifetime"]}

    def step_native(self, targets, *, observation_tick):
        """Commit model-native integration in the joint physics barrier.

        Each live policy actor supplies xyz,yaw,vxyz,a_long,a_lat,steer in scene
        map axes. Dynamics belong to the model executor, collisions to the world.
        """
        if observation_tick != self._read()[4]:
            raise ValueError("Stale native policy proposal")
        if getattr(self._native, "native_policy_state_version", 0) != 1:
            raise RuntimeError("Rebuild native adapter for model-native control")
        live = {a for a, valid in zip(self.actor_ids, self._alive) if valid}
        if not isinstance(targets, Mapping) or set(targets) != live:
            raise ValueError("Native targets must cover exactly the live controlled actors")
        rows = []
        for actor in self.actor_ids:
            if actor not in live:
                continue
            v = np.asarray(targets[actor], dtype=float)
            if v.shape != (10,) or not np.isfinite(v).all() or np.abs(v).max() > 1e7:
                raise ValueError("Invalid native next-state vector")
            i = self._scene.actor_ids.index(actor)
            if np.linalg.norm(v[:2] - self.state()[i, :2]) > 15:
                raise ValueError("Native proposal exceeds per-step displacement support")
            rows.append([i, *v])
        return self._step(np.zeros(len(self.actor_ids), dtype=np.int32),
                          np.asarray(rows, dtype=float).reshape(-1, 11))

    def step(self, actions: Mapping[str, int] | np.ndarray) -> Transition:
        return self._step(actions)

    def _step(self, actions, native_targets=None) -> Transition:
        if self.episode_done:
            raise RuntimeError("Episode is done; reset explicitly before stepping")
        if isinstance(actions, Mapping):
            if set(actions) != set(self.actor_ids):
                raise ValueError("Joint action keys must match every controlled actor ID")
            actions = np.asarray([actions[x] for x in self.actor_ids])
        else:
            actions = np.asarray(actions)
        if (actions.shape != (len(self.actor_ids),) or actions.dtype.kind not in "iu"
                or np.any(actions < 0) or np.any(actions >= 12)):
            raise ValueError("Expected one integer jerk action in [0, 11] per controlled actor")
        valid = self._alive.copy()
        tick = self._read()[4]
        traffic_controls = ()
        if self._background.tasks:
            current = self.state()
            background, decisions = self._background.propose(current, self.presence(), self._scene.presence_at(tick+1))
            if self._lifecycle:
                lifecycle, entrants = self._lifecycle.propose(tick, current, background)
                membership = np.asarray(lifecycle["present"], bool)
                background = background[membership[background[:,0].astype(int)]]
                decisions = [d for d in decisions if membership[self._scene.actor_ids.index(d["actor"])]]
                background = np.vstack((background, entrants))
                self._native.step(self._handle, actions.astype(np.int32), background, membership, native_targets)
                self._lifecycle.restore(lifecycle)
            else:
                self._native.step(self._handle, actions.astype(np.int32), background, None, native_targets)
            if self._background.fleet:
                traffic_controls = tuple({**d, "observation_tick": tick, "execution_tick": tick+1} for d in decisions)
        else:
            self._native.step(self._handle, actions.astype(np.int32), None, None, native_targets)
        _, state, indices, _, step = self._read()
        controlled = state[indices]
        events = {name: (controlled[:, column] != 0) & valid
                  for name, column in (("collision", 10), ("offroad", 11), ("goal", 12))}
        if self._corner_boundary:
            events["road_boundary_contact"] = np.asarray(self._native.boundary_contacts(self._handle))[indices] & valid
        failed = events["collision"] | events["offroad"]
        support_end = np.zeros(len(self.actor_ids), dtype=bool)
        if self._tasks:
            support_end = events["goal"] & np.asarray([
                self._tasks[a][1]["endpoint"] == "support_boundary" for a in self.actor_ids]) & ~failed
            events["goal"] &= np.asarray([self._tasks[a][1]["endpoint"] == "destination" for a in self.actor_ids])
            events["route_support_end"] = support_end
        rewards = np.where(failed, -1.0, events["goal"].astype(float)).astype(np.float32)
        terminated = valid & (controlled[:, 9] != 0)
        self._alive &= ~terminated
        terminated &= ~support_end
        truncated = support_end | (self._alive & (step >= self.max_steps))
        self.episode_done = bool(step >= self.max_steps or (not self._alive.any() and not self._background.fleet))
        return Transition(self.observe(), rewards, valid, terminated, truncated,
                          self.episode_done, events, traffic_controls)

    def snapshot(self) -> Snapshot:
        self._read()
        return Snapshot(self._scene_key, self._native.build_id,
                        self._native.snapshot(self._handle), tuple(bool(x) for x in self._alive),
                        bool(self.episode_done),
                        copy.deepcopy(self._lifecycle.state) if self._lifecycle else {})

    def restore(self, snapshot: Snapshot) -> Observation:
        self._read()
        if snapshot.scene_key != self._scene_key or snapshot.build_id != self._native.build_id:
            raise ValueError("Snapshot scene or backend build mismatch")
        if len(snapshot.alive) != len(self.actor_ids):
            raise ValueError("Snapshot actor mask mismatch")
        if self._lifecycle:
            # Validate before native mutation; never leave half-restored state.
            candidate = TrafficLifecycle(self._scene)
            candidate.restore(snapshot.runtime)
        elif snapshot.runtime:
            raise ValueError("Unexpected traffic lifecycle snapshot")
        previous_native = self._native.snapshot(self._handle)
        self._native.restore(self._handle, snapshot.native_state)
        if self._lifecycle:
            if (candidate.state["tick"] != self._read()[4]
                    or not np.array_equal(candidate.state["present"], self._native.presence(self._handle))):
                self._native.restore(self._handle, previous_native)
                raise ValueError("Lifecycle snapshot differs from native clock/membership")
            self._lifecycle = candidate
        self._alive = np.asarray(snapshot.alive, dtype=bool)
        self.episode_done = snapshot.episode_done
        return self.observe()

    def fork(self) -> "PufferDriveWorld":
        snapshot = self.snapshot()
        child = type(self)(self._scene)
        try:
            child.restore(snapshot)
        except Exception:
            child.close()
            raise
        return child
