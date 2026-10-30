"""Managed visual decisions over one existing world; no second physics engine."""

import copy
import hashlib
import json
import os
import time

import numpy as np

from driversi.agents.visual_control.image_moment import ImageMomentPolicy
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.sampling import ActorVisualInput, CameraInput, policy_description


def visual_population(controlled, camera_owners, selected, source, desired_speed, policy=None):
    description = policy_description(policy or ImageMomentPolicy())
    actors = tuple(selected) if selected else tuple(a for a in controlled if a in camera_owners)
    if (not actors or len(set(actors)) != len(actors)
            or not set(actors).issubset(set(controlled) & set(camera_owners))):
        raise ValueError("Visual actors must be unique controlled camera owners")
    if source is None:
        source = EpisodePopulation((PopulationPolicy("default", "route-idm-v2", "route-idm-v2",
                                   (("desired_speed", desired_speed),)),),
                                   tuple(ActorPolicyBinding(a, "default") for a in controlled))
    source.validate_actors(tuple(controlled))
    if source.external_actor_ids:
        raise ValueError("Managed visual mode cannot invent inputs for existing external policies")
    identity = ("managed-visual-diagnostic" if description == ImageMomentPolicy().describe()
                else "managed-visual")
    if any(p.policy_id == identity for p in source.policies):
        raise ValueError("Reserved visual diagnostic policy ID")
    assignments = tuple(ActorPolicyBinding(b.actor_id, identity if b.actor_id in actors else b.policy_id, b.role)
                        for b in source.assignments)
    used = {b.policy_id for b in assignments}
    policies = tuple(p for p in source.policies if p.policy_id in used) + (
        PopulationPolicy(identity, description["version"], "external-joint-v1"),)
    return EpisodePopulation(policies, assignments), actors


def body_camera_transform(camera_pose, body_pose):
    """MTGS right/up/back camera -> optical right/down/forward -> body FLU."""
    camera = np.eye(4)
    camera[:3] = camera_pose
    return np.linalg.inv(body_pose) @ camera @ np.diag([1., -1., -1., 1.])


class ManagedVisualSampler:
    version = "managed-visual-sampling-v1"

    def __init__(self, session, actor_ids, *, history=2, seed=0, policy=None):
        if type(history) is not int or not 1 <= history <= 16 or type(seed) is not int or seed < 0:
            raise ValueError("Invalid visual history/seed")
        self.session, self.actor_ids = session, tuple(actor_ids)
        self.history_size, self.seed = history, seed
        self.policy = policy or ImageMomentPolicy()
        self.description = policy_description(self.policy)
        self.reset()
        session.manifest["sampling"] = {"version": self.version, "decision_mode": "visual",
            "visual_actor_ids": list(self.actor_ids), "history": history, "seed": seed,
            "policy": self.description, "trainable": False,
            "scope": "synthetic_sampling_not_qualified_for_training",
            "input_version": "actor-owned-heterogeneous-rgb-v1", "recording": "synchronous_lossless"}
        session.manifest["decisions"] = []
        session.manifest["protocol"].update(kind="managed_visual_sampling", rgb_is_policy_input=True,
                                             policy_inputs_recorded=True)
        session.manifest["limits"] = [x for x in session.manifest["limits"]
                                      if x != "external_policy_inputs_unrecorded"]
        session.manifest["limits"].append("visual_policy_driving_quality_unverified")

    def reset(self):
        self.histories = {a: [] for a in self.actor_ids}
        self.memories = np.array(self.policy.initial_memories(len(self.actor_ids)), copy=True)
        self.rngs = tuple(np.random.default_rng(np.random.SeedSequence([
            self.seed, int.from_bytes(hashlib.sha256(a.encode()).digest()[:8], "little")])) for a in self.actor_ids)
        if self.memories.ndim != 2 or self.memories.shape[0] != len(self.actor_ids) or not np.isfinite(self.memories).all():
            raise ValueError("Policy initial memories must be finite [actor,memory]")
        self.held_actions, self.held_record = None, None
        self.observed_tick = -1
        self.file_hashes = {}

    def _camera_record(self, camera):
        def artifact(path):
            p = (self.session.output/path).resolve()
            if not p.is_relative_to(self.session.output.resolve()) or not p.is_file():
                raise ValueError("Input artifacts must belong to the session")
            if path not in self.file_hashes:
                self.file_hashes[path] = hashlib.sha256(p.read_bytes()).hexdigest()
            return {"path": path, "sha256": self.file_hashes[path]}
        return {"camera_id": camera.camera_id, "width": camera.rgb.shape[1], "height": camera.rgb.shape[0],
                "rgb": artifact(camera.image_path), "pixel_valid": artifact(camera.valid_path),
                "rgb_pixel_sha256": hashlib.sha256(camera.rgb.tobytes()).hexdigest(),
                "valid_pixel_sha256": hashlib.sha256(camera.pixel_valid.tobytes()).hexdigest(),
                "intrinsic": camera.intrinsic.tolist(), "distortion": camera.distortion.tolist(),
                "camera_to_body": camera.camera_to_body.tolist(), "camera_model": camera.camera_model,
                "optical_axes": "right_down_forward", "body_axes": "forward_left_up",
                "timestamp_us": camera.timestamp_us}

    def _observe(self):
        session = self.session
        tick = session.current["tick"]
        if session.latest["tick"] != tick or session.latest["generation"] != session.generation:
            raise ValueError("A new visual decision requires a current-generation current-tick capture")
        states = np.asarray(session.current["state"])
        motion = np.asarray(session.current["motion"])
        observations = []
        for actor in self.actor_ids:
            views = []
            for metadata in session.latest["cameras"]:
                if metadata["owner_actor_id"] != actor:
                    continue
                pixels = session._sensor_buffers[metadata["camera_id"]]
                views.append(CameraInput(metadata["camera_id"], pixels["rgb"], pixels["valid"],
                    np.asarray(metadata["K"]), np.asarray(metadata["distortion"]),
                    np.asarray(metadata["camera_to_body"]), metadata["camera_model"],
                    metadata["camera_timestamp_us"], metadata["image"], metadata["valid"]))
            history = self.histories[actor]
            history.append((tick, tuple(views)))
            del history[:-self.history_size]
            i = session.current["all_actor_ids"].index(actor)
            s = states[i]
            c, sn = np.cos(s[3]), np.sin(s[3])
            goal = s[7:9]-s[:2]
            telemetry = np.asarray([np.hypot(s[4], s[5]), *motion[i]])
            observations.append(ActorVisualInput(actor, tuple(v for _, v in history),
                tuple(t for t, _ in history), telemetry, np.asarray([c*goal[0]+sn*goal[1], -sn*goal[0]+c*goal[1]]),
                bool(session.current["alive"][session.current["controlled_ids"].index(actor)])))
        return tuple(observations)

    def advance(self, overrides=None):
        session = self.session
        if policy_description(self.policy) != self.description:
            raise ValueError("Policy descriptor changed within a collection session")
        tick = session.current["tick"]
        identity = session.decision_identity()
        started = time.perf_counter()
        new = self.held_actions is None or session.latest["tick"] > self.observed_tick
        if new:
            # Inference/serialization failure leaves no silently advanced actor state.
            before_histories = {a: list(h) for a, h in self.histories.items()}
            before_memory = self.memories.copy()
            before_rng = [copy.deepcopy(r.bit_generator.state) for r in self.rngs]
            try:
                observations = self._observe()
                result = self.policy.infer(observations, self.memories.copy(), self.rngs)
                result.validate(len(self.actor_ids))
                if result.memories.shape != before_memory.shape:
                    raise ValueError("Policy memory shape changed within a session")
                identifier = len(session.manifest["decisions"])
                record = {"schema_version": 1, "decision_id": identifier, "identity": identity,
                    "observation_tick": tick, "policy": self.description,
                    "preprocessing": "identity_uint8_rgb_v1",
                    "adapter_preprocessing": self.description["preprocessing"], "actors": [{
                        "actor_id": obs.actor_id, "history_ticks": list(obs.history_ticks),
                        "history": [[self._camera_record(v) for v in views] for views in obs.history],
                        "telemetry": obs.telemetry.tolist(), "navigation": obs.navigation.tolist(), "alive": obs.alive}
                        for obs in observations],
                    "memory_before": before_memory.tolist(), "memory_after": result.memories.tolist(),
                    "rng_before": before_rng, "rng_after": [copy.deepcopy(r.bit_generator.state) for r in self.rngs],
                    "actions": {a: int(v) for a, v in zip(self.actor_ids, result.actions)},
                    "diagnostics": list(result.diagnostics), "trainable": False}
                raw = json.dumps(record, allow_nan=False, ensure_ascii=False)
                with (session.output/"decisions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(raw+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                session.manifest["decisions"].append(record)
                self.memories = np.array(result.memories, copy=True)
                self.held_actions, self.held_record, self.observed_tick = record["actions"], record, tick
            except BaseException:
                self.histories, self.memories = before_histories, before_memory
                for rng, state in zip(self.rngs, before_rng):
                    rng.bit_generator.state = state
                raise
        reference = {"decision_id": self.held_record["decision_id"],
                     "observation_tick": self.observed_tick, "newly_decided": new,
                     "input_mode": "visual", "collector_seconds": time.perf_counter()-started}
        return session.advance(overrides, external_actions=dict(self.held_actions), expected_tick=tick,
                               decision_identity=identity, policy_decision=reference)

    def snapshot(self):
        def camera(view):
            return {name: copy.deepcopy(getattr(view, name)) for name in CameraInput.__dataclass_fields__}
        return {"version": "managed-sampler-state-v1", "actor_ids": list(self.actor_ids),
                "policy": copy.deepcopy(self.description), "history_size": self.history_size, "seed": self.seed,
                "histories": {a: [(tick, [camera(v) for v in views]) for tick, views in h]
                              for a, h in self.histories.items()},
                "memories": self.memories.copy(), "rngs": [copy.deepcopy(r.bit_generator.state) for r in self.rngs],
                "held_actions": copy.deepcopy(self.held_actions), "held_record": copy.deepcopy(self.held_record),
                "observed_tick": self.observed_tick}

    def restore(self, value):
        if (value.get("version") != "managed-sampler-state-v1" or value["actor_ids"] != list(self.actor_ids)
                or value["policy"] != self.description or value["history_size"] != self.history_size
                or value["seed"] != self.seed):
            raise ValueError("Incompatible managed sampler checkpoint")
        memories = np.asarray(value["memories"])
        if memories.shape != self.memories.shape or not np.isfinite(memories).all():
            raise ValueError("Invalid restored policy memories")
        histories = {}
        for actor in self.actor_ids:
            rows = value["histories"][actor]
            ticks = [t for t, _ in rows]
            if len(rows) > self.history_size or any(type(t) is not int or t < 0 for t in ticks) or ticks != sorted(set(ticks)):
                raise ValueError("Invalid restored image history")
            histories[actor] = [(t, tuple(CameraInput(**v) for v in views)) for t, views in rows]
        rngs = tuple(np.random.default_rng() for _ in self.actor_ids)
        for rng, state in zip(rngs, value["rngs"], strict=True):
            rng.bit_generator.state = copy.deepcopy(state)
        actions, record, tick = value["held_actions"], value["held_record"], value["observed_tick"]
        if (type(tick) is not int or tick < -1 or (actions is None) != (record is None)
                or (actions is not None and (set(actions) != set(self.actor_ids)
                    or any(type(a) is not int or not 0 <= a < 12 for a in actions.values())
                    or record["actions"] != actions or record["observation_tick"] != tick
                    or any(not h or h[-1][0] != tick for h in histories.values())))):
            raise ValueError("Invalid restored held decision")
        self.histories, self.memories, self.rngs = histories, memories.copy(), rngs
        self.held_actions, self.held_record = copy.deepcopy(actions), copy.deepcopy(record)
        self.observed_tick, self.file_hashes = tick, {}
