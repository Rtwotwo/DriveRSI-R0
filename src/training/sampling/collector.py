"""Deterministic batched sampling over serial or process-sharded joint worlds."""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from driversi.contracts.learning import OBSERVATION_SIZE, JointDecision
from driversi.contracts.scene import MAX_WORLD_ACTORS
from driversi.data.scenes.scene_store import LazyScenes, scene_header
from driversi.training.legacy.advantages import generalized_advantage
from driversi.training.self_play.opponents import FrozenOpponentPool
from driversi.training.sampling.world_pool import WorldPool


def rollout_nbytes(steps, worlds, opponents=False, world_capacity=32, route_tasks=False):
    """Exact persistent NumPy array bytes, including GAE; excludes model and temporaries."""
    if any(type(v) is not int or v < 1 for v in (steps, worlds)):
        raise ValueError("Positive integer rollout steps/world count required")
    if type(opponents) is not bool:
        raise ValueError("opponents must be boolean")
    if type(world_capacity) is not int or not 32 <= world_capacity <= MAX_WORLD_ACTORS:
        raise ValueError(f"world_capacity must be 32..{MAX_WORLD_ACTORS}")
    extra = (32 * (1 + 8 + 1) if opponents else 0) + (32 if route_tasks else 0)
    return steps * worlds * (extra + 32 * (2 * OBSERVATION_SIZE * 4 + 4 * 4 + 4 * 8 + 3 + 3 + 2 * 4)
                             + 2 * world_capacity * (13 * 8 + 3 * 8 + 1) + 8)


class Collector:
    """Freeze one behavior policy per batch; persist final observations before resetting."""
    def __init__(self, paths, num_worlds=4, seed=0, progress_weight=.02, *, workers=0,
                 scene_cache_size=32, max_rollout_mib=1024, worker_timeout_seconds=30.,
                 opponent_pool=None, learner_count=1):
        if not paths or type(num_worlds) is not int or num_worlds < 1:
            raise ValueError("A nonempty scene set and positive world count are required")
        if type(max_rollout_mib) is not int or max_rollout_mib < 1:
            raise ValueError("Positive max_rollout_mib required")
        self.paths = [str(Path(p).resolve()) for p in paths]
        if type(learner_count) is not int or not 1 <= learner_count <= 31:
            raise ValueError("learner_count must be between 1 and 31")
        self.opponents = (opponent_pool if isinstance(opponent_pool, FrozenOpponentPool)
                          else FrozenOpponentPool(opponent_pool) if opponent_pool is not None else None)
        self.learner_count = learner_count
        if self.opponents is not None:
            for path in self.paths:
                if len(scene_header(path)["controlled_ids"]) <= learner_count:
                    raise ValueError("Mixed scenes require at least one learner and one frozen opponent")
            self.opponent_rng = np.random.default_rng(np.random.SeedSequence([seed, 741921]))
        self.route_tasks = any(scene_header(p)["spec_options"] for p in self.paths)
        self.world_capacity = max(32, max(len(scene_header(p)["actor_ids"]) for p in self.paths))
        self.scenes = LazyScenes(self.paths, scene_cache_size)
        self.rng = np.random.default_rng(seed)
        self.progress_weight = float(progress_weight)
        self.scene_indices = [int(self.rng.integers(len(paths))) for _ in range(num_worlds)]
        self.episodes = [0] * num_worlds
        self.slots = num_worlds * 32
        self.max_rollout_bytes = max_rollout_mib * 1024**2
        self.pool_options = dict(workers=workers, progress_weight=self.progress_weight,
                                 cache_size=scene_cache_size, timeout=worker_timeout_seconds)
        self.pool = WorldPool([self.paths[i] for i in self.scene_indices], **self.pool_options)
        self.assignments = {}
        self._adaptive_sampling = None
        self._sampling_revisions = [0] * num_worlds
        if self.opponents is not None:
            for world in range(num_worlds):
                self._assign(world)
        self.last_metrics = {}

    def _assign(self, world):
        ids = self.pool.current[world].observation.actor_ids
        indices = np.zeros(len(ids), dtype=np.int64)
        if self._adaptive_sampling is None:
            indices[self.learner_count:] = self.opponents.sample_indices(
                self.opponent_rng, len(ids) - self.learner_count)
        else:
            probabilities = [row["probability"] for row in self._adaptive_sampling["sampling"]["evidence"]]
            indices[self.learner_count:] = self.opponent_rng.choice(
                len(self.opponents), size=len(ids) - self.learner_count, p=probabilities) + 1
        self._sampling_revisions[world] = 0 if self._adaptive_sampling is None else self._adaptive_sampling["revision"]
        self.assignments[world] = {"actor_ids": list(ids), "policy_indices": indices.tolist()}

    @property
    def adaptive_sampling(self):
        return copy.deepcopy(self._adaptive_sampling)

    def _validate_curriculum_sources(self, report):
        from driversi.training.sampling.environment import validate_splits
        if (self.opponents is None or report.get("learner_count") != self.learner_count
                or set(report.get("learner_training_paths", [])) != set(self.paths)):
            raise ValueError("Curriculum learner roles or training paths differ from collector")
        validate_splits(self.paths, report.get("evaluation_paths", []))

    def refresh_opponent_sampling(self, report, *, optimizer_update, learner_version, **settings):
        """Validate fully before publishing. Existing assignments and RNG do not change."""
        from driversi.training.self_play.adaptive_opponents import refreshed_state
        self._validate_curriculum_sources(report)
        pending = refreshed_state(self._adaptive_sampling, report, self.opponents.identity,
                                  optimizer_update=optimizer_update, learner_version=learner_version, **settings)
        self._adaptive_sampling = pending
        return self.adaptive_sampling

    def behavior_policies(self, policy_version):
        if self.opponents is None:
            return [{"index": 0, "role": "learner", "policy_version": policy_version}]
        return self.opponents.behavior_policies(policy_version)

    @property
    def envs(self):
        """Legacy local inspection only; process-owned worlds use snapshots instead."""
        if self.pool.group is None:
            raise RuntimeError("Process-owned worlds cannot be inspected as local env handles")
        return [self.pool.group.envs[i] for i in range(self.pool.count)]

    def close(self):
        self.pool.close()
        self.scenes.cache.entries.clear()

    def snapshot(self):
        result = {"paths": self.paths.copy(), "progress_weight": self.progress_weight,
                  "scene_indices": self.scene_indices.copy(), "episodes": self.episodes.copy(),
                  "rng": copy.deepcopy(self.rng.bit_generator.state), "worlds": self.pool.snapshot()}
        if self.opponents is not None:
            result["opponents"] = {"identity": self.opponents.identity, "learner_count": self.learner_count,
                "rng": copy.deepcopy(self.opponent_rng.bit_generator.state),
                "assignments": [copy.deepcopy(self.assignments[w]) for w in range(self.pool.count)]}
        if self._adaptive_sampling is not None:
            result["adaptive_opponents"] = {"state": self.adaptive_sampling,
                                             "assignment_revisions": self._sampling_revisions.copy()}
        return result

    def restore(self, state):
        if (state["paths"] != self.paths or state["progress_weight"] != self.progress_weight
                or len(state["worlds"]) != self.pool.count
                or len(state["scene_indices"]) != self.pool.count
                or len(state["episodes"]) != self.pool.count
                or any(type(i) is not int or not 0 <= i < len(self.paths) for i in state["scene_indices"])
                or any(type(i) is not int or i < 0 for i in state["episodes"])):
            raise ValueError("Collector checkpoint configuration mismatch")
        extra = state.get("opponents")
        if (extra is not None) != (self.opponents is not None):
            raise ValueError("Collector opponent configuration mismatch")
        if self.opponents is not None:
            if (not isinstance(extra, dict) or set(extra) != {"identity", "learner_count", "rng", "assignments"}
                    or extra["identity"] != self.opponents.identity
                    or type(extra["learner_count"]) is not int or extra["learner_count"] != self.learner_count
                    or not isinstance(extra["assignments"], list) or len(extra["assignments"]) != self.pool.count):
                raise ValueError("Collector opponent configuration mismatch")
            for w, assignment in enumerate(extra["assignments"]):
                ids = scene_header(self.paths[state["scene_indices"][w]])["controlled_ids"]
                if (not isinstance(assignment, dict) or set(assignment) != {"actor_ids", "policy_indices"}
                        or assignment["actor_ids"] != ids or not isinstance(assignment["policy_indices"], list)
                        or len(assignment["policy_indices"]) != len(ids)
                        or any(type(v) is not int for v in assignment["policy_indices"])
                        or assignment["policy_indices"][:self.learner_count] != [0] * self.learner_count
                        or any(not 1 <= v <= len(self.opponents)
                               for v in assignment["policy_indices"][self.learner_count:])):
                    raise ValueError("Invalid checkpoint opponent assignment")
            opponent_rng = np.random.default_rng()
            opponent_rng.bit_generator.state = extra["rng"]
        adaptive, revisions = None, [0] * self.pool.count
        if "adaptive_opponents" in state:
            from driversi.training.self_play.adaptive_opponents import validate_state
            item = state["adaptive_opponents"]
            if self.opponents is None or not isinstance(item, dict) or set(item) != {"state", "assignment_revisions"}:
                raise ValueError("Invalid adaptive opponent checkpoint")
            adaptive = validate_state(item["state"], self.opponents.identity)
            self._validate_curriculum_sources(adaptive["report"])
            revisions = item["assignment_revisions"]
            if (not isinstance(revisions, list) or len(revisions) != self.pool.count
                    or any(type(v) is not int or not 0 <= v <= adaptive["revision"] for v in revisions)):
                raise ValueError("Invalid episode sampling revision")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        replacement = WorldPool([self.paths[i] for i in state["scene_indices"]],
                                snapshots=state["worlds"], **self.pool_options)
        self.pool.close()
        self.pool = replacement
        self.scene_indices = list(state["scene_indices"])
        self.episodes = list(state["episodes"])
        self.rng = rng
        self._adaptive_sampling = adaptive
        self._sampling_revisions = revisions.copy()
        if self.opponents is not None:
            self.opponent_rng = opponent_rng
            self.assignments = {w: copy.deepcopy(a) for w, a in enumerate(extra["assignments"])}

    def collect(self, model, steps, device="cpu", policy_version="development-unspecified",
                *, before_step=None):
        expected_bytes = rollout_nbytes(steps, self.pool.count, self.opponents is not None, self.world_capacity, self.route_tasks)
        if self._adaptive_sampling is not None:
            expected_bytes += steps * self.pool.count * 8
        if expected_bytes > self.max_rollout_bytes:
            raise MemoryError(f"Rollout needs {expected_bytes} array bytes, limit {self.max_rollout_bytes}; "
                              "reduce worlds/rollout_steps or explicitly raise max_rollout_mib")
        if self.pool.closed:
            raise RuntimeError("Collector world pool is closed")
        start = time.perf_counter()
        shape = (steps, self.slots)
        data = {k: np.zeros(shape + (OBSERVATION_SIZE,), np.float32)
                for k in ("observations", "next_observations")}
        data.update({k: np.zeros(shape, np.float32) for k in
                     ("rewards", "values", "next_values", "log_probs")})
        data.update({k: np.zeros(shape, np.int64) for k in ("actions", "scene_index", "episode", "tick")})
        data.update({k: np.zeros(shape, bool) for k in ("valid", "terminated", "truncated")})
        if self.opponents is not None:
            data["learner_mask"] = np.zeros(shape, bool)
            data["behavior_policy_index"] = np.full(shape, -1, np.int64)
            data["on_policy_eligible"] = np.zeros(shape, bool)
        worlds = self.pool.count
        if self._adaptive_sampling is not None:
            data["opponent_sampling_revision"] = np.zeros((steps, worlds), np.int64)
        data["world_before"] = np.zeros((steps, worlds, self.world_capacity, 13), np.float64)
        data["world_after"] = data["world_before"].copy()
        data["world_motion_before"] = np.zeros((steps, worlds, self.world_capacity, 3), np.float64)
        data["world_motion_after"] = data["world_motion_before"].copy()
        data["world_presence_before"] = np.zeros((steps, worlds, self.world_capacity), bool)
        data["world_presence_after"] = data["world_presence_before"].copy()
        data["world_actor_count"] = np.zeros((steps, worlds), np.int64)
        if self.route_tasks:
            data["route_support_end"] = np.zeros(shape, bool)
        data["events"] = np.zeros(shape + (3,), bool)
        model.eval()
        behavior = self.behavior_policies(policy_version) if self.opponents is not None else None
        inference_seconds = physics_seconds = reset_seconds = 0.
        resets = 0
        try:
            for t in range(steps):
                if self._adaptive_sampling is not None:
                    data["opponent_sampling_revision"][t] = self._sampling_revisions
                requests = []
                for w in range(worlds):
                    frame = self.pool.current[w]
                    request = frame.observation
                    n = len(request.actor_ids)
                    sl = slice(w * 32, w * 32 + n)
                    requests.append(request)
                    data["observations"][t, sl] = request.features
                    data["valid"][t, sl] = request.alive
                    if self.opponents is not None:
                        assignment = self.assignments[w]
                        if assignment["actor_ids"] != list(request.actor_ids):
                            raise ValueError("Controlled actor order changed within episode")
                        ownership = np.asarray(assignment["policy_indices"], dtype=np.int64)
                        data["behavior_policy_index"][t, sl] = ownership
                        data["learner_mask"][t, sl] = ownership == 0
                    data["scene_index"][t, sl] = self.scene_indices[w]
                    data["episode"][t, sl] = self.episodes[w]
                    data["tick"][t, sl] = request.key.tick
                    state = frame.state
                    data["world_presence_before"][t, w, :len(state)] = frame.presence
                    data["world_before"][t, w, :len(state)] = state
                    data["world_motion_before"][t, w, :len(state)] = frame.motion
                    data["world_actor_count"][t, w] = len(state)
                begin = time.perf_counter()
                if self.opponents is None:
                    with torch.no_grad():
                        logits, values = model(torch.as_tensor(data["observations"][t], device=device))
                        distribution = Categorical(logits=logits)
                        actions = distribution.sample()
                        data["actions"][t] = actions.cpu().numpy()
                        data["values"][t] = values.cpu().numpy()
                        data["log_probs"][t] = distribution.log_prob(actions).cpu().numpy()
                else:
                    self._infer_mixed(model, data, t, device)
                inference_seconds += time.perf_counter() - begin
                decisions = []
                for w, request in enumerate(requests):
                    sl = slice(w * 32, w * 32 + len(request.actor_ids))
                    versions = None
                    if self.opponents is not None:
                        versions = tuple(behavior[int(i)]["policy_version"]
                                         for i in data["behavior_policy_index"][t, sl])
                    decisions.append(JointDecision(request.key, request.actor_ids,
                        data["actions"][t, sl], policy_version, data["log_probs"][t, sl],
                        actor_policy_versions=versions))
                begin = time.perf_counter()
                if before_step is not None:
                    before_step(sum(int(request.alive.sum()) for request in requests))
                results = self.pool.step(decisions)
                physics_seconds += time.perf_counter() - begin
                replacements = []
                for w, result in enumerate(results):
                    frame = self.pool.current[w]
                    sl = slice(w * 32, w * 32 + len(requests[w].actor_ids))
                    data["next_observations"][t, sl] = frame.observation.features
                    data["rewards"][t, sl] = result.rewards
                    if self.opponents is not None:
                        if result.on_policy_eligible is None:
                            raise ValueError("Mixed sampling requires explicit execution eligibility")
                        data["on_policy_eligible"][t, sl] = result.on_policy_eligible
                    data["terminated"][t, sl] = result.terminated
                    data["truncated"][t, sl] = result.truncated
                    if self.route_tasks:
                        data["route_support_end"][t, sl] = result.events.get("route_support_end", False)
                    data["events"][t, sl] = np.stack([result.events[k] for k in
                                                      ("collision", "offroad", "goal")], axis=-1)
                    data["world_presence_after"][t, w, :len(frame.state)] = frame.presence
                    data["world_after"][t, w, :len(frame.state)] = frame.state
                    data["world_motion_after"][t, w, :len(frame.state)] = frame.motion
                    if result.episode_done:
                        index = int(self.rng.integers(len(self.paths)))
                        replacements.append((w, self.paths[index]))
                        self.scene_indices[w] = index
                        self.episodes[w] += 1
                begin = time.perf_counter()
                self.pool.reset(replacements)
                if self.opponents is not None:
                    for world, _path in replacements:
                        self._assign(world)
                reset_seconds += time.perf_counter() - begin
                resets += len(replacements)
                begin = time.perf_counter()
                if self.opponents is None:
                    with torch.no_grad():
                        data["next_values"][t] = model(torch.as_tensor(
                            data["next_observations"][t], device=device))[1].cpu().numpy()
                else:
                    self._infer_mixed(model, data, t, device, successor=True)
                inference_seconds += time.perf_counter() - begin
            learning = data["valid"]
            if self.opponents is not None:
                learning = learning & data["learner_mask"] & data["on_policy_eligible"]
            data["advantages"], data["returns"] = generalized_advantage(
                data["rewards"], data["values"], data["next_values"], learning,
                data["terminated"], data["truncated"])
        except BaseException:
            self.close()
            raise
        elapsed = time.perf_counter() - start
        valid = int(data["valid"].sum())
        self.last_metrics = {"sample_seconds": elapsed, "inference_seconds": inference_seconds,
            "physics_and_ipc_seconds": physics_seconds, "reset_seconds": reset_seconds,
            "resets": resets, "world_steps": steps * worlds, "valid_agent_transitions": valid,
            "valid_transitions_per_second": valid / elapsed, "slot_utilization": valid / data["valid"].size,
            "rollout_array_bytes": sum(value.nbytes for value in data.values())}
        if self.opponents is not None:
            self.last_metrics.update(learner_transitions=int((data["valid"] & data["learner_mask"]).sum()),
                opponent_transitions=int((data["valid"] & ~data["learner_mask"]).sum()),
                learning_transitions=int(learning.sum()))
        if self._adaptive_sampling is not None:
            self.last_metrics["opponent_sampling_revision"] = self._adaptive_sampling["revision"]
        return data

    def _infer_mixed(self, model, data, tick, device, *, successor=False):
        ownership = data["behavior_policy_index"][tick]
        features = data["next_observations" if successor else "observations"][tick]
        target = "next_values" if successor else "values"
        # Only live current actors need decisions. Successor values may be read
        # on a finite-task terminal row, but GAE masks all such bootstrap values.
        learner = (ownership == 0) & data["valid"][tick]
        if learner.any():
            with torch.no_grad():
                logits, values = model(torch.as_tensor(features[learner], device=device))
                if (tuple(logits.shape) != (int(learner.sum()), 12)
                        or tuple(values.shape) != (int(learner.sum()),)
                        or not torch.isfinite(logits).all() or not torch.isfinite(values).all()):
                    raise ValueError("Learner inference violated structured actor-critic contract")
                data[target][tick, learner] = values.cpu().numpy()
                if not successor:
                    distribution = Categorical(logits=logits)
                    actions = distribution.sample()
                    data["actions"][tick, learner] = actions.cpu().numpy()
                    data["log_probs"][tick, learner] = distribution.log_prob(actions).cpu().numpy()
        for index in range(len(self.opponents)):
            selected = (ownership == index + 1) & data["valid"][tick]
            if not selected.any():
                continue
            logits, values = self.opponents.predict(index, features[selected])
            data[target][tick, selected] = values
            if not successor:
                logits = logits.astype(np.float64)
                shifted = logits - logits.max(axis=1, keepdims=True)
                logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                cumulative = np.exp(logp).cumsum(axis=1)
                # Accumulation roundoff cannot manufacture an action outside [0, 11].
                cumulative[:, -1] = 1.
                actions = (self.opponent_rng.random(len(logp))[:, None] >= cumulative).sum(axis=1)
                data["actions"][tick, selected] = actions
                data["log_probs"][tick, selected] = logp[np.arange(len(logp)), actions]
