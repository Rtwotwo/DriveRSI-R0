"""Continuous structured sessions: recording boundaries never reset a physical world.

Policies must keep all mutable inference state in memories/RNG, not hidden on self.
The fixed description includes model/version and preprocessing identity. This adapter
retains the existing discrete-action contract; native trajectory models need their
own executor and are not silently quantized here.
"""

import copy
import json
from collections import deque

import numpy as np

from driversi.contracts.checkpoint import decode_world, encode_world
from driversi.contracts.map_support import crop_support
from driversi.contracts.transition import MetricObservation, Observation
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.execution.structured_history import StructuredHistory


def json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def pack_observation(obs):
    metric = (
        None
        if obs.metric is None
        else {
            "ego": obs.metric.ego.tolist(),
            "neighbors": obs.metric.neighbors.tolist(),
            "visible": obs.metric.visible.tolist(),
            "neighbor_ids": obs.metric.neighbor_ids,
        }
    )
    return {
        "actor_ids": list(obs.actor_ids),
        "features": obs.features.tolist(),
        "alive": obs.alive.tolist(),
        "step": obs.step,
        "metric": metric,
    }


def unpack_observation(value, expected):
    if value["actor_ids"] != list(expected.actor_ids) or type(value["step"]) is not int:
        raise ValueError("History actor identity or clock mismatch")
    features, alive = (
        np.asarray(value["features"], dtype=expected.features.dtype),
        np.asarray(value["alive"]),
    )
    if (
        features.shape != expected.features.shape
        or not np.isfinite(features).all()
        or alive.dtype != np.bool_
        or alive.shape != expected.alive.shape
    ):
        raise ValueError("Invalid history observation")
    m = value["metric"]
    if m is None or expected.metric is None:
        raise ValueError("Continuous structured session requires metric history")
    metric = MetricObservation(
        np.asarray(m["ego"], dtype=expected.metric.ego.dtype),
        np.asarray(m["neighbors"], dtype=expected.metric.neighbors.dtype),
        np.asarray(m["visible"]),
        tuple(tuple(x) for x in m["neighbor_ids"]),
    )
    for key in ("ego", "neighbors", "visible"):
        a, b = getattr(metric, key), getattr(expected.metric, key)
        if a.shape != b.shape or not np.isfinite(a).all():
            raise ValueError("Invalid metric history")
    if metric.visible.dtype != np.bool_ or len(metric.neighbor_ids) != len(expected.actor_ids):
        raise ValueError("Invalid history visibility")
    return Observation(expected.actor_ids, features, alive, value["step"], metric)


class ControllerPolicy:
    """Explicit bridge for existing stateless rule controllers, not a learned teacher."""

    def __init__(self, controller):
        self.controller = controller

    def describe(self):
        return {
            "version": "structured-controller-session-v1",
            "action_space": "jerk_discrete_12",
            "controller": self.controller.describe(),
            "hidden_state": False,
        }

    def initial_memories(self, actor_ids):
        return {a: {"decisions": 0} for a in actor_ids}

    def infer(self, observation, history, memories, rngs):
        commands = self.controller.commands(observation)
        for actor in observation.actor_ids:
            memories[actor]["decisions"] += 1
        return {c["actor"]: int(c["action"]) for c in commands}, memories, commands


class ContinuousSession:
    def __init__(self, scene, policy, *, seed=0, history_steps=20, window_steps=90):
        if type(history_steps) is not int or not 2 <= history_steps <= 256:
            raise ValueError("history_steps must be2..256")
        if type(window_steps) is not int or window_steps <= 0:
            raise ValueError("window_steps must be positive")
        self.policy = policy
        self.configuration = {
            "policy": json_copy(policy.describe()),
            "history_steps": history_steps,
            "window_steps": window_steps,
            "version": "continuous-session-v2",
        }
        self.world = PufferDriveWorld(scene)
        self.initial = self.world.snapshot()
        self.memories = json_copy(policy.initial_memories(self.world.actor_ids))
        if set(self.memories) != set(self.world.actor_ids):
            self.close()
            raise ValueError("Policy memories must cover every actor")
        seeds = np.random.SeedSequence(seed).spawn(len(self.world.actor_ids))
        self.rngs = {a: np.random.default_rng(s) for a, s in zip(self.world.actor_ids, seeds)}
        self.history = deque([self.world.observe()], maxlen=history_steps)
        try:
            self.structured_history = StructuredHistory(scene, self.world, history_steps)
        except BaseException:
            self.close()
            raise
        self.last_decision = None

    def model_input(self, **request):
        """Read-only causal input for a model-specific adapter, never source future."""
        return self.structured_history.sample(**request)

    def close(self):
        self.world.close()

    def snapshot(self):
        tick = self.world.observe().step
        return {
            "version": "continuous-session-v2",
            "world": encode_world(
                self.world.snapshot(),
                tick=tick,
                configuration=self.configuration,
                actor_ids=self.world._scene.actor_ids,
            ),
            "memories": json_copy(self.memories),
            "rngs": {a: copy.deepcopy(g.bit_generator.state) for a, g in self.rngs.items()},
            "history": [pack_observation(x) for x in self.history],
            "structured_history": self.structured_history.snapshot(),
        }

    def restore(self, value):
        if (
            set(value) != {"version", "world", "memories", "rngs", "history", "structured_history"}
            or value["version"] != "continuous-session-v2"
        ):
            raise ValueError("Unsupported session checkpoint")
        snap = decode_world(
            value["world"],
            initial=self.initial,
            configuration=self.configuration,
            actor_ids=self.world._scene.actor_ids,
        )
        memory = json_copy(value["memories"])
        if set(memory) != set(self.world.actor_ids) or set(value["rngs"]) != set(memory):
            raise ValueError("Checkpoint memory/RNG actor mismatch")
        rngs = {}
        for actor, state in value["rngs"].items():
            rngs[actor] = np.random.default_rng()
            rngs[actor].bit_generator.state = copy.deepcopy(state)
        history = [unpack_observation(v, self.world.observe()) for v in value["history"]]
        tick = value["world"]["tick"]
        expected_ticks = list(range(max(0, tick - self.history.maxlen + 1), tick + 1))
        if [o.step for o in history] != expected_ticks:
            raise ValueError("Checkpoint history must be contiguous and end at current tick")
        before = self.world.snapshot()
        try:
            observation = self.world.restore(snap)
            if pack_observation(observation) != pack_observation(history[-1]):
                raise ValueError("Checkpoint history terminal state differs from physical world")
            self.structured_history.restore(value["structured_history"], self.world)
        except BaseException:
            self.world.restore(before)
            raise
        self.memories, self.rngs = memory, rngs
        self.history = deque(history, maxlen=self.history.maxlen)
        self.last_decision = None
        return observation

    def step(self, overrides=None):
        if self.world.episode_done:
            raise RuntimeError("Episode done; no implicit reset")
        overrides = {} if overrides is None else overrides
        if (
            not isinstance(overrides, dict)
            or set(overrides) - set(self.world.actor_ids)
            or any(type(a) is not int or not 0 <= a < 12 for a in overrides.values())
        ):
            raise ValueError("Invalid explicit action override")
        if json_copy(self.policy.describe()) != self.configuration["policy"]:
            raise ValueError("Policy version changed during continuous episode")
        before = self.world.snapshot()
        proposal_rngs = copy.deepcopy(self.rngs)
        history_before = self.structured_history.records.copy()
        try:
            obs = self.world.observe()
            history_ticks = [x.step for x in self.history]
            request = self.configuration["policy"].get("structured_history")
            if request is not None:
                if not isinstance(request, dict) or not callable(
                    getattr(self.policy, "infer_structured", None)
                ):
                    raise ValueError("Declared structured history requires infer_structured")
                actions, memories, diagnostics = self.policy.infer_structured(
                    copy.deepcopy(obs),
                    self.model_input(**request),
                    json_copy(self.memories),
                    proposal_rngs,
                )
            else:
                actions, memories, diagnostics = self.policy.infer(
                    copy.deepcopy(obs),
                    tuple(copy.deepcopy(self.history)),
                    json_copy(self.memories),
                    proposal_rngs,
                )
            if json_copy(self.policy.describe()) != self.configuration["policy"]:
                raise ValueError("Policy version changed during inference")
            memories = json_copy(memories)
            if (
                set(memories) != set(self.world.actor_ids)
                or not isinstance(actions, dict)
                or set(actions) != set(self.world.actor_ids)
                or any(type(a) is not int or not 0 <= a < 12 for a in actions.values())
            ):
                raise ValueError("Policy must return complete valid actions and memories")
            if set(proposal_rngs) != set(self.rngs) or any(
                not isinstance(g, np.random.Generator)
                or type(g.bit_generator) is not type(self.rngs[a].bit_generator)
                for a, g in proposal_rngs.items()
            ):
                raise ValueError("Policy changed the per-actor RNG contract")
            executed = {**actions, **overrides}
            diagnostics = json_copy(diagnostics)
            # Distinguish the policy proposal from the actual diagnostic intervention.
            if isinstance(diagnostics, list):
                for command in diagnostics:
                    if isinstance(command, dict) and command.get("actor") in overrides:
                        actor = command["actor"]
                        command.update(
                            policy_proposed_action=actions[actor],
                            action=executed[actor],
                            intervention="explicit_same_anchor_action_override",
                        )
            transition = self.world.step(executed)
            support = crop_support(self.world._scene, self.world.state(), self.world.presence())
            self.structured_history.append_world(self.world)
        except BaseException:
            self.world.restore(before)
            self.structured_history.records = history_before
            raise
        self.memories = memories
        self.rngs = proposal_rngs
        self.history.append(copy.deepcopy(transition.observation))
        self.last_decision = {
            "tick": obs.step,
            "policy_actions": actions,
            "executed_actions": executed,
            "overrides": dict(overrides),
            "history_ticks": history_ticks,
            "window_index": obs.step // self.configuration["window_steps"],
            "diagnostics": diagnostics,
            "map_support": support,
        }
        return transition

    def advance(self, steps, *, record=None):
        """A recording/scheduling chunk, not an episode/reset boundary."""
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        count = 0
        while count < steps and not self.world.episode_done:
            transition = self.step()
            if record is not None:
                record.append(
                    self.world,
                    self.last_decision["executed_actions"],
                    transition,
                    self.last_decision["diagnostics"],
                )
            count += 1
        return count
