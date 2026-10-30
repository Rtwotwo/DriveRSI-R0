"""Bounded native-control training with persistent per-episode reward memory.

Every reward uses the actual current world; earlier rewards are never recomputed
using a later map or actor selection. World/native terminal disagreement rejects
learning samples instead of silently changing either collision definition.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from driversi.training.online_rl.online import Step
from driversi.training.sampling.environment import source_log_group
from driversi_agents.driverl.learning.reward_memory import RewardMemory, world_baselines


def training_goals(session, seed):
    """Appendix A reset-time weighted future goals, optionally lane projected."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)
    elapsed = (session.ego.time_us - session.clock0) / 1e6
    states = [
        e
        for e in session.scenario.get_ego_future_trajectory(
            0, 22.0 + elapsed, int(round((22.0 + elapsed) * 5))
        )
        if e.time_us > session.ego.time_us
    ]
    if len(states) < 2:
        raise ValueError("training goals require actual future source support")
    count = 1 if rng.random() < 0.5 else 2
    weights = np.arange(1, len(states) + 1, dtype=float)
    weights /= weights.sum()
    selected = sorted(rng.choice(len(states), size=count, replace=False, p=weights).tolist())
    points = []
    evidence = []
    for index in selected:
        state = states[index]
        point = Point(state.rear_axle.x, state.rear_axle.y)
        snap = rng.random() < 0.5
        candidates = []
        if snap:
            lanes = session.initialization.map_api.get_proximal_map_objects(
                Point2D(point.x, point.y), 20.0, [SemanticMapLayer.LANE]
            )[SemanticMapLayer.LANE]
            for lane in lanes:
                line = lane.baseline_path.linestring
                projected = line.interpolate(line.project(point))
                distance = point.distance(projected)
                if distance < 20:
                    candidates.append((distance, str(lane.id), projected))
            candidates.sort(key=lambda x: (x[0], x[1]))
            candidates = candidates[:4]
            if candidates:
                w = np.array([(1 - d / 20) ** 2 for d, _, _ in candidates])
                w /= w.sum()
                _, _, point = candidates[int(rng.choice(len(candidates), p=w))]
        points.append([point.x, point.y])
        evidence.append(
            dict(future_index=index, snapping_requested=snap, snapping_candidates=len(candidates))
        )
    if len(points) == 1:
        points *= 2
    return np.asarray(points), evidence


class NativeEnvironment:
    split = "train"

    def __init__(self, session, *, source_log, protected_logs, seed=42, horizon=110):
        from driversi_agents.driverl.closed_loop import ego_vector, pack_objects

        if source_log_group(source_log) in {source_log_group(x) for x in protected_logs}:
            raise ValueError("protected source")
        if session.reference or session.repair or session.search:
            raise ValueError("training cannot use correction/search")
        if type(horizon) is not int or not 1 <= horizon <= 110:
            raise ValueError("bounded 110-decision episode required")
        self.session = session
        self.source_log = source_log
        self.horizon = horizon
        self.steps = 0
        self.ended = False
        if int(session.world.observe().step) >= 20:
            self.history = copy.deepcopy(session.history)
        else:
            egos = list(
                session.scenario.get_ego_past_trajectory(0, time_horizon=2.1, num_samples=42)
            )
            observations = list(
                session.scenario.get_past_tracked_objects(0, time_horizon=2.1, num_samples=42)
            )
            if len(egos) != len(observations):
                raise ValueError("source history alignment")
            self.history = [
                dict(ego=ego_vector(e), objects=pack_objects(o), origin="source")
                for e, o in zip(egos, observations)
                if e.time_us < session.clock0
            ]
            # Runtime history may also contain source frames. Merge timestamps,
            # preferring actual runtime entries, and never duplicate/backtrack time.
            merged = {int(row["ego"][-1]): row for row in self.history}
            merged.update({int(row["ego"][-1]): copy.deepcopy(row) for row in session.history})
            self.history = [merged[t] for t in sorted(merged)]
        if self.history[0]["ego"][-1] > self.history[-1]["ego"][-1] - 2_000_000:
            raise ValueError("2s actual initialization history required")
        self.goals, self.goal_evidence = training_goals(session, seed)
        self.reward_components = np.zeros(9, dtype=np.float32)
        self.reward_builder = copy.deepcopy(session.planner._feature_builder)
        self.reward_builder.config.history_steps = 11
        from driverl.env.engine.config import EngineRuntimeConfig
        from driverl.env.engine.engine import Engine

        loaded = session.planner._loaded_agent
        runtime = EngineRuntimeConfig(
            batch_size=1,
            max_agents=self.reward_builder.config.max_agents,
            frame_time_interval=0.2,
            dynamics_model=loaded.env_config.dynamics_model,
            device=str(session.planner._device),
            num_steps=110,
            init_steps=10,
            enable_occupancy_grid=loaded.env_config.enable_occupancy_grid,
        )
        self.engine = Engine(copy.deepcopy(loaded.engine_config), runtime)
        self.last_reward_evidence = {}
        self.reward_memory = RewardMemory()
        self.physics_steps = 0
        self.world_events = []

    def _goals(self, data, ego=None):
        ego = self.session.ego if ego is None else ego
        h = ego.rear_axle.heading
        delta = self.goals - np.array([ego.rear_axle.x, ego.rear_axle.y])
        rotated = delta @ np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        # Native anchor origin is rear axle and orientation is current ego heading.
        goal = torch.as_tensor(
            rotated.reshape(4), dtype=data.goal_positions.dtype, device=data.goal_positions.device
        )
        data.goal_positions[:, 0, :, :] = goal

    @torch.no_grad()
    def observation(self):
        planner = self.session.planner
        data = planner._feature_builder.build(
            self.session.planner_input(), self.session.initialization
        )
        self._goals(data)
        features = planner._agent.preprocess(data)
        return {
            "features": tuple(
                None if x is None else x.detach().cpu().numpy().copy() for x in features
            ),
            "reward_components": self.reward_components.copy(),
            "tick": int(self.session.world.observe().step),
            "reward_protocol": "paper-hard-goal-native-six-soft-v1",
        }

    @torch.no_grad()
    def _reward(self, reward_ego=None):
        from driverl.env.engine.reward_calculator import InfoDimension

        from driversi_agents.driverl.closed_loop import unpack_ego, unpack_objects

        egos = [unpack_ego(x["ego"]) for x in self.history]
        if reward_ego is not None:
            egos[-1] = reward_ego
        objects = [unpack_objects(x["objects"], int(x["ego"][-1])) for x in self.history]
        inp = replace(
            self.session.planner_input(),
            history=SimpleNamespace(
                ego_states=egos, observations=objects, current_state=(egos[-1], objects[-1])
            ),
        )
        self.reward_builder.config.history_steps = 12
        current = self.reward_builder.build(inp, self.session.initialization)
        self._goals(current, egos[-1])
        config = current.randomized_features.config
        for name, value in (
            ("goal_reaching_threshold", 1.5),
            ("goal_reaching_weight", 0.5),
            ("goal_reaching_distance_weight", 0.0),
            ("survival_reward_numerator", 0.0),
            ("collision_reward_weight", -1.0),
        ):
            feature = getattr(config, name)
            feature.default = value
            feature.calculator_default = value
            current.randomized_features.values[name].fill_(value)
        keys = self.reward_builder._last_slot_keys
        baselines = world_baselines(current, egos[-1])
        self.reward_memory.apply(current, keys, baselines)
        current.clean_polygon_cache()
        current.update_nearest_neighbors()
        current.update_occ_surface_points(self.engine.config, self.engine._occ_ray_cache)
        results = {}
        for _, calculator in self.engine._reward_calculators:
            calculator.forward(current, current, results)
        mask = (results["OffRoad"]["info"] == 1) | results["OffRoad"]["occ_hit"].bool()
        results["OffRoad"]["reward"] = torch.where(
            mask, 2 * results["OffRoad"]["reward"], results["OffRoad"]["reward"]
        )
        total, done, infos, components = self.engine._combine_rewards(results, current)
        self.reward_memory.capture(
            current, keys, baselines, infos[0, 0, InfoDimension.GOAL_REACHED_FLAG.value]
        )
        if not torch.isfinite(components[0, 0]).all() or not torch.allclose(
            components.sum(-1), total, atol=1e-6
        ):
            raise ValueError("invalid decomposed reward")
        self.reward_components = components[0, 0].cpu().numpy().copy()
        self.last_reward_evidence = {
            key: float(val["reward"][0, 0]) for key, val in results.items()
        }
        self.last_reward_evidence.update(
            native_collision=bool(results["NuPlanCollision"]["info"][0, 0]),
            native_offroad=bool(
                results["OffRoad"]["info"][0, 0] in (1, 4) or results["OffRoad"]["occ_hit"][0, 0]
            ),
        )
        return float(total[0, 0]), bool(done[0, 0])

    def step(self, action):
        if self.ended:
            raise RuntimeError("episode ended")
        if not self.session.world.observe().alive[0]:
            raise RuntimeError("dead ego cannot produce on-policy samples")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("invalid action")
        executed = 0
        events = {}
        world_terminal = False
        world_truncated = False
        reward_ego = None
        for substep in range(2):
            if self.session.world.episode_done:
                break
            tr = self.session.step(native_action=action, policy_forward=substep == 0)
            self.physics_steps += 1
            self.history.append(copy.deepcopy(self.session.history[-1]))
            executed += 1
            events.update({k: bool(v[0]) or events.get(k, False) for k, v in tr.events.items()})
            world_terminal = bool(tr.terminated[0])
            world_truncated = bool(tr.truncated[0])
            if world_terminal or world_truncated or not tr.observation.alive[0]:
                # Controller state retains pre-contact kinematics, before the world's
                # absorbing-state zeroing. Only reward computation uses this state.
                reward_ego = self.session.controller.get_state()
                break
        self.steps += 1
        reward, terminal = self._reward(reward_ego)
        mismatch = (
            events.get("collision", False) and not self.last_reward_evidence["native_collision"]
        ) or (events.get("offroad", False) and not self.last_reward_evidence["native_offroad"])
        invalid = mismatch or (world_terminal and not terminal) or (executed != 2 and not terminal)
        reason = (
            "world_native_terminal_disagreement"
            if mismatch or (world_terminal and not terminal)
            else "partial_policy_interval"
            if invalid
            else None
        )
        truncated = not terminal and (
            invalid
            or world_truncated
            or self.steps >= self.horizon
            or self.session.world.episode_done
        )
        self.ended = terminal or truncated
        self.world_events.append(
            dict(
                tick=int(self.session.world.observe().step),
                events=events,
                physics_steps=executed,
                native_terminal=terminal,
                platform_valid=not invalid,
                reason=reason,
            )
        )
        return Step(
            self.observation(), action.copy(), reward, terminal, truncated, not invalid, reason
        )

    def snapshot(self):
        return copy.deepcopy(
            {
                "session": self.session.snapshot(),
                "source_log": self.source_log,
                "horizon": self.horizon,
                "steps": self.steps,
                "ended": self.ended,
                "history": self.history,
                "goals": self.goals,
                "goal_evidence": self.goal_evidence,
                "reward_components": self.reward_components,
                "reward_memory": self.reward_memory.snapshot(),
                "physics_steps": self.physics_steps,
                "world_events": self.world_events,
                "last_reward_evidence": self.last_reward_evidence,
            }
        )

    def restore(self, state):
        if state["source_log"] != self.source_log or state["horizon"] != self.horizon:
            raise ValueError("wrong environment recovery")
        self.session.restore(state["session"])
        self.reward_memory.restore(state["reward_memory"])
        for key in (
            "steps",
            "ended",
            "history",
            "goals",
            "goal_evidence",
            "reward_components",
            "physics_steps",
            "world_events",
            "last_reward_evidence",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
