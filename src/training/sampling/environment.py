"""Finite-horizon privileged observations, declared rewards and source splits."""
import json
import re
import uuid
from pathlib import Path

import numpy as np

from driversi.contracts.driving_task import route_progress, spec_options, tasks_from_metadata
from driversi.contracts.learning import ACTION_SIZE as ACTION_SIZE
from driversi.contracts.learning import (
    END_REASONS,
    ClosedLoopSpec,
    DecisionKey,
    JointDecision,
    LearningObservation,
    LearningStep,
)
from driversi.contracts.learning import OBSERVATION_SIZE as OBSERVATION_SIZE
from driversi.contracts.learning import OBSERVATION_VERSION as OBSERVATION_VERSION
from driversi.contracts.scene import Scene
from driversi.data.scenes.scene_store import scene_header
from driversi.sim.backends.pufferdrive import PufferDriveWorld


def features(observation, max_steps):
    x = np.asarray(observation.features, dtype=np.float32)
    if x.shape != (len(observation.actor_ids), 1124) or not np.isfinite(x).all():
        raise ValueError("Expected finite native current-state observations")
    remaining = (max_steps - observation.step) * .1 / 9.
    return np.column_stack((x, np.full(len(x), remaining, dtype=np.float32)))


def source_group(scene):
    # NAVSIM shards from one original drive must remain in one partition.
    return source_log_group(scene.metadata["source_log"])


def source_log_group(source_log):
    name = Path(source_log).name
    if name.endswith(".pkl"):
        name = name[:-4]
    return re.sub(r"_\d+_\d+$", "", name)


def validate_development_split(split):
    """Reject protected benchmark aliases before any development use.

    This validates declared metadata; upstream source provenance must still be
    checked independently. Case, separator and benchmark-version variants are
    not alternate routes for admitting navtest/navhard into a curriculum.
    """
    tokens = re.findall(r"[a-z0-9]+", str(split).casefold())
    compact = "".join(tokens)
    protected = {"test", "heldout", "holdout", "navtest", "navhard"}
    if (any(token in protected for token in tokens)
            or compact in protected or compact.startswith(("navtest", "navhard"))):
        raise ValueError("Official heldout/test/navtest/navhard data is not development data")


def validate_development_paths(paths):
    """Validate one development partition; return original source-log groups."""
    if not paths:
        raise ValueError("Development scene paths must not be empty")
    resolved = [str(Path(p).resolve()) for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate scene path in a development split")
    partition_groups = set()
    for path in resolved:
        metadata = scene_header(path)["metadata"]
        validate_development_split(metadata["split"])
        partition_groups.add(source_log_group(metadata["source_log"]))
    return partition_groups


def validate_splits(train, evaluation):
    if not train or not evaluation:
        raise ValueError("Both development training and evaluation scenes are required")
    if validate_development_paths(train) & validate_development_paths(evaluation):
        raise ValueError("Original source log overlaps training and evaluation")


def validate_initial_world(env):
    """Operational multi-actor admission, not route legality or driving quality."""
    state = env.world.state()[env.indices]
    if len(env.indices) < 2:
        raise ValueError("Fewer than two controlled actors")
    if not env.world.observe().alive.all() or state[:, 10:12].any():
        raise ValueError("Initially invalid controlled actor")
    if (np.linalg.norm(state[:, :2] - state[:, 7:9], axis=1) <= 2).any():
        raise ValueError("Controlled actor already at endpoint goal")



class TeacherEnvironment:
    """No auto-reset. All active controlled vehicles use one simultaneous joint action."""
    def __init__(self, scene, progress_weight=.02):
        if not np.isfinite(progress_weight) or progress_weight < 0:
            raise ValueError("progress_weight must be finite and nonnegative")
        self.world = PufferDriveWorld(scene)
        self.scene = scene
        self.progress_weight = float(progress_weight)
        self.indices = np.flatnonzero(scene.controlled)
        self.driving_tasks = tasks_from_metadata(self.world._scene.metadata, self.world.actor_ids)
        self.last_task_diagnostics = {}
        self.initial = self.world.snapshot()
        self.session_id = uuid.uuid4().hex
        self.generation = 0
        self.spec = ClosedLoopSpec(self.world.max_steps, self.progress_weight, dynamic_population=scene.presence is not None, **spec_options(scene.metadata))

    def observe(self):
        return features(self.world.observe(), self.world.max_steps)

    def observation(self):
        raw = self.world.observe()
        return LearningObservation(DecisionKey(self.session_id, self.generation, raw.step),
                                   raw.actor_ids, features(raw, self.world.max_steps), raw.alive,
                                   self.world.episode_done)

    def reset_episode(self) -> LearningObservation:
        """External model reset: return the new generation's identified observation."""
        self.world.restore(self.initial)
        self.last_task_diagnostics = {}
        self.generation += 1
        return self.observation()

    def reset(self):
        """Legacy writable ndarray reset; model adapters should use reset_episode."""
        return self.reset_episode().features.copy()

    def snapshot(self):
        return self.world.snapshot()

    def restore(self, snapshot):
        self.world.restore(snapshot)
        self.last_task_diagnostics = {}
        self.generation += 1  # pending pre-restore actions must never be reused
        return self.observation()

    def fork(self):
        child = type(self)(self.scene, self.progress_weight)
        try:
            child.world.restore(self.snapshot())
            return child
        except Exception:
            child.close()
            raise

    def advance(self, decision: JointDecision, overrides=None) -> LearningStep:
        current = self.observation()
        if decision.key != current.key or decision.actor_ids != current.actor_ids:
            raise ValueError("Stale/foreign decision identity or actor order")
        overrides = {} if overrides is None else overrides
        if not isinstance(overrides, dict) or set(overrides) - set(current.actor_ids):
            raise ValueError("Overrides must refer to controlled actors")
        if any(type(a) is not int or not 0 <= a < ACTION_SIZE for a in overrides.values()):
            raise ValueError("Invalid override action")
        executed = decision.actions.copy()
        for actor, action in overrides.items():
            executed[current.actor_ids.index(actor)] = action
        obs, rewards, raw, terms = self._apply(executed)
        reasons = np.zeros(len(current.actor_ids), dtype=np.uint8)
        for name in ("collision", "offroad", "goal"):
            reasons |= raw.events[name].astype(np.uint8) * END_REASONS[name]
        support = raw.events.get("route_support_end", np.zeros_like(raw.valid))
        reasons |= support.astype(np.uint8) * END_REASONS["route_support_end"]
        reasons |= (raw.truncated & ~support).astype(np.uint8) * END_REASONS["time_budget"]
        reasons |= (raw.terminated & (reasons == 0)).astype(np.uint8) * END_REASONS["other_absorbing"]
        next_observation = LearningObservation(
            DecisionKey(self.session_id, self.generation, raw.observation.step),
            raw.observation.actor_ids, obs, raw.observation.alive, raw.episode_done)
        return LearningStep(decision, next_observation, executed, rewards, terms, reasons,
                            raw.valid & ~raw.terminated & ~raw.truncated,
                            raw.valid & (executed == decision.actions) & (decision.log_probs is not None), raw)

    def step(self, actions):
        """Trusted internal compatibility API; external adapters should use advance."""
        obs, reward, transition, _terms = self._apply(actions)
        return obs, reward, transition

    def _apply(self, actions):
        before = self.world.state()[self.indices]
        transition = self.world.step(actions)
        after = self.world.state()[self.indices]
        progress = (np.linalg.norm(before[:, :2] - before[:, 7:9], axis=1)
                    - np.linalg.norm(after[:, :2] - after[:, 7:9], axis=1))
        self.last_task_diagnostics = {}
        if self.driving_tasks:
            for i, actor in enumerate(self.world.actor_ids):
                route, task = self.driving_tasks[actor]
                diagnostic = route_progress(route, before[i, :2], after[i, :2], corridor_m=task["corridor_m"])
                diagnostic["valid"] = bool(transition.valid[i])
                self.last_task_diagnostics[actor] = diagnostic
                progress[i] = diagnostic["progress_m"]
        reward = (transition.rewards + self.progress_weight * progress) * transition.valid
        terms = np.column_stack((transition.rewards, self.progress_weight * progress)) * transition.valid[:, None]
        return (features(transition.observation, self.world.max_steps), reward.astype(np.float32),
                transition, terms)

    def close(self):
        self.world.close()


def prepare_suite(scene_dir, output, *, train_per_city=8, eval_per_city=2):
    """Bounded engineering split, with every rejection retained. No official-test claim."""
    if output.exists():
        raise FileExistsError(output)
    if min(train_per_city, eval_per_city) < 1:
        raise ValueError("Positive scene limits required")
    candidates, rejected = {}, []
    for path in sorted(Path(scene_dir).glob("*.npz")):
        env = None
        try:
            scene = Scene.load(path)
            validate_development_split(scene.metadata["split"])
            env = TeacherEnvironment(scene)
            validate_initial_world(env)
            candidates.setdefault(scene.metadata["map_name"], []).append({
                "path": str(path.resolve()), "source_group": source_group(scene),
                "actors": len(env.indices), "seconds": env.world.max_steps * .1})
        except (ValueError, RuntimeError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
        finally:
            if env is not None:
                env.close()
    train, evaluation, unused = [], [], []
    for city, entries in sorted(candidates.items()):
        groups = sorted({e["source_group"] for e in entries})
        # Fixed, deterministic groups; all windows from selected eval logs excluded from train.
        eval_groups = set(groups[::4])
        for target, selected, limit in [(evaluation, eval_groups, eval_per_city),
                                         (train, set(groups) - eval_groups, train_per_city)]:
            eligible = [dict(e, city=city) for e in entries if e["source_group"] in selected]
            target.extend(eligible[:limit])
            unused.extend(eligible[limit:])
    validate_splits([e["path"] for e in train], [e["path"] for e in evaluation])
    result = {"schema_version": 1, "kind": "teacher-development-suite",
              "official_test": False, "train": train, "evaluation": evaluation,
              "rejected": rejected, "unused_eligible": unused,
              "limits": ["Short scenes with fixed policy ownership and optional logged replay lifecycles; goals follow compiled task endpoints",
                         "Geometry/traffic semantics and teacher quality not certified"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-city", type=int, default=8)
    parser.add_argument("--eval-per-city", type=int, default=2)
    args = parser.parse_args()
    result = prepare_suite(args.scene_dir, args.output, train_per_city=args.train_per_city,
                           eval_per_city=args.eval_per_city)
    print(json.dumps({k: len(result[k]) for k in ("train", "evaluation", "rejected", "unused_eligible")}))


if __name__ == "__main__":
    main()
