"""Fixed-policy cross-matchups on source-separated development scenes.

Each opponent is evaluated separately against the same learner, scenes and random
streams. Endpoint/event metrics are engineering diagnostics, not PDMS or win rate.
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from driversi.agents.interfaces.config import load_inference, read_agent_config
from driversi.contracts.learning import ACTION_SIZE, JointDecision
from driversi.contracts.scene import Scene
from driversi.training.sampling.environment import (
    TeacherEnvironment,
    source_group,
    validate_initial_world,
    validate_splits,
)

_METRICS = ("unsafe", "collision", "offroad", "goal", "return_sum", "travel_m", "endpoint_progress_m")


def _predict(policy, observations):
    logits, values = policy.predict(observations)
    logits, values = np.asarray(logits), np.asarray(values)
    if (logits.shape != (len(observations), ACTION_SIZE) or values.shape != (len(observations),)
            or logits.dtype.kind not in "fiu" or values.dtype.kind not in "fiu"
            or not np.isfinite(logits).all() or not np.isfinite(values).all()):
        raise ValueError("Matchup inference must return finite logits and values")
    return logits


def _random_stream(seed, source, scene, actor):
    payload = json.dumps([source, scene, actor], ensure_ascii=True, separators=(",", ":"))
    words = np.frombuffer(hashlib.sha256(payload.encode()).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([seed, *map(int, words)]))


def _sample(logits, rngs):
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.array([rng.choice(ACTION_SIZE, p=p) for rng, p in zip(rngs, weights)], np.int64)


def _metrics(mask, events, returns, travel, progress, active_steps, action_counts):
    count = int(mask.sum())
    totals = {
        "actors": count,
        "unsafe": int(np.any(events[mask, :2], axis=1).sum()),
        "collision": int(events[mask, 0].sum()),
        "offroad": int(events[mask, 1].sum()),
        "goal": int(events[mask, 2].sum()),
        "return_sum": float(returns[mask].sum()),
        "travel_m": float(travel[mask].sum()),
        "endpoint_progress_m": float(progress[mask].sum()),
        "valid_agent_steps": int(active_steps[mask].sum()),
        "action_histogram": action_counts[mask].sum(axis=0).tolist(),
    }
    return totals


def _aggregate(episodes, role):
    total = {"actor_episodes": sum(e[role]["actors"] for e in episodes),
             "valid_agent_steps": sum(e[role]["valid_agent_steps"] for e in episodes)}
    for name in _METRICS:
        total[name] = sum(e[role][name] for e in episodes)
        total[name + "_per_actor"] = total[name] / total["actor_episodes"]
    return total


def evaluate_matchups(policy, paths, opponent_pool, *, training_paths, seed=0,
                      learner_count=1, progress_weight=.02, on_episode=None, replay_dir=None):
    """Run every frozen opponent on the same fixed development scene set.

    Training paths are explicit caller declarations, not independently certified
    model lineage. Missing declarations fail; original source-log overlaps fail.
    Recurrent/stateful policies are not supported by this stateless inference API.
    replay_dir optionally saves exact full-prefix BEV records; it never implies GS coverage.
    on_episode receives the cumulative completed-scene count across opponents,
    after world cleanup; it is an optional synchronous progress boundary.
    """
    from driversi.training.self_play.opponents import FrozenOpponentPool

    if type(seed) is not int or seed < 0:
        raise ValueError("Matchup seed must be a nonnegative integer")
    if type(learner_count) is not int or learner_count < 1:
        raise ValueError("learner_count must be a positive integer")
    if (policy is None or not callable(getattr(policy, "predict", None))
            or not isinstance(getattr(policy, "metadata", None), dict)
            or not isinstance(policy.metadata.get("policy_version"), str)
            or not policy.metadata["policy_version"].strip()):
        raise ValueError("A versioned frozen learner inference policy is required")
    if training_paths is None:
        raise ValueError("Learner training_paths declaration is required")
    paths = [str(Path(p).resolve()) for p in paths]
    train = [str(Path(p).resolve()) for p in training_paths]
    validate_splits(train, paths)
    pool = opponent_pool if isinstance(opponent_pool, FrozenOpponentPool) else FrozenOpponentPool(opponent_pool)
    validate_splits(list(pool.training_paths), paths)
    learner_version = policy.metadata["policy_version"]
    if replay_dir is not None:
        replay_dir = Path(replay_dir)
        replay_dir.mkdir(parents=True, exist_ok=False)
    matchups = []
    completed = 0
    for index, opponent in enumerate(pool.policies):
        episodes = []
        for scene_index, path in enumerate(paths):
            scene = Scene.load(Path(path))
            env = TeacherEnvironment(scene, progress_weight)
            try:
                validate_initial_world(env)
                n = len(env.indices)
                if learner_count >= n:
                    raise ValueError("Each matchup needs at least one learner and one frozen opponent")
                current = env.observation()
                learner_mask = np.arange(n) < learner_count
                rngs = [_random_stream(seed, source_group(scene), scene.metadata["scenario_id"], actor)
                        for actor in current.actor_ids]
                initial = env.world.state()[env.indices]
                events = np.zeros((n, 3), bool)
                returns, travel = np.zeros(n), np.zeros(n)
                active_steps = np.zeros(n, np.int64)
                action_counts = np.zeros((n, ACTION_SIZE), np.int64)
                record = None
                if replay_dir is not None:
                    from driversi.evaluation.benchmarks.matchup_replay import MatchupReplay
                    record = MatchupReplay(env.world)
                digest = hashlib.sha256()
                while not env.world.episode_done:
                    current = env.observation()
                    actions = np.full(n, 7, np.int64)
                    for mask, predict in ((learner_mask, lambda x: _predict(policy, x)),
                                          (~learner_mask, lambda x: pool.predict(index, x)[0])):
                        selected = np.flatnonzero(mask & current.alive)
                        if len(selected):
                            logits = predict(current.features[selected])
                            actions[selected] = _sample(logits, [rngs[i] for i in selected])
                    before = env.world.state()[env.indices]
                    decision = JointDecision(current.key, current.actor_ids, actions,
                                             f"matchup:{learner_version}:{opponent['policy_version']}")
                    step = env.advance(decision)
                    result = step.transition
                    if record is not None:
                        record.append(env.world, current.actor_ids, actions, result)
                    after = env.world.state()[env.indices]
                    travel += np.linalg.norm(after[:, :2] - before[:, :2], axis=1) * result.valid
                    returns += step.rewards
                    active_steps += result.valid
                    active = np.flatnonzero(result.valid)
                    np.add.at(action_counts, (active, actions[active]), 1)
                    events |= np.stack([result.events[k] for k in ("collision", "offroad", "goal")], axis=-1)
                    digest.update(actions.astype("<i8").tobytes())
                    digest.update(after.astype("<f8").tobytes())
                    digest.update(result.valid.tobytes())
                final = env.world.state()[env.indices]
                progress = (np.linalg.norm(initial[:, :2] - initial[:, 7:9], axis=1)
                            - np.linalg.norm(final[:, :2] - final[:, 7:9], axis=1))
                args = events, returns, travel, progress, active_steps, action_counts
                episodes.append({
                    "scene": path, "scenario_id": scene.metadata["scenario_id"],
                    "source_log": scene.metadata["source_log"], "source_group": source_group(scene),
                    "map_name": scene.metadata["map_name"], "seed": seed,
                    "steps": env.observation().key.tick,
                    "seconds": env.observation().key.tick * env.spec.dt_seconds,
                    "roles": [{"actor_id": actor, "role": "learner" if learner_mask[i] else "opponent",
                               "policy_version": learner_version if learner_mask[i] else opponent["policy_version"],
                               "policy_id": "learner" if learner_mask[i] else opponent["id"]}
                              for i, actor in enumerate(current.actor_ids)],
                    "learner": _metrics(learner_mask, *args),
                    "opponent": _metrics(~learner_mask, *args),
                    "all_world": _metrics(np.ones(n, bool), *args),
                    "joint_action_state_sha256": digest.hexdigest(),
                })
                if record is not None:
                    destination = replay_dir / f"opponent-{index:03d}" / f"scene-{scene_index:04d}"
                    episodes[-1]["replay"] = record.export(scene, destination, dict(episodes[-1]))
            finally:
                env.close()
            completed += 1
            if on_episode is not None:
                on_episode(completed)
        matchups.append({"opponent_id": opponent["id"], "opponent_version": opponent["policy_version"],
                         "opponent_checkpoint_sha256": opponent["checkpoint_sha256"],
                         "episodes": episodes, "physical_world_steps": sum(e["steps"] for e in episodes),
                         **{role: _aggregate(episodes, role) for role in ("learner", "opponent", "all_world")}})
    return {
        "schema_version": 1, "kind": "frozen-opponent-development-matchups",
        "learner_version": learner_version, "opponent_pool": pool.identity,
        "seed": seed, "learner_count": learner_count,
        "sampling": "categorical_numpy_per_actor_common_random_numbers_v1",
        "role_rule": "first_learner_count_controlled_in_stable_scene_order",
        "learner_training_paths": train, "opponent_training_paths": list(pool.training_paths),
        "evaluation_paths": paths, "source_separation_checked": True,
        "source_provenance": "declared_training_paths_not_independently_certified",
        "official_test": False, "matchups": matchups,
        "limits": ["Current collision/offroad/endpoint metrics; no PDMS, right-of-way or win-rate claim",
                   "All-world metrics include controlled learner and opponent actors, not replay actor episodes",
                   "Endpoint progress is Euclidean distance reduction, not legal route progress",
                   "Frozen stateless inference only; no policy update or adversarial curriculum"],
    }


def _write_report(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".matchup-", suffix=".json", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--learner", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--opponent-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, help="Optional new directory for exact BEV matchup replays")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learner-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    suite_bytes = args.suite.read_bytes()
    suite = json.loads(suite_bytes)
    if not isinstance(suite, dict) or any(not isinstance(suite.get(k), list) for k in ("train", "evaluation")):
        raise ValueError("Suite must declare learner train and evaluation scene paths")
    def suite_paths(key):
        result = []
        for item in suite[key]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
                raise ValueError("Suite scene entries need an explicit path")
            path = Path(item["path"])
            result.append(path if path.is_absolute() else args.suite.resolve().parent / path)
        return result
    config = read_agent_config(args.agent_config)
    learner_sha = hashlib.sha256(args.learner.read_bytes()).hexdigest()
    policy = load_inference(args.learner, config)
    if hashlib.sha256(args.learner.read_bytes()).hexdigest() != learner_sha:
        raise ValueError("Learner checkpoint changed while loading")
    report = evaluate_matchups(policy, suite_paths("evaluation"), args.opponent_pool,
                               training_paths=suite_paths("train"), seed=args.seed,
                               learner_count=args.learner_count, replay_dir=args.replay_dir)
    report["learner_checkpoint"] = {"path": str(args.learner.resolve()),
                                    "sha256": learner_sha,
                                    "agent_config": config}
    report["suite"] = {"path": str(args.suite.resolve()),
                       "sha256": hashlib.sha256(suite_bytes).hexdigest(),
                       "training_declaration": "suite.train"}
    _write_report(args.output, report)
    print(json.dumps({"matchups": len(report["matchups"]),
                      "actor_episodes": sum(m["all_world"]["actor_episodes"] for m in report["matchups"]),
                      "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
