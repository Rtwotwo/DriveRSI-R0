"""Independent actor-episode metrics for frozen structured policies."""
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.training.sampling.environment import (
    TeacherEnvironment,
    validate_development_paths,
    validate_initial_world,
    validate_splits,
)


def evaluate(policy, paths, progress_weight=.02, *, training_paths=None, before_step=None, on_episode=None):
    """Development evaluation; on_episode runs after each completed world's cleanup.

    Unknown training sources cannot be checked for overlap. The optional callback
    is a synchronous progress boundary; failures propagate without hidden retries.
    """
    paths = list(paths)
    if training_paths is None:
        validate_development_paths(paths)
    else:
        validate_splits(list(training_paths), paths)
    episodes = []
    for path in paths:
        env = TeacherEnvironment(Scene.load(Path(path)), progress_weight)
        try:
            validate_initial_world(env)
            n = len(env.indices)
            events = np.zeros((n, 3), bool)
            endings = {k: np.zeros(n, bool) for k in ("route_support_end", "time_budget", "other_absorbing")}
            returns = np.zeros(n)
            distance = np.zeros(n)
            initial = env.world.state()[env.indices]
            while not env.world.episode_done:
                before = env.world.state()[env.indices]
                actions = np.full(n, 7, np.int64) if policy is None else policy.actions(env.observe())
                if before_step is not None:
                    before_step(int(env.observation().alive.sum()))
                _, reward, result = env.step(actions)
                after = env.world.state()[env.indices]
                distance += np.linalg.norm(after[:, :2] - before[:, :2], axis=1) * result.valid
                returns += reward
                events |= np.stack([result.events[k] for k in ("collision", "offroad", "goal")], axis=-1)
                support = result.events.get("route_support_end", np.zeros(n, bool))
                endings["route_support_end"] |= support
                endings["time_budget"] |= result.truncated & ~support
                endings["other_absorbing"] |= result.terminated & ~support & ~events.any(axis=1)
            final = env.world.state()[env.indices]
            episodes.append({"scene": str(path), "actors": n, "steps": env.world.observe().step,
                             "collision": int(events[:, 0].sum()), "offroad": int(events[:, 1].sum()),
                             "goal": int(events[:, 2].sum()), "return_sum": float(returns.sum()),
                             "endings": {k: int(v.sum()) for k, v in endings.items()},
                             "travel_m": float(distance.sum()),
                             "endpoint_progress_m": float((np.linalg.norm(initial[:, :2] - initial[:, 7:9], axis=1)
                                                           - np.linalg.norm(final[:, :2] - final[:, 7:9], axis=1)).sum())})
        finally:
            env.close()
        if on_episode is not None:
            on_episode(len(episodes))
    actors = sum(e["actors"] for e in episodes)
    return {"episodes": episodes, "actor_episodes": actors,
            "source_separation_checked": training_paths is not None,
            **{k + "_per_actor": sum(e[k] for e in episodes) / actors for k in
               ("collision", "offroad", "goal", "return_sum", "travel_m", "endpoint_progress_m")}}


