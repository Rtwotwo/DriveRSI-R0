"""Safety-based PFSP refreshes; frozen opponents remain immutable within episodes.

This adapts match difficulty, not a zero-sum win rate. No collision/offroad is
insufficient evidence of task completion or realistic traffic. Matchups use one
opponent policy for all other actors; this is a proxy for the independently
sampled per-actor mixtures used in training, not measured mixed-team difficulty.
"""
import copy

import numpy as np

from driversi.training.self_play.opponent_priority import priority_from_report


def pfsp_from_report(report, identity, *, uniform_mix=.2, power=1., min_actor_episodes=20):
    sampling = priority_from_report(report, identity, uniform_mix=uniform_mix,
                                    power=power, min_actor_episodes=min_actor_episodes)
    failure = np.array([row["smoothed_failure_rate"] for row in sampling["evidence"]])
    weights = (failure * (1 - failure)) ** power
    probabilities = uniform_mix / len(weights) + (1 - uniform_mix) * weights / weights.sum()
    for row, probability in zip(sampling["evidence"], probabilities):
        row["probability"] = float(probability)
    sampling.update(method="pfsp_safety_variance_v1",
                    scope="adaptive_safety_curriculum_not_game_win_rate_or_task_completion",
                    weighting="(smoothed_failure * (1 - smoothed_failure)) ** power")
    return sampling


def refreshed_state(previous, report, identity, *, optimizer_update, learner_version, **settings):
    if type(optimizer_update) is not int or optimizer_update < 0:
        raise ValueError("Nonnegative optimizer update required")
    if report.get("learner_version") != learner_version:
        raise ValueError("Curriculum learner version differs from the frozen current learner")
    if previous is not None and optimizer_update <= previous["optimizer_update"]:
        raise ValueError("A refresh requires a newer optimizer update")
    sampling = pfsp_from_report(report, identity, **settings)
    revision = 1 if previous is None else previous["revision"] + 1
    history = [] if previous is None else copy.deepcopy(previous["history"])
    history.append({"revision": revision, "optimizer_update": optimizer_update,
                    "learner_version": learner_version, "sampling": sampling})
    return copy.deepcopy({"schema_version": 1, "revision": revision,
                          "optimizer_update": optimizer_update, "learner_version": learner_version,
                          "settings": settings, "report": report, "sampling": sampling, "history": history})


def validate_state(value, identity):
    fields = {"schema_version", "revision", "optimizer_update", "learner_version",
              "settings", "report", "sampling", "history"}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Invalid adaptive opponent checkpoint")
    revision, history = value["revision"], value["history"]
    if type(revision) is not int or revision < 1 or not isinstance(history, list) or len(history) != revision:
        raise ValueError("Invalid adaptive opponent revision history")
    previous = -1
    for index, row in enumerate(history, 1):
        if (not isinstance(row, dict) or set(row) != {"revision", "optimizer_update", "learner_version", "sampling"}
                or type(row["revision"]) is not int or row["revision"] != index
                or type(row["optimizer_update"]) is not int or row["optimizer_update"] <= previous):
            raise ValueError("Invalid adaptive opponent revision history")
        previous = row["optimizer_update"]
    expected = refreshed_state(None, value["report"], identity, optimizer_update=value["optimizer_update"],
                               learner_version=value["learner_version"], **value["settings"])
    if (expected["sampling"] != value["sampling"] or history[-1] != {
            "revision": revision, "optimizer_update": value["optimizer_update"],
            "learner_version": value["learner_version"], "sampling": value["sampling"]}):
        raise ValueError("Adaptive checkpoint evidence and probabilities disagree")
    return copy.deepcopy(value)


DEFAULT_ADAPTIVE = dict(interval_updates=10, uniform_mix=.2, power=1.,
                        min_actor_episodes=20, max_world_steps=100000, seed=0)


def resolve_adaptive(raw):
    """Opt-in config. interval_updates counts successful PPO batch updates."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_ADAPTIVE):
        raise ValueError("Unknown adaptive_opponents configuration")
    result = {**DEFAULT_ADAPTIVE, **raw}
    for key in ("interval_updates", "min_actor_episodes", "max_world_steps", "seed"):
        if type(result[key]) is not int or result[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid adaptive_opponents {key}")
    from driversi.training.self_play.opponent_priority import failure_probabilities
    failure_probabilities([(result["min_actor_episodes"], 0)], **sampling_settings(result))
    return result


def sampling_settings(settings):
    return {key: settings[key] for key in ("uniform_mix", "power", "min_actor_episodes")}


def validate_curriculum(paths, training_paths, evaluation_paths, opponents, learner_count, settings):
    """Separate train/curriculum/final evaluation, and bound each complete refresh."""
    from driversi.data.scenes.scene_store import scene_header
    from driversi.training.sampling.environment import validate_splits
    if opponents is None:
        raise ValueError("Adaptive opponents require a frozen opponent pool")
    validate_splits(training_paths, paths)
    validate_splits(opponents.training_paths, paths)
    validate_splits(paths, evaluation_paths)
    headers = [scene_header(path) for path in paths]
    if any(len(h["controlled_ids"]) <= learner_count for h in headers):
        raise ValueError("Curriculum scenes require learner and opponent actors")
    if len(paths) * learner_count < settings["min_actor_episodes"]:
        raise ValueError("Insufficient curriculum actor episodes per opponent")
    bound = sum(h["max_steps"] for h in headers) * len(opponents.identity["policies"])
    if bound > settings["max_world_steps"]:
        raise ValueError(f"Curriculum step bound {bound} exceeds max_world_steps")
    return bound


def refresh_for_training(model, agent, collector, paths, evaluation_paths, output,
                         settings, optimizer_updates, learner_version):
    """Freeze current learner, evaluate a disjoint cohort, then publish atomically.

    A refresh is synchronous at an optimizer boundary. Failure leaves the sampler
    unchanged and aborts training; no incomplete report is used as evidence.
    Simulation-step budget is separate from training samples, not a wall timeout.
    """
    import json
    import time
    from pathlib import Path

    from driversi.agents.interfaces.config import load_inference
    from driversi.evaluation.benchmarks.matchup import evaluate_matchups

    state = collector.adaptive_sampling
    if state is not None:
        delta = optimizer_updates - state["optimizer_update"]
        if delta < 0:
            raise ValueError("Adaptive checkpoint update exceeds training counter")
        if delta < settings["interval_updates"]:
            return None
    begin = time.perf_counter()
    bound = validate_curriculum(paths, collector.paths, evaluation_paths,
                                collector.opponents, collector.learner_count, settings)
    revision = 1 if state is None else state["revision"] + 1
    directory = Path(output) / "curriculum" / f"revision-{revision:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = directory / "learner.npz"
        model.export(checkpoint, learner_version)
        report = evaluate_matchups(load_inference(checkpoint, agent), paths, collector.opponents,
                                  training_paths=collector.paths, seed=settings["seed"],
                                  learner_count=collector.learner_count,
                                  progress_weight=collector.progress_weight)
        actual = sum(row["physical_world_steps"] for row in report["matchups"])
        if actual > bound:
            raise RuntimeError("Curriculum exceeded its declared simulation horizon")
        pending = refreshed_state(state, report, collector.opponents.identity,
                                  optimizer_update=optimizer_updates, learner_version=learner_version,
                                  **sampling_settings(settings))
        summary = {"revision": revision, "optimizer_update": optimizer_updates,
                   "learner_version": learner_version, "physical_world_steps": actual,
                   "reserved_world_steps": bound, "max_world_steps": settings["max_world_steps"],
                   "source_separation_checked": True, "sampling": pending["sampling"],
                   "seconds_before_recording": time.perf_counter() - begin}
        (directory / "matchups.json").write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        collector.refresh_opponent_sampling(report, optimizer_update=optimizer_updates,
                                           learner_version=learner_version, **sampling_settings(settings))
        summary["seconds_including_recording"] = time.perf_counter() - begin
        # This returned cost includes all synchronous refresh work. The training metrics
        # persist it; the report above deliberately labels its narrower timing scope.
        return summary
    except Exception as exc:
        failure = {"status": "failed", "revision": revision, "optimizer_update": optimizer_updates,
                   "reserved_world_steps": bound, "actual_world_steps": None,
                   "actual_world_steps_known": False, "seconds_including_failure": time.perf_counter() - begin,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "sampling_published": collector.adaptive_sampling != state}
        (directory / "failure.json").write_text(json.dumps(failure, allow_nan=False), encoding="utf-8")
        raise
