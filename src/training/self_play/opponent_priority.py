"""Frozen development-failure prioritization, not an online adversarial league."""

import math
from pathlib import Path

import numpy as np

from driversi.training.sampling.environment import validate_splits


def failure_probabilities(counts, *, uniform_mix=0.2, power=2.0, min_actor_episodes=20):
    if (
        isinstance(uniform_mix, bool)
        or not isinstance(uniform_mix, (int, float))
        or not math.isfinite(uniform_mix)
        or not 0 < uniform_mix <= 1
        or isinstance(power, bool)
        or not isinstance(power, (int, float))
    ):
        raise ValueError("Finite positive uniform_mix <= 1 and power required")
    if not math.isfinite(power) or not 0 < power <= 8:
        raise ValueError("power must be finite in (0,8]")
    if type(min_actor_episodes) is not int or min_actor_episodes < 1 or not counts:
        raise ValueError("Positive minimum episode count and nonempty counts required")
    rates = []
    for n, unsafe in counts:
        if (
            type(n) is not int
            or type(unsafe) is not int
            or n < min_actor_episodes
            or not 0 <= unsafe <= n
        ):
            raise ValueError("Insufficient or invalid learner actor-episode evidence")
        rates.append((unsafe + 1) / (n + 2))
    weights = np.asarray(rates, dtype=np.float64) ** power
    return uniform_mix / len(weights) + (1 - uniform_mix) * weights / weights.sum()


def priority_from_report(report, pool_identity, *, uniform_mix, power, min_actor_episodes):
    """Revalidate local episode evidence; upstream lineage remains declared, not certified."""
    expected = {
        "schema_version": 1,
        "kind": "frozen-opponent-development-matchups",
        "source_separation_checked": True,
        "official_test": False,
    }
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("A source-separated development matchup report is required")
    if report.get("opponent_pool") != pool_identity:
        raise ValueError("Matchup report does not match the frozen opponent pool")
    if not isinstance(report.get("learner_version"), str) or not report["learner_version"]:
        raise ValueError("Missing report learner version")
    paths = report.get("evaluation_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Missing evaluation cohort")
    paths = [str(Path(p).resolve()) for p in paths]
    validate_splits(report.get("learner_training_paths"), paths)
    train = [p for row in pool_identity["policies"] for p in row["training_paths"]]
    validate_splits(list(dict.fromkeys(train)), paths)
    rows = report.get("matchups")
    if not isinstance(rows, list) or len(rows) != len(pool_identity["policies"]):
        raise ValueError("Every frozen opponent needs matchup evidence")
    by_id = {row["opponent_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate opponent evidence")
    evidence, counts, common = [], [], None
    for policy in pool_identity["policies"]:
        row = by_id.get(policy["id"], {})
        if (
            row.get("opponent_version") != policy["policy_version"]
            or row.get("opponent_checkpoint_sha256") != policy["checkpoint_sha256"]
        ):
            raise ValueError("Opponent evidence version differs")
        episodes = row.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != len(paths):
            raise ValueError("Incomplete matchup cohort")
        cohort, n, unsafe = [], 0, 0
        for ep in episodes:
            values = ep.get("learner", {})
            actors, failed = values.get("actors"), values.get("unsafe")
            collision, offroad = values.get("collision"), values.get("offroad")
            if (
                any(type(v) is not int for v in (actors, failed, collision, offroad))
                or actors < 1
                or not 0 <= collision <= actors
                or not 0 <= offroad <= actors
                or not max(collision, offroad) <= failed <= min(actors, collision + offroad)
            ):
                raise ValueError(
                    "Missing or inconsistent unsafe actor-episode count; rerun matchups"
                )
            if actors != report.get("learner_count"):
                raise ValueError("Learner role count differs")
            if ep.get("seed") != report.get("seed"):
                raise ValueError("Mixed evaluation seeds")
            cohort.append((str(Path(ep["scene"]).resolve()), ep["scenario_id"], ep["seed"]))
            n += actors
            unsafe += failed
        if len(set(cohort)) != len(cohort) or sorted(x[0] for x in cohort) != sorted(paths):
            raise ValueError("Duplicated or missing evaluation scenes")
        cohort = sorted(cohort)
        if common is not None and cohort != common:
            raise ValueError("Opponents must share the same scene/seed cohort")
        common = cohort
        counts.append((n, unsafe))
        evidence.append(
            {
                "id": policy["id"],
                "actor_episodes": n,
                "unsafe": unsafe,
                "smoothed_failure_rate": (unsafe + 1) / (n + 2),
            }
        )
    probs = failure_probabilities(
        counts, uniform_mix=uniform_mix, power=power, min_actor_episodes=min_actor_episodes
    )
    for row, prob in zip(evidence, probs):
        row["probability"] = float(prob)
    return {
        "method": "failure_priority_v1",
        "uniform_mix": uniform_mix,
        "power": power,
        "min_actor_episodes": min_actor_episodes,
        "evidence": evidence,
        "reference_learner_version": report["learner_version"],
        "evaluation_paths": paths,
        "cohort": [list(row) for row in common],
        "metric": "collision_or_offroad_actor_episode_union",
        "scope": "frozen_stage_distribution_not_online_PFSP_or_official_driving_score",
    }
