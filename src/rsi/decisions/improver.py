"""Versioned, constrained development-only curriculum proposal."""

import math
import random


def classify(row):
    """Observed safety events are evidence, not a causal diagnosis or solvability proof."""
    if row["collision"] or row["offroad"]:
        return "observed_safety_failure"
    endings = row.get("endings", {})
    if endings.get("route_support_end", 0):
        return "data_support_end"
    if endings.get("time_budget", 0) or row["goal"] == row["actors"]:
        return "normal_end"
    return "insufficient_evidence"


def propose(evidence, config, recipes, seed):
    if config["name"] != "hardcase_curriculum_v1":
        raise ValueError("Unknown registered improver")
    if evidence.get("purpose") != "development_mining":
        raise ValueError("Improver accepts development mining only")
    rows = evidence["episodes"]
    if not rows or len({r["scene"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate mining evidence")
    ranked = sorted(
        rows,
        key=lambda row: (
            -(100 * row["collision"] + 50 * row["offroad"])
            / row["actors"],
            row["scene"],
        ),
    )
    count = min(config["max_scenes"], len(rows))
    hard_count = max(1, count - math.ceil(count * config["replay_fraction"]))
    failures = [row for row in ranked if classify(row) == "observed_safety_failure"]
    hard = failures[:hard_count]
    hard_paths = {row["scene"] for row in hard}
    rest = sorted([row for row in ranked if row["scene"] not in hard_paths], key=lambda row: row["scene"])
    hard_count = len(hard)
    replay = random.Random(seed).sample(rest, count - hard_count)
    selected = hard + replay
    return {
        "improver_version": "hardcase_safety_evidence_v2",
        "recipe": config["recipe"],
        "selected_paths": [row["scene"] for row in selected],
        "hard_cases": [
            {
                "scene": row["scene"],
                "collision": row["collision"],
                "offroad": row["offroad"],
                "classification": classify(row),
                "endings": row.get("endings", {}),
            }
            for row in hard
        ],
        "replay_paths": [row["scene"] for row in replay],
        "reason": "Rank observed development safety events only; support/time endings do not imply driving failure; retain replay mixture",
    }


def validate_proposal(value, train_paths, recipes):
    expected = {
        "improver_version",
        "recipe",
        "selected_paths",
        "hard_cases",
        "replay_paths",
        "reason",
    }
    if not isinstance(value, dict) or set(value) != expected or value["recipe"] not in recipes:
        raise ValueError("Proposal cannot change evaluator, budgets or tool configuration")
    paths = value["selected_paths"]
    if not paths or len(paths) != len(set(paths)) or not set(paths) <= set(train_paths):
        raise ValueError("Proposal selected paths outside training pool")
    return value
