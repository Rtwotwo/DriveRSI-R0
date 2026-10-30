"""Resolve a fixed research campaign before any automatic proposal is run."""

import math
from pathlib import Path

from driversi.agents.interfaces.config import read_agent_config
from driversi.configuration import read_mapping
from driversi.rsi.contracts import BUDGET_FIELDS, _cost, _identity, _keys
from driversi.rsi.decisions.promotion import validate_policy
from driversi.training.sampling.environment import source_log_group, validate_development_paths


def resolve_campaign(raw, base):
    _keys(
        raw,
        (
            "schema_version",
            "id",
            "suite",
            "agent_config",
            "initial_model",
            "initial_weights",
            "initial_training_groups",
            "rounds",
            "seed",
            "device",
            "gpu_id",
            "wall_seconds",
            "budget",
            "stage_budget",
            "gate",
            "improver",
            "recipes",
            "training",
        ),
        "campaign",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported campaign schema")
    _identity(raw["id"], "campaign id")
    if len(raw["id"]) > 64:
        raise ValueError("Campaign ID must leave room for round job IDs")
    _identity(raw["initial_model"], "initial model")
    for key in ("rounds", "seed", "gpu_id"):
        if type(raw[key]) is not int or raw[key] < (0 if key != "rounds" else 1):
            raise ValueError("Invalid campaign integer")
    if (
        raw["device"] not in ("cpu", "cuda")
        or type(raw["wall_seconds"]) not in (int, float)
        or not math.isfinite(raw["wall_seconds"])
        or raw["wall_seconds"] <= 0
    ):
        raise ValueError("Invalid campaign device/deadline")
    spec = dict(raw)

    def path(value):
        return (Path(base) / value).resolve(strict=True)

    suite_path = path(raw["suite"])
    suite = read_mapping(suite_path)
    if suite.get("official_test") is not False:
        raise ValueError("Official test use is forbidden in improvement campaigns")

    def paths(rows):
        value = [str((suite_path.parent / row["path"]).resolve(strict=True)) for row in rows]
        validate_development_paths(value)
        return value

    spec["train_paths"], spec["development_paths"] = (
        paths(suite["train"]),
        paths(suite["development"]),
    )
    spec["promotion_slots"] = [paths(rows) for rows in suite["promotion_slots"]]
    if len(spec["promotion_slots"]) != spec["rounds"]:
        raise ValueError("Each round requires one predeclared fresh promotion slot")
    partitions = [spec["train_paths"], spec["development_paths"], *spec["promotion_slots"]]
    group_sets = [validate_development_paths(p) for p in partitions]
    seen = set()
    for groups in group_sets:
        if seen & groups:
            raise ValueError("Campaign data roles/slots overlap in original logs")
        seen |= groups
    spec["train_groups"], spec["development_groups"] = map(sorted, group_sets[:2])
    spec["promotion_groups"] = [sorted(g) for g in group_sets[2:]]
    if not isinstance(raw["initial_training_groups"], list) or any(
        not isinstance(x, str) or not x for x in raw["initial_training_groups"]
    ):
        raise ValueError("Declare initial trained-model lineage")
    initial_groups = {source_log_group(x) for x in raw["initial_training_groups"]}
    if initial_groups & set().union(*group_sets[2:]):
        raise ValueError("Initial model declared training overlaps promotion logs")
    spec["initial_training_groups"] = sorted(initial_groups)
    spec["agent"] = read_agent_config(path(raw["agent_config"]))
    spec["initial_weights"] = str(path(raw["initial_weights"])) if raw["initial_weights"] else None
    if spec["initial_weights"] and not initial_groups:
        raise ValueError("Existing weights require declared nonempty training lineage")
    validate_policy(spec["gate"])
    _cost(spec["budget"], "campaign budget", positive=True)
    _keys(spec["stage_budget"], ("mining", "training", "gate"), "stage budget")
    for stage, budget in spec["stage_budget"].items():
        _cost(budget, stage, positive=True)
        if (
            budget["trials"] != 1
            or budget["cpu_seconds"] <= 0
            or budget["llm_tokens"]
            or budget["rendered_frames"]
        ):
            raise ValueError("One bounded supported trial per stage is required")
        if stage != "training" and budget["gpu_seconds"]:
            raise ValueError("Mining/gate are CPU tools")
    if any(
        sum(v[k] for v in spec["stage_budget"].values()) > spec["budget"][k] for k in BUDGET_FIELDS
    ):
        raise ValueError("Campaign cannot reserve one complete round")
    _keys(spec["improver"], ("name", "max_scenes", "replay_fraction", "recipe"), "improver")
    im = spec["improver"]
    if (
        im["name"] != "hardcase_curriculum_v1"
        or type(im["max_scenes"]) is not int
        or im["max_scenes"] < 1
    ):
        raise ValueError("Unsupported improver/config")
    if type(im["replay_fraction"]) not in (int, float) or not 0 <= im["replay_fraction"] < 1:
        raise ValueError("Invalid replay fraction")
    if not isinstance(spec["recipes"], dict) or im["recipe"] not in spec["recipes"]:
        raise ValueError("Unknown allowed recipe")
    for key, recipe in spec["recipes"].items():
        _identity(key, "recipe")
        _keys(
            recipe,
            (
                "target_transitions",
                "max_train_agent_steps",
                "learning_rate",
                "epochs",
                "minibatch_size",
            ),
            "recipe",
        )
        for name in ("target_transitions", "max_train_agent_steps", "epochs", "minibatch_size"):
            if type(recipe[name]) is not int or recipe[name] < 1:
                raise ValueError("Positive recipe integers required")
        if (
            recipe["target_transitions"] > recipe["max_train_agent_steps"]
            or type(recipe["learning_rate"]) not in (float, int)
            or not math.isfinite(recipe["learning_rate"])
            or recipe["learning_rate"] <= 0
        ):
            raise ValueError("Invalid recipe budget/rate")
    _keys(spec["training"], ("worlds", "rollout_steps", "max_rollout_mib", "recording"), "training")
    for key in ("worlds", "rollout_steps", "max_rollout_mib"):
        if type(spec["training"][key]) is not int or spec["training"][key] < 1:
            raise ValueError("Positive training limits required")
    if spec["training"]["recording"] not in ("metrics", "full"):
        raise ValueError("Unknown recording mode")
    spec["sources"] = []
    for value in [p for partition in partitions for p in partition] + (
        [spec["initial_weights"]] if spec["initial_weights"] else []
    ):
        stat = Path(value).stat()
        spec["sources"].append({"path": value, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return spec
