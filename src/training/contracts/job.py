"""Common job semantics; native backends retain their numerical algorithms."""

import math
from pathlib import Path

from driversi.agents.backends.native import digest, path_at
from driversi.configuration import read_mapping
from driversi.training.contracts.protocols import BACKENDS, LIFECYCLE_BACKENDS, MODES

BUDGET_KEYS = {
    "max_wall_seconds",
    "max_optimizer_steps",
    "max_environment_steps",
    "max_scoring_agent_steps",
}


def validate_request(value):
    if set(value) != (
        {"schema_version", "id", "model_config", "mode", "initialization", "budget"}
        | ({"data", "pilot"} if value.get("schema_version") in (2, 3, 4) else set())
    ) or value["schema_version"] not in (1, 2, 3, 4):
        raise ValueError("Malformed unified training request")
    if any(
        not isinstance(value[k], str) or not value[k]
        for k in ("id", "model_config", "mode", "initialization")
    ):
        raise ValueError("Explicit training identity and model required")
    if value["mode"] not in MODES:
        raise ValueError("Unknown learning mode")
    if value["initialization"] != "model_weights":
        raise ValueError(
            "Exact optimizer/RNG/data-cursor resume is not supported by these adapters"
        )
    if value["schema_version"] in (2, 3, 4):
        p = value["pilot"]
        if (
            set(p) != {"seed", "batch_size", "evaluation_steps"}
            or any(type(p[k]) is not int for k in p)
            or not 1 <= p["batch_size"] <= (64 if value["schema_version"] == 4 else 4)
            or not 1 <= p["evaluation_steps"] <= (20000 if value["schema_version"] in (3, 4) else 100)
            or p["seed"] < 0
        ):
            raise ValueError("Bounded explicit single-GPU pilot controls required")
        if not isinstance(value["data"], str) or not value["data"]:
            raise ValueError("Explicit real-source manifest required")
    b = value["budget"]
    if not isinstance(b, dict) or set(b) != BUDGET_KEYS:
        raise ValueError("Explicit training resource limits required")
    if (
        type(b["max_wall_seconds"]) not in (int, float)
        or not math.isfinite(b["max_wall_seconds"])
        or not 0 < b["max_wall_seconds"] <= 86400
    ):
        raise ValueError("Bounded wall time required")
    for k in BUDGET_KEYS - {"max_wall_seconds"}:
        if type(b[k]) is not int or b[k] < (1 if k == "max_optimizer_steps" else 0):
            raise ValueError("Invalid discrete training budget: " + k)
    if (value["mode"] == "online_rl") != (b["max_environment_steps"] > 0):
        raise ValueError(
            "Online RL requires live environment steps; offline jobs cannot spend them"
        )
    return value


def resolve_request(path):
    path = Path(path).resolve(strict=True)
    request = validate_request(read_mapping(path))
    config = path_at(path.parent, request["model_config"])
    cfg = read_mapping(config)
    if request["schema_version"] == 3 and cfg["model"] not in ("chainflow_stage1", "augustus"):
        raise ValueError("Official navtest campaign requires a registered benchmark adapter")
    if request["schema_version"] == 4 and cfg["model"] != "chainflow_stage1":
        raise ValueError("Epoch distributed backend currently requires ChainFlow stage1")
    if request["schema_version"] == 4:
        from driversi_agents.chainflow.sampling import distributed_settings
        distributed_settings(cfg["settings"])
    backend = (LIFECYCLE_BACKENDS if request["schema_version"] in (2, 3, 4) else BACKENDS).get(
        (cfg["model"], request["mode"])
    )
    return {
        "request": request,
        "request_path": str(path),
        "request_sha256": digest(path),
        "model_config": str(config),
        "model_config_sha256": digest(config),
        "model": cfg["model"],
        "family": cfg["family"],
        "backend": backend,
        "native_learner_registered": (cfg["model"], request["mode"]) in BACKENDS,
        "learning_backend_registered": backend is not None,
        "reason": None if backend else "native_learning_backend_not_connected",
        "initialization": "model_weights_not_exact_training_resume",
        "evaluation_status": "independent_evaluation_required",
        "promoted": False,
    }
