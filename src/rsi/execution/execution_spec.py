"""Resolve trusted tool configuration into an immutable, source-checked job description."""
import hashlib
import math
from pathlib import Path

from driversi.agents.interfaces.config import read_agent_config
from driversi.configuration import read_mapping
from driversi.data.scenes.scene_store import scene_header
from driversi.rsi.contracts import _identity
from driversi.training.sampling.environment import validate_development_paths, validate_splits

EVALUATOR = "driversi.teacher.v1"
METRICS = {k + "_per_actor" for k in
           ("collision", "offroad", "goal", "return_sum", "travel_m", "endpoint_progress_m")}
DEFAULTS = {"seed": 0, "device": "cpu", "gpu_id": 0, "wall_seconds": 180.0,
            "worlds": 2, "rollout_steps": 8, "epochs": 2, "minibatch_size": 128,
            "learning_rate": 0.0003, "max_rollout_mib": 256, "recording": "metrics",
            "initial_weights": None, "reference_weights": None, "reference_model": None}


def resolve_execution(raw, base: Path, plan):
    required = {"schema_version", "id", "plan_id", "arm", "tool", "suite", "agent_config",
                "target_transitions", "max_train_agent_steps"}
    if not isinstance(raw, dict) or required - set(raw) or set(raw) - required - set(DEFAULTS):
        raise ValueError("Invalid execution fields")
    job = {**DEFAULTS, **raw}
    if type(job["schema_version"]) is not int or job["schema_version"] != 1:
        raise ValueError("Unknown execution schema")
    _identity(job["id"], "job id")
    if job["plan_id"] != plan["id"] or job["arm"] not in ("baseline", "candidate"):
        raise ValueError("Plan/arm mismatch")
    expected_role = "promotion" if job["tool"] == "structured_gate" else "development"
    if plan["target"] != "model" or plan["protocol"]["role"] != expected_role:
        raise ValueError("Tool/protocol role mismatch")
    if plan["protocol"]["evaluator_version"] != EVALUATOR or plan["protocol"]["metric"] not in METRICS:
        raise ValueError("Unsupported independent evaluator or metric")
    if job["tool"] not in ("structured_ppo", "structured_sample", "structured_probe", "structured_gate"):
        raise ValueError("Unknown allowlisted tool")
    if job["device"] not in ("cpu", "cuda") or job["recording"] not in ("metrics", "full"):
        raise ValueError("Unsupported device/recording")
    readonly = job["tool"] in ("structured_probe", "structured_gate")
    if readonly and (job["target_transitions"] != 0 or job["max_train_agent_steps"] != 0):
        raise ValueError("Probe/gate must have zero training budget")
    for key in ("seed", "gpu_id", "worlds", "rollout_steps", "epochs", "minibatch_size",
                "target_transitions", "max_train_agent_steps", "max_rollout_mib"):
        if type(job[key]) is not int or job[key] < (0 if key in ("seed", "gpu_id") or readonly and key in ("target_transitions", "max_train_agent_steps") else 1):
            raise ValueError(f"Invalid {key}")
    for key in ("wall_seconds", "learning_rate"):
        if type(job[key]) not in (int, float) or not math.isfinite(job[key]) or job[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if job["target_transitions"] > job["max_train_agent_steps"]:
        raise ValueError("Target exceeds training simulation cap")
    def path(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Expected explicit input path")
        return (base / value).resolve(strict=True)
    suite_path = path(job.pop("suite"))
    suite = read_mapping(suite_path)
    if suite.get("official_test") is not False:
        raise ValueError("An explicitly non-official suite is required")
    def paths(split):
        return [str((suite_path.parent / row["path"]).resolve(strict=True)) for row in suite[split]]
    job["train_paths"], job["eval_paths"] = paths("train"), paths("evaluation")
    validate_splits(job["train_paths"], job["eval_paths"])
    for key, actual in (
        ("development_groups", validate_development_paths(job["train_paths"])),
        ("evaluation_groups", validate_development_paths(job["eval_paths"])),
    ):
        if set(plan["data"][key]) != actual:
            raise ValueError(f"Actual scene lineage differs from declared {key}")
    if not set(plan["scene_ids"]) <= {Path(x).stem for x in job["train_paths"] + job["eval_paths"]}:
        raise ValueError("Plan links scenes outside the execution suite")
    job["agent"] = read_agent_config(path(job.pop("agent_config")))
    evaluation_paths = job["train_paths"] if job["tool"] == "structured_probe" else job["eval_paths"]
    headers = [scene_header(p) for p in evaluation_paths]
    job["evaluation_max_agent_steps"] = sum(len(h["controlled_ids"]) * h["max_steps"] for h in headers)
    if job["tool"] == "structured_gate":
        job["evaluation_max_agent_steps"] *= 2
        if not job["initial_weights"] or not job["reference_weights"]:
            raise ValueError("Gate requires candidate and reference weights")
        _identity(job["reference_model"], "reference model")
    elif job["reference_weights"] is not None or job["reference_model"] is not None:
        raise ValueError("Reference weights are gate-only")
    job["sources"] = []
    all_sources = job["train_paths"] + job["eval_paths"]
    for key in ("initial_weights", "reference_weights"):
        if job[key] is not None:
            job[key] = str(path(job[key]))
            all_sources.append(job[key])
    for source in all_sources:
        stat = Path(source).stat()
        job["sources"].append({"path": source, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    identity = f"{plan['id']}/{job['id']}"
    job["candidate_version"] = (job["id"][:64] + ".model." +
                                hashlib.sha256(identity.encode()).hexdigest()[:16])
    job["initial_model"] = plan["arms"][job["arm"]]["initial_model"]
    job["evaluator_version"] = EVALUATOR
    job["task_set"] = plan["protocol"]["task_set"]
    return job
