"""PPO run configuration and CLI resolution; no model/runtime imports at CLI setup."""
import argparse
from pathlib import Path

from driversi.agents.interfaces.catalog import resolve_agent
from driversi.agents.interfaces.config import normalize_agent, read_agent_config
from driversi.configuration import read_mapping

DEFAULT_RUN = {"budget": 100000, "worlds": 4, "rollout_steps": 32, "seed": 0, "device": "cpu"}
DEFAULT_SAMPLING = {"workers": 0, "scene_cache_size": 32, "max_rollout_mib": 1024,
                    "worker_timeout_seconds": 30.}
DEFAULT_OPTIMIZATION = {"learning_rate": 3e-4, "epochs": 4, "minibatch_size": 512}


def add_training_arguments(parser):
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--distributed-timeout-seconds", type=float)
    parser.add_argument("--config", type=Path, help="Training JSON/YAML; agent paths are relative to it")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--agent", help="Select a named config from src/config/agents")
    selection.add_argument("--agent-config", type=Path, help="Select an explicit agent JSON/YAML")
    parser.add_argument("--agent-dir", type=Path, help="Config directory for --agent (source checkout by default)")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for key in ("budget", "worlds", "rollout_steps", "seed"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--opponent-pool", type=Path, help="Frozen history manifest; no pool preserves shared-policy sampling")
    parser.add_argument("--learner-count", type=int, help="First N controlled actors learn in mixed mode")
    parser.add_argument("--recording", choices=("full", "metrics"))
    for key in ("workers", "scene_cache_size", "max_rollout_mib"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--worker-timeout-seconds", type=float)


def resolve_distribution(args):
    result = {"enabled": False, "timeout_seconds": 600.0}
    config = getattr(args, "config", None)
    if config is not None:
        raw = read_mapping(Path(config).resolve()).get("distribution", {})
        if not isinstance(raw, dict) or set(raw) - set(result):
            raise ValueError("unknown distribution configuration")
        result.update(raw)
    if getattr(args, "distributed", None) is not None:
        result["enabled"] = args.distributed
    if getattr(args, "distributed_timeout_seconds", None) is not None:
        result["timeout_seconds"] = args.distributed_timeout_seconds
    import math
    timeout = result["timeout_seconds"]
    if type(result["enabled"]) is not bool or type(timeout) not in (int, float):
        raise ValueError("invalid distribution configuration")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("distributed timeout must be finite and positive")
    return result


def resolve_training(args):
    import math

    run = dict(DEFAULT_RUN)
    optimization = dict(DEFAULT_OPTIMIZATION)
    sampling = dict(DEFAULT_SAMPLING)
    recording = "full"
    opponent_pool = None
    learner_count = 1
    adaptive = None
    agent = None
    config_path = getattr(args, "config", None)
    override = getattr(args, "agent_config", None)
    name = getattr(args, "agent", None)
    directory = getattr(args, "agent_dir", None)
    if name is not None:
        if override is not None:
            raise ValueError("--agent and --agent-config are mutually exclusive")
        override = resolve_agent(name, directory)
    elif directory is not None:
        raise ValueError("--agent-dir requires --agent")
    if config_path is not None:
        config_path = Path(config_path).resolve()
        raw = read_mapping(config_path)
        allowed = {"schema_version", "trainer", "agent", "optimization", "sampling", "recording", "distribution", "opponent_pool", "learner_count", "adaptive_opponents", *DEFAULT_RUN}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError("Unknown training config fields")
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1 or raw.get("trainer") != "ppo":
            raise ValueError("This entry point supports trainer=ppo, schema_version=1 only")
        if not isinstance(raw.get("agent"), str) or not raw["agent"]:
            raise ValueError("Training config requires an agent config path")
        if override is None:
            agent = read_agent_config(config_path.parent / raw["agent"])
        run.update({k: raw[k] for k in DEFAULT_RUN if k in raw})
        supplied = raw.get("optimization", {})
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_OPTIMIZATION):
            raise ValueError("Unknown optimization config fields")
        optimization.update(supplied)
        supplied_sampling = raw.get("sampling", {})
        if not isinstance(supplied_sampling, dict) or set(supplied_sampling) - set(DEFAULT_SAMPLING):
            raise ValueError("Unknown sampling config fields")
        sampling.update(supplied_sampling)
        recording = raw.get("recording", recording)
        if "opponent_pool" in raw:
            if not isinstance(raw["opponent_pool"], str) or not raw["opponent_pool"]:
                raise ValueError("opponent_pool must be a manifest path")
            opponent_pool = (config_path.parent / raw["opponent_pool"]).resolve()
        learner_count = raw.get("learner_count", learner_count)
        from driversi.training.self_play.adaptive_opponents import resolve_adaptive
        adaptive = resolve_adaptive(raw.get("adaptive_opponents"))
    if getattr(args, "opponent_pool", None) is not None:
        opponent_pool = Path(args.opponent_pool).resolve()
    if getattr(args, "learner_count", None) is not None:
        learner_count = args.learner_count
    if type(learner_count) is not int or not 1 <= learner_count <= 31:
        raise ValueError("learner_count must be an integer in [1,31]")
    if opponent_pool is None and learner_count != 1:
        raise ValueError("learner_count requires an opponent pool")
    recording = getattr(args, "recording", None) or recording
    if recording not in ("full", "metrics"):
        raise ValueError("recording must be full or metrics")
    if override is not None:
        agent = read_agent_config(Path(override))
    agent = normalize_agent(agent)
    for key in DEFAULT_RUN:
        value = getattr(args, key, None)
        if value is not None:
            run[key] = value
    for key in DEFAULT_SAMPLING:
        value = getattr(args, key, None)
        if value is not None:
            sampling[key] = value
    for key in ("workers", "scene_cache_size", "max_rollout_mib"):
        if type(sampling[key]) is not int or sampling[key] < (0 if key == "workers" else 1):
            raise ValueError(f"Invalid sampling {key}")
    timeout = sampling["worker_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker_timeout_seconds must be finite and positive")
    for key in ("budget", "worlds", "rollout_steps", "seed"):
        if type(run[key]) is not int or run[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid training {key}")
    if sampling["workers"] > run["worlds"]:
        raise ValueError("workers cannot exceed worlds")
    if run["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    for key in ("epochs", "minibatch_size"):
        if type(optimization[key]) is not int or optimization[key] < 1:
            raise ValueError(f"Invalid optimization {key}")
    lr = optimization["learning_rate"]
    if type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    distribution = resolve_distribution(args)
    if adaptive is not None and (opponent_pool is None or distribution["enabled"]):
        raise ValueError("Adaptive opponents require a pool and single-process PPO; DDP refresh is not supported")
    return argparse.Namespace(**run, adaptive_opponents=adaptive, opponent_pool=opponent_pool, learner_count=learner_count, distribution=distribution, agent=agent, optimization=optimization, sampling=sampling, recording=recording,
                              suite=Path(args.suite), output_dir=Path(args.output_dir),
                              resume=Path(args.resume) if getattr(args, "resume", None) else None)
