"""Pinned, isolated native model jobs. Plans are not performance certificates."""

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from driversi.agents.backends.sources import source_state, source_root
from driversi.configuration import read_mapping

ADAPTERS = {
    "driverl": "driversi_agents.driverl.adapter",
    "fpd_drive": "driversi_agents.fpd_drive.adapter",
    "augustus": "driversi_agents.augustus.adapter",
}


class NativeCancelled(Exception):
    """Cancellation must not inherit OSError: /proc readers catch transient I/O errors."""


def project_root():
    return Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def path_at(root, value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Explicit nonempty path required")
    return Path(os.path.abspath(root / value))


def runtime_fingerprint(python, env, cwd):
    # Metadata only: no Torch initialization, model loading or device allocation.
    code = r"""
import hashlib, importlib.metadata as m, importlib.util, json, pathlib, sys
packages = {}
for d in m.distributions():
    name = d.metadata.get('Name', '').lower()
    if name:
        packages[name] = {'version': d.version,
            'metadata_sha256': hashlib.sha256((d.read_text('METADATA') or '').encode()).hexdigest(),
            'record_sha256': hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest(),
            'direct_url': d.read_text('direct_url.json')}
origins = {}
for name in ['torch', 'numpy', 'nuplan', 'navsim', 'timm', 'transformers', 'omegaconf']:
    spec = importlib.util.find_spec(name)
    origin = spec.origin if spec else None
    path = pathlib.Path(origin) if origin and origin not in ('built-in','frozen') else None
    origins[name] = {'origin':origin, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None}
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'version':sys.version,'packages':packages,'origins':origins},sort_keys=True))
"""
    raw = subprocess.check_output(
        [str(python), "-B", "-c", code], env=env, cwd=cwd, text=True, timeout=30
    )
    result = json.loads(raw)
    result["interpreter_sha256"] = digest(python)
    return result


def asset_files(path):
    path = Path(path)
    if not path.is_dir():
        return [str(path)]
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    if any(not p.resolve().is_relative_to(path.resolve()) for p in files):
        raise ValueError("Checkpoint directory contains external file references")
    return [str(p) for p in files] or [str(path)]


def catalog(root=None):
    root = Path(root or project_root())
    return [read_mapping(p) for p in sorted((root / "src/config/agents").glob("*/native.yaml"))]


def prepare(config_path, operation, output, *, root=None):
    root = Path(root or project_root()).resolve()
    path = Path(config_path).resolve()
    cfg = read_mapping(path)
    required = {
        "schema_version",
        "model",
        "family",
        "repository",
        "revision",
        "python",
        "weights",
        "paths",
        "settings",
        "capabilities",
        "input_contract",
        "output_contract",
    }
    if set(cfg) != required or cfg["schema_version"] != 1 or cfg["model"] not in ADAPTERS:
        raise ValueError("Unsupported native model configuration")
    if operation not in cfg["capabilities"] or not cfg["capabilities"][operation]:
        raise ValueError("Native operation not released or not adapted: " + operation)
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["revision"]):
        raise ValueError("Full upstream commit required")
    output = Path(output).resolve()
    if not any(output.is_relative_to(root / n) for n in ("outputs", "exp")):
        raise ValueError("Native output must be under project outputs")
    if operation == "train" and not output.is_relative_to(root / "exp"):
        raise ValueError("New native training artifacts belong under exp")
    repo = path_at(root, cfg["repository"])
    if repo.resolve() != root / "agents" / cfg["model"] / "native":
        raise ValueError("Model implementation must be owned under agents/<model>/native")
    python = path_at(root, cfg["python"])
    weights = {k: str(path_at(root, v)) for k, v in cfg["weights"].items()}
    for p in weights.values():
        if not any(Path(p).resolve().is_relative_to(root / n) for n in ("weights", "exp")):
            raise ValueError("All active checkpoint/backbone files must be under weights")
    paths = {k: str(path_at(root, v)) for k, v in cfg["paths"].items()}
    adapter = importlib.import_module(ADAPTERS[cfg["model"]])
    launch = adapter.build(cfg, root, repo, str(python), weights, paths, output, operation)
    if set(launch) - {
        "argv",
        "env",
        "files",
        "directories",
        "protocol",
        "generated_files",
        "checkpoint_output",
        "input_dataset",
    } or not {"argv", "env", "files", "directories", "protocol"} <= set(launch):
        raise ValueError("Malformed native launch contract")
    generated = launch.get("generated_files", {})
    for name, content in generated.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".yaml")
            or not isinstance(content, str)
        ):
            raise ValueError("Generated native config must be a local YAML filename")
    checkpoint_output = launch.get("checkpoint_output")
    if checkpoint_output and (
        not Path(checkpoint_output).resolve().is_relative_to(root / "exp")
        or Path(checkpoint_output).exists()
    ):
        raise ValueError("New native checkpoint output must be unused and under exp")
    missing = []
    input_dataset = launch.get("input_dataset")
    if input_dataset and not input_dataset["ready"]:
        missing.append("native_dataset_missing_images:" + str(input_dataset["missing_count"]))
    if not python.is_file() or not os.access(python, os.X_OK):
        missing.append("python:" + str(python))
    if type(cfg["settings"].get("gpu")) is not int or cfg["settings"]["gpu"] < 0:
        raise ValueError("One explicit nonnegative GPU index required")
    if "workers" in cfg["settings"] and (
        type(cfg["settings"]["workers"]) is not int or not 1 <= cfg["settings"]["workers"] <= 64
    ):
        raise ValueError("Native workers must be in [1,64]")
    source = source_state(repo, cfg["revision"])
    if source["actual"] != cfg["revision"] or not source["clean"]:
        missing.append("owned_source_revision_or_content")
    lock_path = root / "src/config/agents/native-sources.lock.json"
    lock = read_mapping(lock_path)
    dependencies = {}
    for item in paths.values():
        candidate = Path(item).resolve()
        if candidate.is_relative_to(root / "clones"):
            raise ValueError("Runtime dependencies cannot come from clones")
        if candidate.is_relative_to(root / "src/vendor"):
            dependency = source_root(candidate)
            name = "DrivoR" if dependency.name == "drivor" else dependency.name
            expected = lock.get(name, {}).get("revision")
            state = source_state(candidate, expected)
            dependencies[str(candidate)] = state
            if not expected or state["actual"] != expected or not state["clean"]:
                missing.append("dependency_source_changed:" + str(candidate))
    implementation = [
        Path(__file__),
        Path(__file__).with_name("native_process.py"),
        Path(__file__).with_name("sources.py"),
        repo / "source-manifest.json",
        Path(adapter.__file__),
        lock_path,
    ]
    files = {
        str(p): digest(p)
        for p in [path, *implementation, *map(Path, launch["files"])]
        if p.is_file()
    }
    for f in launch["files"]:
        if not Path(f).is_file():
            missing.append("file:" + str(f))
        elif Path(f).stat().st_size < 256 and Path(f).read_bytes().startswith(
            b"version https://git-lfs"
        ):
            missing.append("lfs_pointer:" + str(f))
    for d in launch["directories"]:
        if not Path(d).is_dir():
            missing.append("directory:" + str(d))
    # No inherited model/scoring overrides; these can silently change reference policies.
    env = {k: os.environ[k] for k in ("HOME", "USER", "LANG", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(
        PATH=str(python.parent) + ":/usr/local/bin:/usr/bin:/bin",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(root / "weights/.cache/huggingface"),
        TORCH_HOME=str(root / "weights/.cache/torch"),
        XDG_CACHE_HOME=str(root / "weights/.cache"),
        TRITON_CACHE_DIR=str(root / "weights/.cache/triton"),
    )
    env.update(launch["env"])
    runtime = None
    if python.is_file() and os.access(python, os.X_OK):
        try:
            runtime = runtime_fingerprint(python, env, repo)
            for name, evidence in runtime.get("origins", {}).items():
                origin = evidence.get("origin")
                if origin and Path(origin).resolve().is_relative_to(root / "clones"):
                    missing.append("runtime_import_from_clones:" + name)

        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            missing.append("runtime_probe:" + type(exc).__name__)
    plan = {
        "schema_version": 1,
        "model": cfg["model"],
        "operation": operation,
        "config_path": str(path),
        "project_root": str(root),
        "output": str(output),
        "cwd": str(repo),
        "argv": launch["argv"],
        "env": env,
        "source": source,
        "runtime": runtime,
        "dependencies": dependencies,
        "files": files,
        "missing": missing,
        "ready_to_attempt": not missing,
        "protocol": launch["protocol"],
        "generated_files": generated,
        "checkpoint_output": checkpoint_output,
        "input_dataset": input_dataset,
        "input_contract": cfg["input_contract"],
        "output_contract": cfg["output_contract"],
        "performance_parity": "not_measured",
        "driversi_online_control": False,
        "trainable_data_admitted": False,
    }
    plan["plan_sha256"] = signature(plan)
    return plan


def verify(plan):
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if signature(unsigned) != plan["plan_sha256"]:
        raise ValueError("Native job plan was modified")
    actual = prepare(
        plan["config_path"], plan["operation"], plan["output"], root=plan["project_root"]
    )
    if actual != plan:
        raise ValueError("Native code/config/checkpoint/environment changed since preparation")
    if not plan["ready_to_attempt"]:
        raise ValueError("Missing native prerequisites: " + ", ".join(plan["missing"]))


def execute(plan, *, timeout_seconds, own_process_group=True, deadline_monotonic=None):
    import signal
    import time

    from driversi.agents.backends.native_process import OwnedTree
    from driversi.runtime.processes import subreaper

    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 86400
    ):
        raise ValueError("Explicit bounded wall time required")
    verify(plan)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    for name, content in plan.get("generated_files", {}).items():
        (output / name).write_text(content)
    if plan.get("checkpoint_output"):
        Path(plan["checkpoint_output"]).mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {
        "status": "running",
        "model": plan["model"],
        "plan_sha256": plan["plan_sha256"],
        "performance_parity": "not_measured",
        "promoted": False,
        "trainable": False,
        "evaluation_status": "not_validated",
        "scope": "native_process_execution_only",
    }
    report = output / "report.json"
    report.write_text(json.dumps(result, indent=2))
    process = None
    prior_subreaper = subreaper(True)
    tree = OwnedTree()
    old_handlers = {}

    def cancel(signum, frame):
        raise NativeCancelled("Native launcher cancelled by signal " + str(signum))

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, cancel)
        with (output / "native.log").open("wb") as log:
            remaining = timeout_seconds if deadline_monotonic is None else min(
                timeout_seconds, deadline_monotonic - time.monotonic())
            if remaining <= 0:
                result.update(status="timeout", reason="deadline_exhausted_before_launch", returncode=None)
            else:
                process = subprocess.Popen(
                    plan["argv"],
                    cwd=plan["cwd"],
                    env=plan["env"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=own_process_group,
                )
                tree.attach(process)
                try:
                    code = tree.wait(timeout=remaining)
                    result.update(status="completed" if code == 0 else "failed", returncode=code)
                except subprocess.TimeoutExpired:
                    result.update(status="timeout", returncode=None)
    except (KeyboardInterrupt, NativeCancelled) as exc:
        result.update(status="cancelled", reason=str(exc), returncode=None)
    except OSError as exc:
        result.update(status="failed", reason=str(exc), returncode=None)
    finally:
        try:
            tree.stop()
        except RuntimeError as exc:
            result.update(status="outcome_unknown", reason=str(exc))
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            if prior_subreaper is not None:
                subreaper(prior_subreaper)
            if plan.get("checkpoint_output"):
                checkpoint_root = Path(plan["checkpoint_output"])
                result["checkpoint_output"] = str(checkpoint_root)
                result["checkpoints"] = {
                    str(p): digest(p)
                    for p in checkpoint_root.rglob("*")
                    if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt", ".safetensors")
                }
                if result["status"] == "completed" and not result["checkpoints"]:
                    result.update(status="failed", reason="native_training_produced_no_checkpoint")
            result["wall_seconds"] = time.monotonic() - start
            gpu_count = len(
                [v for v in plan["env"].get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
            )
            result["reserved_gpu_seconds"] = gpu_count * result["wall_seconds"]
            if (output / "native.log").exists():
                result["native_log_sha256"] = digest(output / "native.log")
            if plan["operation"] == "preflight" and result["status"] == "completed":
                if (
                    "PREFLIGHT_OK"
                    not in (output / "native.log").read_text(errors="replace").splitlines()
                ):
                    result.update(status="failed", reason="missing_official_preflight_marker")
            report.write_text(json.dumps(result, indent=2))
    return result
