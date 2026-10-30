"""Read-only inventory and isolated official preflight for trusted DriveRL releases.

This module never advances a DriveRSI world, rewrites upstream files, downloads data,
or installs dependencies. Native nuPlan evaluation and DriveRSI execution still need
separate validation. Merely finding a file does not validate its contents.
"""
import os
import signal
import subprocess
from pathlib import Path

from driversi.configuration import read_mapping

_FIELDS = {"schema_version", "runtime", "release_id", "python", "driverl_root",
           "nuplan_root", "data_root", "maps_root", "split", "preflight_timeout_seconds"}
_PATHS = ("python", "driverl_root", "nuplan_root", "data_root", "maps_root")


def load_config(path):
    path = Path(path).resolve()
    value = read_mapping(path)
    if set(value) - {"checkpoint"} != _FIELDS:
        raise ValueError("DriveRL runtime config fields must match schema version 1")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported DriveRL schema_version")
    if value["runtime"] != "driverl_nuplan" or value["split"] not in ("mini", "trainval", "test"):
        raise ValueError("Unsupported DriveRL runtime or nuPlan split")
    if not isinstance(value["release_id"], str) or not value["release_id"].strip():
        raise ValueError("release_id must be a nonempty string")
    timeout = value["preflight_timeout_seconds"]
    if type(timeout) not in (int, float) or not 0 < timeout <= 600:
        raise ValueError("preflight_timeout_seconds must be in (0, 600]")
    for key in (*_PATHS, *(("checkpoint",) if "checkpoint" in value else ())):
        raw = value[key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key} must be a nonempty path")
        # Do NOT resolve the executable symlink: /venv/bin/python must retain
        # its virtual-environment location rather than becoming /usr/bin/python.
        value[key] = os.path.abspath(path.parent / raw)
    return value


def _release_file(root, name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Release manifest requires a file path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release file is outside driverl_root")
    return path


def inspect_runtime(config_path):
    value = load_config(config_path)
    root = Path(value["driverl_root"])
    manifest = root / "release/driverl_checkpoints.yaml"
    checks = {}
    files = {"manifest": manifest, "preflight_script": root / "scripts/driverl_release_preflight.py"}
    if manifest.is_file():
        entries = read_mapping(manifest).get("artifacts", [])
        if not isinstance(entries, list):
            raise ValueError("Release manifest artifacts must be a list")
        matches = [a for a in entries if isinstance(a, dict) and a.get("id") == value["release_id"]]
        if len(matches) != 1:
            raise ValueError("release_id must identify exactly one manifest artifact")
        artifact = matches[0]
        files["runtime_config"] = _release_file(root, artifact.get("config_file"))
        files["checkpoint"] = (Path(value["checkpoint"]) if "checkpoint" in value
                               else _release_file(root, artifact.get("checkpoint_file")))
    else:
        checks.update(runtime_config=False, checkpoint=False)
    checks.update({key: p.is_file() for key, p in files.items()})
    executable = Path(value["python"])
    checks["configured_python_executable"] = executable.is_file() and os.access(executable, os.X_OK)
    checks["driverl_source"] = (root / "src/driverl").is_dir()
    checks["nuplan_source"] = (Path(value["nuplan_root"]) / "nuplan").is_dir()
    # Inventory only: a .db extension or map metadata is not schema/coverage validation.
    database_dir = Path(value["data_root"]) / "splits" / value["split"]
    db_count = sum(1 for p in database_dir.glob("*.db") if p.is_file())
    maps = Path(value["maps_root"])
    argv = None
    if all(checks.values()):
        argv = [value["python"], "-B", str(files["preflight_script"]),
                "--manifest", str(manifest), "--config", str(files["runtime_config"]),
                "--checkpoint", str(files["checkpoint"]), "--release-id", value["release_id"],
                "--driverl-root", str(root), "--nuplan-root", value["nuplan_root"],
                "--device", "cpu", "--check-simulation-import"]
    return {"schema_version": 1, "kind": "driverl-runtime-inventory", "config": value,
            "checks": checks, "missing": [k for k, ok in checks.items() if not ok],
            "can_attempt_official_preflight": all(checks.values()), "preflight_argv": argv,
            "dataset_inventory": {"database_dir": str(database_dir), "database_count": db_count,
                                  "maps_metadata_present": (maps / "nuplan-maps-v1.0.json").is_file(),
                                  "contents_validated": False, "required_for_preflight": False},
            "official_preflight": {"status": "not_run"},
            "driversi_closed_loop_adapter": "not_implemented", "trainable": False}


def run_preflight(report):
    """Execute only the official CPU preflight, in a separate process/environment.

    A successful import/checkpoint load is not an evaluation result. Report construction
    alone is read-only and does not execute trusted upstream code.
    """
    if not report["can_attempt_official_preflight"]:
        raise ValueError("Missing runtime prerequisites: " + ", ".join(report["missing"]))
    value = report["config"]
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(key, None)
    env.update(PYTHONPATH=os.pathsep.join([value["nuplan_root"],
                                         str(Path(value["driverl_root"]) / "src")]),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               NUPLAN_DATA_ROOT=value["data_root"], NUPLAN_MAPS_ROOT=value["maps_root"])
    with subprocess.Popen(report["preflight_argv"], cwd=value["driverl_root"], env=env,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          start_new_session=True) as process:
        status = "failed"
        try:
            stdout, stderr = process.communicate(timeout=value["preflight_timeout_seconds"])
            if process.returncode == 0 and "PREFLIGHT_OK" in stdout.splitlines():
                status = "passed"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            status = "timeout"
    return {**report, "official_preflight": {
        "status": status, "returncode": process.returncode,
        "stdout": stdout[-32000:], "stderr": stderr[-32000:],
        "scope": "official CPU checkpoint load and simulation import only"}}
