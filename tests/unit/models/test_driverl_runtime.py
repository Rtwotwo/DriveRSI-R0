"""Preparation is distinct from model execution and DriveRSI closed-loop qualification."""
import copy
import json
import os
import sys

import pytest
import yaml

from tests.support.diagnostic_config import read_agent_config
from driversi_agents.driverl.__main__ import main
from driversi_agents.driverl.runtime import inspect_runtime, load_config, run_preflight


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "upstream"
    for folder in ("release/configs", "release/checkpoints", "scripts", "src/driverl", "nuplan-devkit/nuplan"):
        (root / folder).mkdir(parents=True)
    (root / "release/configs/teacher.yaml").write_text("{}")
    (root / "release/checkpoints/model.pt").write_bytes(b"test fixture, not a real checkpoint")
    (root / "release/driverl_checkpoints.yaml").write_text(yaml.safe_dump({"artifacts": [{
        "id": "fixture", "config_file": "release/configs/teacher.yaml",
        "checkpoint_file": "release/checkpoints/model.pt"}]}))
    script = root / "scripts/driverl_release_preflight.py"
    script.write_text('import os,json; print(json.dumps(dict(os.environ))); print("PREFLIGHT_OK")')
    cfg = dict(schema_version=1, runtime="driverl_nuplan", release_id="fixture",
               python=sys.executable, driverl_root="upstream", nuplan_root="upstream/nuplan-devkit",
               data_root="dataset", maps_root="maps", split="mini", preflight_timeout_seconds=10)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, script


def test_inventory_is_read_only_not_checkpoint_or_dataset_validation(runtime, tmp_path, monkeypatch):
    path, cfg, script = runtime
    script.write_text('raise RuntimeError("must not execute during inspection")')
    monkeypatch.chdir("/")
    result = inspect_runtime(path)
    assert result["can_attempt_official_preflight"]
    assert result["dataset_inventory"]["database_count"] == 0
    assert not result["dataset_inventory"]["contents_validated"]
    assert not result["dataset_inventory"]["required_for_preflight"]
    assert result["official_preflight"]["status"] == "not_run"
    assert result["driversi_closed_loop_adapter"] == "not_implemented"
    assert result["trainable"] is False
    assert result["config"]["driverl_root"] == str(tmp_path / "upstream")


def test_preflight_is_explicit_cpu_child_and_keeps_parent_environment(runtime, monkeypatch):
    path, cfg, script = runtime
    monkeypatch.setenv("PYTHONPATH", "/wrong-model-environment")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    before = dict(os.environ)
    report = inspect_runtime(path)
    original = copy.deepcopy(report)
    result = run_preflight(report)
    assert report == original
    assert dict(os.environ) == before
    assert result["official_preflight"]["status"] == "passed"
    env = json.loads(result["official_preflight"]["stdout"].splitlines()[0])
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep) == [report["config"]["nuplan_root"],
                                                   report["config"]["driverl_root"] + "/src"]
    assert not result["trainable"]


@pytest.mark.parametrize("text,status", [('print("PREFLIGHT_OK"); raise SystemExit(3)', "failed"),
                                        ('print("nothing validated")', "failed"),
                                        ('import time; time.sleep(5)', "timeout")])
def test_failed_or_incomplete_preflight_never_passes(runtime, text, status):
    path, cfg, script = runtime
    cfg["preflight_timeout_seconds"] = .2 if status == "timeout" else 10
    path.write_text(yaml.safe_dump(cfg))
    script.write_text(text)
    assert run_preflight(inspect_runtime(path))["official_preflight"]["status"] == status


def test_missing_interpreter_never_falls_back_to_platform(runtime, capsys):
    path, cfg, script = runtime
    cfg["python"] = "missing-venv/bin/python"
    path.write_text(yaml.safe_dump(cfg))
    result = inspect_runtime(path)
    assert not result["can_attempt_official_preflight"]
    assert result["preflight_argv"] is None
    assert result["missing"] == ["configured_python_executable"]
    with pytest.raises(ValueError, match="Missing"):
        run_preflight(result)
    assert main(["--config", str(path), "--run-preflight"]) == 2
    assert "Missing runtime prerequisites" in capsys.readouterr().out


def test_python_symlink_preserves_virtual_environment_location(runtime, tmp_path):
    path, cfg, _ = runtime
    exe = tmp_path / "venv/bin/python"
    exe.parent.mkdir(parents=True)
    exe.symlink_to(sys.executable)
    cfg["python"] = "venv/bin/python"
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)["python"] == str(exe)


@pytest.mark.parametrize("change", [{"extra": True}, {"schema_version": True},
    {"runtime": "ppo"}, {"split": "../test"}, {"python": ""}, {"release_id": 1},
    {"preflight_timeout_seconds": True}, {"preflight_timeout_seconds": 0},
    {"preflight_timeout_seconds": float("inf")}])
def test_invalid_config_rejected(runtime, change):
    path, cfg, _ = runtime
    cfg.update(change)
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_external_runtime_not_accepted_as_structured_ppo(runtime):
    with pytest.raises(ValueError, match="Agent config"):
        read_agent_config(runtime[0])


def test_release_file_escape_rejected(runtime):
    path, cfg, _ = runtime
    manifest = path.parent / "upstream/release/driverl_checkpoints.yaml"
    data = yaml.safe_load(manifest.read_text())
    data["artifacts"][0]["checkpoint_file"] = "../other.pt"
    manifest.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="outside"):
        inspect_runtime(path)
