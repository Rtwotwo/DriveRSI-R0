"""Command catalog, owned implementation and retired entry contracts."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from driversi.architecture import script_violations

ROOT = Path(__file__).resolve().parents[3]


def test_all_commands_have_declared_ownership():
    assert script_violations(ROOT) == []
    assert not list((ROOT / "scripts").glob("*.py"))
    for name in ("run_chainflow_train_mining.py", "run_chainflow_navtest_campaign.py",
                 "run_confirmed_navtrain_teachers.py"):
        assert not list((ROOT / "scripts").rglob(name))


def test_an_unregistered_command_is_detected(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "catalog.json").write_text(json.dumps({"entries": []}))
    (scripts / "unexpected.py").write_text("pass")
    assert script_violations(tmp_path) == ["scripts/unexpected.py: missing command purpose"]


def test_wheel_does_not_require_checkout_scripts(tmp_path):
    assert script_violations(tmp_path) == []


@pytest.mark.parametrize("command", [
    "scripts/data/manage_scene_catalog.py", "scripts/data/run_scene_corpus.py",
    "scripts/rendering/publish_whitebox.py", "scripts/rendering/export_showcase_pages.py",
    "scripts/rendering/publish_interaction_showcase.py",
    "scripts/rendering/run_multiview_gs_window.py",
    "scripts/rendering/inspect_multiview_gs_window.py",
    "scripts/rsi/run_platform_spine.py",
])
def test_owned_cli_help_from_unrelated_working_directory(command, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / command), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def load_command(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("owned_command_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_publisher_dispatches_existing_canonical_scripts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = load_command("scripts/evaluation/publish_driverl_progress.py")
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"pid": 123}))
    commands = []
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    monkeypatch.setattr(sys, "argv", ["publisher", "--launch", str(launch)])
    def fake_run(command, **kwargs):
        commands.append(command)
        assert Path(command[1]).is_file()
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.main()
    assert len(commands) == 3
    assert all(Path(c[1]).parent != ROOT / "scripts" for c in commands)


@pytest.mark.parametrize("final", [False, True])
def test_multicity_dispatches_canonical_verifiers_only_in_temporary_tree(tmp_path, monkeypatch, final):
    from types import SimpleNamespace
    module = load_command("scripts/simulation/run_driverl_multicity.py")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/simulation/run.py"))
    monkeypatch.setattr(sys, "argv", ["multicity"] + (["--final"] if final else []))
    quality = tmp_path / "outputs/quality/driverl-multicity-v1"
    quality.mkdir(parents=True)
    for i in range(1, 9):
        (tmp_path / "outputs/demonstrations/driverl-multicity-v1" / f"log-{i:02}").mkdir(parents=True)
    commands = []
    def fake_run(command, **kwargs):
        relative = Path(command[1]).relative_to(tmp_path)
        assert relative.parts[:2] == ("scripts", "simulation")
        assert (ROOT / relative).is_file()
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.main()
    assert len(commands) == (8 if final else 14)
