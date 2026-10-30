"""Selected driving models and relocated rendering entrypoints remain executable."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from driversi.agents.backends.native import ADAPTERS
from driversi.agents.backends.sources import require_source
from driversi.agents.interfaces.config import normalize_agent
from driversi.training.orchestration.lifecycle import WORKERS
ROOT = Path(__file__).resolve().parents[3]

def test_selected_portfolio_and_retired_defaults():
    portfolio = json.loads((ROOT / "src/config/model-selection.json").read_text())
    assert set(portfolio["active_models"]) == {"driverl", "augustus", "fpd_drive", "chainflow_stage1"}
    assert set(WORKERS) == {"driverl", "augustus", "chainflow_stage1"}
    assert set(ADAPTERS) == {"driverl", "augustus", "fpd_drive"}
    assert {p.name for p in (ROOT / "agents").iterdir() if p.is_dir() and not p.name.startswith("__")} == {"driverl", "augustus", "fpd_drive", "chainflow"}
    planned = next(x for x in portfolio["models"] if x["model"] == "drive_jepa")
    assert not planned["training_connected"] and not planned["weights_present"]
    assert not (ROOT / "weights/wa_jepa").exists()
    with pytest.raises(ValueError, match="Explicit agent"):
        normalize_agent()

@pytest.mark.parametrize("name", ["mtgs", "streetcrafter", "worlddreamer"])
def test_render_sources_preserve_recorded_content(name):
    from driversi.vendor import path as dependency_path
    assert require_source(dependency_path(name))["clean"]
    assert not (ROOT / "agents" / name).exists()

@pytest.mark.parametrize("name", ["streetcrafter", "worlddreamer"])
def test_render_worker_is_launchable_without_agents_alias(tmp_path, name):
    worker = ROOT / "src/rendering/video" / name / "runtime.py"
    result = subprocess.run([sys.executable, "-I", str(worker), "--help"], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--prepared" in result.stdout

def test_harness_keeps_active_training_and_render_services():
    from driversi.rsi.workflow.providers import builtin_providers
    names = {p["name"] for p in builtin_providers().describe()}
    assert {"native_model_unified_training", "conditioned_video_render", "mtgs_selected_branch_render"} <= names
    assert "native_model_training" not in names  # retired WA-only provider

@pytest.mark.parametrize("name,worker", [("streetcrafter", "diagnostic.py"), ("worlddreamer", "prepare.py")])
def test_isolated_native_renderer_imports_do_not_load_driving_models(name, worker, tmp_path):
    python = ROOT / ".venvs" / name / "bin/python"
    if not python.is_file():
        pytest.skip("Native renderer environment unavailable")
    path = ROOT / "src/rendering/video" / name / worker
    code = "import runpy,sys; runpy.run_path(" + repr(str(path)) + ", run_name='import_probe'); assert 'driversi_agents' not in sys.modules"
    result = subprocess.run([str(python), "-I", "-B", "-c", code], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_active_model_preflight_dependency_lock_is_present():
    from driversi.agents.backends.native import catalog
    from driversi.configuration import read_mapping
    lock = read_mapping(ROOT / "src/config/agents/native-sources.lock.json")
    assert "WA-JEPA" not in lock
    for cfg in catalog(ROOT):
        for value in cfg["paths"].values():
            path = (ROOT / value).resolve()
            if path.is_relative_to(ROOT / "src/vendor"):
                from driversi.agents.backends.sources import source_root
                name = source_root(path).name
                key = "DrivoR" if name == "drivor" else name
                assert len(lock[key]["revision"]) == 40
