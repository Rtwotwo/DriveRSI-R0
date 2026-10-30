"""Configuration relocation preserves targets and uses the real provider registry."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from driversi.config import catalog, load, path, root
from driversi.rsi.services.catalog import describe
from driversi.vendor import catalog as dependencies, path as dependency_path

ROOT = Path(__file__).resolve().parents[3]


def test_presets_are_locatable_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-m", "driversi", "config", "path", "rsi/tasks/preflight/driverl.json"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == path("rsi/tasks/preflight/driverl.json")
    rows = catalog("rsi")
    assert rows and all(Path(x["path"]).is_relative_to(root()) for x in rows)


@pytest.mark.parametrize("name", ["../README.md", "/etc/passwd", "local/key.json", ".secret/key.json"])
def test_preset_id_cannot_load_private_or_external_paths(name):
    with pytest.raises(ValueError):
        path(name)


def test_relative_task_and_weight_targets_survive_move():
    task = path("rsi/tasks/training/augustus.json")
    request = load("rsi/tasks/training/augustus.json")["steps"][0]["config"]["request"]
    assert (task.parent / request).resolve() == path("training/augustus.json")
    teacher = path("agents/driverl/teacher.yaml")
    cfg = load("agents/driverl/teacher.yaml")
    assert (teacher.parent / cfg["checkpoint"]).resolve() == ROOT / "weights/driverl/checkpoint_2400.pt"
    render = load("rendering/mtgs-branch-boston-v1.json")
    assert Path(render["population_config"]) == path("populations/navsim-mtgs-boston-v1.json")


def test_service_catalog_resolves_actual_handlers_and_is_lazy(tmp_path):
    rows = describe()
    from driversi.rsi.workflow.providers import builtin_providers
    actual = {row["name"] for row in builtin_providers().describe()}
    assert {row["name"] for row in rows} == actual
    assert len(rows) == len(actual)
    assert all(row["dispatcher"] == "driversi.rsi.workflow.engine" for row in rows)
    command = "from driversi.rsi.services.catalog import describe; import sys; assert describe(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", command], cwd=tmp_path, check=True)
    with pytest.raises(ValueError):
        describe("unknown")


def test_dependencies_are_classified_without_old_root_aliases():
    assert not (ROOT / "third_party").exists()
    assert not (ROOT / "configs").exists()
    for name, value in dependencies().items():
        folder = dependency_path(name)
        assert folder.is_relative_to(ROOT / "src/vendor" / value["domain"])
        assert json.loads((folder / "source-manifest.json").read_text())["revision"] == value["revision"]
    with pytest.raises(ValueError):
        dependency_path("missing")


@pytest.mark.parametrize("preset", ["backends/pufferdrive.toml", "agents/augustus/scoring-requirements.txt", "rsi/examples/drivetto-cases.json"])
def test_config_show_handles_advertised_resource_types(preset, tmp_path):
    assert preset in {item["id"] for item in catalog()}
    expected = load(preset)
    result = subprocess.run([sys.executable, "-I", "-m", "driversi", "config", "show", preset],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == expected
    assert expected


def test_platform_adapters_cannot_refer_to_removed_vendor_directory():
    # Native repositories may have their own nested third_party directories.
    for adapter in (ROOT / "agents").rglob("*.py"):
        if "native" in adapter.relative_to(ROOT / "agents").parts:
            continue
        content = adapter.read_text()
        assert "third_party/navsim-v1.1" not in content, str(adapter)
        if "augustus" not in adapter.relative_to(ROOT / "agents").parts:
            assert "third_party/nuplan-devkit" not in content, str(adapter)


def test_all_catalogued_resources_can_be_read():
    for item in catalog():
        assert load(item["id"]) is not None, item["id"]
