"""Active models must run from owned sources, without a reference clone checkout."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from driversi.agents.backends import native
from driversi.agents.backends.sources import require_source, source_state

ROOT = Path(__file__).resolve().parents[3]


def test_owned_source_detects_changed_missing_and_added_runtime_code(tmp_path):
    code = tmp_path / "model.py"
    code.write_text("original network")
    (tmp_path / "source-manifest.json").write_text(json.dumps({
        "schema_version": 1, "revision": "a" * 40,
        "files": {"model.py": hashlib.sha256(code.read_bytes()).hexdigest()},
    }))
    assert require_source(tmp_path, "a" * 40)["clean"]
    code.write_text("different network")
    with pytest.raises(ValueError, match="changed"):
        require_source(tmp_path)
    code.write_text("original network")
    (tmp_path / "injected.py").write_text("extra model")
    assert not source_state(tmp_path)["clean"]
    (tmp_path / "injected.py").unlink()
    code.unlink()
    assert not source_state(tmp_path)["clean"]


def test_active_model_catalog_owns_implementation_and_has_no_clone_launch_paths():
    import importlib
    for cfg in native.catalog():
        repo = ROOT / cfg["repository"]
        assert repo == ROOT / "agents" / cfg["model"] / "native"
        assert require_source(repo, cfg["revision"])["clean"]
        adapter = importlib.import_module(native.ADAPTERS[cfg["model"]])
        plan = adapter.build(cfg, ROOT, repo, "/native/python",
            {k: str(ROOT / v) for k, v in cfg["weights"].items()},
            {k: str(ROOT / v) for k, v in cfg["paths"].items()},
            ROOT / "outputs/ownership-test", "preflight" if cfg["model"] == "driverl" else "evaluate")
        assert "/clones/" not in json.dumps(plan)
        assert not any(p.is_symlink() for p in repo.rglob("*"))


def test_owned_native_sources_and_shared_sdk_are_separately_versioned():
    from driversi.vendor import catalog, path as dependency_path
    for path in [*(dependency_path(name) for name in catalog()),
                 ROOT / "agents/driverl/learning/muon"]:
        assert require_source(path)["clean"]


def test_old_clone_runtime_configuration_is_rejected_before_import(tmp_path):
    cfg = yaml.safe_load((ROOT / "src/config/agents/driverl/native.yaml").read_text())
    cfg["repository"] = "clones/DriveZero"
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="owned under agents"):
        native.prepare(path, "preflight", ROOT / "outputs/ownership-test")


def test_preloaded_reference_module_cannot_be_reused(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from driversi.agents.backends.sources import activate_packages

    monkeypatch.setitem(sys.modules, "owned_fixture", SimpleNamespace(
        __file__=str(tmp_path / "clones/fixture/__init__.py")))
    with pytest.raises(ValueError, match="Conflicting"):
        activate_packages({"owned_fixture": tmp_path / "agents/fixture/native/fixture"})
