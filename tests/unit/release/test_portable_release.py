import json
import subprocess
import sys
from pathlib import Path

import pytest

from driversi.configuration import project_paths, read_mapping
from driversi.visualization.catalog import scene_index
from driversi.config import path as preset_path

ROOT = Path(__file__).resolve().parents[3]


def test_checkout_token_resolves_recursively_without_shell_interpolation(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"paths": ["@DRIVERSI_ROOT@/weights/test.ckpt"], "plain": "$HOME"}))
    result = read_mapping(p)
    assert result["paths"] == [str(ROOT / "weights/test.ckpt")]
    assert result["plain"] == "$HOME"
    p.write_text('{"value":"${HOME}"}')
    with pytest.raises(ValueError, match="interpolation"):
        read_mapping(p)


def test_empty_view_does_not_require_old_scene_payload(tmp_path):
    assert scene_index(tmp_path / "not-created", []) == []


def test_materialized_native_yaml_keeps_a_valid_absolute_interpreter():
    value = read_mapping(preset_path("agents/chainflow_vla/lifecycle.yaml"))
    assert value["python"] == str(ROOT / ".venvs/chainflow/bin/python")


def test_missing_private_configuration_does_not_consume_handoff(tmp_path, monkeypatch):
    from driversi.rsi.experiments import adaptive_handoff as module
    root = tmp_path / "exp/test"
    root.mkdir(parents=True)
    (root / "primary-generation-policy.json").write_text('{}')
    monkeypatch.setattr(module, "project_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="Configure adaptive_context"):
        module.run(root, 999999999)
    assert not (root / "protocol-revisions/adaptive-diversity-v1/handoff/started.json").exists()


def test_release_view_has_no_previous_experiments_or_public_deployment():
    config = json.loads((ROOT / "src/config/visualization/platform.json").read_text())
    for key in ("experiments", "training", "production", "scenes", "whitebox", "harness"):
        assert config[key] == []
    assert config["site_root"] == "exp/local-view"
    assert not (ROOT / "src/config/visualization/release.json").exists()
    assert not (ROOT / ".github/workflows/pages.yml").exists()
    exporter = (ROOT / "src/visualization/publication/pages.py").read_text()
    assert "actions/deploy-pages" not in exporter


def test_materializer_is_explicit_and_does_not_overwrite(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"path":"@DRIVERSI_ROOT@/data/maps"}')
    output = tmp_path / "config.json"
    command = [sys.executable, str(ROOT / "scripts/build/materialize_config.py"),
               "--input", str(source), "--output", str(output)]
    subprocess.run(command, check=True)
    assert json.loads(output.read_text())["path"] == str(ROOT / "data/maps")
    assert subprocess.run(command, capture_output=True).returncode != 0
