import json
from pathlib import Path

import pytest

from driversi.config import load, path
from driversi.visualization.catalog import config_guide, experiment, scalar, typed_scalar

ROOT = Path(__file__).resolve().parents[3]


def test_missing_evidence_is_not_success(tmp_path):
    (tmp_path / "exp").mkdir()
    row = dict(
        id="trial", title="Trial", model="Student", scope="pilot", receipt="exp/missing.json"
    )
    assert experiment(row, tmp_path)["status"] == "missing"


def test_projection_preserves_scope_and_excludes_raw_fields(tmp_path):
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp/run.json").write_text(
        json.dumps(
            {
                "status": "done",
                "score": 0.91,
                "secret": "sk-secret123456789012345",
                "navtest_tokens": ["test-token"],
                "path": "/data/private/key",
                "promoted": False,
            }
        )
    )
    row = dict(
        id="x",
        title="X",
        model="S",
        scope="four-scene pilot",
        receipt="exp/run.json",
        fields=[
            dict(label="PDMS", path="score", type="percent"),
            dict(label="promoted", path="promoted", type="bool"),
        ],
    )
    value = experiment(row, tmp_path)
    assert value["fields"][0]["value"] == 0.91
    assert value["fields"][1]["value"] is False
    assert value["scope"] == "four-scene pilot"
    assert "test-token" not in json.dumps(value) and "secret" not in json.dumps(value)
    assert value["status"] == "recorded"


def test_nonfinite_or_nested_payload_is_not_a_display_scalar():
    for value in (float("nan"), float("inf"), {"token": "private"}, [1, 2]):
        with pytest.raises(ValueError):
            scalar(value)
    assert scalar("/data/private/key") == "内部记录（未导出）"


def test_receipts_cannot_read_external_files(tmp_path):
    row = dict(id="x", title="X", model="S", scope="pilot", receipt="/etc/passwd")
    with pytest.raises(ValueError):
        experiment(row, tmp_path)


def test_config_directory_has_explicit_roles_and_preserved_dependencies():
    rows = config_guide(ROOT)
    assert all(r["exists"] for r in rows)
    assert "policies/continual-scene-learning.json" in {r["id"] for r in rows}
    assert all(
        r["role"] in {"examples", "tasks", "skills", "studies", "policies", "registry"}
        for r in rows
    )
    assert not path("rsi/catalog.json").parent.joinpath("chainflow_navtrain_mining.json").exists()
    cfg = load("rsi/tasks/training/augustus.json")
    source = path("rsi/tasks/training/augustus.json")
    assert (source.parent / cfg["steps"][0]["config"]["request"]).resolve() == path(
        "training/augustus.json"
    )


def test_published_metrics_are_aggregate_selections_only():
    cfg = load("visualization/platform.json")
    for exp in cfg["experiments"]:
        for row in exp.get("fields", []):
            assert not any(
                k in row["path"].lower() for k in ["token", "checkpoint", "parent_assets"]
            )
            assert not any(
                k in row["path"].lower().split(".") for k in ["source_log", "source_logs"]
            )


def test_boolean_cannot_be_reported_as_cost_or_score():
    with pytest.raises(ValueError):
        typed_scalar(False, "number")
    with pytest.raises(ValueError):
        typed_scalar(True, "percent")
    with pytest.raises(ValueError):
        typed_scalar(95, "percent")
    with pytest.raises(ValueError):
        typed_scalar(1, "bool")
    assert typed_scalar(False, "bool") is False


def test_local_publish_preserves_legacy_and_is_repeatable(tmp_path, monkeypatch):
    from driversi.visualization import publish as module

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"site_root": str(tmp_path)}))
    (tmp_path / "index.html").write_text("original-page")
    monkeypatch.setattr(module, "build", lambda _: {"status": "built"})
    first = module.publish(config)
    second = module.publish(config)
    assert first == second and not second["git_pushed"]
    assert (tmp_path / "legacy-showcase.html").read_text() == "original-page"
    assert module.MARKER in (tmp_path / "index.html").read_text()
    (tmp_path / "index.html").write_text("different-old-page")
    with pytest.raises(ValueError):
        module.publish(config)
    assert (tmp_path / "legacy-showcase.html").read_text() == "original-page"
