"""The full split protocol must reach the real lifecycle, not the legacy trainer."""
from pathlib import Path

import pytest

from driversi.training.orchestration import engine, lifecycle
from driversi.rsi.plugins import training


def test_schema3_engine_routes_to_native_lifecycle(tmp_path, monkeypatch):
    resolved = {"request": {"schema_version": 3}}
    monkeypatch.setattr(engine, "resolve_request", lambda _: resolved)
    seen = []
    def prepare(value, output):
        seen.append((value, output))
        return {"native_lifecycle": True}
    monkeypatch.setattr(lifecycle, "prepare", prepare)
    output = tmp_path / "exp"
    assert engine.prepare(tmp_path / "request.json", output) == {"native_lifecycle": True}
    assert seen == [(resolved, output)]


def test_schema3_harness_uses_declared_benchmark_sources(tmp_path, monkeypatch):
    request, data, model = [tmp_path / n for n in ("request.json", "data.json", "model.json")]
    request.write_text("{}")
    data.write_text('{}')
    model.write_text('{}')
    monkeypatch.setattr(training, "resolve_request", lambda _: dict(
        request=dict(schema_version=3, data=str(data), budget=dict(max_scoring_agent_steps=971680,
                                                               max_environment_steps=0)),
        model="chainflow_stage1", model_config=str(model), model_config_sha256="fixed"))
    monkeypatch.setattr(lifecycle, "sources", lambda _: {"train": ["train"], "development": ["test"]})
    config, cap = training.prepare_job({"request": str(request)}, tmp_path, tmp_path / "registry")
    assert config["source_groups"] == ["test", "train"]
    assert cap == 971680


def test_legacy_test_data_stays_forbidden_without_explicit_protocol(tmp_path, monkeypatch):
    import yaml
    monkeypatch.setattr(lifecycle, "project_root", lambda: tmp_path)
    source = tmp_path / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump({"log_names": ["test-log"]}))
    data = {"parent_provenance": "unknown", "train": {"source_logs": ["train-log"]},
            "development": {"source_logs": ["test-log"]}}
    with pytest.raises(ValueError):
        lifecycle.sources(data)


def test_original_gap_compatibility_never_relaxes_synthetic_continuity():
    from driversi_agents.chainflow.data import validate_window_times
    times = [0, 500000, 1500000, 2000000]
    original = {"kind": "original_navtest"}
    with pytest.raises(ValueError, match="Discontinuous"):
        validate_window_times(original, times)
    assert not validate_window_times(original, times, native_original_windows=True)
    with pytest.raises(ValueError, match="Discontinuous"):
        validate_window_times({"kind": "synthetic_navtrain"}, times, native_original_windows=True)
    with pytest.raises(ValueError, match="non-increasing"):
        validate_window_times(original, [0, 0, 500000, 1000000], native_original_windows=True)
