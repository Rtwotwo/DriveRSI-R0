"""Admission, budget and evidence separation for bounded native confirmation."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from driversi_agents.chainflow import mining


def config():
    return {
        "schema_version": 1,
        "manifest": "manifest.json",
        "selected_tokens": ["train"],
        "allowed_source_logs": ["drive_001_002"],
        "stage1_config": "stage1.json",
        "output_dir": "exp/confirmation",
        "cuda_device": 0,
        "max_samples": 1,
        "max_scoring_steps": 80,
    }


def manifest():
    return {
        "schema_version": 1,
        "kind": "chainflow_native_scene_dataset",
        "files": {},
        "splits": {
            "train": [
                {
                    "sample_id": "train",
                    "token": "train",
                    "source_log": "drive_001_002",
                    "kind": "original_navtrain",
                    "admission": None,
                    "anchor_index": 3,
                }
            ],
            "development": [{"token": "dev", "source_log": "heldout_003_004"}],
        },
    }


def test_train_selection_preserves_requested_order_and_requires_both_allowlists():
    cfg, data = config(), manifest()
    assert mining.select_records(data, cfg, {"drive_001_002"}) == data["splits"]["train"]
    with pytest.raises(ValueError, match="allowlists"):
        mining.select_records(data, cfg, set())
    cfg["allowed_source_logs"] = ["other_001_002"]
    with pytest.raises(ValueError, match="allowlists"):
        mining.select_records(data, cfg, {"drive_001_002"})


@pytest.mark.parametrize("split", ["development", "test", "protected", "unknown"])
def test_any_nontrain_split_is_ineligible(split):
    cfg, data = config(), manifest()
    data["splits"][split] = data["splits"].pop("development")
    cfg["selected_tokens"] = ["dev"]
    with pytest.raises(ValueError, match="outside eligible train"):
        mining.select_records(data, cfg, {"drive_001_002", "heldout_003_004"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("protected_tokens", "train"),
        ("development_tokens", "train"),
        ("protected_source_logs", "drive_999_1000"),
        ("development_source_logs", "drive_999_1000"),
    ],
)
def test_explicit_deny_lists_include_source_group_relatives(field, value):
    cfg = config()
    cfg[field] = [value]
    with pytest.raises(ValueError, match="eligible train|overlaps"):
        mining.select_records(manifest(), cfg, {"drive_001_002"})


def test_manifest_heldout_group_and_duplicate_identity_rejected():
    data = manifest()
    data["splits"]["development"][0]["source_log"] = "drive_003_004"
    with pytest.raises(ValueError, match="overlaps"):
        mining.select_records(data, config(), {"drive_001_002"})
    data["splits"]["development"][0]["token"] = "train"
    with pytest.raises(ValueError, match="Duplicate"):
        mining.select_records(data, config(), {"drive_001_002"})


def test_synthetic_records_cannot_enter_confirmation():
    data = manifest()
    data["splits"]["train"][0]["kind"] = "synthetic_navtrain"
    with pytest.raises(ValueError, match="original navtrain"):
        mining.select_records(data, config(), {"drive_001_002"})


def test_allowed_log_does_not_admit_a_token_outside_exact_navtrain_filter():
    rows = mining.select_records(manifest(), config(), {"drive_001_002"})
    with pytest.raises(ValueError, match="exact official navtrain filter"):
        mining.validate_official_tokens(rows, {"other-token-from-same-log"})
    mining.validate_official_tokens(rows, {"train"})


@pytest.mark.parametrize(
    "changes",
    [
        {"max_samples": 65},
        {"max_samples": True},
        {"max_scoring_steps": 79},
        {"max_scoring_steps": 5121},
        {"selected_tokens": ["train", "train"]},
        {"cuda_device": -1},
        {"gpu_memory_fraction": 0.5},
        {"score_threshold": 0.9},
    ],
)
def test_invalid_or_unbounded_requests_fail_before_runtime(changes):
    cfg = dict(config(), **changes)
    with pytest.raises(ValueError):
        mining.validate_config(cfg)


def test_expanded_confirmation_still_has_a_fixed_scoring_cap():
    cfg = dict(config(), selected_tokens=['t' + str(i) for i in range(48)],
               max_samples=48, max_scoring_steps=48 * 80)
    mining.validate_config(cfg)
    with pytest.raises(ValueError):
        mining.validate_config(dict(cfg, max_scoring_steps=48 * 80 - 1))


def test_threshold_boundary_and_diagnostics_do_not_assert_cause():
    values = dict.fromkeys(mining.METRICS, 1.0)
    values.update(score=0.95, ego_progress=0.9)
    result = mining.classify_metrics(values)
    assert result["classification"] == "threshold_met"
    assert result["diagnostic_submetrics_below_one"] == ["ego_progress"]
    assert result["causal_model_failure"] is None
    values["score"] = 0.9499999
    assert mining.classify_metrics(values)["classification"] == "threshold_shortfall"
    values["score"] = float("nan")
    with pytest.raises(ValueError, match="Invalid native"):
        mining.classify_metrics(values)


def test_selected_file_seals_exclude_heldout_and_detect_mutation(tmp_path):
    row = {}
    for key in ("log_path", "metric_cache", "training_metric_cache"):
        p = tmp_path / key
        p.write_text(key)
        row[key] = str(p)
    row["image_files"] = []
    for i in range(4):
        p = tmp_path / (str(i) + ".jpg")
        p.write_text(str(i))
        row["image_files"].append(str(p))
    files = {str(p): mining.digest(p) for p in tmp_path.iterdir()}
    files[str(tmp_path / "unavailable-heldout.pkl")] = "never-read"
    assert len(mining.verify_seals([row], files)) == 7
    Path(row["metric_cache"]).write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        mining.verify_seals([row], files)


def test_runtime_failure_remains_separate_from_shortfall_receipt(tmp_path, monkeypatch):
    """Mock transport/runtime only: assert per-row failures cannot become failures labels."""
    from driversi_agents.chainflow import data as native_data

    cfg, data = config(), manifest()
    cfg.update(selected_tokens=["train", "train2"], max_samples=2, max_scoring_steps=160)
    data["splits"]["train"].append(
        dict(deepcopy(data["splits"]["train"][0]), sample_id="train2", token="train2")
    )
    stage = {
        "model": "chainflow_stage1",
        "native_root": "native",
        "native_config": "agent.yaml",
        "checkpoint": "checkpoint.ckpt",
        "backbone": "backbone.pt",
        "inference_adapter": "driversi_agents.chainflow.stage1:trajectory_from_input",
    }
    paths = [
        "checkpoint.ckpt",
        "backbone.pt",
        "native/agent.yaml",
        "agents/chainflow/stage1.py",
        "agents/chainflow/data.py",
        "agents/chainflow/mining.py",
        "native/navsim/planning/metric_caching/metric_cache_processor.py",
        "native/navsim/evaluate/pdm_score.py",
        "native/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        "native/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sealed")
    for name, value in (("config.json", cfg), ("manifest.json", data), ("stage1.json", stage)):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(mining, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mining, "official_train_logs", lambda: {"drive_001_002"})
    monkeypatch.setattr(mining, "official_train_tokens", lambda: {"train", "train2"})
    monkeypatch.setattr(mining, "verify_seals", lambda rows, files: {})
    monkeypatch.setattr(mining, "_runtime", lambda *args: None)
    verified = []
    monkeypatch.setattr(native_data, "verify_records", lambda rows: verified.extend(rows))
    monkeypatch.setattr(mining, "_gpu_snapshot", lambda device: {"index": device})
    monkeypatch.setattr(mining, "_agent", lambda *args: object())

    def confirm(agent, scenes, index, row):
        if index == 1:
            raise RuntimeError("injected native runtime error")
        metrics = dict.fromkeys(mining.METRICS, 1.0)
        metrics["score"] = 0.8
        return dict(mining.classify_metrics(metrics), metrics=metrics, poses=[[0, 0, 0]] * 8)

    monkeypatch.setattr(mining, "_confirm_one", confirm)
    receipt = mining.run(tmp_path / "config.json")
    assert [r["token"] for r in verified] == ["train", "train2"]
    assert receipt["status"] == "completed_with_runtime_failures"
    assert receipt["counts"] == {"threshold_shortfall": 1, "threshold_met": 0, "runtime_failure": 1}
    assert receipt["rows"][1]["classification"] is None
    assert receipt["rows"][1]["causal_model_failure"] is None
    assert receipt["budget"]["reserved_scoring_steps"] == 160
    assert receipt["budget"]["completed_scoring_steps"] == 80
    assert receipt["budget"]["optimizer_steps"] == 0
    assert receipt["checkpoint_unchanged"]
    assert json.loads((tmp_path / "exp/confirmation/receipt.json").read_text()) == receipt


def test_missing_reference_uses_native_builder_and_preserves_training_cache(tmp_path, monkeypatch):
    import lzma
    import pickle
    import sys
    import types

    source = tmp_path / "training.pkl"
    source.write_bytes(lzma.compress(pickle.dumps(types.SimpleNamespace(pdm_progress=7))))
    before = source.read_bytes()
    out = tmp_path / "attempt" / "evaluation-cache"
    target = out / "drive" / "unknown" / "token" / "metric_cache.pkl"
    calls = []

    class Processor:
        def __init__(self, path, force):
            calls.append((path, force))

        def compute_metric_cache(self, scene):
            calls.append(scene)
            target.parent.mkdir(parents=True)
            target.write_bytes(lzma.compress(pickle.dumps(types.SimpleNamespace(
                trajectory="native reference", file_path=target))))
            return types.SimpleNamespace(file_name=target)

    processor = types.ModuleType("navsim.planning.metric_caching.metric_cache_processor")
    processor.MetricCacheProcessor = Processor
    scenario = types.ModuleType("navsim.planning.scenario_builder.navsim_scenario")
    scenario.NavSimScenario = lambda scene, **kw: (scene, kw)
    monkeypatch.setitem(sys.modules, processor.__name__, processor)
    monkeypatch.setitem(sys.modules, scenario.__name__, scenario)
    monkeypatch.setenv("NUPLAN_MAPS_ROOT", "/maps")
    scenes = types.SimpleNamespace(evaluation_cache_dir=out,
                                  scene=lambda i, sensors: (i, sensors))
    row = dict(metric_cache=str(source), kind="original_navtrain", token="token", source_log="drive")
    cache, proof = mining.evaluation_cache(scenes, 2, row)
    assert source.read_bytes() == before
    assert cache.trajectory == "native reference" and proof["rebuilt"]
    assert calls == [(str(out), False), ((2, False), dict(map_root="/maps", map_version="nuplan-maps-v1.0"))]
    assert proof["sha256"] == mining.digest(target)
    with pytest.raises(ValueError, match="outside original"):
        mining.evaluation_cache(scenes, 2, dict(row, kind="synthetic_navtrain"))
