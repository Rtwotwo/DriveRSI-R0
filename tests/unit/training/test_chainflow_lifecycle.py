import json

import pytest

from driversi.agents.backends.native import digest
from driversi.training.contracts.protocols import LIFECYCLE_BACKENDS
from driversi.training.orchestration import lifecycle
from driversi_agents.chainflow.data import validate_manifest


@pytest.fixture(autouse=True)
def source_allowlist(monkeypatch):
    monkeypatch.setattr(
        "driversi_agents.chainflow.data.official_train_tokens", lambda: {"train", "development"}
    )
    monkeypatch.setattr(
        "driversi_agents.chainflow.data.official_train_logs",
        lambda: {"train-log", "development-log", "drive_00001_00100", "drive_00101_00200"},
    )


def manifest(tmp_path):
    files = {}
    splits = {}
    for side in ("train", "development"):
        paths = []
        for suffix in ("log", "metric", "training", "f0", "l0", "r0", "b0"):
            p = tmp_path / (side + suffix)
            p.write_text("sealed")
            paths.append(str(p))
            files[str(p)] = digest(p)
        splits[side] = [
            {
                "sample_id": side,
                "source_log": side + "-log",
                "kind": "original_navtrain",
                "log_path": paths[0],
                "sensor_root": str(tmp_path),
                "anchor_index": 3,
                "token": side,
                "metric_cache": paths[1],
                "training_metric_cache": paths[2],
                "image_files": paths[3:],
                "admission": None,
            }
        ]
    p = tmp_path / "manifest.json"
    value = {
        "schema_version": 1,
        "kind": "chainflow_native_scene_dataset",
        "splits": splits,
        "files": files,
    }
    p.write_text(json.dumps(value))
    return p, value


def test_chainflow_uses_existing_lifecycle_worker():
    assert lifecycle.WORKERS["chainflow_stage1"] == "driversi_agents.chainflow.training"
    assert ("chainflow_stage1", "supervised") in LIFECYCLE_BACKENDS
    assert ("chainflow_stage1", "online_rl") not in LIFECYCLE_BACKENDS


def test_sealed_original_inputs_are_accepted(tmp_path):
    p, value = manifest(tmp_path)
    assert validate_manifest(p) == value


def test_modified_image_cannot_enter_training(tmp_path):
    p, value = manifest(tmp_path)
    from pathlib import Path

    Path(value["splits"]["train"][0]["image_files"][0]).write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        validate_manifest(p)


def test_synthetic_inspection_is_never_training_admission(tmp_path):
    p, value = manifest(tmp_path)
    value["splits"]["train"][0].update(kind="synthetic_navtrain", admission={"trainable": True})
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="canonical sealed visual admission"):
        validate_manifest(p)


def test_same_original_drive_shards_are_not_holdout(tmp_path):
    p, value = manifest(tmp_path)
    value["splits"]["train"][0]["source_log"] = "drive_00001_00100"
    value["splits"]["development"][0]["source_log"] = "drive_00101_00200"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(p)


def test_native_scene_requires_explicit_temporal_and_cache_inputs(tmp_path):
    p, value = manifest(tmp_path)
    del value["splits"]["train"][0]["training_metric_cache"]
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_manifest(p)


def test_source_group_preserves_dots_in_original_timestamp():
    from driversi_agents.chainflow.data import source_group

    assert source_group("2021.08.17.18.11.12_veh-08_00234_00611") == "2021.08.17.18.11.12_veh-08"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from driversi.training.contracts.job import resolve_request

    root = tmp_path
    monkeypatch.setattr(lifecycle, "project_root", lambda: root)
    monkeypatch.setattr(lifecycle, "require_source", lambda *a: None)
    monkeypatch.setattr(
        lifecycle,
        "sources",
        lambda data: {"train": ["train-log"], "development": ["development-log"]},
    )
    data_manifest, _ = manifest(root)
    for name in (
        "agents/chainflow/data.py",
        "agents/chainflow/benchmark.py",
        "agents/chainflow/synthetic.py",
        "src/rsi/correction/trajectory_admission.py",
        "src/data/datasets/navsim_synthetic.py",
        "src/rendering/diagnostics/worldengine_admission.py",
        "agents/chainflow/training.py",
        "agents/chainflow/distributed.py",
        "agents/chainflow/sampling.py",
        "agents/chainflow/stage1.py",
        "agents/chainflow/native/source-manifest.json",
        "src/training/orchestration/worker.py",
        "src/training/orchestration/state.py",
        "agents/chainflow/native/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
        "agents/chainflow/native/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
        "weights/parent.ckpt",
        "weights/backbone.pt",
        "python",
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixed")
    cfg = {
        "model": "chainflow_stage1",
        "family": "end_to_end_planner",
        "repository": "agents/chainflow/native",
        "revision": "pinned",
        "python": "python",
        "weights": {"checkpoint": "weights/parent.ckpt", "backbone": "weights/backbone.pt"},
        "settings": {"gpu": 0},
    }
    (root / "model.json").write_text(json.dumps(cfg))
    (root / "data.json").write_text(
        json.dumps({"manifest": str(data_manifest), "train": {}, "development": {}})
    )
    request = {
        "schema_version": 2,
        "id": "test",
        "model_config": "model.json",
        "mode": "supervised",
        "initialization": "model_weights",
        "data": "data.json",
        "pilot": {"seed": 1, "batch_size": 1, "evaluation_steps": 1},
        "budget": {
            "max_wall_seconds": 20,
            "max_optimizer_steps": 1,
            "max_environment_steps": 0,
            "max_scoring_agent_steps": 2560,
        },
    }
    (root / "request.json").write_text(json.dumps(request))
    resolved = resolve_request(root / "request.json")
    plan = lifecycle.prepare(resolved, root / "exp/first")
    receipt = root / "exp/first/training-result.json"
    receipt.write_text(
        json.dumps({"status": "completed", "context": plan["context"], "checkpoint_files": {}})
    )
    return root, resolved, receipt


def test_changed_manifest_anchor_cannot_resume_at_same_path(prepared):
    root, resolved, receipt = prepared
    p = root / "manifest.json"
    value = json.loads(p.read_text())
    value["splits"]["train"][0]["anchor_index"] = 4
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Continuation context"):
        lifecycle.prepare(resolved, root / "exp/second", phase="resume", previous=receipt)


def test_changed_parent_weight_bytes_cannot_resume_at_same_path(prepared):
    root, resolved, receipt = prepared
    (root / "weights/parent.ckpt").write_text("changed")
    with pytest.raises(ValueError, match="Continuation context"):
        lifecycle.prepare(resolved, root / "exp/second", phase="resume", previous=receipt)


def test_changed_adapter_cannot_resume_under_old_recipe(prepared):
    root, resolved, receipt = prepared
    (root / "agents/chainflow/training.py").write_text("changed")
    with pytest.raises(ValueError, match="Continuation context"):
        lifecycle.prepare(resolved, root / "exp/second", phase="resume", previous=receipt)


def test_actual_camera_bindings_are_validated(tmp_path):
    import lzma
    import pickle
    from types import SimpleNamespace

    from driversi_agents.chainflow.data import validate_consumption

    images = []
    cameras = {}
    for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
        p = tmp_path / (cam + ".png")
        p.write_text("pixels")
        images.append(str(p))
        cameras[cam] = {"data_path": p.name}
    cache_path = tmp_path / "train-log/unknown/token/metric_cache.pkl"
    cache_path.parent.mkdir(parents=True)
    cache = SimpleNamespace(
        file_path=cache_path,
        ego_state=SimpleNamespace(time_point=SimpleNamespace(time_us=1500000)),
        pdm_progress=1.0,
    )
    with lzma.open(cache_path, "wb") as stream:
        pickle.dump(cache, stream)
    row = {
        "source_log": "train-log",
        "log_path": str(tmp_path / "train-log.pkl"),
        "token": "token",
        "sensor_root": str(tmp_path),
        "image_files": images,
        "metric_cache": str(cache_path),
        "training_metric_cache": str(cache_path),
    }
    window = [
        {"log_name": "train-log", "timestamp": 500000 * i, "token": "token", "cams": cameras}
        for i in range(14)
    ]
    validate_consumption(row, window)
    row["image_files"] = images[::-1]
    with pytest.raises(ValueError, match="Actual consumed images"):
        validate_consumption(row, window)
    row["image_files"] = images
    window[3]["timestamp"] = 1500001
    with pytest.raises(ValueError, match="another anchor"):
        validate_consumption(row, window)


def test_native_long_target_fallback_is_rejected():
    pytest.importorskip("scipy")
    from types import SimpleNamespace

    import numpy as np

    from driversi_agents.chainflow.data import validated_native_targets

    raw = np.stack([np.arange(10), np.zeros(10), np.zeros(10)], axis=-1).astype(float)
    scene = SimpleNamespace(
        get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
            poses=raw[:num_trajectory_frames]
        )
    )
    builder = SimpleNamespace(compute_targets=lambda s: {"trajectory_long": raw[:8]})
    agent = SimpleNamespace(get_target_builders=lambda: [builder])
    with pytest.raises(ValueError, match="fell back"):
        validated_native_targets(scene, agent)
    raw[9, 0] = np.nan
    with pytest.raises(ValueError, match="Finite complete"):
        validated_native_targets(scene, agent)
