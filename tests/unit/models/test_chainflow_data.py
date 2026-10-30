"""Exact split admission, bounded source transport, and unchanged native targets."""

import importlib.util
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

from driversi.agents.backends.native import digest, project_root
from driversi_agents.chainflow import data


@pytest.fixture
def sealed_manifest(tmp_path, monkeypatch):
    splits, files = {}, {}
    for side in ("train", "development"):
        row = {
            "sample_id": side,
            "token": side,
            "source_log": side + "-log",
            "kind": "original_navtrain",
            "admission": None,
            "anchor_index": 3,
            "sensor_root": str(tmp_path),
        }
        for field in ("log_path", "metric_cache", "training_metric_cache"):
            path = tmp_path / (side + "-" + field)
            path.write_text("sealed")
            row[field] = str(path)
            files[str(path)] = digest(path)
        row["image_files"] = []
        for camera in range(4):
            path = tmp_path / (side + "-" + str(camera) + ".jpg")
            path.write_text("sealed")
            row["image_files"].append(str(path))
            files[str(path)] = digest(path)
        splits[side] = [row]
    value = {
        "schema_version": 1,
        "kind": "chainflow_native_scene_dataset",
        "splits": splits,
        "files": files,
    }
    monkeypatch.setattr(data, "official_train_logs", lambda: {"train-log", "development-log"})
    monkeypatch.setattr(data, "official_train_tokens", lambda: {"train", "development"})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("split", ["train", "development"])
def test_same_allowed_log_does_not_admit_nonofficial_anchor(sealed_manifest, split):
    path, value = sealed_manifest
    assert data.validate_manifest(path) == value
    value["splits"][split][0]["token"] = "same-log-forged-token"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact official navtrain allowlist"):
        data.validate_manifest(path)


def test_token_alias_cannot_cross_train_and_development(sealed_manifest):
    path, value = sealed_manifest
    value["splits"]["development"][0]["token"] = "train"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Duplicate native token"):
        data.validate_manifest(path)


def test_synthetic_remains_blocked_with_otherwise_valid_identity(sealed_manifest):
    path, value = sealed_manifest
    value["splits"]["train"][0].update(kind="synthetic_navtrain", admission={"trainable": True})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="canonical sealed visual admission"):
        data.validate_manifest(path)


@pytest.mark.parametrize("value", [{}, {"tokens": []}, {"tokens": "not-a-list"}])
def test_missing_exact_official_filter_fails_closed(tmp_path, monkeypatch, value):
    path = tmp_path / (
        "agents/chainflow/native/navsim/planning/script/config/common/"
        "train_test_split/scene_filter/navtrain.yaml"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    monkeypatch.setattr(data, "project_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="Exact official navtrain token allowlist unavailable"):
        data.official_train_tokens()


def source_rows(tmp_path, count=3, future=10):
    rows = []
    for index in range(count):
        token = "token-" + str(index)
        path = tmp_path / ("log-" + str(index) + ".pkl")
        frames = [{"timestamp": i * 500000, "token": token} for i in range(4 + future)]
        with path.open("wb") as stream:
            pickle.dump(frames, stream)
        rows.append(
            {"log_path": str(path), "anchor_index": 3, "token": token, "sensor_root": str(tmp_path)}
        )
    return rows


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, 17])
def test_cache_bound_must_be_explicit_small_positive_integer(maximum):
    with pytest.raises(ValueError, match="max_cached_logs"):
        data.NativeScenes([], object(), max_cached_logs=maximum)
    with pytest.raises(ValueError, match="max_cached_logs"):
        data.verify_records([], max_cached_logs=maximum)


def test_lru_evicts_before_loading_and_reuses_recent_log(tmp_path, monkeypatch):
    rows = source_rows(tmp_path)
    cache = data._LogFrameCache(max_cached_logs=2)
    loaded, load = [], pickle.load

    def observe(stream):
        assert len(cache) < 2  # Capacity is released before the new pickle allocation.
        loaded.append(stream.name)
        return load(stream)

    monkeypatch.setattr(pickle, "load", observe)
    for i in (0, 1, 0, 2):
        assert cache.read(rows[i]["log_path"])[3]["token"] == rows[i]["token"]
        assert len(cache) <= 2
    assert len(loaded) == 3
    assert list(cache) == [rows[0]["log_path"], rows[2]["log_path"]]
    cache.read(rows[1]["log_path"])
    assert len(loaded) == 4


def install_scene_transport(monkeypatch):
    module = ModuleType("navsim.common.dataclasses")

    def build(window, sensor_root, history, future, cameras):
        return SimpleNamespace(window=window, history=history, future=future, cameras=cameras)

    module.Scene = SimpleNamespace(from_scene_dict_list=build)
    module.SensorConfig = SimpleNamespace(build_no_sensors=lambda: "no-sensors")
    monkeypatch.setitem(sys.modules, "navsim.common.dataclasses", module)
    monkeypatch.setattr(data, "validate_consumption", lambda row, window: None)


def test_native_scenes_default_retains_one_log_and_preserves_four_plus_ten(tmp_path, monkeypatch):
    install_scene_transport(monkeypatch)
    rows = source_rows(tmp_path)
    scenes = data.NativeScenes(rows, object())
    for index in (0, 1, 2, 0):
        scene = scenes.scene(index, sensors=False)
        assert len(scenes._frames) == 1
        assert scene.history == 4 and scene.future == 10
        assert len(scene.window) == 14
        assert scene.window[3]["token"] == rows[index]["token"]


def test_verifier_cache_is_bounded_and_preserves_requested_order(tmp_path, monkeypatch):
    rows = source_rows(tmp_path)
    trace, capacities = [], []
    original = data._LogFrameCache.read

    def observe(cache, path):
        frames = original(cache, path)
        capacities.append(len(cache))
        return frames

    monkeypatch.setattr(data._LogFrameCache, "read", observe)
    monkeypatch.setattr(
        data, "validate_consumption", lambda row, window: trace.append(row["token"])
    )
    requested = [rows[i] for i in (0, 1, 2, 0, 1)]
    data.verify_records(requested)
    assert trace == [r["token"] for r in requested]
    assert max(capacities) == 1


def test_eight_future_frames_are_rejected_without_padding(tmp_path, monkeypatch):
    install_scene_transport(monkeypatch)
    rows = source_rows(tmp_path, count=1, future=8)
    with pytest.raises(ValueError, match="4 history \\+ 10 future"):
        data.NativeScenes(rows, object()).scene(0, sensors=False)
    with pytest.raises(ValueError, match="4\\+10"):
        data.verify_records(rows)


def test_native_short_and_stretched_long_targets_are_returned_unchanged(monkeypatch):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    # Only the abstract import is stubbed; execute the actual unmodified native
    # target builder so its dual-horizon interpolation remains the oracle.
    abstract = ModuleType("navsim.planning.training.abstract_feature_target_builder")
    abstract.AbstractTargetBuilder = object
    monkeypatch.setitem(sys.modules, abstract.__name__, abstract)
    path = (
        project_root()
        / "agents/chainflow/native/navsim/agents/chainflow_vla/builders/target_builder.py"
    )
    spec = importlib.util.spec_from_file_location("native_chainflow_target_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.ChainFlowVLATargetBuilder(
        SimpleNamespace(
            trajectory_sampling=SimpleNamespace(num_poses=8),
            long_trajectory_additional_poses=2,
        )
    )
    axis = np.arange(10, dtype=np.float64)
    raw = np.stack((axis * axis, np.sin(axis), axis / 13), axis=1)
    scene = SimpleNamespace(
        get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
            poses=raw[:num_trajectory_frames]
        ),
        scene_metadata=SimpleNamespace(initial_token="native-token"),
    )
    expected = builder.compute_targets(scene)
    actual = data.validated_native_targets(
        scene, SimpleNamespace(get_target_builders=lambda: [builder])
    )
    assert set(actual) == set(expected) == {"trajectory", "trajectory_long", "token"}
    assert actual["token"] == expected["token"]
    for key in ("trajectory", "trajectory_long"):
        assert torch.equal(actual[key], expected[key])
        assert tuple(actual[key].shape) == (8, 3)
    assert not torch.equal(actual["trajectory"], actual["trajectory_long"])
    short_scene = SimpleNamespace(
        get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(
            poses=raw[: min(num_trajectory_frames, 8)]
        )
    )
    with pytest.raises(ValueError, match="Finite complete ten-pose"):
        data.validated_native_targets(
            short_scene, SimpleNamespace(get_target_builders=lambda: [builder])
        )
