import copy

import numpy as np
import pytest

from driversi.contracts.transition import Observation
from driversi.training.legacy.advantages import generalized_advantage
from driversi.training.sampling.environment import features, source_group, validate_splits


def test_time_is_seconds_and_not_number_of_input_frames():
    obs = Observation(("a",), np.zeros((1, 1124)), np.array([True]), 10)
    assert features(obs, 60)[0, -1] == pytest.approx(5 / 9)


@pytest.mark.parametrize("end", ["terminated", "truncated"])
def test_gae_does_not_cross_reset_or_bootstrap_finite_task(end):
    rewards = np.array([[1.], [100.]])
    z = np.zeros((2, 1))
    valid = np.ones((2, 1), bool)
    terminal = np.zeros((2, 1), bool)
    truncated = terminal.copy()
    (terminal if end == "terminated" else truncated)[0] = True
    adv, ret = generalized_advantage(rewards, z, z + 999, valid, terminal, truncated,
                                     gamma=1., lam=1.)
    assert adv[0, 0] == ret[0, 0] == 1


def test_collection_boundary_bootstraps_but_invalid_agent_does_not():
    adv, ret = generalized_advantage(np.array([[2., 999.]]), np.array([[3., 888.]]),
                                     np.array([[10., 777.]]), np.array([[True, False]]),
                                     np.zeros((1, 2), bool), np.zeros((1, 2), bool),
                                     gamma=.5)
    np.testing.assert_array_equal(adv, [[4, 0]])
    np.testing.assert_array_equal(ret, [[7, 0]])


def test_shards_from_same_original_drive_cannot_cross_split(straight_scene, tmp_path):
    a = copy.deepcopy(straight_scene)
    b = copy.deepcopy(straight_scene)
    a.metadata["source_log"] = "2021.01.01_veh-001_00010_00020"
    b.metadata["source_log"] = "2021.01.01_veh-001_00500_00600"
    assert source_group(a) == source_group(b) == "2021.01.01_veh-001"
    a.save(tmp_path / "a.npz")
    b.save(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits([tmp_path / "a.npz"], [tmp_path / "b.npz"])


def test_direct_training_rejects_test_data_and_duplicate_paths(straight_scene, tmp_path):
    a = copy.deepcopy(straight_scene)
    b = copy.deepcopy(straight_scene)
    a.metadata["source_log"] = "train-only"
    b.metadata["source_log"] = "eval-only"
    a.save(tmp_path / "a.npz")
    b.metadata["split"] = "test"
    b.save(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="heldout/test"):
        validate_splits([tmp_path / "a.npz"], [tmp_path / "b.npz"])
    b.metadata["split"] = "unit"
    b.save(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits([tmp_path / "a.npz"] * 2, [tmp_path / "b.npz"])


def test_scene_inventory_is_lazy_and_cache_bounded(straight_scene, tmp_path, monkeypatch):
    from driversi.contracts.scene import Scene
    from driversi.data.scenes.scene_store import LazyScenes, scene_header

    paths = []
    for i in range(4):
        path = tmp_path / f"scene-{i}.npz"
        straight_scene.save(path)
        paths.append(path)
    original = Scene.load
    reads = []

    def load(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Scene, "load", load)
    store = LazyScenes(paths, 2)
    assert len(store) == 4 and not reads
    assert scene_header(paths[0])["controlled_ids"] == ["a", "b"]
    assert not reads
    _ = store[0], store[0], store[1], store[2]
    assert len(reads) == 3 and len(store.cache.entries) == 2
    _ = store[0]
    assert len(reads) == 4 and len(store.cache.entries) == 2


@pytest.mark.parametrize("invalid", ["test", "heldout", "duplicate", "empty", "overlap"])
def test_direct_evaluation_rejects_invalid_partition_before_native_load(straight_scene, tmp_path, invalid):
    from driversi.evaluation.search.teacher import evaluate

    train = copy.deepcopy(straight_scene)
    train.metadata["source_log"] = "training-log"
    train_path = tmp_path / "train.npz"
    train.save(train_path)
    evaluation = copy.deepcopy(straight_scene)
    evaluation.metadata["source_log"] = "training-log" if invalid == "overlap" else "evaluation-log"
    if invalid in ("test", "heldout"):
        evaluation.metadata["split"] = invalid
    path = tmp_path / "evaluation.npz"
    evaluation.save(path)
    paths = [] if invalid == "empty" else [path] * (2 if invalid == "duplicate" else 1)
    reason = {"test": "heldout/test", "heldout": "heldout/test", "duplicate": "Duplicate",
              "empty": "empty", "overlap": "overlaps"}[invalid]
    with pytest.raises(ValueError, match=reason):
        evaluate(None, paths, **({"training_paths": [train_path]} if invalid == "overlap" else {}))


@pytest.mark.parametrize("split", ["navtest", "navhard", "NAVTEST", " NavHard ",
                                  "navtest-v2", "navhard_two_stage", "nav-test",
                                  "nav_hard", "held-out", "holdout", "test-mini"])
def test_protected_benchmark_aliases_cannot_drive_development(straight_scene, tmp_path, split):
    from driversi.training.sampling.environment import validate_development_paths

    scene = copy.deepcopy(straight_scene)
    scene.metadata["split"] = split
    path = tmp_path / "protected.npz"
    scene.save(path)
    with pytest.raises(ValueError, match="not development"):
        validate_development_paths([path])
