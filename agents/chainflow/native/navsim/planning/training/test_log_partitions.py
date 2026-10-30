from types import SimpleNamespace

import pytest

from navsim.planning.training.log_partitions import (
    resolve_cache_training_log_names,
    resolve_training_log_names,
)


class SplitConfig(dict):
    __getattr__ = dict.__getitem__


def _config(split):
    return SimpleNamespace(
        train_logs=["train-a", "train-b"],
        val_logs=["val-a"],
        train_test_split=SplitConfig(split),
    )


def test_legacy_navtrain_behavior_is_preserved():
    assert resolve_training_log_names(_config({})) == ["train-a", "train-b"]


def test_legacy_navtrainval_behavior_is_preserved():
    cfg = _config({"include_val_logs_in_train": True})
    assert resolve_training_log_names(cfg) == ["train-a", "train-b", "val-a"]


def test_explicit_navval_partition_selects_only_validation_logs():
    cfg = _config({"training_log_partitions": ["val"]})
    assert resolve_training_log_names(cfg) == ["val-a"]
    assert resolve_cache_training_log_names(cfg) == ["val-a"]


def test_cache_only_legacy_navtrainval_behavior_is_preserved():
    cfg = _config({"include_val_logs_in_train": True})
    assert resolve_cache_training_log_names(cfg) == ["train-a", "train-b"]


@pytest.mark.parametrize(
    "partitions",
    [[], ["test"], ["train", "train"]],
)
def test_invalid_partition_contract_is_rejected(partitions):
    cfg = _config({"training_log_partitions": partitions})
    with pytest.raises(ValueError):
        resolve_training_log_names(cfg)
