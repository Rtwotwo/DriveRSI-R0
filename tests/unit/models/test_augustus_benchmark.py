import copy
import json

import pytest

from driversi.agents.backends.native import digest
from driversi_agents.augustus import benchmark


@pytest.fixture
def locked(tmp_path, monkeypatch):
    subset = tmp_path / "subset.json"
    subset.write_text("locked population")
    monkeypatch.setattr(benchmark, "read_pilot_subset", lambda p: {
        "records": [{"token": "test-anchor", "source_log": "test-log"}],
        "test_index": "index", "test_index_sha256": "sealed",
    })
    monkeypatch.setattr(benchmark, "official_filter", lambda split: (
        None, {"tokens": ["train-anchor", "validation-anchor"],
               "log_names": ["train-log", "validation-log"]}
    ))
    return {
        "benchmark": {"protocol": "official_navtest_v1_locked_pilot",
                      "test_subset": str(subset), "test_subset_sha256": digest(subset)},
        "splits": {
            "train": {"records": [{"frame_id": "train-anchor", "log_name": "train-log"}]},
            "validation": {"records": [{"frame_id": "validation-anchor", "log_name": "validation-log"}]},
            "development": {"records": [{"frame_id": "test-anchor", "log_name": "test-log"}]},
        },
    }


def test_exact_locked_population(locked):
    assert benchmark.validate_membership(locked)["records"]
    for records in ([], locked["splits"]["development"]["records"] * 2,
                    [{"frame_id": "other", "log_name": "test-log"}]):
        changed = copy.deepcopy(locked)
        changed["splits"]["development"]["records"] = records
        with pytest.raises(ValueError, match="locked official population"):
            benchmark.validate_membership(changed)


@pytest.mark.parametrize("side", ["train", "validation"])
def test_test_token_or_log_cannot_become_optimization_input(locked, side):
    for field, value in (("frame_id", "test-anchor"), ("log_name", "test-log")):
        changed = copy.deepcopy(locked)
        changed["splits"][side]["records"][0][field] = value
        with pytest.raises(ValueError, match="Non-navtrain"):
            benchmark.validate_membership(changed)


def test_changed_selection_file_rejected(locked):
    from pathlib import Path

    Path(locked["benchmark"]["test_subset"]).write_text("resampled after seeing scores")
    with pytest.raises(ValueError, match="subset changed"):
        benchmark.validate_membership(locked)


def test_joint_original_pool_still_checked(locked):
    block = locked["splits"]["train"]
    block["joint_source_blocks"] = [copy.deepcopy(block), {"records": []}]
    block["records"].append({"frame_id": "synthetic-id", "log_name": "train-log"})
    benchmark.validate_membership(locked)
    block["joint_source_blocks"][0]["records"][0]["frame_id"] = "test-anchor"
    with pytest.raises(ValueError, match="Non-navtrain"):
        benchmark.validate_membership(locked)
