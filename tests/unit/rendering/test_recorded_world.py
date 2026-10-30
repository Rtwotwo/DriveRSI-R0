import hashlib
import json

import pytest

from driversi.rendering.gaussian.recorded_world import RecordedWorldWorker


@pytest.fixture
def record(tmp_path):
    paths = {k: tmp_path / k for k in ("scene", "road_graph")}
    for p in paths.values():
        p.write_bytes(b"asset")
    data = {
        "inputs": {
            k: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()
        },
        "initial_world_by_generation": {"0": {"tick": 0, "state": [0], "static": "retained"}},
        "steps": [
            {"generation": 0, "tick": 1, "state": [1]},
            {"generation": 0, "tick": 2, "state": [2]},
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    return dict(manifest=manifest, **paths), data


def test_read_only_sequence_and_copy(record):
    args, _ = record
    worker = RecordedWorldWorker(**args)
    result = worker.request("step", expected_tick=0)
    result["state"][0] = 99
    assert worker.current["state"] == [1]
    assert worker.request("step", expected_tick=1)["static"] == "retained"
    with pytest.raises(ValueError, match="exhausted"):
        worker.request("step", expected_tick=2)


@pytest.mark.parametrize(
    "operation,extra",
    [
        ("reset", {}),
        ("snapshot", {}),
        ("step", {"overrides": {"ego": 1}}),
        ("step", {"actions": [1]}),
    ],
)
def test_new_simulation_rejected(record, operation, extra):
    args, _ = record
    with pytest.raises(ValueError, match="cannot simulate"):
        RecordedWorldWorker(**args).request(operation, expected_tick=0, **extra)


@pytest.mark.parametrize("change", ["hash", "generation", "gap"])
def test_corrupt_sequence_rejected(record, change):
    args, data = record
    if change == "hash":
        data["inputs"]["scene"]["sha256"] = "bad"
    elif change == "generation":
        for step in data["steps"]:
            step["generation"] = 1
    else:
        data["steps"][1]["tick"] = 3
    args["manifest"].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        RecordedWorldWorker(**args)


def test_stale_and_closed_rejected(record):
    args, _ = record
    worker = RecordedWorldWorker(**args)
    with pytest.raises(ValueError, match="Stale"):
        worker.request("step", expected_tick=1)
    worker.close()
    with pytest.raises(ValueError, match="cannot simulate"):
        worker.request("step", expected_tick=0)
