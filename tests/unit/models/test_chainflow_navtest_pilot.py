"""Bounded test access remains explicit, immutable, and separate from full scores."""
import copy
import json

import pytest

from driversi.agents.backends.native import digest
from driversi_agents.chainflow import benchmark as b


@pytest.fixture
def population(tmp_path, monkeypatch):
    specs, indices = {}, {}
    for split in ("navtrain", "navtest"):
        rows = [dict(sample_id=f"{split}{i}", token=f"{split}{i}",
                     source_log=f"{split}-log-{i // 2}_00000_00010",
                     metric_cache=str(tmp_path / "cache" / f"{split}{i}" / "metric_cache.pkl"),
                     kind="original_" + split) for i in range(8)]
        source = tmp_path / (split + ".yaml")
        source.write_text(split)
        specs[split] = source, dict(tokens=[r["token"] for r in rows],
                                   log_names=sorted({r["source_log"] for r in rows}))
        index = tmp_path / (split + "-index.json")
        index.write_text(json.dumps(dict(split=split, complete=True, source_stats={},
                                         official_filter_sha256=digest(source),
                                         expected_tokens=specs[split][1]["tokens"], records=rows)))
        indices[split] = index
    monkeypatch.setattr(b, "official_filter", lambda split: specs[split])
    return indices


def pilot(population, tmp_path):
    subset = tmp_path / "subset.json"
    b.make_pilot_subset(population["navtest"], subset, count=3, seed="fixed-before-scores")
    return b.make_pilot_manifest(population["navtrain"], population["navtest"], subset,
                                 tmp_path / "manifest.json")


def test_selection_is_score_blind_stable_and_one_per_original_log(population):
    rows = json.loads(population["navtest"].read_text())["records"]
    expected = b.select_pilot_rows(rows, 3, "fixed-before-scores")
    assert b.select_pilot_rows(list(reversed(rows)), 3, "fixed-before-scores") == expected
    scored = [{**r, "score": 0 if i % 2 else 1} for i, r in enumerate(rows)]
    assert [r["token"] for r in b.select_pilot_rows(scored, 3, "fixed-before-scores")] == [r["token"] for r in expected]
    assert len({b.source_group(r["source_log"]) for r in expected}) == 3
    with pytest.raises(ValueError, match="Insufficient"):
        b.select_pilot_rows(rows, 5, "fixed-before-scores")


def test_pilot_requires_full_training_pool_and_exact_locked_subset(population, tmp_path):
    value = pilot(population, tmp_path)
    assert b.validate_manifest(value)
    assert value["schema_version"] == 3
    for side in ("train", "development"):
        changed = copy.deepcopy(value)
        changed["splits"][side].pop()
        with pytest.raises(ValueError, match="Full navtrain"):
            b.validate_manifest(changed)


def test_pilot_cannot_masquerade_as_complete_benchmark(population, tmp_path):
    value = pilot(population, tmp_path)
    value["schema_version"] = 2
    value["benchmark"]["protocol"] = "official_navtest_v1"
    del value["benchmark"]["test_subset"]
    with pytest.raises(ValueError, match="full navtest"):
        b.validate_manifest(value)


def test_test_rows_never_become_training_rows(population, tmp_path):
    value = pilot(population, tmp_path)
    value["splits"]["train"].append(value["splits"]["development"][0])
    with pytest.raises(ValueError, match="unique admitted navtrain"):
        b.validate_manifest(value)


def test_modified_or_unsealed_pilot_receipt_rejected(population, tmp_path):
    value = pilot(population, tmp_path)
    from pathlib import Path
    path = Path(value["benchmark"]["test_subset"])
    value["files"].pop(str(path))
    with pytest.raises(ValueError, match="Unsealed"):
        b.validate_manifest(value)
    changed = json.loads(path.read_text())
    changed["records"][0]["token"] = "train-token"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="score-blind"):
        b.read_pilot_subset(path)


def test_formal_aggregator_never_accepts_a_subset_as_full_population(population, tmp_path):
    value = pilot(population, tmp_path)
    rows = [{"token": r["token"], "valid": True, **dict.fromkeys(b.METRICS, 1.)}
            for r in value["splits"]["development"]]
    full = json.loads(population["navtest"].read_text())["expected_tokens"]
    result = b.summarize(rows, full)
    assert result["complete"] is False
    assert result["pdms"] is None


def test_standalone_baseline_uses_only_locked_rows(population, tmp_path):
    value = pilot(population, tmp_path)
    config = dict(index=str(population["navtest"]), pilot_subset=value["benchmark"]["test_subset"])
    index = b.read_index(config["index"], "navtest")
    assert b.evaluation_population(config, index) == value["splits"]["development"]
    assert b.evaluation_population({"index": config["index"]}, index) == index["records"]
    with pytest.raises(ValueError, match="cannot select training"):
        b.evaluation_population({**config, "split": "navtrain"}, index)


def test_formal_merge_rejects_even_completed_pilot_receipts(population, tmp_path):
    folder = tmp_path / "pilot-output"
    folder.mkdir()
    (folder / "identity.json").write_text(json.dumps(dict(pilot_protocol=b.PILOT_PROTOCOL)))
    (folder / "scores.jsonl").write_text("")
    with pytest.raises(ValueError, match="cannot enter formal"):
        b.merge(dict(index=str(population["navtest"]), shard_outputs=[str(folder)]))


def test_native_numpy_metrics_have_identical_in_memory_and_serialized_summary():
    np = pytest.importorskip("numpy")
    row = dict(token="native", valid=True, **{k: np.float64(0.95) for k in b.METRICS})
    actual = b.summarize([row], ["native"])
    assert actual == b.summarize(json.loads(json.dumps([row])), ["native"])
    assert actual["complete"] and actual["pdms"] == 95.0
    for value in (True, np.bool_(True), float("nan"), float("inf"), -0.1, 1.1, "0.95"):
        invalid = b.summarize([{**row, "score": value}], ["native"])
        assert not invalid["complete"] and invalid["pdms"] is None
