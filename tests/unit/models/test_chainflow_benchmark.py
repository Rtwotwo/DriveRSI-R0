"""No trained-source test substitution, partial benchmark claims, or lost mixing."""
import json
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest
from driversi_agents.chainflow import benchmark as b


def test_missing_failed_duplicate_and_nonfinite_never_report_complete():
    row = {"token": "a", "valid": True, **{k: 1.0 for k in b.METRICS}}
    assert b.summarize([row], ["a"])["pdms"] == 100
    for rows, expected in (([row], ["a", "b"]), ([row, row], ["a"]),
                           ([{**row, "valid": False}], ["a"]),
                           ([{**row, "score": float("nan")}], ["a"]),
                           ([{**row, "token": "train"}], ["a"])):
        result = b.summarize(rows, expected)
        assert not result["complete"]
        assert result["pdms"] is None


def test_requested_shutdown_finishes_sample_and_restores_signal_handlers():
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    with b.sample_boundary_stop() as stop:
        assert stop['value'] is False
        signal.raise_signal(signal.SIGTERM)
        assert stop['value'] is True
    assert signal.getsignal(signal.SIGTERM) == previous
    with pytest.raises(ValueError):
        with b.sample_boundary_stop():
            raise ValueError('sample failed')
    assert signal.getsignal(signal.SIGTERM) == previous


def test_full_pool_sampling_not_a_fixed_eight_sample_cycle():
    real = [{"kind": "original_navtrain", "sample_id": str(i)} for i in range(100)]
    synth = [{"kind": "synthetic_navtrain", "sample_id": "s" + str(i)} for i in range(3)]
    control = b.sample_indices(real, 100, 42)
    treatment = b.sample_indices(real + synth, 128, 42)
    assert set(control) == set(range(100))
    assert set(range(100, 103)) <= set(treatment)
    assert treatment[:64] == b.sample_indices(real + synth, 64, 42)
    assert [i for i in treatment[:112] if i < 100] == control[:98]
    assert sum(i >= 100 for i in treatment) == 16


@pytest.fixture
def benchmark_manifest(tmp_path, monkeypatch):
    specs = {}
    indices = {}
    for split in ("navtrain", "navtest"):
        row = {"sample_id": split, "token": split, "source_log": split + "-log", "kind": "original_" + split}
        source = tmp_path / (split + ".yaml")
        source.write_text(split)
        specs[split] = (source, {"tokens": [split], "log_names": [split + "-log"]})
        index = tmp_path / (split + "-index.json")
        index.write_text(json.dumps(dict(split=split, complete=True, source_stats={},
                         official_filter_sha256=digest(source), expected_tokens=[split], records=[row])))
        indices[split] = str(index)
    monkeypatch.setattr(b, "official_filter", lambda split: specs[split])
    result = b.make_manifest(indices["navtrain"], indices["navtest"], tmp_path / "manifest.json")
    return result


def test_cannot_replace_navtest_with_navtrain(benchmark_manifest):
    assert b.validate_manifest(benchmark_manifest)
    benchmark_manifest["splits"]["development"] = benchmark_manifest["splits"]["train"]
    with pytest.raises(ValueError, match="full navtest"):
        b.validate_manifest(benchmark_manifest)


def test_cannot_shrink_real_training_pool(benchmark_manifest):
    benchmark_manifest["splits"]["train"] = []
    with pytest.raises(ValueError, match="Full navtrain"):
        b.validate_manifest(benchmark_manifest)


def test_unsealed_official_index_rejected(benchmark_manifest):
    benchmark_manifest["files"] = {}
    with pytest.raises(ValueError, match="Unsealed"):
        b.validate_manifest(benchmark_manifest)


def test_fake_synthetic_admission_cannot_enter_full_pool(benchmark_manifest):
    benchmark_manifest["splits"]["train"].append(dict(kind="original_navtest", sample_id="fake"))
    with pytest.raises(ValueError, match="unique admitted navtrain"):
        b.validate_manifest(benchmark_manifest)


def test_indices_with_changed_official_population_rejected(benchmark_manifest):
    path = Path(benchmark_manifest["benchmark"]["test_index"])
    data = json.loads(path.read_text())
    data["expected_tokens"].append("another")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete or changed"):
        b.read_index(path, "navtest")
