import pytest

from driversi.rsi.decisions.paired_evidence import assess


def row(i, group=None, gain=0.05):
    return {
        "sample_id": str(i),
        "source_log": group or "log-" + str(i),
        "control": {"score": 0.8, "safe": 1.0},
        "treatment": {"score": 0.8 + gain, "safe": 1.0},
    }


def evidence(rows):
    return assess(rows, primary_metric="score", safety_metrics=["safe"])


def test_many_correlated_samples_do_not_become_independent_trials():
    result = evidence([row(i, "same-log") for i in range(1000)])
    assert result["original_logs"] == 1
    assert result["status"] == "insufficient_independent_logs"
    assert result["log_bootstrap_interval"] is None
    assert not result["promotion_authorized"]


def test_consistent_gain_is_only_exploratory_replication_evidence():
    result = evidence([row(i) for i in range(8)])
    assert result["status"] == "positive_exploratory_evidence"
    assert result["ready_for_independent_replication"]
    assert not result["promotion_authorized"]


def test_single_dense_log_cannot_outweigh_all_other_logs():
    rows = [row(i, "dense", 0.1) for i in range(100)]
    rows += [row(100 + i, "separate-" + str(i), -0.08) for i in range(7)]
    result = evidence(rows)
    assert result["sample_mean_delta"] > 0
    assert result["equal_log_mean_delta"] < 0
    assert result["status"] == "gain_not_established"


def test_small_float_gain_cannot_trigger_replication():
    assert evidence([row(i, gain=1e-10) for i in range(8)])["status"] == "gain_not_established"


def test_local_safety_regression_blocks_average_gain():
    rows = [row(i) for i in range(8)]
    rows[0]["treatment"]["safe"] = 0
    assert evidence(rows)["status"] == "safety_regression"


def test_pair_order_does_not_affect_deterministic_interval():
    rows = [row(i, gain=(i - 2) / 100) for i in range(20)]
    assert evidence(rows) == evidence(list(reversed(rows)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_values_rejected(bad):
    item = row(0)
    item["treatment"]["score"] = bad
    with pytest.raises(ValueError, match="finite"):
        evidence([item])


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError, match="Unique"):
        evidence([row(0), row(0)])
