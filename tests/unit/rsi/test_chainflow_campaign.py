import copy
import csv
import json

import pytest

from driversi.data.datasets.native_admission import validate_sample
from driversi.rsi.experiments import chainflow_campaign as campaign
from driversi.rsi.plugins.chainflow_pre_eval import verify_pair


def baseline(tmp_path, monkeypatch):
    from driversi_agents.chainflow import benchmark

    tokens = [f"token-{i}" for i in range(12146)]
    records = [dict(token=t, source_log="test-log") for t in tokens]
    for n in ("checkpoint", "backbone", "index"):
        (tmp_path / n).write_text("fixture")
    index = dict(expected_tokens=tokens, records=records)
    monkeypatch.setattr(benchmark, "read_index", lambda p, s: index)
    monkeypatch.setattr(benchmark, "evaluation_identity", lambda c: dict(shard=0, model="native"))
    source = tmp_path / "source"
    source.mkdir()
    with (source / "per-scene.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "log_name", "valid", "score", "error", "comfort"])
        for t in tokens:
            writer.writerow([t, "test-log", "True", 0.9, "", 1.0])
    (source / "summary.json").write_text(
        json.dumps(
            dict(
                complete=True,
                succeeded=12146,
                failed=0,
                identity=dict(model="native"),
                metrics=dict(score=0.9, comfort=1.0),
                pdms=90.0,
            )
        )
    )
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            dict(
                output=str(tmp_path / "out"),
                checkpoint=str(tmp_path / "checkpoint"),
                backbone=str(tmp_path / "backbone"),
                test_index=str(tmp_path / "index"),
            )
        )
    )
    return source, cfg


def test_full_normalization_and_reuse(tmp_path, monkeypatch):
    src, cfg = baseline(tmp_path, monkeypatch)
    a = campaign.import_baseline(src, cfg)
    b = campaign.import_baseline(src, cfg)
    assert a == b and a["completed"] == 12146 and a["pdms"] == pytest.approx(90.0)
    assert a["below_095"] == 12146 and not a["dit"]


@pytest.mark.parametrize("replacement", ["True,90.0,", "False,0.9,"])
def test_reject_non_normalized_or_invalid_score(tmp_path, monkeypatch, replacement):
    src, cfg = baseline(tmp_path, monkeypatch)
    p = src / "per-scene.csv"
    p.write_text(p.read_text().replace("True,0.9,", replacement, 1))
    with pytest.raises(ValueError, match="normalized"):
        campaign.import_baseline(src, cfg)


def test_duplicate_token_is_not_full_coverage(tmp_path, monkeypatch):
    src, cfg = baseline(tmp_path, monkeypatch)
    p = src / "per-scene.csv"
    p.write_text(p.read_text().replace("token-1,", "token-0,", 1))
    with pytest.raises(ValueError, match="coverage"):
        campaign.import_baseline(src, cfg)


def receipts():
    student = dict(
        model="chainflow_stage1",
        status="completed",
        total_updates=2,
        context="ctx",
        resume_from="checkpoint",
        state_fingerprint="state",
        worker_pid=1,
        sample_trace=[
            dict(optimizer_step=1, sample_ids=["original"], learning_rates_before=[0.01]),
            dict(optimizer_step=2, sample_ids=["synthetic"], learning_rates_before=[0.01]),
        ],
    )
    return copy.deepcopy(student), student, dict(student, worker_pid=2)


def test_gate_requires_replay_and_real_positive_lr_exposure():
    a, b, c = receipts()
    assert verify_pair(a, b, c, ["synthetic"])["status"] == "passed"
    c["state_fingerprint"] = "different"
    with pytest.raises(ValueError, match="recovery"):
        verify_pair(a, b, c, ["synthetic"])
    a, b, c = receipts()
    b["sample_trace"][1]["learning_rates_before"] = [0.0]
    with pytest.raises(ValueError, match="positive-LR"):
        verify_pair(a, b, c, ["synthetic"])


def test_chainflow_sample_does_not_require_augustus_language(tmp_path, monkeypatch):
    from driversi_agents.chainflow import synthetic

    called = []
    monkeypatch.setattr(synthetic, "validate_admission", lambda row, files: called.append(row))
    value = dict(
        kind="chainflow_admitted_synthetic_sample",
        trainable=True,
        row=dict(sample_id="new", source_log="train"),
        files={},
    )
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(value))
    assert validate_sample(p)["sample_id"] == "new" and called == [value["row"]]
    value["trainable"] = False
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_sample(p)
