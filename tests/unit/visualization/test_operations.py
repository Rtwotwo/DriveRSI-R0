import json
import re
import sqlite3
from pathlib import Path

import pytest

from driversi.rsi.workflow.store import public_tasks
from driversi.visualization.catalog import experiment
from driversi.visualization.operations import parse_augustus_progress
from driversi.visualization.watch import fingerprint


def test_native_progress_keeps_final_scalar_per_batch_and_drops_sample_text():
    text = """Epoch 1/100: 100%|aa| 2/2 [00:01, diff=0.1, lm=0.2, loss=0.3, lr=1e-7]
Epoch 1/100: 100%|aa| 2/2 [00:01, diff=0.4, lm=0.5, loss=0.6, lr=2e-7]
[step 2] train_data/sample_text: private sample content
"""
    rows = parse_augustus_progress(text, "train")
    assert len(rows) == 1 and rows[0]["loss"] == 0.6 and rows[0]["learning_rate"] == 2e-7
    assert "private" not in json.dumps(rows)


def test_missing_or_invalid_receipt_isolated_and_failure_status_preserved(tmp_path):
    (tmp_path / "exp").mkdir()
    row = dict(id="a", title="A", model="M", scope="pilot", receipt="exp/a.json")
    (tmp_path / row["receipt"]).write_text("incomplete JSON")
    assert experiment(row, tmp_path)["status"] == "invalid"
    (tmp_path / row["receipt"]).write_text('{"status":"failed"}')
    assert experiment(row, tmp_path)["status"] == "failed"


def test_explicit_harness_task_survives_more_than_100_new_records(tmp_path):
    dbpath = tmp_path / "registry.sqlite"

    class Registry:
        def _connect(self):
            return sqlite3.connect(dbpath)

        def protected_groups(self):
            return set()

    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE harness_tasks (id TEXT PRIMARY KEY, payload TEXT)")
        for i in range(102):
            task = dict(label="task", providers=[], schema_version=2, steps=[])
            row = dict(
                id=str(i),
                task=task,
                status="completed",
                failure_reason=None,
                sim_agent_steps=0,
                stages=[],
            )
            db.execute("INSERT INTO harness_tasks VALUES (?,?)", (str(i), json.dumps(row)))
    assert len(public_tasks(Registry())) == 100
    assert public_tasks(Registry(), task_ids=["0"])[0]["id"] == "0"
    assert public_tasks(Registry(), task_ids=[]) == []


def test_refresh_detects_new_receipt(tmp_path):
    p = tmp_path / "new.json"
    before = fingerprint([p])
    p.write_text("{}")
    assert before != fingerprint([p])


def test_new_rsi_configuration_roles_and_names_are_registered():
    root = Path(__file__).resolve().parents[3] / "src/config/rsi"
    rows = json.loads((root / "catalog.json").read_text())["presets"]
    registered = {row["id"] for row in rows}
    actual = {str(p.relative_to(root)) for p in root.rglob("*.json") if p.name != "catalog.json"}
    assert actual == registered
    for identifier in actual:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*\.json", Path(identifier).name)


def test_absent_status_is_unknown(tmp_path):
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp/a.json").write_text("{}")
    row = dict(id="a", title="A", model="M", scope="pilot", receipt="exp/a.json")
    assert experiment(row, tmp_path)["status"] == "unknown"


def test_watcher_survives_missing_config_and_keeps_previous_snapshot(tmp_path, monkeypatch, capsys):
    from driversi.visualization import watch as module

    class EndLoop(Exception):
        pass

    monkeypatch.setattr(module.time, "sleep", lambda _: (_ for _ in ()).throw(EndLoop()))
    with pytest.raises(EndLoop):
        module.watch(tmp_path / "missing.json")
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "refresh_failed" and result["previous_snapshot_retained"]
