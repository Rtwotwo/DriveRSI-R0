"""Resident encoding preserves profiles and fails closed on unknown native jobs."""

import json
import os
import signal
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest, signature
from driversi.data.catalog import feature_session as sessions
from driversi.data.catalog import feature_worker as worker


def config_at(tmp_path):
    cfg = dict(
        kind="resident_generated_features_v1",
        database=str(tmp_path / "generated.sqlite"),
        source_catalog=str(tmp_path / "source.sqlite"),
        profile_id="profile",
        directory=str(tmp_path / "service"),
        gpu=0,
        idle_seconds=1,
        max_seconds=30,
        max_jobs=2,
        job_seconds=20,
    )
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def job_at(tmp_path, cfg, name="job"):
    args = dict(
        export_path=str(tmp_path / "export.json"),
        source_catalog=cfg["source_catalog"],
        profile_id=cfg["profile_id"],
        database=cfg["database"],
        output=str(tmp_path / name),
        sample_report=str(tmp_path / "sample.json"),
    )
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(dict(action="features", arguments=args)))
    return path


def test_session_verifies_once_and_reuses_exact_encoder(monkeypatch):
    calls = []
    expected = dict(config=dict(model_path="test"))
    monkeypatch.setattr(sessions, "snapshot", lambda c: {"weights": (1, 2, 3)})
    monkeypatch.setattr(
        sessions.features, "feature_profile", lambda c: calls.append("profile") or dict(config=c)
    )

    class Encoder:
        def __init__(self, cfg):
            calls.append("load")

        def __call__(self, images):
            calls.append("encode")
            return images, {"seconds": 1}

    monkeypatch.setattr(sessions.features, "VisionEncoder", Encoder)
    session = sessions.FeatureSession()
    assert session.profile(expected) == session.profile(expected)
    assert session.encode([1])[1]["resident_encoder_reused"] is False
    assert session.encode([2])[1]["resident_encoder_reused"] is True
    assert calls == ["profile", "load", "encode", "encode"]


def test_session_invalidates_mutation_and_profile_change(monkeypatch):
    state = {"weight": 1}
    monkeypatch.setattr(sessions, "snapshot", lambda c: dict(state))
    monkeypatch.setattr(sessions.features, "feature_profile", lambda c: dict(config=c))
    session = sessions.FeatureSession()
    expected = {"config": {"model_path": "a"}}
    session.profile(expected)
    with pytest.raises(ValueError, match="mix"):
        session.profile({"config": {"model_path": "b"}})
    state["weight"] = 2
    with pytest.raises(ValueError, match="changed"):
        session.profile(expected)


def test_snapshot_detects_same_length_rewrite_restored_mtime(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weights.safetensors"
    weight.write_bytes(b"aaaa")
    for name in ("features.py", "vectors.py", "observations.py"):
        (tmp_path / name).write_text("# fake")
    monkeypatch.setattr(sessions.features, "__file__", str(tmp_path / "features.py"))
    before = sessions.snapshot(dict(model_path=str(model)))
    stat = weight.stat()
    # Let the filesystem timestamp clock advance before a separate write event.
    import time

    time.sleep(0.02)
    weight.write_bytes(b"bbbb")
    os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert sessions.snapshot(dict(model_path=str(model))) != before


@pytest.mark.parametrize(
    "field,value", [("gpu", 8), ("job_seconds", 1801), ("max_jobs", 0), ("idle_seconds", True)]
)
def test_worker_limits(tmp_path, field, value):
    path, cfg = config_at(tmp_path)
    cfg[field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        worker.load_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("database", "/other.sqlite"),
        ("source_catalog", "/navtest.sqlite"),
        ("profile_id", "other"),
        ("sample_report", None),
    ],
)
def test_worker_rejects_mixed_inputs(tmp_path, field, value):
    path, cfg = config_at(tmp_path)
    job = json.loads(job_at(tmp_path, cfg).read_text())
    job["arguments"][field] = value
    with pytest.raises(ValueError):
        worker.validate_job(job, cfg)


def test_unknown_running_job_is_not_restarted(tmp_path, monkeypatch):
    path, cfg = config_at(tmp_path)
    job = job_at(tmp_path, cfg)
    queue = Path(cfg["directory"]) / "jobs"
    queue.mkdir(parents=True)
    key = signature(dict(path=str(job.resolve()), sha256=digest(job)))
    (queue / (key + ".running.json")).write_text("{}")
    monkeypatch.setattr(
        worker.subprocess, "Popen", lambda *a, **k: pytest.fail("must not restart unknown work")
    )
    with pytest.raises(ValueError, match="unknown outcome"):
        worker.submit(job, path)


def test_real_queue_process_reuses_session_and_reports_failure(tmp_path, monkeypatch):
    import time

    from driversi.rsi.experiments import reactive_postprocess

    path, cfg = config_at(tmp_path)
    queue = Path(cfg["directory"]) / "jobs"
    queue.mkdir(parents=True)
    jobs = [job_at(tmp_path, cfg, n) for n in ["one", "two"]]
    for index, job in enumerate(jobs):
        (queue / (str(index) + ".request.json")).write_text(
            json.dumps(
                dict(
                    job=str(job),
                    job_sha256=digest(job),
                    config_sha256=digest(path),
                    deadline=time.time() + 20,
                )
            )
        )

    class Session:
        def __init__(self):
            self.calls = 0

        def validate(self):
            pass

    def execute(job, *, feature_session):
        feature_session.calls += 1
        if feature_session.calls == 2:
            raise ValueError("actual sample error")
        return dict(calls=feature_session.calls)

    monkeypatch.setattr(sessions, "FeatureSession", Session)
    monkeypatch.setattr(reactive_postprocess, "execute_job", execute)
    original = signal.getsignal(signal.SIGALRM)
    try:
        worker.serve(path)
    finally:
        signal.signal(signal.SIGALRM, original)
    first = json.loads((queue / "0.result.json").read_text())
    second = json.loads((queue / "1.result.json").read_text())
    assert first["ok"] and first["result"]["calls"] == 1
    assert second["ok"] is False and "actual sample error" in second["error"]
    assert json.loads((Path(cfg["directory"]) / "status.json").read_text())["completed"] == 2


def test_result_cannot_cross_config(tmp_path):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(dict(ok=True, result={"x": 1}, config_sha256="a")))
    with pytest.raises(ValueError, match="configuration"):
        worker.read_result(p, "b")


def test_completed_rpc_is_not_a_validation_bypass(tmp_path):
    path, cfg = config_at(tmp_path)
    job = job_at(tmp_path, cfg)
    queue = Path(cfg["directory"]) / "jobs"
    queue.mkdir(parents=True)
    key = signature(dict(path=str(job.resolve()), sha256=digest(job)))
    (queue / (key + ".result.json")).write_text("{}")
    with pytest.raises(ValueError, match="native exit receipt"):
        worker.submit(job, path)


def test_unknown_job_not_owned_by_new_service(tmp_path, monkeypatch):
    path, cfg = config_at(tmp_path)
    job = job_at(tmp_path, cfg)
    queue = Path(cfg["directory"]) / "jobs"
    queue.mkdir(parents=True)
    key = signature(dict(path=str(job.resolve()), sha256=digest(job)))
    (queue / (key + ".running.json")).write_text("{}")
    monkeypatch.setattr(worker, "is_running", lambda d: True)
    with pytest.raises(ValueError, match="unknown outcome"):
        worker.submit(job, path)


def test_external_watchdog_stops_unresponsive_process(tmp_path):
    import subprocess
    import sys
    import time

    path, cfg = config_at(tmp_path)
    directory = Path(cfg["directory"])
    directory.mkdir()
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert p.stdout.readline().strip() == "ready"
        birth = worker.process_birth(p.pid)
        started = time.monotonic()
        worker.watchdog(path, p.pid, birth, time.monotonic() - 1)
        p.wait(timeout=3)
        assert p.returncode == -signal.SIGKILL
        assert time.monotonic() - started < 4
        receipt = json.loads((directory / "watchdog-termination.json").read_text())
        assert receipt["reason"] == "service_deadline"
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def test_submit_bootstraps_real_worker_and_surfaces_native_error(tmp_path):
    path, cfg = config_at(tmp_path)
    cfg["max_jobs"] = 1
    path.write_text(json.dumps(cfg))
    job = job_at(tmp_path, cfg)
    with pytest.raises(ValueError, match="Resident features failed"):
        worker.submit(job, path)
    directory = Path(cfg["directory"])
    launch = json.loads((directory / "launch.json").read_text())
    assert launch["pid"] > 0
    results = list((directory / "jobs").glob("*.result.json"))
    assert len(results) == 1 and json.loads(results[0].read_text())["ok"] is False


def test_member_map_reuses_full_check_and_refreshes_commits_and_split(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace

    from driversi.rsi.experiments import navval_retrieval

    path = tmp_path / "catalog.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE members(token TEXT, log TEXT)")
    db.execute("INSERT INTO members VALUES ('a','log')")
    db.commit()
    spec = tmp_path / "split.yaml"
    spec.write_text("version1")

    class Cat:
        def __init__(self, path, split, readonly=True):
            self.path = Path(path)
            self.db = sqlite3.connect(path)

        def close(self):
            self.db.close()

    calls = []

    def build(cat, population):
        calls.append(1)
        return dict(cat.db.execute("SELECT token,log FROM members")), {str(spec): digest(spec)}

    monkeypatch.setattr(sessions, "Catalog", Cat)
    monkeypatch.setattr(navval_retrieval, "training_members", build)
    monkeypatch.setattr(navval_retrieval, "official_filter", lambda _: (spec, {}))
    session = sessions.FeatureSession()
    source = SimpleNamespace(path=path)
    assert dict(session.member_map(source)) == {"a": "log"}
    assert dict(session.member_map(source)) == {"a": "log"}
    assert len(calls) == 1
    with pytest.raises(TypeError):
        session.member_map(source)["wrong"] = "bad"
    db.execute("INSERT INTO members VALUES ('b','log')")
    db.commit()
    assert dict(session.member_map(source)) == {"a": "log", "b": "log"}
    assert len(calls) == 2
    spec.write_text("version2")
    session.member_map(source)
    assert len(calls) == 3
    session.members_catalog.close()
    db.close()


def test_member_map_split_change_during_build_does_not_cache(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace

    from driversi.rsi.experiments import navval_retrieval

    path = tmp_path / "catalog.sqlite"
    sqlite3.connect(path).close()
    spec = tmp_path / "split.yaml"
    spec.write_text("old")

    class Cat:
        def __init__(self, path, split, readonly=True):
            self.path = Path(path)
            self.db = sqlite3.connect(path)

        def close(self):
            self.db.close()

    def changing(cat, population):
        spec.write_text("new")
        return {"old": "log"}, {str(spec): digest(spec)}

    monkeypatch.setattr(sessions, "Catalog", Cat)
    monkeypatch.setattr(navval_retrieval, "official_filter", lambda _: (spec, {}))
    monkeypatch.setattr(navval_retrieval, "training_members", changing)
    session = sessions.FeatureSession()
    with pytest.raises(ValueError, match="split changed"):
        session.member_map(SimpleNamespace(path=path))
    assert session.members is None and session.members_catalog is None
    new = tmp_path / "replacement.sqlite"
    db = sqlite3.connect(new)
    db.execute("CREATE TABLE verified(value TEXT)")
    db.execute("INSERT INTO verified VALUES ('new')")
    db.commit()
    db.close()
    os.replace(new, path)

    def repaired(cat, population):
        token = cat.db.execute("SELECT value FROM verified").fetchone()[0]
        return {token: "log"}, {str(spec): digest(spec)}

    monkeypatch.setattr(navval_retrieval, "training_members", repaired)
    assert dict(session.member_map(SimpleNamespace(path=path))) == {"new": "log"}
    session.members_catalog.close()


def test_member_map_initial_filter_failure_closes_connection(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from driversi.data.catalog.feature_session import FeatureSession
    from driversi.data.catalog.store import Catalog
    from driversi.rsi.experiments import navval_retrieval

    path = tmp_path / "catalog.sqlite"
    Catalog(path, "navtrain").close()
    split = tmp_path / "official.yaml"
    split.write_text("valid")
    session = FeatureSession.__new__(FeatureSession)
    session.members = session.members_state = session.members_catalog = None
    session.members_files = {}
    monkeypatch.setattr(
        navval_retrieval,
        "official_filter",
        lambda _: (_ for _ in ()).throw(ValueError("invalid split")),
    )
    with pytest.raises(ValueError, match="invalid split"):
        session.member_map(SimpleNamespace(path=path))
    assert session.members_catalog is None
    path.unlink()
    Catalog(path, "navtrain").close()
    monkeypatch.setattr(navval_retrieval, "official_filter", lambda _: (split, {}))
    monkeypatch.setattr(
        navval_retrieval,
        "training_members",
        lambda *_: ({"new": "log"}, {str(split): digest(split)}),
    )
    assert session.member_map(SimpleNamespace(path=path)) == {"new": "log"}
    session.members_catalog.close()
