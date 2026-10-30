"""Native adapters must preserve policy behavior and fail closed at integration gaps."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from driversi.agents.backends import native
from driversi.contracts.native_model import (
    DecisionIdentity,
    NavsimPolicyAdapter,
    NuplanPlannerAdapter,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def job(tmp_path, monkeypatch):
    root = tmp_path
    repo = root / "agents/driverl/native"
    repo.mkdir(parents=True)
    (repo / "model.py").write_text("# original upstream\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    import hashlib
    (repo / "source-manifest.json").write_text(json.dumps({
        "schema_version": 1, "revision": revision,
        "files": {"model.py": hashlib.sha256((repo / "model.py").read_bytes()).hexdigest()},
    }))
    weights = root / "weights/test.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"weights fixture")
    config = root / "src/config/agents/driverl/native.yaml"
    config.parent.mkdir(parents=True)
    (config.parents[1] / "native-sources.lock.json").write_text("{}")
    value = dict(
        schema_version=1,
        model="driverl",
        family="closed_loop_rl",
        repository="agents/driverl/native",
        revision=revision,
        python=sys.executable,
        weights={"checkpoint": "weights/test.pt"},
        paths={},
        settings={"gpu": 0},
        capabilities={"preflight": True, "evaluate": True, "train": False},
        input_contract="native",
        output_contract="native",
    )
    config.write_text(yaml.safe_dump(value))
    code = ["print('PREFLIGHT_OK')"]

    def build(cfg, root, repo, python, weights, paths, output, operation):
        return dict(
            argv=[python, "-c", code[0]],
            env={},
            files=list(weights.values()),
            directories=[],
            protocol="fixture",
        )

    adapter = SimpleNamespace(build=build, __file__=__file__)
    original = native.importlib.import_module
    monkeypatch.setattr(
        native.importlib,
        "import_module",
        lambda n: adapter if n == native.ADAPTERS["driverl"] else original(n),
    )
    return root, config, weights, repo, code


def prepare(job, operation="preflight"):
    root, config, *_ = job
    return native.prepare(config, operation, root / "outputs/run", root=root)


def test_frozen_native_job_executes_original_command_and_records_no_qualification(job, monkeypatch):
    monkeypatch.setenv("EVAL_EXTRA_OVERRIDES", "replace_the_scorer=true")
    plan = prepare(job)
    assert plan["ready_to_attempt"]
    assert "EVAL_EXTRA_OVERRIDES" not in plan["env"]
    result = native.execute(plan, timeout_seconds=3)
    assert result["status"] == "completed"
    assert result["performance_parity"] == "not_measured" and not result["trainable"]
    assert (Path(plan["output"]) / "native.log").read_text().strip() == "PREFLIGHT_OK"
    with pytest.raises(FileExistsError):
        native.execute(plan, timeout_seconds=3)


@pytest.mark.parametrize("target", ["weights", "source", "config", "plan"])
def test_changed_evidence_rejected_before_native_launch(job, target):
    plan = prepare(job)
    root, config, weights, repo, _ = job
    if target == "weights":
        weights.write_bytes(b"changed checkpoint")
    if target == "source":
        (repo / "model.py").write_text("# changed upstream")
    if target == "config":
        config.write_text(config.read_text() + "# edited\n")
    if target == "plan":
        plan["argv"].append("changed")
    with pytest.raises(ValueError, match="changed|modified"):
        native.execute(plan, timeout_seconds=3)
    assert not (root / "outputs/run").exists()


def test_missing_weight_cannot_be_treated_as_random_initialization(job):
    job[2].unlink()
    plan = prepare(job)
    assert not plan["ready_to_attempt"]
    with pytest.raises(ValueError, match="Missing"):
        native.execute(plan, timeout_seconds=3)


def test_weight_outside_central_store_and_unsupported_training_rejected(job):
    with pytest.raises(ValueError, match="not released"):
        prepare(job, "train")
    value = yaml.safe_load(job[1].read_text())
    value["weights"]["checkpoint"] = "clones/example/model.py"
    job[1].write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="under weights"):
        prepare(job)


def test_zero_exit_without_preflight_marker_is_not_passed(job):
    job[4][0] = "print('loaded nothing')"
    result = native.execute(prepare(job), timeout_seconds=3)
    assert result["status"] == "failed"
    assert result["reason"] == "missing_official_preflight_marker"


def test_timeout_stops_native_process(job):
    job[4][0] = "import time; time.sleep(60)"
    result = native.execute(prepare(job), timeout_seconds=0.1)
    assert result["status"] == "timeout" and result["wall_seconds"] < 5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_wall_budget_rejected_before_output(job, value):
    with pytest.raises(ValueError, match="wall time"):
        native.execute(prepare(job), timeout_seconds=value)


def test_native_trajectory_exact_passthrough_and_stale_identity():
    poses = np.arange(24, dtype=np.float64).reshape(8, 3)
    output = SimpleNamespace(
        poses=poses, trajectory_sampling=SimpleNamespace(num_poses=8, interval_length=0.5)
    )
    observed = []
    policy = SimpleNamespace(compute_trajectory=lambda x: observed.append(x) or output)
    adapter = NavsimPolicyAdapter(policy, "weights-hash")
    identity = DecisionIdentity("s", "e", 1, 20, "weights-hash")
    agent_input = object()
    result = adapter.predict(agent_input, identity, expected_identity=identity)
    assert result.output is output and result.output.poses is poses
    assert observed == [agent_input] and result.likelihood is None
    with pytest.raises(ValueError, match="Stale"):
        adapter.predict(agent_input, identity, expected_identity=replace(identity, generation=2))
    assert len(observed) == 1
    output.trajectory_sampling.interval_length = 0.1
    with pytest.raises(ValueError, match="clock"):
        adapter.predict(agent_input, identity, expected_identity=identity)


def test_nuplan_planner_history_and_trajectory_are_not_converted():
    output = SimpleNamespace(get_state_at_time=lambda t: t)
    seen = []
    planner = SimpleNamespace(
        initialize=lambda x: seen.append(x),
        compute_planner_trajectory=lambda x: seen.append(x) or output,
    )
    adapter = NuplanPlannerAdapter(planner, "v1")
    init, history = object(), object()
    identity = DecisionIdentity("s", "e", 0, 0, "v1")
    adapter.initialize(init)
    result = adapter.predict(history, identity, expected_identity=identity)
    assert seen == [init, history] and result.output is output


def test_public_catalog_is_native_and_implicit_ppo_is_retired(capsys):
    from driversi.cli import main

    assert main(["agents"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {row["model"] for row in data["agents"]} == set(native.ADAPTERS)
    with pytest.raises(SystemExit):
        main(["train"])


def test_official_launchers_keep_model_parameters_and_release_capability():
    import importlib

    for cfg in native.catalog():
        adapter = importlib.import_module(native.ADAPTERS[cfg["model"]])
        paths = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
        weights = {k: str(ROOT / v) for k, v in cfg["weights"].items()}
        operation = "preflight" if cfg["model"] == "driverl" else "evaluate"
        plan = adapter.build(
            cfg,
            ROOT,
            ROOT / cfg["repository"],
            "/native/python",
            weights,
            paths,
            ROOT / "outputs/test",
            operation,
        )
        assert "/native/python" in plan["argv"] or "/native/python" in plan["env"].values()
        assert not cfg["capabilities"]["online_control"]
        assert cfg["capabilities"]["train"] == (cfg["model"] == "augustus")
        if cfg["model"] == "fpd_drive":
            assert plan["env"]["CUDA_VISIBLE_DEVICES"] == str(cfg["settings"]["gpu"])
            assert "--denoise-steps" not in plan["argv"]
            assert "EVAL_EXTRA_OVERRIDES" not in plan["env"]


def test_harness_cannot_dispatch_evaluation_as_zero_sim_cost(tmp_path):
    from driversi.rsi.plugins.native import run_preflight

    p = tmp_path / "report.json"
    p.write_text(json.dumps({"operation": "evaluate"}))
    with pytest.raises(ValueError, match="zero simulator cost"):
        run_preflight(
            {"max_wall_seconds": 2},
            {"inventory": {"artifacts": {"report": str(p)}}},
            tmp_path / "stage",
            tmp_path,
        )


def test_runtime_change_invalidates_prepared_model(job, monkeypatch):
    plan = prepare(job)
    monkeypatch.setattr(native, "runtime_fingerprint", lambda *a: {"changed_runtime": True})
    with pytest.raises(ValueError, match="environment changed"):
        native.verify(plan)


def test_zero_exit_evaluation_is_not_benchmark_admission(job):
    result = native.execute(prepare(job, "evaluate"), timeout_seconds=3)
    assert result["status"] == "completed"
    assert result["evaluation_status"] == "not_validated"
    assert result["scope"] == "native_process_execution_only"


def test_harness_inventory_creates_and_seals_stage(job, monkeypatch):
    from driversi.rsi.plugins import native as harness_native

    plan = prepare(job)
    monkeypatch.setattr(harness_native, "prepare", lambda *a: plan)
    output = job[0] / "stage"
    result = harness_native.run_inventory({"plan": plan}, {}, output, job[0])
    assert result["status"] == "completed"
    assert json.loads((output / "report.json").read_text()) == plan


def test_harness_timeout_is_known_budget_failure(job, monkeypatch):
    from driversi.rsi.plugins import native as harness_native

    plan = prepare(job)
    report = job[0] / "inventory.json"
    report.write_text(json.dumps(plan))
    monkeypatch.setattr(harness_native, "prepare", lambda *a: plan)
    monkeypatch.setattr(harness_native, "execute", lambda *a, **k: {"status": "timeout"})
    result = harness_native.run_preflight(
        {"max_wall_seconds": 1},
        {"inventory": {"artifacts": {"report": str(report)}}},
        job[0] / "stage",
        job[0],
    )
    assert (
        result["status"] == "budget_exhausted"
        and result["reason"] == "native_preflight_wall_timeout"
    )


def test_fpd_benchmarks_keep_distinct_official_filters():
    from driversi_agents.fpd_drive.adapter import build

    cfg = yaml.safe_load((ROOT / "src/config/agents/fpd_drive/native.yaml").read_text())
    for benchmark, ending in [("v1", "navtest.yaml"), ("v2", "navhard_two_stage.yaml")]:
        cfg["settings"]["benchmark"] = benchmark
        plan = build(
            cfg,
            ROOT,
            ROOT / cfg["repository"],
            "/python",
            {k: str(ROOT / v) for k, v in cfg["weights"].items()},
            {k: str(ROOT / v) for k, v in cfg["paths"].items()},
            ROOT / "exp/run",
            "evaluate",
        )
        assert plan["env"]["SCENE_FILTER_PATH"].endswith(ending)
        assert Path(plan["env"]["SCENE_FILTER_PATH"]).is_file()


def test_native_training_requires_source_isolation(tmp_path):
    from driversi.training.backends.native_backend import validate_dataset

    root = tmp_path / "data"
    (root / "navsim_logs/trainval").mkdir(parents=True)
    (root / "navsim_logs/trainval/log-a.pkl").touch()
    train, test = tmp_path / "train.yaml", tmp_path / "test.yaml"
    train.write_text("log_names: [log-a]\n")
    test.write_text("log_names: [log-b]\n")
    manifest = tmp_path / "manifest.json"
    value = dict(
        schema_version=1,
        dataset_version="v1",
        dataset_kind="original_navsim",
        data_root=str(root),
        train_filter=str(train),
    )
    manifest.write_text(json.dumps(value))
    assert validate_dataset(manifest, root, test) == train
    test.write_text("log_names: [log-a]\n")
    with pytest.raises(ValueError, match="overlap"):
        validate_dataset(manifest, root, test)
    value["dataset_kind"] = "synthetic"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="admission"):
        validate_dataset(manifest, root, test)




def test_native_cancel_reaps_model_and_grandchild(job):
    import os
    import signal
    import time

    pid_file = job[0] / "pids.json"
    job[4][0] = (
        "import subprocess,sys,os,json,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True); "
        f"open({str(pid_file.with_suffix('.tmp'))!r},'w').write(json.dumps([os.getpid(),p.pid])); "
        f"os.replace({str(pid_file.with_suffix('.tmp'))!r},{str(pid_file)!r}); time.sleep(60)"
    )
    plan = prepare(job)
    # Fork inherits the fixture adapter so verification is identical to the real launcher.
    pid = os.fork()
    if pid == 0:
        try:
            native.execute(plan, timeout_seconds=30)
        finally:
            os._exit(0)
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_file.exists()
    children = json.loads(pid_file.read_text())
    os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)
    result = json.loads((Path(plan["output"]) / "report.json").read_text())
    assert result["status"] == "cancelled"
    assert all(not Path(f"/proc/{child}").exists() for child in children)


@pytest.mark.parametrize("own_group", [True, False])
def test_native_timeout_reaps_detached_rank_workers(job, own_group):
    pid_file = job[0] / "rank.json"
    job[4][0] = (
        "import subprocess,sys,os,json,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True); "
        f"open({str(pid_file)!r},'w').write(json.dumps([p.pid]));time.sleep(60)"
    )
    result = native.execute(prepare(job), timeout_seconds=0.3, own_process_group=own_group)
    assert result["status"] == "timeout"
    assert all(not Path(f"/proc/{pid}").exists() for pid in json.loads(pid_file.read_text()))


def test_training_fingerprint_rejects_same_path_log_or_image_replacement(tmp_path):
    import pickle

    from driversi.training.backends.native_dataset import fingerprint_inputs

    logs = tmp_path / "navsim_logs/trainval"
    logs.mkdir(parents=True)
    sensor = tmp_path / "sensor_blobs/trainval/log-a/CAM_F0/frame.jpg"
    sensor.parent.mkdir(parents=True)
    sensor.write_bytes(b"first image")
    log = logs / "log-a.pkl"
    log.write_bytes(pickle.dumps([{"image": "log-a/CAM_F0/frame.jpg", "state": 1}]))
    first = fingerprint_inputs(tmp_path, ["log-a"])
    assert first["ready"]
    log.write_bytes(pickle.dumps([{"image": "log-a/CAM_F0/frame.jpg", "state": 2}]))
    second = fingerprint_inputs(tmp_path, ["log-a"])
    assert first["sha256"] != second["sha256"]
    sensor.write_bytes(b"different image")
    third = fingerprint_inputs(tmp_path, ["log-a"])
    assert second["sha256"] != third["sha256"]
    sensor.unlink()
    assert not fingerprint_inputs(tmp_path, ["log-a"])["ready"]


@pytest.mark.parametrize("worker_died", [False, True])
def test_outer_harness_stops_detached_ranks_after_worker_failure(tmp_path, worker_died):
    import os
    import signal
    import time

    from driversi.agents.backends.native_process import OwnedTree
    from driversi.rsi.workflow.engine import _stop_owned
    from driversi.runtime.processes import subreaper

    previous = subreaper(True)
    pids = tmp_path / "ranks.json"
    tree = OwnedTree()
    child = None
    try:
        code = (
            "import subprocess,sys,os,json,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True); "
            f"open({str(pids.with_suffix('.tmp'))!r},'w').write(json.dumps([p.pid])); "
            f"os.replace({str(pids.with_suffix('.tmp'))!r},{str(pids)!r});time.sleep(60)"
        )
        child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        child._native_tree = tree
        child._native_receipt = tmp_path / "processes.json"
        tree.attach(child)
        deadline = time.monotonic() + 10
        while not pids.exists() and time.monotonic() < deadline:
            tree.capture()
            time.sleep(0.02)
        assert pids.exists()
        ranks = json.loads(pids.read_text())
        tree.capture()
        if worker_died:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        outcome = _stop_owned(child)
        assert outcome == {"status": "stopped"}
        assert not any(Path(f"/proc/{pid}").exists() for pid in ranks)
        assert json.loads(child._native_receipt.read_text())["status"] == "stopped"
    finally:
        if child is not None:
            tree.stop()
        subreaper(previous)


def test_cancel_during_proc_scan_is_not_swallowed_as_io_error(job, monkeypatch):
    import os
    import signal

    from driversi.agents.backends import native_process

    original = Path.read_text
    sent = False

    def interrupted_read(path, *args, **kwargs):
        nonlocal sent
        if not sent and str(path).startswith("/proc/"):
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return original(path, *args, **kwargs)

    def wait(self, timeout):
        monkeypatch.setattr(Path, "read_text", interrupted_read)
        native_process.snapshot()
        # This is reached if snapshot wrongly catches the cancellation as OSError.
        raise subprocess.TimeoutExpired(self.child.args, timeout)

    monkeypatch.setattr(native_process.OwnedTree, "wait", wait)
    result = native.execute(prepare(job), timeout_seconds=1)
    assert sent and result["status"] == "cancelled"


def test_common_deadline_exhausted_during_verification_does_not_launch(job, monkeypatch):
    import time
    plan=prepare(job)
    native.verify(plan)
    clock=[time.monotonic()]
    def verify(value):
        clock[0]+=4
    monkeypatch.setattr(native,"verify",verify)
    monkeypatch.setattr(time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(native.subprocess,"Popen",lambda *a,**k:pytest.fail("must not launch after verification budget"))
    result=native.execute(plan,timeout_seconds=3,deadline_monotonic=clock[0]+3)
    assert result["status"]=="timeout" and result["reason"]=="deadline_exhausted_before_launch"
