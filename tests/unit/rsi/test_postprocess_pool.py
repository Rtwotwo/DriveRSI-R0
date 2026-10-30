import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.experiments.postprocess_pool import PostprocessPool, read_result


def test_parallel_gpu_slots_are_distinct_bounded_and_ordered(tmp_path):
    barrier = threading.Barrier(2)
    seen = []

    def worker(job):
        seen.append(job["gpu"])
        barrier.wait(timeout=5)
        return job["value"]

    def job(i):
        return dict(directory=str(tmp_path / str(i)), value=i)

    with PostprocessPool(tmp_path, [1, 2], factory=ThreadPoolExecutor, worker=worker) as pool:
        pool.submit("a", job(1))
        pool.submit("b", job(2))
        with pytest.raises(ValueError, match="capacity"):
            pool.submit("c", job(3))
        assert pool.result("a") == 1
        assert pool.result("b") == 2
        assert sorted(seen) == [1, 2]
        pool.submit("a", job(1))
        assert len(pool.jobs) == 2
        with pytest.raises(ValueError, match="changed"):
            pool.submit("a", job(9))
    saved = json.loads((tmp_path / "postprocess-status.json").read_text())
    assert saved["status"] == "settled" and saved["consumed"] == 2


@pytest.mark.parametrize("gpus", [[0], [6], [1, 1], [True], [1, 2, 3, 4, 5], (1, 2)])
def test_gpu_contract(gpus, tmp_path):
    with pytest.raises(ValueError):
        PostprocessPool(tmp_path, gpus)


def test_completed_result_checks_inputs_and_output_bytes(tmp_path):
    job = {"x": 1}
    (tmp_path / "job.json").write_text(json.dumps(job))
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("native")
    result = {
        "job_sha256": digest(tmp_path / "job.json"),
        "outcome": {"status": "ok"},
        "files": {str(evidence): digest(evidence)},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "result-seal.json").write_text(
        json.dumps({"sha256": digest(tmp_path / "result.json")})
    )
    assert read_result(tmp_path, job) == {"status": "ok"}
    with pytest.raises(ValueError):
        read_result(tmp_path, {"x": 2})
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        read_result(tmp_path, job)


def test_quarantined_process_never_publishes_canonical_sample(tmp_path, monkeypatch):
    from driversi.agents.backends import native
    from driversi.rendering.diagnostics import numeric_admission
    from driversi.rsi.experiments import chainflow_campaign, reactive_postprocess

    root = tmp_path / "exp" / "benchmark"
    root.mkdir(parents=True)
    monkeypatch.setattr(native, "project_root", lambda: tmp_path)

    def save(p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    execution = root / "execution"
    execution.mkdir()
    context = execution / "teacher-counterfactual-context.pkl"
    context.write_bytes(b"fixture")
    branch = save(
        root / "branch.json", dict(execution=str(execution), files={str(context): digest(context)})
    )
    generation = save(
        root / "generation.json",
        dict(branch_report=str(branch), render_payload=str(root / "pixels.json")),
    )
    config = save(
        root / "config.json",
        dict(
            production_gpu=0,
            output=str(root),
            generated_catalog=str(root / "features.sqlite"),
            visual_quality_profile="experimental_numeric27_ade05_v1",
            maps_root=str(root / "maps"),
            chainflow_python="unused",
            feature_python="unused",
            train_catalog="unused",
            profile="fixture",
        ),
    )
    save(
        root / "quarantine.json",
        dict(
            kind="isolated_postprocess_benchmark",
            generated_catalog=str(root / "features.sqlite"),
            production_training=False,
        ),
    )
    out = root / "out"

    def admission(_g, _d, output, **_kw):
        save(output / "admission.json", {})
        return dict(admitted=True, files={}, trajectory_window="fixture")

    monkeypatch.setattr(numeric_admission, "admit", admission)
    seen = []

    def run(action, args, directory, **kw):
        seen.append((action, kw["gpu"]))
        if action == "chainflow_export":
            assert args["plan"]["canonical_dataset"] is False
            save(out / "native/dataset.json", {})
            save(out / "native/native/export.json", {})
        else:
            save(out / "features/report.json", dict(trainable=True, parent_vectors_copied=False))
            save(directory / "result.json", {})

    monkeypatch.setattr(reactive_postprocess, "run_native", run)
    result = chainflow_campaign.process(generation, config, out, execution_gpu=2, quarantine=True)
    assert result["status"] == "benchmark_sample_verified" and not result["trainable"]
    assert result["dataset_reference"] is None and seen == [
        ("chainflow_export", 2),
        ("features", 2),
    ]
    assert json.loads((out / "inputs.json").read_text())["quarantine"] is True

    wrong = json.loads(config.read_text())
    wrong["generated_catalog"] = str(tmp_path / "exp" / "production" / "features.sqlite")
    save(config, wrong)
    with pytest.raises(ValueError, match="dedicated quarantined"):
        chainflow_campaign.process(generation, config, out / "other", quarantine=True)


def test_batch_rejects_overlapping_stage_directories_before_dispatch(tmp_path, monkeypatch):
    from driversi.agents.backends import native
    from driversi.rsi.experiments.postprocess_pool import run_batch

    monkeypatch.setattr(native, "project_root", lambda: tmp_path)
    candidate = tmp_path / "exp" / "candidate"
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version=1,
                output=str(tmp_path / "exp" / "batch"),
                gpus=[1],
                jobs=[
                    dict(
                        generation=dict(output=str(candidate / "generation")),
                        admission=str(candidate / "generation"),
                        directory=str(candidate / "receipt"),
                    )
                ],
            )
        )
    )
    with pytest.raises(ValueError, match="must be distinct"):
        run_batch(manifest)
    assert not candidate.exists()


def test_busy_worker_and_native_child_exit_when_supervisor_dies(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import textwrap
    import time

    script = tmp_path / "supervisor.py"
    script.write_text(
        textwrap.dedent("""
        import json,os,sys,time
        from pathlib import Path
        from driversi.rsi.experiments import postprocess_pool as pp,reactive_postprocess as rp
        root=Path(sys.argv[1])
        def task(job):
            original=rp.subprocess.Popen
            def start(*args,**kwargs):
                return original([sys.executable,'-c','import time; time.sleep(120)'],**kwargs)
            rp.subprocess.Popen=start
            def execute(job):
                (root/'worker.json').write_text(json.dumps(dict(pid=os.getpid())))
                rp.run_native('test_parent_lifetime',{},root/'native',python=sys.executable)
            pp._branch_job=execute
            return pp.branch_job(job)
        if __name__=='__main__':
            with pp.PostprocessPool(root/'pool',[1],worker=task) as pool:
                pool.submit('a',dict(directory=str(root/'job')))
                pool.result('a')
        """)
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids = []

    def live(pid):
        path = __import__("pathlib").Path(f"/proc/{pid}/stat")
        try:
            return path.read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False

    try:
        deadline = time.monotonic() + 15
        launch = tmp_path / "native" / "launch.json"
        while not launch.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert launch.exists()
        pids = [
            json.loads((tmp_path / "worker.json").read_text())["pid"],
            json.loads(launch.read_text())["pid"],
        ]
        proc.kill()
        proc.wait(timeout=5)
        deadline = time.monotonic() + 10
        while any(live(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(live(pid) for pid in pids)
        assert json.loads((tmp_path / "native" / "exit.json").read_text())["returncode"] == 130
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for pid in pids:
            if live(pid):
                os.kill(pid, signal.SIGKILL)


def test_unsettled_native_job_is_never_relaunched(tmp_path):
    from driversi.rsi.experiments.postprocess_pool import UnsettledPostprocess, branch_job

    directory = tmp_path / "candidate"
    directory.mkdir()
    job = dict(directory=str(directory))
    (directory / "job.json").write_text(json.dumps(job))
    (directory / "launch.json").write_text(json.dumps(dict(pid=123)))
    with pytest.raises(UnsettledPostprocess):
        branch_job(job)
    assert not (directory / "result.json").exists()
