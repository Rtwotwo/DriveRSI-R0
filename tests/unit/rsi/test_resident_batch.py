import json
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.experiments import resident_batch as batch


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "project_root", lambda: tmp_path)
    runtime = tmp_path / "python"
    runtime.write_text("not executed")
    bundle = tmp_path / "asset"
    bundle.mkdir()
    config = save(
        tmp_path / "config.json",
        dict(
            student_model="chainflow_stage1",
            train_catalog=str(tmp_path / "train.sqlite"),
            profile="native-profile",
            output="historical",
            generated_catalog="historical.sqlite",
            chainflow_python=str(runtime),
            feature_python=str(runtime),
            runtime=dict(gs_python=str(runtime), cuda_home="/usr/local/cuda"),
        ),
    )
    feature = save(
        tmp_path / "feature.json",
        dict(
            kind="resident_generated_features_v1",
            database=str(tmp_path / "old.sqlite"),
            source_catalog=str(tmp_path / "train.sqlite"),
            profile_id="native-profile",
            directory=str(tmp_path / "old-service"),
            gpu=7,
            idle_seconds=600,
            max_seconds=7200,
            max_jobs=8,
            job_seconds=1800,
        ),
    )
    jobs = []
    for i in range(4):
        directory = tmp_path / "prepared" / str(i)
        gen = save(directory / "report.json", dict(teacher=dict(status="trajectory_admitted")))
        save(
            directory / "generation.json",
            dict(asset=dict(bundle_root=str(bundle)), config=dict(input=dict(kind="assignment"))),
        )
        div = save(directory / "diversity.json", dict(decision=dict(distinct=True)))
        jobs.append(
            dict(
                generation=str(gen),
                sha256=digest(gen),
                diversity=str(div),
                diversity_sha256=digest(div),
            )
        )
    return save(
        tmp_path / "batch.json",
        dict(
            schema_version=1,
            output=str(tmp_path / "exp" / "isolated"),
            config=str(config),
            feature_worker_config=str(feature),
            gpus=[1, 2],
            feature_gpu=3,
            mode="resident",
            max_seconds=120,
            jobs=jobs,
        ),
    )


def test_preflight_is_read_only_preserves_sources_and_partitions_work(manifest):
    spec = json.loads(manifest.read_text())
    originals = {p: Path(p).read_bytes() for p in (spec["config"], spec["feature_worker_config"])}
    plan = batch.prepare(manifest)
    assert [len(lane["jobs"]) for lane in plan["lanes"]] == [2, 2]
    assert plan["lanes"][0]["jobs"][1] == spec["jobs"][2]
    assert plan["feature"]["database"] == str(Path(spec["output"]) / "features.sqlite")
    assert plan["feature"]["gpu"] == 3 and plan["production_increment"] == 0
    assert not Path(spec["output"]).exists()
    assert all(Path(p).read_bytes() == raw for p, raw in originals.items())


def test_training_only_preflight_needs_no_feature_service(manifest,monkeypatch):
    from driversi.data.datasets import postprocessing
    monkeypatch.setattr(postprocessing,'training_only',lambda _:True)
    spec=json.loads(manifest.read_text());spec['purpose']='production'
    cfg=json.loads(Path(spec['config']).read_text())
    cfg.update(postprocessing_mode='training_only',visual_quality_profile='experimental_numeric25_spatial05_v1',
               strict_source_psnr_gt=25.0)
    for k in ['feature_python','train_catalog','profile','generated_catalog']:cfg.pop(k)
    save(Path(spec['config']),cfg)
    for k in ['feature_gpu','feature_worker_config']:spec.pop(k)
    reservation=save(manifest.parent/'reservation.json',dict(config_sha256=digest(spec['config']),jobs=spec['jobs']))
    spec.update(reservation=str(reservation),reservation_sha256=digest(reservation))
    save(manifest,spec)
    plan=batch.prepare(manifest)
    assert plan['feature'] is None and plan['feature_template_sha256'] is None
    assert len(plan['lanes'])==2


@pytest.mark.parametrize(
    "change",
    [
        dict(gpus=[1, 1]),
        dict(gpus=[True]),
        dict(feature_gpu=1),
        dict(max_seconds=0),
        dict(mode="production"),
        dict(gpus=[0, 1, 2, 3, 4]),
    ],
)
def test_preflight_rejects_unsafe_resources_before_creating_outputs(manifest, change):
    spec = json.loads(manifest.read_text())
    save(manifest, dict(spec, **change))
    with pytest.raises(ValueError):
        batch.prepare(manifest)
    assert not Path(spec["output"]).exists()


def test_changed_or_duplicate_jobs_cannot_be_dispatched(manifest):
    spec = json.loads(manifest.read_text())
    Path(spec["jobs"][0]["generation"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        batch.prepare(manifest)


def test_asset_mixing_rejected(manifest):
    spec = json.loads(manifest.read_text())
    other = manifest.parent / "other-asset"
    other.mkdir()
    p = Path(spec["jobs"][0]["generation"]).parent / "generation.json"
    value = json.loads(p.read_text())
    value["asset"]["bundle_root"] = str(other)
    save(p, value)
    with pytest.raises(ValueError, match="one GS asset"):
        batch.prepare(manifest)


def test_default_cli_does_not_launch_execution(manifest, monkeypatch, capsys):
    monkeypatch.setattr(batch.sys, "argv", ["batch", "--manifest", str(manifest)])
    monkeypatch.setattr(batch, "run", lambda *a: pytest.fail("must not execute"))
    batch.main()
    assert json.loads(capsys.readouterr().out)["execution_started"] is False


@pytest.mark.parametrize("fail", [False, True])
def test_supervisor_isolates_outputs_and_settles_failed_lanes(manifest, monkeypatch, fail):
    from driversi.rendering.gaussian import image_store
    from driversi.runtime import experiment_guard

    spec = json.loads(manifest.read_text())
    out = Path(spec["output"])
    commands, stops, settled = [], [], []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(commands)
            self.returncode = 1 if fail and "--spec" in command else 0
            commands.append(command)
            if "--spec" in command:
                lane = json.loads(Path(command[command.index("--spec") + 1]).read_text())
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(lane["gpu"])
                save(
                    Path(lane["output"]) / "report.json",
                    dict(
                        rows=[
                            dict(outcome=dict(status="benchmark_sample_verified"))
                            for _ in lane["jobs"]
                        ]
                    ),
                )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(batch.subprocess, "Popen", Process)
    monkeypatch.setattr(batch, "_stop", lambda p: stops.append(p.pid))
    monkeypatch.setattr(experiment_guard, "birth", lambda pid: "fixture")
    monkeypatch.setattr(experiment_guard, "settle", lambda root: settled.append(root) or [])
    monkeypatch.setattr(image_store, "flush_outputs", lambda root: dict(status="fixture"))
    if fail:
        with pytest.raises(RuntimeError, match="lane failed"):
            batch.run(manifest)
        assert not (out / "report.json").exists()
        assert json.loads((out / "status.json").read_text())["status"] == "failed"
    else:
        result = batch.run(manifest)
        assert result["verified"] == 4 and not result["training_started"]
        assert result["production_increment"] == 0
    assert len(commands) == 3 and len(stops) == 3
    assert settled and set(settled) == {out}
    assert json.loads((out / "quarantine.json").read_text())["production_training"] is False
    assert not (manifest.parent / "dataset").exists()


def test_prefetch_parameter_defaults_bounds_and_requires_family():
    from driversi.rsi.experiments.reactive_autoloop import physical_prefetch_workers

    assert physical_prefetch_workers({}) == 1
    assert (
        physical_prefetch_workers(
            dict(trajectory_family={"method": "test"}, physical_prefetch_workers=4)
        )
        == 4
    )
    for value in (True, 0, 5, 1.5, "4"):
        with pytest.raises(ValueError):
            physical_prefetch_workers(
                dict(trajectory_family={"method": "test"}, physical_prefetch_workers=value)
            )
    with pytest.raises(ValueError, match="trajectory family"):
        physical_prefetch_workers(dict(physical_prefetch_workers=4))


@pytest.mark.parametrize("workers", [1, 4])
def test_campaign_forwards_prefetch_to_existing_family_before_any_training(
    tmp_path, monkeypatch, workers
):
    from driversi.rsi.experiments import reactive_autoloop, reactive_cohort, reactive_selection

    config = save(
        tmp_path / "campaign.json",
        dict(
            output=str(tmp_path / "run"),
            max_wall_seconds=60,
            baseline_config="unused",
            coverage="unused",
            max_queries=1,
            max_candidates=1,
            trajectory_family={"method": "existing"},
            physical_prefetch_workers=workers,
        ),
    )
    monkeypatch.setattr(reactive_autoloop, "wait_baseline", lambda *args: None)
    monkeypatch.setattr(reactive_autoloop, "status", lambda *args, **kwargs: None)
    monkeypatch.setattr(reactive_selection, "select", lambda *args, **kwargs: {"plan": "selected"})

    class ReachedFamily(Exception):
        pass

    def produce(selection, config_path, root, deadline, status, *, prefetch_workers):
        assert selection["plan"] == "selected" and prefetch_workers == workers
        raise ReachedFamily()

    monkeypatch.setattr(reactive_cohort, "produce", produce)
    with pytest.raises(ReachedFamily):
        reactive_autoloop.production(config)


@pytest.mark.parametrize("signal_race", ["term", "kill"])
def test_stop_tolerates_child_exit_between_poll_and_signal(monkeypatch, signal_race):
    class Child:
        pid = 123
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout):
            self.waits += 1
            if signal_race == "kill" and self.waits == 1:
                raise batch.subprocess.TimeoutExpired("child", timeout)
            return 0

    def signal(pid, sig):
        if signal_race == "term" or sig == batch.signal.SIGKILL:
            raise ProcessLookupError()

    monkeypatch.setattr(batch.os, "killpg", signal)
    child = Child()
    batch._stop(child)
    assert child.waits == (2 if signal_race == "kill" else 1)


def test_cleanup_attempts_every_lane_and_retains_guard_on_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    children = [SimpleNamespace(pid=1), SimpleNamespace(pid=2)]
    guard = SimpleNamespace(pid=3)
    stopped, settled = [], []

    def stop(process):
        stopped.append(process.pid)
        if process.pid == 1:
            raise batch.subprocess.TimeoutExpired("child", 10)

    monkeypatch.setattr(batch, "_stop", stop)
    save(tmp_path / "report.json", dict(status="completed"))
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        batch._cleanup(
            [(i, p, []) for i, p in enumerate(children)],
            guard,
            tmp_path,
            lambda root: settled.append(root) or [],
        )
    assert stopped == [1, 2] and settled == [tmp_path]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "cleanup_failed"
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "cleanup_failed"


def production_spec(manifest):
    spec=json.loads(manifest.read_text())
    cfg=json.loads(Path(spec['config']).read_text())
    feature=json.loads(Path(spec['feature_worker_config']).read_text())
    cfg.update(visual_quality_profile='experimental_numeric27_strict_ade05_v1',
               strict_source_psnr_gt=27., generated_catalog=feature['database'])
    save(Path(spec['config']),cfg)
    spec.update(purpose='production',feature_gpu=feature['gpu'])
    reservation=save(manifest.parent/'reservation.json',dict(config_sha256=digest(spec['config']),jobs=spec['jobs']))
    spec.update(reservation=str(reservation),reservation_sha256=digest(reservation))
    save(manifest,spec)
    return spec


def test_production_keeps_canonical_feature_database_and_explicit_lanes(manifest):
    spec=production_spec(manifest)
    plan=batch.prepare(manifest)
    assert plan['production']
    assert plan['config']['generated_catalog']==json.loads(Path(spec['config']).read_text())['generated_catalog']
    assert all(lane['production'] for lane in plan['lanes'])
    assert plan['feature']==json.loads(Path(spec['feature_worker_config']).read_text())
    assert not Path(spec['output']).exists()


@pytest.mark.parametrize('change',[{'strict_source_psnr_gt':25.},
    {'visual_quality_profile':'experimental_numeric25_v1'}, {'generated_catalog':'other.sqlite'}])
def test_production_cannot_relax_gate_or_change_feature_library(manifest,change):
    spec=production_spec(manifest);p=Path(spec['config']);cfg=json.loads(p.read_text());cfg.update(change);save(p,cfg)
    with pytest.raises(ValueError):batch.prepare(manifest)
    assert not Path(spec['output']).exists()


def test_production_cannot_dispatch_unreserved_job(manifest):
    spec=production_spec(manifest);spec['jobs']=spec['jobs'][:2];save(manifest,spec)
    with pytest.raises(ValueError,match='Reservation'):batch.prepare(manifest)


def test_symmetric_history_grid_preserves_envelope_and_both_sides():
    from driversi.rsi.experiments.generation_policy import symmetric_grid
    grid=symmetric_grid()
    assert len(grid)==64
    for row in grid:
        assert abs(row['longitudinal_m'])<=5 and abs(row['lateral_m'])<=2 and abs(row['speed_delta_mps'])<=2
        assert dict(row,lateral_m=-row['lateral_m']) in grid
    assert all(row['lateral_m']==0 for row in grid[:4])


def test_bounded_grid_prefix_covers_speed_signs_without_changing_candidates():
    from driversi.rsi.experiments.generation_policy import symmetric_grid, balanced_grid
    old=symmetric_grid();new=balanced_grid()
    key=lambda r:tuple(sorted(r.items()))
    assert set(map(key,old))==set(map(key,new))
    assert len(set(map(key,new)))==64
    for budget in range(10,65):
        assert {r['speed_delta_mps'] for r in new[:budget]}=={-1.,0.,1.,2.}
    assert {-1.,0.}=={r['speed_delta_mps'] for r in old[:32]}
    assert {r['lateral_m'] for r in new[:32]}=={-.5,0.,.5}
