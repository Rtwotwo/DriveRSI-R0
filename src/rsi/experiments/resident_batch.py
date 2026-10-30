"""Bounded, isolated multi-GPU postprocessing of already executed branches.

This promotes the verified experiment supervisor without importing experiment
scripts. It does not generate physical trajectories or publish training samples.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def prepare(manifest):
    """Read-only preflight and deterministic lane assignment; no CUDA imports."""
    manifest = Path(manifest).resolve(strict=True)
    spec = read_mapping(manifest)
    root = project_root().resolve()
    output = Path(spec["output"]).resolve()
    if spec.get("schema_version") != 1 or not output.is_relative_to(root / "exp"):
        raise ValueError("Versioned manifest and isolated exp output required")
    if output == root / "exp" or output.exists():
        raise ValueError("A fresh isolated output directory is required")
    gpus = spec["gpus"]
    if (
        not isinstance(gpus, list)
        or not 1 <= len(gpus) <= 4
        or any(type(g) is not int or not 0 <= g <= 7 for g in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("Select one to four distinct GPU IDs")
    seconds = spec.get("max_seconds", 2400)
    if type(seconds) is not int or not 60 <= seconds <= 7200:
        raise ValueError("Bounded 60..7200 second execution required")
    if spec["mode"] not in {"baseline", "resident"}:
        raise ValueError("Unknown postprocessing mode")
    jobs = spec["jobs"]
    if not isinstance(jobs, list) or not len(gpus) <= len(jobs) <= 64:
        raise ValueError("One to64 explicit prepared jobs, at least one per lane, required")
    production = spec.get("purpose", "benchmark") == "production"
    if spec.get("purpose", "benchmark") not in ("benchmark", "production"):
        raise ValueError("Unknown batch purpose")
    config_path = Path(spec["config"]).resolve(strict=True)
    config = read_mapping(config_path)
    from driversi.data.datasets.postprocessing import mode, training_only
    deferred = mode(config, production=production) == "training_only"
    if deferred and (not production or not training_only(root / "dataset")):
        raise ValueError("Training-only rendering requires the active production policy")
    feature_gpu = None if deferred else spec["feature_gpu"]
    if not deferred and (type(feature_gpu) is not int or not 0 <= feature_gpu <= 7 or feature_gpu in gpus):
        raise ValueError("Feature GPU must be explicit and separate from rendering")
    if config.get("student_model") != "chainflow_stage1":
        raise ValueError("This verified exporter supports native ChainFlow stage1 only")
    executables = [config["runtime"]["gs_python"], config["chainflow_python"]]
    if not deferred: executables.append(config["feature_python"])
    for executable in executables:
        if not Path(executable).is_absolute() or not Path(executable).is_file():
            raise ValueError("Native runtime executable missing")
    feature_path, feature = None, None
    if not deferred:
        from driversi.data.catalog.feature_worker import load_config
        feature_path = Path(spec["feature_worker_config"]).resolve(strict=True)
        feature = load_config(feature_path)
        if (
            Path(feature["source_catalog"]).resolve() != Path(config["train_catalog"]).resolve()
            or feature["profile_id"] != config["profile"]
        ):
            raise ValueError("Feature source/profile differs from native export configuration")
    verified = []
    bundles = set()
    for job in jobs:
        value = dict(job)
        for name in ("generation", "diversity"):
            path = Path(job[name]).resolve(strict=True)
            key = "sha256" if name == "generation" else "diversity_sha256"
            if digest(path) != job[key]:
                raise ValueError("Prepared branch or diversity evidence changed")
            value[name] = str(path)
        generation = read_mapping(value["generation"])
        if generation["teacher"]["status"] != "trajectory_admitted":
            raise ValueError("Completed admitted physical execution required")
        setup = read_mapping(Path(value["generation"]).parent / "generation.json")
        if setup["config"]["input"]["kind"] != "assignment":
            raise ValueError("Explicit sealed teacher assignment required")
        bundles.add(str(Path(setup["asset"]["bundle_root"]).resolve(strict=True)))
        if not read_mapping(value["diversity"])["decision"]["distinct"]:
            raise ValueError("Prepared diversity evidence is rejected")
        verified.append(value)
    if len({j["generation"] for j in verified}) != len(verified):
        raise ValueError("Duplicate prepared branch input")
    if len(bundles) != 1:
        raise ValueError("Group one GS asset per resident batch")
    if production:
        from driversi.rendering.diagnostics.numeric_admission import strict_problem_profile
        if (not strict_problem_profile(config.get('visual_quality_profile'),config.get('strict_source_psnr_gt'))
                or spec.get("reservation_sha256") != digest(spec["reservation"])
                or (not deferred and (feature_gpu != feature["gpu"]
                or Path(feature["database"]).resolve() != Path(config["generated_catalog"]).resolve()))):
            raise ValueError("Production requires declared strict admission and a bound reservation/feature service")
        reservation = read_mapping(spec["reservation"])
        if reservation.get("config_sha256") != digest(config_path) or reservation.get("jobs") != verified:
            raise ValueError("Reservation does not bind these prepared jobs")
    else:
        config.update(output=str(output), generated_catalog=str(output / "features.sqlite"))
    if not production: feature.update(
        database=config["generated_catalog"],
        directory=str(output / "feature-worker"),
        gpu=feature_gpu,
        max_jobs=len(jobs),
        max_seconds=seconds,
    )
    lanes = [
        dict(
            output=str(output / f"lane-{i}"),
            gpu=gpu,
            mode=spec["mode"],
            production=production,
            bundle=next(iter(bundles)),
            config=str(output / "config.json"),
            staging=spec.get("staging"),
            jobs=verified[i :: len(gpus)],
        )
        for i, gpu in enumerate(gpus)
    ]
    return dict(
        manifest=str(manifest),
        manifest_sha256=digest(manifest),
        config_sha256=digest(config_path),
        feature_template_sha256=digest(feature_path) if feature_path else None,
        config=config,
        feature=feature,
        lanes=lanes,
        max_seconds=seconds,
        production_increment=0,
        production=production,
        output=str(output),
        training_started=False,
    )


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The owned child may exit between poll and signalling.
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def _cleanup(processes, guard, output, settle):
    """One failed cleanup cannot prevent settlement of the other owned jobs."""
    errors = []
    for _, process, _ in processes:
        try:
            _stop(process)
        except Exception as exc:
            errors.append(dict(pid=process.pid, error_type=type(exc).__name__))
    rows = []
    try:
        rows = settle(output)
        if any(not row["settled"] for row in rows):
            errors.append(dict(error_type="UnsettledService"))
    except Exception as exc:
        errors.append(dict(error_type=type(exc).__name__))
    # If settlement failed, retain the ownership guard as a last-resort cleanup.
    if guard is not None and not errors:
        try:
            _stop(guard)
        except Exception as exc:
            errors.append(dict(pid=guard.pid, error_type=type(exc).__name__))
    write_json(output / "final-service-settlement.json", dict(rows=rows, errors=errors))
    if errors:
        write_json(
            output / "status.json",
            dict(status="cleanup_failed", errors=errors, production_increment=0),
        )
        report = output / "report.json"
        if report.exists():
            write_json(report, dict(read_mapping(report), status="cleanup_failed"))
        raise RuntimeError("Owned batch cleanup incomplete; inspect settlement receipt")


def run(manifest):
    """Execute only a preflighted isolated batch; settle every owned service."""
    plan = prepare(manifest)
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "request.json", plan)
    write_json(out / "config.json", plan["config"])
    if not plan["production"]:
        write_json(out / "quarantine.json", dict(kind="isolated_postprocess_benchmark",
            generated_catalog=plan["config"]["generated_catalog"], production_training=False))
        write_json(Path(plan["feature"]["database"] + ".feature-worker.json"), plan["feature"])
    else:
        write_json(out / "production-binding.json", dict(manifest=plan["manifest"],
            manifest_sha256=plan["manifest_sha256"], native_gates_unchanged=True))
    from driversi.runtime.experiment_guard import birth, settle

    processes, guard = [], None
    started = time.time()
    try:
        # This guard also settles services if the supervisor is killed abruptly.
        guard_cmd = [
            sys.executable,
            "-m",
            "driversi.runtime.experiment_guard",
            "--owner",
            str(os.getpid()),
            "--birth",
            str(birth(os.getpid())),
            "--root",
            str(out),
            "--seconds",
            str(plan["max_seconds"] + 120),
        ]
        with (out / "guard.log").open("w") as log:
            guard = subprocess.Popen(
                guard_cmd,
                cwd=project_root(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json(out / "guard-launch.json", dict(pid=guard.pid, command=guard_cmd))
        for i, lane in enumerate(plan["lanes"]):
            spec_path = out / f"lane-{i}-spec.json"
            write_json(spec_path, lane)
            runtime = plan["config"]["runtime"]
            command = [
                runtime["gs_python"],
                "-u",
                "-m",
                "driversi.rsi.experiments.resident_postprocess",
                "--spec",
                str(spec_path),
                "--parent-pid",
                str(os.getpid()),
                "--max-seconds",
                str(plan["max_seconds"]),
            ]
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(lane["gpu"]),
                "CUDA_HOME": runtime["cuda_home"],
                "PATH": runtime["cuda_home"] + "/bin:" + os.environ.get("PATH", ""),
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            with (out / f"lane-{i}.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=project_root(),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append((i, process, command))
            write_json(
                out / "launch.json",
                dict(
                    started=started,
                    processes=[dict(lane=j, pid=p.pid, command=cmd) for j, p, cmd in processes],
                ),
            )
        deadline = time.monotonic() + plan["max_seconds"] + 30
        while any(p.poll() is None for _, p, _ in processes):
            write_json(
                out / "status.json",
                dict(
                    status="running",
                    elapsed_seconds=time.time() - started,
                    lanes=[dict(lane=i, exit_code=p.poll()) for i, p, _ in processes],
                ),
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("Resident batch deadline reached")
            time.sleep(1)
        if any(p.returncode != 0 for _, p, _ in processes):
            raise RuntimeError("A postprocessing lane failed; retained evidence must be inspected")
        rows = [
            dict(lane=i, **row)
            for i, _, _ in processes
            for row in read_mapping(out / f"lane-{i}/report.json")["rows"]
        ]
        settlement = settle(out)
        write_json(out / "service-settlement.json", dict(rows=settlement))
        if any(not row["settled"] for row in settlement):
            raise RuntimeError("An owned service did not settle")
        from driversi.rendering.gaussian.image_store import flush_outputs

        flush = flush_outputs(out)
        report = dict(
            status="completed",
            source_assets=1,
            prepared_jobs=len(rows),
            rows=rows,
            verified=sum(r["outcome"]["status"] ==
                ("training_sample_ready" if plan["production"] else "benchmark_sample_verified") for r in rows),
            wall_seconds=time.time() - started,
            output_flush=flush,
            production_increment=sum(bool(r["outcome"].get("trainable")) for r in rows) if plan["production"] else 0,
            training_started=False,
            timing_scope=("render_export_storage_settlement; retrieval_processing_deferred; physics_and_reconstruction_excluded"
                          if plan["feature"] is None else
                          "render_export_features_storage_settlement; physics_and_reconstruction_excluded"),
        )
        write_json(out / "report.json", report)
        write_json(
            out / "status.json",
            dict(status="completed", verified=report["verified"], production_increment=report["production_increment"]),
        )
        return report
    except BaseException as exc:
        write_json(
            out / "status.json",
            dict(status="failed", error_type=type(exc).__name__, production_increment=0),
        )
        raise
    finally:
        _cleanup(processes, guard, out, settle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Run GPU jobs; default is read-only preflight"
    )
    args = parser.parse_args()
    if args.execute:
        result = run(args.manifest)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("status", "verified", "wall_seconds", "production_increment")
                }
            )
        )
    else:
        plan = prepare(args.manifest)
        print(
            json.dumps(
                dict(
                    status="preflight_passed",
                    lanes=len(plan["lanes"]),
                    jobs=sum(len(lane["jobs"]) for lane in plan["lanes"]),
                    execution_started=False,
                )
            )
        )


if __name__ == "__main__":
    main()
