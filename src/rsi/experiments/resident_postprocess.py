"""Explicit isolated prepared-branch benchmark using original provider/export checks."""

import argparse
import ctypes
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def parent_bound_process(argv, output, seconds, *, environment=None):
    from driversi.rsi.plugins.synthesis import run_process

    run_process(
        [
            sys.executable,
            str(project_root() / "src/runtime/parent_exec.py"),
            str(os.getpid()),
            *argv,
        ],
        output,
        seconds,
        environment=environment,
    )


def render_prepared(generation, output, gpu, runner=None, *, production=False):
    from driversi.rsi.plugins import worldengine
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage
    from driversi.rsi.workflow.artifacts import seal

    generation, output = Path(generation), Path(output)
    original = read_mapping(generation)
    setup = read_mapping(generation.parent / "generation.json")["config"]
    registry = Path(setup["registry"])
    task = setup["id"] + "-teacher"
    if (
        setup["input"]["kind"] != "assignment"
        or original["teacher"]["status"] != "trajectory_admitted"
    ):
        raise ValueError("Only an actually completed native teacher assignment is supported")
    _, selected, _ = _stored_stage(
        dict(source_task=task, source_stage="admission"), "selected_native_branch", registry
    )
    cfg, _ = worldengine.prepare(dict(setup["render"]["config"], gpu=gpu), project_root(), registry)
    source_cfg, _ = worldengine.prepare_source(
        dict(setup["render"]["source_config"], gpu=gpu), project_root(), registry
    )
    started = time.perf_counter()
    rendered = worldengine.execute(
        cfg, {"branch": selected}, output / "render", registry, process_runner=runner
    )
    rendered["summary"]["artifact_reference"] = seal(
        rendered,
        output / "render",
        output.name,
        dict(
            id="render",
            output_kind="worldengine_inspection_window",
            provider="resident_native_render_production" if production else "isolated_native_render_benchmark",
            provider_version="1",
        ),
    )
    source = worldengine.execute_source(
        source_cfg, {"render": rendered}, output / "source", registry, process_runner=runner
    )
    result = dict(
        original,
        status="rendered",
        render_status="completed",
        branch_report=selected["artifacts"]["report"],
        render_report=rendered["artifacts"]["report"],
        source_report=source["artifacts"]["report"],
        render_payload=rendered["artifacts"]["render_report"],
        benchmark_only=not production,
        production_training=production,
    )
    result["files"] = {
        result[k]: digest(result[k])
        for k in ("branch_report", "render_report", "source_report", "render_payload")
    }
    write_json(output / "report.json", result)
    return time.perf_counter() - started


def run_lane(spec_path, *, parent_pid=None, max_seconds=2400):
    if parent_pid is not None:
        if (
            os.getppid() != parent_pid
            or ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0
            or os.getppid() != parent_pid
        ):
            raise RuntimeError("Benchmark lane lost its owning supervisor")
    if not 1 <= max_seconds <= 7200:
        raise ValueError("Explicit bounded lane budget required")
    watchdog = threading.Timer(max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    from driversi.rsi.experiments.chainflow_campaign import process
    from driversi.rsi.services.native_export_worker import NativeExportSession

    spec = read_mapping(spec_path)
    cfg = read_mapping(spec["config"])
    from driversi.data.datasets.postprocessing import mode
    processing_mode = mode(cfg, production=spec.get('production',False))
    out = Path(spec["output"]).resolve()
    if not out.is_relative_to(project_root() / "exp") or out.exists():
        raise ValueError("A fresh isolated benchmark directory under exp is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["gpu"]):
        raise ValueError("Fixed per-lane GPU required")
    if not 1 <= len(spec["jobs"]) <= 64:
        raise ValueError("Bounded explicit prepared jobs required")
    out.mkdir(parents=True)
    frozen_json(out / "request.json", spec)
    start = time.time()
    runner, exporter = parent_bound_process, None
    try:
        if spec["mode"] == "resident":
            from driversi.rendering.gaussian.image_store import PngWriter
            from driversi.rendering.gaussian.resident_session import ResidentRenderSession

            runner = ResidentRenderSession(
                spec["bundle"],
                spec["gpu"],
                image_writer=PngWriter(compression=1, staging=spec.get("staging")),
            )
            exporter = NativeExportSession(
                cfg["chainflow_python"], spec["gpu"], out / "export-service"
            )
        elif spec["mode"] != "baseline":
            raise ValueError("Unknown benchmark mode")
        rows = []
        for index, job in enumerate(spec["jobs"]):
            if (
                digest(job["generation"]) != job["sha256"]
                or digest(job["diversity"]) != job["diversity_sha256"]
            ):
                raise ValueError("Prepared physical branch or diversity changed")
            directory = out / f"branch-{index:03d}"
            directory.mkdir()
            frozen_json(directory / "diversity.json", read_mapping(job["diversity"]))
            began = time.time()
            seconds = render_prepared(
                job["generation"], directory / "generation", spec["gpu"], runner, production=spec.get("production",False)
            )
            outcome = process(
                directory / "generation/report.json",
                spec["config"],
                directory / "admission",
                execution_gpu=spec["gpu"],
                quarantine=not spec.get("production",False),
                native_runner=exporter,
            )
            rows.append(
                dict(
                    index=index,
                    prepared_generation=job["generation"],
                    render_seconds=seconds,
                    wall_seconds=time.time() - began,
                    outcome=outcome,
                    generation=str(directory / "generation/report.json"),
                )
            )
            write_json(
                out / "progress.json",
                dict(
                    status="running", rows=rows, started=start, elapsed_seconds=time.time() - start
                ),
            )
        closing = time.perf_counter()
        if exporter:
            exporter.close()
            exporter = None
        settlement_seconds = time.perf_counter() - closing
        from driversi.rendering.gaussian.image_store import flush_outputs

        flush = flush_outputs(out)
        write_json(
            out / "report.json",
            dict(
                status="completed",
                started=start,
                finished=time.time(),
                wall_seconds=time.time() - start,
                mode=spec["mode"],
                gpu=spec["gpu"],
                rows=rows,
                export_settlement_seconds=settlement_seconds,
                output_flush=flush,
                scope=("render_native_export_durable_files; retrieval_processing_deferred; prepared_physics_excluded"
                       if processing_mode == 'training_only' else
                       "render_native_export_fresh_features_durable_files; prepared_physics_excluded"),
                production_increment=sum(bool(r["outcome"].get("trainable")) for r in rows),
            ),
        )
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if exporter:
            exporter.close()
        watchdog.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, default=2400)
    args = parser.parse_args()
    run_lane(args.spec, parent_pid=args.parent_pid, max_seconds=args.max_seconds)
