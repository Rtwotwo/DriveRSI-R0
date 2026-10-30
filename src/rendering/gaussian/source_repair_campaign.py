"""Bounded immutable appearance-repair rounds; never grants data admission."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def run(source_report, asset, sensor_root, output, *, rounds=8, max_wall_seconds=3600,
        minimum_source_psnr_db=25.0, optimize_geometry=False):
    from driversi.rendering.gaussian.source_refinement import run as refine
    from driversi.rendering.gaussian.source_preflight import run as inspect

    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("One to eight bounded repair rounds required")
    if type(optimize_geometry) is not bool or (optimize_geometry and rounds > 3):
        raise ValueError("Geometry escalation allows at most three rounds")
    steps = 3600 if optimize_geometry else 1200
    if minimum_source_psnr_db not in (25.0, 27.0):
        raise ValueError("Explicit supported source threshold required")
    if type(max_wall_seconds) is not int or not 120 <= max_wall_seconds <= 7200:
        raise ValueError("Finite source-repair wall budget required")
    out = Path(output).resolve()
    if not out.is_relative_to(project_root() / "exp") or out.exists():
        raise ValueError("New owned experiment directory required")
    previous = Path(source_report).resolve(strict=True)
    initial = read_mapping(previous)
    if initial.get("source_mode") != "canonical_source_pose_without_motion_v1":
        raise ValueError("Canonical source preflight required")
    if initial["asset"] != asset:
        raise ValueError("Source asset identity differs from preflight")
    out.mkdir(parents=True)
    write_json(out / "request.json", dict(source_report=str(previous), source_sha256=digest(previous),
        asset=asset, sensor_root=str(sensor_root), rounds=rounds, max_wall_seconds=max_wall_seconds,
        optimizer_steps_per_round=steps, optimize_geometry=optimize_geometry,
        threshold=minimum_source_psnr_db, operator='>', trainable=False,
        producer_sha256=digest(__file__),
        holdout_role="excluded_from_gradients_but_included_in_mean_selection"))
    started = time.monotonic()
    rows = []
    stalled = 0
    score = initial["mean_psnr_db"]
    state = "source_threshold_met" if score > minimum_source_psnr_db else "round_budget_exhausted"
    try:
        for index in range(rounds):
            if score > minimum_source_psnr_db:
                break
            remaining = max_wall_seconds - (time.monotonic()-started)
            if remaining < 180:
                state = "wall_budget_exhausted"
                break
            extra = dict(optimize_geometry=True) if optimize_geometry else {}
            child = refine(previous, out / f"round-{index:02d}-refinement", steps=steps,
                           max_wall_seconds=int(remaining-90), **extra)
            current_asset = {**asset, "bundle_root": child["child_asset"]}
            checked = inspect(current_asset, sensor_root, out / f"round-{index:02d}-preflight",
                              minimum_source_psnr_db=minimum_source_psnr_db)
            heldout = [r["measured_psnr_db"] for r in checked["frames"] if r["tick"] in (5, 35, 65)]
            gain = checked["mean_psnr_db"] - score
            score = checked["mean_psnr_db"]
            previous = out / f"round-{index:02d}-preflight/report.json"
            rows.append(dict(round=index, source_report=str(previous), source_sha256=digest(previous),
                child_asset=child["child_asset"], mean_psnr_db=score, improvement_db=gain,
                repair_holdout_mean_psnr_db=sum(heldout)/len(heldout),
                repair_holdout_scope="excluded_from_gradients_but_used_in_mean_selection_not_independent_test",
                optimizer_steps=steps, trainable=False))
            stalled = stalled + 1 if gain < .01 else 0
            state = "source_threshold_met" if score > minimum_source_psnr_db else "insufficient_repair_progress" if stalled >= 2 else "round_budget_exhausted"
            write_json(out / "progress.json", dict(status=state, rows=rows, trainable=False))
            print(json.dumps(rows[-1]), flush=True)
            if score > minimum_source_psnr_db or stalled >= 2:
                break
    except BaseException as exc:
        write_json(out / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), rows=rows, trainable=False))
        raise
    result = dict(status=state, rows=rows, source_report=str(previous), mean_psnr_db=score,
        threshold=minimum_source_psnr_db, operator='>',
        next_action='fresh_branch_admission' if state=='source_threshold_met' else 'reconstruct_geometry',
        wall_seconds=time.monotonic()-started, trainable=False,
        scope="source_asset_only_requires_fresh_same_branch_admission")
    write_json(out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    job = read_mapping(args.job)
    if args.worker:
        print(json.dumps(run(**job)), flush=True)
        return
    timeout = job.get("max_wall_seconds", 3600)
    if type(timeout) is not int or not 120 <= timeout <= 7200:
        raise ValueError("Finite source-repair wall budget required")
    # Own one worker (no child jobs); subprocess.run kills and reaps it on timeout,
    # including blocked setup, CUDA calls and source-image evaluation.
    command = [sys.executable, "-u", "-m", "driversi.rendering.gaussian.source_repair_campaign",
               "--job", str(Path(args.job).resolve()), "--worker"]
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        out = Path(job["output"]).resolve()
        if out.is_relative_to(project_root()/"exp") and out.is_dir():
            write_json(out/"supervisor-timeout.json", dict(status="hard_wall_timeout",
                       max_wall_seconds=timeout, worker_reaped=True, trainable=False))
        raise


if __name__ == "__main__":
    main()
