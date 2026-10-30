"""Bounded physical-only prefetch; consumers retain ordered ADE and admission."""

import fcntl
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def completed_physical_result(job, report):
    """Reuse terminal evidence without overwriting the original execution clock."""
    from driversi.agents.backends.native import digest
    from driversi.configuration import read_mapping
    from driversi.rsi.plugins.synthesis import check_files
    if not report.exists():return None
    if read_mapping(report).get('status')!='completed':
        raise ValueError('Unsettled physical prefetch may not be automatically repeated')
    output=Path(job['output']).resolve();path=output/'report.json'
    result=read_mapping(path);teacher=result.get('teacher',{})
    setup=read_mapping(output/'generation.json')
    if (setup['asset']!=job['asset'] or setup['plan_sha256']!=digest(job['plan_path'])
            or teacher.get('plan_sha256')!=digest(job['plan_path'])
            or teacher.get('assignment_id')!=job['assignment_id']
            or teacher.get('harness_status')!='completed'
            or teacher.get('status') not in ('trajectory_admitted','execution_or_admission_rejected')
            or result.get('status') not in ('teacher_rejected','trajectory_ready','rendered')
            or Path(setup['config']['input']['database']).resolve()!=Path(job['database']).resolve()):
        raise ValueError('Completed physical receipt does not match the requested assignment')
    native_path=Path(teacher['teacher_report']).resolve(strict=True)
    if not native_path.is_relative_to(output):raise ValueError('Foreign physical teacher receipt')
    native=read_mapping(native_path)
    if native.get('status') not in ('completed','rejected'):raise ValueError('Unknown native teacher outcome')
    check_files(native['source_files']);check_files(native.get('files',{}))
    frozen_json(report.with_name('physical-prefetch-reuse.json'),dict(
        original_report=str(report),original_sha256=digest(report),
        generation=str(path),generation_sha256=digest(path),
        native_receipt=str(native_path),native_sha256=digest(native_path),
        physical_execution_repeated=False,original_execution_time_preserved=True))
    return dict(result,status='trajectory_ready') if result['status']=='rendered' else result


def physical_job(job):
    from driversi.rsi.experiments.reactive_generation import generate

    if job.get("render") is not False:
        raise ValueError("Prefetch may only execute physical generation")
    report = Path(job["output"]).parent / "physical-prefetch.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with (report.parent / "physical-prefetch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = completed_physical_result(job, report)
        if cached is not None:return cached
        started = time.time()
        write_json(report, dict(status="running", started=started, pid=os.getpid()))
        try:
            result = generate(**job)
        except BaseException as exc:
            write_json(
                report,
                dict(
                    status="failed",
                    started=started,
                    finished=time.time(),
                    error_type=type(exc).__name__,
                    pid=os.getpid(),
                ),
            )
            raise
        write_json(
            report,
            dict(
                status="completed",
                physical_status=result["status"],
                started=started,
                finished=time.time(),
                pid=os.getpid(),
            ),
        )
        return result


class PhysicalPrefetch:
    """At most workers outstanding; never decides novelty or writes vector indices."""

    def __init__(self, root, workers=1, *, executor_factory=None, worker=physical_job):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError("Physical prefetch requires 1..4 workers")
        self.root = Path(root)
        self.workers = workers
        self.factory = executor_factory
        self.worker = worker
        self.executor = None
        self.futures = {}
        self.jobs = {}
        self.consumed = set()
        self.errors = {}

    def __enter__(self):
        if self.workers > 1:
            frozen_json(
                self.root / "physical-prefetch-policy.json",
                dict(
                    workers=self.workers,
                    render=False,
                    ordered_admission=True,
                    maximum_unused_lookahead=self.workers,
                ),
            )
            self.executor = (
                self.factory(self.workers)
                if self.factory
                else ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")
                )
            )
        return self

    def schedule(self, jobs):
        if self.executor is None:
            return
        for cid, job in jobs:
            if job.get("render") is not False:
                raise ValueError("Rendering is forbidden in physical prefetch")
            if cid in self.jobs:
                if self.jobs[cid] != job:
                    raise ValueError("Prefetch assignment changed")
                continue
            if len(self.futures.keys() - self.consumed) >= self.workers:
                break
            self.jobs[cid] = job
            self.futures[cid] = self.executor.submit(self.worker, job)
        self.snapshot("running")

    def has(self, cid):
        return cid in self.futures

    def result(self, cid):
        try:
            return self.futures[cid].result()
        except BaseException as exc:
            self.errors[cid] = type(exc).__name__
            raise
        finally:
            self.consumed.add(cid)
            self.snapshot("running")

    def snapshot(self, status):
        if self.executor is None:
            return
        write_json(
            self.root / "physical-prefetch-status.json",
            dict(
                status=status,
                workers=self.workers,
                submitted=len(self.futures),
                completed=sum(f.done() for f in self.futures.values()),
                consumed=len(self.consumed),
                errors=self.errors,
                unconsumed=[cid for cid in self.futures if cid not in self.consumed],
                qualified_samples_certified=False,
                jobs={
                    cid: dict(
                        done=f.done(),
                        consumed=cid in self.consumed,
                        output=self.jobs[cid]["output"],
                    )
                    for cid, f in self.futures.items()
                },
            ),
        )

    def __exit__(self, exc_type, exc, tb):
        if self.executor is None:
            return
        # Finish all owned work before training. Surplus trajectories remain physical
        # evidence only, are reported, and are never counted as admitted samples.
        for cid, future in self.futures.items():
            if cid not in self.consumed:
                try:
                    future.result()
                except BaseException as error:
                    self.errors[cid] = type(error).__name__
        self.executor.shutdown(wait=True)
        self.snapshot("settled_after_error" if exc_type else "settled")
