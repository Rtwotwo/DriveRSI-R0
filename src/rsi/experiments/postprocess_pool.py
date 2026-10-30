"""Bounded independent GS/export lanes; parent owns ordered ADE and accounting."""

import ctypes
import fcntl
import multiprocessing
import os
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json

_termination_requested = False


class UnsettledPostprocess(RuntimeError):
    """An existing launch lacks a verified result; never implicitly repeat it."""


def read_result(directory, job):
    directory = Path(directory)
    if read_mapping(directory / "job.json") != job:
        raise ValueError("Parallel postprocessing input changed")
    if "config" in job and digest(job["config"]) != job["config_sha256"]:
        raise ValueError("Parallel configuration changed")
    if "generation" in job and digest(job["generation"]["plan_path"]) != job["plan_sha256"]:
        raise ValueError("Parallel physical plan changed")
    path = directory / "result.json"
    seal = read_mapping(directory / "result-seal.json")
    if seal["sha256"] != digest(path):
        raise ValueError("Parallel postprocessing result changed")
    result = read_mapping(path)
    if result["job_sha256"] != digest(directory / "job.json"):
        raise ValueError("Result belongs to another postprocessing job")
    for path, sha in result["files"].items():
        if digest(path) != sha:
            raise ValueError("Parallel output evidence changed")
    outcome = result["outcome"]
    if outcome.get("sample_report"):
        from driversi.data.datasets.native_admission import validate_sample

        validate_sample(outcome["sample_report"])
    return outcome


def _branch_job(job):
    from driversi.rsi.experiments.chainflow_campaign import process
    from driversi.rsi.experiments.reactive_generation import generate

    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "owner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnsettledPostprocess("Parallel branch still has an owner") from exc
        frozen_json(directory / "job.json", job)
        if (directory / "result-seal.json").exists():
            return read_result(directory, job)
        if (directory / "launch.json").exists():
            raise UnsettledPostprocess("Unsettled prior postprocess; no retry")
        if digest(job["config"]) != job["config_sha256"]:
            raise ValueError("Parallel configuration changed")
        if digest(job["generation"]["plan_path"]) != job["plan_sha256"]:
            raise ValueError("Parallel physical plan changed")
        config = read_mapping(job["config"])
        from driversi.data.datasets.postprocessing import mode
        if mode(config, production=not job.get("quarantine", False)) != "training_only":
            from driversi.data.catalog.feature_worker import descriptor, load_config
            service = load_config(descriptor(config["generated_catalog"]))
            if (
                service["database"] != str(Path(config["generated_catalog"]).resolve())
                or service["source_catalog"] != str(Path(config["train_catalog"]).resolve())
                or service["profile_id"] != config["profile"]
                or (not job.get("quarantine") and service["gpu"] != 0)
            ):
                raise ValueError("Parallel features require the matching serialized service")
        started = time.time()
        write_json(directory / "launch.json", dict(pid=os.getpid(), started=started))
        try:
            generated = generate(**job["generation"], render_gpu=job["gpu"])
            if generated["status"] == "rendered":
                outcome = process(
                    Path(job["generation"]["output"]) / "report.json",
                    job["config"],
                    job["admission"],
                    execution_gpu=job["gpu"],
                    quarantine=job.get("quarantine", False),
                )
            else:
                outcome = dict(status=generated["status"], trainable=False)
            files = dict(outcome.get("files", {}))
            if outcome.get("sample_report"):
                files.update(read_mapping(outcome["sample_report"])["files"])
            report = Path(job["generation"]["output"]) / "report.json"
            files[str(report)] = digest(report)
            files.update(generated.get("files", {}))
        except Exception as exc:
            (directory / "failure.txt").write_text(traceback.format_exc())
            outcome = dict(
                status="execution_failed", trainable=False, error_type=type(exc).__name__
            )
            files = {str(directory / "failure.txt"): digest(directory / "failure.txt")}
        result = dict(
            job_sha256=digest(directory / "job.json"),
            outcome=outcome,
            files=files,
            started=started,
            finished=time.time(),
            gpu=job["gpu"],
        )
        write_json(directory / "result.json", result)
        frozen_json(directory / "result-seal.json", dict(sha256=digest(directory / "result.json")))
        return outcome


def branch_job(job):
    try:
        return _branch_job(job)
    finally:
        # Executor workers catch BaseException; exit after owned-child cleanup
        # rather than returning to the task queue after supervisor termination.
        if _termination_requested:
            os._exit(130)


def _bind_parent():
    parent = os.getppid()

    def terminate(_signum, _frame):
        global _termination_requested
        _termination_requested = True
        raise KeyboardInterrupt("Parallel supervisor ended")

    signal.signal(signal.SIGTERM, terminate)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != parent:
        raise RuntimeError("Cannot bind parallel worker lifetime")


class PostprocessPool:
    def __init__(self, root, gpus=None, *, factory=None, worker=branch_job):
        self.root = Path(root)
        self.gpus = [] if gpus is None else gpus
        if self.gpus and (
            not isinstance(self.gpus, list)
            or not 1 <= len(self.gpus) <= 4
            or len(set(self.gpus)) != len(self.gpus)
            or any(type(g) is not int or not 1 <= g <= 5 for g in self.gpus)
        ):
            raise ValueError("Postprocessing uses distinct GPUs1..5, at most4 lanes")
        self.factory, self.worker = factory, worker
        self.executor = None
        self.jobs, self.futures, self.consumed, self.errors = {}, {}, set(), {}

    def __enter__(self):
        if self.gpus:
            self.root.mkdir(parents=True, exist_ok=True)
            frozen_json(
                self.root / "postprocess-policy.json",
                dict(
                    gpus=self.gpus,
                    ordered_ade=True,
                    ordered_consumption=True,
                    feature_writer="resident_serial",
                ),
            )
            self.executor = (
                self.factory(len(self.gpus))
                if self.factory
                else ProcessPoolExecutor(
                    max_workers=len(self.gpus),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_bind_parent,
                )
            )
        return self

    @property
    def outstanding(self):
        return len(self.futures.keys() - self.consumed)

    def submit(self, cid, job):
        if self.executor is None:
            raise ValueError("Parallel postprocessing is not enabled")
        if cid in self.jobs:
            previous = {k: v for k, v in self.jobs[cid].items() if k != "gpu"}
            if previous != job:
                raise ValueError("Parallel candidate input changed")
            return
        busy = {self.jobs[c]["gpu"] for c in self.futures if c not in self.consumed}
        free = [g for g in self.gpus if g not in busy]
        if not free:
            raise ValueError("Parallel postprocessing capacity exhausted")
        saved = Path(job["directory"]) / "job.json"
        gpu = read_mapping(saved)["gpu"] if saved.exists() else free[0]
        if gpu not in free:
            raise ValueError("Previously bound GPU is not available")
        bound = dict(job, gpu=gpu)
        if saved.exists() and read_mapping(saved) != bound:
            raise ValueError("Saved parallel job differs")
        self.jobs[cid] = bound
        self.futures[cid] = self.executor.submit(self.worker, bound)
        self.snapshot("running")

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
            self.root / "postprocess-status.json",
            dict(
                status=status,
                gpus=self.gpus,
                submitted=len(self.futures),
                completed=sum(f.done() for f in self.futures.values()),
                consumed=len(self.consumed),
                errors=self.errors,
                jobs={
                    cid: dict(
                        gpu=self.jobs[cid]["gpu"], done=f.done(), consumed=cid in self.consumed
                    )
                    for cid, f in self.futures.items()
                },
                qualified_samples_certified=False,
            ),
        )

    def __exit__(self, kind, exc, tb):
        if self.executor is None:
            return
        for cid, future in self.futures.items():
            if cid not in self.consumed:
                try:
                    future.result()
                except BaseException as error:
                    self.errors[cid] = type(error).__name__
        self.executor.shutdown(wait=True)
        self.snapshot("settled_after_error" if kind else "settled")


def run_batch(manifest_path):
    """Process an ordered, already physically executed and ADE-screened batch."""
    from driversi.agents.backends.native import project_root
    from driversi.data.catalog.store import identity
    from driversi.rendering.diagnostics.numeric_admission import numeric_settings
    from driversi.rsi.decisions.trajectory_diversity import compare, describe

    path = Path(manifest_path).resolve(strict=True)
    spec = read_mapping(path)
    root = Path(spec["output"]).resolve()
    if spec.get("schema_version") != 1 or not root.is_relative_to(project_root().resolve() / "exp"):
        raise ValueError("A versioned batch under exp is required")
    jobs = spec["jobs"]
    if not spec.get("gpus"):
        raise ValueError("Batch requires explicit postprocessing GPUs")
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 64:
        raise ValueError("One to64 explicit prepared branches required")
    destinations = []
    retained = {}
    for job in jobs:
        d = Path(job["generation"]["output"]).resolve()
        for destination in [d, Path(job["directory"]).resolve(), Path(job["admission"]).resolve()]:
            if not destination.is_relative_to(project_root().resolve() / "exp"):
                raise ValueError("Batch output escapes exp")
        parent = d.parent
        if len({d, Path(job["directory"]).resolve(), Path(job["admission"]).resolve()}) != 3:
            raise ValueError("Generation, admission and receipt directories must be distinct")
        if (
            Path(job["directory"]).resolve().parent != parent
            or Path(job["admission"]).resolve().parent != parent
            or job["generation"].get("render") is not True
            or type(job.get("quarantine", False)) is not bool
        ):
            raise ValueError("One isolated candidate directory and explicit render are required")
        destinations.append(str(parent))
        config = read_mapping(job["config"])
        if (
            digest(job["config"]) != job["config_sha256"]
            or digest(job["generation"]["plan_path"]) != job["plan_sha256"]
        ):
            raise ValueError("Prepared batch input changed")
        generated = read_mapping(d / "report.json")
        if generated["status"] not in ("trajectory_ready", "rendered"):
            raise ValueError("Batch requires completed admitted physical execution")
        evidence = read_mapping(d.parent / "diversity.json")
        description = describe(
            dict(generated, status="trajectory_ready"), evidence["description"]["candidate_id"]
        )
        if description != evidence["description"]:
            raise ValueError("Prepared ADE differs from actual execution")
        minimum = numeric_settings(config["visual_quality_profile"])["minimum_ade_m"]
        spatial=numeric_settings(config['visual_quality_profile']).get('minimum_path_ade_m')
        for previous in evidence.get("retained", []):
            for source, sha in previous["files"].items():
                if digest(source) != sha:
                    raise ValueError("Prior ADE evidence changed")
        if (
            compare(description, evidence.get("retained", []), minimum_ade_m=minimum,minimum_path_ade_m=spatial)
            != evidence["decision"]
            or not evidence["decision"]["distinct"]
        ):
            raise ValueError("Prepared ADE decision rejected")
        key = identity(description["comparison_key"])
        if not compare(description, retained.get(key, []), minimum_ade_m=minimum,minimum_path_ade_m=spatial)["distinct"]:
            raise ValueError("Duplicate branches within parallel batch")
        retained.setdefault(key, []).append(description)
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate branch output in batch")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_json(root / "batch-input.json", dict(manifest=str(path), sha256=digest(path)))
        results = []
        with PostprocessPool(root, spec["gpus"]) as pool:
            pending = []
            for index, job in enumerate(jobs):
                if len(pending) >= len(pool.gpus):
                    cid = pending.pop(0)
                    results.append(dict(candidate=cid, outcome=pool.result(cid)))
                cid = f"branch-{index:03d}"
                pool.submit(cid, job)
                pending.append(cid)
            for cid in pending:
                results.append(dict(candidate=cid, outcome=pool.result(cid)))
        unsettled = any(
            row["outcome"].get("status")
            in {"teacher_outcome_unknown", "render_running", "render_registered"}
            for row in results
        )
        report = dict(
            status="awaiting_settlement" if unsettled else "settled",
            rows=results,
            ordered=True,
            ready=sum(row["outcome"].get("trainable") is True for row in results),
            training_started=False,
            automatic_promotion=False,
        )
        write_json(root / "batch-report.json", report)
        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run_batch(args.manifest)
