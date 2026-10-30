"""Bounded CPU prefetch, native GPU encoding and durable single-writer corpus dump."""

import os
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .features import VisionEncoder, feature_profile
from .observations import history_evidence
from .store import encode
from .vectors import VectorStore


def write_report(path, report):
    temp = Path(str(path) + ".tmp")
    temp.write_text(encode(report))
    os.replace(temp, path)


def load_evidence(scene, config):
    try:
        evidence, images = history_evidence(scene, config["cameras"], config["max_side"])
        return evidence, images, None
    except Exception as exc:
        return [], [], type(exc).__name__ + ": " + str(exc)[:2000]


def dump_corpus(catalog, config, output, *, workers=4, tokens=None, backend_factory=VisionEncoder):
    """Only the main thread touches SQLite/GPU. Prefetch cannot change image order."""
    import fcntl

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use 1..8 bounded image-prefetch workers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    purpose = "training" if catalog.split == "navtrain" else "evaluation"
    sql, args = catalog.token_query(purpose)
    population = [row[0] for row in catalog.db.execute(sql, args)]
    if tokens is None:
        tokens = population
    elif not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= set(population):
        raise ValueError("Explicit tokens must be unique members of this catalog")
    # Locality changes scheduling only, not the set or per-image encoder contract.
    selected = set(tokens)
    tokens = [row[0] for row in catalog.db.execute(
        "SELECT token FROM scenes ORDER BY source_log,token") if row[0] in selected]
    stop = {"requested": False}
    previous = {}
    def request_stop(signum, frame):
        stop["requested"] = True

    with Path(str(catalog.path) + ".feature.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = VectorStore(catalog, writable=True)
        data = feature_profile(config)
        profile = store.profile(data)
        (output / "profile.json").write_text(encode(data))
        report = {"split": catalog.split, "profile": profile, "requested": len(tokens),
                  "population": len(population), "processed": 0, "ready": 0, "reused": 0,
                  "failed": 0, "complete": False, "status": "running", "workers": workers,
                  "trainable": False}
        backend, started = None, time.monotonic()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool, (output / "attempts.jsonl").open("a", buffering=1) as log:
                pending = deque()
                iterator = iter(tokens)
                def fill():
                    while len(pending) < workers * 2 and not stop["requested"]:
                        token = next(iterator, None)
                        if token is None:
                            break
                        pending.append((token, pool.submit(load_evidence, catalog.scene(token), config)))
                fill()
                while pending and not stop["requested"]:
                    token, future = pending.popleft()
                    evidence, images, error = future.result()
                    values, cost, state = None, {}, "failed"
                    try:
                        if error is None and store.completed(token, profile, evidence):
                            state = "reused"
                        elif error is None:
                            if backend is None:
                                backend = backend_factory(config)
                            values, cost = backend(images)
                            store.put(token, profile, evidence, values, cost)
                            state = "ready"
                    except Exception as exc:
                        error = type(exc).__name__ + ": " + str(exc)[:2000]
                    if state == "failed":
                        store.put(token, profile, evidence, None, cost, error)
                    log.write(encode({"token": token, "status": state, "error": error, "cost": cost}) + "\n")
                    report[state] += 1
                    report["processed"] += 1
                    report["elapsed_seconds"] = time.monotonic() - started
                    write_report(output / "report.json", report)
                    fill()
                for _, future in pending:
                    future.cancel()
            report["complete"] = report["processed"] == len(tokens)
            report["status"] = ("complete_with_failures" if report["failed"] else "complete") if report["complete"] else "interrupted"
            report["stats"] = store.stats(profile)
            write_report(output / "report.json", report)
            return report
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
