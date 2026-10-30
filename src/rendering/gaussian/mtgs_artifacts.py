"""Bounded CPU artifact writes scoped to one MTGS capture.

This changes scheduling only: PNG level 1, compressed NPZ arrays and hashes retain
their existing semantics. A caller must finish before publishing artifact paths.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from driversi.rendering.gaussian.mtgs import _sha256


def validate_image_write_workers(workers: int) -> int:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("image_write_workers must be an integer in [1,4]")
    return workers


def _snapshot(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _save_png(path, array, with_hash):
    from PIL import Image

    started = time.perf_counter()
    Image.fromarray(array).save(path, compress_level=1)
    sha = _sha256(path) if with_hash else None
    return {"seconds": time.perf_counter() - started,
            "sha256": sha}


def _save_source(path, source, size):
    from PIL import Image

    started = time.perf_counter()
    with Image.open(source) as raw:
        raw.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(path, compress_level=1)
    return {"seconds": time.perf_counter() - started, "sha256": None}


def _save_surface(path, accumulation, depth, valid):
    started = time.perf_counter()
    with path.open("xb") as stream:
        np.savez_compressed(stream, accumulation=accumulation,
                            expected_depth_m=depth, lens_valid=valid)
    sha = _sha256(path)
    return {"seconds": time.perf_counter() - started, "sha256": sha}


class CaptureArtifactWriter:
    """At most 2*workers submitted jobs, with immutable owned array snapshots.

    Jobs may leave unreferenced files on failure, as the former serial path did.
    Closing cancels queued work and joins every running job, including on errors.
    No background task survives capture and no durability/fsync claim is made.
    """

    def __init__(self, workers: int = 1):
        self.workers = validate_image_write_workers(workers)
        self.max_pending = 2 * self.workers
        self.peak_pending = 0
        self._pending = deque()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mtgs-save") if workers > 1 else None
        self._closed = False

    def __enter__(self):
        return self

    def _admit(self):
        if self._closed:
            raise RuntimeError("Artifact writer is closed")
        while len(self._pending) >= self.max_pending:
            self._pending.popleft().result()

    def _submit(self, function, *args):
        if self._pool is None:
            future = Future()
            future.set_result(function(*args))
            return future
        future = self._pool.submit(function, *args)
        self._pending.append(future)
        self.peak_pending = max(self.peak_pending, len(self._pending))
        return future

    def png(self, path: Path, array, *, with_hash=False):
        self._admit()
        return self._submit(_save_png, Path(path), _snapshot(array), with_hash)

    def source(self, path: Path, source: Path, size):
        self._admit()
        return self._submit(_save_source, Path(path), Path(source), tuple(size))

    def surface(self, path: Path, accumulation, depth, valid):
        self._admit()
        return self._submit(_save_surface, Path(path), _snapshot(accumulation),
                            _snapshot(depth), _snapshot(valid))

    def finish(self):
        while self._pending:
            self._pending.popleft().result()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.finish()
        finally:
            for future in self._pending:
                future.cancel()
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
            self._pending.clear()
            self._closed = True
        return False
