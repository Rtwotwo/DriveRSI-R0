import hashlib
import threading

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_artifacts import CaptureArtifactWriter


@pytest.fixture
def pillow():
    # Pillow is an optional runtime dependency, absent in the minimal core venv.
    return pytest.importorskip("PIL.Image")


@pytest.mark.parametrize("workers", [0, 5, True, 1.5, "4"])
def test_writer_rejects_unbounded_or_ambiguous_worker_counts(workers):
    with pytest.raises(ValueError, match="image_write_workers"):
        CaptureArtifactWriter(workers)


def test_serial_parallel_files_pixels_arrays_and_hashes_match(tmp_path, pillow):
    Image = pillow
    rng = np.random.default_rng(17)
    rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    alpha = rng.random((24, 32), dtype=np.float32)
    depth = rng.random((24, 32), dtype=np.float32) * 80
    valid = alpha > 0.1
    source = tmp_path / "source.png"
    Image.fromarray(rgb).save(source)
    for workers in (1, 4):
        directory = tmp_path / str(workers)
        directory.mkdir()
        with CaptureArtifactWriter(workers) as writer:
            image = writer.png(directory / "image.png", rgb, with_hash=True)
            surface = writer.surface(directory / "surface.npz", alpha, depth, valid)
            writer.source(directory / "source.png", source, (32, 24))
            writer.finish()
            for name, future in (("image.png", image), ("surface.npz", surface)):
                assert future.result()["sha256"] == hashlib.sha256((directory/name).read_bytes()).hexdigest()
        np.testing.assert_array_equal(np.asarray(Image.open(directory/"image.png")), rgb)
        with np.load(directory/"surface.npz") as archive:
            for key, array in (("accumulation", alpha), ("expected_depth_m", depth), ("lens_valid", valid)):
                assert archive[key].dtype == array.dtype
                np.testing.assert_array_equal(archive[key], array)
    for name in ("image.png", "surface.npz", "source.png"):
        assert (tmp_path/"1"/name).read_bytes() == (tmp_path/"4"/name).read_bytes()


def test_writer_owns_snapshot_and_bounds_jobs(tmp_path, monkeypatch, pillow):
    Image = pillow
    from driversi.rendering.gaussian import mtgs_artifacts as module

    original = module._save_png
    release, four_started = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, peak = 0, 0

    def delayed(path, array, with_hash):
        nonlocal active, peak
        assert not array.flags.writeable
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 4:
                four_started.set()
        assert release.wait(5)
        try:
            return original(path, array, with_hash)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(module, "_save_png", delayed)
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    with CaptureArtifactWriter(4) as writer:
        for i in range(8):
            writer.png(tmp_path/f"{i}.png", array)
        assert four_started.wait(5)
        assert writer.peak_pending == 8
        array[:] = 255
        release.set()
        writer.png(tmp_path/"8.png", array)
        writer.finish()
        assert writer.peak_pending <= 8 and peak <= 4
    for i in range(8):
        assert not np.asarray(Image.open(tmp_path/f"{i}.png")).any()
    assert np.asarray(Image.open(tmp_path/"8.png")).min() == 255


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("failure", ["surface", "hash"])
def test_writer_propagates_save_and_hash_failure_and_joins(tmp_path, monkeypatch, workers, failure, pillow):
    from driversi.rendering.gaussian import mtgs_artifacts as module

    def fail(*_args, **_kwargs):
        raise OSError("injected write or hash failure")

    monkeypatch.setattr(module, "_save_surface" if failure == "surface" else "_sha256", fail)
    array = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(OSError, match="injected write or hash failure"):
        with CaptureArtifactWriter(workers) as writer:
            if failure == "surface":
                writer.surface(tmp_path/"surface.npz", array, array, array.astype(bool))
            else:
                writer.png(tmp_path/"alpha.png", array.astype(np.uint8), with_hash=True)
            writer.finish()
    assert not any(t.name.startswith("mtgs-save") for t in threading.enumerate())


def test_render_exception_still_joins_writes(tmp_path, pillow):
    with pytest.raises(RuntimeError, match="render failed"):
        with CaptureArtifactWriter(4) as writer:
            future = writer.png(tmp_path/"image.png", np.zeros((64, 64, 3), np.uint8))
            raise RuntimeError("render failed")
    assert future.done()
    assert not any(t.name.startswith("mtgs-save") for t in threading.enumerate())
