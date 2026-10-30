import numpy as np
import pytest
from PIL import Image

from driversi.rendering.gaussian.image_store import PngWriter


def test_local_png_publication_exact_no_clobber(tmp_path):
    pixels = np.random.default_rng(3).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    staging = tmp_path / "staging"
    destination = tmp_path / "image.png"
    writer = PngWriter(compression=1, staging=staging)
    writer(pixels, destination)
    assert np.array_equal(np.asarray(Image.open(destination)), pixels)
    assert writer.rows[0]["bytes"] == destination.stat().st_size
    assert not list(staging.iterdir())
    assert not list(tmp_path.glob(".png-*"))
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="replace"):
        writer(pixels * 0, destination)
    assert destination.read_bytes() == before


def test_publication_failure_leaves_no_final_file(tmp_path, monkeypatch):
    import driversi.rendering.gaussian.image_store as module

    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(module.os, "link", fail)
    with pytest.raises(OSError, match="publication failed"):
        PngWriter()(np.zeros((10, 10, 3), dtype=np.uint8), tmp_path / "image.png")
    assert not (tmp_path / "image.png").exists()
    assert not list(tmp_path.glob(".png-*"))
