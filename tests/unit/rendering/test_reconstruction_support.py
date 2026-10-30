import numpy as np
import pytest

from driversi.rendering.gaussian.reconstruction_support import reconstruction_loss_support

cv2 = pytest.importorskip("cv2")


def test_no_retained_window_reads_missing_border_or_actor():
    y, x = np.mgrid[:40, :50].astype(np.float32)
    x[:, :6] = -10
    rect = (20, 10, 27, 22)
    support = reconstruction_loss_support(x, y, source_size=(50, 40), excluded_rectangles=[rect])
    assert support.any()
    forbidden = x < 0
    forbidden[10:22, 20:27] = True
    for cy, cx in zip(*np.where(support[..., 0])):
        assert not forbidden[cy-5:cy+6, cx-5:cx+6].any()
        assert 5 <= cy < 35 and 5 <= cx < 45


def test_support_rejects_nonfinite_maps_and_bad_geometry():
    y, x = np.mgrid[:20, :20].astype(np.float32)
    with pytest.raises(ValueError):
        reconstruction_loss_support(x*np.nan, y, source_size=(20, 20))
    with pytest.raises(ValueError):
        reconstruction_loss_support(x, y, source_size=(20, 20), excluded_rectangles=[(-1, 1, 5, 5)])
    with pytest.raises(ValueError):
        reconstruction_loss_support(x, y, source_size=(20, 20), ssim_window_size=4)
