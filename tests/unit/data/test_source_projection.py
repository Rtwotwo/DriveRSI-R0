"""Independent pinhole geometry and source timestamp boundary checks."""

import numpy as np
import pytest

from driversi.data.scenes.source_gallery import select_source_window
from driversi.rendering.visual.source_projection import project_source_box


def camera():
    transform = np.eye(4)
    transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    transform[:3, 3] = [0, 0, 1.5]
    return np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]]), transform


def test_box_analytical_pinhole_extent_and_front_direction():
    intrinsic, transform = camera()
    lines = np.asarray(project_source_box([10, 0, 1.5, 4, 2, 2, 0], intrinsic, transform, 640, 480))
    pixels = lines.reshape(-1, 2)
    # The nearest face is 8 m from camera; 1 m half-width at fx=100 is 12.5 pixels.
    np.testing.assert_allclose(pixels.min(axis=0), [307.5, 227.5])
    np.testing.assert_allclose(pixels.max(axis=0), [332.5, 252.5])
    assert len(lines) == 12
    assert project_source_box([-10, 0, 1.5, 4, 2, 2, 0], intrinsic, transform, 640, 480) == []


def test_side_camera_uses_full_rotation_and_translation():
    intrinsic, transform = camera()
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform[:3, :3] = quarter_turn @ transform[:3, :3]
    transform[:3, 3] = [3, 4, 1.5]
    lines = project_source_box([3, 14, 1.5, 4, 2, 2, np.pi/2], intrinsic, transform, 640, 480)
    pixels = np.asarray(lines).reshape(-1, 2)
    np.testing.assert_allclose((pixels.min(axis=0)+pixels.max(axis=0))/2, [320, 240])


def test_near_crossing_and_viewport_clipping_do_not_explode():
    intrinsic, transform = camera()
    lines = np.asarray(project_source_box([.1, 0, 1.5, .3, 2, .2, 0], intrinsic, transform, 640, 480))
    assert lines.size and np.isfinite(lines).all()
    pixels = lines.reshape(-1, 2)
    assert np.all(pixels >= 0) and np.all(pixels <= [640, 480])


def test_projection_rejects_nonrigid_calibration():
    intrinsic, transform = camera()
    transform[0, 2] = 2
    with pytest.raises(ValueError, match="Invalid source"):
        project_source_box([10, 0, 1.5, 4, 2, 2, 0], intrinsic, transform, 640, 480)


def source_times():
    return [{"timestamp": t, "log_name": "a", "map_location": "m"}
            for t in [1_000_000, 1_500_100, 1_999_900, 2_500_000]]


def test_source_window_retains_native_nonuniform_time_and_rejects_padding():
    frames = source_times()
    assert select_source_window(frames, 1, 3) == frames[1:]
    with pytest.raises(ValueError, match="no padding"):
        select_source_window(frames, 1, 4)


def test_source_window_rejects_gap_and_identity_change():
    frames = source_times()
    frames[2]["timestamp"] = 2_499_999
    with pytest.raises(ValueError, match="time gap"):
        select_source_window(frames, 0, 4)
    frames = source_times()
    frames[-1]["log_name"] = "b"
    with pytest.raises(ValueError, match="boundaries"):
        select_source_window(frames, 0, 4)
