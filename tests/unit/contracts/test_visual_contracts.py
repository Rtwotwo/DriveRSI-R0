import numpy as np
import pytest

from driversi.contracts.visual import CameraCalibration, CameraRig, WorldFrame


def test_optical_front_axes_and_owned_buffers():
    camera = CameraCalibration.front()
    assert np.allclose(camera.camera_to_body[:3, :3] @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(camera.camera_to_body[:3, :3] @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(camera.camera_to_body[:3, :3] @ [0, 1, 0], [0, 0, -1])
    poses = np.zeros((1, 4))
    frame = WorldFrame(("ego",), poses, 2, "episode", "main", 0)
    poses[0, 0] = 10
    assert frame.poses[0, 0] == 0
    assert frame.sim_time == .2
    with pytest.raises(ValueError):
        frame.poses[0, 0] = 1


def test_reject_nonrigid_camera_and_mixed_resolution():
    front = CameraCalibration.front()
    transform = front.camera_to_body.copy()
    transform[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        CameraCalibration("bad", 160, 96, front.intrinsic, transform)
    with pytest.raises(ValueError, match="unique"):
        CameraRig((front, front))
    second = CameraCalibration.front(320, 192)
    second = CameraCalibration("other", second.width, second.height,
                               second.intrinsic, second.camera_to_body)
    with pytest.raises(ValueError, match="image size"):
        CameraRig((front, second))
