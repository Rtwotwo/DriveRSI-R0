import numpy as np
import pytest

from driversi.rendering.gaussian.gaussian_support import obb_support_indices


def test_center_only_misses_crossing_support():
    means = np.array([[0., 0., 0.], [1.2, 0., 0.], [1.5, 0., 0.]])
    ids = obb_support_indices(means, np.full((3, 3), .1), np.tile([1., 0., 0., 0.], (3, 1)),
                              np.zeros(3), np.eye(3), np.ones(3))
    assert ids.tolist() == [0, 1]


def test_rotation_and_anisotropic_covariance():
    # Major Gaussian axis rotates to y; x support remains tiny.
    q = np.array([[2**-.5, 0., 0., 2**-.5]] * 2)
    ids = obb_support_indices([[1.3, 0., 0.], [0., 1.3, 0.]], [[.5, .01, .01]] * 2, q,
                              np.zeros(3), np.eye(3), np.ones(3))
    assert ids.tolist() == [1]


def test_global_rigid_transform_invariance():
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    points = np.array([[0., 1.2, 0.], [2., 2., 0.]])
    result = obb_support_indices(points @ rot.T + 7, [[.1]*3]*2,
                                [[2**-.5, 0., 0., 2**-.5]]*2,
                                np.ones(3)*7, rot, np.ones(3))
    assert result.tolist() == [0]


@pytest.mark.parametrize('q', [[[0., 0., 0., 0.]], [[float('nan'), 0., 0., 0.]]])
def test_invalid_quaternions_rejected(q):
    with pytest.raises(ValueError):
        obb_support_indices([[0., 0., 0.]], [[.1]*3], q, np.zeros(3), np.eye(3), np.ones(3))


def test_empty_scene():
    assert len(obb_support_indices(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 4)),
                                   np.zeros(3), np.eye(3), np.ones(3))) == 0
