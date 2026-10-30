import numpy as np
import pytest

from driversi.rendering.gaussian.source_refinement import object_rectangles


def frame(box):
    return dict(ego2global=np.eye(4), gt_boxes=[box])


def test_boxes_behind_camera_are_not_fabricated_on_image():
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    assert object_rectangles(frame([0, 0, -5, 2, 2, 2, 0]), np.eye(4), k, 100, 100) == []


def test_near_plane_crossing_is_conservatively_clipped():
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    regions = object_rectangles(frame([0, 0, .15, .2, .2, .2, 0]), np.eye(4), k, 100, 100)
    assert regions == [[0, 0, 100, 100]]


def test_far_object_excludes_bounded_region_and_invalid_box_rejects():
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    region = object_rectangles(frame([0, 0, 20, 2, 2, 2, 0]), np.eye(4), k, 100, 100)[0]
    assert 0 < region[0] < 50 < region[2] < 100
    with pytest.raises(ValueError, match="Finite source"):
        object_rectangles(frame([0, 0, np.nan, 2, 2, 2, 0]), np.eye(4), k, 100, 100)
