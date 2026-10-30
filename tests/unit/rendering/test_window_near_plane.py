import itertools

import numpy as np
import pytest

from driversi.rendering.diagnostics.window_review import clipped_box_projection, project_camera


def evidence():
    return dict(
        camera_to_world=np.eye(4)[:3].tolist(),
        distorted_intrinsic=[[100, 0, 100], [0, 100, 50], [0, 0, 1]],
        distortion=[0, 0, 0, 0, 0],
    )


def box(z):
    return np.asarray(list(itertools.product((-0.5, 0.5), (-0.5, 0.5), z)))


def test_front_box_preserves_original_projected_corners():
    corners = box((3, 4))
    p = clipped_box_projection(corners, evidence())
    uv, _ = project_camera(corners, evidence())
    points = np.asarray(p["pixel_edges"]).reshape(-1, 2)
    assert p["projection_supported"] and p["clipped_edges"] == 0
    for point in uv:
        assert np.linalg.norm(points - point, axis=1).min() < 1e-10
    assert not p["appearance_certified"]


def test_partial_box_is_clipped_instead_of_dropped():
    corners = box((-1, 2))
    p = clipped_box_projection(corners, evidence())
    assert p["projection_supported"] and p["clipped_edges"] == 4
    assert np.isfinite(p["pixel_edges"]).all()
    assert min(p["original_camera_depth"]) < 0
    assert not p["appearance_certified"]


def test_real_failure_depths_under_ten_centimeters_remain_reviewable():
    p = clipped_box_projection(box((0.0107189, 4.629796)), evidence())
    assert p["projection_supported"] and p["clipped_edges"] == 4
    assert np.isfinite(p["pixel_edges"]).all()


def test_behind_box_has_no_supported_projection_and_nan_is_rejected():
    assert not clipped_box_projection(box((-2, -1)), evidence())["projection_supported"]
    corners = box((1, 2))
    corners[0, 0] = np.nan
    with pytest.raises(ValueError):
        clipped_box_projection(corners, evidence())


def test_real_distorted_camera_visible_edge_error_under_point_three_pixels():
    import cv2

    corners = np.array(
        [
            [-1.0481812205389236, 1.9490008528037506, 4.629796280720979],
            [-1.0145327881615127, 0.3417449448867093, 4.607116768217099],
            [0.09667959742767404, 1.9958288864380753, 3.009749910953202],
            [0.13032802980508576, 0.388572978521034, 2.9870703984493217],
            [-5.258009093614195, 1.90286513334654, 1.6534447902502387],
            [-5.224360661236793, 0.2956092254294986, 1.6307652777463557],
            [-4.113148275647593, 1.9496931669808648, 0.03339842048246505],
            [-4.079499843270189, 0.34243725906382355, 0.010718907978583143],
        ]
    )
    ev = evidence()
    ev.update(
        distorted_intrinsic=[
            [774.3051552094769, 0, 485.22992892604515],
            [0, 775.7631759676525, 289.2068893579094],
            [0, 0, 1],
        ],
        distortion=[
            -0.32835500713595794,
            0.10539508935644087,
            -0.002809684095076108,
            0.0005862805521374153,
            0.0,
        ],
    )
    result = clipped_box_projection(corners, ev)
    segments = iter(result["pixel_edges"])
    errors = []
    for i in range(8):
        for j in range(i + 1, 8):
            if i ^ j not in (1, 2, 4):
                continue
            a, b = corners[i].copy(), corners[j].copy()
            if max(a[2], b[2]) < 0.1:
                continue
            if a[2] < 0.1:
                a = a + (0.1 - a[2]) / (b[2] - a[2]) * (b - a)
            if b[2] < 0.1:
                b = b + (0.1 - b[2]) / (a[2] - b[2]) * (a - b)
            line = np.asarray(next(segments))
            dense = a[None, :] + np.linspace(0, 1, 4097)[:, None] * (b - a)[None, :]
            uv = cv2.projectPoints(
                dense,
                np.zeros(3),
                np.zeros(3),
                np.asarray(ev["distorted_intrinsic"], dtype=float),
                np.asarray(ev["distortion"]),
            )[0].reshape(-1, 2)
            visible = uv[(uv[:, 0] >= 0) & (uv[:, 0] < 960) & (uv[:, 1] >= 0) & (uv[:, 1] < 540)]
            if not len(visible):
                continue
            delta = line[1:] - line[:-1]
            diff = visible[:, None, :] - line[:-1][None, :, :]
            t = np.clip(
                (diff * delta[None, :, :]).sum(-1) / np.maximum((delta * delta).sum(-1), 1e-20),
                0,
                1,
            )
            distance = np.linalg.norm(diff - t[:, :, None] * delta[None, :, :], axis=-1).min(axis=1)
            errors.extend(distance.tolist())
    assert len(errors) > 10000
    assert max(errors) < 0.3
    assert not result["appearance_certified"]
