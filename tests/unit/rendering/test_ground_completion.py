import numpy as np
import pytest

from driversi.rendering.gaussian.ground_completion import fit_ground_plane, ray_ground_intersections


def test_plane_fit_from_measurements_and_upward_ray_rejection():
    x, y = np.meshgrid(np.linspace(-10, 10, 30), np.linspace(-10, 10, 30))
    pts = np.column_stack((x.ravel(), y.ravel(), (0.02 * x + 0.03 * y + 0.5).ravel()))
    plane, support, stats = fit_ground_plane(pts, [0, 0])
    np.testing.assert_allclose(plane, [0.02, 0.03, 0.5], atol=1e-9)
    assert stats["rmse_m"] < 1e-9 and len(support) > 100
    pose = np.column_stack((np.eye(3), [0, 0, 2]))
    points, valid = ray_ground_intersections(
        np.array([[[0, 0, -1], [0, 0, 1], [1, 0, 0]]]), pose, plane
    )
    assert valid.tolist() == [[True, False, False]]
    np.testing.assert_allclose(points[0, 0], [0, 0, 0.5])


@pytest.mark.parametrize("kind", ["few", "nan", "steep", "line"])
def test_unsupported_ground_fails(kind):
    x, y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
    pts = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    if kind == "few":
        pts = pts[:10]
    elif kind == "nan":
        pts[0, 0] = np.nan
    elif kind == "steep":
        pts[:, 2] = pts[:, 0]
    else:
        pts[:, 1] = 0
    with pytest.raises(ValueError):
        fit_ground_plane(pts, [0, 0])


def test_observation_supported_completion_preserves_opaque_and_inputs():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from driversi.rendering.gaussian.ground_completion import complete_observed_ground

    h, w = 32, 48
    pose = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 2.0]], float)
    k = np.array([[30, 0, 24], [0, 30, 12], [0, 0, 1.0]], float)
    camera = dict(camera_to_world=pose, K=k, distortion=np.zeros(5))
    x, y = np.meshgrid(np.linspace(-2, 25, 110), np.linspace(-12, 12, 100))
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    rgb = np.full((h, w, 3), 17, np.uint8)
    alpha = np.zeros((h, w))
    alpha[:, : w // 2] = 0.9
    depth = np.full((h, w), 100.0)
    valid = np.ones((h, w), bool)
    sources = []
    for i in range(2):
        sp = pose.copy()
        sp[0, 3] = -1 - i
        sources.append(
            dict(id=str(i), pose=sp, K=k, rgb=np.full((h, w, 3), 80, np.uint8), road=valid.copy())
        )
    result, filled, owner, report = complete_observed_ground(
        rgb, alpha, depth, valid, camera, ground, sources
    )
    assert filled.any() and not filled[:, : w // 2].any()
    np.testing.assert_array_equal(result[~filled], rgb[~filled])
    assert (result[filled] == 80).all() and (rgb == 17).all()
    assert (alpha[:, : w // 2] == 0.9).all() and not report["trainable"]
    # Sparse foreground is still geometry: never paint road over a positive
    # near-depth sample just because its alpha falls below the former .05 cut.
    for opacity in (0.0, 0.01, 0.049, 0.05, 0.49):
        near, near_filled, _, _ = complete_observed_ground(
            rgb, np.full_like(alpha, opacity), np.full_like(depth, 0.1),
            valid, camera, ground, sources,
        )
        assert not near_filled.any()
        np.testing.assert_array_equal(near, rgb)
    sources[1]["id"] = "0"
    with pytest.raises(ValueError, match="Duplicate"):
        complete_observed_ground(rgb, alpha, depth, valid, camera, ground, sources)
