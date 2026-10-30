"""Source-ego annotation wireframes; pinhole diagnostics, not visible contours."""

import numpy as np

_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7))


def _clip_image(start: np.ndarray, end: np.ndarray, width: int, height: int):
    delta = end - start
    low, high = 0., 1.
    for axis, limit in enumerate((width, height)):
        if abs(delta[axis]) < 1e-12:
            if not 0 <= start[axis] <= limit:
                return None
            continue
        a, b = sorted((-start[axis] / delta[axis], (limit - start[axis]) / delta[axis]))
        low, high = max(low, a), min(high, b)
        if low > high:
            return None
    return np.r_[start + low * delta, start + high * delta].tolist()


def project_source_box(box: np.ndarray, intrinsic: np.ndarray, ego_from_camera: np.ndarray,
                       width: int, height: int, *, near: float = .1) -> list[list[float]]:
    """Transform all eight source-ego corners before projecting/clipping edges.

    Distortion is intentionally not applied. A projected line may be occluded in
    the photograph; this output is not an instance segmentation annotation.
    """
    box, intrinsic, transform = (np.asarray(value, dtype=float)
                                 for value in (box, intrinsic, ego_from_camera))
    if (box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0)
            or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or not np.allclose(intrinsic[2], [0, 0, 1])
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)
            or isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or min(width, height) <= 0 or not np.isfinite(near) or near <= 0):
        raise ValueError("Invalid source pinhole projection inputs")
    corners = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    corners *= box[3:6] / 2
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    corners[:, :2] = corners[:, :2] @ np.asarray([[cosine, sine], [-sine, cosine]])
    corners += box[:3]
    corners = (corners - transform[:3, 3]) @ transform[:3, :3]
    segments = []
    for i, j in _EDGES:
        a, b = corners[i].copy(), corners[j].copy()
        if a[2] < near and b[2] < near:
            continue
        if a[2] < near:
            a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
        if b[2] < near:
            b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
        projected = np.stack((a, b)) @ intrinsic.T
        xy = projected[:, :2] / projected[:, 2:3]
        clipped = _clip_image(xy[0], xy[1], width, height)
        if clipped is not None:
            segments.append(clipped)
    return segments
