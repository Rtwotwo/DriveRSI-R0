"""Conservative spatial support for bounded GS refinement, not object identity."""
from __future__ import annotations

import numpy as np


def obb_support_indices(means, scales, quaternions, center, axes, half_size, *, sigma=3.0):
    """Return indices whose ellipsoid AABB in the OBB frame overlaps the box.

    scales are positive standard deviations; quaternions use w,x,y,z. axes
    contains box axes as columns in world coordinates. This broad-phase test
    can include false positives; neither visibility nor semantic ownership is
    established. No scene/object annotation is enlarged or changed.
    """
    means, scales, quaternions = [np.asarray(x, dtype=np.float64) for x in
                                  (means, scales, quaternions)]
    center, axes, half_size = [np.asarray(x, dtype=np.float64) for x in
                               (center, axes, half_size)]
    n = len(means)
    if (means.shape != (n, 3) or scales.shape != (n, 3)
            or quaternions.shape != (n, 4) or center.shape != (3,)
            or axes.shape != (3, 3) or half_size.shape != (3,)):
        raise ValueError("Invalid Gaussian/OBB shapes")
    if not all(np.isfinite(x).all() for x in (means, scales, quaternions, center, axes, half_size)):
        raise ValueError("Non-finite geometry")
    if sigma <= 0 or not np.isfinite(sigma) or np.any(scales <= 0) or np.any(half_size <= 0):
        raise ValueError("Support sizes must be positive")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6) or np.linalg.det(axes) <= 0:
        raise ValueError("Box axes must be a proper rotation")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Zero quaternion")
    local = (means - center) @ axes
    # Circumscribed sphere cheaply excludes far Gaussian means before covariance.
    broad = np.flatnonzero(np.all(np.abs(local) <= half_size + sigma * scales.max(1)[:, None], axis=1))
    q = quaternions[broad] / norms[broad, None]
    w, x, y, z = q.T
    rotations = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                          2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                          2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)
    basis = np.einsum('ij,njk->nik', axes.T, rotations)
    extent = sigma * np.sqrt(np.sum((basis * scales[broad, None, :])**2, axis=2))
    return broad[np.all(np.abs(local[broad]) <= half_size + extent, axis=1)]
