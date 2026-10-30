"""Observed-road reprojection pilot; never changes the GS checkpoint or world.

A local plane and semantically filtered source views supply a diagnostic layer.
The output is approximate, explicitly masked, and unsuitable as certified labels.
No generative filling, extrapolation outside point support, or alpha rewriting.
"""

import numpy as np


def fit_ground_plane(points, center_xy, *, radius=18.0, max_rmse=0.12):
    points = np.asarray(points, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or center.shape != (2,)
        or not np.isfinite(points).all()
        or not np.isfinite(center).all()
        or not np.isfinite([radius, max_rmse]).all()
        or radius <= 0
        or max_rmse <= 0
    ):
        raise ValueError("Finite ground points/center and positive bounds required")
    selected = points[np.linalg.norm(points[:, :2] - center, axis=1) <= radius]
    if len(selected) < 100:
        raise ValueError("Insufficient local measured road support")
    x = np.column_stack((selected[:, :2] - center, np.ones(len(selected))))
    keep = np.ones(len(x), bool)
    for _ in range(5):
        if keep.sum() < 100 or np.linalg.matrix_rank(x[keep]) < 3:
            raise ValueError("Degenerate local ground support")
        coef = np.linalg.lstsq(x[keep], selected[keep, 2], rcond=None)[0]
        residual = np.abs(x @ coef - selected[:, 2])
        keep = residual <= max(0.04, min(0.25, np.quantile(residual, 0.8)))
    coef = np.linalg.lstsq(x[keep], selected[keep, 2], rcond=None)[0]
    rmse = float(np.sqrt(np.mean((x[keep] @ coef - selected[keep, 2]) ** 2)))
    if rmse > max_rmse or np.linalg.norm(coef[:2]) > 0.2:
        raise ValueError("Local plane residual/slope unsupported")
    coef[2] -= coef[:2] @ center
    return (
        coef,
        selected[keep],
        {
            "points": int(keep.sum()),
            "rmse_m": rmse,
            "radius_m": radius,
            "slope": float(np.linalg.norm(coef[:2])),
        },
    )


def ray_ground_intersections(rays, camera_to_world, plane, *, max_range=25.0):
    rays, pose, plane = map(lambda x: np.asarray(x, dtype=float), (rays, camera_to_world, plane))
    if (
        rays.ndim != 3
        or rays.shape[-1] != 3
        or pose.shape != (3, 4)
        or plane.shape != (3,)
        or not all(np.isfinite(x).all() for x in (rays, pose, plane))
        or not np.allclose(pose[:, :3].T @ pose[:, :3], np.eye(3), atol=1e-5)
        or not np.isfinite(max_range)
        or max_range <= 0
    ):
        raise ValueError("Invalid rays, rigid camera or plane")
    directions = rays @ pose[:, :3].T
    normal = np.array([-plane[0], -plane[1], 1.0])
    denom = directions @ normal
    t = np.divide(
        plane[2] - pose[:, 3] @ normal,
        denom,
        out=np.full(denom.shape, -1.0),
        where=np.abs(denom) > 1e-8,
    )
    points = pose[:, 3] + t[..., None] * directions
    distance = np.linalg.norm(points - pose[:, 3], axis=-1)
    valid = (t > 0) & (distance <= max_range) & (denom < -0.05)
    return points, valid


def complete_observed_ground(rgb, alpha, depth, lens_valid, camera, ground_xyz, sources):
    """Physical distorted target lens; pinhole source images use half-pixel K.

    Two separate source cameras must agree within 30/255 per RGB channel. Pixels
    require measured road support within .6 m and a downward ray <=25 m. Original
    alpha>=.5 and near occluders are retained. Sources must already exclude ego,
    non-road semantics and eroded label boundaries. Validity is an approximation,
    not a physical occlusion certificate, so training admission stays false.
    """
    import cv2
    from scipy.spatial import cKDTree

    rgb, alpha, depth, lens_valid = map(np.asarray, (rgb, alpha, depth, lens_valid))
    h, w = rgb.shape[:2]
    if (
        rgb.shape != (h, w, 3)
        or rgb.dtype != np.uint8
        or any(a.shape != (h, w) for a in (alpha, depth, lens_valid))
        or lens_valid.dtype != bool
        or not np.isfinite(alpha).all()
        or not np.isfinite(depth).all()
        or (alpha < 0).any()
        or (alpha > 1).any()
        or (depth < 0).any()
    ):
        raise ValueError("Invalid source render channels")
    pose = np.asarray(camera["camera_to_world"], float)
    plane, support, fit = fit_ground_plane(ground_xyz, pose[:2, 3])
    y, x = np.mgrid[:h, :w]
    pixel = np.stack((x + 0.5, y + 0.5), -1).reshape(-1, 1, 2)
    normalized = cv2.undistortPointsIter(
        pixel,
        np.asarray(camera["K"], float),
        np.asarray(camera["distortion"], float),
        None,
        None,
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
    ).reshape(h, w, 2)
    rays = np.dstack((normalized[..., 0], -normalized[..., 1], -np.ones((h, w))))
    points, valid = ray_ground_intersections(rays, pose, plane)
    nearest = cKDTree(support[:, :2]).query(points[..., :2].reshape(-1, 2))[0].reshape(h, w)
    # Expected GS depth uses camera Z, not Euclidean ray length.
    target_depth = -((points - pose[:, 3]) @ pose[:, :3])[..., 2]
    valid &= (nearest <= 0.6) & lens_valid & (alpha < 0.5)
    # Low opacity is not proof that foreground geometry is absent. Retain
    # every positive near-depth sample, even if it is sparsely reconstructed.
    valid &= ((depth == 0) & (alpha == 0)) | (depth >= target_depth - 0.5)
    colors, masks, seen = [], [], set()
    for source in sources:
        if source["id"] in seen:
            raise ValueError("Duplicate observation cannot supply independent support")
        seen.add(source["id"])
        sp, sk = np.asarray(source["pose"], float), np.asarray(source["K"], float)
        image, road = np.asarray(source["rgb"]), np.asarray(source["road"])
        if (
            sp.shape != (3, 4)
            or sk.shape != (3, 3)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or road.shape != image.shape[:2]
            or road.dtype != bool
            or not np.isfinite(sp).all()
            or not np.isfinite(sk).all()
        ):
            raise ValueError("Invalid observed ground source")
        local = (points - sp[:, 3]) @ sp[:, :3]
        optical = local * [1, -1, -1]
        projected = optical @ sk.T
        uv = projected[..., :2] / np.maximum(projected[..., 2:3], 1e-6) - 0.5
        mx, my = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
        mask = (
            cv2.remap(
                road.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            >= 1 - 1e-6
        )
        mask &= (optical[..., 2] > 0.5) & (np.linalg.norm(local, axis=-1) < 50)
        colors.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        masks.append(mask)
    repaired = rgb.copy()
    filled = np.zeros((h, w), bool)
    owner = np.full((h, w), -1, np.int16)
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            # Source selection orders nearest observed cameras first. Texture is
            # copied from one source, not averaged into a blurry blend.
            consistent = np.max(np.abs(colors[i].astype(float) - colors[j]), axis=-1) <= 30
            use = valid & ~filled & masks[i] & masks[j] & consistent
            repaired[use] = colors[i][use]
            filled[use] = True
            owner[use] = i
    return (
        repaired,
        filled,
        owner,
        {
            "kind": "observed-local-ground-preview-v1",
            "plane": plane.tolist(),
            "plane_fit": fit,
            "filled_pixels": int(filled.sum()),
            "lens_valid_low_alpha_pixels": int(((alpha < 0.5) & lens_valid).sum()),
            "residual_low_alpha_unfilled_pixels": int(((alpha < 0.5) & lens_valid & ~filled).sum()),
            "trainable": False,
            "original_alpha_unchanged": True,
            "limits": [
                "Local planar and semantic approximation; no heldout photometric proof",
                "Source road observations do not certify current dynamic occlusion",
                "Does not reconstruct missing pedestrians, vehicles or general novel views",
            ],
        },
    )
