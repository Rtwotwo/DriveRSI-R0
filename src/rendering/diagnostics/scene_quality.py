"""Explicit low-detail GS experiment rules; no geometry or scoring exemptions."""

import math
import numpy as np

PROFILE = "experimental_scene24_v1"
ALLOWED_ISSUES = {"limited_texture_blur", "small_static_appearance_uncertain"}
STATIC_KINDS = {"traffic_cone", "barrier", "generic_object", "construction_zone_sign", "czone_sign"}


def hole_fractions(mask):
    """Binary independently reviewed void mask; report full-frame connected area."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isin(mask, [0, 255]).all():
        raise ValueError("Binary 0/255 void mask required")
    seen = mask.astype(bool)
    total = int(seen.sum())
    largest = 0
    for y, x in np.argwhere(seen):
        if not seen[y, x]:
            continue
        todo = [(int(y), int(x))]
        seen[y, x] = False
        size = 0
        while todo:
            a, b = todo.pop()
            size += 1
            for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if 0 <= c < seen.shape[0] and 0 <= d < seen.shape[1] and seen[c, d]:
                    seen[c, d] = False
                    todo.append((c, d))
        largest = max(largest, size)
    return dict(total_fraction=total / mask.size, largest_component_fraction=largest / mask.size)


CRITICAL_CHECKS = (
    "road_geometry_consistent",
    "lane_markings_consistent",
    "on_road_vehicles_consistent",
    "occlusion_consistent",
)


def frame_allowed(review, holes, *, profile=PROFILE):
    if profile not in (PROFILE, "experimental_scene25_v1", "experimental_egoview25_v1"):
        raise ValueError("Unsupported limited-appearance profile")
    if profile in ("experimental_scene25_v1", "experimental_egoview25_v1") and not all(
        review.get(key) is True for key in CRITICAL_CHECKS
    ):
        return False
    return (
        review.get("verdict") in ("pass", "experimental_pass")
        and isinstance(review.get("issues"), list)
        and set(review["issues"]) <= ALLOWED_ISSUES
        and review.get("dynamic_objects_consistent") is True
        and review.get("multiview_temporal_consistent") is True
        and review.get("no_severe_geometry_misalignment") is True
        and holes["total_fraction"] <= 0.05
        and holes["largest_component_fraction"] <= 0.02
    )


def small_static_appearance_allowed(kind, evidence, binding, projected_fraction):
    return (
        kind in STATIC_KINDS
        and evidence.get("physical_immobile") is True
        and evidence.get("physical_background_geometry", {}).get("gaussian_count", 0) > 0
        and binding.get("physical_box_alignment_confirmed") is True
        and binding.get("appearance_uncertainty_accepted_for_experiment") is True
        and math.isfinite(projected_fraction)
        and 0 <= projected_fraction <= 0.005
    )


def projected_box_fraction(corners, camera):
    """Conservative box rectangle from native optical projection; near-plane ambiguity rejects."""
    points = np.asarray(corners, float)
    pose = np.asarray(camera["camera_to_world"], float)
    if points.shape != (8, 3) or pose.shape != (3, 4):
        return float("inf")
    xyz = (points - pose[:3, 3]) @ pose[:3, :3]
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0.15):
        return float("inf")

    # Interval bounds include nonlinear distorted edges, not just eight corners.
    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    def mul(a, b):
        values = [u * v for u in a for v in b]
        return min(values), max(values)

    def scale(a, v):
        return mul(a, (v, v))

    def square(a):
        return (0 if a[0] <= 0 <= a[1] else min(v * v for v in a), max(v * v for v in a))

    xy = xyz[:, :2] / xyz[:, 2:]
    x = (min(xy[:, 0]), max(xy[:, 0]))
    y = (min(xy[:, 1]), max(xy[:, 1]))
    xx, yy = square(x), square(y)
    r = add(xx, yy)
    r2 = mul(r, r)
    r3 = mul(r2, r)
    k1, k2, p1, p2, k3 = camera["distortion"]
    radial = add((1, 1), add(scale(r, k1), add(scale(r2, k2), scale(r3, k3))))
    u = add(mul(x, radial), add(scale(mul(x, y), 2 * p1), scale(add(r, scale(xx, 2)), p2)))
    v = add(mul(y, radial), add(scale(add(r, scale(yy, 2)), p1), scale(mul(x, y), 2 * p2)))
    k = np.asarray(camera["distorted_intrinsic"])
    w, h = camera.get("output_image_size", camera["native_calibration_size"])
    px = np.clip(add(add(scale(u, k[0, 0]), scale(v, k[0, 1])), (k[0, 2], k[0, 2])), 0, w)
    py = np.clip(add(scale(v, k[1, 1]), (k[1, 2], k[1, 2])), 0, h)
    return float(np.ptp(px) * np.ptp(py) / (w * h))
