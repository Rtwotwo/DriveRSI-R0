"""Bounded reconstruction geometry; these facts are not image admission.

Released road maps use Sim2 p_grid = s * (R @ p_reconstruction + t).
Unlike the upstream nearest/clamping alternatives, all interpolation support
must be finite and inside the map. A planar simulation is lifted by preserving
source clearance while using the road-height difference at the new position.
"""

import json
from pathlib import Path

import numpy as np


class UnsupportedTerrain(ValueError):
    pass


def terrain_failure_evidence(error):
    """Keep the failed geometry proof; search exhaustion is not a second road.

    This only explains an existing rejection. It grants no height support and
    intentionally preserves the original rejection reason and thresholds.
    """
    import copy

    evidence = copy.deepcopy(getattr(error, "evidence", {}))
    witnesses, unresolved = [], []

    def visit(value):
        if isinstance(value, dict):
            check = value.get("secondary_layer_check")
            if isinstance(check, dict):
                if check.get("status") == "ambiguous" and isinstance(check.get("supported_secondary"), dict):
                    witnesses.append(check["supported_secondary"])
                elif check.get("status") == "unresolved":
                    unresolved.append(check.get("reason", "unspecified"))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    diagnosis = (
        "confirmed_competing_surface_fit" if witnesses else
        "secondary_surface_not_resolved" if unresolved else
        "independent_traversals_disagree" if "disagree" in str(error) else
        "surface_support_not_established"
    )
    return dict(status="unsupported", reason=str(error), diagnosis=diagnosis,
                evidence=evidence, confirmed_secondary_fits=len(witnesses),
                unresolved_secondary_reasons=sorted(set(unresolved)),
                interpretation="geometric_support_failure_not_proof_of_a_physical_overpass")


class RoadHeightMap:
    def __init__(self, grid, transform):
        self.grid = np.asarray(grid, dtype=float)
        self.rotation = np.asarray(transform["R"], dtype=float)
        self.translation = np.asarray(transform["t"], dtype=float)
        self.scale = float(transform["s"])
        if (
            self.grid.ndim != 2
            or min(self.grid.shape) < 2
            or self.rotation.shape != (2, 2)
            or self.translation.shape != (2,)
            or not np.isfinite([*self.rotation.ravel(), *self.translation, self.scale]).all()
            or self.scale <= 0
            or not np.allclose(self.rotation.T @ self.rotation, np.eye(2), atol=1e-8)
            or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-8)
        ):
            raise ValueError("Malformed released terrain map or Sim2 transform")

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        return cls(
            np.load(folder / "road_height_map.npy", allow_pickle=False),
            json.loads((folder / "sim2.json").read_text()),
        )

    def sample(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite reconstruction xy required")
        uv = self.scale * (self.rotation @ xy + self.translation)
        col, row = map(int, np.floor(uv))
        if row < 0 or col < 0 or row + 1 >= self.grid.shape[0] or col + 1 >= self.grid.shape[1]:
            raise UnsupportedTerrain("Outside finite height-map interpolation domain")
        values = self.grid[row : row + 2, col : col + 2]
        if not np.isfinite(values).all():
            raise UnsupportedTerrain("Incomplete finite height-map interpolation support")
        x, y = uv - [col, row]
        value = (
            values[0, 0] * (1 - x) * (1 - y)
            + values[0, 1] * x * (1 - y)
            + values[1, 0] * (1 - x) * y
            + values[1, 1] * x * y
        )
        return {
            "height_m": float(value),
            "grid_xy": uv.tolist(),
            "support_cells": [[row, col], [row, col + 1], [row + 1, col], [row + 1, col + 1]],
            "support_min_m": float(values.min()),
            "support_max_m": float(values.max()),
        }

    def lift(self, source_center, planar_translation):
        center, delta = (
            np.asarray(source_center, dtype=float),
            np.asarray(planar_translation, dtype=float),
        )
        if center.shape != (3,) or delta.shape != (3,) or not np.isfinite([*center, *delta]).all():
            raise ValueError("Finite source center and reconstruction translation required")
        source = self.sample(center[:2])
        target = self.sample((center + delta)[:2])
        corrected = delta.copy()
        corrected[2] = target["height_m"] - source["height_m"]
        return corrected, {
            "status": "supported",
            "method": "finite_bilinear_source_clearance_v1",
            "source": source,
            "target": target,
            "source_clearance_m": float(center[2] - source["height_m"]),
            "vertical_correction_m": float(corrected[2] - delta[2]),
            "vertical_displacement_m": float(corrected[2]),
            "orientation": "source_pitch_roll_with_transported_physical_yaw",
            "scope": "vertical_road_support_not_surface_orientation_certificate",
        }


def background_box_support(means, opacity, box, ego_transform):
    """Count native background Gaussians inside a source annotation, no dilation.

    This proves spatial support, not semantic identity or photo quality. The
    caller must separately compare real/source/novel-view crops and immobility.
    """
    means, opacity = np.asarray(means), np.asarray(opacity).reshape(-1)
    box, transform = np.asarray(box, dtype=float), np.asarray(ego_transform, dtype=float)
    if (
        means.ndim != 2
        or means.shape[1] != 3
        or opacity.shape != (len(means),)
        or box.shape != (7,)
        or transform.shape != (4, 4)
        or not np.isfinite(box).all()
        or not np.isfinite(transform).all()
        or np.any(box[3:6] <= 0)
    ):
        raise ValueError("Finite source 3D annotation and background cloud required")
    c, s = np.cos(box[6]), np.sin(box[6])
    local_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = transform[:3, :3] @ local_rotation
    center = transform[:3, :3] @ box[:3] + transform[:3, 3]
    local = (means - center) @ rotation
    inside = np.all(np.abs(local) <= box[3:6] / 2, axis=1) & (opacity > 0.05)
    points = means[inside]
    return {
        "center_reconstruction": center.tolist(),
        "rotation_reconstruction": rotation.tolist(),
        "dimensions_lwh": box[3:6].tolist(),
        "gaussian_count": int(inside.sum()),
        "opacity_sum": float(opacity[inside].sum()),
        "point_bounds": [points.min(0).tolist(), points.max(0).tolist()] if len(points) else None,
        "scope": "background_geometry_support_not_semantic_or_visual_certificate",
    }


def physical_box_corners(state, dimensions, scene_origin, asset_origin, source):
    """Transport the unchanged physical OBB into the refined reconstruction frame."""
    import itertools

    state = np.asarray(state, dtype=float)
    dims = np.asarray(dimensions, dtype=float)
    original = np.asarray(source["ego2global_original"], dtype=float)
    refined = np.asarray(source["ego2global"], dtype=float)
    if (
        state.shape[0] < 4
        or dims.shape != (3,)
        or np.any(dims <= 0)
        or original.shape != (4, 4)
        or refined.shape != (4, 4)
    ):
        raise ValueError("Finite physical box and source registration required")
    center = state[:3] + np.asarray(scene_origin) - np.asarray(asset_origin)
    c, s = np.cos(state[3]), np.sin(state[3])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * dims / 2
    corners = corners @ rotation.T + center
    transport = refined @ np.linalg.inv(original)
    result = corners @ transport[:3, :3].T + transport[:3, 3]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite physical box registration")
    return result


def box_frustum_support(corners, camera_to_world, ray_bounds):
    """Conservative OBB/frustum half-space exclusion; no occlusion assumption.

    ray_bounds must enclose every distorted output-pixel ray. An intersecting
    box remains unresolved; this never treats a small/distant object as absent.
    """
    corners = np.asarray(corners, dtype=float)
    pose = np.asarray(camera_to_world, dtype=float)
    bounds = np.asarray(ray_bounds, dtype=float)
    if (
        corners.shape != (8, 3)
        or pose.shape != (3, 4)
        or bounds.shape != (4,)
        or not np.isfinite([*corners.ravel(), *pose.ravel(), *bounds]).all()
        or not bounds[0] < bounds[1]
        or not bounds[2] < bounds[3]
    ):
        raise ValueError("Finite box, camera and enclosing ray bounds required")
    camera = (corners - pose[:, 3]) @ pose[:, :3]
    xmin, xmax, ymin, ymax = bounds
    planes = np.array([[0, 0, 1], [1, 0, -xmin], [-1, 0, xmax], [0, 1, -ymin], [0, -1, ymax]])
    maxima = (camera @ planes.T).max(axis=0)
    excluded = np.flatnonzero(maxima < -1e-6).tolist()
    return {
        "outside_frustum": bool(excluded),
        "separating_planes": excluded,
        "plane_support_maxima": maxima.tolist(),
        "ray_bounds": bounds.tolist(),
        "method": "all_obb_corners_halfspace_separation_v1",
        "scope": "camera_frustum_only_not_occlusion_or_appearance_certificate",
    }


def background_physical_box_support(means, opacity, corners):
    """Measure static background support in the actual physical OBB, not a drifting annotation."""
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("Finite physical OBB corners required")
    axes = np.stack(
        (corners[4] - corners[0], corners[2] - corners[0], corners[1] - corners[0]), axis=1
    )
    dims = np.linalg.norm(axes, axis=0)
    if np.any(dims <= 0):
        raise ValueError("Positive physical box dimensions required")
    transform = np.eye(4)
    transform[:3, :3] = axes / dims
    transform[:3, 3] = corners.mean(0)
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("Orthogonal physical OBB required")
    return background_box_support(means, opacity, [0, 0, 0, *dims, 0], transform)
