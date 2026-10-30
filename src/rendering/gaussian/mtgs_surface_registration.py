"""Audited cross-traversal vertical registration for measured GS ground support.

Registration never changes the source Gaussian model or the fitted source road.
The declared transform is recomputed from raw measured clouds, using calibration
planes disjoint from the declared physical branch and buffered spatial holdouts.
"""
import json
from pathlib import Path

import numpy as np

from .mtgs_road_surface import (
    IndependentlyValidatedRoadSurface,
    MeasuredRoadSurface,
    _file_sha,
)
from .worldengine_geometry import UnsupportedTerrain

POLICY = {
    "method": "A13_v1", "min_calibration_planes": 64,
    "query_exclusion_m": 5.0, "surface_radius_m": 2.0,
    "holdout_neighborhood_gap_m": 4.0, "holdout_folds": 4,
    "min_training_planes_per_fold": 16, "min_test_planes_per_fold": 8,
    "holdout_max_residual_m": .08, "holdout_max_rmse_m": .03,
    "fit_surface_method": "A11-v2",
}


def derive_rigid_registration(xy, differences):
    xy, differences = np.asarray(xy, float), np.asarray(differences, float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or differences.shape != (len(xy),)
            or len(xy) < 3 or not np.isfinite(xy).all() or not np.isfinite(differences).all()):
        raise ValueError("Finite corresponding static plane centers required")
    center = xy.mean(0)
    design = np.column_stack((xy-center, np.ones(len(xy))))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Static plane centers must constrain both axes")
    coef = np.linalg.lstsq(design, differences, rcond=None)[0]
    normal = np.r_[-coef[:2], 1.]
    normal /= np.linalg.norm(normal)
    axis = np.cross(normal, [0., 0., 1.])
    sine, cosine = np.linalg.norm(axis), float(normal[2])
    if sine < 1e-14:
        rotation = np.eye(3)
    else:
        a = axis / sine
        cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        rotation = np.eye(3) + sine * cross + (1-cosine)*(cross@cross)
    pivot = np.r_[center, 0.]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot - rotation@pivot - [0, 0, coef[2]]
    return transform, coef, center


def require_derived_transform(declared, derived):
    matrix = np.asarray(declared, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix, derived, rtol=0, atol=1e-10):
        raise ValueError("Declared registration is not the transform derived from measured evidence")


def buffered_spatial_holdouts(rows):
    xy = np.asarray([x["xy"] for x in rows], float)
    delta = np.asarray([x["moving"]["height_m"]-x["reference"]["height_m"] for x in rows])
    axis = int(np.argmax(np.ptp(xy, axis=0)))
    order = np.argsort(xy[:, axis], kind="stable")
    groups = np.empty(len(xy), int)
    for fold, indices in enumerate(np.array_split(order, 4)):
        groups[indices] = fold
    records = []
    for fold in range(4):
        test = groups == fold
        distances = np.sqrt(((xy[:, None]-xy[test][None])**2).sum(-1)).min(1)
        train = (~test) & (distances > 4.+1e-9)
        if train.sum() < 16 or test.sum() < 8:
            raise ValueError("Insufficient independent buffered calibration holdout")
        transform, _, _ = derive_rigid_registration(xy[train], delta[train])
        rotation, translation = transform[:3, :3], transform[:3, 3]
        residuals = []
        for i in np.flatnonzero(test):
            source = rows[i]["moving"]
            coefficients = np.asarray(source["plane_centered_reconstruction"])
            normal = np.r_[-coefficients[:2], 1.]
            offset = source["height_m"]-coefficients[:2]@xy[i]
            target_normal = rotation@normal
            target_offset = offset+target_normal@translation
            height = (target_offset-target_normal[:2]@xy[i])/target_normal[2]
            residuals.append(float(height-rows[i]["reference"]["height_m"]))
        residuals = np.asarray(residuals)
        maximum, rmse = float(abs(residuals).max()), float(np.sqrt(np.mean(residuals**2)))
        if maximum > .08 or rmse > .03:
            raise ValueError("Measured rigid calibration failed unchanged spatial holdout limits")
        records.append({"fold": fold, "training_count": int(train.sum()),
            "test_count": int(test.sum()), "train_test_center_gap_m": float(distances[train].min()),
            "max_residual_m": maximum, "rmse_m": rmse})
    return records



class RegisteredRoadSurface(IndependentlyValidatedRoadSurface):
    """Enforce calibration independence for every actual source/target query."""

    def __init__(self, measured, independent, calibration_centers):
        super().__init__(measured, independent)
        centers = np.asarray(calibration_centers, dtype=float)
        if (centers.ndim != 2 or centers.shape[1] != 2 or not len(centers)
                or not np.isfinite(centers).all()):
            raise ValueError("Finite recomputed calibration centers required")
        self.calibration_centers = centers.copy()
        self.calibration_centers.setflags(write=False)
        self.provenance["actual_query_calibration_exclusion"] = {
            "minimum_center_distance_m": POLICY["query_exclusion_m"],
            "calibration_center_count": len(centers),
            "source": "raw_measurement_recomputed_calibration_centers",
        }

    def sample(self, xy):
        query = np.asarray(xy, dtype=float)
        if query.shape != (2,) or not np.isfinite(query).all():
            raise ValueError("Finite reconstruction XY required")
        distances = np.linalg.norm(self.calibration_centers - query, axis=1)
        nearest = int(np.argmin(distances))
        evidence = {
            "query_xy": query.tolist(),
            "nearest_calibration_center_xy": self.calibration_centers[nearest].tolist(),
            "actual_center_distance_m": float(distances[nearest]),
            "minimum_center_distance_m": POLICY["query_exclusion_m"],
        }
        if distances[nearest] < POLICY["query_exclusion_m"]:
            error = UnsupportedTerrain("actual_query_overlaps_registration_calibration_support")
            error.evidence = {**evidence, "status": "unsupported",
                "reason": "actual_query_overlaps_registration_calibration_support"}
            raise error
        result = super().sample(query)
        return {**result, "calibration_query_exclusion": {**evidence, "status": "supported"}}


def load_registered_road_surface(fit_pcd, fit_labels, reference_pcd, reference_labels,
                                 moving_pcd, moving_labels, certificate_path):
    certificate_path = Path(certificate_path).resolve(strict=True)
    certificate = json.loads(certificate_path.read_text())
    try:
        fit_id, reference_id, moving_id = [int(Path(p).stem.rsplit("-", 1)[1]) for p in (fit_pcd, reference_pcd, moving_pcd)]
    except (ValueError, IndexError) as error:
        raise ValueError("Native traversal-labelled PCD paths required") from error
    expected_policy = {**POLICY, "no_source_fit_traversal_used": fit_id}
    if (len({fit_id, reference_id, moving_id}) != 3
            or certificate.get("method") != "rigid_vertical_registration_disjoint_static_surface_v1"
            or certificate.get("frozen_policy") != expected_policy
            or certificate.get("reference_traversal") != reference_id
            or certificate.get("aligned_traversal") != moving_id
            or certificate.get("source_fit_traversal_unused") != fit_id):
        raise ValueError("Explicit frozen A13 calibration policy required")
    evidence_path = Path(certificate["calibration_evidence"]).resolve(strict=True)
    excluded_path = Path(certificate["experiment_exclusion_evidence"]).resolve(strict=True)
    for path, expected in ((evidence_path, certificate["calibration_evidence_sha256"]),
            (excluded_path, certificate["experiment_exclusion_evidence_sha256"])):
        if _file_sha(path) != expected:
            raise ValueError("Registration evidence changed")
    evidence = json.loads(evidence_path.read_text())
    excluded = json.loads(excluded_path.read_text())
    queries = np.asarray([xy for row in excluded["rows"] if "source_center" in row
        for xy in (row["source_center"][:2], (np.asarray(row["source_center"])+row["translation"])[:2])])
    if len(queries) == 0 or not np.isfinite(queries).all():
        raise ValueError("Explicit finite excluded experiment queries required")
    reference = MeasuredRoadSurface.from_pcd(reference_pcd, reference_labels)
    moving = MeasuredRoadSurface.from_pcd(moving_pcd, moving_labels)
    raw_ids = {reference.provenance["pcd_sha256"], moving.provenance["pcd_sha256"]}
    if len(raw_ids) != 2:
        raise ValueError("Distinct traversal point measurements required")
    rows = []
    seen = set()
    for candidate in evidence["rows"]:
        if not candidate.get("accepted_geometry"):
            continue
        xy = np.asarray(candidate["xy"], float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or tuple(xy) in seen:
            raise ValueError("Unique finite calibration centers required")
        seen.add(tuple(xy))
        if np.linalg.norm(queries-xy, axis=1).min() < 5.:
            raise ValueError("Calibration overlaps declared experiment support")
        recalculated = {}
        for traversal, surface in ((reference_id, reference), (moving_id, moving)):
            old = candidate["surfaces"][str(traversal)]
            provenance = old["provenance"]
            for key in ("pcd_sha256", "semantic_labels_sha256"):
                if provenance[key] != surface.provenance[key]:
                    raise ValueError("Registration proof belongs to different measured assets")
            if old["policy"] != surface.policy.__dict__:
                raise ValueError("Registration surface policy changed")
            recalculated[traversal], _ = IndependentlyValidatedRoadSurface._layer(surface, xy)
            if abs(recalculated[traversal]["height_m"]-old["height_m"]) > 1e-9:
                raise ValueError("Recorded static surface does not reproduce from actual raw points")
        rows.append({"xy": xy.tolist(), "reference": recalculated[reference_id], "moving": recalculated[moving_id]})
    if len(rows) < 64:
        raise ValueError("Insufficient measured calibration planes")
    checks = buffered_spatial_holdouts(rows)
    xy = np.asarray([x["xy"] for x in rows])
    differences = np.asarray([x["moving"]["height_m"]-x["reference"]["height_m"] for x in rows])
    transform, _, _ = derive_rigid_registration(xy, differences)
    require_derived_transform(certificate["rigid_transform"], transform)
    transformed_points = moving.points@transform[:3, :3].T+transform[:3, 3]
    fit = MeasuredRoadSurface.from_pcd(fit_pcd, fit_labels)
    if fit.provenance["pcd_sha256"] in raw_ids:
        raise ValueError("Fitted source traversal participated in its own independent calibration")
    independent = MeasuredRoadSurface(np.concatenate((reference.points, transformed_points)), {
        "traversals": [reference.provenance, {**moving.provenance, "coordinate_transform": transform.tolist()}],
        "registration": {"certificate": str(certificate_path), "sha256": _file_sha(certificate_path),
            "raw_source_recomputed": True, "calibration_plane_count": len(rows),
            "buffered_spatial_holdouts": checks,
            "scope": "global_rigid_registration_not_local_interpolation"}})
    return RegisteredRoadSurface(fit, independent, xy)
