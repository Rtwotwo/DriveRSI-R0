"""Finite local road-height support from the pinned MTGS measured LiDAR assets.

Road labels are upstream semantic predictions, not geometric ground truth. Every
query must also pass bounded support, planar consistency and spatial holdouts.
No nearest-point fallback, extrapolation, outlier pruning or repaired support is
used. This is vertical support evidence, never visual or training admission.
"""
import copy
import functools
import hashlib
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .worldengine_geometry import UnsupportedTerrain
from .worldengine_terrain_repair import _inside_hull

TERRAIN_PROOF_VERSION = "measured_surface_residual_proof_v2"


@dataclass(frozen=True)
class MeasuredRoadPolicy:
    radius_m: float = 2.0
    min_support: int = 16
    max_slope: float = 0.2
    max_rmse_m: float = 0.03
    max_residual_m: float = 0.08
    max_holdout_residual_m: float = 0.08
    max_condition: float = 20.0
    fold_cell_m: float = 0.25

    def __post_init__(self):
        values = [v for k, v in asdict(self).items() if k != "min_support"]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Positive finite measured road policy required")
        if type(self.min_support) is not int or self.min_support < 12:
            raise ValueError("At least 12 original support points required")


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class MeasuredRoadSurface:
    """RoadHeightMap-compatible provider in reconstruction metres.

    The immutable input must be actual measured road-labelled points. Native MTGS
    PCD coordinates already use the refined reconstruction frame; a rigid optional
    transform may reproduce an explicitly pinned dataparser transform.
    """
    def __init__(self, points_xyz, provenance, policy=None):
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Finite measured XYZ array required; nonfinite points are not silently dropped")
        self.policy = policy or MeasuredRoadPolicy()
        if len(points) < self.policy.min_support:
            raise ValueError("Insufficient measured road cloud")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("Explicit measured cloud provenance required")
        self.points = np.array(points, copy=True)
        self.points.flags.writeable = False
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self.tree = None
        else:
            self.tree = cKDTree(self.points[:, :2])
        self.provenance = dict(provenance)
        self.array_sha256 = hashlib.sha256(self.points.tobytes()).hexdigest()

    @classmethod
    def from_pcd(cls, pcd_path, semantic_labels_path, *, coordinate_transform=None, policy=None, surface_labels=(0,)):
        import open3d as o3d
        pcd_path, semantic_labels_path = Path(pcd_path), Path(semantic_labels_path)
        cloud = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points)
        labels = np.load(semantic_labels_path, allow_pickle=False)
        if labels.shape != (len(cloud),) or labels.dtype.kind not in "iu":
            raise ValueError("Measured PCD and semantic labels must have exact one-to-one correspondence")
        if not np.isfinite(cloud).all():
            raise ValueError("Nonfinite measured PCD")
        transform = np.eye(4) if coordinate_transform is None else np.asarray(coordinate_transform, dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-8)):
            raise ValueError("Rigid metric reconstruction transform required")
        if (not isinstance(surface_labels, (tuple, list)) or not surface_labels
                or any(type(v) is not int or v not in (0, 1, 9) for v in surface_labels)
                or len(set(surface_labels)) != len(surface_labels)):
            raise ValueError("Explicit measured support labels must be road, sidewalk or terrain")
        road = cloud[np.isin(labels, surface_labels)]
        points = road @ transform[:3, :3].T + transform[:3, 3]
        return cls(points, {
            "pcd_path": str(pcd_path.resolve()), "pcd_sha256": _file_sha(pcd_path),
            "semantic_labels_path": str(semantic_labels_path.resolve()),
            "semantic_labels_sha256": _file_sha(semantic_labels_path),
            "input_count": len(cloud), "road_count": int((labels == 0).sum()),
            "surface_count": len(road), "surface_labels": list(surface_labels),
            "surface_label_meanings": {str(k): {0: "road", 1: "sidewalk", 9: "terrain"}[k] for k in surface_labels},
            "support_scope": "physical_vertical_surface_not_drivable_area_or_right_of_way",
            "semantic_provenance": "pinned_MTGS_projected_Cityscapes_prediction_not_ground_truth",
            "geometry_provenance": "pinned_MTGS_actual_LiDAR_accumulation_lidar2global",
            "coordinate_transform": transform.tolist(), "scale": 1.0,
        }, policy)

    def sample(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite reconstruction XY required")
        policy = self.policy
        if self.tree is None:
            dxy = self.points[:, :2] - xy
            ids = np.flatnonzero(np.einsum("ij,ij->i", dxy, dxy) <= policy.radius_m**2)
        else:
            ids = np.asarray(sorted(self.tree.query_ball_point(xy, policy.radius_m)), dtype=np.int64)
        points = self.points[ids]
        relative = points[:, :2] - xy
        evidence = {
            "query_xy": xy.tolist(), "support_count": len(ids), "policy": asdict(policy),
            "support_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "cloud_array_sha256": self.array_sha256, "provenance": self.provenance,
            "all_points_inside_radius_used": True, "outlier_pruning": False,
            "input_scope": "selected_geometric_layer" if "geometric_layer" in self.provenance else "all_input_surface_label_points",
        }
        def reject(reason):
            error = UnsupportedTerrain(reason)
            error.evidence = {**evidence, "status": "unsupported", "reason": reason}
            raise error
        if len(ids) < policy.min_support:
            reject("insufficient_measured_road_support")
        if not _inside_hull(relative):
            reject("outside_measured_road_convex_hull")
        design = np.column_stack((relative, np.ones(len(ids))))
        condition = float(np.linalg.cond(design))
        evidence["condition"] = condition if np.isfinite(condition) else None
        if not np.isfinite(condition) or condition > policy.max_condition:
            reject("ill_conditioned_measured_road_surface")
        fit = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
        errors = points[:, 2] - design @ fit
        evidence.update(plane_centered_reconstruction=fit.tolist(),
                        slope=float(np.linalg.norm(fit[:2])),
                        rmse_m=float(np.sqrt(np.mean(errors**2))),
                        max_residual_m=float(np.abs(errors).max()))
        if evidence["slope"] > policy.max_slope:
            reject("excessive_measured_road_slope")
        if evidence["rmse_m"] > policy.max_rmse_m or evidence["max_residual_m"] > policy.max_residual_m:
            reject("inconsistent_measured_road_surface")
        cells = np.floor(points[:, :2] / policy.fold_cell_m).astype(np.int64)
        folds = (cells[:, 0] % 2) * 2 + cells[:, 1] % 2
        records = []
        evidence["spatial_holdouts"] = records
        for fold in range(4):
            test = folds == fold
            train = ~test
            record = {"fold": fold, "training_count": int(train.sum()), "held_out_count": int(test.sum())}
            records.append(record)
            if test.sum() < 2 or train.sum() < 6:
                reject("insufficient_measured_spatial_holdout")
            cond = float(np.linalg.cond(design[train]))
            if not np.isfinite(cond) or cond > policy.max_condition:
                reject("ill_conditioned_measured_holdout")
            coeff = np.linalg.lstsq(design[train], points[train, 2], rcond=None)[0]
            held = points[test, 2] - design[test] @ coeff
            record.update(max_residual_m=float(np.abs(held).max()), rmse_m=float(np.sqrt(np.mean(held**2))))
            if record["max_residual_m"] > policy.max_holdout_residual_m:
                reject("measured_spatial_holdout_failed")
        return {**evidence, "height_m": float(fit[2]), "support_min_m": float(points[:, 2].min()),
                "support_max_m": float(points[:, 2].max()), "status": "supported",
                "method": "bounded_measured_road_plane_spatial_holdout_v1"}

    def lift(self, source_center, planar_translation):
        center, delta = np.asarray(source_center, dtype=float), np.asarray(planar_translation, dtype=float)
        if center.shape != (3,) or delta.shape != (3,) or not np.isfinite([*center, *delta]).all():
            raise ValueError("Finite source center and reconstruction translation required")
        source, target = self.sample(center[:2]), self.sample((center + delta)[:2])
        corrected = delta.copy()
        corrected[2] = target["height_m"] - source["height_m"]
        return corrected, {
            "status": "supported", "method": "measured_road_source_clearance_v1",
            "source": source, "target": target,
            "source_clearance_m": float(center[2] - source["height_m"]),
            "vertical_correction_m": float(corrected[2] - delta[2]),
            "vertical_displacement_m": float(corrected[2]),
            "orientation": "source_pitch_roll_with_transported_physical_yaw",
            "scope": "vertical_road_support_not_surface_orientation_or_visual_certificate",
        }


class IndependentlyValidatedRoadSurface(MeasuredRoadSurface):
    """Fixed robust geometric separation, checked on disjoint traversal returns.

    Every excluded return is retained in query evidence. A dominant layer is
    required in both partitions; ordinary planar/holdout limits are unchanged.
    This handles projected semantic errors without treating semantics as truth.
    """
    def __init__(self, measured, independent):
        if measured.array_sha256 == independent.array_sha256:
            raise ValueError("Independent measured cloud required")
        if measured.policy != independent.policy:
            raise ValueError("Identical frozen surface policies required")
        self.measured = measured
        self.independent = independent
        self.policy = measured.policy
        self.provenance = {"measured": measured.provenance, "independent": independent.provenance}

    @staticmethod
    def _layer(surface, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite reconstruction XY required")
        if surface.tree is None:
            dxy = surface.points[:, :2] - xy
            ids = np.flatnonzero(np.einsum("ij,ij->i", dxy, dxy) <= surface.policy.radius_m**2)
        else:
            ids = np.asarray(sorted(surface.tree.query_ball_point(xy, surface.policy.radius_m)), dtype=np.int64)
        points = surface.points[ids]
        evidence = {"source": surface.provenance, "cloud_array_sha256": surface.array_sha256,
                    "total_support_count": len(ids), "max_excluded_fraction": 0.05,
                    "huber_delta_m": 0.03, "iterations": 20, "inlier_distance_m": 0.06,
                    "terrain_proof_version": TERRAIN_PROOF_VERSION}
        def reject(reason):
            error = UnsupportedTerrain(reason)
            error.evidence = {**evidence, "status": "unsupported", "reason": reason}
            raise error
        if len(ids) < surface.policy.min_support:
            reject("insufficient_independent_measured_layer_support")
        design = np.column_stack((points[:, :2] - xy, np.ones(len(ids))))
        fit = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
        for _ in range(20):
            residual = points[:, 2] - design @ fit
            weights = np.minimum(1.0, 0.03 / np.maximum(np.abs(residual), 1e-15))
            root = np.sqrt(weights)
            fit = np.linalg.lstsq(design * root[:, None], points[:, 2] * root, rcond=None)[0]
        residual = points[:, 2] - design @ fit
        keep = np.abs(residual) <= 0.06
        evidence.update(excluded_fraction=float((~keep).mean()),
                        excluded_point_ids=ids[~keep].tolist(),
                        excluded_points_xyz=points[~keep].tolist(),
                        excluded_signed_residuals_m=residual[~keep].tolist(),
                        excluded_reason="outside_fixed_6cm_dominant_layer_band",
                        robust_plane=fit.tolist(),
                        included_point_ids_sha256=hashlib.sha256(ids[keep].tobytes()).hexdigest())
        if (~keep).mean() > 0.05:
            reject("ambiguous_measured_road_layers")
        if keep.sum() < surface.policy.min_support:
            reject("insufficient_dominant_layer_support")
        evidence["secondary_layer_check"] = _layer_residual_check(evidence, xy)
        if evidence["secondary_layer_check"]["status"] != "no_supported_secondary":
            reject("ambiguous_measured_road_secondary_layer")
        layer = MeasuredRoadSurface(points[keep], {**surface.provenance, "geometric_layer": evidence}, surface.policy)
        try:
            result = layer.sample(xy)
        except UnsupportedTerrain as error:
            error.evidence = {**getattr(error, "evidence", {}), "layer_separation": evidence}
            raise
        return result, points[keep]

    def sample(self, xy):
        fit, points = self._layer(self.measured, xy)
        check, independent_points = self._layer(self.independent, xy)
        fitcoef = np.asarray(fit["plane_centered_reconstruction"])
        checkcoef = np.asarray(check["plane_centered_reconstruction"])
        relative = np.concatenate((points[:, :2], independent_points[:, :2])) - np.asarray(xy)
        design = np.column_stack((relative, np.ones(len(relative))))
        max_difference = float(np.abs(design @ (fitcoef - checkcoef)).max())
        center_difference = float(abs(fit["height_m"] - check["height_m"]))
        evidence = {"method": "dominant_measured_road_layer_disjoint_traversal_validation_v1",
                    "fit": fit, "independent": check,
                    "independent_center_difference_m": center_difference,
                    "independent_max_plane_difference_m": max_difference,
                    "max_center_difference_m": .03, "max_plane_difference_m": .08,
                    "quality_admitted": False,
                    "terrain_proof_version": TERRAIN_PROOF_VERSION}
        if center_difference > .03 or max_difference > .08:
            error = UnsupportedTerrain("independent_measured_road_surface_disagrees")
            error.evidence = {**evidence, "status": "unsupported"}
            raise error
        result = {**fit, "method": evidence["method"], "independent_surface_validation": evidence}
        validate_independent_surface_evidence(result)
        return result


CONSENSUS_METHOD = "traversal_consensus_measured_road_v1"


@functools.lru_cache(maxsize=256)
def _residual_layer_check(points_tuple, xy_tuple):
    """Find conflicts; bounded proposal failure cannot prove absence of a layer."""
    points, xy = np.asarray(points_tuple, float), np.asarray(xy_tuple, float)
    count = len(points)
    evidence = {"method": "bounded_secondary_surface_witness_search_v2",
                "residual_point_count": count, "max_residual_points": 40,
                "minimum_secondary_support": 16, "hypothesis_count": 0,
                "supported_secondary": None}
    if count < 16:
        return {**evidence, "status": "no_supported_secondary"}
    if not _inside_hull(points[:, :2] - xy):
        return {**evidence, "status": "no_supported_secondary",
                "proof": "query_outside_all_residual_points_convex_hull"}
    # The original eight-centimetre fit can accept a surface that no six-centimetre
    # proposal band contains. Test all residuals first; proposal search below is
    # only a way to produce a counterexample, never a proof of absence.
    try:
        candidate = MeasuredRoadSurface(
            points, {"secondary_layer_hypothesis": "all_residual_points"}
        ).sample(xy)
    except UnsupportedTerrain:
        pass
    else:
        return {**evidence, "status": "ambiguous", "supported_secondary": candidate}
    if count > 40:
        return {**evidence, "status": "unresolved", "reason": "residual_plane_budget_exceeded"}
    design = np.column_stack((points[:, :2] - xy, np.ones(count)))
    seen = set()
    for indices in itertools.combinations(range(count), 3):
        matrix = design[list(indices)]
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        coefficients = np.linalg.solve(matrix, points[list(indices), 2])
        if np.linalg.norm(coefficients[:2]) > MeasuredRoadPolicy().max_slope:
            continue
        keep = np.abs(points[:, 2] - design @ coefficients) <= .06
        key = tuple(np.flatnonzero(keep))
        if keep.sum() < 16 or key in seen:
            continue
        seen.add(key)
        evidence["hypothesis_count"] += 1
        if not _inside_hull(points[keep, :2] - xy):
            continue
        try:
            candidate = MeasuredRoadSurface(
                points[keep], {"secondary_layer_hypothesis": list(indices)}
            ).sample(xy)
        except UnsupportedTerrain:
            continue
        return {**evidence, "status": "ambiguous", "supported_secondary": candidate}
    return {**evidence, "status": "unresolved",
            "reason": "bounded_proposals_do_not_prove_secondary_surface_absence"}


def _layer_residual_check(layer, query):
    """Recompute the exact excluded measurements, including small fractions."""
    if layer.get("terrain_proof_version") != TERRAIN_PROOF_VERSION:
        raise ValueError("Current terrain residual proof version required")
    raw = layer.get("excluded_points_xyz")
    ids = layer.get("excluded_point_ids")
    if not isinstance(raw, list) or not isinstance(ids, list):
        raise ValueError("Complete finite source residual measurements required")
    points = np.asarray(raw, dtype=float).reshape((-1, 3)) if raw == [] else np.asarray(raw, dtype=float)
    xy = np.asarray(query, float)
    total = layer.get("total_support_count")
    fraction = layer.get("excluded_fraction")
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or xy.shape != (2,) or not np.isfinite(xy).all()
            or len(points) != len(ids) or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != len(ids) or type(total) is not int or total < max(1, len(ids))
            or type(fraction) not in (int, float) or not np.isfinite(fraction)
            or not np.isclose(fraction, len(ids) / total, rtol=0, atol=1e-12)):
        raise ValueError("Complete finite source residual measurements required")
    return _residual_layer_check(tuple(map(tuple, points)), tuple(xy))


def _source_rejection_check(rejection, query):
    if rejection.get("reason") not in {
        "ambiguous_measured_road_layers", "ambiguous_measured_road_secondary_layer"
    }:
        raise ValueError("Unresolved source measurement cannot be replaced by consensus")
    evidence = _layer_residual_check(rejection, query)
    if evidence["status"] != "no_supported_secondary":
        raise ValueError("Source secondary layer remains ambiguous: " + evidence["status"])
    return evidence


def _validate_supported_layer(fit, query):
    layer = fit.get("provenance", {}).get("geometric_layer", {})
    expected = _layer_residual_check(layer, query)
    if expected["status"] != "no_supported_secondary":
        raise ValueError("Source secondary layer remains ambiguous: " + expected["status"])
    if layer.get("secondary_layer_check") != expected:
        raise ValueError("Observed secondary-layer check differs from retained measurements")
    if (layer.get("max_excluded_fraction") != .05 or layer.get("inlier_distance_m") != .06
            or not 0 <= layer["excluded_fraction"] <= .05):
        raise ValueError("Original independent layer policy required")
    return _strict_fit_coefficients(fit, query)


def _strict_fit_coefficients(fit, query):
    if fit.get("status") != "supported" or fit.get("policy") != asdict(MeasuredRoadPolicy()):
        raise ValueError("Frozen measured terrain fit required")
    for key, limit in (("rmse_m", .03), ("max_residual_m", .08), ("slope", .2), ("condition", 20.)):
        value = fit.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Measured terrain limits violated")
    folds = fit.get("spatial_holdouts", [])
    if len(folds) != 4 or {x.get("fold") for x in folds} != set(range(4)):
        raise ValueError("Original spatial holdouts required")
    for fold in folds:
        residual = fold.get("max_residual_m")
        if (type(residual) not in (int, float) or not np.isfinite(residual)
                or not 0 <= residual <= .08 or fold.get("training_count", 0) < 6
                or fold.get("held_out_count", 0) < 2):
            raise ValueError("Measured spatial holdout failed")
    coefficients = np.asarray(fit.get("plane_centered_reconstruction"), float)
    height = fit.get("height_m")
    if (coefficients.shape != (3,) or not np.isfinite(coefficients).all()
            or type(height) not in (int, float) or not np.isfinite(height)
            or not np.isclose(coefficients[2], height, rtol=0, atol=1e-10)
            or fit.get("query_xy") != list(query)):
        raise ValueError("Terrain query and plane evidence differ")
    return coefficients



def validate_independent_surface_evidence(sample):
    """Reject legacy pooled evidence that lacks complete residual-layer proof."""
    validation = sample.get("independent_surface_validation", {})
    method = "dominant_measured_road_layer_disjoint_traversal_validation_v1"
    if (sample.get("status") != "supported" or sample.get("method") != method
            or validation.get("method") != method
            or validation.get("terrain_proof_version") != TERRAIN_PROOF_VERSION
            or validation.get("max_center_difference_m") != .03
            or validation.get("max_plane_difference_m") != .08):
        raise ValueError("Current independent terrain residual proof required")
    query = sample.get("query_xy")
    fit, check = validation.get("fit", {}), validation.get("independent", {})
    left, right = _validate_supported_layer(fit, query), _validate_supported_layer(check, query)
    _validate_supported_layer(sample, query)
    if (fit.get("provenance", {}).get("geometric_layer", {}).get("cloud_array_sha256")
            == check.get("provenance", {}).get("geometric_layer", {}).get("cloud_array_sha256")):
        raise ValueError("Independent measured cloud required")
    if (not np.allclose(sample.get("plane_centered_reconstruction"), left, rtol=0, atol=1e-10)
            or abs(left[2] - right[2]) > .03):
        raise ValueError("Independent terrain planes disagree")
    for key, limit in (("independent_center_difference_m", .03),
                       ("independent_max_plane_difference_m", .08)):
        value = validation.get(key)
        if type(value) not in (float, int) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Independent terrain plane difference evidence invalid")


def _retained_primary(surface, xy, rejection):
    """Retain failed traversal's height constraint without qualifying it as a vote."""
    if surface.tree is None:
        dxy = surface.points[:, :2] - xy
        ids = np.flatnonzero(np.einsum("ij,ij->i", dxy, dxy) <= surface.policy.radius_m**2)
    else:
        ids = np.asarray(sorted(surface.tree.query_ball_point(xy, surface.policy.radius_m)), dtype=np.int64)
    retained = ids[~np.isin(ids, rejection["excluded_point_ids"])]
    primary = MeasuredRoadSurface(surface.points[retained], {
        "rejected_cloud_sha256": surface.array_sha256,
        "retained_original_point_ids_sha256": hashlib.sha256(retained.tobytes()).hexdigest(),
        "scope": "contradiction_constraint_only_not_qualified_traversal",
    }, surface.policy).sample(xy)
    _strict_fit_coefficients(primary, xy)
    return primary


def validate_traversal_consensus_evidence(sample):
    """Validate independently measured support; this never grants visual admission."""
    validation = sample.get("independent_surface_validation", {})
    if (sample.get("status") != "supported" or sample.get("method") != CONSENSUS_METHOD
            or validation.get("method") != CONSENSUS_METHOD
            or validation.get("terrain_proof_version") != TERRAIN_PROOF_VERSION
            or validation.get("minimum_supported_references") != 2
            or validation.get("max_center_difference_m") != .03
            or validation.get("max_plane_difference_m") != .08):
        raise ValueError("Unsupported traversal consensus evidence")
    records = validation.get("traversals", [])
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("Source and at least two independent traversals required")
    hashes = [row.get("cloud_array_sha256") for row in records]
    if any(not isinstance(h, str) or len(h) != 64 for h in hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("Distinct measured traversal arrays required")
    raw_hashes = [row.get("pcd_sha256") for row in records if row.get("pcd_sha256")]
    if len(raw_hashes) != len(set(raw_hashes)):
        raise ValueError("Independent raw traversal measurements required")
    if sum(row.get("role") == "source" for row in records) != 1:
        raise ValueError("Exactly one source traversal required")
    fits, comparison_fits = [], []
    references = 0
    for row in records:
        if row.get("role") not in {"source", "reference"}:
            raise ValueError("Explicit traversal role required")
        if row.get("status") == "unsupported":
            if not row.get("reason") or not isinstance(row.get("rejection"), dict):
                raise ValueError("Failed measurement evidence must be retained")
            if row["role"] == "source" or row["reason"] in {"ambiguous_measured_road_layers", "ambiguous_measured_road_secondary_layer"}:
                primary = row.get("retained_primary_fit", {})
                if primary.get("provenance", {}).get("rejected_cloud_sha256") != row["cloud_array_sha256"]:
                    raise ValueError("Rejected traversal primary measurement missing")
                comparison_fits.append(_strict_fit_coefficients(primary, sample.get("query_xy")))
                expected = _source_rejection_check(row["rejection"], sample.get("query_xy"))
                if row.get("secondary_layer_check") != expected:
                    raise ValueError("Observed secondary-layer check differs from retained measurements")
            continue
        fit = row.get("fit", {})
        if row.get("status") != "supported" or fit.get("status") != "supported":
            raise ValueError("Unverified terrain traversal")
        if fit.get("policy") != asdict(MeasuredRoadPolicy()):
            raise ValueError("Frozen measured terrain policy required")
        layer = fit.get("provenance", {}).get("geometric_layer", {})
        if (layer.get("max_excluded_fraction") != .05
                or layer.get("inlier_distance_m") != .06
                or layer.get("cloud_array_sha256") != row["cloud_array_sha256"]):
            raise ValueError("Original independent layer policy required")
        fraction = layer.get("excluded_fraction")
        if type(fraction) not in (int, float) or not np.isfinite(fraction) or not 0 <= fraction <= .05:
            raise ValueError("Measured terrain limits violated")
        coefficients = _validate_supported_layer(fit, sample.get("query_xy"))
        fits.append(coefficients)
        comparison_fits.append(coefficients)
        references += row["role"] == "reference"
    if references < 2:
        raise ValueError("At least two supported independent references required")
    for left, right in itertools.combinations(comparison_fits, 2):
        diff = left - right
        if abs(diff[2]) > .03 or abs(diff[2]) + 2. * np.linalg.norm(diff[:2]) > .08:
            raise ValueError("Validated traversal planes disagree")
    mean = np.mean(fits, axis=0)
    if (not np.allclose(sample.get("plane_centered_reconstruction"), mean, rtol=0, atol=1e-10)
            or not np.isclose(sample.get("height_m"), mean[2], rtol=0, atol=1e-10)):
        raise ValueError("Consensus is not the equal-traversal measured mean")


class TraversalConsensusRoadSurface(MeasuredRoadSurface):
    """Separate traversals before consensus; retain rejected measurements in full.

    Source appearance and road measurements are distinct assets. A contaminated
    source cloud cannot erase agreement between at least two other traversals.
    Conversely any independently valid conflicting plane rejects the query.
    No point-density vote, best-fit selection, registration or local repair occurs.
    """
    def __init__(self, measured, independent):
        independent = tuple(independent)
        if len(independent) < 2:
            raise ValueError("At least two independent reference traversals required")
        self.surfaces = (measured, *independent)
        self.policy = measured.policy
        if self.policy != MeasuredRoadPolicy() or any(s.policy != self.policy for s in independent):
            raise ValueError("Original frozen measured road policy required")
        hashes = [s.array_sha256 for s in self.surfaces]
        raw = [s.provenance["pcd_sha256"] for s in self.surfaces if "pcd_sha256" in s.provenance]
        if len(set(hashes)) != len(hashes) or len(set(raw)) != len(raw):
            raise ValueError("Distinct source and reference traversal measurements required")
        self.provenance = {"source": measured.provenance,
                           "independent_traversals": [s.provenance for s in independent]}
        self._cache = {}

    def sample(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite reconstruction XY required")
        key = tuple(xy)
        if key in self._cache:
            return copy.deepcopy(self._cache[key])
        records, fits = [], []
        for index, surface in enumerate(self.surfaces):
            row = {"role": "source" if index == 0 else "reference",
                   "cloud_array_sha256": surface.array_sha256,
                   "pcd_sha256": surface.provenance.get("pcd_sha256"),
                   "provenance": surface.provenance}
            try:
                fit, _ = IndependentlyValidatedRoadSurface._layer(surface, xy)
            except UnsupportedTerrain as error:
                row.update(status="unsupported", reason=str(error),
                           rejection=getattr(error, "evidence", {}))
                if index == 0 or str(error) in {"ambiguous_measured_road_layers", "ambiguous_measured_road_secondary_layer"}:
                    try:
                        row["retained_primary_fit"] = _retained_primary(surface, xy, row["rejection"])
                        row["secondary_layer_check"] = _source_rejection_check(row["rejection"], xy)
                    except (ValueError, UnsupportedTerrain, KeyError) as cause:
                        row["secondary_layer_check"] = {"status": "unresolved", "reason": str(cause)}
            else:
                row.update(status="supported", fit=fit)
                fits.append(fit["plane_centered_reconstruction"])
            records.append(row)
        result = {"status": "supported", "method": CONSENSUS_METHOD,
                  "query_xy": xy.tolist(), "policy": asdict(self.policy),
                  "provenance": self.provenance, "quality_admitted": False,
                  "independent_surface_validation": {
                      "method": CONSENSUS_METHOD, "minimum_supported_references": 2,
                      "terrain_proof_version": TERRAIN_PROOF_VERSION,
                      "max_center_difference_m": .03, "max_plane_difference_m": .08,
                      "comparison_domain": "entire_query_centered_2m_disk",
                      "traversals": records, "quality_admitted": False}}
        if fits:
            mean = np.mean(fits, axis=0)
            result.update(plane_centered_reconstruction=mean.tolist(), height_m=float(mean[2]))
        try:
            validate_traversal_consensus_evidence(result)
        except ValueError as cause:
            error = UnsupportedTerrain("independent_traversal_consensus_failed: " + str(cause))
            error.evidence = {**result, "status": "unsupported", "reason": str(cause)}
            raise error from cause
        self._cache[key] = copy.deepcopy(result)
        return result
