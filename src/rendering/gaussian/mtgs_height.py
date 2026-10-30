"""Bounded height differences along one actor's own reference trajectory.

The samples are reconstructed *actor reference-point* heights (e.g. box centres),
not a terrain surface. This proxy does not change physics, XY, or orientation.
Unsupported queries return an explicit zero-correction fallback.
"""

from __future__ import annotations

import numpy as np


class ReferenceTrackHeight:
    """Piecewise-linear spatial proxy with no endpoint extrapolation.

    Projection ambiguity means candidates within 0.05 m of the best distance
    disagree in height by more than 0.10 m. Adjacent segments are treated the
    same as distant ones: geometry, not source time, determines support.
    Repeated XY samples do not create interpolation segments. Their differing
    heights remain ambiguity evidence near that point.
    """

    VERSION = "reference-track-height-v1"
    AMBIGUITY_DISTANCE_M = 0.05
    AMBIGUITY_HEIGHT_M = 0.10
    MIN_SEGMENT_M = 1e-6
    ENDPOINT_NUMERICAL_TOLERANCE_M = 1e-4

    def __init__(self, xy, z, *, max_lateral_m=2., max_segment_m=20.,
                 max_grade=.25, max_correction_m=2.):
        points = np.asarray(xy, dtype=np.float64)
        heights = np.asarray(z, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1
                or heights.shape != (len(points),)
                or not np.isfinite(points).all() or not np.isfinite(heights).all()):
            raise ValueError("Reference height samples require finite XY[N,2] and Z[N], N>=1")
        values = (max_lateral_m, max_segment_m, max_grade, max_correction_m)
        if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Reference height limits must be positive finite numbers")
        self.max_lateral_m, self.max_segment_m = float(max_lateral_m), float(max_segment_m)
        self.max_grade, self.max_correction_m = float(max_grade), float(max_correction_m)
        self._xy, self._z = points.copy(), heights.copy()
        self._delta = np.diff(self._xy, axis=0)
        self._length = np.linalg.norm(self._delta, axis=1)
        self._moving = np.flatnonzero(self._length > self.MIN_SEGMENT_M)
        repeated = np.flatnonzero(self._length <= self.MIN_SEGMENT_M)
        self._repeated_samples = np.unique(np.concatenate((repeated, repeated+1)))
        self._arclength = np.r_[0., np.cumsum(self._length)]

    @property
    def configuration(self) -> dict:
        """Return a detached JSON-safe copy of every numerical support limit."""
        return {
            "version": self.VERSION,
            "semantics": "same_actor_reference_point_height_not_terrain",
            "max_lateral_m": self.max_lateral_m,
            "max_segment_m": self.max_segment_m,
            "max_grade": self.max_grade,
            "max_correction_m": self.max_correction_m,
            "ambiguity_distance_m": self.AMBIGUITY_DISTANCE_M,
            "ambiguity_height_m": self.AMBIGUITY_HEIGHT_M,
            "min_segment_m": self.MIN_SEGMENT_M,
            "endpoint_numerical_tolerance_m": self.ENDPOINT_NUMERICAL_TOLERANCE_M,
        }

    @staticmethod
    def _point(value):
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Height query requires a finite XY pair")
        return point

    def _project(self, point):
        if not len(self._moving):
            return False, "no_moving_reference_segment", None
        indices = self._moving
        lengths = self._length[indices]
        delta = self._delta[indices]
        raw = np.einsum("ij,ij->i", point-self._xy[indices], delta) / lengths**2
        fraction = np.clip(raw, 0., 1.)
        projected = self._xy[indices] + fraction[:, None]*delta
        distance = np.linalg.norm(projected-point, axis=1)
        values = self._z[indices] + fraction*(self._z[indices+1]-self._z[indices])
        best = int(np.argmin(distance))
        segment = int(indices[best])
        result = {
            "height_m": float(values[best]), "distance_m": float(distance[best]),
            "segment_index": segment, "fraction": float(fraction[best]),
            "unclamped_fraction": float(raw[best]),
            "xy": projected[best].tolist(),
            "along_track_m": float(self._arclength[segment]+fraction[best]*lengths[best]),
        }
        if distance[best] > self.max_lateral_m:
            return False, "outside_reference_corridor", result
        before = best == 0 and raw[best]*lengths[best] < -self.ENDPOINT_NUMERICAL_TOLERANCE_M
        after = (best == len(indices)-1
                 and (raw[best]-1)*lengths[best] > self.ENDPOINT_NUMERICAL_TOLERANCE_M)
        if before or after:
            return False, "outside_reference_extent", result
        near = distance <= distance[best]+self.AMBIGUITY_DISTANCE_M
        candidate_heights = values[near]
        if len(self._repeated_samples):
            repeat_distance = np.linalg.norm(self._xy[self._repeated_samples]-point, axis=1)
            repeated_near = repeat_distance <= distance[best]+self.AMBIGUITY_DISTANCE_M
            candidate_heights = np.r_[candidate_heights,
                                     self._z[self._repeated_samples[repeated_near]]]
        if np.ptp(candidate_heights) > self.AMBIGUITY_HEIGHT_M:
            return False, "ambiguous_reference_height", result
        # Do not jump over the nearest unsupported segment to a farther valid one.
        if lengths[best] > self.max_segment_m:
            return False, "reference_segment_gap", result
        if abs(self._z[segment+1]-self._z[segment])/lengths[best] > self.max_grade:
            return False, "reference_grade_exceeded", result
        return True, "supported", result

    def correction(self, current_xy, reference_xy) -> dict:
        """Return H(current XY)-H(reference XY), or an audited zero fallback.

        Both points are queried against the same profile. Consequently exact
        zero displacement is exactly zero whenever the location is supported.
        Failure never silently clamps the queried location into supported data.
        """
        current, reference = self._point(current_xy), self._point(reference_xy)
        source_ok, source_reason, source = self._project(reference)
        target_ok, target_reason, target = self._project(current)
        result = {
            "version": self.VERSION, "supported": False, "reason": "unresolved",
            "correction_m": 0., "source_projection": source, "target_projection": target,
            "fallback": "retain_source_time_height", "semantics": "actor_reference_height_proxy",
            "terrain_validated": False,
        }
        if not source_ok:
            result["reason"] = "source_"+source_reason
        elif not target_ok:
            result["reason"] = "target_"+target_reason
        else:
            correction = 0. if np.array_equal(current, reference) else (
                target["height_m"]-source["height_m"])
            if abs(correction) > self.max_correction_m:
                result["reason"] = "correction_limit_exceeded"
            else:
                result.update(supported=True, reason="supported", correction_m=float(correction),
                              fallback=None)
        return result
