"""Bounded C2 reference proposals; only executed states count as new data."""

import numpy as np

BACKEND = "source_history_perturbation_v1"
FAMILY = "history_endpoint_grid_tts_v1"
FIELDS = ("longitudinal_m", "lateral_m", "speed_delta_mps")


def validate_perturbation(value):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError("Historical perturbation requires exact endpoint fields")
    limits = (5.0, 2.0, 2.0)
    for key, limit in zip(FIELDS, limits):
        v = value[key]
        if type(v) not in (int, float) or not np.isfinite(v) or abs(v) > limit:
            raise ValueError("Out-of-envelope historical perturbation: " + key)
    return {key: float(value[key]) for key in FIELDS}


def perturbation_grid():
    # Alternate lateral and longitudinal targets before expanding speed variants.
    return [
        dict(zip(FIELDS, (lon, lat, dv)))
        for dv in (0.0, -1.0, 1.0)
        for lon, lat in ((0, -1), (0, 1), (-2, 0), (2, 0), (-2, -1), (-2, 1), (2, -1), (2, 1))
    ]


def quintic_reference(start, end, times, duration=2.0):
    """Analytic position/velocity/acceleration, then constant endpoint velocity.

    Endpoint acceleration is zero; this makes extrapolated tracking lookahead C2.
    Start/end contain world-frame XY position, velocity and acceleration (3,2).
    """
    start, end, times = map(lambda x: np.asarray(x, dtype=float), (start, end, times))
    if start.shape != (3, 2) or end.shape != (3, 2) or times.ndim != 1:
        raise ValueError("Expected two planar boundary states and vector times")
    if duration != 2.0 or not np.isfinite(np.r_[start.ravel(), end.ravel(), times]).all():
        raise ValueError("Finite two-second boundary states required")
    if np.any(times < 0) or np.any(end[2] != 0):
        raise ValueError("Nonnegative times and zero terminal acceleration required")
    c = np.zeros((6, 2))
    c[:3] = [start[0], start[1], start[2] / 2]
    t = duration
    matrix = np.array(
        [[t**3, t**4, t**5], [3 * t * t, 4 * t**3, 5 * t**4], [6 * t, 12 * t * t, 20 * t**3]]
    )
    c[3:] = np.linalg.solve(
        matrix,
        [end[0] - c[0] - c[1] * t - c[2] * t * t, end[1] - c[1] - 2 * c[2] * t, end[2] - 2 * c[2]],
    )
    bounded = np.minimum(times, t)
    p = np.polynomial.polynomial.polyval(bounded, c).T
    v = np.polynomial.polynomial.polyval(bounded, c[1:] * np.arange(1, 6)[:, None]).T
    a = np.polynomial.polynomial.polyval(
        bounded, c[2:] * (np.arange(2, 6) * np.arange(1, 5))[:, None]
    ).T
    after = times > t
    p[after] = end[0] + (times[after] - t)[:, None] * end[1]
    v[after], a[after] = end[1], end[2]
    return p, v, a


def arrival_search_compatible(tick, search):
    """Disjoint actual-control windows; never let a teacher overwrite active TTS."""
    return (
        type(tick) is int
        and 0 <= tick < 20
        and search is not None
        and search["start_tick"] == 20
        and search["stop_tick"] == 70
    )
