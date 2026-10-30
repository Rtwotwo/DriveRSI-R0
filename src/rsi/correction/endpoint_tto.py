"""Finite endpoint-reference construction; proposals are never executed labels.

Local TTO-inspired construction, not the official DriveTTO implementation.
Only training-source references are supported by the execution adapter.
"""
import numpy as np

BACKEND = "source_endpoint_tto_v1"
FAMILY = "history_endpoint_grid_tto_v1"


def endpoint_grid(history):
    lateral = float(history["lateral_m"])
    if not np.isfinite(lateral) or abs(lateral) > 2:
        raise ValueError("A bounded historical endpoint is required")
    # Preference is the requested side; zero is not an implicit optimum.
    return [dict(longitudinal_m=x, lateral_m=float(np.clip(lateral+y, -1.25, 1.25)))
            for x in (0., -1., 1.) for y in (0., -.25, .25)]


def warp_reference(xy, times, offset):
    xy, times = np.asarray(xy, float), np.asarray(times, float)
    delta = np.array([offset["longitudinal_m"], offset["lateral_m"]], float)
    if (xy.shape != (len(times), 2) or len(times) < 51
            or not np.isfinite(np.r_[xy.ravel(), times, delta]).all()
            or not np.allclose(times, np.arange(len(times))*.1, atol=1e-8)
            or abs(delta[0]) > 1 or abs(delta[1]) > 1.25):
        raise ValueError("Finite bounded 10Hz endpoint reference required")
    velocity = np.gradient(xy, .1, axis=0, edge_order=2)
    length = np.linalg.norm(velocity, axis=1)
    if np.any(length < .05):
        raise ValueError("Endpoint construction requires a moving source reference")
    tangent = velocity/length[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    u = np.clip(times/3., 0., 1.)
    blend = 10*u**3-15*u**4+6*u**5
    return xy + blend[:, None]*(delta[0]*tangent+delta[1]*normal)


def select_candidate(branches, keys):
    valid = [k for k in keys if branches[k]["trajectory_admitted"]]
    if not valid:
        return None
    # Score first, deterministic grid order for an exact tie. Independent
    # diversity and visual gates still decide whether the winner is usable.
    return max(valid, key=lambda k: branches[k]["score"]["score"])
