"""Footprint support in an extracted map crop, distinct from road legality."""

import numpy as np


def crop_support(scene, state, present, margin=2.0):
    bounds = scene.metadata.get("map_bounds_global")
    if bounds is None:
        return {"status": "unknown", "reason": "map_crop_not_declared", "actors": []}
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("Invalid declared map crop or support margin")
    rows = np.flatnonzero(present)
    xy = state[rows, :2] + np.asarray(scene.metadata["origin_xyz"][:2])
    c, s = np.abs(np.cos(state[rows, 3])), np.abs(np.sin(state[rows, 3]))
    length, width = scene.dimensions[rows, 0], scene.dimensions[rows, 1]
    radius = np.column_stack((length * c + width * s, length * s + width * c)) * 0.5
    clearance = np.min(
        np.column_stack((xy - radius - bounds[:2], bounds[2:] - xy - radius)), axis=1
    )
    actors = [
        {
            "actor": scene.actor_ids[int(i)],
            "footprint_clearance_m": float(d),
            "outside_crop": bool(d < 0),
        }
        for i, d in zip(rows, clearance)
        if d < margin
    ]
    return {
        "status": "boundary_warning" if actors else "inside_declared_crop",
        "margin_m": float(margin),
        "actors": actors,
        "scope": "extraction extent only; not proof of mapped road or legal right of way",
    }
