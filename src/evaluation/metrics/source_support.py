"""Source-entry geometry evidence, independent of simulated collision scores."""

import numpy as np

from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap


def source_entry_overlaps(scene):
    """Audit simultaneous source boxes at first annotations and reappearances.

    Fixed compiled dimensions are used, not per-frame raw DB dimensions. An
    overlap is a confounder; it does not establish duplicate identity or fault.
    Missing source annotations are never filled with simulated positions.
    """
    support = np.asarray(scene.metadata.get("reactive_background", {}).get(
        "source_presence", [scene.presence_at(t) for t in range(len(scene.times))]
    ), dtype=bool)
    if support.shape != scene.poses.shape[:2]:
        raise ValueError("Source support must match scene time and actor axes")
    entries = support & ~np.vstack((np.zeros_like(support[:1]), support[:-1]))
    radii = 0.5 * np.linalg.norm(scene.dimensions[:, :2], axis=1)
    events = []
    for tick in np.flatnonzero(entries.any(axis=1)):
        seen = set()
        for i in np.flatnonzero(entries[tick]):
            distance = np.linalg.norm(scene.poses[tick, :, :2] - scene.poses[tick, i, :2], axis=1)
            candidates = np.flatnonzero(support[tick] & (distance <= radii + radii[i]))
            for j in candidates:
                pair = tuple(sorted((int(i), int(j))))
                if i == j or pair in seen:
                    continue
                seen.add(pair)
                boxes = [rectangle_corners(*scene.poses[tick, n, :2], scene.poses[tick, n, 3],
                                           *scene.dimensions[n, :2]) for n in pair]
                if not rectangles_overlap(*boxes):
                    continue
                events.append({
                    "tick": int(tick), "time_seconds": float(scene.times[tick]),
                    "actors": [scene.actor_ids[n] for n in pair],
                    "entering_actors": [scene.actor_ids[n] for n in pair if entries[tick, n]],
                    "kind": "source_initial_overlap" if tick == 0 else "source_entry_overlap",
                    "dimensions_m": scene.dimensions[list(pair)].tolist(),
                    "causal_blame": "not_established",
                })
    return {"version": 1, "geometry": "compiled_fixed_boxes_SAT_no_margin",
            "scope": "source_initial_and_entry_frames_only_not_full_safety_assessment",
            "status": "source_overlap_confounder" if events else "no_entry_overlap_observed",
            "events": events, "simulation_qualified": False,
            "interpretation": "Do not merge IDs, suppress contacts, or label driver failures from this evidence alone."}
