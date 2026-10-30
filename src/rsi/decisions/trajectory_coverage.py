"""Reference-grounded coverage diagnostics, never an admission or skill certificate."""

from collections import Counter
from itertools import combinations

import numpy as np

from driversi.rsi.decisions.trajectory_diversity import compare, distances

SIMSCALE_REFERENCE = "https://arxiv.org/html/2511.23369v3#S2.SS3"
MTR_REFERENCE = "https://github.com/sshaoshuai/MTR/blob/master/mtr/utils/motion_utils.py"
GRID_STEPS_M = np.array([5.0, 0.5])


def _mean_window(values, count):
    return np.convolve(values, np.ones(count) / count, mode="valid")


def pair_metrics(first, second, reference_heading_rad):
    """Same-time XY distances; future starts at 2s, including both boundary states."""
    base = distances(first, second)  # Also validates complete finite 71-state arrays.
    if not np.isfinite(reference_heading_rad):
        raise ValueError("Finite common physical heading required")
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    c, s = np.cos(reference_heading_rad), np.sin(reference_heading_rad)
    delta = (a - b) @ np.array([[c, -s], [s, c]])
    separation = np.linalg.norm(delta, axis=1)
    future_windows = _mean_window(separation[20:], 11)  # 11 states span 1 second.
    speed_a = np.linalg.norm(np.diff(a, axis=0), axis=1) / 0.1
    speed_b = np.linalg.norm(np.diff(b, axis=0), axis=1) / 0.1
    speed_delta = np.abs(_mean_window(speed_a[20:], 10) - _mean_window(speed_b[20:], 10))
    return dict(
        **base,
        history_2s_ade_m=float(separation[:21].mean()),
        anchor_2s_distance_m=float(separation[20]),
        anchor_2s_longitudinal_difference_m=float(delta[20, 0]),
        anchor_2s_lateral_difference_m=float(delta[20, 1]),
        final_longitudinal_difference_m=float(delta[-1, 0]),
        final_lateral_difference_m=float(delta[-1, 1]),
        max_future_1s_mean_separation_m=float(future_windows.max()),
        max_separation_window_start_s=float(2 + np.argmax(future_windows) * 0.1),
        max_future_1s_mean_speed_difference_mps=float(speed_delta.max()),
    )


def family_coverage(descriptions):
    """Annotate a single source-anchor family without selecting or approving samples.

    SimScale uses sparse perturbation endpoints; these two-origin occupancy counts
    are sensitivity diagnostics in a fixed initial-ego frame, not its sampler.
    MTR's endpoint suppression motivates FDE reporting, not an adopted NMS gate.
    """
    result = dict(
        schema_version=1,
        kind="trajectory_family_coverage_diagnostics",
        admission_effect="none",
        distinct_samples_certified=False,
        causal_behavior_verified=False,
        references=dict(simscale=SIMSCALE_REFERENCE, endpoint_nms=MTR_REFERENCE),
        total_trajectories=len(descriptions),
    )
    if not descriptions:
        return dict(result, status="empty")
    names = [row["candidate_id"] for row in descriptions]
    if len(set(names)) != len(names):
        raise ValueError("Repeated candidate ID in coverage evidence")
    key = descriptions[0]["comparison_key"]
    for row in descriptions:
        if row["comparison_key"] != key:
            raise ValueError("Coverage requires a single source state and coordinate frame")
        distances(row["xy"], row["xy"])
    if any("initial_heading_rad" not in row for row in descriptions):
        return dict(result, status="unavailable_missing_initial_heading")
    headings = np.array([row["initial_heading_rad"] for row in descriptions], dtype=float)
    if not np.isfinite(headings).all():
        raise ValueError("Nonfinite initial heading")
    difference = np.arctan2(np.sin(headings - headings[0]), np.cos(headings - headings[0]))
    if np.any(np.abs(difference) > 1e-6):
        raise ValueError("Different initial orientations cannot share a coverage grid")
    xy = np.asarray([row["xy"] for row in descriptions], dtype=float)
    origin = xy[0, 0]
    if not np.allclose(xy[:, 0], origin, rtol=0, atol=1e-6):
        raise ValueError("Different initial positions cannot share a coverage grid")
    c, s = np.cos(headings[0]), np.sin(headings[0])
    local = (xy - origin) @ np.array([[c, -s], [s, c]])
    grids = []
    for phase in (0.0, 0.5):
        cells = np.floor(local[:, [20, 70]] / GRID_STEPS_M + phase).astype(np.int64)
        grids.append(
            dict(
                phase_fraction=phase,
                anchor_2s_occupied_cells=len({tuple(x) for x in cells[:, 0]}),
                final_7s_occupied_cells=len({tuple(x) for x in cells[:, 1]}),
                assignments=[
                    dict(candidate_id=name, anchor_2s=cell[0].tolist(), final_7s=cell[1].tolist())
                    for name, cell in zip(names, cells, strict=True)
                ],
            )
        )
    pairs = [
        dict(
            first=a["candidate_id"],
            second=b["candidate_id"],
            **pair_metrics(a["xy"], b["xy"], float(headings[0])),
        )
        for a, b in combinations(descriptions, 2)
    ]
    retained = []
    for row in descriptions:
        if compare(row, retained)["distinct"]:
            retained.append(row)
    return dict(
        result,
        status="diagnostic_only",
        coordinate_frame="fixed_initial_ego_heading_not_frenet",
        grid_steps_m=GRID_STEPS_M.tolist(),
        grid_semantics="SimScale_spacing_reference_only_not_interleaved_sampler_reproduction",
        grid_origin_sensitivity=grids,
        pairs=pairs,
        v1_retained_in_input_order=[row["candidate_id"] for row in retained],
        v1_rule="both_full_7s_and_future_5s_ADE_ge_0.5m_engineering_pilot_not_paper_threshold",
        behavior_hypothesis_attempt_counts=dict(
            Counter(
                label
                for row in descriptions
                for label in row.get("behavior_types", ["unclassified"])
            )
        ),
    )
