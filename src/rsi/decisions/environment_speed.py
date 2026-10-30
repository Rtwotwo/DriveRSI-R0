"""Frozen source-history heuristics for bounded proposals and speed novelty.

These engineering thresholds do not certify traffic safety or replace native
admission. No query position, future training answer, or candidate score is used.
"""
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest

VERSION = "source_history_speed_envelope_v1"


def policy_from_facts(facts):
    speed = float(facts["anchor_speed_mps"])
    curvature = facts["observed_curvature_inv_m"]
    gap = facts["forward_box_gap_m"]
    values = [speed] + [float(v) for v in (curvature, gap) if v is not None]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Finite nonnegative source speed facts required")
    novelty = float(np.clip(.08 * speed, .3, 1.))
    cap = min(2., max(.5, .25 * speed))
    positive, negative = cap, min(cap, speed)
    reasons = ["native_reference_envelope_2mps", "relative_source_speed"]
    if curvature is None:
        positive *= .5
        reasons.append("unknown_curvature_conservative")
    elif curvature > 1e-4:
        positive = min(positive, max(0., np.sqrt(2. / curvature) - speed))
        reasons.append("observed_curve_lateral_acceleration_proxy")
    if not facts["objects_observed"]:
        positive = min(positive, .5 * cap)
        reasons.append("unknown_forward_objects_conservative")
    elif gap is not None:
        # 2 m reserve, 1 s reaction, 2 m/s² braking: a conservative stationary
        # object proxy, not a leader-velocity estimate or a road-speed limit.
        limit = max(0., -2. + np.sqrt(4. + 4. * max(0., gap - 2.)))
        positive = min(positive, max(0., limit - speed))
        reasons.append("forward_object_gap_proxy")
    return dict(version=VERSION, facts=facts, minimum_speed_mae_mps=novelty,
        minimum_above_threshold_seconds=1., max_positive_delta_mps=float(positive),
        max_negative_delta_mps=float(negative), reasons=reasons,
        scope="source_history_reference_bounds_and_actual_pair_speed_classification",
        spatial_gate_required=True, traffic_safety_certified=False,
        speed_limit_known=False)


def source_policy(logs_root, source_log, token):
    from driversi.data.sources.navsim import read_log, yaw_from_quaternion
    root = Path(logs_root).resolve(strict=True)
    path = (root / (source_log + ".pkl")).resolve(strict=True)
    if Path(source_log).name != source_log or not path.is_relative_to(root):
        raise ValueError("Source log must remain inside the declared training logs")
    frames = read_log(path, trusted=True)
    indices = [i for i, row in enumerate(frames) if row["token"] == token]
    if len(indices) != 1:
        raise ValueError("Unique source anchor required")
    index = indices[0]
    if index < 4:
        raise ValueError("Two seconds of source history required for speed context")
    history = frames[index-4:index+1]
    times = np.array([row["timestamp"] for row in history], dtype=float) / 1e6
    if np.any(np.abs(np.diff(times) - .5) > .08):
        raise ValueError("Do not bridge gaps when deriving source speed context")
    xy = np.array([row["ego2global_translation"][:2] for row in history], dtype=float)
    yaw = np.unwrap([yaw_from_quaternion(row["ego2global_rotation"]) for row in history])
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    supported = lengths >= .25
    curvature = float(np.max(np.abs(np.diff(yaw)[supported]) / lengths[supported])) if supported.all() else None
    frame = history[-1]
    speed = float(np.linalg.norm(np.array(frame["ego_dynamic_state"], dtype=float)[:2]))
    boxes = np.asarray(frame.get("anns", {}).get("gt_boxes", []), dtype=float)
    observed = bool(boxes.ndim == 2 and boxes.shape[1] == 7 and np.isfinite(boxes).all())
    gaps = []
    if observed:
        # Native NAVSIM boxes are in ego coordinates; use yaw-aware extents.
        for x, y, _, length, width, _, angle in boxes:
            ex = .5 * (abs(np.cos(angle))*length + abs(np.sin(angle))*width)
            ey = .5 * (abs(np.sin(angle))*length + abs(np.cos(angle))*width)
            if x > 0 and abs(y) <= 1.5 + ey:
                gaps.append(max(0., float(x - ex - 2.5)))
    facts = dict(source_log=source_log, token=token, anchor_timestamp_us=int(frame["timestamp"]),
        anchor_speed_mps=speed, observed_curvature_inv_m=curvature,
        forward_box_gap_m=min(gaps) if gaps else None, objects_observed=observed,
        history_seconds=float(times[-1]-times[0]), source_file=str(path), source_sha256=digest(path),
        observation_scope="training_source_history_only", future_used=False)
    return policy_from_facts(facts)


def adapt_grid(grid, policy, count):
    if policy != policy_from_facts(policy["facts"]):
        raise ValueError("Speed policy must replay from frozen source facts")
    # Remap levels within the frozen envelope, deduplicate after clipping. Keep
    # actual fewer proposals; never fill a budget with repeated references.
    rows = []
    seen = set()
    for row in grid:
        delta = row["speed_delta_mps"]
        if delta > 0:
            delta = policy["max_positive_delta_mps"] * min(delta/2., 1.)
        elif delta < 0:
            delta = -policy["max_negative_delta_mps"]
        if abs(delta) < policy["minimum_speed_mae_mps"]:
            delta = 0.
        value = dict(row, speed_delta_mps=round(float(delta), 8))
        key = tuple(value[k] for k in ("longitudinal_m", "lateral_m", "speed_delta_mps"))
        if key not in seen:
            seen.add(key)
            rows.append(value)
        if len(rows) == count:
            break
    return rows


def pair_speed(first_xy, second_xy, policy):
    if policy != policy_from_facts(policy["facts"]):
        raise ValueError("Non-replayable speed thresholds")
    a, b = [np.linalg.norm(np.diff(np.asarray(x, dtype=float), axis=0), axis=1)/.1
            for x in (first_xy, second_xy)]
    difference = np.abs(a-b)
    threshold = policy["minimum_speed_mae_mps"]
    active = difference >= threshold
    longest = run = 0
    for value in active:
        run = run+1 if value else 0
        longest = max(longest, run)
    return dict(speed_mae_mps=float(difference.mean()), speed_threshold_mps=threshold,
        speed_max_abs_difference_mps=float(difference.max()),
        speed_difference_longest_seconds=float(longest*.1),
        speed_distinct=bool(difference.mean() >= threshold and longest*.1 >= policy["minimum_above_threshold_seconds"]))
