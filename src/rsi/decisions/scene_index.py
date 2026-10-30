"""History-only NAVSIM scene descriptions and independent navtrain retrieval.

Current/history annotations are privileged pool metadata, never student inputs.
Retrieval proposes observable scenario matches; it does not establish a failure.
The serialized native format requires one whole log in memory, never the corpus.
"""

import hashlib
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from driversi.agents.backends.native import project_root
from driversi.rsi.decisions.capability_feedback import CAPABILITIES, priorities
from driversi.training.sampling.environment import source_log_group

CONDITION_TAGS = frozenset(
    {
        "traffic_sparse",
        "traffic_moderate",
        "traffic_dense",
        "nearby_vehicle",
        "nearby_static",
        "ego_stopped",
        "ego_low_speed",
        "ego_moving",
        "history_speed_change",
        "history_turning",
        "nav_command_0",
        "nav_command_1",
        "nav_command_2",
        "nav_command_3",
    }
)
CONDITIONS = CONDITION_TAGS
DYNAMIC_CLASSES = frozenset({"vehicle", "pedestrian", "bicycle"})
STATIC_CLASSES = frozenset({"generic_object", "traffic_cone", "barrier", "czone_sign"})
# Heuristics describe exposure, not difficulty, causality or compliance.
HEURISTIC_TAGS = {
    "collision": {"nearby_vehicle", "nearby_static", "traffic_moderate", "traffic_dense"},
    "interaction": {"nearby_vehicle", "traffic_moderate", "traffic_dense"},
    "comfort": {"history_speed_change", "history_turning"},
    "route_progress": {"ego_stopped", "ego_low_speed"},
}
FEATURE_KEYS = {
    "dynamic_count_30m",
    "vehicle_count_30m",
    "pedestrian_count_30m",
    "bicycle_count_30m",
    "static_count_30m",
    "nearby_vehicle_count_15m",
    "nearby_static_count_15m",
    "ego_speed_mps",
    "history_speed_range_mps",
    "history_yaw_range_rad",
    "nav_command_slot",
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split(path=None):
    path = (
        Path(path)
        if path
        else project_root()
        / (
            "agents/chainflow/native/navsim/planning/script/config/common/"
            "train_test_split/scene_filter/navtrain.yaml"
        )
    )
    value = yaml.safe_load(path.read_text())
    logs, tokens = value.get("log_names"), value.get("tokens")
    if (
        not isinstance(logs, list)
        or not logs
        or not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(x, str) or not x for x in logs + tokens)
    ):
        raise ValueError("Official navtrain exact log AND token allowlists required")
    return path.resolve(), set(logs), set(tokens)


def _number_array(value, shape=None):
    try:
        array = np.asarray(value, dtype=float)
        if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
            return None
        return array
    except (TypeError, ValueError):
        return None


def extract_history_descriptors(history):
    """Describe exactly the provided history window, ending at the current anchor."""
    if not isinstance(history, (list, tuple)) or not history:
        raise ValueError("Nonempty current/history frame sequence required")
    features = dict.fromkeys(sorted(FEATURE_KEYS))
    tags, unknown = set(), set()
    anchor = history[-1]
    anns = anchor.get("anns", {})
    boxes, names = _number_array(anns.get("gt_boxes")), anns.get("gt_names")
    if (
        boxes is not None
        and boxes.ndim == 2
        and boxes.shape[1] >= 2
        and names is not None
        and len(names) == len(boxes)
    ):
        radius = np.linalg.norm(boxes[:, :2], axis=1)
        names = np.asarray(names, dtype=str)
        dynamic = np.isin(names, sorted(DYNAMIC_CLASSES))
        static = np.isin(names, sorted(STATIC_CLASSES))
        features["dynamic_count_30m"] = int(np.sum(dynamic & (radius <= 30)))
        features["static_count_30m"] = int(np.sum(static & (radius <= 30)))
        for name in ("vehicle", "pedestrian", "bicycle"):
            features[name + "_count_30m"] = int(np.sum((names == name) & (radius <= 30)))
        features["nearby_vehicle_count_15m"] = int(np.sum((names == "vehicle") & (radius <= 15)))
        features["nearby_static_count_15m"] = int(np.sum(static & (radius <= 15)))
        count = features["dynamic_count_30m"]
        tags.add(
            "traffic_sparse"
            if count <= 2
            else "traffic_moderate"
            if count <= 7
            else "traffic_dense"
        )
        for name in ("vehicle", "static"):
            if features["nearby_" + name + "_count_15m"]:
                tags.add("nearby_" + name)
        if any(name not in DYNAMIC_CLASSES | STATIC_CLASSES for name in names):
            unknown.add("unrecognized_actor_classes")
    else:
        unknown.add("current_actor_annotations")

    speeds, yaws = [], []
    for frame in history:
        dynamic = _number_array(frame.get("ego_dynamic_state"))
        if dynamic is not None and dynamic.ndim == 1 and len(dynamic) >= 2:
            speeds.append(float(np.linalg.norm(dynamic[:2])))
        rotation = _number_array(frame.get("ego2global_rotation"), (4,))
        if rotation is not None and np.linalg.norm(rotation) > 0:
            w, x, y, z = rotation / np.linalg.norm(rotation)
            # Match NAVSIM pyquaternion.yaw_pitch_roll (w, x, y, z).
            yaws.append(math.atan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z)))
    current = _number_array(anchor.get("ego_dynamic_state"))
    if current is not None and current.ndim == 1 and len(current) >= 2:
        speed = features["ego_speed_mps"] = float(np.linalg.norm(current[:2]))
        tags.add("ego_stopped" if speed < 0.5 else "ego_low_speed" if speed < 3 else "ego_moving")
    else:
        unknown.add("current_ego_speed")
    if len(speeds) == len(history) and len(history) >= 2:
        features["history_speed_range_mps"] = max(speeds) - min(speeds)
        if features["history_speed_range_mps"] >= 1:
            tags.add("history_speed_change")
    else:
        unknown.add("history_ego_speed")
    if len(yaws) == len(history) and len(history) >= 2:
        features["history_yaw_range_rad"] = float(np.ptp(np.unwrap(yaws)))
        if features["history_yaw_range_rad"] >= 0.1:
            tags.add("history_turning")
    else:
        unknown.add("history_ego_yaw")
    command = _number_array(anchor.get("driving_command"), (4,))
    if command is not None and np.all((command == 0) | (command == 1)) and np.sum(command) == 1:
        slot = features["nav_command_slot"] = int(np.argmax(command))
        tags.add("nav_command_" + str(slot))
    else:
        unknown.add("navigation_command")
    return {"features": features, "conditions": sorted(tags), "unknown": sorted(unknown)}


def _identity(row, history_frames):
    token, source = row.get("token"), row.get("source_log")
    if (
        not isinstance(token, str)
        or not token
        or not isinstance(source, str)
        or not source
        or type(row.get("anchor_index")) is not int
        or row["anchor_index"] < history_frames - 1
    ):
        raise ValueError("Actual token, source log and sufficient history anchor required")
    path = Path(row["log_path"]).resolve(strict=True)
    if path.name != source + ".pkl":
        raise ValueError("Physical source filename differs from declared source log")
    return token, source, path


def _describe(row, frames, history_frames, allow_irregular_history=False):
    token, source, path = _identity(row, history_frames)
    anchor = row["anchor_index"]
    if anchor >= len(frames):
        raise ValueError("Anchor outside physical source log")
    # Deliberately no access to frames after the anchor.
    history = frames[anchor - history_frames + 1 : anchor + 1]
    if (
        len(history) != history_frames
        or history[-1].get("token") != token
        or any(frame.get("log_name") != source for frame in history)
    ):
        raise ValueError("Raw history token/log identity differs from declaration")
    times = [frame.get("timestamp") for frame in history]
    if any(not isinstance(t, (int, np.integer)) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Continuous increasing NAVSIM history timestamps required")
    gaps = [int(b - a) for a, b in zip(times, times[1:])]
    continuous = all(450000 <= gap <= 550000 for gap in gaps)
    if not continuous and not allow_irregular_history:
        raise ValueError("Continuous 0.5 second NAVSIM history required")
    # Official originals can contain missing samples. Retain their population
    # membership, but do not infer temporal features across a missing interval.
    described = extract_history_descriptors(history if continuous else history[-1:])
    if not continuous:
        described["unknown"] = sorted(set(described["unknown"]) | {"history_sampling_gap"})
    return {
        "token": token,
        "source_token": token,
        "sample_id": token,
        "source_log": source,
        "source_group": source_log_group(source),
        "log_path": str(path),
        "anchor_index": anchor,
        "history": {
            "frames": history_frames,
            "start_timestamp_us": int(times[0]),
            "anchor_timestamp_us": int(times[-1]),
            "information_set": "current_state_history",
            "continuous": continuous,
            "actual_intervals_us": gaps,
        },
        **described,
    }


def _check_window(history_frames):
    if type(history_frames) is not int or history_frames < 2:
        raise ValueError("At least two current/history frames required")


def describe_records(records, history_frames=4, *, allow_irregular_history=False):
    """Describe trusted raw records for aggregation; this does not admit any split."""
    _check_window(history_frames)
    rows, seen, loaded_path, frames = [], set(), None, None
    for row in records:
        token, _, path = _identity(row, history_frames)
        if token in seen:
            raise ValueError("Duplicate scene token")
        seen.add(token)
        if loaded_path != path:
            # Release the previous log before unpickling the next one.
            frames = None
            with path.open("rb") as stream:
                frames = pickle.load(stream)
            loaded_path = path
        rows.append(_describe(row, frames, history_frames, allow_irregular_history))
    return rows


def _allowed(row, allowed_logs, allowed_tokens, excluded):
    if (
        row.get("source_log") not in allowed_logs
        or row.get("token") not in allowed_tokens
        or source_log_group(row["source_log"]) in excluded
        or row.get("source_split", "navtrain") != "navtrain"
        or row.get("kind", "original_navtrain") != "original_navtrain"
    ):
        raise ValueError("Scene is outside official, unprotected navtrain token/log membership")


def build_index(
    records=None,
    *,
    official_split_path=None,
    development_groups,
    protected_groups,
    history_frames=4,
    log_root=None,
    max_scenes=None,
    allow_irregular_history=False,
):
    """Index supplied anchors, or stream official log files one at a time.

    Pickle input must be trusted local NAVSIM data. Exact official token membership
    is combined with raw token/log identity; a declared split alone never admits.
    """
    _check_window(history_frames)
    if max_scenes is not None and (type(max_scenes) is not int or max_scenes <= 0):
        raise ValueError("Positive bounded scene limit required")
    if (records is None) == (log_root is None):
        raise ValueError("Provide records OR a log root")
    split_path, allowed_logs, allowed_tokens = _split(official_split_path)
    excluded = {source_log_group(x) for x in [*development_groups, *protected_groups]}
    implicit_protected = []
    for name in ("navtest.yaml", "navhard.yaml"):
        path = split_path.with_name(name)
        if path.is_file():
            value = yaml.safe_load(path.read_text())
            logs = value.get("log_names") or []
            excluded.update(source_log_group(x) for x in logs)
            implicit_protected.append({"path": str(path), "sha256": _sha(path)})
    result, seen, files, skipped = [], set(), {}, Counter()

    def append(row, frames):
        _allowed(row, allowed_logs, allowed_tokens, excluded)
        if row["token"] in seen:
            raise ValueError("Duplicate scene token")
        described = _describe(row, frames, history_frames, allow_irregular_history)
        seen.add(row["token"])
        result.append(described)

    if records is not None:
        loaded_path, frames = None, None
        for row in records:
            _allowed(row, allowed_logs, allowed_tokens, excluded)
            _, _, path = _identity(row, history_frames)
            if path != loaded_path:
                frames = None
                with path.open("rb") as stream:
                    frames = pickle.load(stream)
                files[str(path)] = _sha(path)
                loaded_path = path
            append(row, frames)
            if max_scenes is not None and len(result) >= max_scenes:
                break
    else:
        root = Path(log_root).resolve(strict=True)
        for source in sorted(allowed_logs):
            if source_log_group(source) in excluded:
                skipped["excluded_log_groups"] += 1
                continue
            path = root / (source + ".pkl")
            if not path.is_file():
                skipped["missing_log_files"] += 1
                continue
            with path.open("rb") as stream:
                frames = pickle.load(stream)
            files[str(path)] = _sha(path)
            for anchor, frame in enumerate(frames):
                if frame.get("token") not in allowed_tokens:
                    continue
                row = {
                    "token": frame["token"],
                    "source_log": source,
                    "log_path": str(path),
                    "anchor_index": anchor,
                }
                append(row, frames)
                if max_scenes is not None and len(result) >= max_scenes:
                    break
            del frames
            if max_scenes is not None and len(result) >= max_scenes:
                break
    result.sort(key=lambda x: (x["source_log"], x["token"]))
    return {
        "schema_version": 1,
        "kind": "navtrain_history_scene_index",
        "selection_information_set": "current_state_history",
        "policy": {
            "official_split_path": str(split_path),
            "official_split_sha256": _sha(split_path),
            "excluded_source_groups": sorted(excluded),
            "implicit_protected_splits": implicit_protected,
        },
        "records": result,
        "source_files": files,
        "coverage": {
            "indexed_scenes": len(result),
            "source_logs": len(files),
            "conditions": dict(sorted(Counter(t for r in result for t in r["conditions"]).items())),
            "unknown": dict(sorted(Counter(t for r in result for t in r["unknown"]).items())),
            "skipped": dict(skipped),
            "scope": "supplied_records" if records is not None else "streamed_official_logs",
            "max_scenes": max_scenes,
        },
    }


def _validated_cohorts(cohorts):
    result = []
    seen = set()
    for row in cohorts or []:
        if (
            not isinstance(row, dict)
            or set(row) != {"capability", "conditions", "count", "failures"}
            or row["capability"] not in CAPABILITIES
            or not isinstance(row["conditions"], list)
            or not row["conditions"]
            or any(t not in CONDITION_TAGS for t in row["conditions"])
            or len(set(row["conditions"])) != len(row["conditions"])
            or type(row["count"]) is not int
            or type(row["failures"]) is not int
            or not 0 <= row["failures"] <= row["count"]
            or row["count"] <= 0
        ):
            raise ValueError("Sanitized bounded capability/condition cohort required")
        key = (row["capability"], tuple(sorted(row["conditions"])))
        if key in seen:
            raise ValueError("Duplicate capability cohort")
        seen.add(key)
        result.append(row)
    if len(result) > 256:
        raise ValueError("At most 256 aggregate cohorts allowed")
    return result


def retrieve(index, capability_feedback, limit, cohorts=None, *, per_source_limit=None,
             require_continuous_history=False, capability_budget=None):
    """Rank independent train proposals; all need current-checkpoint verification."""
    weights = priorities(capability_feedback)
    if type(limit) is not int or limit <= 0:
        raise ValueError("Positive retrieval limit required")
    if capability_budget is not None and (
        not isinstance(capability_budget, dict) or not capability_budget
        or any(c not in weights or type(n) is not int or n <= 0
               for c, n in capability_budget.items())
        or sum(capability_budget.values()) != limit
    ):
        raise ValueError("Capability budget must sum to limit and use observed shortfall capabilities")
    if per_source_limit is not None and (type(per_source_limit) is not int or per_source_limit <= 0):
        raise ValueError("Positive per-source candidate limit required")
    if (
        index.get("kind") != "navtrain_history_scene_index"
        or index.get("schema_version") != 1
        or index.get("selection_information_set") != "current_state_history"
    ):
        raise ValueError("History-only navtrain index required")
    policy = index["policy"]
    path, logs, tokens = _split(policy["official_split_path"])
    if _sha(path) != policy["official_split_sha256"]:
        raise ValueError("Official navtrain allowlist changed")
    excluded = set(policy["excluded_source_groups"])
    for protected in policy["implicit_protected_splits"]:
        if _sha(protected["path"]) != protected["sha256"]:
            raise ValueError("Protected allowlist changed")
        value = yaml.safe_load(Path(protected["path"]).read_text())
        excluded.update(source_log_group(x) for x in value.get("log_names") or [])
    seen, rows = set(), index["records"]
    for row in rows:
        _allowed(row, logs, tokens, excluded)
        if (
            row["token"] in seen
            or row["source_group"] != source_log_group(row["source_log"])
            or set(row["features"]) != FEATURE_KEYS
            or not set(row["conditions"]) <= CONDITION_TAGS
            or row["history"]["information_set"] != "current_state_history"
        ):
            raise ValueError("Invalid, duplicate or contaminated indexed scene")
        seen.add(row["token"])
    cohort_rows = _validated_cohorts(cohorts)
    cohort_rows = [r for r in cohort_rows if r["failures"] and r["capability"] in weights]
    base_rates = {r["capability"]: r["failures"] / r["count"]
                  for r in capability_feedback["slices"] if r["count"]}
    candidates, covered, exact_covered = [], Counter(), Counter()
    for row in rows:
        if require_continuous_history and row["history"].get("continuous") is not True:
            continue
        tags = set(row["conditions"])
        exact = [c for c in cohort_rows if set(c["conditions"]) <= tags]
        broad = sorted(c for c in weights if tags & HEURISTIC_TAGS.get(c, set()))
        matched = sorted({c["capability"] for c in exact} | set(broad))
        if not matched:
            continue
        exact_caps = {c["capability"] for c in exact}
        # Every common tag can have some failures. Matching any such tag alone
        # made all candidates tie and degenerated into alphabetical selection.
        # Shrink each conditional rate toward its global rate with 20 evidence
        # units, then take one maximum per capability (correlated tags must not
        # be added as independent evidence). This is a retrieval heuristic only.
        risk = {cap: max([(c["failures"] + 20 * base_rates[cap]) / (c["count"] + 20)
                         for c in exact if c["capability"] == cap],
                        default=base_rates[cap]) for cap in matched}
        score = sum(weights[c] * risk[c] for c in matched)
        candidates.append(
            {
                **row,
                "status": "proposed",
                "requires_model_verification": True,
                "match": {
                    "mode": "exact_observable_cohort" if exact else "capability_heuristic_fallback",
                    "capabilities": matched,
                    "cohorts": [
                        {"capability": c["capability"], "conditions": c["conditions"]}
                        for c in exact
                    ],
                    "fallback_capabilities": sorted(set(broad) - exact_caps),
                    "score": score,
                    "conditional_risk": risk,
                    "ranking": "max_shrunk_conditional_shortfall_rate_per_capability",
                    "interpretation": "observable exposure only; failure not established",
                },
            }
        )
        covered.update(matched)
        exact_covered.update(exact_caps)
    candidates.sort(
        key=lambda r: (
            r["match"]["mode"] != "exact_observable_cohort",
            -r["match"]["score"],
            r["source_log"],
            r["token"],
        )
    )
    selected, source_counts, assigned_counts = [], Counter(), Counter()
    if capability_budget is None:
        for row in candidates:
            group = row["source_group"]
            if per_source_limit is not None and source_counts[group] >= per_source_limit:
                continue
            selected.append(row)
            source_counts[group] += 1
            if len(selected) == limit:
                break
    else:
        # Reserve separate opportunities for rare safety shortfalls. A large
        # progress failure rate must not consume every verification slot.
        queues = {}
        for cap in sorted(capability_budget):
            queue = [r for r in candidates if cap in r["match"]["capabilities"]]
            queue.sort(key=lambda r: r["token"])
            random.Random("navtrain-capability-v1:" + cap).shuffle(queue)
            queue.sort(key=lambda r: -r["match"]["conditional_risk"][cap])
            queues[cap] = iter(queue)
        selected_tokens = set()
        while True:
            added = False
            for cap, queue in queues.items():
                if assigned_counts[cap] >= capability_budget[cap]:
                    continue
                for row in queue:
                    group = row["source_group"]
                    if row["token"] in selected_tokens or (
                        per_source_limit is not None and source_counts[group] >= per_source_limit
                    ):
                        continue
                    selected.append({**row, "match": {**row["match"], "target_capability": cap}})
                    selected_tokens.add(row["token"])
                    source_counts[group] += 1
                    assigned_counts[cap] += 1
                    added = True
                    break
            if not added:
                break
    return {
        "schema_version": 1,
        "kind": "navtrain_candidate_proposals",
        "checkpoint": capability_feedback["checkpoint"],
        "candidates": selected,
        "coverage": {
            "pool_scenes": len(rows),
            "eligible_scenes": len(candidates),
            "selected_scenes": len(selected),
            "selected_source_groups": len(source_counts),
            "per_source_limit": per_source_limit,
            "capability_budget": capability_budget,
            "assigned_by_capability": dict(assigned_counts),
            "unfilled_capability_budget": {
                c: n - assigned_counts[c] for c, n in (capability_budget or {}).items()
                if n > assigned_counts[c]
            },
            "ranking_policy": "per_capability_risk_round_robin" if capability_budget else "weighted_risk",
            "tie_break_seed": "navtrain-capability-v1:<capability>" if capability_budget else None,
            "excluded_irregular_history": sum(r["history"].get("continuous") is not True for r in rows)
            if require_continuous_history else 0,
            "selected_by_capability": dict(
                Counter(c for r in selected for c in r["match"]["capabilities"])
            ),
            "unselected_capabilities": sorted(
                set(weights) - {c for r in selected for c in r["match"]["capabilities"]}
            ),
            "by_capability": dict(sorted(covered.items())),
            "exact_cohort_by_capability": dict(sorted(exact_covered.items())),
            "uncovered_capabilities": sorted(set(weights) - set(covered)),
            "unsupported_heuristic_capabilities": sorted(set(weights) - set(HEURISTIC_TAGS)),
            "unmatched_cohorts": [
                c
                for c in cohort_rows
                if not any(set(c["conditions"]) <= set(r["conditions"]) for r in rows)
            ],
            "unknown_descriptors": index["coverage"]["unknown"],
        },
        "purpose": "independent_training_candidates_requiring_actual_model_verification",
    }
