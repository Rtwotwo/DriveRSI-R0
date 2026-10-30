"""Read trusted NAVSIM logs without importing its training or sensor stack."""

import pickle
from decimal import ROUND_FLOOR, Decimal
from numbers import Real
from pathlib import Path

import numpy as np

from driversi.contracts.scene import MAX_WORLD_ACTORS, SCENE_VERSION, Scene
from driversi.data.participants.motion import source_vehicle_headings

KINDS = {"vehicle": 1, "pedestrian": 2, "bicycle": 3, "cyclist": 3}
SIM_DT_SECONDS = 0.1
BACKEND_MAX_DURATION_SECONDS = 600.0
MAX_SOURCE_INTERVAL_SECONDS = 0.75
REQUIRED = {
    "timestamp", "token", "log_name", "map_location", "ego2global_translation",
    "ego2global_rotation", "anns",
}


class HorizonPlanError(ValueError):
    """A valid horizon request has no supported plan; preserve its diagnostics."""

    def __init__(self, plan: dict):
        self.plan = plan
        super().__init__("Unsupported horizon: " + "; ".join(plan["rejection_reasons"]))


def validate_horizon_request(duration: float, mode: str, min_duration: float) -> None:
    """Reject malformed parameters before catalogues or output directories exist."""
    if not isinstance(mode, str) or mode not in {"exact", "fit_source"}:
        raise ValueError("duration mode must be exact or fit_source")
    for name, value in (("duration", duration), ("min_duration", min_duration)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite number")
    if min_duration < SIM_DT_SECONDS:
        raise ValueError("min_duration must be at least 0.1 seconds")
    if mode == "exact" and (duration < SIM_DT_SECONDS or not np.isclose(
            duration, round(duration, 1), rtol=0, atol=1e-9)):
        raise ValueError("exact duration must be a multiple of 0.1 seconds")


def plan_horizon(frames: list[dict], *, start: int, duration: float,
                 mode: str = "exact", min_duration: float = SIM_DT_SECONDS) -> dict:
    """Select source-supported time, never shorten because a simulation fails.

    Valid requests return a supported or rejected plan. Malformed input raises
    ValueError. A rejected plan has no selected endpoint or planned duration,
    but retains the quantized candidate and limiting conditions for auditing.
    Only source continuity and backend time support are checked here.
    """
    validate_horizon_request(duration, mode, min_duration)
    if not isinstance(frames, list) or not frames:
        raise ValueError("Horizon planning requires a nonempty list of source frames")
    if (isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer))
            or not 0 <= start < len(frames)):
        raise ValueError("Invalid start frame")
    validate_log(frames)
    start = int(start)
    timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.int64)
    first = frames[start]
    end = start
    boundary = []
    for index in range(start + 1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        reasons = []
        if int(timestamps[index]) - int(timestamps[index - 1]) > round(MAX_SOURCE_INTERVAL_SECONDS * 1e6):
            reasons.append("source_gap_exceeds_0.75_seconds")
        if current["log_name"] != previous["log_name"]:
            reasons.append("source_log_boundary")
        if current["map_location"] != previous["map_location"]:
            reasons.append("source_map_boundary")
        if reasons:
            boundary = reasons
            break
        end = index
    if not boundary:
        boundary = ["source_exhaustion"]
    available = float((int(timestamps[end]) - int(timestamps[start])) / 1e6)
    requested = float(duration)
    if mode == "exact":
        requested = round(requested, 1)
    limit = min(requested, available, BACKEND_MAX_DURATION_SECONDS)
    ticks = int((Decimal(str(limit)) / Decimal(str(SIM_DT_SECONDS))).to_integral_value(
        rounding=ROUND_FLOOR))
    candidate = round(ticks * SIM_DT_SECONDS, 1)
    limiting = []
    if available < requested:
        limiting.extend(boundary)
    if BACKEND_MAX_DURATION_SECONDS < requested:
        limiting.append("backend_cap")
    loss = max(0.0, limit - candidate)
    if loss > 0:
        limiting.append("step_quantization")
    rejected = []
    if mode == "exact" and available < requested:
        rejected.append("source_does_not_cover_requested_duration")
        rejected.extend(boundary)
    if mode == "exact" and requested > BACKEND_MAX_DURATION_SECONDS:
        rejected.append("requested_duration_exceeds_backend_cap")
    if candidate < min_duration:
        rejected.append("candidate_below_min_duration")
    accepted = not rejected
    stop = int(np.searchsorted(timestamps, int(timestamps[start]) + round(candidate * 1e6))) if accepted else None
    # This is an invariant of downward quantization plus same-segment bracketing.
    if stop is not None and not start < stop <= end:
        raise ValueError("Planned endpoint is outside the continuous source segment")
    population = None
    if stop is not None:
        track_sets = [set(f["anns"]["track_tokens"]) for f in frames[start:stop + 1]]
        common = set.intersection(*track_sets)
        population = {
            "initial_tracks": len(track_sets[0]), "persistent_tracks": len(common),
            "initial_excluded_incomplete_tracks": len(track_sets[0] - common),
            "new_arriving_tracks": len(set.union(*track_sets) - track_sets[0]),
        }
    return {
        "schema_version": 1, "status": "supported" if accepted else "rejected", "mode": mode,
        "requested_duration_seconds": float(duration), "min_duration_seconds": float(min_duration),
        "planned_duration_seconds": candidate if accepted else None,
        "candidate_duration_seconds": candidate, "available_future_seconds": available,
        "backend_max_duration_seconds": BACKEND_MAX_DURATION_SECONDS,
        "dt_seconds": SIM_DT_SECONDS, "quantization_loss_seconds": loss,
        "limiting_factors": limiting, "rejection_reasons": rejected,
        "source_start_frame": start, "source_start_timestamp_us": int(timestamps[start]),
        "source_end_frame": stop, "source_end_timestamp_us": int(timestamps[stop]) if stop is not None else None,
        "source_end_offset_seconds": float((timestamps[stop] - timestamps[start]) / 1e6) if stop is not None else None,
        "source_segment_end_frame": end, "source_segment_end_timestamp_us": int(timestamps[end]),
        "source_boundary_reasons": boundary, "source_log": first["log_name"],
        "map_name": first["map_location"], "population": population,
        "scope": "time_support_only_not_map_route_population_or_interaction_qualification",
        "population_scope": "raw_annotation_track_ids_non_ego_including_unsupported_kinds",
    }


def read_log(
    path: Path, *, trusted: bool = False, allow_single_frame: bool = False,
) -> list[dict]:
    """Deserialize trusted input; field validation is a separate, required step."""
    if not trusted:
        raise ValueError("NAVSIM uses pickle; pass --trusted-pickle only for your trusted local logs")
    with path.open("rb") as stream:
        frames = pickle.load(stream)
    minimum_frames = 1 if allow_single_frame else 2
    if not isinstance(frames, list) or len(frames) < minimum_frames:
        raise ValueError(f"Expected a NAVSIM list containing at least {minimum_frames} frames")
    return frames


def validate_log(frames: list[dict]) -> None:
    """Validate all fields used by the scene compiler, including annotation shapes."""
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict) or not REQUIRED.issubset(frame):
            raise ValueError(f"Frame {i} is missing required NAVSIM fields")
        timestamp = frame["timestamp"]
        if (isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, (int, np.integer))
                or not np.iinfo(np.int64).min <= timestamp <= np.iinfo(np.int64).max):
            raise ValueError(f"Frame {i} timestamp must be an integer number of microseconds")
        anns = frame["anns"]
        if not {"gt_boxes", "gt_names", "track_tokens"}.issubset(anns):
            raise ValueError(f"Frame {i} has incomplete annotations")
        count = len(anns["track_tokens"])
        if np.asarray(anns["gt_boxes"]).shape != (count, 7) or len(anns["gt_names"]) != count:
            raise ValueError(f"Frame {i} has inconsistent annotation shapes")
        if len(set(anns["track_tokens"])) != count:
            raise ValueError(f"Frame {i} has duplicate track IDs")
    timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.int64)
    if np.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError("NAVSIM timestamps must be strictly increasing")


def load_log(path: Path, *, trusted: bool = False) -> list[dict]:
    frames = read_log(path, trusted=trusted)
    validate_log(frames)
    return frames


def yaw_from_quaternion(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def transform_box(box: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    """NAVSIM box is in the current ego frame; output a global centre and heading."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z, _, _, _, heading = np.asarray(box, dtype=np.float64)
    return np.asarray([
        translation[0] + c * x - s * y, translation[1] + s * x + c * y,
        translation[2] + z, np.arctan2(np.sin(yaw + heading), np.cos(yaw + heading)),
    ])


def audit_logs(paths: list[Path], *, trusted: bool) -> dict:
    records, errors = [], []
    for path in paths:
        try:
            frames = load_log(path, trusted=trusted)
            intervals = np.diff([f["timestamp"] for f in frames]) / 1e6
            records.append({
                "path": str(path), "frames": len(frames),
                "logs": sorted({f["log_name"] for f in frames}),
                "maps": sorted({f["map_location"] for f in frames}),
                "interval_seconds": {
                    "min": float(intervals.min()), "median": float(np.median(intervals)),
                    "max": float(intervals.max()),
                },
                "max_annotated_actors": max(len(f["anns"]["track_tokens"]) for f in frames),
                "frames_with_signal_records": sum(bool(f.get("traffic_lights")) for f in frames),
                "actor_kinds": sorted({str(k) for f in frames for k in f["anns"]["gt_names"]}),
            })
        except (ValueError, KeyError, TypeError, OSError, pickle.UnpicklingError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return {"schema_version": 1, "files_read": len(records), "records": records, "errors": errors}


def audit_horizons(frames: list[dict], *, start: int, durations: list[float]) -> dict:
    """Inspect source support and raw track retention, without compiling a scene.

    A requested endpoint uses the same left search as ``compile_frames``: the
    first source frame at or after that time is included. A gap or identity
    change before that bracketing frame makes the whole window unsupported.
    Compiler support below covers only its duration limit, not map, actor,
    annotation-kind, route or simulation validity.
    """
    if not isinstance(frames, list) or not frames:
        raise ValueError("Horizon audit requires a nonempty list of source frames")
    if (isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer))
            or not 0 <= start < len(frames)):
        raise ValueError("Invalid start frame")
    if not isinstance(durations, list) or not durations:
        raise ValueError("Require a nonempty list of durations")
    for duration in durations:
        if (isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real)
                or not np.isfinite(duration) or duration < SIM_DT_SECONDS
                or not np.isclose(duration, round(duration, 1), rtol=0, atol=1e-9)):
            raise ValueError("Durations must be positive finite multiples of 0.1 seconds")
    validate_log(frames)
    start = int(start)
    timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.int64)
    intervals = np.diff(timestamps) / 1e6
    identities = [(frame["log_name"], frame["map_location"]) for frame in frames]
    boundaries = [0, *(i + 1 for i, gap in enumerate(intervals)
                       if gap > MAX_SOURCE_INTERVAL_SECONDS or identities[i] != identities[i + 1]), len(frames)]
    segments = []
    start_segment = 0
    for index, (first, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        last = stop - 1
        segments.append({
            "start_frame": first, "end_frame": last,
            "start_timestamp_us": int(timestamps[first]), "end_timestamp_us": int(timestamps[last]),
            "source_log": identities[first][0], "map_name": identities[first][1],
            "duration_seconds": float((timestamps[last] - timestamps[first]) / 1e6),
        })
        if first <= start <= last:
            start_segment = index
    segment = segments[start_segment]
    relative_times = (timestamps - timestamps[start]) / 1e6
    initial_tracks = set(frames[start]["anns"]["track_tokens"])
    horizons = []
    for duration in durations:
        duration = float(duration)
        stop = int(np.searchsorted(relative_times, duration, side="left"))
        covered = stop < len(frames)
        reasons = []
        population = None
        if not covered:
            reasons.append("file_does_not_cover_requested_duration")
        else:
            if np.any(intervals[start:stop] > MAX_SOURCE_INTERVAL_SECONDS):
                reasons.append("source_gap_exceeds_0.75_seconds")
            if any(identity[0] != identities[start][0] for identity in identities[start:stop + 1]):
                reasons.append("source_log_boundary")
            if any(identity[1] != identities[start][1] for identity in identities[start:stop + 1]):
                reasons.append("source_map_boundary")
            if not reasons:
                track_sets = [set(frame["anns"]["track_tokens"]) for frame in frames[start:stop + 1]]
                persistent = set.intersection(*track_sets)
                population = {
                    "initial_tracks": len(initial_tracks), "persistent_tracks": len(persistent),
                    "initial_excluded_incomplete_tracks": len(initial_tracks - persistent),
                    "new_arriving_tracks": len(set.union(*track_sets) - initial_tracks),
                }
        horizons.append({
            "duration_seconds": duration, "file_covers_duration": covered,
            "source_continuous": covered and not reasons,
            "compiler_duration_supported": SIM_DT_SECONDS <= duration <= BACKEND_MAX_DURATION_SECONDS,
            "source_end_frame": stop if covered else None,
            "source_end_timestamp_us": int(timestamps[stop]) if covered else None,
            "source_end_offset_seconds": float(relative_times[stop]) if covered else None,
            "source_frames": stop - start + 1 if covered else None,
            "unavailable_reasons": reasons, "population": population,
        })
    return {
        "schema_version": 1, "kind": "navsim_horizon_audit",
        "source": {
            "frame_count": len(frames), "first_timestamp_us": int(timestamps[0]),
            "last_timestamp_us": int(timestamps[-1]),
            "span_seconds": float((timestamps[-1] - timestamps[0]) / 1e6),
            "interval_seconds": {"min": float(intervals.min()), "median": float(np.median(intervals)),
                                 "max": float(intervals.max())} if len(intervals) else None,
            "continuous_segment_count": len(segments), "continuous_segments": segments,
            "longest_continuous_seconds": max(item["duration_seconds"] for item in segments),
        },
        "start": {
            "frame_index": start, "timestamp_us": int(timestamps[start]),
            "source_log": identities[start][0], "map_name": identities[start][1],
            "continuous_segment_index": start_segment,
            "available_history_seconds": float((timestamps[start] - segment["start_timestamp_us"]) / 1e6),
            "available_future_seconds": float((segment["end_timestamp_us"] - timestamps[start]) / 1e6),
            "remaining_file_seconds": float(relative_times[-1]),
        },
        "continuity_policy": {"max_interval_seconds": MAX_SOURCE_INTERVAL_SECONDS, "same_log_and_map": True,
                              "endpoint": "first_source_frame_at_or_after_requested_time"},
        "population_scope": "raw_annotation_track_ids_non_ego_including_unsupported_kinds",
        "compiler": {"min_duration_seconds": SIM_DT_SECONDS,
                     "max_duration_seconds": BACKEND_MAX_DURATION_SECONDS, "dt_seconds": SIM_DT_SECONDS,
                     "scope": "duration_constraint_only_not_full_compilability"},
        "limitations": ["history_reported_not_loaded_into_scene", "map_and_sensors_not_checked",
                        "population_counts_are_not_controlled_vehicle_counts"],
        "horizons": horizons,
    }


def _compile_persistent_frames(
    frames: list[dict], *, start: int, duration: float, split: str, max_actors: int = 32,
    duration_mode: str = "exact", min_duration: float = SIM_DT_SECONDS,
    allow_stationary_ego: bool = False,
) -> Scene:
    """Build an interpolated, fully tracked initial population. Map is attached separately."""
    if (isinstance(max_actors, (bool, np.bool_)) or not isinstance(max_actors, (int, np.integer))
            or not 1 <= max_actors <= 32):
        raise ValueError("Invalid max_actors")
    plan = plan_horizon(frames, start=start, duration=duration, mode=duration_mode,
                        min_duration=min_duration)
    if plan["status"] != "supported":
        raise HorizonPlanError(plan)
    start = plan["source_start_frame"]
    duration = plan["planned_duration_seconds"]
    if split not in {"mini", "train", "development", "heldout", "unassigned"}:
        raise ValueError("An explicit supported data split is required")
    all_times = (np.asarray([f["timestamp"] for f in frames], np.int64) - frames[start]["timestamp"]) / 1e6
    stop = plan["source_end_frame"]
    window = frames[start:stop + 1]
    if len({(f["log_name"], f["map_location"]) for f in window}) != 1:
        raise ValueError("A scene cannot cross log or map boundaries")
    source_times = all_times[start:stop + 1]
    if np.any(np.diff(source_times) > MAX_SOURCE_INTERVAL_SECONDS):
        raise ValueError("A source gap exceeds 0.75 seconds; interpolation is not justified")
    indexes = [{token: i for i, token in enumerate(f["anns"]["track_tokens"])} for f in window]
    common = set.intersection(*(set(index) for index in indexes))
    initial = window[0]
    origin = np.asarray(initial["ego2global_translation"], dtype=np.float64)
    eligible = []
    for token in common:
        i = indexes[0][token]
        kind = str(initial["anns"]["gt_names"][i])
        if kind not in KINDS:
            continue
        if any(str(f["anns"]["gt_names"][index[token]]) != kind
               for f, index in zip(window, indexes)):
            raise ValueError(f"Actor kind changes within the window: {token}")
        eligible.append((float(np.linalg.norm(initial["anns"]["gt_boxes"][i][:2])), token))
    eligible.sort()
    chosen = [token for _, token in eligible[:max_actors - 1]]
    ids = ("ego", *chosen)
    n = len(ids)
    source_poses = np.empty((len(window), n, 4), dtype=np.float64)
    dimensions = np.empty((n, 3), dtype=np.float64)
    kinds = np.ones(n, dtype=np.int32)
    # nuPlan Pacifica: front=4.049, rear=1.127 measured from rear axle.
    dimensions[0] = [5.176, 2.297, 1.777]
    for a, token in enumerate(chosen, 1):
        i = indexes[0][token]
        dimensions[a] = initial["anns"]["gt_boxes"][i][3:6]
        kinds[a] = KINDS[str(initial["anns"]["gt_names"][i])]
    for t, (frame, index) in enumerate(zip(window, indexes)):
        translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
        yaw = yaw_from_quaternion(frame["ego2global_rotation"])
        source_poses[t, 0] = [
            translation[0] + 1.461 * np.cos(yaw), translation[1] + 1.461 * np.sin(yaw),
            translation[2], yaw,
        ]
        for a, token in enumerate(chosen, 1):
            source_poses[t, a] = transform_box(frame["anns"]["gt_boxes"][index[token]], translation, yaw)
    source_poses[..., :3] -= origin
    source_poses[..., 3], heading_audit = source_vehicle_headings(
        source_poses, source_times, ids, kinds)
    times = np.arange(round(duration / SIM_DT_SECONDS) + 1, dtype=np.float64) / round(1 / SIM_DT_SECONDS)
    poses = np.empty((len(times), n, 4), dtype=np.float64)
    for a in range(n):
        for c in range(4):
            poses[:, a, c] = np.interp(times, source_times, source_poses[:, a, c])
    poses[..., 3] = np.arctan2(np.sin(poses[..., 3]), np.cos(poses[..., 3]))
    velocities = np.gradient(poses[..., :3], SIM_DT_SECONDS, axis=0)
    displacement = np.linalg.norm(poses[-1, :, :2] - poses[0, :, :2], axis=1)
    controlled = (kinds == 1) & (displacement > 2.0)
    needs_task = []
    if not controlled.any():
        if not allow_stationary_ego:
            raise ValueError("No moving vehicle with a goal more than 2 metres away")
        controlled[0] = True
        needs_task = ["ego"]

    metadata = {
        "schema_version": SCENE_VERSION, "scenario_id": str(initial["token"]),
        "source_log": initial["log_name"], "map_name": initial["map_location"], "split": split,
        "split_verification": "unverified_manual_label",
        "origin_xyz": origin.tolist(), "coordinate_frame": "map_axes_translated_from_initial_ego",
        "pose_reference": "vehicle_center", "source_start_frame": start,
        "source_start_timestamp_us": int(initial["timestamp"]), "source_frames": len(window),
        "time_model": "linear_position_unwrapped_yaw_interpolation_10hz",
        "velocity_model": "position_finite_difference", "map_complete": False,
        "excluded_incomplete_tracks": len(set(indexes[0]) - common),
        "excluded_over_capacity": max(0, len(eligible) - len(chosen)),
        "excluded_unsupported_initial": sum(str(k) not in KINDS for k in initial["anns"]["gt_names"]),
        "traffic_lights": [
            {"time": float(t), "available": "traffic_lights" in f,
             "records": [[str(i), bool(red)] for i, red in f.get("traffic_lights", [])]}
            for t, f in zip(source_times, window)
        ],
        "horizon_plan": plan,
        "heading_audit": heading_audit,
        "source_timing": {
            "sample_count": len(source_times),
            "interval_seconds": {
                "min": float(np.diff(source_times).min()),
                "median": float(np.median(np.diff(source_times))),
                "max": float(np.diff(source_times).max()),
            },
            "target_dt_seconds": SIM_DT_SECONDS,
            "interpolation": "linear_position_and_audited_unwrapped_yaw",
        },
        "population_policy": "initial_population_full_window_track_intersection",
        "control_selection_policy": ("stationary_ego_requires_explicit_task" if needs_task else
                                     "vehicle_with_window_endpoint_displacement_greater_than_2m"),
        **({"unbound_controlled_tasks": needs_task} if needs_task else {}),
        "limitations": ["no_spawn_despawn", "no_signal_dynamics", "not_native_10hz_logs"],
    }
    scene = Scene(
        metadata, ids, kinds, dimensions, times, poses, velocities, controlled,
        np.empty((0, 3)), np.asarray([0], np.int64), np.empty(0, np.int32),
    )
    scene.validate()
    return scene



def source_motion_evidence(window, actor_ids):
    """Measured first-run motion for privileged traffic, never student input.

    NAVSIM annotations may drift while measured velocity remains zero. Preserve
    both signals for the existing fleet rule; do not infer zero from missing data
    or merge a disappearance and later reappearance into one supported run.
    """
    allowed, ended, active, evidence = set(actor_ids), set(), set(), {}
    for frame in window:
        anns = frame['anns']
        ids = [str(t) for t in anns['track_tokens']]
        ended.update(active - set(ids))
        active = set(ids)
        yaw = yaw_from_quaternion(frame['ego2global_rotation'])
        velocities = anns.get('gt_velocity_3d')
        timestamp = int(frame['timestamp'])
        for index, actor in enumerate(ids):
            if actor not in allowed or actor in ended:
                continue
            position = transform_box(anns['gt_boxes'][index], frame['ego2global_translation'], yaw)[:2]
            velocity = None if velocities is None else np.asarray(velocities[index], dtype=float)
            speed = (float(np.linalg.norm(velocity[:2])) if velocity is not None
                     and velocity.shape == (3,) and np.isfinite(velocity).all() else None)
            item = evidence.setdefault(actor, dict(
                samples=0, finite_velocity_samples=0, max_reported_speed_mps=0.0,
                max_position_excursion_m=0.0, reported_travel_m=0.0,
                position_path_length_m=0.0, first_position=position.tolist(),
                first_timestamp_us=timestamp))
            if 'last_position' in item:
                item['position_path_length_m'] += float(np.linalg.norm(position - item['last_position']))
                if speed is not None and item['last_speed'] is not None:
                    item['reported_travel_m'] += .5 * (speed + item['last_speed']) * (timestamp - item['last_timestamp_us']) * 1e-6
            item.update(last_position=position.tolist(), last_speed=speed, last_timestamp_us=timestamp)
            item['samples'] += 1
            item['max_position_excursion_m'] = max(item['max_position_excursion_m'], float(np.linalg.norm(position-item['first_position'])))
            if speed is not None:
                item['finite_velocity_samples'] += 1
                item['max_reported_speed_mps'] = max(item['max_reported_speed_mps'], speed)
    for item in evidence.values():
        for key in ('first_position', 'last_position', 'last_speed'):
            item.pop(key)
    return dict(version=1, scope='first_contiguous_run_privileged_navigation', actors=evidence)


def compile_frames(frames, *, start, duration, split, max_actors=32,
                   duration_mode="exact", min_duration=SIM_DT_SECONDS,
                   population_mode="logged", max_world_actors=MAX_WORLD_ACTORS,
                   allow_stationary_ego=False):
    """Compile selected policy cars plus the complete logged participant union.

    max_actors bounds initial policy selection; max_world_actors bounds all IDs.
    Persistent mode is retained only for reproducing existing population ablations.
    """
    if type(allow_stationary_ego) is not bool:
        raise ValueError("allow_stationary_ego must be boolean")
    if population_mode not in {"logged", "persistent"}:
        raise ValueError("Unknown population mode")
    scene = _compile_persistent_frames(frames, start=start, duration=duration, split=split,
        max_actors=max_actors, duration_mode=duration_mode, min_duration=min_duration,
        allow_stationary_ego=allow_stationary_ego)
    if population_mode == "persistent":
        return scene
    from driversi.data.participants.lifecycle import attach_logged_population
    result = attach_logged_population(scene, frames, max_world_actors=max_world_actors)
    first = scene.metadata["source_start_frame"]
    window = frames[first:first + scene.metadata["source_frames"]]
    result.metadata["source_motion_evidence"] = source_motion_evidence(window, result.actor_ids)
    return result
