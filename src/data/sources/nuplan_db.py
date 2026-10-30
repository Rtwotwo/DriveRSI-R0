"""Bounded, read-only nuPlan v1.1 SQLite input; no SDK, pickle or sensor load."""

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np

from driversi.data.sources.navsim import (
    BACKEND_MAX_DURATION_SECONDS,
    compile_frames,
    validate_horizon_request,
    validate_log,
    yaw_from_quaternion,
)

MAX_FRAMES = 13000
MAX_GAP_US = 150000
MAX_POSE_ASSOCIATION_SKEW_US = 50000
CATEGORIES = {
    "vehicle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
}


@contextmanager
def open_db(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix != ".db":
        raise ValueError("Expected an existing nuPlan .db file")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        yield conn


def raw_token(value):
    try:
        result = bytes.fromhex(value)
    except (TypeError, ValueError):
        raise ValueError("Expected hexadecimal lidar token") from None
    if len(result) != 8:
        raise ValueError("nuPlan lidar token must contain 8 bytes")
    return result


def log_header(conn):
    rows = conn.execute("SELECT token,logfile,location,map_version FROM log").fetchall()
    if len(rows) != 1:
        raise ValueError("Expected exactly one original log per DB")
    row = rows[0]
    if not row["logfile"] or not row["map_version"]:
        raise ValueError("Missing or inconsistent nuPlan log/map identity")
    return {
        "source_log": row["logfile"],
        "log_token": row["token"].hex(),
        "map_name": row["map_version"],
        "source_location": row["location"],
    }


def _finite(values, name):
    value = np.asarray(values, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite nuPlan {name}")
    return value


def read_window(db_path, *, token, duration=9.0, history_seconds=1.5):
    """Keep measured past and future separate; source frames bracket the 10Hz horizon.

    No interpolation, padding, source object filtering or future-observation assembly
    occurs here. The shared compiler owns interpolation and participant lifecycles.
    """
    validate_horizon_request(duration, "fit_source", 0.1)
    if (
        isinstance(history_seconds, bool)
        or not np.isfinite(history_seconds)
        or not 0 <= history_seconds <= 5
    ):
        raise ValueError("history_seconds must be finite and between 0 and 5")
    path = Path(db_path).resolve(strict=True)
    before = path.stat()
    with open_db(path) as conn:
        header = log_header(conn)
        anchor = conn.execute(
            "SELECT * FROM lidar_pc WHERE token=?", (raw_token(token),)
        ).fetchone()
        if anchor is None:
            raise ValueError("Lidar token not found in this DB")
        owner = conn.execute(
            "SELECT log_token FROM lidar WHERE token=?", (anchor["lidar_token"],)
        ).fetchone()
        if owner is None or owner[0].hex() != header["log_token"]:
            raise ValueError("Lidar stream does not belong to source log")
        anchor_us = int(anchor["timestamp"])
        first_us = anchor_us - round(history_seconds * 1e6)
        # One source interval on each side permits true timestamp bracketing.
        rows = conn.execute(
            """SELECT p.token AS pc_token,p.timestamp AS pc_us,p.scene_token,
            e.* FROM lidar_pc p LEFT JOIN ego_pose e ON p.ego_pose_token=e.token
            WHERE p.lidar_token=? AND p.timestamp BETWEEN ? AND ?
            ORDER BY p.timestamp LIMIT ?""",
            (
                anchor["lidar_token"],
                first_us - MAX_GAP_US,
                anchor_us + round(min(duration, BACKEND_MAX_DURATION_SECONDS) * 1e6) + MAX_GAP_US,
                MAX_FRAMES + 1,
            ),
        ).fetchall()
        if len(rows) > MAX_FRAMES:
            raise ValueError("nuPlan window exceeds bounded source frame capacity")
        times = np.asarray([row["pc_us"] for row in rows], dtype=np.int64)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("Insufficient or non-monotonic nuPlan source frames")
        left = int(np.searchsorted(times, first_us, side="right") - 1)
        if left < 0:
            raise ValueError("Insufficient measured history; padding is forbidden")
        right = min(
            len(rows),
            int(
                np.searchsorted(
                    times, anchor_us + round(min(duration, BACKEND_MAX_DURATION_SECONDS) * 1e6)
                )
            )
            + 1,
        )
        rows, times = rows[left:right], times[left:right]
        if np.any(np.diff(times) > MAX_GAP_US):
            raise ValueError("nuPlan source gap exceeds 0.15 seconds")
        if any(row["token"] is None for row in rows):
            raise ValueError("Missing ego pose for lidar source frame")
        offsets = [int(row["timestamp"]) - int(row["pc_us"]) for row in rows]
        if max(map(abs, offsets)) > MAX_POSE_ASSOCIATION_SKEW_US:
            raise ValueError("nuPlan pose/lidar association exceeds 50ms alignment tolerance")
        at = [i for i, row in enumerate(rows) if row["pc_token"] == raw_token(token)]
        if len(at) != 1 or at[0] == len(rows) - 1:
            raise ValueError("Anchor requires a unique frame and future support")
        tokens = [row["pc_token"] for row in rows]
        placeholders = ",".join("?" for _ in tokens)
        objects = {item: [] for item in tokens}
        for box in conn.execute(
            f"""SELECT b.*,c.name AS category_name FROM lidar_box b
            LEFT JOIN track t ON b.track_token=t.token LEFT JOIN category c ON t.category_token=c.token
            WHERE b.lidar_pc_token IN ({placeholders}) ORDER BY b.lidar_pc_token,b.track_token""",
            tokens,
        ):
            if box["category_name"] not in CATEGORIES or not box["track_token"]:
                raise ValueError("Unknown object category or missing stable track identity")
            objects[box["lidar_pc_token"]].append(dict(box))
        lights = {item: [] for item in tokens}
        for light in conn.execute(
            f"SELECT * FROM traffic_light_status WHERE lidar_pc_token IN ({placeholders})", tokens
        ):
            status = str(light["status"]).lower()
            lights[light["lidar_pc_token"]].append(
                {
                    "lane_connector_id": str(light["lane_connector_id"]),
                    "status": status if status in {"red", "yellow", "green"} else "unknown",
                    "source_status": status,
                }
            )
        source_scene = conn.execute(
            "SELECT * FROM scene WHERE token=?", (anchor["scene_token"],)
        ).fetchone()
        if source_scene is None:
            raise ValueError("Missing source scene route metadata")
        goal = conn.execute(
            "SELECT * FROM ego_pose WHERE token=?", (source_scene["goal_ego_pose_token"],)
        ).fetchone()
        goal_pose = (
            None
            if goal is None
            else _finite(
                [
                    goal["x"],
                    goal["y"],
                    yaw_from_quaternion([goal[k] for k in ("qw", "qx", "qy", "qz")]),
                ],
                "mission goal",
            ).tolist()
        )
        tags = [
            r[0]
            for r in conn.execute(
                "SELECT type FROM scenario_tag WHERE lidar_pc_token=? ORDER BY type",
                (raw_token(token),),
            )
        ]
        frames, records = [], []
        for row in rows:
            xyz = _finite([row[k] for k in ("x", "y", "z")], "ego pose")
            quat = _finite([row[k] for k in ("qw", "qx", "qy", "qz")], "ego quaternion")
            yaw = yaw_from_quaternion(quat)
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            boxes, names, ids, velocities, object_records = [], [], [], [], []
            for box in objects[row["pc_token"]]:
                pos = _finite([box[k] for k in ("x", "y", "z", "yaw")], "object pose")
                dims = _finite([box[k] for k in ("length", "width", "height")], "object dimensions")
                if (dims <= 0).any():
                    raise ValueError("Non-positive nuPlan object dimensions")
                xy = rot.T @ (pos[:2] - xyz[:2])
                identity = box["track_token"].hex()
                vel = np.asarray([box[k] for k in ("vx", "vy", "vz")], dtype=float)
                boxes.append([*xy, 0.0, *dims, pos[3] - yaw])
                velocities.append([*(rot.T @ vel[:2]), vel[2]])
                names.append(box["category_name"])
                ids.append(identity)
                object_records.append(
                    {
                        "track_token": identity,
                        "category": names[-1],
                        "pose_global": pos.tolist(),
                        "dimensions_lwh": dims.tolist(),
                        "velocity_global": vel.tolist() if np.isfinite(vel).all() else None,
                    }
                )
            stamp = int(row["pc_us"])
            record = {
                "token": row["pc_token"].hex(),
                "timestamp_us": stamp,
                "ego_pose_timestamp_us": int(row["timestamp"]),
                "ego_rear_axle_global": xyz.tolist(),
                "ego_quaternion_wxyz": quat.tolist(),
                "ego_velocity_db": _finite(
                    [row[k] for k in ("vx", "vy", "vz")], "ego velocity"
                ).tolist(),
                "ego_acceleration_db": _finite(
                    [row[k] for k in ("acceleration_x", "acceleration_y", "acceleration_z")],
                    "ego acceleration",
                ).tolist(),
                "objects": object_records,
                "signals": lights[row["pc_token"]],
            }
            records.append(record)
            frames.append(
                {
                    "timestamp": stamp,
                    "token": record["token"],
                    "log_name": header["source_log"],
                    "log_token": header["log_token"],
                    "map_location": header["map_name"],
                    "ego2global_translation": [float(xyz[0]), float(xyz[1]), 0.0],
                    "ego2global_rotation": [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    "anns": {
                        "gt_boxes": np.asarray(boxes).reshape(-1, 7),
                        "gt_names": names,
                        "track_tokens": ids,
                        "gt_velocity_3d": np.asarray(velocities).reshape(-1, 3),
                    },
                }
            )
        validate_log(frames)  # Include history, not only the compiled future.
        # Raw DB labels are never projected to a false red/not-red interpretation.
        result = {
            **header,
            "db_path": str(path),
            "anchor_token": token.lower(),
            "anchor_timestamp_us": anchor_us,
            "route_roadblock_ids": str(source_scene["roadblock_ids"] or "").split(),
            "mission_goal_global": goal_pose,
            "scenario_types": tags,
            "history_seconds_requested": history_seconds,
            "timestamp_alignment": {
                "basis": "lidar_associated_frame; ego_pose_has_separate_measurement_time",
                "exact_sensor_synchrony": False,
                "min_ego_pose_minus_lidar_us": min(offsets),
                "max_ego_pose_minus_lidar_us": max(offsets),
                "max_abs_tolerance_us": MAX_POSE_ASSOCIATION_SKEW_US,
            },
            "history": records[: at[0] + 1],
            "future_source": records[at[0] + 1 :],
            "frames": frames[at[0] :],
        }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source database changed while reading")
    result["source_stat"] = {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    return result


def compile_window(
    window, *, duration, split="unassigned", duration_mode="fit_source", max_actors=32
):
    scene = compile_frames(
        window["frames"],
        start=0,
        duration=duration,
        split=split,
        duration_mode=duration_mode,
        max_actors=max_actors,
        population_mode="logged",
        allow_stationary_ego=True,
    )
    # Avoid initializing closed-loop dynamics from finite differences of future poses.
    history = window["history"]
    if len(history) < 2:
        raise ValueError("Compilation needs at least two measured history frames for causal motion")
    previous, current = history[-2:]
    dt = (current["timestamp_us"] - previous["timestamp_us"]) / 1e6
    if not 0 < dt <= 0.15:
        raise ValueError("Causal initial motion lacks continuous history")

    from driversi.data.participants.nuplan_motion import ego_center_motion

    _, _, center_velocity, _ = ego_center_motion(current, previous)
    scene.velocities[0, scene.actor_ids.index("ego"), :2] = center_velocity
    old_objects = {obj["track_token"]: obj for obj in previous["objects"]}
    for obj in current["objects"]:
        i = scene.actor_ids.index(obj["track_token"])
        velocity = obj["velocity_global"]
        if velocity is None:
            old = old_objects.get(obj["track_token"])
            if old is None:
                raise ValueError("New object lacks observed velocity and causal history")
            velocity = (np.asarray(obj["pose_global"][:2]) - old["pose_global"][:2]) / dt
        scene.velocities[0, i, :2] = velocity[:2]
    scene.velocities[0, :, 2] = 0.0
    scene.metadata["velocity_model"] = (
        "causal_initial; position_finite_difference_source_replay_suffix"
    )
    end = scene.metadata["horizon_plan"]["source_end_frame"]
    source = {key: value for key, value in window.items() if key not in {"frames", "future_source"}}
    future = window["future_source"][:end]
    source.update(
        schema_version=2,
        initial_velocity_source="ego_reported_local_rear_axle_plus_causal_yaw_rate_center_shift; objects_logged_global_or_backward_history",
        reader="readonly_sqlite_v1",
        future_source_tokens=[v["token"] for v in future],
        future_source_timestamps_us=[v["timestamp_us"] for v in future],
        signals=[
            {"timestamp_us": v["timestamp_us"], "states": v["signals"]}
            for v in [window["history"][-1], *future]
        ],
        coordinates="projected XY map axes; DB ego is rear axle, object poses are centers",
        height="physical scene planar; original elevations retained in structured source history",
        rules="source roadblocks and lights retained, not an executable routing/right-of-way policy",
        history_scope="measured <= anchor only; future source records are supervision, never policy observations",
    )
    # Retain actual DB motion evidence separately from position-difference replay
    # velocities. Static boxes may jitter even when every reported velocity is zero.
    # These are privileged source-run summaries, not online policy observations.
    evidence = {}
    active, ended = set(), set()
    for record in [current, *future]:
        objects = {obj["track_token"]: obj for obj in record["objects"]}
        ended.update(active - objects.keys())
        active = set(objects)
        for actor, obj in objects.items():
            if actor in ended or actor not in scene.actor_ids:
                continue
            position = np.asarray(obj["pose_global"][:2])
            item = evidence.setdefault(
                actor,
                {
                    "samples": 0,
                    "finite_velocity_samples": 0,
                    "max_reported_speed_mps": 0.0,
                    "max_position_excursion_m": 0.0,
                    "reported_travel_m": 0.0,
                    "position_path_length_m": 0.0,
                    "first_position": position.tolist(),
                    "first_timestamp_us": record["timestamp_us"],
                },
            )
            velocity = obj["velocity_global"]
            reported_speed = (
                float(np.linalg.norm(velocity[:2]))
                if velocity is not None and np.isfinite(velocity).all()
                else None
            )
            if "last_position" in item:
                dt = (record["timestamp_us"] - item["last_timestamp_us"]) * 1e-6
                item["position_path_length_m"] += float(
                    np.linalg.norm(position - item["last_position"])
                )
                if reported_speed is not None and item["last_speed"] is not None:
                    item["reported_travel_m"] += 0.5 * (reported_speed + item["last_speed"]) * dt
            item["last_position"] = position.tolist()
            item["last_speed"] = reported_speed
            item["samples"] += 1
            item["last_timestamp_us"] = record["timestamp_us"]
            item["max_position_excursion_m"] = max(
                item["max_position_excursion_m"],
                float(np.linalg.norm(position - item["first_position"])),
            )
            velocity = obj["velocity_global"]
            if velocity is not None and np.isfinite(velocity).all():
                item["finite_velocity_samples"] += 1
                item["max_reported_speed_mps"] = max(
                    item["max_reported_speed_mps"], float(np.linalg.norm(velocity[:2]))
                )
    for item in evidence.values():
        item.pop("first_position")
        item.pop("last_position")
        item.pop("last_speed")
    scene.metadata["source_motion_evidence"] = {
        "version": 1,
        "scope": "first_contiguous_run_privileged_navigation",
        "actors": evidence,
    }
    scene.metadata.update(
        source_dataset="nuplan_db",
        source_version="v1.1",
        source_log_token=window["log_token"],
        nuplan_source=source,
        trainable=False,
        training_admission="not_established_by_data_compilation",
    )
    scene.metadata["limitations"].extend(
        ["nuplan_SE2_planar_height", "nuplan_signal_execution_not_implemented"]
    )
    scene.validate()
    return scene
