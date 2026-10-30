"""Optional nuPlan scenario/DB input. No teacher model or sensor stack is required.

nuPlan SDK is imported only by load_db_scenario. The shared compiler remains
responsible for time support, population and10Hz interpolation. Source z is not
available through this SE2 interface: all actors use a declared planar height.
"""

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from driversi.data.sources.navsim import BACKEND_MAX_DURATION_SECONDS, compile_frames, validate_horizon_request


def scenario_frames(scenario, *, max_source_frames=20000):
    count = scenario.get_number_of_iterations()
    if type(count) is not int or not 2 <= count <= max_source_frames:
        raise ValueError("nuPlan scenario source-frame count outside bounded adapter capacity")
    frames, signals, dimensions = [], [], []
    for i in range(count):
        ego = scenario.get_ego_state_at_iteration(i)
        centre, box = ego.center, ego.car_footprint.oriented_box
        yaw = float(centre.heading)
        # Shared compiler expects Pacifica rear-axle coordinates. Normalize using
        # the actual SDK centre, then restore actual ego dimensions below.
        translation = np.array(
            [centre.x - 1.461 * np.cos(yaw), centre.y - 1.461 * np.sin(yaw), 0.0]
        )
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        boxes, names, tokens = [], [], []
        for obj in scenario.get_tracked_objects_at_iteration(i).tracked_objects:
            token = obj.track_token
            if not isinstance(token, str) or not token or token == "ego":
                raise ValueError(
                    "nuPlan object needs a stable non-ego track token; no silent omission"
                )
            centre_obj, obox = obj.center, obj.box
            xy = rotation.T @ (np.array([centre_obj.x, centre_obj.y]) - translation[:2])
            boxes.append([*xy, 0.0, obox.length, obox.width, obox.height, centre_obj.heading - yaw])
            names.append(str(obj.tracked_object_type.name).lower())
            tokens.append(token)
        timestamp = int(scenario.get_time_point(i).time_us)
        lights = [
            {
                "lane_connector_id": str(light.lane_connector_id),
                "status": str(light.status.name).lower(),
                "timestamp_us": int(light.timestamp),
            }
            for light in scenario.get_traffic_light_status_at_iteration(i)
        ]
        if any(item["status"] not in {"red", "yellow", "green", "unknown"} for item in lights):
            raise ValueError("Unsupported nuPlan signal state")
        # Deliberately omit NAVSIM's red/not-red projection; yellow/unknown must
        # never become green. Full states are retained in source metadata.
        frames.append(
            {
                "timestamp": timestamp,
                "token": hashlib.sha256(f"{scenario.token}:{timestamp}".encode()).hexdigest()[:16],
                "log_name": str(scenario.log_name),
                "map_location": str(scenario.map_api.map_name),
                "ego2global_translation": translation,
                "ego2global_rotation": [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                "anns": {
                    "gt_boxes": np.asarray(boxes).reshape(-1, 7),
                    "gt_names": names,
                    "track_tokens": tokens,
                },
            }
        )
        signals.append({"timestamp_us": timestamp, "states": lights})
        dimensions.append([box.length, box.width, box.height])
    if not np.isfinite(dimensions).all() or not np.allclose(
        dimensions, dimensions[0], rtol=0, atol=1e-6
    ):
        raise ValueError("Ego dimensions must remain fixed and finite")
    return frames, signals, dimensions[0]


def compile_scenario(
    scenario, *, duration, split, start=0, duration_mode="fit_source", max_actors=32
):
    frames, signals, dimensions = scenario_frames(scenario)
    scene = compile_frames(
        frames,
        start=start,
        duration=duration,
        split=split,
        duration_mode=duration_mode,
        max_actors=max_actors,
        population_mode="logged",
    )
    scene.dimensions[scene.actor_ids.index("ego")] = dimensions
    goal = scenario.get_mission_goal()
    scene.metadata["nuplan_source"] = {
        "schema_version": 1,
        "scenario_token": str(scenario.token),
        "source_timestamp_us": frames[start]["timestamp"],
        "source_log": str(scenario.log_name),
        "route_roadblock_ids": list(map(str, scenario.get_route_roadblock_ids())),
        "mission_goal_global": None
        if goal is None
        else [float(goal.x), float(goal.y), float(goal.heading)],
        "signals": signals[start : scene.metadata["horizon_plan"]["source_end_frame"] + 1]
        if "horizon_plan" in scene.metadata
        else signals,
        "coordinates": "nuPlan projected XY; synthetic rear-axle normalization preserves SDK centre",
        "height": "planar_zero_not_sensor_elevation",
        "rules": "signals preserved as source records; not executed or inferred",
        "source_scenario_frames": len(frames),
        "real_db_validation": "not_established_by_adapter",
    }
    scene.metadata["limitations"].extend(
        ["nuplan_SE2_planar_height", "nuplan_signal_execution_not_implemented"]
    )
    scene.validate()
    return scene


def db_anchor(db_path, token):
    """Read-only local DB lookup; never creates a missing database or downloads."""
    db_path = Path(db_path).resolve(strict=True)
    if not db_path.is_file() or db_path.suffix != ".db":
        raise ValueError("Expected an existing local nuPlan.db")
    try:
        raw = bytes.fromhex(token)
    except (ValueError, TypeError):
        raise ValueError("Lidar token must be hexadecimal") from None
    if not raw:
        raise ValueError("Empty lidar token")
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute(
            """SELECT p.timestamp, l.map_version FROM lidar_pc p
             JOIN lidar s ON p.lidar_token=s.token JOIN log l ON s.log_token=l.token
             WHERE p.token=?""",
            (raw,),
        ).fetchone()
    if row is None:
        raise ValueError("Lidar token not found in this DB")
    return int(row[0]), str(row[1])


def load_db_scenario(db_path, *, token, maps_root, map_version="nuplan-maps-v1.0", duration=9.0):
    validate_horizon_request(duration, "fit_source", 0.1)
    timestamp, map_name = db_anchor(db_path, token)
    try:
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
            ScenarioExtractionInfo,
        )
    except ImportError as error:
        raise RuntimeError(
            "nuPlan SDK must be installed in the explicitly selected interpreter"
        ) from error
    path = Path(db_path).resolve()
    return NuPlanScenario(
        data_root=str(path.parent),
        log_file_load_path=str(path),
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type="driversi_input",
        map_root=str(maps_root),
        map_version=map_version,
        map_name=map_name,
        ego_vehicle_parameters=get_pacifica_parameters(),
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_duration=min(duration, BACKEND_MAX_DURATION_SECONDS) + 0.1, extraction_offset=0.0, subsample_ratio=0.5
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--lidar-token", required=True)
    parser.add_argument("--maps-root", required=True, type=Path)
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--reader", choices=["sqlite", "sdk"], default="sqlite")
    parser.add_argument("--history-seconds", type=float, help="SQLite only; default 1.5 seconds")
    parser.add_argument(
        "--split", required=True, choices=["mini", "train", "development", "heldout", "unassigned"]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.split in {"train", "development"}:
        parser.error("Use index-nuplan --split-policy then compile-catalog for provenance-checked training lists")
    if args.db.parent.name in {"val", "test"} and args.split != "heldout":
        parser.error("nuPlan val/test databases must remain heldout")
    if args.reader == "sqlite":
        from driversi.data.sources.nuplan_db import compile_window, read_window
        window = read_window(args.db, token=args.lidar_token, duration=args.duration,
                             history_seconds=1.5 if args.history_seconds is None else args.history_seconds)
        scene = compile_window(window, duration=args.duration, split=args.split)
    else:
        if args.history_seconds is not None:
            parser.error("--history-seconds is supported only by the SQLite reader")
        scenario = load_db_scenario(args.db, token=args.lidar_token, maps_root=args.maps_root,
                                    map_version=args.map_version, duration=args.duration)
        scene = compile_scenario(scenario, duration=args.duration, split=args.split)
    from driversi.data.road_network.maps import attach_nuplan_map

    attach_nuplan_map(scene, args.maps_root)
    scene.metadata["nuplan_source"]["db_path"] = str(args.db.resolve())
    scene.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "actors": len(scene.actor_ids),
                "seconds": float(scene.times[-1]),
                "validation": "compiled_only_not_interaction_or_traffic_qualification",
            }
        )
    )


if __name__ == "__main__":
    main()
