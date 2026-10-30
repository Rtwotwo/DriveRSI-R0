"""Export two inspection modalities; never modifies a scene or runs a policy."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.data.scenes.context import read_map_context
from driversi.data.sources.navsim import load_log, transform_box, yaw_from_quaternion
from driversi.rendering.bev.replay import _load_run, _scene_key
from driversi.rendering.whitebox.schema import validate_pair

KINDS = {1: "vehicle", 2: "pedestrian", 3: "bicycle", 10: "other"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def object_row(actor, kind, pose, size, control):
    return dict(
        actor_id=str(actor),
        kind=str(kind),
        pose_xyz_yaw=np.asarray(pose, float).tolist(),
        dimensions_lwh=np.asarray(size, float).tolist(),
        control_source=control,
    )


def source_frames(scene, *, trusted=False):
    meta = scene.metadata
    origin = np.asarray(meta["origin_xyz"], float)
    start_us = meta["source_start_timestamp_us"]
    duration = float(scene.times[-1])
    measured = {}
    if "nuplan_source" in meta:
        from driversi.data.sources.nuplan_db import read_window

        src = meta["nuplan_source"]
        record = read_window(
            src["db_path"], token=src["anchor_token"], duration=duration, history_seconds=0
        )
        raw = record["frames"]
        measured = {x["timestamp_us"]: x for x in [record["history"][-1], *record["future_source"]]}
        family, path = "nuplan", src["db_path"]
    elif "navsim_source" in meta:
        path = meta["navsim_source"]["log_path"]
        raw = load_log(Path(path), trusted=trusted)[meta["source_start_frame"] :]
        family = "navsim"
    else:
        raise ValueError("Original NAVSIM or nuPlan source required")
    if not raw or raw[0]["timestamp"] != start_us or raw[0]["token"] != meta["scenario_id"]:
        raise ValueError("Source anchor does not match the scene")
    if not np.allclose(raw[0]["ego2global_translation"], origin, rtol=0, atol=1e-6):
        raise ValueError("Source coordinate origin does not match scene")
    result = []
    for frame in raw:
        time = (frame["timestamp"] - start_us) / 1e6
        if time > duration:
            break
        if frame["log_name"] != meta["source_log"] or frame["map_location"] != meta["map_name"]:
            raise ValueError("Source log/map changed within window")
        xyz = np.asarray(frame["ego2global_translation"], float)
        yaw = yaw_from_quaternion(frame["ego2global_rotation"])
        ego = np.r_[xyz - origin, yaw]
        ego[:2] += 1.461 * np.array([np.cos(yaw), np.sin(yaw)])
        objects = [
            object_row(
                "ego", "ego", ego, scene.dimensions[scene.actor_ids.index("ego")], "source_log"
            )
        ]
        ann = frame["anns"]
        for token, kind, box in zip(
            ann["track_tokens"], ann["gt_names"], ann["gt_boxes"], strict=True
        ):
            pose = transform_box(np.asarray(box), xyz, yaw)
            pose[:3] -= origin
            objects.append(object_row(token, kind, pose, box[3:6], "source_log"))
        signals = dict(
            available="traffic_lights" in frame,
            records=frame.get("traffic_lights", []),
            semantics="connector_red_boolean",
        )
        # Keep measured DB elevation and separate lidar/ego measurement timestamps.
        evidence = measured.get(frame["timestamp"])
        if evidence:
            objects[0]["pose_xyz_yaw"][2] = evidence["ego_rear_axle_global"][2] - origin[2]
            by_id = {x["track_token"]: x for x in evidence["objects"]}
            for obj in objects[1:]:
                obj["pose_xyz_yaw"][2] = by_id[obj["actor_id"]]["pose_global"][2] - origin[2]
            signals = dict(
                available=True, records=evidence["signals"], semantics="raw_db_connector_status"
            )
        result.append(
            dict(
                time_seconds=time,
                timestamp_us=int(frame["timestamp"]),
                token=frame["token"],
                ego_pose_timestamp_us=evidence["ego_pose_timestamp_us"] if evidence else None,
                objects=objects,
                signals=signals,
                events={},
            )
        )
    if len(result) < 2:
        raise ValueError("Two actual source frames required")
    return result, dict(
        family=family,
        path=str(path),
        anchor_timestamp_us=start_us,
        pose_scope="measured annotations; ego rear axle shifted 1.461m to body center",
    )


def interaction_frames(scene, run):
    result = []
    for tick, (states, present) in enumerate(zip(run["states"], run["presence"], strict=True)):
        commands = run["frames"][tick].get("control_sources", {})
        life = run["lifecycle"][tick] if run["lifecycle"] else {}
        background = {x["actor"]: x for x in life.get("background_decisions", [])}
        continuation = life.get("continuity", {})
        statuses = continuation.get("status", [])
        lifecycle = {
            key: life.get(key, []) for key in ("entered", "exited", "contacts", "source_supported")
        }
        lifecycle["available"] = bool(life)
        if not life:
            lifecycle["contacts"] = None
        lifecycle["status_by_actor"] = dict(zip(scene.actor_ids, statuses)) if statuses else {}
        lifecycle["events"] = continuation.get("events", [])
        objects = []
        for i, actor in enumerate(scene.actor_ids):
            if present[i]:
                control = commands.get(actor)
                if control is None and actor in background:
                    diagnostic = background[actor]
                    control = dict(
                        controller=diagnostic.get("controller"),
                        control_mode=diagnostic.get("control_mode"),
                        phase="recorded_background_diagnostic",
                        learning_slot=False,
                    )
                if control is None:
                    control = "recorded_world_without_control_diagnostic"
                objects.append(
                    object_row(
                        actor,
                        "ego" if actor == "ego" else KINDS[int(scene.kinds[i])],
                        states[i][:4],
                        scene.dimensions[i],
                        control,
                    )
                )
        time = float(scene.times[tick])
        result.append(
            dict(
                time_seconds=time,
                timestamp_us=int(scene.metadata["source_start_timestamp_us"] + round(time * 1e6)),
                tick=tick,
                objects=objects,
                events=run["frames"][tick].get("arrivals", {}),
                lifecycle=lifecycle,
                signals=dict(
                    available=False, records=[], semantics="not_asserted_by_execution_record"
                ),
            )
        )
    return result


def sequence(scene, role, frames, provenance):
    catalog = {
        obj["actor_id"]: dict(kind=obj["kind"], dimensions_lwh=obj["dimensions_lwh"])
        for frame in frames
        for obj in frame["objects"]
    }
    if role == "executed_branch":
        catalog = {
            actor: dict(
                kind="ego" if actor == "ego" else KINDS[int(scene.kinds[i])],
                dimensions_lwh=scene.dimensions[i].tolist(),
            )
            for i, actor in enumerate(scene.actor_ids)
        }
    return dict(
        schema_version=1,
        kind="whitebox_structured_sequence",
        role=role,
        actors=catalog,
        source_log=scene.metadata["source_log"],
        source_split=scene.metadata["split"],
        scenario_id=scene.metadata["scenario_id"],
        origin_xyz=scene.metadata["origin_xyz"],
        coordinate_frame="local_metric_map_axes",
        pose_reference="vehicle_or_object_center",
        frames=frames,
        provenance=provenance,
        trainable=False,
        max_sample_age_seconds=(0.75 if provenance.get("family") == "navsim" else 0.15)
        if role == "source_log"
        else 1.5 * float(scene.times[1] - scene.times[0]),
        data_use="inspection_only_not_training_admission",
        temporal_policy="previous_recorded_sample_no_interpolation",
        rendering_geometry="planar_ground_boxes_bottom_z0; stored measured elevations remain unchanged",
    )


def export(scene_path, rollout_path, maps_root, output, *, trusted=False):
    scene_path, rollout_path, output = Path(scene_path), Path(rollout_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    scene = Scene.load(scene_path)
    run = _load_run(scene, rollout_path, _scene_key(scene))
    raw, provenance = source_frames(scene, trusted=trusted)
    physical = interaction_frames(scene, run)
    xy = np.asarray(
        [obj["pose_xyz_yaw"][:2] for frame in [*raw, *physical] for obj in frame["objects"]]
    )
    bounds = np.r_[xy.min(0) - 400, xy.max(0) + 400]
    features, layers, map_source = read_map_context(
        scene.metadata["map_name"],
        Path(maps_root),
        scene.metadata["origin_xyz"],
        bounds,
        expected_version=scene.metadata.get("map_version"),
    )
    source = sequence(scene, "source_log", raw, provenance)
    branch = sequence(
        scene,
        "executed_branch",
        physical,
        dict(
            scene=str(scene_path.resolve()),
            rollout=str(rollout_path.resolve()),
            anchor_timestamp_us=scene.metadata["source_start_timestamp_us"],
            branch_id=sha256(rollout_path),
            scene_sha256=sha256(scene_path),
            controller=run["metadata"].get("controller_configuration", {}),
            scene_key=run["metadata"]["scene_key"],
            validation="existing_strict_engineering_replay_loader",
        ),
    )
    validate_pair(source, branch)
    manifest = dict(
        schema_version=1,
        kind="whitebox_replay",
        source_log=scene.metadata["source_log"],
        source_family=provenance["family"],
        map="map.json",
        streams={
            "source_log": "source-structure.json",
            "executed_branch": "interaction-structure.json",
        },
        timeline_end_seconds=max(raw[-1]["time_seconds"], physical[-1]["time_seconds"]),
        counts=dict(
            source_frames=len(raw),
            interaction_frames=len(physical),
            map_features=len(features),
            source_unique_actors=len({o["actor_id"] for f in raw for o in f["objects"]}),
            physical_unique_actors=len({o["actor_id"] for f in physical for o in f["objects"]}),
            physical_actor_slots=len(scene.actor_ids),
        ),
        visual_style="semantic_whitebox",
        trainable=False,
        export_runs_simulation=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".whitebox-", dir=output.parent) as temporary:
        folder = Path(temporary) / "bundle"
        folder.mkdir()
        dump(folder / "source-structure.json", source)
        dump(folder / "interaction-structure.json", branch)
        dump(
            folder / "map.json",
            dict(
                schema_version=1,
                origin_xyz=scene.metadata["origin_xyz"],
                features=features,
                layers=layers,
                source=map_source,
                bounds_local=bounds.tolist(),
                surface_model="official_polygons_flattened_z0",
                baseline_semantics="navigation_centerlines_not_lane_marking_truth",
            ),
        )
        manifest["sha256"] = {
            name: sha256(folder / name)
            for name in ["source-structure.json", "interaction-structure.json", "map.json"]
        }
        dump(folder / "manifest.json", manifest)
        for name in ("viewer.html", "geometry.js", "viewer.js"):
            template = Path(__file__).with_name(name)
            if not template.exists():
                raise FileNotFoundError(f"Whitebox renderer resource missing: {template}")
            shutil.copyfile(template, folder / name)
        folder.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("scene", "rollout", "maps-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--trusted-source", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.scene, args.rollout, args.maps_root, args.output, trusted=args.trusted_source
            )
        )
    )


if __name__ == "__main__":
    main()
