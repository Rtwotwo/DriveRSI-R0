"""Source-supported kinematic participant slots, separate from policy ownership."""

import copy

import numpy as np

from driversi.contracts.scene import MAX_WORLD_ACTORS, OTHER_OBJECT
from driversi.data.participants.motion import source_vehicle_headings


def attach_logged_population(scene, frames, *, max_world_actors=MAX_WORLD_ACTORS):
    from driversi.data.sources.navsim import KINDS, transform_box, validate_log, yaw_from_quaternion

    scene.validate()
    if type(max_world_actors) is not int or not 1 <= max_world_actors <= MAX_WORLD_ACTORS:
        raise ValueError(f"World capacity must be 1..{MAX_WORLD_ACTORS}")
    scene = copy.deepcopy(scene)
    start = scene.metadata["source_start_frame"]
    window = frames[start : start + scene.metadata["source_frames"]]
    if (
        not window
        or window[0]["token"] != scene.metadata["scenario_id"]
        or window[0]["timestamp"] != scene.metadata["source_start_timestamp_us"]
        or any(f["log_name"] != scene.metadata["source_log"] for f in window)
    ):
        raise ValueError("Lifecycle source does not match scene")
    validate_log(window)
    ts = (np.array([f["timestamp"] for f in window], dtype=np.int64) - window[0]["timestamp"]) / 1e6
    if (
        len(ts) < 2
        or np.any(np.diff(ts) <= 0)
        or np.any(np.diff(ts) > 0.75)
        or ts[-1] < scene.times[-1]
    ):
        raise ValueError("Lifecycle source lacks continuous horizon support")
    records = {}
    first_observations = {}
    for j, frame in enumerate(window):
        yaw = yaw_from_quaternion(frame["ego2global_rotation"])
        for row_index, (token, kind, box) in enumerate(zip(
            frame["anns"]["track_tokens"],
            frame["anns"]["gt_names"],
            frame["anns"]["gt_boxes"],
            strict=True,
        )):
            if token == "ego":
                raise ValueError("Source track ID collides with ego identity")
            pose = transform_box(box, frame["ego2global_translation"], yaw)
            pose[:3] -= scene.metadata["origin_xyz"]
            records.setdefault(str(token), []).append((j, str(kind), np.asarray(box[3:6]), pose))
            if str(token) not in first_observations:
                raw = frame["anns"].get("gt_velocity_3d")
                velocity = None
                if raw is not None:
                    value = np.asarray(raw[row_index], dtype=float)
                    if value.shape == (3,) and np.isfinite(value).all():
                        velocity = [float(np.cos(yaw)*value[0] - np.sin(yaw)*value[1]),
                                    float(np.sin(yaw)*value[0] + np.cos(yaw)*value[1]), float(value[2])]
                first_observations[str(token)] = {
                    "source_timestamp_us": int(frame["timestamp"]),
                    "source_index": j, "velocity_world": velocity,
                    "velocity_source": "gt_velocity_3d_rotated_by_ego_yaw" if velocity is not None else "unavailable",
                }
    ids = (*scene.actor_ids, *sorted(set(records) - set(scene.actor_ids)))
    if len(ids) > max_world_actors:
        raise ValueError(
            f"Logged population exceeds capacity: {len(ids)}>{max_world_actors}; no actors discarded"
        )
    t, n = len(scene.times), len(ids)
    old = len(scene.actor_ids)
    poses = np.zeros((t, n, 4))
    vel = np.zeros((t, n, 3))
    dims = np.ones((n, 3))
    kinds = np.ones(n, np.int32)
    controlled = np.zeros(n, bool)
    present = np.zeros((t, n), bool)
    poses[:, :old] = scene.poses
    vel[:, :old] = scene.velocities
    dims[:old] = scene.dimensions
    kinds[:old] = scene.kinds
    controlled[:old] = scene.controlled
    present[:, :old] = True
    source_kinds = {"ego": "vehicle"}
    audit = copy.deepcopy(scene.metadata.get("heading_audit", {}))
    owned = {actor for actor, flag in zip(ids, controlled, strict=True) if flag}
    corrections = [item for item in audit.get("corrections", []) if item["actor_id"] in owned]
    previous = np.searchsorted(ts, scene.times, side="right") - 1
    for a, token in enumerate(ids):
        if token == "ego":
            continue
        rows = records[token]
        names = {row[1] for row in rows}
        if len(names) != 1:
            raise ValueError(f"Actor kind changes: {token}")
        kind = rows[0][1]
        source_kinds[token] = kind
        kinds[a] = KINDS.get(kind, OTHER_OBJECT)
        dims[a] = rows[0][2]
        if not np.isfinite(dims[a]).all() or (dims[a] <= 0).any():
            raise ValueError(f"Invalid collision box for {token}")
        if controlled[a]:
            continue
        samples = np.zeros((len(ts), 4))
        observed = np.zeros(len(ts), bool)
        for j, _, _, pose in rows:
            samples[j] = pose
            observed[j] = True
        present[:, a] = observed[previous]
        poses[:, a] = 0
        vel[:, a] = 0
        # Continuous runs preserve heading correction but never bridge a missing frame.
        positions = np.flatnonzero(observed)
        runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        for run in runs:
            if len(run) > 1:
                corrected, run_audit = source_vehicle_headings(
                    samples[run, None, :], ts[run], (token,), np.array([kinds[a]])
                )
                samples[run, 3] = corrected[:, 0]
                for item in run_audit["corrections"]:
                    item["source_start_index"] = int(run[item["source_start_index"]])
                    item["source_end_index"] = int(run[item["source_end_index"]])
                    corrections.append(item)
                if not audit:
                    audit = {key: value for key, value in run_audit.items() if key != "corrections"}
            for j in run:
                mask = (previous == j) & present[:, a]
                if j + 1 < len(ts) and observed[j + 1]:
                    u = (scene.times[mask] - ts[j]) / (ts[j + 1] - ts[j])
                    poses[mask, a] = samples[j] + u[:, None] * (samples[j + 1] - samples[j])
                    vel[mask, a] = (samples[j + 1, :3] - samples[j, :3]) / (ts[j + 1] - ts[j])
                else:
                    poses[mask, a] = samples[j]
        poses[:, a, 3] = np.arctan2(np.sin(poses[:, a, 3]), np.cos(poses[:, a, 3]))
    audit["corrections"] = corrections
    scene.metadata["heading_audit"] = audit
    scene.actor_ids = ids
    scene.kinds = kinds
    scene.dimensions = dims
    scene.poses = poses
    scene.velocities = vel
    scene.controlled = controlled
    scene.presence = present
    scene.metadata.update(
        population_policy="logged_union_stable_slots_v1",
        source_actor_kinds=source_kinds,
        first_observations=first_observations,
        world_actor_capacity=max_world_actors,
        excluded_incomplete_tracks=0,
        excluded_over_capacity=0,
        excluded_unsupported_initial=0,
        lifecycle={
            "version": 1,
            "source_membership": "previous_annotation_until_next_source_frame",
            "controlled_lifetime": "simulation_owned",
            "new_actor_controller": "kinematic_log_replay",
            "other_object_backend": "noncontrolled_vehicle_box_collision_proxy",
            "presence_counts": present.sum(axis=1).tolist(),
        },
        limitations=list(dict.fromkeys([item for item in scene.metadata.get("limitations", [])
                                       if item != "no_spawn_despawn"] + [
            "log_replay_not_socially_reactive",
            "discrete_10hz_box_contacts",
            "annotation_absence_not_proof_of_physical_exit",
            "no_signal_dynamics",
        ])),
    )
    scene.validate()
    return scene
