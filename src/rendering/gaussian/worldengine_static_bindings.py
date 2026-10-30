"""Observed static background registration; geometry evidence never grants visual admission.

NAVSIM physics intentionally compiles boxes using yaw and planar Z. A source
background OBB must therefore use its original full 3D observation rather than
transporting that planar Z through the current frame's pitch/roll. The fixed
background anchor is usable only while the executed physical pose is unchanged.
"""

import copy

import numpy as np

from driversi.data.participants.traffic_continuity import STATIC_SOURCE_KINDS
from driversi.data.scenes.pufferdrive import native_observed_pose
from driversi.data.sources.navsim import transform_box, yaw_from_quaternion
from driversi.rendering.gaussian.mtgs_bindings import exact_float32_identity, observed_3d_box_corners


def _observations(raw_frames, actor_id, source_log):
    result = []
    for frame in raw_frames:
        annotation = frame.get("anns", {})
        tokens = list(annotation.get("track_tokens", ()))
        if actor_id not in tokens:
            continue
        if tokens.count(actor_id) != 1 or frame.get("log_name") != source_log:
            raise ValueError("Static appearance observation identity is ambiguous")
        index = tokens.index(actor_id)
        box = np.asarray(annotation["gt_boxes"][index], dtype=float)
        if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise ValueError("Finite original source box required")
        result.append((frame, index, box))
    if not result:
        raise ValueError("Static appearance has no observed original annotation")
    times = [int(row[0]["timestamp"]) for row in result]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Static appearance observations must be chronologically unique")
    return result


def _parking_provenance(observations):
    """Reuse the existing fleet's 0.2 m/s, 1.5 m source evidence limits."""
    rows, positions = [], []
    for frame, index, box in observations:
        annotation = frame["anns"]
        values = annotation.get("gt_velocity_3d")
        if values is None:
            values = annotation.get("gt_velocity")
        if values is None or len(values) != len(annotation["track_tokens"]):
            raise ValueError("Complete native vehicle velocity evidence required")
        velocity = np.asarray(values[index], dtype=float)
        if velocity.shape not in {(2,), (3,)} or not np.isfinite(velocity).all():
            raise ValueError("Complete finite native vehicle velocity evidence required")
        pose = transform_box(box, np.asarray(frame["ego2global_translation"]),
                             yaw_from_quaternion(frame["ego2global_rotation"]))
        positions.append(pose[:2])
        rows.append(dict(source_token=frame["token"], timestamp_us=int(frame["timestamp"]),
                         native_velocity=velocity.tolist(), speed_xy_mps=float(np.linalg.norm(velocity[:2])),
                         compiled_global_xy=pose[:2].tolist()))
    points = np.asarray(positions)
    excursion = float(np.linalg.norm(points - points[0], axis=1).max())
    maximum = max(row["speed_xy_mps"] for row in rows)
    if len(rows) < 2 or maximum > 0.2 or excursion > 1.5:
        raise ValueError("Native vehicle observations do not establish parked background support")
    return dict(rule="complete_source_speed_le_0.2_and_excursion_le_1.5_v1",
                scope="offline_source_appearance_audit_not_online_policy_input",
                samples=len(rows), max_reported_speed_mps=maximum,
                max_position_excursion_m=excursion, observations=rows)



def _raw_reference_pose(scene, raw_frames, frame, index, box, actor_id, first_time, surveyed):
    """Recompute the native compiler's first observation/interpolation, not its receipt."""
    from driversi.data.participants.motion import source_vehicle_headings

    origin = np.asarray(scene.metadata["origin_xyz"], dtype=float)
    pose = transform_box(box, np.asarray(frame["ego2global_translation"]),
                         yaw_from_quaternion(frame["ego2global_rotation"]))
    pose[:3] -= origin
    evidence = dict(method="original_raw_planar_box_v1", source_token=frame["token"],
                    source_timestamp_us=int(frame["timestamp"]), interpolation_fraction=0.0)
    delta = int(first_time) - int(frame["timestamp"])
    if not surveyed and delta > 0:
        positions = [i for i, item in enumerate(raw_frames) if item["token"] == frame["token"]]
        if len(positions) != 1:
            raise ValueError("First static source frame is ambiguous")
        n = positions[0] + 1
        if n < len(raw_frames):
            following = raw_frames[n]
            tokens = list(following["anns"]["track_tokens"])
            if actor_id in tokens:
                k = tokens.index(actor_id)
                box2 = np.asarray(following["anns"]["gt_boxes"][k], dtype=float)
                end = int(following["timestamp"])
                if (following["log_name"] != frame["log_name"] or
                        end <= int(first_time) or end - int(frame["timestamp"]) > 750000):
                    raise ValueError("First static compiler interpolation lacks source support")
                next_pose = transform_box(box2, np.asarray(following["ego2global_translation"]),
                                           yaw_from_quaternion(following["ego2global_rotation"]))
                next_pose[:3] -= origin
                poses = np.array([pose, next_pose])
                kind = 1 if frame["anns"]["gt_names"][index] == "vehicle" else 10
                times = np.array([0., (end-int(frame["timestamp"]))/1e6])
                headings, _ = source_vehicle_headings(poses[:,None,:], times, (actor_id,), np.array([kind]))
                poses[:,3] = headings[:,0]
                fraction = delta / (end-int(frame["timestamp"]))
                pose = poses[0] + fraction * (poses[1]-poses[0])
                pose[3] = np.arctan2(np.sin(pose[3]), np.cos(pose[3]))
                evidence.update(method="original_raw_first_contiguous_interpolation_v1",
                                next_source_token=following["token"],
                                next_source_timestamp_us=end, interpolation_fraction=float(fraction))
    evidence["compiled_planar_pose"] = pose.tolist()
    return pose, evidence

def build_static_background_binding(scene, states, presence, raw_frames, resolve_source, actor_id):
    """Build an independently checkable fixed 3D anchor for one unbound actor.

    The source observation is never relabelled as an earlier observation. This
    routine does not modify source annotations, physics, Gaussians, or admission.
    """
    states, presence = np.asarray(states), np.asarray(presence)
    count = len(scene.actor_ids)
    if (actor_id not in scene.actor_ids or actor_id == "ego" or
            states.ndim != 3 or states.shape[:2] != (len(scene.times), count) or states.shape[2] < 4 or
            presence.shape != states.shape[:2] or presence.dtype.kind != "b" or
            not np.isfinite(states[..., :4]).all()):
        raise ValueError("Sealed matching finite physical branch required")
    times = np.asarray(scene.times, dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or times[0] != 0 or not np.allclose(np.diff(times), 0.1, atol=1e-9, rtol=0):
        raise ValueError("Static appearance binding requires the sealed 10 Hz physical timeline")
    index = scene.actor_ids.index(actor_id)
    if scene.controlled[index]:
        raise ValueError("A controlled policy actor cannot be a fixed background appearance")
    hits = np.flatnonzero(presence[:, index])
    if not len(hits) or not presence[hits[0]:, index].all():
        raise ValueError("Static background physical presence must continue after admission")
    first_tick = int(hits[0])
    reference = native_observed_pose(scene, scene.poses[first_tick, index])
    initial = states[first_tick, index, :4]
    identity = exact_float32_identity(reference, initial)
    if not identity["equal"]:
        raise ValueError("Static background initial pose differs from native compiled observation")
    actual = states[hits, index, :4].astype(np.float32)
    if not np.array_equal(actual, np.broadcast_to(reference.astype(np.float32), actual.shape)):
        raise ValueError("Static background physical actor moved within the sealed branch")
    observations = _observations(raw_frames, actor_id, scene.metadata["source_log"])
    kinds = {str(frame["anns"]["gt_names"][j]) for frame, j, box in observations}
    if len(kinds) != 1:
        raise ValueError("Static background source category changes")
    kind = next(iter(kinds))
    if scene.metadata.get("source_actor_kinds", {}).get(actor_id) != kind:
        raise ValueError("Static background compiled source category differs")
    if kind in STATIC_SOURCE_KINDS:
        provenance = dict(rule="native_nuplan_static_object_type_v1", source_category=kind,
                          observations=len(observations))
    elif kind in {"vehicle", "bicycle"}:
        provenance = _parking_provenance(observations)
        if kind == "bicycle":
            provenance["source_category"] = "bicycle"
            provenance["scope"] = "observed_stationary_bicycle_background_candidate_not_static_class_or_visual_admission"
    else:
        raise ValueError("No native static or confirmed parked vehicle appearance evidence")
    start = int(scene.metadata["source_start_timestamp_us"])
    first_time = start + round(float(scene.times[first_tick]) * 1e6)
    prior = None
    if actor_id in scene.metadata.get("static_scene_context", {}).get("actors", {}):
        from driversi.data.scenes.static_context import validate_static_scene_context

        context = validate_static_scene_context(scene, raw_frames)
        prior = context["actors"][actor_id]
        if kind not in STATIC_SOURCE_KINDS or first_tick != 0:
            raise ValueError("Survey prior binding requires an initial native static object")
        eligible = [row for row in observations if row[0]["token"] == prior["source_token"]]
        if len(eligible) != 1:
            raise ValueError("Survey prior raw source identity differs")
        frame, j, box = eligible[0]
        # The source timestamp remains the real later annotation time. The
        # immutable geometry was separately surveyed and reviewed in initial
        # real images, not manufactured as a past dynamic observation.
        if (int(frame["timestamp"]) != prior["source_timestamp_us"] or
                not np.array_equal(box, np.asarray(prior["raw_box"]))):
            raise ValueError("Survey prior source box or timestamp differs")
        temporal_policy = "offline_survey_prior_static_native_v1"
    else:
        eligible = [row for row in observations if start <= int(row[0]["timestamp"]) <= first_time]
        if not eligible:
            raise ValueError("Static background cannot use a future first observation")
        frame, j, box = eligible[0]
        if first_time - int(frame["timestamp"]) > 100000:
            raise ValueError("Static background observation lacks first-admission temporal support")
        temporal_policy = "past_observed_static_background_v1"
    raw_pose, raw_reference = _raw_reference_pose(
        scene, raw_frames, frame, j, box, actor_id, first_time, prior is not None)
    raw_native = native_observed_pose(scene, raw_pose)
    raw_identity = exact_float32_identity(raw_native, reference)
    if not raw_identity["equal"]:
        raise ValueError("Static physical reference differs from original raw source geometry")
    raw_reference["native_identity"] = raw_identity
    canonical = resolve_source(frame["token"])
    if canonical["token"] != frame["token"] or int(canonical["timestamp"]) != int(frame["timestamp"]):
        raise ValueError("Static background reconstruction observation identity differs")
    # Native metadata can intentionally omit static classes. Use exact original
    # raw annotations under the same immutable source pose, never a generated box.
    source = dict(canonical, track_tokens=list(frame["anns"]["track_tokens"]),
                  gt_boxes=np.asarray(frame["anns"]["gt_boxes"]))
    corners, observation = observed_3d_box_corners(source, actor_id, scene.dimensions[index])
    return dict(schema_version=1, policy="immutable_static_background_observation_v1",
                actor_id=actor_id, source_log=scene.metadata["source_log"], source_kind=kind,
                source_token=frame["token"], source_timestamp_us=int(frame["timestamp"]),
                source_box=box.tolist(), source_observation=observation,
                temporal_policy=temporal_policy, survey_prior=copy.deepcopy(prior),
                source_refined_ego2global=np.asarray(canonical["ego2global"]).tolist(),
                scene_start_timestamp_us=start,
                reference_tick=first_tick, reference_timestamp_us=first_time,
                reference_contract="native_centered_scene_observation_v2",
                reference_physical_pose=reference.tolist(), reference_identity=identity,
                raw_reference=raw_reference,
                reconstruction_corners=corners.tolist(),
                physical_dimensions=np.asarray(scene.dimensions[index]).tolist(),
                whole_branch_immobility=dict(method="identical_float32_pose_v1", first_tick=first_tick,
                                            last_tick=int(hits[-1]), present_state_count=len(hits),
                                            physical_state_span=np.ptp(states[hits, index, :4], axis=0).tolist()),
                kind_or_parking_provenance=provenance,
                visual_admitted=False,
                scope="fixed_observed_background_geometry_candidate_not_visual_admission")


def validate_static_background_binding(binding, scene, states, presence, raw_frames, resolve_source):
    """Recompute from sealed inputs; no mutable certificate can substitute inputs."""
    if not isinstance(binding, dict):
        raise ValueError("Static background binding must be a mapping")
    expected = build_static_background_binding(scene, states, presence, raw_frames,
                                                resolve_source, binding.get("actor_id"))
    if binding != expected:
        raise ValueError("Static background binding differs from sealed physical/source evidence")
    return copy.deepcopy(expected)


def static_background_box_at_tick(binding, state, tick, timestamp_us):
    """Return unchanged source OBB only at a chronologically valid identical pose."""
    if (binding.get("policy") != "immutable_static_background_observation_v1" or
            not isinstance(tick, (int, np.integer)) or tick < binding["reference_tick"] or
            tick > binding["whole_branch_immobility"]["last_tick"] or
            timestamp_us < binding["reference_timestamp_us"] or
            timestamp_us != binding["scene_start_timestamp_us"] + int(tick) * 100000):
        raise ValueError("Static background binding is not valid before its actual observation/presence")
    temporal = binding.get("temporal_policy")
    if temporal == "offline_survey_prior_static_native_v1":
        prior = binding.get("survey_prior")
        if (binding.get("source_kind") not in STATIC_SOURCE_KINDS or not isinstance(prior, dict) or
                prior.get("policy") != "survey_prior_static_native_v1" or
                prior.get("source_observations_preserved") is not True or
                prior.get("source_token") != binding["source_token"] or
                prior.get("source_timestamp_us") != binding["source_timestamp_us"]):
            raise ValueError("Explicit validated static survey prior is required")
    elif temporal != "past_observed_static_background_v1" or timestamp_us < binding["source_timestamp_us"]:
        raise ValueError("Static background cannot backdate a source observation")
    identity = exact_float32_identity(binding["reference_physical_pose"], np.asarray(state)[:4])
    if not identity["equal"]:
        raise ValueError("Static background target physical pose changed")
    corners = np.asarray(binding["reconstruction_corners"], dtype=float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("Finite fixed source 3D corners required")
    return corners.copy(), dict(policy=binding["policy"], actor_id=binding["actor_id"],
                                source_token=binding["source_token"], source_timestamp_us=binding["source_timestamp_us"],
                                temporal_policy=temporal,
                                reference_tick=binding["reference_tick"], target_tick=int(tick),
                                target_timestamp_us=int(timestamp_us), physical_identity=identity,
                                visual_admitted=False)
