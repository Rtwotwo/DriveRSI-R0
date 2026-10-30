"""Explicit offline surveyed static geometry; never future dynamic actor backfill."""

import copy
import hashlib
from pathlib import Path

import numpy as np

from driversi.contracts.scene import OTHER_OBJECT
from driversi.data.participants.traffic_continuity import STATIC_SOURCE_KINDS
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap

POLICY = "survey_prior_static_native_v1"


def static_prior_actors(scene):
    """Validate self-contained sealed prior metadata and return its actor IDs."""
    context = scene.metadata.get("static_scene_context")
    if context is None:
        return set()
    if context.get("version") != 1 or context.get("policy") != POLICY:
        raise ValueError("Unsupported static scene prior")
    actors = context.get("actors")
    if not isinstance(actors, dict) or not actors:
        raise ValueError("Static prior requires explicit actors")
    for actor, row in actors.items():
        if actor not in scene.actor_ids:
            raise ValueError("Unknown static prior actor")
        i = scene.actor_ids.index(actor)
        if scene.controlled[i] or scene.kinds[i] != OTHER_OBJECT:
            raise ValueError("Only native static categories may receive surveyed priors")
        kind = scene.metadata.get("source_actor_kinds", {}).get(actor)
        if kind not in STATIC_SOURCE_KINDS or row.get("source_kind") != kind:
            raise ValueError("Static prior category mismatch")
        if row.get("policy") != POLICY or row.get("source_observations_preserved") is not True:
            raise ValueError("Static prior must preserve source observations")
        pose = np.asarray(row.get("initial_physical_pose"), float)
        box = np.asarray(row.get("raw_box"), float)
        if pose.shape != (4,) or box.shape != (7,) or not np.isfinite(pose).all() or not np.isfinite(box).all():
            raise ValueError("Static prior requires finite surveyed pose and raw box")
        if not np.allclose(scene.poses[0, i], pose, atol=1e-6, rtol=0):
            raise ValueError("Static prior initial pose changed")
        if not np.allclose(scene.dimensions[i], box[3:6], atol=1e-6, rtol=0):
            raise ValueError("Static prior collision dimensions changed")
        if not isinstance(row.get("evidence_refs"), list) or not row["evidence_refs"]:
            raise ValueError("Static prior needs evidence references")
        support = np.asarray(scene.metadata["reactive_background"]["source_presence"], bool)
        hits = np.flatnonzero(support[:, i])
        if not len(hits) or int(hits[0]) != row.get("source_first_seen_tick"):
            raise ValueError("Static prior raw first-seen metadata changed")
        if not scene.presence[:, i].all() or np.any(scene.velocities[:, i] != 0):
            raise ValueError("Static prior must be physically present and stationary")
    return set(actors)


def apply_static_scene_context(scene, raw_frames, manifest):
    """Import reviewed immutable geometry into physics without changing raw support.

    The geometry is an explicit offline scene prior, like surveyed map geometry.
    It is not an online observation and cannot be applied to dynamic categories.
    """
    from driversi.data.sources.navsim import transform_box, yaw_from_quaternion

    scene.validate()
    if "static_scene_context" in scene.metadata:
        raise ValueError("Static scene context already configured")
    if manifest.get("version") != 1 or manifest.get("policy") != POLICY:
        raise ValueError("Unsupported static scene context manifest")
    if manifest.get("source_log") != scene.metadata["source_log"]:
        raise ValueError("Static scene prior belongs to another log")
    requests = manifest.get("actors")
    if not isinstance(requests, dict) or not requests:
        raise ValueError("Static context must name reviewed actors")
    result = copy.deepcopy(scene)
    support = np.asarray(result.metadata["reactive_background"]["source_presence"], bool)
    entries = {}
    for actor, request in requests.items():
        if actor not in result.actor_ids:
            raise ValueError("Static prior cannot invent actor identities")
        i = result.actor_ids.index(actor)
        kind = result.metadata.get("source_actor_kinds", {}).get(actor)
        if result.controlled[i] or result.kinds[i] != OTHER_OBJECT or kind not in STATIC_SOURCE_KINDS:
            raise ValueError("Only native static categories may receive surveyed priors")
        hits = np.flatnonzero(support[:, i])
        if not len(hits) or hits[0] == 0:
            raise ValueError("Prior is only for a later observed existing static object")
        observations = []
        for raw_index, frame in enumerate(raw_frames):
            tokens = list(frame["anns"]["track_tokens"])
            if actor not in tokens:
                continue
            j = tokens.index(actor)
            if frame["anns"]["gt_names"][j] != kind:
                raise ValueError("Static source category changes")
            velocities = frame["anns"].get("gt_velocity_3d")
            if velocities is None or len(velocities) != len(tokens):
                raise ValueError("Static prior needs measured velocity for every annotation")
            velocity = np.asarray(velocities[j], float)
            if velocity.shape != (3,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity) > 0.2:
                raise ValueError("Static prior contradicted by measured motion")
            observations.append((raw_index, frame, j, float(np.linalg.norm(velocity))))
        if not observations:
            raise ValueError("No measured static geometry")
        raw_index, frame, j, _ = observations[0]
        if raw_index < scene.metadata["source_start_frame"]:
            raise ValueError("Expected the first actual annotation inside the source window")
        evidence = request.get("evidence_refs")
        if request.get("immutable_visible_at_scene_start_reviewed") is not True or not request.get("reviewer"):
            raise ValueError("Earlier physical existence needs explicit pixel review")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Earlier existence evidence missing")
        has_initial = False
        checked = []
        for ref in evidence:
            idx, camera = ref.get("raw_index"), ref.get("camera")
            if type(idx) is not int or not 0 <= idx < raw_index:
                raise ValueError("Earlier existence evidence must precede first annotation")
            item = raw_frames[idx]["cams"].get(camera)
            path = Path(ref.get("path", ""))
            if item is None or not path.is_file() or not str(path).endswith(item["data_path"]):
                raise ValueError("Evidence image does not match raw camera")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != ref.get("sha256"):
                raise ValueError("Static prior image evidence changed")
            has_initial |= idx == scene.metadata["source_start_frame"]
            checked.append(dict(ref))
        if not has_initial:
            raise ValueError("Static prior requires reviewed scene-initial image")
        box = np.asarray(frame["anns"]["gt_boxes"][j], float)
        pose = transform_box(box, frame["ego2global_translation"], yaw_from_quaternion(frame["ego2global_rotation"]))
        pose[:3] -= np.asarray(scene.metadata["origin_xyz"])
        result.poses[: int(hits[0]), i] = pose
        result.velocities[:, i] = 0
        result.presence[:, i] = True
        direction = np.array([np.cos(pose[3]), np.sin(pose[3])])
        result.metadata["reactive_background"]["assignments"][actor] = {
            "points": [pose[:2].tolist(), (pose[:2] + direction * .01).tolist()],
            "desired_speed": 0.0,
            "navigation_source": "observed_stationary_hold",
        }
        entries[actor] = {
            "policy": POLICY, "source_kind": kind, "source_token": frame["token"],
            "source_timestamp_us": int(frame["timestamp"]), "source_raw_index": raw_index,
            "source_first_seen_tick": int(hits[0]), "raw_box": box.tolist(),
            "initial_physical_pose": pose.tolist(), "evidence_refs": checked,
            "reviewer": request["reviewer"], "observed_records": len(observations),
            "max_measured_speed_mps": max(x[3] for x in observations),
            "source_observations_preserved": True,
            "information_scope": "offline_immutable_surveyed_geometry_not_online_future_dynamic_state",
        }
    result.metadata["static_scene_context"] = {"version": 1, "policy": POLICY, "actors": entries}
    for actor in entries:
        i = result.actor_ids.index(actor)
        box = rectangle_corners(*result.poses[0, i, :2], result.poses[0, i, 3], *result.dimensions[i, :2])
        for j in np.flatnonzero(result.presence_at(0)):
            if i == j:
                continue
            other = rectangle_corners(*result.poses[0, j, :2], result.poses[0, j, 3], *result.dimensions[j, :2])
            if rectangles_overlap(box, other):
                raise ValueError("Surveyed static prior overlaps an initial physical object")
    result.validate()
    return result


def validate_static_scene_context(scene, raw_frames):
    """Recompute raw annotation and image-hash evidence at the render boundary."""
    actors = static_prior_actors(scene)
    if not actors:
        return {}
    original = copy.deepcopy(scene)
    stored = original.metadata.pop("static_scene_context")
    support = np.asarray(original.metadata["reactive_background"]["source_presence"], bool)
    requests = {}
    for actor in actors:
        i = original.actor_ids.index(actor)
        first = int(np.flatnonzero(support[:, i])[0])
        original.presence[:, i] = np.arange(len(original.times)) >= first
        original.poses[:first, i] = 0
        row = stored["actors"][actor]
        requests[actor] = {
            "immutable_visible_at_scene_start_reviewed": True,
            "reviewer": row["reviewer"], "evidence_refs": row["evidence_refs"],
        }
    recomputed = apply_static_scene_context(original, raw_frames, {
        "version": 1, "policy": POLICY,
        "source_log": scene.metadata["source_log"], "actors": requests,
    }).metadata["static_scene_context"]
    if stored != recomputed:
        raise ValueError("Static survey receipt differs from raw evidence")
    return recomputed
