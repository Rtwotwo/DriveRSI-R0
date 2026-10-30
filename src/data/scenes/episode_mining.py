"""Hash-verified, non-trainable candidate clips from recorded MTGS episodes.

Mining selects inspection evidence. It neither certifies traffic safety nor
promotes rendered displays to policy inputs or supervised driving targets.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from driversi.contracts.population import EpisodePopulation
from driversi.data.assets.mtgs_episode import (
    EPISODE_VERSION,
    _actions,
    _finite_json,
    _height_binding,
    _ids,
    _integer,
    _mask,
    _matrix,
    _number,
    _world,
)

MINING_VERSION = "mtgs-long-tail-candidates-v1"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, name) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe referenced path")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Referenced path is missing or escapes its root")
    return path


def _read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    _finite_json(value)
    if not isinstance(value, dict):
        raise ValueError("JSON manifest must be an object")
    return value, _digest(raw)


def _load_episode(path: Path) -> tuple[dict, dict, dict, list]:
    """Validate the existing export's integrity and causal sequence contracts."""
    document, digest = _read_json(path)
    if (document.get("version") != EPISODE_VERSION or document.get("schema_version") != 1
            or document.get("kind") != "mtgs_inspection_episode" or document.get("status") != "completed"
            or document.get("trainable") is not False or document.get("display_only") is not True):
        raise ValueError("A completed non-trainable MTGS inspection episode is required")
    eligibility = document.get("eligibility")
    required = ("visual_on_policy_rl", "visual_policy_imitation", "production_world_model_training")
    if not isinstance(eligibility, dict) or any(
            not isinstance(eligibility.get(k), dict) or eligibility[k].get("eligible") is not False
            for k in required):
        raise ValueError("Inspection eligibility cannot promote training")
    dt = _number(document.get("dt_seconds"), "dt_seconds")
    if dt <= 0 or document.get("action_space") != "jerk_discrete_12":
        raise ValueError("Invalid episode time step or action space")
    actors, controlled = _ids(document.get("actor_ids"), "actor_ids"), _ids(document.get("controlled_ids"), "controlled_ids")
    cameras = _ids(document.get("camera_ids"), "camera_ids")
    if not set(controlled).issubset(actors):
        raise ValueError("Controlled actors are missing from world")
    if document.get("state_columns") != ["x", "y", "z", "yaw", "vx", "vy", "vz", "goal_x", "goal_y", "stopped", "collision", "offroad", "reached_goal"]:
        raise ValueError("Unsupported state column contract")
    if document.get("motion_columns") != ["a_long", "a_lat", "steering"]:
        raise ValueError("Unsupported motion column contract")
    source = document.get("source", {})
    session_path = Path(source.get("manifest", "")).resolve()
    session, session_hash = _read_json(session_path)
    if session_hash != source.get("manifest_sha256"):
        raise ValueError("Source session manifest hash mismatch")
    artifact_root = Path(source.get("artifact_root", "")).resolve()
    if artifact_root != session_path.parent:
        raise ValueError("Artifact root differs from source session location")
    if (session.get("status") != "completed" or session.get("trainable") is not False
            or session.get("display_only") is not True):
        raise ValueError("Source session eligibility differs from inspection export")
    for key in ("dt_seconds", "controlled_ids", "camera_ids", "physics_build_id", "controller_configuration",
                "scenario_id", "source_log", "height_mode", "height_configuration",
                "population_configuration", "behavior_policies"):
        if document.get(key) != session.get(key):
            raise ValueError("Episode provenance differs from source session")
    if document.get("branch_lineage", []) != session.get("branch_lineage", []):
        raise ValueError("Branch lineage differs from source session")
    lineage = document.get("branch_lineage", [])
    if lineage:
        last = lineage[-1]
        if (type(last.get("generation")) is not int or type(last.get("tick")) is not int
                or last["generation"] < 0 or last["tick"] < 0):
            raise ValueError("Invalid branch evidence boundary")
    population = None
    if session.get("population_configuration") is not None:
        population = EpisodePopulation.from_dict(session["population_configuration"])
        population.validate_actors(tuple(controlled))
        if (session["population_configuration"] != population.to_dict()
                or document.get("behavior_policies") != population.behavior_policies):
            raise ValueError("Population provenance is not normalized or consistent")
    elif document.get("behavior_policies") is not None:
        raise ValueError("Behavior identities require a declared population")
    if document.get("policy_bindings_recorded", False) is not (population is not None):
        raise ValueError("Population recording flag contradicts the declared population")
    rgb_policy_input = None if population is not None and population.external_actor_ids else False
    protocol = session.get("protocol", {})
    if "rgb_is_policy_input" not in protocol or protocol["rgb_is_policy_input"] is not rgb_policy_input:
        raise ValueError("Source RGB policy input must be unknown for external policies and false for rules")
    tracked = [(path, digest), (session_path, session_hash)]
    summaries = document.get("generations")
    if (not isinstance(summaries, list) or not summaries
            or [s.get("generation") for s in summaries] != list(range(len(summaries)))):
        raise ValueError("Invalid generation coverage")
    expected = {f"generation-{g:03d}-{kind}.jsonl" for g in range(len(summaries))
                for kind in ("physics", "sensors", "camera-intervals")}
    files = document.get("files")
    if (not isinstance(files, list) or len(files) != len(expected)
            or {entry.get("path") for entry in files} != expected):
        raise ValueError("Incomplete or duplicated sequence files")
    contents = {}
    for entry in files:
        f = _relative(path.parent, entry["path"])
        raw = f.read_bytes()
        if _digest(raw) != entry.get("sha256"):
            raise ValueError("Episode sequence hash mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        _finite_json(rows)
        if len(rows) != entry.get("records") or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Sequence record count or shape mismatch")
        contents[entry["path"]] = rows
        tracked.append((f, entry["sha256"]))
    artifacts, groups = {}, {}
    for summary in summaries:
        g = summary["generation"]
        source_initial = session.get("initial_world_by_generation", {}).get(str(g), {})
        if source_initial.get("all_actor_ids") != actors:
            raise ValueError("Episode actor order differs from source initial world")
        physics = contents[f"generation-{g:03d}-physics.jsonl"]
        sensors = contents[f"generation-{g:03d}-sensors.jsonl"]
        intervals = contents[f"generation-{g:03d}-camera-intervals.jsonl"]
        if (len(physics) != summary.get("physics_transitions") or not physics
                or len(sensors) != summary.get("sensor_frames") or not sensors
                or len(intervals) != summary.get("camera_intervals")):
            raise ValueError("Generation summary differs from sequence")
        previous = None
        for i, row in enumerate(physics):
            if (row.get("generation") != g or row.get("observation_tick") != i
                    or row.get("successor_tick") != i + 1 or row.get("controlled_ids") != controlled
                    or row.get("duration_seconds") != dt):
                raise ValueError("Physics tick, generation or actor sequence mismatch")
            worlds = row.get("privileged", {})
            before, after = worlds.get("world_before", {}), worlds.get("world_after", {})
            for world, tick in ((before, i), (after, i+1)):
                if world.get("tick") != tick or type(world.get("done")) is not bool:
                    raise ValueError("World tick/done mismatch")
                _matrix(world.get("state"), len(actors), 13, "world state")
                _matrix(world.get("motion"), len(actors), 3, "world motion")
                _mask(world.get("alive"), len(controlled), "world alive")
            if previous is not None and before != previous:
                raise ValueError("Broken world successor continuity")
            if before["done"] or after["done"] != (i == len(physics)-1):
                raise ValueError("Invalid completed episode boundary")
            previous = after
            masks = row.get("masks", {})
            for name in ("valid", "terminated", "truncated"):
                _mask(masks.get(name), len(controlled), name)
            if masks["valid"] != before["alive"]:
                raise ValueError("Valid mask differs from pre-action alive mask")
            for j in range(len(controlled)):
                if (masks["terminated"][j] and (not masks["valid"][j] or masks["truncated"][j])
                        or masks["truncated"][j] and (not masks["valid"][j] or not after["alive"][j] or not after["done"])
                        or after["alive"][j] != (before["alive"][j] and not masks["terminated"][j])):
                    raise ValueError("Invalid lifecycle masks")
            behavior = row.get("behavior", {})
            policy = _actions(behavior.get("policy_actions"), controlled, "policy_actions")
            executed = _actions(behavior.get("executed_actions"), controlled, "executed_actions")
            if (behavior.get("execution_differs_from_policy") != {a: policy[a] != executed[a] for a in controlled}
                    or "rgb_is_policy_input" not in behavior
                    or behavior["rgb_is_policy_input"] is not rgb_policy_input):
                raise ValueError("Behavior difference or RGB policy boundary mismatch")
            if (behavior.get("configuration_sha256") != document.get("controller_configuration_sha256")
                    or behavior.get("controller_version") != document.get("controller_configuration", {}).get("controller")):
                raise ValueError("Behavior identity differs from episode")
            if set(row.get("events", {})) != {"collision", "offroad", "goal"}:
                raise ValueError("Missing event masks")
            for key, values in row["events"].items():
                _mask(values, len(controlled), key)
        source_steps = [r for r in session.get("steps", []) if r.get("generation") == g]
        if len(source_steps) != len(physics):
            raise ValueError("Episode transition coverage differs from source session")
        source_initial = session.get("initial_world_by_generation", {}).get(str(g), {})
        for row, original in zip(physics, source_steps, strict=True):
            expected_before = _world(source_initial, actors, controlled, dt, document["physics_build_id"])
            expected_after = _world(original, actors, controlled, dt, document["physics_build_id"])
            if (original.get("all_actor_ids") != actors
                    or row["privileged"]["world_before"] != expected_before
                    or row["privileged"]["world_after"] != expected_after
                    or row["events"] != original.get("events")
                    or row["masks"] != {k: original.get(k) for k in ("valid", "terminated", "truncated")}
                    or any(row["behavior"].get(k) != original.get(k) for k in ("policy_actions", "executed_actions",
                           "behavior_policies", "action_origins", "override_actor_ids"))):
                raise ValueError("Episode physics differs from source session")
            source_initial = original
        source_frames = [r for r in session.get("frames", []) if r.get("generation") == g]
        if len(source_frames) != len(sensors):
            raise ValueError("Episode capture coverage differs from source session; missing bracketing")
        last_tick, epoch = -1, None
        for frame, original_frame in zip(sensors, source_frames, strict=True):
            tick = _integer(frame.get("tick"), "capture tick")
            if frame.get("generation") != g or tick <= last_tick or tick > len(physics):
                raise ValueError("Invalid capture sequence")
            last_tick = tick
            if ("used_for_policy" not in frame or frame["used_for_policy"] is not rgb_policy_input
                    or not math.isclose(_number(frame.get("time_seconds"), "capture time"), tick*dt, abs_tol=1e-9)):
                raise ValueError("Capture time or actor-input boundary mismatch")
            stamp = _integer(frame.get("simulation_timestamp_us"), "simulation_timestamp_us")
            epoch = stamp-round(tick*dt*1e6) if epoch is None else epoch
            if stamp != epoch+round(tick*dt*1e6):
                raise ValueError("Capture timestamps are not on the simulation clock")
            views = frame.get("cameras")
            if not isinstance(views, list) or [c.get("camera_id") for c in views] != cameras:
                raise ValueError("Camera coverage mismatch")
            if (frame.get("height_binding") != original_frame.get("height_binding")
                    or frame.get("coverage") != original_frame.get("coverage", {})
                    or frame["tick"] != original_frame.get("tick")):
                raise ValueError("Episode capture diagnostics differ from source session")
            original_cameras = {c["camera_id"]: c for c in original_frame["cameras"]}
            raw_views = []
            for camera in views:
                if ("used_for_policy" not in camera or camera["used_for_policy"] is not rgb_policy_input
                        or camera.get("owner_actor_id") != document.get("camera_owners", {}).get(camera["camera_id"])):
                    raise ValueError("Camera owner or policy boundary mismatch")
                metadata = camera.get("capture_metadata", {})
                references = camera.get("artifacts", {})
                if set(references) != {"image", "pinhole", "valid", "source"}:
                    raise ValueError("Incomplete camera artifact references")
                original = original_cameras[camera["camera_id"]]
                if metadata != {k: v for k, v in original.items() if k not in references}:
                    raise ValueError("Camera metadata differs from source session")
                if (camera.get("virtual", False) != original.get("virtual", False)
                        or any((value.get("path") if isinstance(value, dict) else value) != original.get(k)
                               for k, value in references.items())):
                    raise ValueError("Camera artifact identity differs from source session")
                raw_views.append(metadata)
                for artifact in camera.get("artifacts", {}).values():
                    if artifact is None:
                        continue
                    image = _relative(artifact_root, artifact.get("path"))
                    expected_image = (artifact.get("sha256"), artifact.get("bytes"))
                    if image in artifacts:
                        if artifacts[image] != expected_image:
                            raise ValueError("Contradictory camera artifact references")
                        continue
                    if image.stat().st_size != artifact.get("bytes") or _file_hash(image) != artifact.get("sha256"):
                        raise ValueError("Camera artifact hash or size mismatch")
                    artifacts[image] = expected_image
            _height_binding(frame, raw_views, controlled, document.get("height_mode"))
        if sensors[0]["tick"] != 0 or sensors[-1]["tick"] != len(physics):
            raise ValueError("Capture endpoints are missing; cannot invent bracketing observations")
        expected_intervals = []
        for left, right in zip(sensors, sensors[1:]):
            for camera in cameras:
                begin, end = left["tick"], right["tick"]
                expected_intervals.append({"generation": g, "camera_id": camera,
                    "owner_actor_id": document["camera_owners"][camera], "start_tick": begin, "end_tick": end,
                    "duration_seconds": (end-begin)*dt, "input_frame_tick": begin, "target_frame_tick": end,
                    "used_for_policy": rgb_policy_input, "joint_action_sequence": [{
                        "observation_tick": row["observation_tick"], "successor_tick": row["successor_tick"],
                        "executed_actions": row["behavior"]["executed_actions"]} for row in physics[begin:end]]})
        if intervals != expected_intervals:
            raise ValueError("Camera intervals lost or changed intervening joint actions")
        groups[g] = {"physics": physics, "sensors": sensors, "camera_intervals": intervals}
    tracked.extend((p, pair[0]) for p, pair in artifacts.items())
    provenance = {"episode_manifest": str(path), "episode_manifest_sha256": digest,
                  "session_manifest": str(session_path), "session_manifest_sha256": session_hash,
                  "artifact_root": str(artifact_root), "protocol": session.get("protocol"),
                  "scenario_id": document.get("scenario_id"), "source_log": document.get("source_log")}
    return document, provenance, groups, tracked


def _evidence(document: dict, group: dict, braking: float, speed: float) -> dict:
    points = defaultdict(list)

    def add(channel, reason, tick, *, actor=None, camera=None, scope=None, **details):
        points[(channel, reason, actor, camera, scope)].append({"tick": tick, **details})

    for row in group["physics"]:
        for i, actor in enumerate(document["controlled_ids"]):
            if not row["masks"]["valid"][i]:
                continue
            tick = row["successor_tick"]
            for event in ("collision", "offroad"):
                if row["events"][event][i]:
                    add("driving_event", event, tick, actor=actor)
            wi = document["actor_ids"].index(actor)
            after = row["privileged"]["world_after"]
            state, motion = after["state"][wi], after["motion"][wi]
            velocity = state[4]*math.cos(state[3]) + state[5]*math.sin(state[3])
            if velocity >= speed and motion[0] <= braking:
                add("driving_behavior", "severe_deceleration", tick, actor=actor,
                    speed_mps=velocity, speed_semantics="signed_forward_speed", acceleration_mps2=motion[0])
            behavior = row["behavior"]
            if behavior["execution_differs_from_policy"][actor]:
                add("interaction_intervention", "policy_execution_mismatch", tick, actor=actor,
                    policy_action=behavior["policy_actions"][actor], executed_action=behavior["executed_actions"][actor],
                    semantics="action_difference_not_causal_success_or_override_intent")
    for frame in group["sensors"]:
        tick, binding, coverage = frame["tick"], frame.get("height_binding"), frame.get("coverage", {})
        if binding is None:
            add("render_quality", "height_metadata_unavailable", tick, scope="static_known_limit")
        else:
            failures = {}
            for camera in binding["per_camera"]:
                for actor in camera["actors"]:
                    if not actor["supported"]:
                        failures.setdefault(actor["actor_id"], set()).add(actor["reason"])
            for actor, reasons in failures.items():
                add("render_quality", "height_unsupported", tick, actor=actor,
                    scope="dynamic_support_failure" if document.get("height_mode") == "reference-track" else "static_known_limit",
                    support_reasons=sorted(reasons))
        if not coverage:
            add("render_quality", "coverage_metadata_unavailable", tick, scope="static_known_limit")
        for field, reason in (("world_without_rigid_nodes", "world_actor_without_render_asset"),
                              ("reconstruction_only_nodes", "render_actor_without_physics")):
            if coverage.get(field):
                add("render_quality", reason, tick, scope="asset_coverage_gap", affected_actor_ids=sorted(set(coverage[field])))
        if coverage.get("ego_appearance") == "camera_car_not_reconstructed":
            add("render_quality", "ego_render_asset_missing", tick, actor="ego", scope="static_known_limit")
        for field, reason in (("source_without_physics_actor", "source_actor_without_physics"),
                              ("source_without_reconstruction_node", "source_actor_without_render_asset")):
            missing = {item["track_token"] for inventory in coverage.get("source_inventories", [])
                       for item in inventory.get(field, [])}
            if missing:
                add("render_quality", reason, tick, scope="asset_coverage_gap", affected_actor_ids=sorted(missing))
        for camera in frame["cameras"]:
            support = camera.get("capture_metadata", {}).get("support", {})
            if camera.get("virtual") and support.get("novel_view_quality") != "verified":
                add("render_quality", "novel_view_quality_unverified", tick,
                    actor=camera["owner_actor_id"], camera=camera["camera_id"], scope="static_known_limit")
    return points


def _runs(values: list, ticks: list[int]) -> list[list]:
    order = {tick: i for i, tick in enumerate(ticks)}
    runs = []
    for point in values:
        if not runs or order[point["tick"]] != order[runs[-1][-1]["tick"]]+1:
            runs.append([])
        runs[-1].append(point)
    return runs


def mine_mtgs_episode(episode_manifest: Path, output: Path, *, context_before_seconds: float = 1.,
                      context_after_seconds: float = 1., braking_threshold_mps2: float = -3.,
                      minimum_speed_mps: float = 1.) -> dict:
    """Extract auditable, camera-aligned inspection clips; no image copies."""
    path, output = Path(episode_manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Mining output must be a new directory")
    before = _number(context_before_seconds, "context_before_seconds")
    after = _number(context_after_seconds, "context_after_seconds")
    braking = _number(braking_threshold_mps2, "braking_threshold_mps2")
    speed = _number(minimum_speed_mps, "minimum_speed_mps")
    if before < 0 or after < 0 or braking >= 0 or speed < 0:
        raise ValueError("Context/speed must be nonnegative and braking threshold negative")
    document, source, groups, tracked = _load_episode(path)
    dt = document["dt_seconds"]
    config = {"context_before_seconds": before, "context_after_seconds": after,
              "braking_threshold_mps2": braking, "minimum_speed_mps": speed,
              "alignment": "outward_existing_capture_ticks_within_generation",
              "braking_semantics": "successor_a_long_and_signed_forward_speed_not_unsafe_or_teacher_label",
              "inherited_prefix_evidence": "excluded_context_may_include_prefix"}
    candidates, clips = [], {}
    for generation, group in groups.items():
        physics, sensors = group["physics"], group["sensors"]
        ticks = [f["tick"] for f in sensors]
        points = _evidence(document, group, braking, speed)
        lineage = document.get("branch_lineage", [])
        if lineage:
            last = lineage[-1]
            points = {key: [point for point in values
                       if generation > last["generation"] or
                       (generation == last["generation"] and point["tick"] > last["tick"])]
                      for key, values in points.items()}
            points = {key: values for key, values in points.items() if values}
        for key in sorted(points, key=lambda k: tuple("" if item is None else item for item in k)):
            channel, reason, actor, camera, scope = key
            evidence_clock = ticks if channel == "render_quality" else list(range(1, len(physics)+1))
            for run in _runs(points[key], evidence_clock):
                first, last = run[0]["tick"], run[-1]["tick"]
                # Physics evidence is a transition: always retain its pre-action world,
                # even when the caller asks for zero context.
                event_begin = first if channel == "render_quality" else first-1
                desired_start, desired_end = event_begin-before/dt, last+after/dt
                start = max(t for t in ticks if t <= max(0, desired_start)+1e-9)
                end = min(t for t in ticks if t >= min(len(physics), desired_end)-1e-9)
                identity = {"episode": source["episode_manifest_sha256"], "generation": generation,
                            "channel": channel, "reason": reason, "actor": actor, "camera": camera,
                            "first": first, "last": last, "configuration": config}
                cid = _digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())[:24]
                selected = physics[start:end]
                candidate = {"candidate_id": cid, "channel": channel, "reason": reason,
                    "generation": generation, "actor_id": actor, "camera_id": camera,
                    "quality_scope": scope, "evidence_start_tick": first, "evidence_end_tick": last,
                    "evidence_count": len(run), "start_tick": start, "end_tick": end,
                    "duration_seconds": (end-start)*dt,
                    "context_clipped_before": desired_start < 0, "context_clipped_after": desired_end > len(physics),
                    "capture_aligned": True, "trainable": False}
                initial_world = (physics[start]["privileged"]["world_before"] if start < len(physics)
                                 else physics[-1]["privileged"]["world_after"])
                clip = {"schema_version": 1, "kind": "mtgs_candidate_clip", "version": MINING_VERSION,
                        "candidate": candidate.copy(), "source": source, "evidence": run,
                        "initial_world": initial_world, "physics": selected,
                        "sensors": [f for f in sensors if start <= f["tick"] <= end],
                        "camera_intervals": [i for i in group["camera_intervals"]
                                             if start <= i["start_tick"] and i["end_tick"] <= end],
                        "trainable": False, "eligibility": document["eligibility"],
                        "permissions": document.get("permissions", {}),
                        "unavailable_fields": document.get("unavailable_fields", []),
                        "controller_configuration_sha256": document.get("controller_configuration_sha256"),
                        "actor_ids": document["actor_ids"], "controlled_ids": document["controlled_ids"],
                        "dt_seconds": dt, "action_space": document["action_space"],
                        "state_columns": document["state_columns"], "motion_columns": document["motion_columns"],
                        "source_limits": document.get("source_limits", []),
                        "policy_identity": {"controller_configuration": document.get("controller_configuration"),
                                            "population_configuration": document.get("population_configuration"),
                                            "behavior_policies": document.get("behavior_policies"),
                                            "policy_bindings_recorded": document.get("policy_bindings_recorded", False),
                                            "physics_build_id": document.get("physics_build_id"),
                                            "renderer_backend": document.get("renderer_backend")}}
                raw = (json.dumps(clip, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n").encode()
                candidate["clip"] = {"path": f"clips/{cid}.json", "sha256": _digest(raw)}
                clips[candidate["clip"]["path"]] = raw
                candidates.append(candidate)
    # Validate all inputs before any output is created. This also rejects source
    # changes during mining; immutable production datasets remain recommended.
    if any(_file_hash(p) != expected for p, expected in tracked):
        raise ValueError("An episode input changed during mining")
    result = {"schema_version": 1, "kind": "mtgs_long_tail_candidates", "version": MINING_VERSION,
              "status": "completed", "trainable": False, "display_only": True,
              "eligibility": document["eligibility"], "source": source, "configuration": config,
              "population_configuration": document.get("population_configuration"),
              "behavior_policies": document.get("behavior_policies"),
              "policy_bindings_recorded": document.get("policy_bindings_recorded", False),
              "semantics": "inspection_candidates_not_statistical_long_tail_or_valid_training_labels",
              "verification": {"input_files_verified": len(tracked), "all_hashes_verified": True,
                               "all_joint_actions_retained": True, "images_copied": 0},
              "summary": {"candidates": len(candidates),
                          "by_channel": dict(Counter(c["channel"] for c in candidates)),
                          "by_reason": dict(Counter(c["reason"] for c in candidates))},
              "candidates": candidates}
    output.mkdir(parents=True)
    if clips:
        (output/"clips").mkdir()
    for name, raw in clips.items():
        with (output/name).open("xb") as stream:
            stream.write(raw)
    with (output/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return result
