"""Strict, lossless references for inspecting completed MTGS world sessions.

This exporter never creates policy observations from display images, fills a
missing initial state, copies sensor data, or promotes a diagnostic run to a
training dataset. Only the Python standard library is needed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

EPISODE_VERSION = "mtgs-inspection-episode-v1"
_STATE_COLUMNS = [
    "x",
    "y",
    "z",
    "yaw",
    "vx",
    "vy",
    "vz",
    "goal_x",
    "goal_y",
    "stopped",
    "collision",
    "offroad",
    "reached_goal",
]
_MOTION_COLUMNS = ["a_long", "a_lat", "steering"]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_json(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON data")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _ids(value, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must contain unique nonempty IDs")
    return value


def _mask(value, count: int, name: str) -> list[bool]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(type(x) is not bool for x in value)
    ):
        raise ValueError(f"Invalid {name} mask")
    return value


def _matrix(value, rows: int, columns: int, name: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != rows
        or any(not isinstance(row, list) or len(row) != columns for row in value)
    ):
        raise ValueError(f"Invalid {name} shape")
    for row in value:
        for item in row:
            _number(item, name)


def _world(
    record: dict, actor_ids: list[str], controlled: list[str], dt: float, build: str
) -> dict:
    if not isinstance(record, dict):
        raise ValueError("World record must be an object")
    if record.get("all_actor_ids") != actor_ids or record.get("controlled_ids") != controlled:
        raise ValueError("World actor order changed")
    if record.get("build_id") != build or _number(record.get("dt_seconds"), "world dt") != dt:
        raise ValueError("World build or dt changed")
    _integer(record.get("tick"), "tick")
    _matrix(record.get("state"), len(actor_ids), 13, "state")
    _matrix(record.get("motion"), len(actor_ids), 3, "motion")
    _mask(record.get("alive"), len(controlled), "alive")
    if type(record.get("done")) is not bool:
        raise ValueError("World done must be boolean")
    result = {key: record[key] for key in ("tick", "state", "motion", "alive", "done")}
    if "presence" in record:
        result["presence"] = _mask(record["presence"], len(actor_ids), "presence")
        if not all(result["presence"][actor_ids.index(a)] for a in controlled):
            raise ValueError("Controlled participant cannot disappear")
    if "lifecycle" in record:
        result["lifecycle"] = record["lifecycle"]
    return result


def _actions(value, controlled: list[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(controlled):
        raise ValueError(f"{name} must cover exactly the controlled actor IDs")
    if any(type(action) is not int or not 0 <= action < 12 for action in value.values()):
        raise ValueError(f"{name} requires jerk_discrete_12 integer actions")
    return value


def _native_actions(value, controlled, name):
    if not isinstance(value, dict) or set(value) != set(controlled):
        raise ValueError(f"{name} must cover exactly the controlled actor IDs")
    for row in value.values():
        _matrix([row], 1, 2, name)
    return value


def _native_replay_contract(document, population, controlled):
    native = population is not None and all(
        p["controller"] == "external-native-state-v1" for p in population.behavior_policies.values()
    )
    if not native:
        if document.get("native_replay") is not None:
            raise ValueError("Native replay requires native actor ownership")
        return None
    value = document.get("native_replay")
    if not isinstance(value, dict):
        raise ValueError("Native replay evidence required")
    for key in ("source_rollout", "source_targets"):
        text = value.get(key)
        if (
            not isinstance(text, str)
            or len(text) != 64
            or any(c not in "0123456789abcdef" for c in text)
        ):
            raise ValueError("Native replay source hash required")
    end = _integer(value.get("capture_end_tick"), "native capture end", minimum=1)
    full = _integer(value.get("full_physics_steps"), "native physical steps", minimum=end)
    if (
        value.get("trainable") is not False
        or value.get("rendered_rgb_used_for_actions") is not False
    ):
        raise ValueError("Native replay must preserve diagnostic policy-input boundary")
    return {**value, "capture_end_tick": end, "full_physics_steps": full}


def _artifact(root: Path, value, cache: dict) -> dict:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe camera artifact path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ValueError("Unsafe camera artifact path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Camera artifact is missing or escapes source artifact root")
    if value not in cache:
        cache[value] = {"path": value, "sha256": _hash(path), "bytes": path.stat().st_size}
    return cache[value]


def _camera(
    camera,
    root: Path,
    ids: list[str],
    names: list[str],
    cache: dict,
    used_for_policy: bool | None = False,
) -> dict:
    if not isinstance(camera, dict) or camera.get("camera_id") not in names:
        raise ValueError("Unknown camera ID")
    if camera.get("owner_actor_id") not in ids:
        raise ValueError("Camera owner is absent from the physical world")
    _integer(camera.get("width"), "camera width", minimum=1)
    _integer(camera.get("height"), "camera height", minimum=1)
    _matrix(camera.get("K"), 3, 3, "camera K")
    for key in ("source_timestamp_us", "camera_timestamp_us"):
        _integer(camera.get(key), key)
    if type(camera.get("source_seed_error_us")) is not int:
        raise ValueError("Camera seed offset must be an integer")
    fraction = _number(camera.get("source_lens_valid_fraction"), "camera valid fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("Camera valid fraction outside [0,1]")
    if not isinstance(camera.get("camera_model"), str) or not camera["camera_model"]:
        raise ValueError("Camera model is missing")
    virtual = camera.get("virtual", False)
    if type(virtual) is not bool:
        raise ValueError("Camera virtual flag must be boolean")
    # A virtual owner has no source photograph. It must never borrow the ego
    # reference and label it as an image captured from the virtual vehicle.
    artifacts = {
        key: _artifact(root, camera.get(key), cache) for key in ("image", "pinhole", "valid")
    }
    if virtual and "source" in camera and camera["source"] is None:
        artifacts["source"] = None
        source_comparison = "unavailable_no_real_image_for_virtual_owner"
    else:
        artifacts["source"] = _artifact(root, camera.get("source"), cache)
        source_comparison = (
            "source_seed_reference_not_virtual_owner_ground_truth"
            if virtual
            else "source_camera_reference_only_not_counterfactual_ground_truth"
        )
    return {
        "camera_id": camera["camera_id"],
        "owner_actor_id": camera["owner_actor_id"],
        "virtual": virtual,
        "source_comparison": source_comparison,
        "used_for_policy": used_for_policy,
        "artifacts": artifacts,
        "capture_metadata": {key: value for key, value in camera.items() if key not in artifacts},
    }


def _height_binding(frame: dict, cameras: list[dict], controlled: list[str], mode):
    """Preserve a recorded display-height audit; absence is never support."""
    binding = frame.get("height_binding")
    if binding is None:
        if mode is not None or any("height_binding" in camera for camera in cameras):
            raise ValueError("Declared height mode requires complete frame height_binding")
        return None
    if not isinstance(binding, dict) or binding.get("mode") != mode:
        raise ValueError("Frame height binding mode differs from session")
    per_camera = binding.get("per_camera")
    if (
        not isinstance(per_camera, list)
        or len(per_camera) != len(cameras)
        or any(not isinstance(item, dict) for item in per_camera)
        or [item.get("camera_id") for item in per_camera]
        != [camera["camera_id"] for camera in cameras]
    ):
        raise ValueError("Height binding camera coverage differs from rendered cameras")
    support = []
    for item, camera in zip(per_camera, cameras, strict=True):
        if item.get("source_timestamp_us") != camera["source_timestamp_us"]:
            raise ValueError("Height binding source clock differs from camera")
        actors = item.get("actors")
        if (
            not isinstance(actors, list)
            or len(actors) != len(controlled)
            or any(not isinstance(actor, dict) for actor in actors)
            or [actor.get("actor_id") for actor in actors] != controlled
        ):
            raise ValueError("Height binding actor coverage differs from controlled actor order")
        for actor in actors:
            if (
                actor.get("mode") != mode
                or type(actor.get("supported")) is not bool
                or not isinstance(actor.get("reason"), str)
                or not actor["reason"]
            ):
                raise ValueError("Height binding requires explicit actor mode, support, and reason")
            _number(actor.get("correction_m"), "height correction")
            if not actor["supported"] and actor["correction_m"] != 0:
                raise ValueError("Unsupported height binding must retain zero fallback correction")
        all_supported = all(actor["supported"] for actor in actors)
        if (
            type(item.get("all_controlled_supported")) is not bool
            or item["all_controlled_supported"] != all_supported
            or type(camera.get("scene_height_supported")) is not bool
            or camera["scene_height_supported"] != all_supported
        ):
            raise ValueError("Height binding scene support contradicts actor diagnostics")
        owner = next(
            (actor for actor in actors if actor["actor_id"] == camera["owner_actor_id"]), None
        )
        camera_height = camera.get("height_binding")
        if owner is not None:
            if camera_height != owner:
                raise ValueError("Camera owner height binding differs from whole-scene diagnostics")
        elif (
            not isinstance(camera_height, dict)
            or camera_height.get("actor_id") != camera["owner_actor_id"]
            or camera_height.get("mode") != mode
            or camera_height.get("supported") is not False
            or camera_height.get("reason") != "uncontrolled_source_camera"
            or camera_height.get("fallback") != "retain_source_time_height"
            or _number(camera_height.get("correction_m"), "camera height correction") != 0
        ):
            # An uncontrolled source camera is a physical world actor but is
            # deliberately absent from the all-controlled support aggregation.
            raise ValueError(
                "Uncontrolled camera requires explicit unsupported source-height fallback"
            )
        support.append(all_supported)
    if type(binding.get("all_controlled_supported")) is not bool or binding[
        "all_controlled_supported"
    ] != all(support):
        raise ValueError("Height binding frame support contradicts camera diagnostics")
    return binding


def _population_metadata(step: dict, population, controlled, policy, executed) -> dict:
    fields = ("behavior_policies", "action_origins", "override_actor_ids")
    if population is None:
        if any(step.get(field) is not None for field in fields):
            raise ValueError("Behavior identity requires a population_configuration")
        return {"policy_bindings_recorded": False, **dict.fromkeys(fields)}
    if step.get("behavior_policies") != population.behavior_policies:
        raise ValueError("Behavior policies changed within the fixed population")
    overrides, origins = step.get("override_actor_ids"), step.get("action_origins")
    if (
        not isinstance(overrides, list)
        or any(not isinstance(actor, str) for actor in overrides)
        or len(set(overrides)) != len(overrides)
        or not set(overrides) <= set(controlled)
        or not isinstance(origins, dict)
        or set(origins) != set(controlled)
    ):
        raise ValueError("Action origins and override actors must cover controlled actors")
    for actor in controlled:
        expected = (
            "override" if actor in overrides else population.behavior_policies[actor]["controller"]
        )
        if origins[actor] != expected or (
            policy[actor] != executed[actor] and actor not in overrides
        ):
            raise ValueError("Action origin contradicts the fixed policy or execution override")
    return {"policy_bindings_recorded": True, **{key: step[key] for key in fields}}


def export_mtgs_episode(manifest_path: Path, output: Path) -> dict:
    """Validate, then export completed generations without loading any models.

    Every generation needs the exact worker initial state. Legacy display
    manifests without it are rejected; source logs are never used to invent it.
    The output directory must not exist. Camera intervals retain *all* executed
    joint actions, and privileged physics is separated from sensor references.
    """
    from driversi.rendering.diagnostics.quality import summarize_quality, validate_frame_quality

    quality_reports = []
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Episode output must be a new directory")
    raw = manifest_path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    _finite_json(document)
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("kind") != "mtgs_world_session"
        or document.get("status") != "completed"
    ):
        raise ValueError("A completed MTGS world session schema_version=1 is required")
    if document.get("trainable") is not False or document.get("display_only") is not True:
        raise ValueError("This exporter accepts display-only non-trainable sessions")
    dt = _number(document.get("dt_seconds"), "dt_seconds")
    if dt <= 0:
        raise ValueError("dt_seconds must be positive")
    controlled = _ids(document.get("controlled_ids"), "controlled_ids")
    camera_names = _ids(document.get("camera_ids"), "camera_ids")
    initial = document.get("initial_world_by_generation")
    if not isinstance(initial, dict) or not initial:
        raise ValueError(
            "Missing exact initial_world_by_generation; legacy states cannot be reconstructed"
        )
    generations = []
    for key in initial:
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
            raise ValueError("Initial generation keys must be canonical nonnegative integers")
        generations.append(int(key))
    generations.sort()
    if generations != list(range(len(generations))):
        raise ValueError("Generation coverage must start at zero and be contiguous")
    first = initial["0"]
    if not isinstance(first, dict):
        raise ValueError("Initial world must be an object")
    actor_ids = _ids(first.get("all_actor_ids"), "all_actor_ids")
    if not set(controlled).issubset(actor_ids):
        raise ValueError("Controlled actors must exist in world actor order")
    build = document.get("physics_build_id")
    if not isinstance(build, str) or not build:
        raise ValueError("Physics build ID is missing")
    configuration = document.get("controller_configuration")
    if (
        not isinstance(configuration, dict)
        or not isinstance(configuration.get("controller"), str)
        or not configuration["controller"]
    ):
        raise ValueError("Behavior controller identity/configuration is missing")
    config_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    population = None
    if document.get("population_configuration") is not None:
        from driversi.contracts.population import EpisodePopulation

        population = EpisodePopulation.from_dict(document["population_configuration"])
        population.validate_actors(tuple(controlled))
        if (
            document.get("behavior_policies") != population.behavior_policies
            or configuration.get("population_configuration") != population.to_dict()
        ):
            raise ValueError("Population declaration and controller configuration differ")
    elif document.get("behavior_policies") is not None:
        raise ValueError("Behavior identity requires a population_configuration")
    native_replay = _native_replay_contract(document, population, controlled)
    rgb_policy_input = (
        None
        if population is not None and (population.external_actor_ids or native_replay)
        else False
    )
    protocol = document.get("protocol")
    teacher_inputs = None
    if isinstance(protocol, dict) and protocol.get("kind") == "learned_privileged_population":
        from driversi.data.datasets.teacher_inputs import validate_teacher_inputs

        if population is None or set(population.external_actor_ids) != set(controlled):
            raise ValueError("Learned teacher must own every controlled actor")
        teacher_inputs = validate_teacher_inputs(document, manifest_path.parent)
        rgb_policy_input = False
    if (
        not isinstance(protocol, dict)
        or "rgb_is_policy_input" not in protocol
        or protocol["rgb_is_policy_input"] is not rgb_policy_input
    ):
        raise ValueError(
            "RGB policy input must be unknown for external policies and false for rules"
        )
    steps, frames = document.get("steps"), document.get("frames")
    if not isinstance(steps, list) or not steps or not isinstance(frames, list) or not frames:
        raise ValueError("Session must contain physics steps and frames")
    by_step = {g: [] for g in generations}
    by_frame = {g: [] for g in generations}
    for records, grouped in ((steps, by_step), (frames, by_frame)):
        previous_generation = -1
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Session records must be objects")
            generation = _integer(record.get("generation"), "generation")
            if generation not in grouped or generation < previous_generation:
                raise ValueError("Unknown or interleaved generation")
            previous_generation = generation
            grouped[generation].append(record)
    physics, sensors, intervals, summaries, artifacts = {}, {}, {}, [], {}
    camera_owners = {}
    for generation in generations:
        if (
            population is not None
            and initial[str(generation)].get("behavior_policies") != population.behavior_policies
        ):
            raise ValueError("Initial behavior policies differ from fixed population")
        before = _world(initial[str(generation)], actor_ids, controlled, dt, build)
        if before["tick"] != 0 or before["done"]:
            raise ValueError("Each initial world must be tick zero and not done")
        if not by_step[generation]:
            raise ValueError("Every generation needs completed physics steps")
        records = []
        for tick, step in enumerate(by_step[generation], 1):
            if (
                type(step.get("schema_version")) is not int
                or step.get("schema_version") != 1
                or step.get("status") != "ok"
                or step.get("operation") != ("native_step" if native_replay else "step")
            ):
                raise ValueError(
                    "Each transition must be a successful worker step schema_version=1"
                )
            if step.get("capture_status") not in (None, "captured", "not_due"):
                raise ValueError("Completed transition has an unfinished or failed capture")
            after = _world(step, actor_ids, controlled, dt, build)
            if after["tick"] != tick or before["done"]:
                raise ValueError("Physics ticks must be contiguous with no step after done")
            valid = _mask(step.get("valid"), len(controlled), "valid")
            terminated = _mask(step.get("terminated"), len(controlled), "terminated")
            truncated = _mask(step.get("truncated"), len(controlled), "truncated")
            if valid != before["alive"]:
                raise ValueError("Valid mask differs from exact pre-action alive state")
            events = step.get("events")
            required_events = {"collision", "offroad", "goal"}
            if (
                not isinstance(events, dict)
                or not required_events <= set(events)
                or set(events) - required_events - {"route_support_end"}
            ):
                raise ValueError("Missing or unknown event masks")
            for name, mask in events.items():
                _mask(mask, len(controlled), name)
            support_end = events.get("route_support_end", [False] * len(controlled))
            for i in range(len(controlled)):
                # A route support boundary removes only this actor from sampling.
                # Time truncation instead retains live actors in a finished world.
                expected_alive = valid[i] and not (terminated[i] or support_end[i])
                if (
                    terminated[i]
                    and (not valid[i] or truncated[i])
                    or support_end[i]
                    and (
                        not valid[i]
                        or not truncated[i]
                        or terminated[i]
                        or any(events[name][i] for name in required_events)
                    )
                    or truncated[i]
                    and (
                        not valid[i]
                        or not support_end[i]
                        and (not after["alive"][i] or not after["done"])
                    )
                    or after["alive"][i] != expected_alive
                ):
                    raise ValueError("Inconsistent terminal/truncation/alive masks")
            action_reader = _native_actions if native_replay else _actions
            policy = action_reader(step.get("policy_actions"), controlled, "policy_actions")
            executed = action_reader(step.get("executed_actions"), controlled, "executed_actions")
            native_targets = None
            if native_replay:
                native_targets = step.get("native_targets")
                live = {a for a, v in zip(controlled, before["alive"]) if v}
                if (
                    step.get("action_space") != "native_continuous_state"
                    or not isinstance(native_targets, dict)
                    or set(native_targets) != live
                ):
                    raise ValueError("Native integrator targets must cover live policy actors")
                for target in native_targets.values():
                    _matrix([target], 1, 10, "native target")
            population_metadata = _population_metadata(
                step, population, controlled, policy, executed
            )
            records.append(
                {
                    "generation": generation,
                    "observation_tick": tick - 1,
                    "successor_tick": tick,
                    "duration_seconds": dt,
                    "controlled_ids": controlled,
                    "behavior": {
                        "policy_actions": policy,
                        "executed_actions": executed,
                        "action_space": "native_continuous_state"
                        if native_replay
                        else "jerk_discrete_12",
                        "native_targets": native_targets,
                        "execution_differs_from_policy": {
                            actor: policy[actor] != executed[actor] for actor in controlled
                        },
                        "controller_version": configuration["controller"],
                        "configuration_sha256": config_hash,
                        "policy_input_kind": (
                            "recorded_privileged_teacher"
                            if teacher_inputs
                            else "declared_population_inputs_unrecorded"
                            if population is not None
                            else "privileged_metric_world_rule_controller"
                        ),
                        **population_metadata,
                        "policy_observation_recorded": teacher_inputs is not None,
                        "rgb_is_policy_input": rgb_policy_input,
                    },
                    "masks": {"valid": valid, "terminated": terminated, "truncated": truncated},
                    "events": events,
                    "privileged": {"world_before": before, "world_after": after},
                    "source_step_index": tick - 1,
                }
            )
            before = after
        if native_replay:
            if len(generations) != 1 or len(records) != native_replay["capture_end_tick"]:
                raise ValueError("Native replay capture window is incomplete")
            if not before["done"] and len(records) == native_replay["full_physics_steps"]:
                raise ValueError("Full native replay lacks a terminal successor")
        elif not before["done"]:
            raise ValueError("Completed generation lacks a terminal successor")
        current_frames, seen_ticks = [], set()
        initial_timestamp = None
        for frame in by_frame[generation]:
            if (
                population is not None
                and frame.get("behavior_policies") != population.behavior_policies
            ):
                raise ValueError("Frame behavior policies differ from fixed population")
            tick = _integer(frame.get("tick"), "frame tick")
            if (
                tick > len(records)
                or tick in seen_ticks
                or seen_ticks
                and tick <= max(seen_ticks)
                or _integer(frame.get("render_tick"), "render tick") != tick
            ):
                raise ValueError("Frame ticks must be strictly ordered actual physics ticks")
            seen_ticks.add(tick)
            if not math.isclose(
                _number(frame.get("time_seconds"), "frame time"), tick * dt, abs_tol=1e-9
            ):
                raise ValueError("Frame simulation time does not match tick")
            stamp = _integer(frame.get("simulation_timestamp_us"), "simulation_timestamp_us")
            if initial_timestamp is None:
                initial_timestamp = stamp - round(tick * dt * 1e6)
            if stamp != initial_timestamp + round(tick * dt * 1e6):
                raise ValueError("Frame absolute simulation timestamp is not on the physical clock")
            cameras = frame.get("cameras")
            if (
                not isinstance(cameras, list)
                or len(cameras) != len(camera_names)
                or [camera.get("camera_id") for camera in cameras if isinstance(camera, dict)]
                != camera_names
            ):
                raise ValueError("Frame camera IDs/order must match unique configured cameras")
            normalized = []
            for camera in cameras:
                value = _camera(
                    camera,
                    manifest_path.parent,
                    actor_ids,
                    camera_names,
                    artifacts,
                    rgb_policy_input,
                )
                owner = camera_owners.setdefault(camera["camera_id"], camera["owner_actor_id"])
                if owner != camera["owner_actor_id"]:
                    raise ValueError("Camera owner changed across frames or generations")
                if (
                    camera["source_timestamp_us"] - stamp != camera["source_seed_error_us"]
                    or abs(camera["source_seed_error_us"]) > 60000
                ):
                    raise ValueError("Camera seed offset does not match source/world time")
                normalized.append(value)
            height_binding = _height_binding(
                frame, cameras, controlled, document.get("height_mode")
            )
            quality = validate_frame_quality(frame)
            quality_reports.append(quality)
            current_frames.append(
                {
                    "generation": generation,
                    "tick": tick,
                    "time_seconds": tick * dt,
                    "simulation_timestamp_us": stamp,
                    "cameras": normalized,
                    "used_for_policy": rgb_policy_input,
                    "height_binding": height_binding,
                    "height_metadata_recorded": height_binding is not None,
                    "coverage": frame.get("coverage", {}),
                    "render_quality": quality,
                    "quality_metadata_recorded": "render_quality" in frame,
                }
            )
        if (
            not current_frames
            or current_frames[0]["tick"] != 0
            or current_frames[-1]["tick"] != len(records)
        ):
            raise ValueError("Camera frames must cover initial and terminal worlds")
        camera_intervals = []
        for left, right in zip(current_frames, current_frames[1:]):
            begin, end = left["tick"], right["tick"]
            for camera_name in camera_names:
                camera_intervals.append(
                    {
                        "generation": generation,
                        "camera_id": camera_name,
                        "owner_actor_id": camera_owners[camera_name],
                        "start_tick": begin,
                        "end_tick": end,
                        "duration_seconds": (end - begin) * dt,
                        "input_frame_tick": begin,
                        "target_frame_tick": end,
                        "used_for_policy": rgb_policy_input,
                        "joint_action_sequence": [
                            {
                                "observation_tick": row["observation_tick"],
                                "successor_tick": row["successor_tick"],
                                "executed_actions": row["behavior"]["executed_actions"],
                            }
                            for row in records[begin:end]
                        ],
                    }
                )
        physics[generation], sensors[generation], intervals[generation] = (
            records,
            current_frames,
            camera_intervals,
        )
        summaries.append(
            {
                "generation": generation,
                "physics_transitions": len(records),
                "seconds": len(records) * dt,
                "sensor_frames": len(current_frames),
                "camera_intervals": len(camera_intervals),
            }
        )
    if _hash(manifest_path) != source_hash:
        raise ValueError("Source manifest changed during export validation")
    reasons = [
        "source_marked_non_trainable",
        "display_only_reference_relative_geometry",
        "render_collision_geometry_registration_unverified",
        "rgb_policy_usage_unknown"
        if rgb_policy_input is None
        else "rgb_not_used_by_behavior_policy",
        "population_camera_and_actor_coverage_unverified",
        "pedestrian_and_novel_view_quality_unverified",
    ]
    eligibility = {"dynamics_inspection": {"eligible": True, "reasons": []}}
    for usage in (
        "visual_on_policy_rl",
        "visual_policy_imitation",
        "production_world_model_training",
    ):
        eligibility[usage] = {"eligible": False, "reasons": reasons.copy()}
    result = {
        "schema_version": 1,
        "kind": "mtgs_inspection_episode",
        "version": EPISODE_VERSION,
        "status": "completed",
        "trainable": False,
        "display_only": True,
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": source_hash,
            "artifact_root": str(manifest_path.parent),
        },
        "scenario_id": document.get("scenario_id"),
        "source_log": document.get("source_log"),
        "dt_seconds": dt,
        "action_space": "native_continuous_state" if native_replay else "jerk_discrete_12",
        "actor_ids": actor_ids,
        "controlled_ids": controlled,
        "camera_ids": camera_names,
        "camera_owners": camera_owners,
        "physics_build_id": build,
        "renderer_backend": document.get("backend", {}),
        "binding_mode": document.get("binding_mode"),
        "height_mode": document.get("height_mode"),
        "height_configuration": document.get("height_configuration"),
        "height_profile_unavailable": document.get("height_profile_unavailable"),
        "height_metadata_recorded": document.get("height_mode") is not None,
        "inputs": document.get("inputs", {}),
        "assets": document.get("assets", {}),
        "controller_configuration": configuration,
        "controller_configuration_sha256": config_hash,
        "population_configuration": population.to_dict() if population is not None else None,
        "behavior_policies": population.behavior_policies if population is not None else None,
        "policy_bindings_recorded": population is not None,
        "state_columns": _STATE_COLUMNS,
        "motion_columns": _MOTION_COLUMNS,
        "unavailable_fields": [
            "policy_observations",
            "reward",
            "behavior_log_probability",
            "policy_memory",
            "policy_rng_state",
            "learned_policy_checkpoint",
        ]
        + ([] if population is not None else ["explicit_override_flags"]),
        "permissions": {
            "physics_files": "privileged_truth_not_visual_actor_input",
            "sensor_files": (
                "world_display_observations_policy_usage_unknown"
                if rgb_policy_input is None
                else "world_display_observations_not_behavior_policy_inputs"
            ),
            "source_images": "reference_only_not_counterfactual_observations",
            "reference_futures": "not_exported_to_actor_inputs",
        },
        "eligibility": eligibility,
        "source_limits": document.get("limits", []),
        "render_quality": summarize_quality(quality_reports),
        "summary": {
            "generations": len(generations),
            "physics_transitions": len(steps),
            "joint_simulation_seconds": len(steps) * dt,
            "sensor_frames": len(frames),
            "referenced_artifact_files": len(artifacts),
        },
        "generations": summaries,
        "files": [],
    }
    if teacher_inputs is not None:
        result["teacher_inputs"] = teacher_inputs
        result["unavailable_fields"] = ["reward"]
        result["teacher_artifacts"] = [
            _artifact(manifest_path.parent, name, artifacts)
            for name in ("teacher.npz", "teacher-decisions.jsonl")
        ]
        result["summary"]["referenced_artifact_files"] = len(artifacts)
    result["native_replay"] = native_replay
    result["recording_scope"] = (
        "bounded_native_capture_not_complete_episode"
        if native_replay and native_replay["capture_end_tick"] < native_replay["full_physics_steps"]
        else "complete_episode"
    )
    result["branch_lineage"] = document.get("branch_lineage", [])
    output.mkdir(parents=True)
    for generation in generations:
        for kind, values in (
            ("physics", physics[generation]),
            ("sensors", sensors[generation]),
            ("camera-intervals", intervals[generation]),
        ):
            name = f"generation-{generation:03d}-{kind}.jsonl"
            path = output / name
            with path.open("x", encoding="utf-8") as stream:
                for row in values:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                        + "\n"
                    )
            result["files"].append({"path": name, "sha256": _hash(path), "records": len(values)})
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return result
