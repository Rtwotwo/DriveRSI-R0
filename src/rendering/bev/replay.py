"""Strictly bind recorded transitions to a scene and export a standalone viewer."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from driversi.contracts.context import SceneContext
from driversi.contracts.driving_task import task_identity
from driversi.contracts.road import RoadGraph, Route
from driversi.contracts.scene import Scene
from driversi.data.participants.background import boundary_identity, identity_suffix
from driversi.data.scenes.pufferdrive import write_binary
from driversi.evaluation.metrics.source_support import source_entry_overlaps
from driversi.rendering.bev.template import HTML_TEMPLATE


def _json(value: np.ndarray, name: str):
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"{name} must be a JSON string scalar")

    def invalid(value):
        raise ValueError(f"Non-finite JSON value in {name}: {value}")

    return json.loads(str(value), parse_constant=invalid)


def _scene_key(scene: Scene) -> str:
    # Use the same public binary writer as PufferDriveWorld; no native import.
    with tempfile.TemporaryDirectory(prefix="driversi-replay-identity-") as directory:
        path = Path(directory) / "scene.bin"
        write_binary(scene, path)
        binary = path.read_bytes()
    return hashlib.sha256(binary + boundary_identity(scene) + json.dumps(scene.actor_ids).encode() + identity_suffix(scene) + task_identity(scene)
        + (json.dumps(scene.metadata["native_initial_motion"],sort_keys=True).encode() if "native_initial_motion" in scene.metadata else b"")).hexdigest()


def _array(bundle, name: str, shape: tuple, *, boolean=False, integer=False) -> np.ndarray:
    value = bundle[name]
    if value.shape != shape or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name}: require finite {shape}")
    if boolean and value.dtype.kind != "b":
        raise ValueError(f"{name} must be boolean")
    if integer and value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer")
    return value.copy()


def _ids(bundle, name: str) -> tuple[str, ...]:
    value = bundle[name]
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise ValueError(f"Invalid {name}")
    return tuple(str(item) for item in value)


def _load_run(scene: Scene, path: Path, scene_key: str) -> dict:
    required = {"metadata", "actor_ids", "controlled_ids", "states", "motion", "actions",
                "valid", "terminated", "truncated", "collision", "offroad", "goal",
                "controller_diagnostics"}
    with np.load(path, allow_pickle=False) as bundle:
        if missing := required - set(bundle.files):
            raise ValueError(f"Rollout is missing fields: {', '.join(sorted(missing))}")
        metadata = _json(bundle["metadata"], "metadata")
        if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
                or metadata.get("kind") != "engineering_rollout"):
            raise ValueError("Unsupported rollout metadata schema")
        if metadata.get("scene_key") != scene_key:
            raise ValueError("Rollout scene_key does not match the supplied scene")
        if metadata.get("scenario_id") != scene.metadata["scenario_id"]:
            raise ValueError("Rollout scenario_id does not match scene")
        provenance = metadata.get("scene_metadata", {})
        if not isinstance(provenance, dict) or any(
            provenance.get(key) != scene.metadata.get(key)
            for key in ("source_log", "split", "map_name", "map_version", "origin_xyz")
        ):
            raise ValueError("Rollout source/map/coordinate metadata does not match scene")
        actor_ids = _ids(bundle, "actor_ids")
        controlled_ids = _ids(bundle, "controlled_ids")
        expected_controlled = tuple(actor for actor, active in zip(scene.actor_ids, scene.controlled)
                                    if active)
        if actor_ids != scene.actor_ids or controlled_ids != expected_controlled:
            raise ValueError("Rollout actor identity/order or control mask does not match scene")
        steps = metadata.get("steps")
        dt = metadata.get("dt_seconds")
        if (type(steps) is not int or not 0 <= steps < len(scene.times)
                or not isinstance(dt, (float, int)) or not np.isfinite(dt) or dt <= 0
                or not np.allclose(np.diff(scene.times), dt, atol=1e-9, rtol=0)):
            raise ValueError("Rollout steps/dt do not match the scene time axis")
        native_action = metadata.get("action_space") == "driverl_native_jerk_steering_rate"
        if not native_action and metadata.get("action_space") != "jerk_discrete_12":
            raise ValueError("Unsupported action space; no implicit action conversion")
        n, c = len(actor_ids), len(controlled_ids)
        arrays = {"states": _array(bundle, "states", (steps + 1, n, 13)),
                  "motion": _array(bundle, "motion", (steps + 1, n, 3)),
                  "actions": _array(bundle, "actions", (steps, c, 2) if native_action else (steps, c), integer=not native_action)}
        presence = np.stack([scene.presence_at(i) for i in range(steps+1)])
        if scene.presence is not None and "presence" not in bundle.files:
            raise ValueError("Lifecycle rollout is missing presence")
        dynamic = "traffic_continuity" in scene.metadata
        recorded_presence = _array(bundle,"presence",presence.shape,boolean=True) if "presence" in bundle.files else presence
        if not dynamic and not np.array_equal(recorded_presence, presence):
            raise ValueError("Rollout presence differs from scene")
        presence = recorded_presence
        lifecycle = _json(bundle["lifecycle"], "lifecycle") if "lifecycle" in bundle.files else []
        if "vehicle_fleet" in scene.metadata:
            if len(lifecycle) != steps + 1:
                raise ValueError("Fleet replay requires lifecycle execution records")
            tasks = scene.metadata.get("reactive_background", {}).get("assignments", {})
            if dynamic:
                from driversi.sim.execution.traffic_lifecycle import TrafficLifecycle
                manager = TrafficLifecycle(scene)
                if lifecycle[0].get("continuity") != manager.state or presence[0].tolist() != manager.state["present"]:
                    raise ValueError("Invalid initial traffic membership")
            for tick in range(steps):
                expected = {a for i, a in enumerate(actor_ids)
                            if a in tasks and presence[tick, i] and presence[tick+1, i]}
                controls = lifecycle[tick].get("executed_traffic_controls", [])
                if len(controls) != len(expected) or {d.get("actor") for d in controls} != expected:
                    raise ValueError("Fleet executed controls do not cover active traffic actors")
                if dynamic:
                    rows = np.asarray([[actor_ids.index(d["actor"]), *d["next_state"]] for d in controls], float).reshape(-1,8)
                    derived, entrants = manager.propose(tick, arrays["states"][tick], rows)
                    if derived != lifecycle[tick+1].get("continuity") or derived["present"] != presence[tick+1].tolist():
                        raise ValueError("Recorded admission/departure is not supported by geometry")
                    for row in entrants:
                        if not np.allclose(arrays["states"][tick+1,int(row[0]),:7], row[1:],atol=2e-3,rtol=0):
                            raise ValueError("Admitted state differs from source entry")
                    manager.restore(derived)
                for control in controls:
                    if control.get("observation_tick") != tick or control.get("execution_tick") != tick+1:
                        raise ValueError("Fleet execution clock mismatch")
                    target = np.asarray(control.get("next_state"), float)
                    actual = arrays["states"][tick+1, actor_ids.index(control["actor"]), :7]
                    if target.shape != (7,) or not np.isfinite(target).all():
                        raise ValueError("Invalid executed fleet state")
                    if (not np.allclose(target[[0,1,2,4,5,6]], actual[[0,1,2,4,5,6]], atol=2e-3, rtol=0)
                            or abs(np.arctan2(np.sin(target[3]-actual[3]), np.cos(target[3]-actual[3]))) > 1e-4):
                        raise ValueError("Fleet control differs from actual successor")
            if lifecycle[-1].get("executed_traffic_controls"):
                raise ValueError("Final fleet frame cannot have an executed outgoing action")
        initial = arrays["states"][0]
        yaw_error = np.arctan2(np.sin(initial[:, 3] - scene.poses[0, :, 3]),
                               np.cos(initial[:, 3] - scene.poses[0, :, 3]))
        velocity_matches = np.all(np.isclose(initial[:, 4:7], scene.velocities[0],
                                            atol=2e-3, rtol=0), axis=1)
        # Native stop_entity clears planar velocity only; it preserves source vz.
        stopped_by_initialization = ((initial[:, 9] == 1)
                                     & np.all(initial[:, 4:6] == 0, axis=1)
                                     & np.isclose(initial[:, 6], scene.velocities[0, :, 2],
                                                  atol=2e-3, rtol=0))
        if (not np.allclose(initial[:, :3], scene.poses[0, :, :3], atol=2e-3, rtol=0)
                or not np.all(velocity_matches | stopped_by_initialization)
                or np.any(abs(yaw_error) > 1e-4)):
            raise ValueError("Rollout initial state does not match the scene initial state")
        if steps == 0 and not np.all(initial[scene.controlled, 9] == 1):
            raise ValueError("A zero-step rollout requires all controlled actors to be initially stopped")
        for name in ("valid", "terminated", "truncated", "collision", "offroad", "goal"):
            arrays[name] = _array(bundle, name, (steps, c), boolean=True)
        boundary_profile = scene.metadata.get("road_boundary_policy")
        if boundary_profile and "road_boundary_contact" not in bundle.files:
            raise ValueError("Boundary profile rollout is missing perimeter contact evidence")
        if "road_boundary_contact" in bundle.files:
            arrays["road_boundary_contact"] = _array(bundle, "road_boundary_contact", (steps, c), boolean=True)
            if np.any(arrays["road_boundary_contact"] & ~arrays["valid"]):
                raise ValueError("Boundary contact contradicts valid transitions")
        support = np.zeros((steps, c), dtype=bool)
        if "driving_tasks" in scene.metadata:
            support = _array(bundle, "route_support_end", (steps, c), boolean=True)
            allowed = np.asarray([scene.metadata["driving_tasks"]["actors"][a]["endpoint"] == "support_boundary" for a in controlled_ids])
            if np.any(support & ~arrays["truncated"]) or np.any(support & ~allowed):
                raise ValueError("Invalid route support truncation")
            arrays["route_support_end"] = support
        if not native_action and np.any((arrays["actions"] < 0) | (arrays["actions"] >= 12)):
            raise ValueError("Action index must be in [0, 11]")
        valid, terminated, truncated = (arrays[name] for name in ("valid", "terminated", "truncated"))
        if np.any(terminated & truncated) or any(
            np.any(arrays[name] & ~valid)
            for name in ("terminated", "truncated", "collision", "offroad", "goal")
        ):
            raise ValueError("Rollout events/terminal masks contradict valid transitions")
        ended = np.maximum.accumulate(terminated | truncated, axis=0)
        if np.any(valid[1:] & ended[:-1]):
            raise ValueError("Actor becomes active after its episode has ended")
        flags = arrays["states"][:, scene.controlled, 9:13]
        if not np.isin(flags, (0, 1)).all():
            raise ValueError("State event/stopped flags must be binary")
        if not np.array_equal(valid, flags[:-1, :, 0] == 0):
            raise ValueError("Valid masks contradict current state stopped flags")
        if not np.array_equal(terminated | support, valid & (flags[1:, :, 0] == 1)):
            raise ValueError("Termination contradicts successor stopped flags")
        for column, name in enumerate(("collision", "offroad", "goal"), 1):
            if not np.array_equal(arrays[name], valid & (flags[1:, :, column] == 1)):
                raise ValueError(f"{name} event contradicts successor state flags")
        diagnostics = _json(bundle["controller_diagnostics"], "controller_diagnostics")

    configuration = metadata.get("controller_configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("controller_configuration must be an object")
    routes = configuration.get("routes", {})
    if not isinstance(routes, dict) or not set(routes) <= set(controlled_ids):
        raise ValueError("Routes refer to unknown/uncontrolled actors")
    parsed_routes = {}
    for actor, route in routes.items():
        if not isinstance(route, dict):
            raise ValueError("Malformed route configuration")
        parsed_routes[actor] = Route(
            route["points"], tuple(route.get("node_ids", ())),
            route.get("end_reason", "unspecified"), route.get("speed_limit_mps"),
        )
    if not isinstance(diagnostics, list) or len(diagnostics) not in (0, steps):
        raise ValueError("Diagnostics must be empty or contain one entry per transition")
    diagnostic_fields = (
        "desired_acceleration", "desired_lateral_acceleration", "bumper_gap_m", "ttc_seconds",
        "route_progress_m", "route_distance_m", "remaining_m",
        "actual_speed_mps", "path_speed_mps", "target_speed_mps", "heading_error_radians",
        "preview_curvature_per_m", "curve_speed_limit_mps",
        "configured_desired_speed_mps", "configured_headway_seconds",
    )
    for t, commands in enumerate(diagnostics):
        if not isinstance(commands, list) or len(commands) != c:
            raise ValueError("Diagnostics must contain every controlled actor in control order")
        for i, command in enumerate(commands):
            if not isinstance(command, dict) or command.get("actor") != controlled_ids[i]:
                raise ValueError("Diagnostic actor identity/order mismatch")
            expected_action = arrays["actions"][t, i].tolist() if native_action else int(arrays["actions"][t, i])
            if command.get("action") != expected_action:
                raise ValueError("Diagnostic action contradicts the recorded executed action")
            for name in diagnostic_fields:
                value = command.get(name)
                if value is not None and (
                    not isinstance(value, (int, float)) or not np.isfinite(value)
                ):
                    raise ValueError(f"Invalid diagnostic {name}")
            if command.get("lead_id") is not None and command["lead_id"] not in actor_ids:
                raise ValueError("Diagnostic lead_id is not present in the scene")
            for name in ("controller", "intervention"):
                if name in command and (not isinstance(command[name], str) or not command[name]):
                    raise ValueError(f"Diagnostic {name} must be a nonempty string")

    # State t is paired with action t only if that transition is still valid.
    # Final states have no outgoing action, even when the final action was nonzero.
    frames = []
    last_controls = {}
    overrides = configuration.get("probe_overrides", {})
    protocol = metadata.get("protocol", {})
    if not isinstance(overrides, dict) or not isinstance(protocol, dict):
        raise ValueError("Probe controller overrides/protocol must be objects")
    intervention_time = protocol.get("intervention_seconds")
    override_actors = [overrides.get(key) for key in ("action_replay_actor", "braking_actor")]
    if any(actor is not None and actor not in controlled_ids for actor in override_actors):
        raise ValueError("Probe overrides must refer to controlled actors")
    if any(actor is not None for actor in override_actors) and (
        isinstance(intervention_time, bool) or not isinstance(intervention_time, (int, float))
        or not np.isfinite(intervention_time) or intervention_time < 0
    ):
        raise ValueError("Probe overrides require a finite nonnegative intervention time")
    default_controller = configuration.get("controller") or metadata.get("policy", "unspecified")

    def control_provenance(actor, command, tick):
        after_intervention = (isinstance(intervention_time, (float, int))
                              and tick * dt + 1e-9 >= intervention_time)
        fallback = default_controller
        if after_intervention and overrides.get("action_replay_actor") == actor:
            fallback = "baseline_action_replay"
        controller = command.get("controller") or fallback
        intervention = command.get("intervention")
        if (not intervention and after_intervention
                and overrides.get("braking_actor") == actor):
            intervention = "longitudinal_brake_target"
        lateral = configuration.get("lateral_intervention", {})
        if lateral.get("applied") and lateral.get("actor") == actor and tick >= lateral.get("tick", steps+1):
            intervention = "declared_lateral_route_intent"
        return {"controller": controller, "phase": "outgoing", "command_tick": tick,
                "intervention": intervention,
                "diagnostic_origin": ("unavailable" if not command else "baseline_reference"
                                      if controller == "baseline_action_replay" else "current_world")}

    for t in range(steps + 1):
        outgoing, arrivals, control_sources = {}, {}, {}
        for i, actor in enumerate(controlled_ids):
            if t < steps and valid[t, i]:
                command = diagnostics[t][i] if diagnostics else {}
                provenance = control_provenance(actor, command, t)
                outgoing[actor] = {"action": arrays["actions"][t, i].tolist() if native_action else int(arrays["actions"][t, i]), "diagnostics": command,
                                   "control_source": provenance}
                control_sources[actor] = provenance
                last_controls[actor] = provenance
            elif actor in last_controls:
                control_sources[actor] = {**last_controls[actor], "phase": "last_executed"}
            else:
                control_sources[actor] = {"controller": default_controller,
                                         "phase": "configured_without_action", "command_tick": None,
                                         "intervention": None, "diagnostic_origin": "unavailable"}
            if t > 0:
                events = [name for name in ("collision", "offroad", "goal", "terminated", "truncated")
                          if arrays[name][t - 1, i]]
                if "road_boundary_contact" in arrays and arrays["road_boundary_contact"][t - 1, i]:
                    events.append("road_boundary_contact")
                if support[t - 1, i]:
                    events = [name for name in events if name != "truncated"] + ["route_support_end"]
                if events:
                    arrivals[actor] = events
        frames.append({"outgoing": outgoing, "arrivals": arrivals, "control_sources": control_sources})
    route_distance = {}
    for actor, route in parsed_routes.items():
        i = actor_ids.index(actor)
        route_distance[actor] = [route.project(xy)[1] for xy in arrays["states"][:, i, :2]]
    return {"name": path.name, "metadata": metadata, "states": arrays["states"].tolist(),
            "motion": arrays["motion"].tolist(), "frames": frames,
            "presence": presence.tolist(), "lifecycle": lifecycle,
            "routes": {actor: route.to_dict() for actor, route in parsed_routes.items()},
            "route_distance": route_distance}


def export_replay(scene: Scene, rollout_path: Path, output: Path, *,
                  road_graph: RoadGraph | None = None,
                  compare_path: Path | None = None,
                  scene_context: SceneContext | None = None,
                  inspection: dict | None = None, compress_payload: bool = False) -> dict:
    """Export one strictly matched scene/run pair, optionally with an aligned run.

    Compared runs must use the same scene, controlled actors, dt and frame count.
    No padding or resampling is performed. The HTML is offline and self-contained.
    """
    scene.validate()
    if road_graph is not None:
        road_graph.validate_scene(scene)
    if scene_context is not None:
        scene_context.validate_scene(scene)
    key = _scene_key(scene)
    run = _load_run(scene, Path(rollout_path), key)
    comparison = _load_run(scene, Path(compare_path), key) if compare_path is not None else None
    if comparison is not None and len(comparison["states"]) != len(run["states"]):
        raise ValueError("Comparison time domains differ; padding or alignment is not implicit")
    roads = [{"type": int(kind),
              "points": scene.road_points[scene.road_offsets[i]:scene.road_offsets[i + 1], :2].tolist()}
             for i, kind in enumerate(scene.road_types)]
    steps = len(run["states"]) - 1
    context = scene_context.to_dict() if scene_context is not None else None
    context_indices = [None] * (steps + 1)
    source_only_ids = set()
    if context is not None:
        source_times = np.asarray([frame["time_seconds"] for frame in context["frames"]])
        for frame in context["frames"]:
            source_only_ids.update(obj["id"] for obj in frame["objects"]
                                   if obj["id"] not in scene.actor_ids)
        for i, time in enumerate(scene.times[:steps + 1]):
            # Never borrow a future annotation or extrapolate beyond source coverage.
            index = int(np.searchsorted(source_times, time + 1e-9, side="right")) - 1
            if (index >= 0 and time <= source_times[-1] + 1e-9
                    and time - source_times[index]
                    <= context["metadata"]["max_sample_age_seconds"] + 1e-9):
                context_indices[i] = index
    entity_roles = {
        "controlled": [actor for actor, active in zip(scene.actor_ids, scene.controlled) if active],
        "world_replay": [actor for actor, active in zip(scene.actor_ids, scene.controlled)
                         if not active and actor not in scene.metadata.get("reactive_background", {}).get("assignments", {})],
        "source_context_only": sorted(source_only_ids),
        "map_features": [feature["id"] for feature in context["map_features"]] if context else [],
        "source_context_is_counterfactual_ground_truth": False,
    }
    if scene.metadata.get("reactive_background"):
        entity_roles["traffic_feedback"] = list(scene.metadata["reactive_background"]["assignments"])
    if scene.metadata.get("vehicle_fleet"):
        entity_roles["vehicle_fleet"] = scene.metadata["vehicle_fleet"]
    source_timing = scene.metadata.get("source_timing")
    if source_timing is None and context is not None and len(context["frames"]) > 1:
        intervals = np.diff([frame["time_seconds"] for frame in context["frames"]])
        source_timing = {
            "sample_count": len(context["frames"]),
            "interval_seconds": {"min": float(intervals.min()),
                                 "median": float(np.median(intervals)),
                                 "max": float(intervals.max())},
        }
    data = {"schema_version": 1, "display_mode": "recorded_simulation",
            "source_integrity": source_entry_overlaps(scene), "scene": {"metadata": scene.metadata,
            "actor_ids": scene.actor_ids, "controlled": scene.controlled.tolist(),
            "dimensions": scene.dimensions.tolist(), "kinds": scene.kinds.tolist(),
            "times": scene.times[:steps + 1].tolist(), "roads": roads,
            "reference_poses": scene.poses[:steps + 1].tolist(),
            "reference_planar_speeds_mps": np.linalg.norm(
                scene.velocities[:steps + 1, :, :2], axis=2).tolist(),
            "source_timing": source_timing},
            "run": run, "comparison": comparison,
            "context": context, "context_frame_indices": context_indices,
            "entity_roles": entity_roles,
            "graph": road_graph.nodes if road_graph else None, "inspection": inspection}
    output = Path(output)
    inputs = [Path(rollout_path), *([Path(compare_path)] if compare_path is not None else [])]
    if any(output.resolve() == path.resolve() for path in inputs):
        raise ValueError("Replay output must not overwrite an input")
    write_viewer(data, output, compress_payload=compress_payload)
    return {"kind": "offline_replay", "schema_version": 1, "path": str(output),
            "scenario_id": scene.metadata["scenario_id"], "frames": steps + 1,
            "actors": len(scene.actor_ids), "comparison": comparison is not None,
            "source_context_only_objects": len(source_only_ids),
            "map_features": len(entity_roles["map_features"])}


def write_viewer(data: dict, output: Path, *, compress_payload: bool = False) -> None:
    """Write the common inspector; source-only payloads are explicitly labelled.

    Large raw nuPlan history remains in the linked Scene archive. The HTML keeps
    the displayed world, identities and source clocks, not a second raw DB dump.
    """
    import copy

    data = copy.deepcopy(data)
    metadata = data["scene"]["metadata"]
    source = metadata.get("nuplan_source")
    if source:
        history = source.pop("history", [])
        source["history_frame_count"] = len(history)
        source["history_payload_location"] = "original Scene archive; omitted from display payload"
        for key in ("run", "comparison"):
            if data.get(key):
                data[key]["metadata"].pop("scene_metadata", None)
    payload = (json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
               .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    output = Path(output)
    if output.suffix.lower() != ".html":
        raise ValueError("Replay output must have an .html extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    template = HTML_TEMPLATE
    if compress_payload:
        import base64
        import gzip
        payload = base64.b64encode(gzip.compress(payload.encode("utf-8"), mtime=0)).decode("ascii")
        template = template.replace('<script id="replay-data" type="application/json">',
                                    '<script id="replay-data" type="application/octet-stream">')
        template = template.replace('<script>', '<script type="module">')
        template = template.replace(
            'const DATA = JSON.parse(document.getElementById("replay-data").textContent);',
            '\n'.join([
                'const payloadText = document.getElementById("replay-data").textContent;',
                'const loading = document.createElement("div"); loading.textContent="正在解压完整连续回放，全部对象与事件均保留…"; loading.style="padding:16px;background:#fff7e5"; document.body.prepend(loading);',
                'let DATA;',
                'try {',
                'const bytes = Uint8Array.from(atob(payloadText), c => c.charCodeAt(0));',
                'const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));',
                'DATA = JSON.parse(await new Response(stream).text()); loading.remove();',
                '} catch(error) { loading.textContent="回放解压失败："+error.message; throw error; }',
            ]))
    output.write_text(template.replace("__REPLAY_JSON__", payload), encoding="utf-8")


def export_source_replay(scene: Scene, output: Path, *, inspection: dict | None = None) -> dict:
    """View compiled source positions; do not invent actions or collision results."""
    scene.validate()
    if scene.metadata.get("reactive_background") or scene.metadata.get("vehicle_fleet"):
        raise ValueError("Source replay requires the unmodified source scene, not a configured fleet")
    if scene.metadata.get("source_dataset") != "nuplan_db":
        raise ValueError("Source inspector currently requires a compiled nuPlan scene")
    times, count = len(scene.times), len(scene.actor_ids)
    states = np.zeros((times, count, 13))
    states[:, :, :4] = scene.poses
    states[:, :, 4:7] = scene.velocities
    presence = np.stack([scene.presence_at(i) for i in range(times)])
    metadata = dict(scene.metadata)
    metadata.pop("reactive_background", None)
    metadata.pop("vehicle_fleet", None)
    data = {
        "schema_version": 1, "display_mode": "source_reference", "inspection": inspection,
        "source_integrity": source_entry_overlaps(scene),
        "scene": {"metadata": metadata, "actor_ids": scene.actor_ids,
            "controlled": [False] * count, "dimensions": scene.dimensions.tolist(),
            "kinds": scene.kinds.tolist(), "times": scene.times.tolist(),
            "roads": [{"type": int(kind), "points": scene.road_points[scene.road_offsets[i]:scene.road_offsets[i+1], :2].tolist()}
                      for i, kind in enumerate(scene.road_types)],
            "reference_poses": scene.poses.tolist(),
            "reference_planar_speeds_mps": np.linalg.norm(scene.velocities[:, :, :2], axis=2).tolist(),
            "source_timing": scene.metadata.get("source_timing")},
        "run": {"name": "compiled-source-reference", "metadata": {"kind": "source_reference",
            "policy": "none", "controller_configuration": {}, "controlled_actors": 0,
            "dt_seconds": float(scene.times[1]-scene.times[0])},
            "states": states.tolist(), "motion": np.zeros((times,count,3)).tolist(),
            "presence": presence.tolist(), "lifecycle": [], "routes": {}, "route_distance": {},
            "frames": [{"outgoing": {}, "arrivals": {}, "control_sources": {}} for _ in range(times)]},
        "comparison": None, "context": None, "context_frame_indices": [None]*times,
        "entity_roles": {"controlled": [], "world_replay": list(scene.actor_ids)}, "graph": None}
    write_viewer(data, output)
    return {"kind": "compiled_source_reference", "frames": times, "actors": count,
            "collision_assessed": False, "simulation": False}
