import hashlib
import json
import re

import numpy as np
import pytest

from driversi.contracts.context import SceneContext, scene_fingerprint
from driversi.data.scenes.pufferdrive import write_binary
from driversi.rendering import export_replay


def record(scene, path, *, steps=3, mutations=None, metadata_changes=None):
    binary = path.with_suffix(".bin")
    write_binary(scene, binary)
    key = hashlib.sha256(binary.read_bytes() + json.dumps(scene.actor_ids).encode()).hexdigest()
    binary.unlink()
    metadata = {"schema_version": 1, "kind": "engineering_rollout", "scene_key": key,
                "scenario_id": scene.metadata["scenario_id"], "scene_metadata": scene.metadata,
                "steps": steps, "dt_seconds": .1, "action_space": "jerk_discrete_12",
                "policy": "coast", "controller_configuration": {"controller": "coast"}}
    metadata.update(metadata_changes or {})
    actors, controlled = len(scene.actor_ids), int(scene.controlled.sum())
    states = np.zeros((steps + 1, actors, 13))
    states[:, :, :4] = scene.poses[:steps + 1]
    states[:, :, 4:7] = scene.velocities[:steps + 1]
    arrays = {"metadata": json.dumps(metadata), "actor_ids": np.asarray(scene.actor_ids),
              "controlled_ids": np.asarray(scene.actor_ids)[scene.controlled],
              "states": states, "motion": np.zeros((steps + 1, actors, 3)),
              "actions": np.full((steps, controlled), 7, dtype=np.int32),
              "valid": np.ones((steps, controlled), dtype=bool),
              "controller_diagnostics": "[]"}
    arrays.update({name: np.zeros((steps, controlled), dtype=bool)
                   for name in ("terminated", "truncated", "collision", "offroad", "goal")})
    if mutations:
        mutations(arrays)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def payload(path):
    html = path.read_text(encoding="utf-8")
    return json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',
                                html, re.S).group(1))


def test_standalone_export_escapes_untrusted_scene_text(straight_scene, tmp_path):
    scene = straight_scene
    attack = '</script><img src="https://invalid.example/" onerror="alert(1)">'
    scene.actor_ids = (attack, *scene.actor_ids[1:])
    source = record(scene, tmp_path / "run.npz")
    output = tmp_path / "viewer.html"
    result = export_replay(scene, source, output)
    html = output.read_text(encoding="utf-8")
    assert result["frames"] == 4
    assert payload(output)["scene"]["actor_ids"][0] == attack
    assert attack not in html
    assert not re.search(r'<(?:script|link|img)\b[^>]*(?:src|href)\s*=', html)
    assert "fetch(" not in html and "XMLHttpRequest" not in html


def test_actions_and_events_belong_to_different_sides_of_transition(straight_scene, tmp_path):
    def stop_first_actor(arrays):
        arrays["actions"][0, 0] = 1
        arrays["valid"][1:, 0] = False
        arrays["terminated"][0, 0] = True
        arrays["collision"][0, 0] = True
        arrays["states"][1:, 0, 9:11] = 1

    source = record(straight_scene, tmp_path / "run.npz", mutations=stop_first_actor)
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output)
    frames = payload(output)["run"]["frames"]
    assert frames[0]["outgoing"]["a"]["action"] == 1
    assert frames[0]["arrivals"] == {}
    assert frames[1]["arrivals"]["a"] == ["collision", "terminated"]
    assert "a" not in frames[1]["outgoing"]
    assert "b" in frames[1]["outgoing"]
    assert frames[-1]["outgoing"] == {}


def test_scene_identity_cannot_be_replaced_by_same_scenario_name(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz")
    straight_scene.poses[1, 0, 0] += 1
    with pytest.raises(ValueError, match="scene_key"):
        export_replay(straight_scene, source, tmp_path / "viewer.html")


def test_reference_speed_and_source_clock_do_not_follow_controlled_rollout(straight_scene, tmp_path):
    straight_scene.metadata["source_timing"] = {
        "sample_count": 7, "interval_seconds": {"min": .499, "median": .5, "max": .501}}
    straight_scene.metadata["heading_audit"] = {
        "schema_version": 1, "corrections": [{"actor_id": "replay", "start_time_seconds": .5,
                                              "end_time_seconds": 1.}]}
    def accelerate(arrays):
        arrays["states"][1:, 0, 4] = 8.
    source = record(straight_scene, tmp_path / "run.npz", mutations=accelerate)
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output)
    data = payload(output)
    assert data["scene"]["source_timing"]["interval_seconds"]["median"] == .5
    assert data["run"]["metadata"]["dt_seconds"] == .1
    assert data["run"]["states"][1][0][4] == 8.
    assert data["scene"]["reference_planar_speeds_mps"][1][0] == 4.
    assert data["scene"]["metadata"]["heading_audit"]["corrections"][0]["actor_id"] == "replay"


def test_unknown_source_rate_stays_unknown_without_context(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz")
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output)
    assert payload(output)["scene"]["source_timing"] is None


@pytest.mark.parametrize("mutation,error", [
    (lambda a: a["states"].__setitem__((1, 0, 0), np.nan), "states"),
    (lambda a: a["states"].__setitem__((0, 0, 0), 20), "initial state"),
    (lambda a: a.__setitem__("actor_ids", np.asarray(["b", "a", "replay"])), "identity"),
    (lambda a: a.__setitem__("valid", a["valid"].astype(float)), "boolean"),
    (lambda a: a["actions"].__setitem__((0, 0), 12), "Action index"),
    (lambda a: a["states"].__setitem__((1, 0, 9), 1), "Valid masks"),
    (lambda a: a["states"].__setitem__((1, 0, 10), 1), "collision event"),
    (lambda a: a["states"].__setitem__((1, 0, 11), 0.5), "binary"),
    (lambda a: a.__setitem__("controller_diagnostics", "[NaN]"), "Non-finite"),
])
def test_invalid_recordings_are_rejected(straight_scene, tmp_path, mutation, error):
    source = record(straight_scene, tmp_path / "run.npz", mutations=mutation)
    output = tmp_path / "viewer.html"
    with pytest.raises(ValueError, match=error):
        export_replay(straight_scene, source, output)
    assert not output.exists()


def test_comparison_requires_exact_time_domain(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "main.npz")
    other = record(straight_scene, tmp_path / "other.npz", steps=2)
    with pytest.raises(ValueError, match="time domains"):
        export_replay(straight_scene, source, tmp_path / "viewer.html", compare_path=other)
    record(straight_scene, other, metadata_changes={"policy": "brake"})
    export_replay(straight_scene, source, tmp_path / "viewer.html", compare_path=other)
    assert payload(tmp_path / "viewer.html")["comparison"]["metadata"]["policy"] == "brake"


def test_stale_action_after_termination_is_rejected(straight_scene, tmp_path):
    def invalid_masks(arrays):
        arrays["terminated"][0, 0] = True

    source = record(straight_scene, tmp_path / "run.npz", mutations=invalid_masks)
    with pytest.raises(ValueError, match="active after"):
        export_replay(straight_scene, source, tmp_path / "viewer.html")


def test_diagnostic_action_must_equal_recorded_action(straight_scene, tmp_path):
    def contradictory(arrays):
        commands = [[{"actor": actor, "action": 1} for actor in ("a", "b")] for _ in range(3)]
        arrays["controller_diagnostics"] = json.dumps(commands)

    source = record(straight_scene, tmp_path / "run.npz", mutations=contradictory)
    with pytest.raises(ValueError, match="Diagnostic action"):
        export_replay(straight_scene, source, tmp_path / "viewer.html")


def test_native_initial_stop_preserves_inspection_without_padding(straight_scene, tmp_path):
    def stop_at_initialization(arrays):
        arrays["states"][0, :2, 9] = 1
        arrays["states"][0, :2, 4:7] = 0

    source = record(straight_scene, tmp_path / "stopped.npz", steps=0, mutations=stop_at_initialization)
    output = tmp_path / "viewer.html"
    result = export_replay(straight_scene, source, output)
    data = payload(output)
    assert result["frames"] == 1
    assert data["scene"]["times"] == [0.0]
    assert data["run"]["states"][0][0][4:7] == [0.0, 0.0, 0.0]
    assert data["run"]["frames"][0]["outgoing"] == {}
    assert data["run"]["frames"][0]["arrivals"] == {}
    assert data["run"]["frames"][0]["control_sources"]["a"]["phase"] == "configured_without_action"


@pytest.mark.parametrize("explicit_command_source", [False, True])
def test_probe_controller_and_reference_diagnostics_follow_each_frame(
        straight_scene, tmp_path, explicit_command_source):
    def probe_commands(arrays):
        commands = [[{"actor": actor, "action": 7, "desired_acceleration": .4, "lead_id": "b"}
                     for actor in ("a", "b")] for _ in range(3)]
        if explicit_command_source:
            for tick in (1, 2):
                commands[tick][0]["controller"] = "baseline_action_replay"
                commands[tick][1]["intervention"] = "longitudinal_brake_target"
        arrays["controller_diagnostics"] = json.dumps(commands)

    source = record(straight_scene, tmp_path / "run.npz", mutations=probe_commands,
                    metadata_changes={
                        "controller_configuration": {
                            "controller": "route-idm-v2",
                            "probe_overrides": {"action_replay_actor": "a", "braking_actor": "b"}},
                        "protocol": {"intervention_seconds": .1}})
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output)
    frames = payload(output)["run"]["frames"]
    prefix = frames[0]["control_sources"]["a"]
    assert prefix["controller"] == "route-idm-v2"
    assert prefix["diagnostic_origin"] == "current_world"
    assert prefix["phase"] == "outgoing"
    assert frames[0]["control_sources"]["b"]["intervention"] is None
    replay = frames[1]["outgoing"]["a"]
    assert replay["action"] == 7
    assert replay["diagnostics"]["desired_acceleration"] == .4
    assert replay["control_source"]["controller"] == "baseline_action_replay"
    assert replay["control_source"]["diagnostic_origin"] == "baseline_reference"
    assert frames[1]["control_sources"]["b"]["intervention"] == "longitudinal_brake_target"
    final = frames[-1]["control_sources"]["a"]
    assert frames[-1]["outgoing"] == {}
    assert final["phase"] == "last_executed"
    assert final["command_tick"] == 2
    assert final["controller"] == "baseline_action_replay"
    assert final["diagnostic_origin"] == "baseline_reference"


def test_initial_velocity_difference_requires_explicit_backend_stop(straight_scene, tmp_path):
    def erase_initial_velocity(arrays):
        arrays["states"][0, 0, 4:7] = 0

    source = record(straight_scene, tmp_path / "run.npz", mutations=erase_initial_velocity)
    with pytest.raises(ValueError, match="initial state"):
        export_replay(straight_scene, source, tmp_path / "viewer.html")


@pytest.mark.parametrize("change_vz", [False, True])
def test_initial_planar_stop_preserves_source_vertical_velocity(straight_scene, tmp_path, change_vz):
    straight_scene.velocities[:, :2, 2] = -.46
    def initial_stop(arrays):
        arrays["states"][0, :2, 9] = 1
        arrays["states"][0, :2, 11] = 1
        arrays["states"][0, :2, 4:6] = 0
        if change_vz:
            arrays["states"][0, 0, 6] = 0
    source = record(straight_scene, tmp_path / "stopped.npz", steps=0, mutations=initial_stop)
    output = tmp_path / "viewer.html"
    if change_vz:
        with pytest.raises(ValueError, match="initial state"):
            export_replay(straight_scene, source, output)
        assert not output.exists()
    else:
        export_replay(straight_scene, source, output)
        state = payload(output)["run"]["states"][0][0]
        assert state[4:7] == [0., 0., -.46]
        assert state[9:12] == [1., 0., 1.]


def source_context(scene):
    """Sparse source annotations deliberately differ from simulated population."""
    metadata = {key: scene.metadata.get(key) for key in (
        "scenario_id", "source_log", "map_name", "map_version", "origin_xyz")}
    metadata.update(schema_version=1, scene_fingerprint=scene_fingerprint(scene),
                    scene_times=scene.times.tolist(), reference_policy="previous_sample",
                    max_sample_age_seconds=.75)
    pedestrian = {"id": "walker", "kind": "pedestrian", "pose": [1, 2, 0, .3],
                  "dimensions": [.6, .5, 1.7]}
    duplicate = {"id": "a", "kind": "vehicle", "pose": [99, 2, 0, .3],
                 "dimensions": [4, 2, 1.5]}
    frames = [
        {"time_seconds": 0., "timestamp_us": 1000000, "objects": [pedestrian, duplicate],
         "signals": {"available": False, "records": []}},
        {"time_seconds": .2, "timestamp_us": 1200000, "objects": [],
         "signals": {"available": True, "records": []}},
        {"time_seconds": 1.2, "timestamp_us": 2200000, "objects": [pedestrian],
         "signals": {"available": True, "records": [{"connector_id": "42", "is_red": False}]}},
        {"time_seconds": 3., "timestamp_us": 4000000, "objects": [],
         "signals": {"available": False, "records": []}},
    ]
    return SceneContext(metadata, frames, [], {
        "traffic_signs": {"availability": "unavailable", "feature_count": 0},
        "crosswalks": {"availability": "loaded_empty", "feature_count": 0}})


def test_context_alignment_never_borrows_future_or_stale_source_frames(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz", steps=30)
    context = source_context(straight_scene)
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output, scene_context=context)
    data = payload(output)
    indices = data["context_frame_indices"]
    assert indices[:3] == [0, 0, 1]
    assert data["context"]["frames"][indices[2]]["objects"] == []
    assert indices[9:13] == [1, None, None, 2]
    assert indices[20] is None
    assert indices[30] == 3
    assert data["context"]["frames"][0]["signals"]["available"] is False
    assert data["context"]["frames"][1]["signals"] == {"available": True, "records": []}
    assert data["context"]["frames"][2]["signals"]["records"][0]["is_red"] is False


def test_context_roles_exclude_world_ids_without_mutating_recorded_states(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz")
    context = source_context(straight_scene)
    output = tmp_path / "viewer.html"
    result = export_replay(straight_scene, source, output, scene_context=context)
    data = payload(output)
    assert data["entity_roles"] == {
        "controlled": ["a", "b"], "world_replay": ["replay"],
        "source_context_only": ["walker"], "map_features": [],
        "source_context_is_counterfactual_ground_truth": False}
    assert result["source_context_only_objects"] == 1
    assert len(data["run"]["states"][0]) == 3
    assert data["run"]["states"][0][0][0] == 0
    assert data["context"]["frames"][0]["objects"][1]["pose"][0] == 99


@pytest.mark.parametrize("field,value", [("source_log", "other-log"),
                                         ("origin_xyz", [10, 0, 0]),
                                         ("scene_fingerprint", "0" * 64)])
def test_context_binding_rejects_wrong_scene_provenance(straight_scene, tmp_path, field, value):
    source = record(straight_scene, tmp_path / "run.npz")
    context = source_context(straight_scene)
    context.metadata[field] = value
    output = tmp_path / "viewer.html"
    with pytest.raises(ValueError, match="Context"):
        export_replay(straight_scene, source, output, scene_context=context)
    assert not output.exists()


def test_unknown_context_text_survives_export_without_executable_html(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz")
    context = source_context(straight_scene)
    attack = '</script><img src="https://invalid.example/" onerror="alert(1)">'
    context.frames[0]["objects"][0].update(id=attack, kind=attack)
    context.map_features = [{"id": "sign:" + attack, "layer": "traffic_signs",
                             "geometry": {"type": "Point", "coordinates": [1, 2]},
                             "properties": {"text": attack}}]
    context.map_layers["traffic_signs"] = {"availability": "available", "feature_count": 1}
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output, scene_context=context)
    data = payload(output)
    assert data["context"]["frames"][0]["objects"][0]["kind"] == attack
    assert data["context"]["map_features"][0]["properties"]["text"] == attack
    html = output.read_text(encoding="utf-8")
    assert attack not in html
    assert "innerHTML" not in html
    assert not re.search(r'<(?:script|link|img)\b[^>]*(?:src|href)\s*=', html)


def test_legacy_replay_explicitly_reports_missing_context(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz")
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output)
    data = payload(output)
    assert data["context"] is None
    assert data["context_frame_indices"] == [None] * 4
    assert data["entity_roles"]["source_context_only"] == []


def test_lateral_provenance_only_changes_after_intent_tick(straight_scene, tmp_path):
    source = record(straight_scene, tmp_path / "run.npz", metadata_changes={
        "controller_configuration": {
            "controller": "route-idm-v2",
            "lateral_intervention": {"actor": "a", "tick": 1, "applied": True},
        }
    })
    output = tmp_path / "viewer.html"
    export_replay(straight_scene, source, output)
    frames = payload(output)["run"]["frames"]
    assert frames[0]["control_sources"]["a"]["intervention"] is None
    assert frames[1]["control_sources"]["a"]["intervention"] == "declared_lateral_route_intent"
    assert frames[1]["control_sources"]["b"]["intervention"] is None
    assert frames[-1]["control_sources"]["a"]["phase"] == "last_executed"
    assert frames[-1]["control_sources"]["a"]["intervention"] == "declared_lateral_route_intent"
