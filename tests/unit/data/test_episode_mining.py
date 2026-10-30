"""Candidate extraction preserves sequence boundaries and cannot authorize training."""
import hashlib
import json
from pathlib import Path

import pytest

from driversi.data.scenes.episode_mining import mine_mtgs_episode
from driversi.data.assets.mtgs_episode import export_mtgs_episode


def _world(tick, generation):
    state = [[float(tick), float(i), 0., 0., 2., 0., 0., 10., 0., 0., 0., 0., 0.] for i in range(2)]
    motion = [[0., 0., 0.], [-4. if generation and tick in (2, 3, 4) else 0., 0., 0.]]
    if tick == 4:
        state[1][4] = 0.
    return {"tick": tick, "all_actor_ids": ["ego", "car"], "controlled_ids": ["ego", "car"],
            "state": state, "motion": motion, "alive": [True, True], "done": tick == 4,
            "build_id": "fixture", "dt_seconds": .1}


@pytest.fixture
def episode(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"camera.png").write_bytes(b"immutable image reference")
    doc = {"schema_version": 1, "kind": "mtgs_world_session", "status": "completed",
           "trainable": False, "display_only": True, "dt_seconds": .1,
           "controlled_ids": ["ego", "car"], "camera_ids": ["front"], "physics_build_id": "fixture",
           "protocol": {"rgb_is_policy_input": False, "kind": "fixture"},
           "controller_configuration": {"controller": "route-idm-v2"},
           "initial_world_by_generation": {str(g): _world(0, g) for g in (0, 1)}, "steps": [], "frames": []}
    for g in (0, 1):
        for t in range(1, 5):
            doc["steps"].append({**_world(t, g), "generation": g, "schema_version": 1,
                "status": "ok", "operation": "step", "policy_actions": {"ego": 2, "car": 7},
                "executed_actions": {"ego": 2, "car": 1 if g and t == 2 else 7},
                "valid": [True, True], "terminated": [False, False], "truncated": [t == 4]*2,
                "events": {"collision": [g == 0 and t == 2, False],
                           "offroad": [False, g == 1 and t == 3], "goal": [False, False]}})
        for t in (0, 2, 4):
            doc["frames"].append({"generation": g, "tick": t, "render_tick": t,
                "time_seconds": t*.1, "simulation_timestamp_us": 1000000+t*100000,
                "coverage": {"world_without_rigid_nodes": ["car"], "reconstruction_only_nodes": ["decoration"],
                             "ego_appearance": "camera_car_not_reconstructed"},
                "cameras": [{"camera_id": "front", "owner_actor_id": "ego", "width": 32, "height": 24,
                    "K": [[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]], "camera_model": "pinhole",
                    "source_lens_valid_fraction": 1., "source_timestamp_us": 1000000+t*100000,
                    "camera_timestamp_us": 999000+t*100000, "source_seed_error_us": 0,
                    **{key: "camera.png" for key in ("image", "pinhole", "source", "valid")}}]})
    session = source/"paired-manifest.json"
    session.write_text(json.dumps(doc))
    destination = tmp_path/"episode"
    export_mtgs_episode(session, destination)
    return destination/"manifest.json"


def _load(path):
    return json.loads(path.read_text())


def _mutate_rows(episode, name, update):
    doc = _load(episode)
    file = episode.parent/name
    values = [json.loads(line) for line in file.read_text().splitlines()]
    update(values)
    file.write_text("".join(json.dumps(v)+"\n" for v in values))
    entry = next(item for item in doc["files"] if item["path"] == name)
    entry.update(sha256=hashlib.sha256(file.read_bytes()).hexdigest(), records=len(values))
    episode.write_text(json.dumps(doc))


def _clip(result, output, channel, reason=None):
    item = next(c for c in result["candidates"] if c["channel"] == channel and (reason is None or c["reason"] == reason))
    return item, _load(output/item["clip"]["path"])


def test_exact_actions_and_bracketing_are_retained_at_zero_context(episode, tmp_path):
    output = tmp_path/"mined"
    result = mine_mtgs_episode(episode, output, context_before_seconds=0, context_after_seconds=0)
    candidate, clip = _clip(result, output, "interaction_intervention")
    assert candidate["evidence_start_tick"] == candidate["evidence_end_tick"] == 2
    assert (candidate["start_tick"], candidate["end_tick"]) == (0, 2)
    assert clip["initial_world"]["tick"] == 0
    assert [p["observation_tick"] for p in clip["physics"]] == [0, 1]
    assert [p["behavior"]["executed_actions"]["car"] for p in clip["physics"]] == [7, 1]
    assert [f["tick"] for f in clip["sensors"]] == [0, 2]
    assert len(clip["camera_intervals"]) == 1
    assert len(clip["camera_intervals"][0]["joint_action_sequence"]) == 2
    assert all(p["generation"] == 1 for p in clip["physics"])
    assert "privileged" not in clip["sensors"][0]
    assert clip["source"]["protocol"]["kind"] == "fixture"
    assert result["verification"]["images_copied"] == 0
    assert not list(output.rglob("*.png"))


def test_candidates_separate_driving_from_quality_and_cannot_promote_training(episode, tmp_path):
    output = tmp_path/"mined"
    report = mine_mtgs_episode(episode, output)
    assert report["summary"]["by_channel"]["driving_event"] == 2
    assert report["summary"]["by_channel"]["driving_behavior"] == 1
    assert report["summary"]["by_channel"]["interaction_intervention"] == 1
    assert report["summary"]["by_channel"]["render_quality"] > 0
    item, clip = _clip(report, output, "driving_behavior")
    assert [p["tick"] for p in clip["evidence"]] == [2, 3]  # stopped tick 4 excluded
    assert item["context_clipped_before"] and item["context_clipped_after"]
    assert not report["trainable"] and not clip["trainable"]
    assert report["eligibility"] == _load(episode)["eligibility"]
    assert not clip["eligibility"]["production_world_model_training"]["eligible"]
    assert not clip["eligibility"]["visual_on_policy_rl"]["eligible"]
    assert "reward" in clip["unavailable_fields"]
    assert all(c["quality_scope"] is not None for c in report["candidates"] if c["channel"] == "render_quality")
    assert all(c["quality_scope"] is None for c in report["candidates"] if c["channel"] != "render_quality")


def test_deterministic_candidate_ids_and_exact_clip_hashes(episode, tmp_path):
    first = mine_mtgs_episode(episode, tmp_path/"a")
    second = mine_mtgs_episode(episode, tmp_path/"b")
    assert first == second
    for candidate in first["candidates"]:
        assert hashlib.sha256((tmp_path/"a"/candidate["clip"]["path"]).read_bytes()).hexdigest() == candidate["clip"]["sha256"]


@pytest.mark.parametrize("target", ["source", "sequence", "image"])
def test_corruption_fails_before_output_creation(episode, tmp_path, target):
    doc = _load(episode)
    path = (episode.parent/doc["files"][0]["path"] if target == "sequence" else
            Path(doc["source"]["manifest"]) if target == "source" else
            Path(doc["source"]["artifact_root"])/"camera.png")
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="hash"):
        mine_mtgs_episode(episode, tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


def test_broken_successor_rejected_even_with_recomputed_file_hash(episode, tmp_path):
    _mutate_rows(episode, "generation-000-physics.jsonl", lambda rows: rows[1]["privileged"]["world_before"]["state"][0].__setitem__(0, 999.))
    with pytest.raises(ValueError, match="continuity"):
        mine_mtgs_episode(episode, tmp_path/"bad-chain")
    assert not (tmp_path/"bad-chain").exists()


def test_missing_between_frame_action_is_rejected(episode, tmp_path):
    _mutate_rows(episode, "generation-000-camera-intervals.jsonl", lambda rows: rows[0]["joint_action_sequence"].pop())
    with pytest.raises(ValueError, match="intervening joint actions"):
        mine_mtgs_episode(episode, tmp_path/"bad-interval")


def test_missing_bracketing_capture_is_rejected(episode, tmp_path):
    _mutate_rows(episode, "generation-000-sensors.jsonl", lambda rows: rows.pop(0))
    doc = _load(episode)
    doc["generations"][0]["sensor_frames"] = 2
    episode.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="bracketing"):
        mine_mtgs_episode(episode, tmp_path/"bad-capture")


def test_recorded_mismatch_flag_cannot_disagree_with_actions(episode, tmp_path):
    _mutate_rows(episode, "generation-001-physics.jsonl", lambda rows: rows[1]["behavior"]["execution_differs_from_policy"].update(car=False))
    with pytest.raises(ValueError, match="difference"):
        mine_mtgs_episode(episode, tmp_path/"bad-difference")


def test_eligibility_promoted_input_rejected(episode, tmp_path):
    doc = _load(episode)
    doc["eligibility"]["visual_on_policy_rl"]["eligible"] = True
    episode.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="promote"):
        mine_mtgs_episode(episode, tmp_path/"bad-eligibility")


@pytest.mark.parametrize("kwargs", [{"context_before_seconds": -1}, {"context_after_seconds": float("nan")},
                                    {"braking_threshold_mps2": 0}, {"minimum_speed_mps": -1},
                                    {"context_before_seconds": True}])
def test_bad_configuration_fails_without_outputs(episode, tmp_path, kwargs):
    with pytest.raises(ValueError):
        mine_mtgs_episode(episode, tmp_path/"bad-config", **kwargs)
    assert not (tmp_path/"bad-config").exists()


def test_never_overwrite_existing_output(episode, tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    (output/"precious").write_text("preserve")
    with pytest.raises(FileExistsError):
        mine_mtgs_episode(episode, output)
    assert (output/"precious").read_text() == "preserve"


def test_path_escape_rejected(episode, tmp_path):
    doc = _load(episode)
    doc["files"][0]["path"] = "../private"
    episode.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sequence files"):
        mine_mtgs_episode(episode, tmp_path/"path-escape")



def test_non_camera_actor_height_failure_is_not_hidden_by_supported_owner(episode, tmp_path):
    session_path = Path(_load(episode)["source"]["manifest"])
    doc = _load(session_path)
    doc.update(height_mode="reference-track", height_configuration={"version": "reference-track-height-v1"})
    for frame in doc["frames"]:
        actors = [{"actor_id": a, "mode": "reference-track", "supported": a == "ego" or frame["tick"] != 4,
                   "reason": "target_outside_reference_corridor" if a == "car" and frame["tick"] == 4 else "supported",
                   "correction_m": 0.} for a in ("ego", "car")]
        supported = all(a["supported"] for a in actors)
        frame["height_binding"] = {"mode": "reference-track", "all_controlled_supported": supported,
            "per_camera": [{"camera_id": "front", "source_timestamp_us": frame["simulation_timestamp_us"],
                            "all_controlled_supported": supported, "actors": actors}]}
        frame["cameras"][0].update(height_binding=actors[0], scene_height_supported=supported)
    session_path.write_text(json.dumps(doc))
    export = tmp_path/"height-episode"
    export_mtgs_episode(session_path, export)
    output = tmp_path/"height-mined"
    report = mine_mtgs_episode(export/"manifest.json", output)
    cases = [c for c in report["candidates"] if c["reason"] == "height_unsupported"]
    assert len(cases) == 2
    assert all(c["actor_id"] == "car" and c["quality_scope"] == "dynamic_support_failure" for c in cases)
    assert all(c["evidence_start_tick"] == c["evidence_end_tick"] == 4 for c in cases)
    assert all(c["channel"] == "render_quality" for c in cases)


def test_physical_evidence_gap_splits_runs_without_manufacturing_severity(episode, tmp_path):
    session_path = Path(_load(episode)["source"]["manifest"])
    doc = _load(session_path)
    for row in doc["steps"]:
        if row["generation"] == 1:
            row["motion"][1][0] = -4. if row["tick"] in (1, 3) else 0.
    session_path.write_text(json.dumps(doc))
    export = tmp_path/"gap-episode"
    export_mtgs_episode(session_path, export)
    report = mine_mtgs_episode(export/"manifest.json", tmp_path/"gap-mined")
    cases = [c for c in report["candidates"] if c["reason"] == "severe_deceleration"]
    assert [(c["evidence_start_tick"], c["evidence_end_tick"], c["evidence_count"]) for c in cases] == [(1, 1, 1), (3, 3, 1)]


def test_forged_export_that_changes_complete_world_chain_still_rejects_source_disagreement(episode, tmp_path):
    def mutate(rows):
        for row in rows:
            for world in row["privileged"].values():
                world["state"][0][0] += 100.
    _mutate_rows(episode, "generation-000-physics.jsonl", mutate)
    with pytest.raises(ValueError, match="physics differs from source"):
        mine_mtgs_episode(episode, tmp_path/"forged-chain")



def test_actor_order_cannot_be_relabelled_by_episode_manifest(episode, tmp_path):
    document = _load(episode)
    document["actor_ids"].reverse()
    episode.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="actor order"):
        mine_mtgs_episode(episode, tmp_path/"relabelled-actors")
    assert not (tmp_path/"relabelled-actors").exists()


def test_every_generation_must_preserve_source_actor_order(episode, tmp_path):
    document = _load(episode)
    source_path = Path(document["source"]["manifest"])
    source = _load(source_path)
    source["initial_world_by_generation"]["1"]["all_actor_ids"].reverse()
    source_path.write_text(json.dumps(source))
    document["source"]["manifest_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    episode.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="actor order"):
        mine_mtgs_episode(episode, tmp_path/"generation-actor-order")


@pytest.mark.parametrize("yaw,velocity,expected", [(0., -2., False), (3.141592653589793, -2., True)])
def test_braking_uses_signed_forward_speed_in_actor_heading(episode, tmp_path, yaw, velocity, expected):
    source_path = Path(_load(episode)["source"]["manifest"])
    source = _load(source_path)
    for world in list(source["initial_world_by_generation"].values())+source["steps"]:
        world["state"][1][3] = yaw
        if world["tick"] != 4:
            world["state"][1][4] = velocity
    source_path.write_text(json.dumps(source))
    destination = tmp_path/"signed-episode"
    export_mtgs_episode(source_path, destination)
    output = tmp_path/"signed-mined"
    result = mine_mtgs_episode(destination/"manifest.json", output)
    candidates = [c for c in result["candidates"] if c["reason"] == "severe_deceleration"]
    assert bool(candidates) is expected
    if expected:
        clip = _load(output/candidates[0]["clip"]["path"])
        assert all(point["speed_mps"] > 0 and point["speed_semantics"] == "signed_forward_speed" for point in clip["evidence"])


def _external_episode(episode, tmp_path):
    from driversi.contracts.population import (
        ActorPolicyBinding,
        EpisodePopulation,
        PopulationPolicy,
    )

    source_path = Path(_load(episode)["source"]["manifest"])
    source = _load(source_path)
    population = EpisodePopulation(
        (PopulationPolicy("rule", "fixture-rule", "route-idm-v2"),
         PopulationPolicy("external", "fixture-external", "external-joint-v1")),
        (ActorPolicyBinding("ego", "rule"), ActorPolicyBinding("car", "external")))
    source["population_configuration"] = population.to_dict()
    source["behavior_policies"] = population.behavior_policies
    source["controller_configuration"].update(controller="episode-population-v1",
                                               population_configuration=population.to_dict())
    source["protocol"] = {"kind": "external_population_interaction", "rgb_is_policy_input": None,
                          "external_policy_inputs": "unrecorded"}
    for world in source["initial_world_by_generation"].values():
        world["behavior_policies"] = population.behavior_policies
    for row in source["steps"]:
        row["behavior_policies"] = population.behavior_policies
        overrides = [actor for actor in ("ego", "car") if row["policy_actions"][actor] != row["executed_actions"][actor]]
        row["override_actor_ids"] = overrides
        row["action_origins"] = {actor: "override" if actor in overrides else population.behavior_policies[actor]["controller"] for actor in ("ego", "car")}
    for frame in source["frames"]:
        frame["behavior_policies"] = population.behavior_policies
    source_path.write_text(json.dumps(source))
    destination = tmp_path/"external-episode"
    export_mtgs_episode(source_path, destination)
    return destination/"manifest.json"


def test_external_actor_input_unknown_survives_mining_and_cannot_become_training(episode, tmp_path):
    external = _external_episode(episode, tmp_path)
    output = tmp_path/"external-mined"
    result = mine_mtgs_episode(external, output)
    assert result["policy_bindings_recorded"] is True
    assert not result["trainable"]
    assert result["source"]["protocol"]["rgb_is_policy_input"] is None
    for item in result["candidates"]:
        clip = _load(output/item["clip"]["path"])
        assert all(row["behavior"]["rgb_is_policy_input"] is None for row in clip["physics"])
        assert all(frame["used_for_policy"] is None for frame in clip["sensors"])
        assert all(camera["used_for_policy"] is None for frame in clip["sensors"] for camera in frame["cameras"])
        assert all(interval["used_for_policy"] is None for interval in clip["camera_intervals"])
        assert not clip["eligibility"]["visual_on_policy_rl"]["eligible"]


@pytest.mark.parametrize("field", ["behavior", "sensor", "camera", "interval"])
def test_external_input_unknown_cannot_be_relabelled_false(episode, tmp_path, field):
    external = _external_episode(episode, tmp_path)
    if field == "behavior":
        name = "generation-000-physics.jsonl"
        def mutate(rows):
            rows[0]["behavior"].update(rgb_is_policy_input=False)
    elif field == "interval":
        name = "generation-000-camera-intervals.jsonl"
        def mutate(rows):
            rows[0].update(used_for_policy=False)
    else:
        name = "generation-000-sensors.jsonl"
        mutate = (lambda rows: rows[0].update(used_for_policy=False)) if field == "sensor" else (lambda rows: rows[0]["cameras"][0].update(used_for_policy=False))
    _mutate_rows(external, name, mutate)
    with pytest.raises(ValueError):
        mine_mtgs_episode(external, tmp_path/"external-relabelled")
    assert not (tmp_path/"external-relabelled").exists()


@pytest.mark.parametrize("field", ["behavior", "sensor", "camera", "interval"])
def test_rule_input_cannot_claim_unknown_without_external_population(episode, tmp_path, field):
    if field == "behavior":
        name = "generation-000-physics.jsonl"
        def mutate(rows):
            rows[0]["behavior"].update(rgb_is_policy_input=None)
    elif field == "interval":
        name = "generation-000-camera-intervals.jsonl"
        def mutate(rows):
            rows[0].update(used_for_policy=None)
    else:
        name = "generation-000-sensors.jsonl"
        mutate = (lambda rows: rows[0].update(used_for_policy=None)) if field == "sensor" else (lambda rows: rows[0]["cameras"][0].update(used_for_policy=None))
    _mutate_rows(episode, name, mutate)
    with pytest.raises(ValueError):
        mine_mtgs_episode(episode, tmp_path/"invented-external")


def test_branch_prefix_is_context_not_new_evidence(episode, tmp_path):
    exported = _load(episode)
    session_path = Path(exported["source"]["manifest"])
    session = _load(session_path)
    session["branch_lineage"] = [{"session_id": "parent", "generation": 1, "tick": 2,
                                   "inherited_steps": 6, "prefix_is_new_sample": False}]
    session_path.write_text(json.dumps(session))
    output_episode = tmp_path/"branch-episode"
    export_mtgs_episode(session_path, output_episode)
    output = tmp_path/"branch-candidates"
    result = mine_mtgs_episode(output_episode/"manifest.json", output)
    assert result["candidates"]
    assert all(c["generation"] == 1 and c["evidence_start_tick"] > 2 for c in result["candidates"])
    assert not any(c["reason"] == "collision" for c in result["candidates"])
    assert not any(c["reason"] == "policy_execution_mismatch" for c in result["candidates"])


def test_branch_lineage_cannot_be_removed_from_export(episode, tmp_path):
    exported = _load(episode)
    session_path = Path(exported["source"]["manifest"])
    session = _load(session_path)
    session["branch_lineage"] = [{"session_id": "parent", "generation": 1, "tick": 2}]
    session_path.write_text(json.dumps(session))
    destination = tmp_path/"branch-episode"
    export_mtgs_episode(session_path, destination)
    path = destination/"manifest.json"
    value = _load(path)
    value.pop("branch_lineage")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="lineage"):
        mine_mtgs_episode(path, tmp_path/"false-new-evidence")


def test_mining_retains_world_presence_and_rejects_source_disagreement(episode, tmp_path):
    session_path = Path(_load(episode)["source"]["manifest"])
    doc = _load(session_path)
    for world in [*doc["initial_world_by_generation"].values(), *doc["steps"]]:
        world["presence"] = [True] * len(world["all_actor_ids"])
        world["lifecycle"] = {"tick": world["tick"], "present": world["presence"],
                              "entered": world["all_actor_ids"] if world["tick"] == 0 else [],
                              "exited": [], "contacts": []}
    session_path.write_text(json.dumps(doc))
    export = tmp_path / "lifecycle-episode"
    export_mtgs_episode(session_path, export)
    output = tmp_path / "lifecycle-mined"
    result = mine_mtgs_episode(export / "manifest.json", output)
    _, clip = _clip(result, output, "interaction_intervention")
    assert clip["initial_world"]["presence"] == [True, True]
    assert "lifecycle" in clip["physics"][0]["privileged"]["world_after"]
    def forge(rows):
        for row in rows:
            for world in row["privileged"].values():
                world.pop("lifecycle")
    _mutate_rows(export / "manifest.json", "generation-000-physics.jsonl", forge)
    with pytest.raises(ValueError, match="physics differs"):
        mine_mtgs_episode(export / "manifest.json", tmp_path / "forged-lifecycle")
