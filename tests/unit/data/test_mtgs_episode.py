"""Inspection exports retain causal timing and fail closed on incomplete runs."""

import copy
import hashlib
import json

import pytest

from driversi.data.assets.mtgs_episode import export_mtgs_episode


def world(tick):
    return {
        "tick": tick,
        "all_actor_ids": ["ego", "car", "replay"],
        "controlled_ids": ["ego", "car"],
        "state": [
            [float(tick), float(i), 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(3)
        ],
        "motion": [[0.0, 0.0, 0.0] for _ in range(3)],
        "alive": [True, True],
        "done": tick == 4,
        "build_id": "native-test",
        "dt_seconds": 0.1,
    }


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "camera.png").write_bytes(b"fixture image bytes preserved without image decoding")
    document = {
        "schema_version": 1,
        "kind": "mtgs_world_session",
        "status": "completed",
        "trainable": False,
        "display_only": True,
        "dt_seconds": 0.1,
        "controlled_ids": ["ego", "car"],
        "camera_ids": ["front"],
        "physics_build_id": "native-test",
        "protocol": {"rgb_is_policy_input": False},
        "controller_configuration": {"controller": "route-idm-v2", "tracking": {"stateful": False}},
        "initial_world_by_generation": {"0": world(0), "1": world(0)},
        "steps": [],
        "frames": [],
        "scenario_id": "fixture",
        "source_log": "log",
    }
    for generation in range(2):
        for tick in range(1, 5):
            document["steps"].append(
                {
                    **world(tick),
                    "generation": generation,
                    "schema_version": 1,
                    "status": "ok",
                    "operation": "step",
                    "policy_actions": {"ego": tick, "car": 7},
                    "executed_actions": {"ego": tick, "car": 1 if generation else 7},
                    "valid": [True, True],
                    "terminated": [False, False],
                    "truncated": [tick == 4] * 2,
                    "events": {key: [False, False] for key in ("collision", "offroad", "goal")},
                }
            )
        for tick in (0, 2, 4):
            document["frames"].append(
                {
                    "generation": generation,
                    "tick": tick,
                    "render_tick": tick,
                    "time_seconds": tick * 0.1,
                    "simulation_timestamp_us": 1000000 + tick * 100000,
                    "cameras": [
                        {
                            "camera_id": "front",
                            "owner_actor_id": "ego",
                            "width": 32,
                            "height": 24,
                            "K": [[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]],
                            "camera_model": "pinhole",
                            "source_lens_valid_fraction": 1.0,
                            "source_timestamp_us": 1000000 + tick * 100000,
                            "camera_timestamp_us": 999000 + tick * 100000,
                            "source_seed_error_us": 0,
                            **{
                                key: "camera.png" for key in ("image", "pinhole", "source", "valid")
                            },
                        }
                    ],
                }
            )
    path = root / "paired-manifest.json"
    path.write_text(json.dumps(document))
    return path, document


def write_source(source):
    path, document = source
    path.write_text(json.dumps(document))
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_retains_complete_joint_action_intervals_and_privilege_boundary(source, tmp_path):
    path, _ = source
    destination = tmp_path / "export"
    report = export_mtgs_episode(path, destination)
    assert report["summary"]["physics_transitions"] == 8
    assert report["summary"]["referenced_artifact_files"] == 1
    assert report["source"]["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["trainable"] is False
    assert report["eligibility"]["dynamics_inspection"]["eligible"]
    assert not report["eligibility"]["production_world_model_training"]["eligible"]
    assert not report["eligibility"]["visual_on_policy_rl"]["eligible"]
    assert "reward" in report["unavailable_fields"]
    intervals = rows(destination / "generation-001-camera-intervals.jsonl")
    assert [(row["start_tick"], row["end_tick"]) for row in intervals] == [(0, 2), (2, 4)]
    assert [x["executed_actions"]["ego"] for x in intervals[0]["joint_action_sequence"]] == [1, 2]
    assert [x["executed_actions"]["car"] for x in intervals[0]["joint_action_sequence"]] == [1, 1]
    assert intervals[0]["used_for_policy"] is False
    physics = rows(destination / "generation-001-physics.jsonl")
    assert physics[0]["privileged"]["world_before"]["tick"] == 0
    assert physics[0]["privileged"]["world_after"]["tick"] == 1
    assert physics[0]["behavior"]["policy_actions"]["car"] == 7
    assert physics[0]["behavior"]["executed_actions"]["car"] == 1
    assert physics[0]["behavior"]["execution_differs_from_policy"]["car"]
    assert "reward" not in physics[0]
    sensors = rows(destination / "generation-001-sensors.jsonl")
    assert "privileged" not in sensors[0]
    camera = sensors[0]["cameras"][0]
    assert camera["owner_actor_id"] == "ego"
    assert (
        camera["artifacts"]["image"]["sha256"]
        == hashlib.sha256((path.parent / "camera.png").read_bytes()).hexdigest()
    )
    assert not list(destination.rglob("*.png"))
    with pytest.raises(FileExistsError):
        export_mtgs_episode(path, destination)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d.pop("initial_world_by_generation"), "initial_world_by_generation"),
        (lambda d: d.update(status="running"), "completed"),
        (lambda d: d.update(schema_version=True), "schema_version"),
        (lambda d: d["steps"][0].update(status="error"), "successful worker"),
        (lambda d: d["steps"][0].update(capture_status="failed"), "failed capture"),
        (lambda d: d["frames"][1].update(simulation_timestamp_us=1200001), "physical clock"),
        (lambda d: d.update(trainable=True), "non-trainable"),
        (lambda d: d["protocol"].update(rgb_is_policy_input=True), "policy input"),
        (lambda d: d["steps"].pop(1), "contiguous"),
        (lambda d: d["steps"][2].update(generation=1), "interleaved"),
        (lambda d: d["initial_world_by_generation"]["1"].update(tick=4), "tick zero"),
        (lambda d: d["steps"][0]["state"].pop(), "state shape"),
        (lambda d: d["steps"][0]["motion"][0].append(0), "motion shape"),
        (lambda d: d["steps"][0].update(alive=[True]), "alive mask"),
        (lambda d: d["steps"][0].update(valid=[True, False]), "Valid mask"),
        (lambda d: d["steps"][0].update(terminated=[True, False]), "terminal"),
        (lambda d: d["steps"][0].update(truncated=[True, False]), "terminal"),
        (lambda d: d["steps"][0]["executed_actions"].update(ego=True), "integer actions"),
        (lambda d: d["steps"][0]["policy_actions"].update(ego=12), "integer actions"),
        (lambda d: d["steps"][0]["executed_actions"].update(replay=2), "exactly"),
        (lambda d: d["steps"][0].update(dt_seconds=0.2), "dt changed"),
        (lambda d: d["steps"][0].update(controlled_ids=["car", "ego"]), "actor order"),
        (lambda d: d["frames"][0]["cameras"][0].update(owner_actor_id="absent"), "owner"),
        (lambda d: d["frames"][1]["cameras"][0].update(owner_actor_id="car"), "owner changed"),
        (
            lambda d: d["frames"][0]["cameras"].append(copy.deepcopy(d["frames"][0]["cameras"][0])),
            "camera IDs",
        ),
        (lambda d: d["frames"][0]["cameras"][0].update(image="../outside.png"), "Unsafe"),
        (lambda d: d["frames"][0]["cameras"][0].update(image="/tmp/outside.png"), "Unsafe"),
        (lambda d: d["frames"][0]["cameras"][0].update(image="C:/outside.png"), "Unsafe"),
        (lambda d: d["frames"][0]["cameras"][0].update(image="missing.png"), "missing"),
        (lambda d: d["frames"][1].update(render_tick=1), "actual physics ticks"),
        (lambda d: d["frames"][1].update(time_seconds=0.1), "simulation time"),
        (lambda d: d["frames"][1]["cameras"][0].update(source_seed_error_us=1), "seed offset"),
        (lambda d: d["frames"].pop(0), "initial and terminal"),
        (lambda d: d["steps"][0]["state"][0].__setitem__(0, float("nan")), "Non-finite"),
    ],
)
def test_reject_incomplete_or_misbound_episode_before_writing(source, tmp_path, mutation, match):
    _, document = source
    mutation(document)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=match):
        export_mtgs_episode(write_source(source), output)
    assert not output.exists()


def test_symlink_cannot_escape_artifact_root(source, tmp_path):
    path, document = source
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (path.parent / "escape.png").symlink_to(outside)
    document["frames"][0]["cameras"][0]["image"] = "escape.png"
    with pytest.raises(ValueError, match="escapes"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected")


def test_reject_cross_generation_frame_continuation(source, tmp_path):
    _, document = source
    document["frames"][3].update(tick=4, render_tick=4, time_seconds=0.4)
    with pytest.raises(ValueError, match="strictly ordered"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected")


def add_virtual_stream(document):
    """Match MTGSWorldSession's actual virtual-front record, without fake photos."""
    stream_id = "car__virtual_front"
    document["camera_ids"].append(stream_id)
    document["virtual_camera_actors"] = ["car"]
    for frame in document["frames"]:
        source = frame["cameras"][0]
        camera = copy.deepcopy(source)
        camera.update(
            camera_id=stream_id,
            owner_actor_id="car",
            sensor_id="virtual_front",
            virtual=True,
            trainable=False,
            calibration_source_camera_id="CAM_F0",
            source=None,
            source_comparison="unavailable_no_real_image_for_virtual_owner",
            camera_to_world=[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            camera_to_world_semantics="effective_nerfstudio_pose_in_reconstruction_axes",
            rig={"version": "roof-front-v1"},
            support={"quality_verified": False},
            source_seed={
                "index": frame["tick"],
                "frame_token": "seed",
                "camera_token": "source-cam",
                "timestamp_us": source["source_timestamp_us"],
            },
        )
        frame["cameras"].append(camera)


def test_actual_virtual_stream_has_no_photographic_reference(source, tmp_path):
    _, document = source
    add_virtual_stream(document)
    output = tmp_path / "virtual-export"
    report = export_mtgs_episode(write_source(source), output)
    assert report["camera_owners"] == {"front": "ego", "car__virtual_front": "car"}
    sensor = rows(output / "generation-000-sensors.jsonl")[0]
    real, virtual = sensor["cameras"]
    assert real["artifacts"]["source"]["path"] == "camera.png"
    assert virtual["artifacts"]["source"] is None
    assert virtual["source_comparison"] == "unavailable_no_real_image_for_virtual_owner"
    assert virtual["virtual"] is True
    assert virtual["capture_metadata"]["camera_to_world"][0][3] == 1.0
    assert virtual["capture_metadata"]["support"]["quality_verified"] is False
    intervals = rows(output / "generation-000-camera-intervals.jsonl")
    assert len(intervals) == 4
    virtual_intervals = [row for row in intervals if row["owner_actor_id"] == "car"]
    assert [
        x["executed_actions"]["ego"] for x in virtual_intervals[0]["joint_action_sequence"]
    ] == [1, 2]
    assert report["trainable"] is False


@pytest.mark.parametrize("missing", ["image", "pinhole", "valid"])
def test_virtual_stream_still_requires_all_rendered_artifacts(source, tmp_path, missing):
    _, document = source
    add_virtual_stream(document)
    document["frames"][0]["cameras"][1][missing] = None
    with pytest.raises(ValueError, match="Unsafe camera artifact"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected-virtual")


@pytest.mark.parametrize("virtual", [False, None])
def test_real_or_undeclared_camera_cannot_omit_reference(source, tmp_path, virtual):
    _, document = source
    camera = document["frames"][0]["cameras"][0]
    camera["source"] = None
    if virtual is not None:
        camera["virtual"] = virtual
    with pytest.raises(ValueError, match="Unsafe camera artifact"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected-real")


def test_virtual_flag_cannot_be_a_truthy_non_boolean(source, tmp_path):
    _, document = source
    camera = document["frames"][0]["cameras"][0]
    camera.update(virtual="true", source=None)
    with pytest.raises(ValueError, match="virtual flag"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected-nonbool")


def add_height_diagnostics(document):
    document.update(
        binding_mode="reference_relative_planar_motion_with_bounded_track_height_correction",
        height_mode="reference-track",
        height_configuration={"version": "reference-track-height-v1", "max_lateral_m": 2.0},
        height_profile_unavailable={},
    )
    for frame in document["frames"]:
        per_camera = []
        for camera in frame["cameras"]:
            # A non-camera actor may fail even when the camera owner is supported.
            actors = [
                {
                    "actor_id": actor,
                    "mode": "reference-track",
                    "supported": actor == "ego",
                    "reason": "supported" if actor == "ego" else "target_outside_reference_extent",
                    "correction_m": -0.4 if actor == "ego" else 0.0,
                    "terrain_validated": False,
                    "source_projection": {"height_m": 1.0},
                    "target_projection": {"height_m": 0.6},
                    "geometry_role": "display_height_proxy_not_physics_or_terrain",
                }
                for actor in document["controlled_ids"]
            ]
            camera["height_binding"] = copy.deepcopy(
                next(a for a in actors if a["actor_id"] == camera["owner_actor_id"])
            )
            camera["scene_height_supported"] = False
            per_camera.append(
                {
                    "camera_id": camera["camera_id"],
                    "source_timestamp_us": camera["source_timestamp_us"],
                    "all_controlled_supported": False,
                    "actors": actors,
                }
            )
        frame["height_binding"] = {
            "mode": "reference-track",
            "all_controlled_supported": False,
            "per_camera": per_camera,
        }


def test_height_provenance_and_whole_scene_failure_survive_export(source, tmp_path):
    _, document = source
    add_virtual_stream(document)
    add_height_diagnostics(document)
    output = tmp_path / "height-export"
    report = export_mtgs_episode(write_source(source), output)
    for key in (
        "binding_mode",
        "height_mode",
        "height_configuration",
        "height_profile_unavailable",
    ):
        assert report[key] == document[key]
    assert report["height_metadata_recorded"] is True
    sensor = rows(output / "generation-000-sensors.jsonl")[0]
    assert sensor["height_binding"] == document["frames"][0]["height_binding"]
    assert sensor["height_metadata_recorded"] is True
    for original, camera in zip(document["frames"][0]["cameras"], sensor["cameras"], strict=True):
        assert camera["capture_metadata"]["height_binding"] == original["height_binding"]
        assert camera["capture_metadata"]["scene_height_supported"] is False
    assert sensor["cameras"][0]["capture_metadata"]["height_binding"]["supported"] is True
    assert sensor["height_binding"]["all_controlled_supported"] is False
    assert report["trainable"] is False
    assert report["eligibility"]["production_world_model_training"]["eligible"] is False


def test_legacy_height_is_unknown_and_not_relabelled_supported(source, tmp_path):
    output = tmp_path / "legacy-height-export"
    report = export_mtgs_episode(source[0], output)
    assert report["height_mode"] is None
    assert report["height_configuration"] is None
    assert report["height_metadata_recorded"] is False
    sensor = rows(output / "generation-000-sensors.jsonl")[0]
    assert sensor["height_binding"] is None
    assert sensor["height_metadata_recorded"] is False
    assert "scene_height_supported" not in sensor["cameras"][0]["capture_metadata"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d["frames"][0].pop("height_binding"), "complete frame"),
        (
            lambda d: d["frames"][0]["height_binding"].update(all_controlled_supported=True),
            "frame support",
        ),
        (lambda d: d["frames"][0]["height_binding"]["per_camera"].clear(), "camera coverage"),
        (
            lambda d: d["frames"][0]["height_binding"]["per_camera"][0]["actors"].pop(),
            "actor coverage",
        ),
        (
            lambda d: d["frames"][0]["cameras"][0].update(scene_height_supported=True),
            "scene support",
        ),
        (
            lambda d: d["frames"][0]["cameras"][0]["height_binding"].update(correction_m=1.0),
            "owner height",
        ),
        (
            lambda d: d["frames"][0]["height_binding"]["per_camera"][0]["actors"][1].update(
                correction_m=1.0
            ),
            "zero fallback",
        ),
    ],
)
def test_contradictory_height_diagnostics_fail_before_writing(source, tmp_path, mutation, match):
    _, document = source
    add_height_diagnostics(document)
    mutation(document)
    output = tmp_path / "rejected-height"
    with pytest.raises(ValueError, match=match):
        export_mtgs_episode(write_source(source), output)
    assert not output.exists()


def add_uncontrolled_source_camera(document):
    add_height_diagnostics(document)
    for frame in document["frames"]:
        camera = frame["cameras"][0]
        camera["owner_actor_id"] = "replay"
        camera["height_binding"] = {
            "mode": "reference-track",
            "actor_id": "replay",
            "supported": False,
            "reason": "uncontrolled_source_camera",
            "correction_m": 0.0,
            "fallback": "retain_source_time_height",
            "terrain_validated": False,
        }
        # Controlled actors may all be supported although this camera is not.
        binding = frame["height_binding"]
        binding["all_controlled_supported"] = True
        for per_camera in binding["per_camera"]:
            per_camera["all_controlled_supported"] = True
            for actor in per_camera["actors"]:
                actor.update(supported=True, reason="supported")
        camera["scene_height_supported"] = True


def test_uncontrolled_source_camera_keeps_separate_unsupported_height(source, tmp_path):
    _, document = source
    add_uncontrolled_source_camera(document)
    output = tmp_path / "uncontrolled-height-export"
    result = export_mtgs_episode(write_source(source), output)
    frame = rows(output / "generation-000-sensors.jsonl")[0]
    assert result["camera_owners"] == {"front": "replay"}
    assert frame["height_binding"]["all_controlled_supported"] is True
    actors = frame["height_binding"]["per_camera"][0]["actors"]
    assert [actor["actor_id"] for actor in actors] == ["ego", "car"]
    camera = frame["cameras"][0]["capture_metadata"]
    assert camera["height_binding"]["supported"] is False
    assert camera["height_binding"]["reason"] == "uncontrolled_source_camera"
    assert camera["scene_height_supported"] is True
    assert result["trainable"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["height_binding"].update(supported=True),
        lambda c: c["height_binding"].update(correction_m=0.3),
        lambda c: c["height_binding"].update(actor_id="ego"),
        lambda c: c["height_binding"].update(reason="supported"),
        lambda c: c.pop("height_binding"),
    ],
)
def test_uncontrolled_camera_cannot_claim_height_support(source, tmp_path, mutation):
    _, document = source
    add_uncontrolled_source_camera(document)
    mutation(document["frames"][0]["cameras"][0])
    output = tmp_path / "invalid-uncontrolled-height"
    with pytest.raises(ValueError, match="Uncontrolled camera"):
        export_mtgs_episode(write_source(source), output)
    assert not output.exists()


def add_population(document):
    from driversi.contracts.population import EpisodePopulation

    spec = {
        "schema_version": 1,
        "policies": [
            {"policy_id": "rules", "version": "r1", "controller": "route-idm-v2", "parameters": {}},
            {
                "policy_id": "external",
                "version": "checkpoint-test",
                "controller": "external-joint-v1",
                "parameters": {},
            },
        ],
        "assignments": [
            {"actor_id": "ego", "policy_id": "rules", "role": "participant"},
            {"actor_id": "car", "policy_id": "external", "role": "candidate"},
        ],
    }
    population = EpisodePopulation.from_dict(spec)
    document["protocol"]["rgb_is_policy_input"] = None
    document.update(
        population_configuration=population.to_dict(),
        behavior_policies=population.behavior_policies,
    )
    document["controller_configuration"]["population_configuration"] = population.to_dict()
    for record in [
        *document["initial_world_by_generation"].values(),
        *document["steps"],
        *document["frames"],
    ]:
        record["behavior_policies"] = population.behavior_policies
    for record in document["steps"]:
        overridden = ["car"] if record["generation"] == 1 else []
        record["override_actor_ids"] = overridden
        record["action_origins"] = {
            "ego": "route-idm-v2",
            "car": "override" if overridden else "external-joint-v1",
        }
    return population


def test_population_provenance_is_preserved_without_inventing_policy_observations(source, tmp_path):
    _, doc = source
    population = add_population(doc)
    result = export_mtgs_episode(write_source(source), tmp_path / "out")
    assert result["behavior_policies"] == population.behavior_policies
    assert result["policy_bindings_recorded"] and not result["trainable"]
    assert "explicit_override_flags" not in result["unavailable_fields"]
    assert "policy_observations" in result["unavailable_fields"]
    behavior = rows(tmp_path / "out/generation-001-physics.jsonl")[0]["behavior"]
    assert behavior["behavior_policies"]["car"]["observation_version"] == "external-unverified"
    assert (
        behavior["override_actor_ids"] == ["car"]
        and behavior["action_origins"]["car"] == "override"
    )
    assert not behavior["policy_observation_recorded"]
    assert behavior["rgb_is_policy_input"] is None
    assert "rgb_policy_usage_unknown" in result["eligibility"]["visual_on_policy_rl"]["reasons"]
    for sensor in rows(tmp_path / "out/generation-001-sensors.jsonl"):
        assert sensor["used_for_policy"] is None
        assert all(camera["used_for_policy"] is None for camera in sensor["cameras"])
    assert all(
        row["used_for_policy"] is None
        for row in rows(tmp_path / "out/generation-001-camera-intervals.jsonl")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["steps"][0]["behavior_policies"]["car"].update(version="changed"),
        lambda d: d["initial_world_by_generation"]["1"].pop("behavior_policies"),
        lambda d: d["frames"][0]["behavior_policies"]["car"].update(role="participant"),
        lambda d: d["steps"][0]["action_origins"].update(car="route-idm-v2"),
        lambda d: d["steps"][4].update(override_actor_ids=[]),
        lambda d: d["steps"][4].update(override_actor_ids=["car", "car"]),
        lambda d: d.pop("population_configuration"),
    ],
)
def test_export_rejects_mutated_population_identity_and_ambiguous_actions(
    source, tmp_path, mutation
):
    _, doc = source
    add_population(doc)
    mutation(doc)
    with pytest.raises(ValueError):
        export_mtgs_episode(write_source(source), tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("external,value", [(True, False), (False, None)])
def test_policy_input_usage_cannot_be_misrepresented(source, tmp_path, external, value):
    _, doc = source
    if external:
        add_population(doc)
    doc["protocol"]["rgb_is_policy_input"] = value
    with pytest.raises(ValueError, match="policy input"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_sdk_population_protocol_is_exportable_without_manual_manifest_mutation(source, tmp_path):
    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    _, doc = source
    add_population(doc)
    doc.pop("protocol")
    doc["limits"] = ["rule_driven_interaction_not_learned_visual_self_play"]
    session = MTGSWorldSession.__new__(MTGSWorldSession)
    session.manifest = doc
    session._record_population(
        {
            key: doc[key]
            for key in ("controlled_ids", "population_configuration", "behavior_policies")
        }
    )
    assert doc["protocol"]["rgb_is_policy_input"] is None
    assert doc["limits"] == ["external_policy_inputs_unrecorded"]
    result = export_mtgs_episode(write_source(source), tmp_path / "sdk")
    assert result["status"] == "completed" and not result["trainable"]


def teacher_source(source):
    import numpy as np

    from driversi.contracts.population import (
        ActorPolicyBinding,
        EpisodePopulation,
        PopulationPolicy,
    )
    from driversi.training.sampling.environment import OBSERVATION_VERSION

    path, doc = source
    pop = EpisodePopulation(
        (PopulationPolicy("teacher", "v1", "external-joint-v1"),),
        tuple(ActorPolicyBinding(a, "teacher") for a in doc["controlled_ids"]),
    )
    doc.update(
        session_id="fixture-session",
        teacher_artifact="teacher.npz",
        protocol={"kind": "learned_privileged_population", "rgb_is_policy_input": False},
        population_configuration=pop.to_dict(),
        behavior_policies=pop.behavior_policies,
    )
    doc["controller_configuration"]["population_configuration"] = pop.to_dict()
    for row in [*doc["initial_world_by_generation"].values(), *doc["steps"], *doc["frames"]]:
        row["behavior_policies"] = pop.behavior_policies
    sizes = {
        "trunk.0.weight": (128, 1125),
        "trunk.0.bias": (128,),
        "trunk.2.weight": (128, 128),
        "trunk.2.bias": (128,),
        "actor.weight": (12, 128),
        "actor.bias": (12,),
        "critic.weight": (1, 128),
        "critic.bias": (1,),
    }
    weights = {k: np.zeros(shape, np.float32) for k, shape in sizes.items()}
    weights["actor.bias"][7] = 1.0
    np.savez(
        path.parent / "teacher.npz",
        metadata=json.dumps({"observation_version": OBSERVATION_VERSION, "policy_version": "v1"}),
        **weights,
    )
    decisions = []
    for row in doc["steps"]:
        actions = {a: 7 for a in doc["controlled_ids"]}
        row.update(
            policy_actions=actions,
            executed_actions=actions,
            override_actor_ids=[],
            action_origins={a: "external-joint-v1" for a in actions},
        )
        decisions.append(
            {
                "identity": {
                    "session_id": doc["session_id"],
                    "tick": row["tick"] - 1,
                    "generation": row["generation"],
                },
                "policy_version": "v1",
                "input_mode": "privileged_structured",
                "rgb_is_policy_input": False,
                "selection": "argmax",
                "behavior_log_probability": 0.0,
                "actor_ids": doc["controlled_ids"],
                "actions": actions,
                "observations": np.zeros((2, 1125)).tolist(),
                "logits": np.tile(weights["actor.bias"], (2, 1)).tolist(),
                "values": [0.0, 0.0],
            }
        )
    (path.parent / "teacher-decisions.jsonl").write_text(
        "\n".join(json.dumps(x) for x in decisions)
    )
    return path, doc, decisions


def test_teacher_export_replays_inputs_and_preserves_their_references(source, tmp_path):
    teacher_source(source)
    output = tmp_path / "teacher-export"
    report = export_mtgs_episode(write_source(source), output)
    assert report["teacher_inputs"]["decisions"] == 8
    assert report["teacher_inputs"]["recomputed_outputs"]
    assert report["unavailable_fields"] == ["reward"]
    assert len(report["teacher_artifacts"]) == 2
    assert rows(output / "generation-000-physics.jsonl")[0]["behavior"][
        "policy_observation_recorded"
    ]
    assert not report["trainable"]


@pytest.mark.parametrize("mutation", ["missing", "tick", "logits", "actions"])
def test_teacher_export_rejects_unreproducible_input_claims(source, tmp_path, mutation):
    path, doc, decisions = teacher_source(source)
    if mutation == "missing":
        decisions.pop()
    elif mutation == "tick":
        decisions[0]["identity"]["tick"] = 9
    elif mutation == "logits":
        decisions[0]["logits"][0][0] = 2
    else:
        decisions[0]["actions"] = {"ego": 0, "car": 7}
    (path.parent / "teacher-decisions.jsonl").write_text(
        "\n".join(json.dumps(x) for x in decisions)
    )
    with pytest.raises(ValueError, match="Teacher"):
        export_mtgs_episode(write_source(source), tmp_path / "reject")


def test_quality_is_recomputed_and_legacy_missing_pose_remains_unknown(source, tmp_path):
    path, _ = source
    report = export_mtgs_episode(path, tmp_path / "quality-export")
    assert report["render_quality"]["camera_frames"] == 6
    assert report["render_quality"]["high_quality_supervision_camera_frames"] == 0
    sensor = rows(tmp_path / "quality-export/generation-000-sensors.jsonl")[0]
    assert not sensor["quality_metadata_recorded"]
    assert sensor["render_quality"]["cameras"][0]["checks"]["finite_rigid_camera_pose"] == "unknown"


def test_forged_quality_report_rejected_before_any_export(source, tmp_path):
    path, document = source
    document["frames"][0]["render_quality"] = {"high_quality_supervision_eligible": True}
    write_source(source)
    destination = tmp_path / "forged-quality"
    with pytest.raises(ValueError, match="quality contradicts"):
        export_mtgs_episode(path, destination)
    assert not destination.exists()


@pytest.mark.parametrize("end_tick", [1, 4])
def test_route_support_truncation_preserves_other_actors(source, tmp_path, end_tick):
    _, document = source
    for step in document["steps"]:
        tick = step["tick"]
        step["events"]["route_support_end"] = [False, tick == end_tick]
        step["valid"][1] = tick <= end_tick
        step["alive"][1] = tick < end_tick
        step["truncated"][1] = tick == end_tick
        step["state"][1][9] = float(tick >= end_tick)
    output = tmp_path / "support-export"
    report = export_mtgs_episode(write_source(source), output)
    physics = rows(output / "generation-000-physics.jsonl")
    end = physics[end_tick - 1]
    assert end["events"]["route_support_end"] == [False, True]
    assert not any(end["events"]["goal"])
    assert end["masks"]["terminated"] == [False, False]
    assert end["masks"]["truncated"][1]
    assert end["privileged"]["world_after"]["alive"] == [True, False]
    assert end["privileged"]["world_after"]["done"] == (end_tick == 4)
    if end_tick < 4:
        assert physics[end_tick]["masks"]["valid"] == [True, False]
        assert physics[end_tick]["masks"]["truncated"] == [False, False]
    assert report["summary"]["physics_transitions"] == 8
    assert not report["trainable"]


def test_optional_false_route_support_event_is_preserved(source, tmp_path):
    _, document = source
    for step in document["steps"]:
        step["events"]["route_support_end"] = [False, False]
    output = tmp_path / "route-aware-export"
    export_mtgs_episode(write_source(source), output)
    assert rows(output / "generation-000-physics.jsonl")[0]["events"]["route_support_end"] == [
        False,
        False,
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda step: step.update(truncated=[False, False]),
        lambda step: step.update(terminated=[False, True]),
        lambda step: step.update(alive=[True, True]),
        lambda step: step["events"].update(collision=[False, True]),
        lambda step: step["events"].update(offroad=[False, True]),
        lambda step: step["events"].update(goal=[False, True]),
    ],
)
def test_reject_contradictory_route_support_masks(source, tmp_path, mutation):
    _, document = source
    step = document["steps"][0]
    step.update(truncated=[False, True], alive=[True, False])
    step["events"]["route_support_end"] = [False, True]
    mutation(step)
    output = tmp_path / "rejected-support"
    with pytest.raises(ValueError, match="terminal/truncation/alive"):
        export_mtgs_episode(write_source(source), output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda events: events.update(unknown=[False, False]),
        lambda events: events.pop("goal"),
    ],
)
def test_reject_unknown_or_missing_events(source, tmp_path, mutation):
    mutation(source[1]["steps"][0]["events"])
    with pytest.raises(ValueError, match="event masks"):
        export_mtgs_episode(write_source(source), tmp_path / "rejected-events")


def native_source(source):
    from driversi.contracts.population import (
        EpisodePopulation,
        PopulationPolicy,
        ActorPolicyBinding,
    )

    path, d = source
    pop = EpisodePopulation(
        (PopulationPolicy("native", "weights", "external-native-state-v1"),),
        tuple(ActorPolicyBinding(a, "native") for a in ("ego", "car")),
    )
    d["population_configuration"] = pop.to_dict()
    d["behavior_policies"] = pop.behavior_policies
    d["controller_configuration"]["population_configuration"] = pop.to_dict()
    d["protocol"]["rgb_is_policy_input"] = None
    d["initial_world_by_generation"] = {"0": d["initial_world_by_generation"]["0"]}
    d["initial_world_by_generation"]["0"]["behavior_policies"] = pop.behavior_policies
    d["steps"] = [s for s in d["steps"] if s["generation"] == 0 and s["tick"] <= 2]
    d["frames"] = [s for s in d["frames"] if s["generation"] == 0 and s["tick"] <= 2]
    for row in d["steps"]:
        row.update(
            operation="native_step",
            action_space="native_continuous_state",
            behavior_policies=pop.behavior_policies,
            action_origins={a: "external-native-state-v1" for a in ("ego", "car")},
            override_actor_ids=[],
            policy_actions={a: [0.1, -0.2] for a in ("ego", "car")},
            executed_actions={a: [0.1, -0.2] for a in ("ego", "car")},
            native_targets={a: [0.0] * 10 for a in ("ego", "car")},
        )
    for row in d["frames"]:
        row["behavior_policies"] = pop.behavior_policies
    d["native_replay"] = dict(
        source_rollout="a" * 64,
        source_targets="b" * 64,
        capture_end_tick=2,
        full_physics_steps=4,
        trainable=False,
        rendered_rgb_used_for_actions=False,
    )
    return path, d


def test_native_capture_prefix_preserves_nonterminal_masks_and_continuous_controls(
    source, tmp_path
):
    path, d = native_source(source)
    write_source((path, d))
    out = tmp_path / "native"
    report = export_mtgs_episode(path, out)
    assert report["action_space"] == "native_continuous_state"
    assert report["recording_scope"] == "bounded_native_capture_not_complete_episode"
    physics = rows(out / "generation-000-physics.jsonl")
    assert physics[-1]["privileged"]["world_after"]["done"] is False
    assert physics[-1]["masks"]["truncated"] == [False, False]
    assert physics[0]["behavior"]["executed_actions"]["ego"] == [0.1, -0.2]
    assert not report["trainable"]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("native_replay"),
        lambda d: d["native_replay"].update(capture_end_tick=3),
        lambda d: d["native_replay"].update(full_physics_steps=2),
        lambda d: d["steps"][0]["native_targets"].pop("ego"),
        lambda d: d["steps"][0].update(operation="step"),
        lambda d: d["steps"][0]["executed_actions"].update(ego=1),
    ],
)
def test_native_export_rejects_forged_or_incomplete_prefix(source, tmp_path, change):
    path, d = native_source(source)
    change(d)
    write_source((path, d))
    with pytest.raises(ValueError):
        export_mtgs_episode(path, tmp_path / "bad")
