"""Surveyed static existence is distinct from annotation discovery, never dynamic backfill."""

import copy
import hashlib

import numpy as np
import pytest

from driversi.contracts.scene import OTHER_OBJECT
from driversi.data.participants.traffic_continuity import configure_traffic_continuity
from driversi.data.scenes.static_context import (
    POLICY,
    apply_static_scene_context,
    validate_static_scene_context,
)
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


class Graph:
    def validate_scene(self, scene):
        scene.validate()


def case(straight_scene, tmp_path, category="generic_object"):
    scene = copy.deepcopy(straight_scene)
    scene.kinds[2] = OTHER_OBJECT
    scene.poses[:, 2] = [10, -8, 0, 0]
    scene.velocities[:, 2] = [.002, 0, 0]
    scene.presence = np.ones(scene.poses.shape[:2], bool)
    scene.presence[:5, 2] = False
    scene.metadata.update(source_start_frame=0, source_actor_kinds={"replay": category},
        first_observations={"replay": {"velocity_world": [.002, 0, 0]}},
        origin_xyz=[0, 0, 0], map_bounds_global=[-100, -100, 100, 100])
    source = configure_traffic_continuity(scene, Graph())
    image = tmp_path / "initial.jpg"
    image.write_bytes(b"private-reviewed-real-image-fixture")
    frames = [
        {"token": "initial", "timestamp": 0, "anns": {"track_tokens": []},
         "cams": {"CAM_B0": {"data_path": "initial.jpg"}}},
        {"token": "first", "timestamp": 500000,
         "ego2global_translation": [0, 0, 0], "ego2global_rotation": [1, 0, 0, 0],
         "anns": {"track_tokens": ["replay"], "gt_names": [category],
                  "gt_boxes": [[10, -8, 0, *scene.dimensions[2], 0]],
                  "gt_velocity_3d": [[.002, 0, 0]]}},
    ]
    manifest = {"version": 1, "policy": POLICY, "source_log": source.metadata["source_log"],
        "actors": {"replay": {"immutable_visible_at_scene_start_reviewed": True,
            "reviewer": "pixel-review-fixture", "evidence_refs": [{"raw_index": 0,
                "camera": "CAM_B0", "path": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}}}
    return source, frames, manifest


def test_actual_world_initialization_keeps_raw_support_and_restores(straight_scene, tmp_path):
    source, frames, manifest = case(straight_scene, tmp_path)
    scene = apply_static_scene_context(source, frames, manifest)
    assert not source.presence[0, 2]
    assert scene.presence[:, 2].all()
    assert scene.metadata["reactive_background"]["source_presence"] == source.metadata["reactive_background"]["source_presence"]
    assert scene.metadata["first_observations"] == source.metadata["first_observations"]
    assert validate_static_scene_context(scene, frames)["actors"]["replay"]["source_first_seen_tick"] == 5
    with PufferDriveWorld(scene) as world:
        initial = world.snapshot()
        assert world.presence()[2]
        position = world.state()[2, :4].copy()
        for _ in range(20):
            world.step([7, 7])
            np.testing.assert_array_equal(world.state()[2, :4], position)
        final = world.snapshot()
        world.restore(initial)
        for _ in range(20):
            world.step([7, 7])
        assert world.snapshot() == final


@pytest.mark.parametrize("bad", ["unknown", "vehicle", "pedestrian", "bicycle"])
def test_no_dynamic_or_unknown_future_backfill(straight_scene, tmp_path, bad):
    source, frames, manifest = case(straight_scene, tmp_path, bad)
    with pytest.raises(ValueError, match="native static"):
        apply_static_scene_context(source, frames, manifest)


@pytest.mark.parametrize("failure", ["tamper", "no_review", "no_initial", "moving", "missing_velocity", "overlap"])
def test_prior_rejects_unsupported_evidence_or_geometry(straight_scene, tmp_path, failure):
    source, frames, manifest = case(straight_scene, tmp_path)
    req = manifest["actors"]["replay"]
    if failure == "tamper":
        req["evidence_refs"][0]["sha256"] = "0" * 64
    elif failure == "no_review":
        req["immutable_visible_at_scene_start_reviewed"] = False
    elif failure == "no_initial":
        req["evidence_refs"] = []
    elif failure == "moving":
        frames[1]["anns"]["gt_velocity_3d"] = [[1, 0, 0]]
    elif failure == "missing_velocity":
        del frames[1]["anns"]["gt_velocity_3d"]
    else:
        frames[1]["anns"]["gt_boxes"][0][:2] = source.poses[0, 0, :2].tolist()
    with pytest.raises(ValueError):
        apply_static_scene_context(source, frames, manifest)


def test_render_revalidation_rejects_modified_source_evidence(straight_scene, tmp_path):
    source, frames, manifest = case(straight_scene, tmp_path)
    scene = apply_static_scene_context(source, frames, manifest)
    frames[1]["anns"]["gt_boxes"][0][0] += .1
    with pytest.raises(ValueError, match="receipt differs"):
        validate_static_scene_context(scene, frames)
