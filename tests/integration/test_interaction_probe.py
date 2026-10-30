"""Causal checks must distinguish execution, response and missing evidence."""

import json

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.contracts.road import RoadGraph
from driversi.evaluation.benchmarks.interaction import probe_interaction
from driversi.evaluation.benchmarks.real_scenes import verify_replay
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.rendering.bev.replay import export_replay

pytestmark = pytest.mark.native


def graph_for(scene):
    return RoadGraph(
        {"schema_version": 1, "map_name": "straight", "map_version": None,
         "origin_xyz": [0, 0, 0], "bounds_local": [-30, -15, 250, 100]},
        {f"lane:{y}": {"kind": "lane", "points": [[-30, y], [250, y]],
                       "outgoing": [], "speed_limit_mps": None} for y in (0, 80)})


@pytest.mark.parametrize("duration", [3., 6., 9.])
def test_same_probe_contract_on_different_scene_lengths(tmp_path, duration):
    scene = following_scene(8, 25, duration=duration)
    report = probe_interaction(scene, graph_for(scene), tmp_path / "probe")
    assert report["status"] == "verified", report
    assert report["brake_intervention"]["target_mps2"] == -3.0
    assert report["observer_id"] == "follower" and report["lead_id"] == "lead"
    assert all(report["checks"].values())
    assert report["metrics"]["desired_response_delay_s"] == pytest.approx(.1)
    assert .2 <= report["metrics"]["actual_response_delay_s"] <= 2
    for name, filename in report["recordings"].items():
        path = tmp_path / "probe" / filename
        assert all(verify_replay(scene, path).values())
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            assert metadata["protocol"]["planned_seconds"] == duration
            intervention = metadata["protocol"]["brake_intervention"]
            assert (intervention is not None) == (name in {"lead_brake", "action_replay_brake"})
            if intervention is not None:
                assert intervention == report["brake_intervention"]
            assert len(data["states"]) == round(duration * 10) + 1
        assert export_replay(scene, path, tmp_path / (name + ".html"))["frames"] > 1


def test_short_window_is_insufficient_without_mutating_or_padding(tmp_path):
    scene = following_scene(8, 25, duration=2.)
    original = scene.poses.copy()
    report = probe_interaction(scene, graph_for(scene), tmp_path / "probe")
    assert report["status"] == "evidence_insufficient"
    assert not report["recordings"]
    np.testing.assert_array_equal(scene.poses, original)


def test_two_controlled_cars_are_not_automatically_interacting(tmp_path):
    scene = following_scene(8, 25, replay=True)
    report = probe_interaction(scene, graph_for(scene), tmp_path / "probe")
    assert scene.controlled.sum() == 2
    assert report["status"] == "not_applicable"
    assert report["candidate_pairs"] == []
    assert not report["recordings"]


def test_probe_does_not_overwrite_formal_evidence(tmp_path):
    scene = following_scene(8, 25, duration=2)
    output = tmp_path / "probe"
    probe_interaction(scene, graph_for(scene), output)
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe_interaction(scene, graph_for(scene), output)
    assert (output / "report.json").read_bytes() == before


def test_early_stop_is_not_interpreted_as_no_collision_or_no_response(tmp_path):
    scene = following_scene(8, 25)
    scene.metadata["task_goals_xy"]["follower"] = [6, 0]
    report = probe_interaction(scene, graph_for(scene), tmp_path / "probe")
    assert report["status"] == "evidence_insufficient", report
    assert report["reason"] in {"episode_ended_before_intervention",
                                 "no_common_precontact_interval",
                                 "contact_or_actor_stop_censored_response_budget"}
