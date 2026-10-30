import copy

import numpy as np
import pytest

from driversi.data.participants.background import configure_background
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.evaluation.metrics.source_support import source_entry_overlaps
from driversi.rendering.bev.replay import export_source_replay


def scene_with_entry():
    s = following_scene(1, 20)
    s.controlled[:] = False
    s.metadata.pop("task_goals_xy", None)
    s.presence = np.ones(s.poses.shape[:2], bool)
    s.presence[:, 2] = False
    s.presence[:3, 1] = False
    s.poses[:, 1] = s.poses[:, 0]
    s.velocities[:, :, 0] = 3
    return s


def test_entry_overlap_keeps_id_and_support_and_reports_source_not_blame():
    s = scene_with_entry()
    before = copy.deepcopy(s)
    audit = source_entry_overlaps(s)
    assert len(audit["events"]) == 1
    event = audit["events"][0]
    assert event["tick"] == 3 and event["actors"] == list(s.actor_ids[:2])
    assert event["entering_actors"] == [s.actor_ids[1]]
    assert event["causal_blame"] == "not_established"
    assert not audit["simulation_qualified"]
    np.testing.assert_array_equal(s.poses, before.poses)
    np.testing.assert_array_equal(s.presence, before.presence)


def test_reactive_lifetime_does_not_fake_source_occupancy():
    s = scene_with_entry()
    s.presence[2:, 0] = False
    s.metadata["reactive_background"] = {"source_presence": s.presence.tolist()}
    s.presence[:, 0] = True
    assert source_entry_overlaps(s)["events"] == []


def test_reentry_is_checked_but_simulation_later_contact_is_not_a_source_entry():
    s = scene_with_entry()
    s.presence[4:6, 1] = False
    assert [e["tick"] for e in source_entry_overlaps(s)["events"]] == [3, 6]
    s.poses[3, 1, 1] += 20
    s.poses[6, 1, 1] += 20
    assert source_entry_overlaps(s)["events"] == []


def test_wide_objects_are_not_lost_in_source_broadphase():
    s = scene_with_entry()
    s.dimensions[0, :2] = [4, 2]
    s.dimensions[1, :2] = [0.5, 12]
    s.poses[:, 1, 1] = s.poses[:, 0, 1] + 5
    assert len(source_entry_overlaps(s)["events"]) == 1


def test_modified_fleet_cannot_be_exported_as_source(tmp_path):
    s = scene_with_entry()
    s.metadata["source_dataset"] = "nuplan_db"
    s.controlled[2] = True
    s.presence[:, 2] = True
    actor = s.actor_ids[0]
    p = s.poses[0, 0, :2]
    configured = configure_background(s, {actor: {"points": [p.tolist(), (p+[10, 0]).tolist()],
                                                 "desired_speed": 1, "navigation_source": "explicit_task"}})
    with pytest.raises(ValueError, match="unmodified source"):
        export_source_replay(configured, tmp_path / "source.html")
    assert not (tmp_path / "source.html").exists()


def test_source_geometry_audit_does_not_relabel_modified_spawn_velocity():
    s = scene_with_entry()
    expected = source_entry_overlaps(s)
    s.velocities[3, 1] = [18, 0, 0]
    assert source_entry_overlaps(s) == expected


def test_report_preserves_hazards_without_lead_and_excludes_inactive_or_replayed():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[3] / "scripts/diagnostics/diagnose_recorded_blockage.py"
    spec = importlib.util.spec_from_file_location("blockage_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = {"actor": "ego", "lead_id": None,
               "traffic_hazards": [{"actor": "other", "reason": "swept"}]}
    rows = module.dependency_intervals([{}, {}, {}, {}],
                                      [[command], [command], [command],
                                       [{**command, "controller": "baseline_action_replay"}]],
                                      np.array([[True], [True], [False], [True]]), ["ego"])
    assert rows == [{"actor": "ego", "target": "other", "reason": "swept",
                     "start_tick": 0, "end_tick_exclusive": 2}]
