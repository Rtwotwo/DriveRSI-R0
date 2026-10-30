import copy
import json

import numpy as np
import pytest

from driversi.evaluation.diagnosis.anchor_diagnosis import diagnose_anchor
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.sim.execution.continuous import ContinuousSession


def long_scene():
    s = following_scene(1, 20)
    s.times = np.arange(301) * 0.1
    s.poses = np.zeros((301, 3, 4))
    s.poses[:, :, 0] = s.times[:, None] + [0, 24, 0]
    s.poses[:, :, 1] = [0, 0, 80]
    s.velocities = np.zeros((301, 3, 3))
    s.velocities[:, :, 0] = 1
    return s


class MemoryPolicy:
    def describe(self):
        return {"version": "test-memory-v1"}

    def initial_memories(self, ids):
        return {a: {"calls": 0, "draws": 0} for a in ids}

    def infer(self, obs, history, memories, rngs):
        assert history[-1].step == obs.step
        for a in obs.actor_ids:
            memories[a]["calls"] += 1
            memories[a]["draws"] += int(rngs[a].integers(0, 100))
        return dict.fromkeys(obs.actor_ids, 7), memories, []


def test_chunking_and_post9s_checkpoint_preserve_world_memory_history_rng():
    a = ContinuousSession(long_scene(), MemoryPolicy(), seed=42)
    b = ContinuousSession(long_scene(), MemoryPolicy(), seed=42)
    try:
        a.advance(275)
        for n in (90, 90, 95):
            b.advance(n)
        assert a.snapshot() == b.snapshot()
        assert b.memories["lead"]["calls"] == 275
        assert [o.step for o in b.history] == list(range(256, 276))
        checkpoint = json.loads(json.dumps(b.snapshot()))
        b.advance(25)
        end = b.snapshot()
        b.restore(checkpoint)
        b.advance(25)
        assert end == b.snapshot()
    finally:
        a.close()
        b.close()


def test_invalid_restore_does_not_change_session_and_future_history_rejected():
    s = ContinuousSession(long_scene(), MemoryPolicy())
    try:
        s.advance(95)
        before = s.snapshot()
        bad = copy.deepcopy(before)
        bad["history"][-1]["step"] += 1
        with pytest.raises(ValueError, match="contiguous"):
            s.restore(bad)
        assert before == s.snapshot()
        bad = copy.deepcopy(before)
        bad["history"][-1]["features"][0][0] += 1
        with pytest.raises(ValueError, match="terminal"):
            s.restore(bad)
        assert before == s.snapshot()
    finally:
        s.close()


def test_same_anchor_branches_preserve_parent_and_do_not_invent_model_blame():
    s = ContinuousSession(long_scene(), MemoryPolicy())
    try:
        s.advance(100)
        before = s.snapshot()
        report, records = diagnose_anchor(s, before, owner="follower", steps=20)
        assert report["repeat_exact"] and report["action_changed"]
        assert report["blame"] == "not_established" and report["official_PDM"] is None
        assert report["anchor_tick"] == 100
        assert s.snapshot() == before
        assert records["baseline"].lifecycle[0]["tick"] == 100
    finally:
        s.close()


def test_failed_policy_call_rolls_back_rng_world_and_memory():
    class Bad(MemoryPolicy):
        def infer(self, *args):
            actions, memory, diagnostics = super().infer(*args)
            actions.pop("lead")
            return actions, memory, diagnostics

    s = ContinuousSession(long_scene(), Bad())
    try:
        before = s.snapshot()
        with pytest.raises(ValueError):
            s.step()
        assert before == s.snapshot()
    finally:
        s.close()


def test_override_keeps_policy_proposal_and_actual_execution_separate():
    class Diagnosed(MemoryPolicy):
        def infer(self, *args):
            actions, memory, _ = super().infer(*args)
            return actions, memory, [{"actor": a, "action": v} for a, v in actions.items()]

    s = ContinuousSession(long_scene(), Diagnosed())
    try:
        tr = s.step({"follower": 1})
        cmd = next(c for c in s.last_decision["diagnostics"] if c["actor"] == "follower")
        assert cmd["action"] == 1 and cmd["policy_proposed_action"] == 7
        assert cmd["intervention"] == "explicit_same_anchor_action_override"
        assert tr.observation.step == 1
    finally:
        s.close()


def test_entry_or_anchor_overlap_is_not_reported_as_model_fault():
    scene = long_scene()
    scene.poses[:, 1, :2] = scene.poses[:, 0, :2]
    s = ContinuousSession(scene, MemoryPolicy())
    try:
        report, _ = diagnose_anchor(s, s.snapshot(), owner="follower", steps=10)
        assert "anchor_or_entry_overlap_confounder" in report["evidence_flags"]
        assert report["blame"] == "not_established"
        assert report["branches"]["baseline"]["contacts"]["owner_contact_ticks"] > 0
        assert any(
            e["source_reference_overlap"]
            for e in report["branches"]["baseline"]["contact_onset_evidence"]
        )
    finally:
        s.close()


def test_map_extraction_covers_entire_present_path_not_only_first_window(monkeypatch, tmp_path):
    from driversi.data.road_network import maps

    scene = long_scene()
    scene.presence = np.ones(scene.poses.shape[:2], bool)
    scene.presence[:, 2] = False
    scene.controlled[2] = False
    scene.metadata["task_goals_xy"].pop(scene.actor_ids[2])
    scene.poses[:, 2, :2] = 1e7

    def extract(name, positions, origin, root, **kwargs):
        assert np.max(positions[:, 0]) == 54
        assert np.max(positions[:, 1]) == 0
        assert len(positions) == 602
        return scene.road_points, scene.road_offsets, scene.road_types, {"map_complete": True}

    monkeypatch.setattr(maps, "extract_map", extract)
    maps.attach_nuplan_map(scene, tmp_path)


def test_json_restore_preserves_history_numeric_dtypes():
    class Typed(MemoryPolicy):
        def infer(self, obs, history, memories, rngs):
            for h in history:
                assert h.features.dtype == obs.features.dtype
                assert h.metric.ego.dtype == obs.metric.ego.dtype
                assert h.metric.neighbors.dtype == obs.metric.neighbors.dtype
            return super().infer(obs, history, memories, rngs)

    s = ContinuousSession(long_scene(), Typed())
    try:
        s.advance(95)
        saved = json.loads(json.dumps(s.snapshot()))
        s.advance(10)
        end = s.snapshot()
        s.restore(saved)
        s.advance(10)
        assert s.snapshot() == end
    finally:
        s.close()


def test_map_crop_checks_whole_vehicle_footprint_and_other_actors():
    from driversi.contracts.map_support import crop_support

    scene = long_scene()
    scene.metadata["map_bounds_global"] = [-30, -15, 250, 100]
    scene.metadata["origin_xyz"] = [0, 0, 0]
    state = np.zeros((3, 13))
    state[:, 0] = [247.6, 20, 30]
    scene.dimensions[0, :2] = [5.176, 2.3]
    report = crop_support(scene, state, np.ones(3, bool))
    assert report["status"] == "boundary_warning"
    assert report["actors"][0]["outside_crop"]
    assert report["actors"][0]["actor"] == scene.actor_ids[0]


def test_policy_rng_mapping_mutation_cannot_break_transaction():
    class Invalid(MemoryPolicy):
        def infer(self, obs, history, memories, rngs):
            rngs.pop(obs.actor_ids[0])
            raise ValueError("policy failure")

    s = ContinuousSession(long_scene(), Invalid())
    try:
        before = s.snapshot()
        with pytest.raises(ValueError, match="policy failure"):
            s.step()
        assert s.snapshot() == before
    finally:
        s.close()


def test_map_support_error_rolls_back_step_and_policy_state():
    scene = long_scene()
    scene.metadata["map_bounds_global"] = [0, 1, 2]
    s = ContinuousSession(scene, MemoryPolicy())
    try:
        before = s.snapshot()
        with pytest.raises(ValueError, match="map crop"):
            s.step()
        assert before == s.snapshot()
    finally:
        s.close()


def test_collision_crosscheck_keeps_wide_objects_in_broadphase():
    from driversi.evaluation.diagnosis.anchor_diagnosis import geometry_contacts

    scene = long_scene()
    scene.dimensions[0, :2] = [4, 2]
    scene.dimensions[1, :2] = [0.5, 12]
    state = np.zeros((3, 13))
    state[:, 1] = [0, 5, 80]
    pair = tuple(sorted(scene.actor_ids[:2]))
    assert pair in geometry_contacts(scene, state, np.ones(3, bool), scene.actor_ids[0])


def test_compressed_viewer_preserves_exact_payload(tmp_path):
    import base64, gzip, re
    from driversi.rendering.bev.replay import write_viewer

    data = {
        "scene": {"metadata": {}},
        "run": {"states": [[[1.23456789, 2]]]},
        "unsafe": "</script>",
    }
    out = tmp_path / "view.html"
    write_viewer(data, out, compress_payload=True)
    text = out.read_text()
    encoded = re.search(r'<script[^>]*id="replay-data"[^>]*>(.*?)</script>', text, re.S).group(1)
    assert json.loads(gzip.decompress(base64.b64decode(encoded))) == data
    assert "DecompressionStream" in text and '<script type="module">' in text
