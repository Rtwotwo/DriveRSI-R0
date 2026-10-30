import copy
from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.rendering.test_visual_sampling import sampling_session

from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession
from driversi.sim.persistence.session_checkpoint import SessionCheckpoint
from driversi.sim.sensors.visual_sampling import ManagedVisualSampler


@pytest.fixture
def session(tmp_path, monkeypatch):
    original = sampling_session(tmp_path)
    sampler = ManagedVisualSampler(original, ("ego", "car"))
    original.visual_sampler = sampler
    original.capture()
    sampler.advance()
    actual = MTGSWorldSession.__new__(MTGSWorldSession)
    actual.__dict__ = original.__dict__.copy()
    sampler.session = actual
    # No geometry renderer in this fixture; real native snapshots are exercised
    # by test_worker and the full MTGS renderer by the server pilot.
    del actual.__dict__["advance"]
    actual.manifest.update(inputs={}, assets={}, backend={}, camera_ids=["CAM_F0", "car__virtual_front"],
                           height_mode="source-time", behavior_policies={})
    for row in actual.manifest["frames"]:
        row["capture_seconds"] = .01
    actual.latest["capture_seconds"] = .01
    for row in actual.manifest["steps"]:
        row["advance_seconds"] = .001
    actual.current["behavior_policies"] = {}
    actual._source_images, actual._samples, actual.map, actual._worker_settings = {}, {}, {}, {}
    initial = copy.deepcopy(actual.current)
    initial["tick"] = 0

    class FakeWorker:
        closed = False
        def __init__(self, **kwargs):
            self.current = copy.deepcopy(initial)
        def request(self, operation, **kwargs):
            if operation == "snapshot":
                return {"snapshot": {"tick": self.current["tick"], "state": copy.deepcopy(self.current)}}
            if operation == "restore":
                self.current = copy.deepcopy(kwargs["snapshot"]["state"])
                return copy.deepcopy(self.current)
            raise AssertionError(operation)
        def close(self):
            self.closed = True
    actual.runtime = SimpleNamespace(scale=.5)
    actual.worker = FakeWorker()
    actual.worker.current = copy.deepcopy(actual.current)
    monkeypatch.setattr("driversi.sim.workers.worker_client.WorldWorker", FakeWorker)
    return actual


def test_disk_checkpoint_fork_owns_actor_state_and_prefix(session, tmp_path):
    checkpoint = session.snapshot()
    path = tmp_path/"state.npz"
    checkpoint.save(path)
    loaded = SessionCheckpoint.load(path)
    child = session.fork(tmp_path/"child", loaded)
    try:
        assert child.current["tick"] == session.current["tick"] == 1
        assert child.session_id != session.session_id
        assert child.visual_sampler.held_actions == session.visual_sampler.held_actions
        assert child.visual_sampler.held_record["identity"]["session_id"] == child.session_id
        assert child.manifest["decisions"][0]["identity"]["session_id"] == child.session_id
        assert session.manifest["decisions"][0]["identity"]["session_id"] == session.session_id
        assert child.manifest["branch_lineage"][-1]["tick"] == 1
        assert child.manifest["performance"]["branch_suffix"]["steps"] == 0
        child.visual_sampler.memories[0, 0] = -10
        assert session.visual_sampler.memories[0, 0] != -10
        assert not child._sensor_buffers["CAM_F0"]["rgb"].flags.writeable
        source = session.latest["cameras"][0]["image"]
        (child.output/source).write_bytes(b"changed child copy")
        assert (session.output/source).read_bytes() != b"changed child copy"
        assert child.worker is not session.worker
    finally:
        child.close()
    assert not session.worker.closed


def test_checkpoint_rejects_changed_dependency_before_start(session, tmp_path):
    saved = session.snapshot()
    image = session.latest["cameras"][0]["image"]
    (session.output/image).write_bytes(b"changed")
    with pytest.raises(ValueError, match="dependency"):
        session.fork(tmp_path/"bad", saved)
    assert not (tmp_path/"bad").exists()


def test_checkpoint_rejects_context_change_and_existing_output(session, tmp_path):
    saved = session.snapshot()
    saved.payload["context"]["sensor_stride"] = 1
    with pytest.raises(ValueError, match="context"):
        session.fork(tmp_path/"bad", saved)
    assert not (tmp_path/"bad").exists()
    with pytest.raises(FileExistsError):
        session.fork(tmp_path)


def test_checkpoint_cannot_overwrite_and_contains_no_pickle(session, tmp_path):
    saved = session.snapshot()
    path = tmp_path/"checkpoint.npz"
    saved.save(path)
    with pytest.raises(FileExistsError):
        saved.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[k].dtype.hasobject for k in archive.files)


def test_structured_checkpoint_does_not_create_a_visual_policy(session, tmp_path):
    session.visual_sampler = None
    session.manifest["sampling"] = {"decision_mode": "structured", "enabled": False}
    session.manifest["decisions"] = []
    child = session.fork(tmp_path/"structured")
    try:
        assert child.visual_sampler is None
        assert child.manifest["sampling"]["decision_mode"] == "structured"
    finally:
        child.close()


def test_render_scale_change_is_rejected_before_fork_creation(session, tmp_path):
    saved = session.snapshot()
    session.runtime = type(session.runtime)(scale=.25)
    with pytest.raises(ValueError, match="context"):
        session.fork(tmp_path/"wrong-resolution", saved)
    assert not (tmp_path/"wrong-resolution").exists()


@pytest.mark.parametrize("field", ["surface", "alpha", "depth"])
def test_surface_evidence_is_copied_and_tampering_blocks_fork(session, tmp_path, field):
    name = "diagnostic-"+field+".bin"
    (session.output/name).write_bytes(b"source diagnostic")
    session.manifest["frames"][-1]["cameras"][0][field] = name
    saved = session.snapshot()
    child = session.fork(tmp_path/("surface-"+field), saved)
    try:
        assert (child.output/name).read_bytes() == b"source diagnostic"
        (child.output/name).write_bytes(b"child only")
        assert (session.output/name).read_bytes() == b"source diagnostic"
    finally:
        child.close()
    (session.output/name).write_bytes(b"tampered diagnostic")
    with pytest.raises(ValueError, match="dependency"):
        session.fork(tmp_path/("bad-"+field), saved)
    assert not (tmp_path/("bad-"+field)).exists()
