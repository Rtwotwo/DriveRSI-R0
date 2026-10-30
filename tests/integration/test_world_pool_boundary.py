"""Joint batch rejection must precede mutation across all process shards."""
# ruff: noqa: E402
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")
from driversi.contracts.learning import JointDecision
from driversi.training.sampling.world_pool import WorldPool


def decisions(pool):
    return [JointDecision(frame.observation.key, frame.observation.actor_ids,
                          np.full(len(frame.observation.actor_ids), 7, dtype=np.int64), "frozen-test-v1")
            for _, frame in sorted(pool.current.items())]


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("invalid", ["empty", "short", "long", "stale_later", "reordered_later", "wrong_type"])
def test_entire_joint_batch_is_validated_before_any_world_advances(straight_scene, tmp_path, workers, invalid):
    path = tmp_path / "scene.npz"
    straight_scene.save(path)
    pool = WorldPool([path, path], workers=workers)
    try:
        before = pool.snapshot()
        submitted = decisions(pool)
        if invalid == "empty":
            submitted = []
        elif invalid == "short":
            submitted = submitted[:1]
        elif invalid == "long":
            submitted.append(submitted[-1])
        elif invalid == "stale_later":
            submitted[1] = replace(submitted[1], key=replace(submitted[1].key, generation=99))
        elif invalid == "reordered_later":
            submitted[1] = replace(submitted[1], actor_ids=tuple(reversed(submitted[1].actor_ids)))
        else:
            submitted[1] = None
        with pytest.raises(ValueError, match="Joint batch"):
            pool.step(submitted)
        assert not pool.closed
        assert pool.snapshot() == before
        assert [frame.observation.key.tick for frame in pool.current.values()] == [0, 0]
        assert len(pool.step(decisions(pool))) == 2
        assert [frame.observation.key.tick for frame in pool.current.values()] == [1, 1]
    finally:
        pool.close()


def test_post_dispatch_partial_failure_closes_pool(straight_scene, tmp_path, monkeypatch):
    path = tmp_path / "scene.npz"
    straight_scene.save(path)
    pool = WorldPool([path, path])
    original = pool.group.handle
    def fail_after_first_world(operation, payload):
        if operation == "step":
            original(operation, payload[:1])
            raise RuntimeError("injected second-world failure")
        return original(operation, payload)
    monkeypatch.setattr(pool.group, "handle", fail_after_first_world)
    submitted = decisions(pool)
    try:
        with pytest.raises(RuntimeError, match="second-world"):
            pool.step(submitted)
        assert pool.closed and not pool.current
        with pytest.raises(RuntimeError, match="closed"):
            pool.step(submitted)
    finally:
        pool.close()


def test_incomplete_worker_response_closes_pool(straight_scene, tmp_path, monkeypatch):
    path = tmp_path / "scene.npz"
    straight_scene.save(path)
    pool = WorldPool([path, path])
    original = pool.group.handle
    def lose_result(operation, payload):
        result = original(operation, payload)
        if operation == "step":
            result.pop(1)
        return result
    monkeypatch.setattr(pool.group, "handle", lose_result)
    try:
        with pytest.raises(RuntimeError, match="incomplete joint batch"):
            pool.step(decisions(pool))
        assert pool.closed and not pool.current
    finally:
        pool.close()
