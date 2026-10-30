import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from driversi.rsi.experiments.reactive_prefetch import PhysicalPrefetch, physical_job


def test_bounded_prefetch_keeps_consumption_order_and_settles_surplus(tmp_path):
    barrier = threading.Barrier(2)

    def worker(job):
        barrier.wait(timeout=5)
        return job["value"]

    def factory(n):
        return ThreadPoolExecutor(max_workers=n)

    jobs = [(str(i), dict(render=False, output=str(tmp_path / str(i)), value=i)) for i in range(3)]
    with PhysicalPrefetch(tmp_path, 2, executor_factory=factory, worker=worker) as pool:
        pool.schedule(jobs)
        assert list(pool.futures) == ["0", "1"]
        assert pool.result("0") == 0
        assert "1" not in pool.consumed
    import json

    saved = json.loads((tmp_path / "physical-prefetch-status.json").read_text())
    assert saved["status"] == "settled" and saved["completed"] == 2
    assert saved["unconsumed"] == ["1"] and not saved["qualified_samples_certified"]


def test_prefetch_does_not_render_and_rejects_mutated_job(tmp_path):
    with PhysicalPrefetch(
        tmp_path, 2, executor_factory=lambda n: ThreadPoolExecutor(n), worker=lambda j: 0
    ) as pool:
        with pytest.raises(ValueError, match="forbidden"):
            pool.schedule([("x", dict(render=True))])
        pool.schedule([("x", dict(render=False, output=str(tmp_path), value=1))])
        with pytest.raises(ValueError, match="changed"):
            pool.schedule([("x", dict(render=False, output=str(tmp_path), value=2))])
        assert pool.result("x") == 0
    with pytest.raises(ValueError, match="only"):
        physical_job(dict(render=True))


def test_error_drain_and_serial_default(tmp_path):
    def bad(job):
        raise RuntimeError("actual worker failure")

    with PhysicalPrefetch(
        tmp_path, 2, executor_factory=lambda n: ThreadPoolExecutor(n), worker=bad
    ) as pool:
        pool.schedule([("x", dict(render=False, output=str(tmp_path)))])
        with pytest.raises(RuntimeError):
            pool.result("x")
        assert pool.errors == {"x": "RuntimeError"}
    with PhysicalPrefetch(tmp_path, 1) as serial:
        serial.schedule([])
        assert serial.executor is None
    for value in [0, 5, True]:
        with pytest.raises(ValueError):
            PhysicalPrefetch(tmp_path, value)


def test_real_prefetch_entry_creates_future_candidate_parent(tmp_path, monkeypatch):
    import json

    from driversi.rsi.experiments import reactive_generation

    monkeypatch.setattr(
        reactive_generation, "generate", lambda **job: {"status": "trajectory_ready"}
    )
    output = tmp_path / "not-yet-created" / "generation"
    assert not output.parent.exists()
    assert physical_job(dict(render=False, output=str(output)))["status"] == "trajectory_ready"
    assert (
        json.loads((output.parent / "physical-prefetch.json").read_text())["status"] == "completed"
    )
