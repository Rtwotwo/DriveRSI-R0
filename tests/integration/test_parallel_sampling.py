# ruff: noqa: E402
import copy
import os
import signal

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from tests.support.diagnostic_agents import ActorCritic
from driversi.training.sampling.collector import Collector, rollout_nbytes
from driversi.training.ppo import update


def scenes(straight_scene, tmp_path):
    paths = []
    for i in range(3):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = f"independent-log-{i}"
        # Distinct lengths exercise staggered resets with stable parent-side RNG.
        size = 11 + 10 * i
        scene.times = scene.times[:size]
        scene.poses = scene.poses[:size]
        scene.velocities = scene.velocities[:size]
        scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
        path = tmp_path / f"scene-{i}.npz"
        scene.save(path)
        paths.append(path)
    return paths


def assert_arrays(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_process_shards_match_serial_across_resets_update_and_restore(straight_scene, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    model = ActorCritic(hidden_sizes=(16,))
    paths = scenes(straight_scene, tmp_path)
    serial = Collector(paths, num_worlds=4, seed=11, scene_cache_size=1)
    parallel = Collector(paths, num_worlds=4, seed=11, workers=2, scene_cache_size=1)
    try:
        rng = torch.get_rng_state()
        expected = serial.collect(model, 36)
        torch.set_rng_state(rng)
        actual = parallel.collect(model, 36)
        assert_arrays(actual, expected)
        assert (actual["episode"] > 0).any()
        assert actual["world_actor_count"].shape == (36, 4)
        assert parallel.last_metrics["rollout_array_bytes"] == rollout_nbytes(36, 4)
        assert all(item["cached_scenes"] <= 1 for item in parallel.pool.stats())
        serial_model, parallel_model = copy.deepcopy(model), copy.deepcopy(model)
        optimizer_a = torch.optim.Adam(serial_model.parameters())
        optimizer_b = torch.optim.Adam(parallel_model.parameters())
        rng = torch.get_rng_state()
        update(serial_model, optimizer_a, expected, epochs=1, minibatch=32)
        torch.set_rng_state(rng)
        update(parallel_model, optimizer_b, actual, epochs=1, minibatch=32)
        for key, value in serial_model.state_dict().items():
            assert torch.equal(value, parallel_model.state_dict()[key])
        snapshot = parallel.snapshot()
        rng = torch.get_rng_state()
        suffix = parallel.collect(model, 18)
        parallel.restore(snapshot)
        torch.set_rng_state(rng)
        assert_arrays(parallel.collect(model, 18), suffix)
        # A sampler snapshot is topology-independent; old one-process continuation works.
        serial.restore(snapshot)
        torch.set_rng_state(rng)
        assert_arrays(serial.collect(model, 18), suffix)
    finally:
        serial.close()
        clients = parallel.pool.clients.copy()
        parallel.close()
        assert all(not process.is_alive() for _, process, _ in clients)


def test_rollout_budget_rejects_before_allocating_or_advancing(straight_scene, tmp_path):
    collector = Collector(scenes(straight_scene, tmp_path), num_worlds=2, max_rollout_mib=1)
    try:
        before = collector.snapshot()
        with pytest.raises(MemoryError, match="Rollout needs"):
            collector.collect(None, 30)
        assert collector.snapshot() == before
        assert not collector.pool.closed
    finally:
        collector.close()


def test_dead_worker_aborts_whole_batch_and_releases_children(straight_scene, tmp_path):
    collector = Collector(scenes(straight_scene, tmp_path), num_worlds=2, workers=2)
    clients = collector.pool.clients.copy()
    clients[0][1].terminate()
    clients[0][1].join(timeout=5)
    try:
        with pytest.raises((RuntimeError, OSError, EOFError)):
            collector.collect(ActorCritic(hidden_sizes=(8,)), 2)
        assert collector.pool.closed
        assert all(not process.is_alive() for _, process, _ in clients)
        with pytest.raises(RuntimeError, match="closed"):
            collector.collect(ActorCritic(hidden_sizes=(8,)), 2)
    finally:
        collector.close()


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="Linux worker timeout probe")
def test_stalled_worker_times_out_instead_of_hanging(straight_scene, tmp_path):
    collector = Collector(scenes(straight_scene, tmp_path), num_worlds=1, workers=1)
    process = collector.pool.clients[0][1]
    collector.pool.timeout = .2
    os.kill(process.pid, signal.SIGSTOP)
    try:
        with pytest.raises(TimeoutError):
            collector.collect(ActorCritic(hidden_sizes=(8,)), 1)
        assert collector.pool.closed and not process.is_alive()
    finally:
        collector.close()


def test_initially_bad_scene_is_rejected_in_worker(straight_scene, tmp_path):
    scene = copy.deepcopy(straight_scene)
    scene.metadata["task_goals_xy"] = {"a": [0, 0], "b": [0, 8]}
    path = tmp_path / "bad.npz"
    scene.save(path)
    with pytest.raises(RuntimeError, match="endpoint goal|no eligible controlled"):
        Collector([path], num_worlds=2, workers=2)


def test_capacity_command_counts_valid_samples_and_cleans_temporary_recordings(straight_scene, tmp_path):
    import json

    from driversi.cli import main

    paths = scenes(straight_scene, tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"train": [{"path": str(paths[0])}],
                                 "evaluation": [{"path": str(paths[1])}]}))
    output = tmp_path / "capacity.json"
    assert main(["benchmark-sampling", "--suite", str(suite), "--output", str(output),
                 "--worlds", "2", "--workers", "1", "--steps", "3", "--batches", "2",
                 "--record"]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "complete" and report["training"] is False
    assert report["world_steps"] == 12 and report["valid_agent_transitions"] == 24
    assert report["observed_scene_count"] == 1 and report["observed_scene_indices"] == [0]
    assert sum(report["valid_transitions_by_map"].values()) == 24
    assert report["observed_source_logs"] == ["independent-log-0"]
    assert all(row["slot_utilization"] == 2 / 32 for row in report["measurements"])
    assert report["valid_transitions_per_second_with_recording"] <= report["valid_transitions_per_second"]
    assert not list(tmp_path.glob("sampling-record-*"))


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="Linux worker blocked-send probe")
def test_stalled_worker_cannot_hang_large_request(straight_scene, tmp_path):
    import socket
    import time

    collector = Collector(scenes(straight_scene, tmp_path), num_worlds=1, workers=1)
    channel, process, _ = collector.pool.clients[0]
    channel.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    collector.pool.timeout = .05
    os.kill(process.pid, signal.SIGSTOP)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="sending"):
            collector.pool.request("stats", [(0, bytes(2 ** 20))])
        assert time.monotonic() - start < 3
        assert collector.pool.closed and not process.is_alive()
    finally:
        collector.close()


def test_capacity_probe_honors_minimum_time_and_retains_actual_batch_count(straight_scene, tmp_path):
    import json

    from driversi.evaluation.benchmarks.sampling import benchmark_sampling

    paths = scenes(straight_scene, tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"train": [{"path": str(paths[0])}],
                                 "evaluation": [{"path": str(paths[1])}]}))
    report = benchmark_sampling(suite, tmp_path / "capacity.json", agent_config=__import__("pathlib").Path(__file__).resolve().parents[2] / "tests/fixtures/legacy/agents/mlp.yaml", worlds=1, steps=1,
                                batches=1, min_seconds=.1)
    assert report["probe_wall_seconds"] >= .1
    assert report["batches"] >= report["requested_batches"] == 1
    assert report["world_steps"] == report["batches"]
    assert report["valid_agent_transitions"] == sum(report["valid_transitions_by_map"].values())
    with pytest.raises(ValueError, match="min_seconds"):
        benchmark_sampling(suite, tmp_path / "invalid.json", agent_config=__import__("pathlib").Path(__file__).resolve().parents[2] / "tests/fixtures/legacy/agents/mlp.yaml", min_seconds=float("nan"))
