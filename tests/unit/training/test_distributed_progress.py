# ruff: noqa: E402
"""Progress bounds each silent evaluation segment, not the whole root workload."""
import copy
import json
import socket
import time
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from driversi.training.legacy.distribution import DistributedContext


def progress_worker(rank, port, directory, mode):
    torch.set_num_threads(1)
    timeout_seconds = 3
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                           rank=rank, world_size=2, timeout=timedelta(seconds=timeout_seconds))
    try:
        context = DistributedContext(rank, 2, torch.device("cpu"))
        markers = []
        def evaluation(progress):
            assert rank == 0
            if mode == "error_before":
                raise ValueError("evaluation failed before first scene")
            if mode == "complete":
                for scene in range(8):
                    time.sleep(.5)
                    markers.append(scene + 1)
                    progress(scene + 1)
                return {"completed": 8}
            progress(1)
            if mode == "interrupt":
                raise KeyboardInterrupt("root evaluation interrupted")
            raise ValueError("evaluation failed after progress")
        begin = time.monotonic()
        try:
            result = context.root_with_progress("scene evaluation", evaluation)
            failure = None
        except RuntimeError as error:
            result, failure = None, str(error)
        elapsed = time.monotonic() - begin
        # Coordinated application errors must not poison the process group.
        tensor = torch.ones(1)
        dist.all_reduce(tensor)
        assert tensor.item() == 2
        payload = {"result": result, "failure": failure, "elapsed": elapsed,
                   "markers": markers, "timeout_seconds": timeout_seconds}
        (Path(directory) / f"progress-{rank}.json").write_text(json.dumps(payload))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["complete", "error_before", "error_after", "interrupt"])
def test_root_progress_avoids_total_duration_timeout_and_coordinates_errors(tmp_path, mode):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(progress_worker, args=(port, str(tmp_path), mode), nprocs=2, join=True)
    root, peer = [json.loads((tmp_path / f"progress-{rank}.json").read_text()) for rank in range(2)]
    assert peer["markers"] == []
    if mode == "complete":
        assert root["result"] == {"completed": 8}
        assert peer["result"] is None
        assert root["markers"] == list(range(1, 9))
        for row in (root, peer):
            assert row["failure"] is None
            assert row["elapsed"] > row["timeout_seconds"]
    else:
        assert root["failure"] == peer["failure"]
        assert "distributed stage scene evaluation failed: rank 0" in root["failure"]
        assert "KeyboardInterrupt" in root["failure"] if mode == "interrupt" else "ValueError" in root["failure"]


def test_evaluators_publish_one_callback_per_completed_scene(straight_scene, tmp_path):
    pytest.importorskip("driversi.sim._native")
    from tests.support.diagnostic_agents import ActorCritic
    from tests.support.diagnostic_config import DEFAULT_AGENT, load_inference
    from driversi.evaluation.benchmarks.matchup import evaluate_matchups
    from driversi.evaluation.search.teacher import evaluate
    from driversi.training.self_play.opponents import FrozenOpponentPool

    training, evaluation = [], []
    for index in range(3):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = f"progress-log-{index}"
        scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
        path = tmp_path / f"scene-{index}.npz"
        scene.save(path)
        (training if index == 0 else evaluation).append(path)
    agent = copy.deepcopy(DEFAULT_AGENT)
    agent["parameters"] = {"hidden_sizes": [8]}
    model = ActorCritic(hidden_sizes=(8,))
    checkpoint = tmp_path / "policy.npz"
    model.export(checkpoint, "progress-policy-v1")
    policy = load_inference(checkpoint, agent)
    pool = FrozenOpponentPool({"schema_version": 1, "policies": [
        {"id": f"opponent-{index}", "checkpoint": str(checkpoint), "agent_config": agent,
         "training_paths": [str(path) for path in training]} for index in range(2)]})
    episodes = []
    baseline = evaluate(policy, evaluation, training_paths=training)
    actual = evaluate(policy, evaluation, training_paths=training, on_episode=episodes.append)
    assert baseline == actual
    assert episodes == [1, 2]
    episodes.clear()
    baseline = evaluate_matchups(policy, evaluation, pool, training_paths=training, seed=31)
    actual = evaluate_matchups(policy, evaluation, pool, training_paths=training,
                               seed=31, on_episode=episodes.append)
    assert baseline == actual
    assert episodes == [1, 2, 3, 4]
    def failing_callback(_completed):
        raise RuntimeError("progress callback failure")
    with pytest.raises(RuntimeError, match="progress callback failure"):
        evaluate(policy, evaluation, training_paths=training, on_episode=failing_callback)
    with pytest.raises(RuntimeError, match="progress callback failure"):
        evaluate_matchups(policy, evaluation, pool, training_paths=training, on_episode=failing_callback)
