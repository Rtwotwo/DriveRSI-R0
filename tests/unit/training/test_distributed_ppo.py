# ruff: noqa: E402
"""CPU collective tests: independent loss reference, empty ranks and sample coverage."""
import copy
import socket
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from driversi.training.legacy.distributed_update import update_distributed
from driversi.training.legacy.distribution import DistributedContext, partition, prepare_ddp
from driversi.training.ppo import update


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(4, 3)
        self.critic = torch.nn.Linear(4, 1)
        self.seen = []

    def forward(self, x):
        self.seen.extend(x[:, 0].detach().cpu().tolist())
        return self.actor(x), self.critic(x).squeeze(-1)


def batch(counts):
    n = sum(counts)
    rng = np.random.default_rng(10)
    observations = rng.normal(size=(n, 4)).astype(np.float32)
    observations[:, 0] = np.arange(1, n + 1)
    return dict(observations=observations, actions=np.arange(n) % 3,
                log_probs=np.full(n, -1.1, np.float32),
                advantages=rng.normal(size=n).astype(np.float32),
                returns=rng.normal(size=n).astype(np.float32),
                values=np.zeros(n, np.float32), valid=np.ones(n, bool))


def worker(rank, size, port, directory, counts, minibatch):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=size, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(42 + rank)
        model = TinyPolicy()
        wrapped = prepare_ddp(model, torch.device("cpu"), 4)
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-5)
        global_batch = batch(counts)
        begin, end = sum(counts[:rank]), sum(counts[:rank+1])
        local = {key: value[begin:end] for key, value in global_batch.items()}
        context = DistributedContext(rank, size, torch.device("cpu"))
        stats = update_distributed(context, wrapped, optimizer, local,
                                   epochs=2, minibatch=minibatch)
        payload = dict(state=model.state_dict(), stats=stats, seen=model.seen)
        if minibatch >= max(counts):
            reference = TinyPolicy()
            reference.load_state_dict(initial)
            opt = torch.optim.Adam(reference.parameters(), lr=0.001, eps=1e-5)
            update(reference, opt, global_batch, epochs=2, minibatch=1000)
            payload["reference"] = reference.state_dict()
        torch.save(payload, Path(directory)/f"{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("counts,minibatch", [([3, 11], 32), ([0, 7], 32), ([3, 11], 4)])
def test_unequal_batches(tmp_path, counts, minibatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(2, port, str(tmp_path), counts, minibatch), nprocs=2, join=True)
    rows = [torch.load(tmp_path/f"{rank}.pt", weights_only=False) for rank in range(2)]
    for rank, row in enumerate(rows):
        assert row["stats"]["batch_valid"] == sum(counts)
        assert row["stats"]["sample_visits"] == 2 * sum(counts)
        assert row["stats"]["dropped_valid"] == 0
        for key in row["state"]:
            torch.testing.assert_close(row["state"][key], rows[0]["state"][key], rtol=0, atol=0)
            if "reference" in row:
                torch.testing.assert_close(row["state"][key], row["reference"][key],
                                           rtol=1e-5, atol=2e-7)
        observed = [int(x) for x in row["seen"] if x != 0]
        expected = list(range(sum(counts[:rank])+1, sum(counts[:rank+1])+1)) * 2
        assert sorted(observed) == sorted(expected)


def test_resource_partition():
    assert partition(19, 8) == [3, 3, 3, 2, 2, 2, 2, 2]
    assert partition(3, 8) == [1, 1, 1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        partition(-1, 8)

class MemoryCollector:
    def __init__(self, rank):
        self.state = {"rank": rank, "counter": 7}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, state):
        self.state = copy.deepcopy(state)


def lifecycle_worker(rank, size, port, directory):
    from driversi.training.legacy.distributed_checkpoint import restore_distributed, save_distributed
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                           rank=rank, world_size=size, timeout=timedelta(seconds=30))
    try:
        context = DistributedContext(rank, size, torch.device("cpu"))
        torch.manual_seed(12)
        model = TinyPolicy()
        wrapped = prepare_ddp(model, torch.device("cpu"), 4)
        optimizer = torch.optim.Adam(model.parameters())
        all_rows = batch([3, 7])
        begin, end = (0, 3) if rank == 0 else (3, 10)
        local = {key: value[begin:end] for key, value in all_rows.items()}
        update_distributed(context, wrapped, optimizer, local, epochs=2, minibatch=32)
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        assert expected_optimizer["state"]
        collector = MemoryCollector(rank)
        config = {"world_size": size, "worlds": 9}
        path = Path(directory)/"checkpoint.pt"
        torch.manual_seed(100+rank)
        save_distributed(path, context, model, optimizer, collector, 2, 35, config)
        expected_rng = torch.rand(4)
        expected_model = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            next(model.parameters()).add_(1)
        for value in optimizer.state.values():
            value["exp_avg"].add_(10)
        collector.state["counter"] = 99
        assert restore_distributed(path, context, model, optimizer, collector, config) == (2, 35)
        torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
        assert collector.state == {"rank": rank, "counter": 7}
        for key, values in expected_optimizer["state"].items():
            for field, value in values.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][key][field], value,
                                           rtol=0, atol=0)
        for key in expected_model:
            torch.testing.assert_close(model.state_dict()[key], expected_model[key], rtol=0, atol=0)
        try:
            restore_distributed(path, context, model, optimizer, collector, {"world_size": size})
        except RuntimeError as error:
            assert "configuration/world-size mismatch" in str(error)
        else:
            raise AssertionError("mismatched checkpoint accepted")
        def fail():
            if rank == 1:
                raise ValueError("injected collection failure")
        try:
            context.stage("collection", fail)
        except RuntimeError as error:
            assert "rank 1" in str(error) and "injected collection failure" in str(error)
        else:
            raise AssertionError("rank failure was not propagated")
        second = Path(directory)/"other-checkpoint.pt"
        save_distributed(second, context, model, optimizer, collector, 3, 50, config)
        try:
            restore_distributed(path if rank == 0 else second, context, model,
                                optimizer, collector, config)
        except ValueError as error:
            assert "different checkpoint identities" in str(error)
        else:
            raise AssertionError("different valid checkpoints accepted across ranks")
        (Path(directory)/f"lifecycle-{rank}").write_text("pass")
    finally:
        dist.destroy_process_group()


def test_checkpoint_and_stage_failure(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(lifecycle_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    assert len(list(tmp_path.glob("lifecycle-*"))) == 2


def test_distributed_cli_and_opt_in(tmp_path, monkeypatch):
    import argparse

    from driversi.training.legacy.config import add_training_arguments, resolve_training
    from driversi.training.legacy.distributed import train_distributed
    from driversi.training.ppo import train
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    base = ["--suite", str(tmp_path/"suite.json"), "--output-dir", str(tmp_path/"output")]
    args = parser.parse_args(base)
    assert resolve_training(args).distribution["enabled"] is False
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="explicit"):
        train(args)
    args = parser.parse_args(base+["--distributed"])
    assert resolve_training(args).distribution["enabled"] is True
    with pytest.raises(ValueError, match="device=cuda"):
        train_distributed(args)
    args = parser.parse_args(base+["--distributed-timeout-seconds", "nan"])
    with pytest.raises(ValueError, match="finite"):
        resolve_training(args)


@pytest.mark.parametrize("failure", ["cleanup", "budget", "resume", "output"])
def test_invocation_and_shutdown_failures(tmp_path, monkeypatch, failure):
    import json
    from types import SimpleNamespace

    import driversi.training.legacy.distributed as training

    suite = tmp_path/"suite.json"
    suite.write_text(json.dumps(dict(train=[{"path": "train"}],
                                     evaluation=[{"path": "evaluation"}], limits=[])))
    args = SimpleNamespace(device="cuda", distribution={"timeout_seconds": 30},
                           worlds=2, rollout_steps=1, budget=1, seed=2, agent={},
                           sampling=dict(workers=0, max_rollout_mib=100),
                           optimization=dict(learning_rate=0.001, epochs=1, minibatch_size=4),
                           suite=suite, output_dir=tmp_path/"output", resume=None,
                           recording="metrics")
    class Context:
        rank, size, device = 0, 2, torch.device("cpu")
        def stage(self, name, fn):
            return fn()
        root = stage
    class Policy(TinyPolicy):
        def export(self, path, version):
            Path(path).write_text(version)
    class Collector:
        progress_weight = 0.02
        def __init__(self, *args, **kwargs):
            pass
        def collect(self, *args, **kwargs):
            raise RuntimeError("collection failed")
        def close(self):
            raise OSError("cleanup failed")
    destroyed = []
    def gather(target, value):
        target[:] = [value, copy.deepcopy(value)]
        if failure != "cleanup":
            target[1][failure] = 2 if failure == "budget" else "different-path"
    monkeypatch.setattr(training, "resolve_training", lambda _: args)
    monkeypatch.setattr(training, "initialize", lambda _: Context())
    monkeypatch.setattr(training, "validate_splits", lambda *a: None)
    monkeypatch.setattr(training, "build_agent", lambda _: Policy())
    monkeypatch.setattr(training, "prepare_ddp", lambda model, *a: model)
    monkeypatch.setattr(training, "Collector", Collector)
    monkeypatch.setattr(training, "source_inventory", lambda *a: {})
    monkeypatch.setattr(training, "evaluate", lambda *a, **kw: {})
    monkeypatch.setattr(training, "load_inference", lambda *a: None)
    monkeypatch.setattr(training.dist, "all_gather_object", gather)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training.dist, "destroy_process_group", lambda: destroyed.append(True))
    if failure == "cleanup":
        with pytest.raises(OSError, match="cleanup failed"):
            training.train_distributed(None)
    else:
        with pytest.raises(ValueError, match="rank configurations differ"):
            training.train_distributed(None)
    assert destroyed == [True]
    assert not (args.output_dir/"report.json").exists()


def mixed_batch():
    """Known membership built independently of the production learning_mask."""
    physical_counts = [5, 8, 12]
    role_counts = [0, 4, 8]
    rollout = batch(physical_counts)
    n = sum(physical_counts)
    rollout["learner_mask"] = np.zeros(n, bool)
    rollout["on_policy_eligible"] = np.ones(n, bool)
    rollout["behavior_policy_index"] = np.ones(n, np.int64)
    eligible = []
    for rank, count in enumerate(role_counts):
        begin = sum(physical_counts[:rank])
        rollout["learner_mask"][begin:begin + count] = True
        rollout["behavior_policy_index"][begin:begin + count] = 0
        if count:
            # One real learner action per rank was overridden and is excluded.
            rollout["on_policy_eligible"][begin] = False
            eligible.extend(range(begin + 1, begin + count))
    eligible = np.asarray(eligible, dtype=np.int64)
    excluded = np.ones(n, bool)
    excluded[eligible] = False
    for key in ("observations", "log_probs", "values", "advantages", "returns"):
        rollout[key][excluded] = np.nan
    rollout["actions"][excluded] = -999
    return rollout, eligible, physical_counts


def mixed_worker(rank, size, port, directory, minibatch):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                           rank=rank, world_size=size, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(17 + rank)
        model = TinyPolicy()
        wrapped = prepare_ddp(model, torch.device("cpu"), 4)
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-5)
        global_batch, eligible, counts = mixed_batch()
        begin, end = sum(counts[:rank]), sum(counts[:rank + 1])
        local = {key: value[begin:end] for key, value in global_batch.items()}
        context = DistributedContext(rank, size, torch.device("cpu"))
        stats = update_distributed(context, wrapped, optimizer, local, epochs=3, minibatch=minibatch)
        payload = {"state": model.state_dict(), "stats": stats, "seen": model.seen,
                   "optimizer_state": optimizer.state_dict(), "physical_count": int(local["valid"].sum())}
        if minibatch >= 7:
            reference = TinyPolicy()
            reference.load_state_dict(initial)
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.001, eps=1e-5)
            # Construct a fresh ordinary on-policy batch from independently known
            # row IDs. It has no mixed-policy fields and never calls learning_mask
            # to decide which original rows should contribute to the reference.
            reference_batch = {key: global_batch[key][eligible].copy() for key in
                ("observations", "actions", "log_probs", "values", "advantages", "returns")}
            reference_batch["valid"] = np.ones(len(eligible), bool)
            update(reference, reference_optimizer, reference_batch, epochs=3, minibatch=1000)
            payload["reference"] = reference.state_dict()
        torch.save(payload, Path(directory) / f"mixed-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("minibatch", [32, 4])
def test_mixed_ddp_empty_learner_rank_uses_only_eligible_samples(tmp_path, minibatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(mixed_worker, args=(3, port, str(tmp_path), minibatch), nprocs=3, join=True)
    rows = [torch.load(tmp_path / f"mixed-{rank}.pt", weights_only=False) for rank in range(3)]
    _, eligible, counts = mixed_batch()
    assert sum(row["physical_count"] for row in rows) == 25
    for rank, row in enumerate(rows):
        assert row["stats"]["rank_valid"] == [0, 3, 7]
        assert row["stats"]["batch_valid"] == row["stats"]["samples_used"] == 10
        assert row["stats"]["sample_visits"] == 30
        assert row["stats"]["dropped_valid"] == 0
        assert row["stats"]["optimizer_steps"] == (3 if minibatch == 32 else 6)
        assert row["optimizer_state"]["state"]  # The empty learner rank joined all updates.
        assert np.isfinite(row["seen"]).all()
        for key, value in row["state"].items():
            torch.testing.assert_close(value, rows[0]["state"][key], rtol=0, atol=0)
            if "reference" in row:
                torch.testing.assert_close(value, row["reference"][key], rtol=1e-5, atol=2e-7)
        begin, end = sum(counts[:rank]), sum(counts[:rank + 1])
        expected = (eligible[(eligible >= begin) & (eligible < end)] + 1).tolist() * 3
        observed = [int(value) for value in row["seen"] if value != 0]
        assert sorted(observed) == sorted(expected)
    assert all(value == 0 for value in rows[0]["seen"])
