# ruff: noqa: E402
import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from tests.support.diagnostic_agents import ActorCritic
from tests.support.diagnostic_config import DEFAULT_AGENT
from driversi.training.sampling.collector import Collector, rollout_nbytes
from driversi.training.self_play.opponents import FrozenOpponentPool


def fixture_pool(straight_scene, tmp_path):
    paths = []
    for index in range(2):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = f"league-source-{index}"
        scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
        count = 11 + index * 10
        scene.times, scene.poses, scene.velocities = scene.times[:count], scene.poses[:count], scene.velocities[:count]
        path = tmp_path / f"scene-{index}.npz"
        scene.save(path)
        paths.append(path)
    config = copy.deepcopy(DEFAULT_AGENT)
    config["parameters"] = {"hidden_sizes": [8]}
    config_path = tmp_path / "agent.json"
    config_path.write_text(json.dumps(config))
    rows = []
    for index, action in enumerate((4, 7)):
        model = ActorCritic(hidden_sizes=(8,))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.actor.bias[action] = 50.
        checkpoint = tmp_path / f"opponent-{index}.npz"
        model.export(checkpoint, f"frozen-v{index}")
        rows.append({"id": f"history-{index}", "checkpoint": checkpoint.name,
                     "agent_config": config_path.name, "training_paths": [p.name for p in paths]})
    manifest = tmp_path / "opponents.json"
    manifest.write_text(json.dumps({"schema_version": 1, "policies": rows}))
    return paths, manifest


def assert_arrays(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_no_pool_explicit_default_preserves_arrays_and_rng(straight_scene, tmp_path):
    paths, _ = fixture_pool(straight_scene, tmp_path)
    model = ActorCritic(hidden_sizes=(8,))
    a = Collector(paths, num_worlds=2, seed=18)
    b = Collector(paths, num_worlds=2, seed=18, opponent_pool=None, learner_count=1)
    try:
        rng = torch.get_rng_state()
        expected = a.collect(model, 24)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        assert_arrays(b.collect(model, 24), expected)
        assert torch.equal(torch.get_rng_state(), after)
        assert "learner_mask" not in expected and "opponents" not in a.snapshot()
        assert a.last_metrics["rollout_array_bytes"] == rollout_nbytes(24, 2)
    finally:
        a.close()
        b.close()


def test_assignments_frozen_actions_and_physical_cost(straight_scene, tmp_path):
    paths, manifest = fixture_pool(straight_scene, tmp_path)
    pool = FrozenOpponentPool(manifest)
    identities = pool.identity
    weights = [{k: v.copy() for k, v in p.weights.items()} for p in pool._policies]
    collector = Collector(paths, num_worlds=4, seed=5, opponent_pool=pool)
    try:
        costs = []
        data = collector.collect(ActorCritic(hidden_sizes=(8,)), 27, before_step=costs.append)
        assert data["valid"].sum() == sum(costs)
        assert collector.last_metrics["rollout_array_bytes"] == rollout_nbytes(27, 4, opponents=True)
        for index, action in ((1, 4), (2, 7)):
            chosen = data["valid"] & (data["behavior_policy_index"] == index)
            assert chosen.any()
            assert np.all(data["actions"][chosen] == action)
        assert np.all(data["advantages"][~data["learner_mask"]] == 0)
        assert np.all(data["returns"][~data["learner_mask"]] == 0)
        assert np.all(data["on_policy_eligible"] == data["valid"])
        for world in range(4):
            slot = world * 32 + 1
            for episode in np.unique(data["episode"][:, slot]):
                assert len(np.unique(data["behavior_policy_index"][data["episode"][:, slot] == episode, slot])) == 1
        assert pool.identity == identities
        for policy, old in zip(pool._policies, weights):
            for key in old:
                np.testing.assert_array_equal(policy.weights[key], old[key])
                assert not policy.weights[key].flags.writeable
        assert collector.behavior_policies("learner-v3")[0]["policy_version"] == "learner-v3"
        assert collector.last_metrics["learner_transitions"] + collector.last_metrics["opponent_transitions"] == sum(costs)
    finally:
        collector.close()


def test_serial_parallel_restore_preserve_ownership_rng_and_next_values(straight_scene, tmp_path):
    torch.set_num_threads(1)
    paths, manifest = fixture_pool(straight_scene, tmp_path)
    model = ActorCritic(hidden_sizes=(8,))
    serial = Collector(paths, num_worlds=4, seed=14, opponent_pool=manifest)
    parallel = Collector(paths, num_worlds=4, seed=14, workers=2, opponent_pool=manifest)
    try:
        rng = torch.get_rng_state()
        expected = serial.collect(model, 23)
        torch.set_rng_state(rng)
        assert_arrays(parallel.collect(model, 23), expected)
        snapshot = parallel.snapshot()
        rng = torch.get_rng_state()
        suffix = parallel.collect(model, 17)
        parallel.restore(snapshot)
        torch.set_rng_state(rng)
        assert_arrays(parallel.collect(model, 17), suffix)
        serial.restore(snapshot)
        torch.set_rng_state(rng)
        assert_arrays(serial.collect(model, 17), suffix)
    finally:
        serial.close()
        parallel.close()


@pytest.mark.parametrize("mutation", ["identity", "assignment", "rng", "world", "missing", "learner_count"])
def test_bad_restore_is_transactional(straight_scene, tmp_path, mutation):
    paths, manifest = fixture_pool(straight_scene, tmp_path)
    collector = Collector(paths, num_worlds=1, opponent_pool=manifest)
    try:
        before = collector.snapshot()
        bad = copy.deepcopy(before)
        if mutation == "identity":
            bad["opponents"]["identity"]["sha256"] = "different"
        elif mutation == "assignment":
            bad["opponents"]["assignments"][0]["policy_indices"][0] = 1
        elif mutation == "rng":
            bad["opponents"]["rng"] = {"bad": True}
        elif mutation == "world":
            bad["worlds"][0]["payload"] = "not-base64"
        elif mutation == "learner_count":
            bad["opponents"]["learner_count"] = True
        else:
            del bad["opponents"]
        with pytest.raises((ValueError, TypeError, KeyError)):
            collector.restore(bad)
        assert collector.snapshot() == before
        assert not collector.pool.closed
    finally:
        collector.close()


def test_manifest_validation_and_content_identity(straight_scene, tmp_path):
    paths, manifest = fixture_pool(straight_scene, tmp_path)
    raw = json.loads(manifest.read_text())
    a = FrozenOpponentPool(manifest)
    other = copy.deepcopy(raw)
    other["policies"][0]["training_paths"] = []
    manifest.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="training_paths"):
        FrozenOpponentPool(manifest)
    manifest.write_text(json.dumps(raw))
    for row in raw["policies"]:
        row["checkpoint"] = str(tmp_path / row["checkpoint"])
        row["agent_config"] = str(tmp_path / row["agent_config"])
        row["training_paths"] = [str(tmp_path / path) for path in row["training_paths"]]
    b = FrozenOpponentPool(raw)
    assert a.identity == b.identity
    policy_path = Path(raw["policies"][0]["checkpoint"])
    model = ActorCritic(hidden_sizes=(8,))
    model.export(policy_path, "frozen-v0")
    assert a.identity != FrozenOpponentPool(raw).identity
    with pytest.raises(ValueError, match="one learner and one frozen"):
        Collector(paths, opponent_pool=a, learner_count=2)
    leaked = copy.deepcopy(straight_scene)
    leaked.metadata["split"] = "test"
    leaked.save(paths[0])
    with pytest.raises(ValueError, match="heldout/test"):
        FrozenOpponentPool(raw)


def test_output_shape_and_numeric_validation(straight_scene, tmp_path):
    _, manifest = fixture_pool(straight_scene, tmp_path)
    pool = FrozenOpponentPool(manifest)
    pool._policies[0].predict = lambda x: (np.zeros((len(x), 11)), np.zeros(len(x)))
    with pytest.raises(ValueError, match="finite"):
        pool.predict(0, np.zeros((1, 1125), dtype=np.float32))
    pool._policies[0].predict = lambda x: (np.full((len(x), 12), np.nan), np.zeros(len(x)))
    with pytest.raises(ValueError, match="finite"):
        pool.predict(0, np.zeros((1, 1125), dtype=np.float32))


def test_last_living_opponent_remains_reactive_after_learner_ends(straight_scene, tmp_path):
    _, manifest = fixture_pool(straight_scene, tmp_path)
    scene = copy.deepcopy(straight_scene)
    scene.metadata["task_goals_xy"] = {"a": [4.5, 0], "b": [100, 8]}
    path = tmp_path / "last-alive.npz"
    scene.save(path)
    pool = FrozenOpponentPool(manifest)
    seen = []
    original = pool.predict
    def tracked(index, x):
        seen.append(x.copy())
        return original(index, x)
    pool.predict = tracked
    learner = ActorCritic(hidden_sizes=(8,))
    with torch.no_grad():
        for value in learner.parameters():
            value.zero_()
        learner.actor.bias[7] = 50.
    collector = Collector([path], num_worlds=1, seed=8, opponent_pool=pool)
    try:
        data = collector.collect(learner, 20)
        rows = ~data["valid"][:, 0] & data["valid"][:, 1]
        assert rows.any()
        assert data["episode"][:, 1].max() == 0
        assert np.all(data["on_policy_eligible"][rows, 1])
        assert np.all(data["advantages"][rows] == 0)
        assert np.linalg.norm(data["world_after"][-1, 0, 1, :2] - data["world_before"][0, 0, 1, :2]) > 0
        assert len(seen) >= 40
        # Current-world features, including remaining task time, are freshly
        # delivered even when the learner no longer supplies training samples.
        assert not np.array_equal(seen[0], seen[-1])
    finally:
        collector.close()
