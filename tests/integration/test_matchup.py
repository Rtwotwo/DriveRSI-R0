"""Executable frozen-matchup boundaries using real native joint worlds."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from tests.support.diagnostic_config import load_inference, normalize_agent
from driversi.evaluation.benchmarks.matchup import evaluate_matchups, main
from driversi.training.self_play.opponents import FrozenOpponentPool


def _model(path, version, action=7):
    config = normalize_agent()
    config["parameters"] = {"hidden_sizes": []}
    logits = np.full(12, -60.0, dtype=np.float32)
    logits[action] = 60.0
    metadata = {k: config[k] for k in ("observation_version", "action_space", "input_mode")}
    metadata.update(
        policy_version=version,
        architecture={"version": 1, "hidden_sizes": [], "activation": "tanh"},
    )
    np.savez(
        path,
        metadata=json.dumps(metadata),
        **{
            "actor.weight": np.zeros((12, 1125), np.float32),
            "actor.bias": logits,
            "critic.weight": np.zeros((1, 1125), np.float32),
            "critic.bias": np.zeros(1, np.float32),
        },
    )
    return config


@pytest.fixture
def matchup(straight_scene, tmp_path):
    paths = {}
    for name in ("learner-train", "opponent-train", "evaluation"):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = name
        path = tmp_path / (name + ".npz")
        scene.save(path)
        paths[name] = path
    learner = tmp_path / "learner.npz"
    config = _model(learner, "learner-v1")
    rows = []
    for name, action in (("coast", 7), ("brake", 1)):
        path = tmp_path / (name + ".npz")
        _model(path, name + "-v1", action)
        rows.append(
            {
                "id": name,
                "checkpoint": str(path),
                "agent_config": config,
                "training_paths": [str(paths["opponent-train"])],
            }
        )
    manifest = {"schema_version": 1, "policies": rows}
    return paths, learner, config, manifest


def _evaluate(matchup, **kwargs):
    paths, learner, config, manifest = matchup
    return evaluate_matchups(
        load_inference(learner, config),
        [paths["evaluation"]],
        manifest,
        training_paths=[paths["learner-train"]],
        seed=43,
        **kwargs,
    )


def test_fixed_matchups_reproducible_with_separate_roles_and_real_actions(matchup):
    first, second = _evaluate(matchup), _evaluate(matchup)
    assert first == second
    assert first["source_separation_checked"] and not first["official_test"]
    assert len(first["matchups"]) == 2
    digests = []
    for row in first["matchups"]:
        episode = row["episodes"][0]
        assert episode["roles"] == [
            {
                "actor_id": "a",
                "role": "learner",
                "policy_version": "learner-v1",
                "policy_id": "learner",
            },
            {
                "actor_id": "b",
                "role": "opponent",
                "policy_version": row["opponent_version"],
                "policy_id": row["opponent_id"],
            },
        ]
        assert episode["learner"]["actors"] == episode["opponent"]["actors"] == 1
        assert episode["all_world"]["valid_agent_steps"] == (
            episode["learner"]["valid_agent_steps"] + episode["opponent"]["valid_agent_steps"]
        )
        assert (
            sum(episode["learner"]["action_histogram"]) == episode["learner"]["valid_agent_steps"]
        )
        expected_action = 7 if row["opponent_id"] == "coast" else 1
        assert (
            episode["opponent"]["action_histogram"][expected_action]
            == episode["opponent"]["valid_agent_steps"]
        )
        assert (
            row["all_world"]["actor_episodes"] == 2
        )  # reference actor is not a controlled episode
        digests.append(episode["joint_action_state_sha256"])
    assert digests[0] != digests[1]


def test_matchup_rng_independent_of_opponent_manifest_order(matchup):
    paths, learner, config, manifest = matchup
    a = _evaluate(matchup)
    manifest["policies"].reverse()
    b = _evaluate((paths, learner, config, manifest))
    assert {x["opponent_id"]: x["episodes"] for x in a["matchups"]} == {
        x["opponent_id"]: x["episodes"] for x in b["matchups"]
    }


@pytest.mark.parametrize("who", ["learner", "opponent"])
def test_declared_training_log_overlap_rejected_before_evaluation(matchup, who):
    paths, learner, config, manifest = matchup
    if who == "learner":
        paths["learner-train"] = paths["evaluation"]
    else:
        manifest["policies"][0]["training_paths"] = [str(paths["evaluation"])]
    with pytest.raises(ValueError, match="overlaps"):
        _evaluate((paths, learner, config, manifest))


@pytest.mark.parametrize("training_paths", [None, []])
def test_unknown_learner_sources_cannot_claim_clean_evaluation(matchup, training_paths):
    paths, learner, config, manifest = matchup
    with pytest.raises(ValueError, match="training_paths|Both development"):
        evaluate_matchups(
            load_inference(learner, config),
            [paths["evaluation"]],
            manifest,
            training_paths=training_paths,
        )


def test_missing_opponent_sources_rejected(matchup):
    matchup[-1]["policies"][0]["training_paths"] = []
    with pytest.raises(ValueError, match="training_paths"):
        _evaluate(matchup)


@pytest.mark.parametrize("value", [0, 2, True])
def test_learner_count_requires_positive_mixed_roles(matchup, value):
    with pytest.raises(ValueError, match="learner_count|one learner"):
        _evaluate(matchup, learner_count=value)


def test_invalid_inference_does_not_advance_world(matchup, monkeypatch):
    from driversi.training.sampling.environment import TeacherEnvironment

    paths, learner, config, manifest = matchup
    policy = load_inference(learner, config)
    monkeypatch.setattr(
        policy, "predict", lambda x: (np.full((len(x), 12), np.nan), np.zeros(len(x)))
    )
    monkeypatch.setattr(TeacherEnvironment, "advance", lambda *a: pytest.fail("world advanced"))
    with pytest.raises(ValueError, match="finite logits"):
        evaluate_matchups(
            policy, [paths["evaluation"]], manifest, training_paths=[paths["learner-train"]]
        )


def test_existing_pool_instance_supported_and_identity_preserved(matchup):
    paths, learner, config, manifest = matchup
    pool = FrozenOpponentPool(manifest)
    result = evaluate_matchups(
        load_inference(learner, config),
        [paths["evaluation"]],
        pool,
        training_paths=[paths["learner-train"]],
    )
    assert result["opponent_pool"] == pool.identity


def test_cli_relative_suite_paths_and_immutable_complete_report(matchup, tmp_path):
    paths, learner, config, manifest = matchup
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "train": [{"path": paths["learner-train"].name}],
                "evaluation": [{"path": paths["evaluation"].name}],
            }
        )
    )
    agent = tmp_path / "agent.json"
    agent.write_text(json.dumps(config))
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps(manifest))
    output = tmp_path / "report.json"
    args = [
        "--suite",
        str(suite),
        "--learner",
        str(learner),
        "--agent-config",
        str(agent),
        "--opponent-pool",
        str(pool),
        "--output",
        str(output),
        "--seed",
        "77",
    ]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["seed"] == 77 and result["suite"]["training_declaration"] == "suite.train"
    assert len(result["learner_checkpoint"]["sha256"]) == 64
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".matchup-*"))


def test_stochastic_matchups_use_repeatable_actor_random_streams(matchup):
    paths, learner, config, manifest = matchup
    for checkpoint in [learner, *[row["checkpoint"] for row in manifest["policies"]]]:
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["actor.bias"][:] = 0
        np.savez(checkpoint, **arrays)
    policy = load_inference(learner, config)

    def evaluate(seed):
        return evaluate_matchups(
            policy,
            [paths["evaluation"]],
            manifest,
            training_paths=[paths["learner-train"]],
            seed=seed,
        )

    first = evaluate(10)
    assert first == evaluate(10)
    assert first["matchups"][0]["episodes"] != evaluate(11)["matchups"][0]["episodes"]
    # Equal frozen logits share common random draws; labels alone cannot change actions.
    a, b = [row["episodes"][0] for row in first["matchups"]]
    assert a["joint_action_state_sha256"] == b["joint_action_state_sha256"]
    assert np.count_nonzero(a["learner"]["action_histogram"]) > 1


def test_distinct_windows_from_same_original_log_are_rejected(matchup):
    from driversi.contracts.scene import Scene

    paths = matchup[0]
    for key, suffix in (("learner-train", "00000_00010"), ("evaluation", "00020_00030")):
        scene = Scene.load(paths[key])
        scene.metadata["source_log"] = "original-drive_" + suffix + ".pkl"
        scene.save(paths[key])
    with pytest.raises(ValueError, match="overlaps"):
        _evaluate(matchup)


def test_cli_changed_learner_file_is_rejected_before_report(matchup, tmp_path, monkeypatch):
    import driversi.evaluation.benchmarks.matchup as module

    paths, learner, config, manifest = matchup
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "train": [{"path": str(paths["learner-train"])}],
                "evaluation": [{"path": str(paths["evaluation"])}],
            }
        )
    )
    agent, pool, output = (tmp_path / name for name in ("agent.json", "pool.json", "report.json"))
    agent.write_text(json.dumps(config))
    pool.write_text(json.dumps(manifest))
    original = module.load_inference

    def changed(path, config):
        policy = original(path, config)
        path.write_bytes(path.read_bytes() + b"modified")
        return policy

    monkeypatch.setattr(module, "load_inference", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        main(
            [
                "--suite",
                str(suite),
                "--learner",
                str(learner),
                "--agent-config",
                str(agent),
                "--opponent-pool",
                str(pool),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_priority_report_weights_and_identity(matchup):
    from driversi.training.self_play.opponent_priority import priority_from_report

    report = _evaluate(matchup)
    pool = FrozenOpponentPool(matchup[-1])
    settings = dict(uniform_mix=0.2, power=2.0, min_actor_episodes=1)
    priority = priority_from_report(report, pool.identity, **settings)
    assert sum(r["probability"] for r in priority["evidence"]) == pytest.approx(1)
    assert all(r["probability"] >= 0.1 for r in priority["evidence"])
    assert all(
        r["learner"]["unsafe"] <= r["learner"]["actors"]
        for m in report["matchups"]
        for r in m["episodes"]
    )
    changed = copy.deepcopy(report)
    changed["matchups"][0]["episodes"][0]["learner"]["unsafe"] = 99
    with pytest.raises(ValueError, match="inconsistent"):
        priority_from_report(changed, pool.identity, **settings)
    changed = copy.deepcopy(report)
    changed["official_test"] = True
    with pytest.raises(ValueError, match="development"):
        priority_from_report(changed, pool.identity, **settings)
    changed = copy.deepcopy(report)
    changed["matchups"][0]["episodes"][0]["scene"] = report["learner_training_paths"][0]
    with pytest.raises(ValueError, match="Duplicated or missing"):
        priority_from_report(changed, pool.identity, **settings)
    with pytest.raises(ValueError, match="Insufficient"):
        priority_from_report(report, pool.identity, **{**settings, "min_actor_episodes": 20})


def test_priority_manifest_sample_restore_and_uniform_compatibility(matchup, tmp_path):
    report = _evaluate(matchup)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    base = FrozenOpponentPool(matchup[-1])
    raw = copy.deepcopy(matchup[-1])
    raw.update(
        schema_version=2,
        sampling=dict(
            method="failure_priority_v1",
            matchup_report=str(path),
            uniform_mix=0.2,
            power=2.0,
            min_actor_episodes=1,
        ),
    )
    pool = FrozenOpponentPool(raw)
    assert pool.identity == json.loads(json.dumps(pool.identity))
    assert pool.identity != base.identity
    a, b = np.random.default_rng(13), np.random.default_rng(13)
    np.testing.assert_array_equal(base.sample_indices(a, 1000), b.integers(1, 3, 1000))
    rng = np.random.default_rng(15)
    state = copy.deepcopy(rng.bit_generator.state)
    first = pool.sample_indices(rng, 5000)
    rng.bit_generator.state = state
    np.testing.assert_array_equal(pool.sample_indices(rng, 5000), first)
    assert set(first) == {1, 2}
    invalid = copy.deepcopy(raw)
    invalid["sampling"]["uniform_mix"] = 0
    with pytest.raises(ValueError):
        FrozenOpponentPool(invalid)


def test_weighted_training_checkpoint_restores_actual_suffix(matchup, tmp_path):
    torch = pytest.importorskip("torch")
    from tests.support.diagnostic_agents import ActorCritic
    from driversi.training.legacy.checkpoint import restore_checkpoint, save_checkpoint
    from driversi.training.sampling.collector import Collector

    report_path = tmp_path / "weighted-report.json"
    report_path.write_text(json.dumps(_evaluate(matchup)))
    raw = copy.deepcopy(matchup[-1])
    raw.update(schema_version=2, sampling=dict(method="failure_priority_v1",
               matchup_report=str(report_path), uniform_mix=0.2, power=2.,
               min_actor_episodes=1))
    collector = Collector([matchup[0]["learner-train"]], num_worlds=2,
                          seed=27, opponent_pool=raw)
    model = ActorCritic(hidden_sizes=())
    try:
        collector.collect(model, 9)
        path = tmp_path / "weighted-checkpoint.pt"
        optimizer = torch.optim.Adam(model.parameters())
        config = {"opponent_identity": FrozenOpponentPool(raw).identity}
        save_checkpoint(path, model, optimizer, collector, 0, 0, config)
        expected = collector.collect(model, 17)
        restore_checkpoint(path, model, optimizer, collector, config)
        actual = collector.collect(model, 17)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    finally:
        collector.close()


def test_optional_bev_recording_preserves_matchup_actions_states_and_metrics(matchup, tmp_path):
    import hashlib

    plain = _evaluate(matchup)
    output = tmp_path / "replays"
    recorded = _evaluate(matchup, replay_dir=output)
    for row in recorded["matchups"]:
        for ep in row["episodes"]:
            replay = ep.pop("replay")
            directory = __import__("pathlib").Path(replay["directory"])
            assert (directory / "viewer.html").is_file()
            assert replay["gs_assets_verified"] is False
            with np.load(directory / "rollout.npz") as arrays:
                digest = hashlib.sha256()
                controlled = arrays["controlled_ids"].tolist()
                indices = [arrays["actor_ids"].tolist().index(a) for a in controlled]
                for actions, after, valid in zip(arrays["actions"], arrays["states"][1:, indices], arrays["valid"]):
                    digest.update(actions.astype("<i8").tobytes())
                    digest.update(after.astype("<f8").tobytes())
                    digest.update(valid.tobytes())
                assert digest.hexdigest() == ep["joint_action_state_sha256"]
                protocol = json.loads(str(arrays["metadata"]))["protocol"]
                assert not protocol["photoreal_training"]
    assert recorded == plain
    with pytest.raises(FileExistsError):
        _evaluate(matchup, replay_dir=output)
