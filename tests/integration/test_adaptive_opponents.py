# ruff: noqa: E402
"""Native sampler refreshes keep ongoing episodes immutable and restore exactly."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from test_matchup import _evaluate
from test_matchup import matchup as _matchup_fixture

from tests.support.diagnostic_agents import ActorCritic
from driversi.training.self_play.adaptive_opponents import pfsp_from_report
from driversi.training.sampling.collector import Collector
from driversi.training.self_play.opponents import FrozenOpponentPool

SETTINGS = dict(uniform_mix=0.2, power=1.0, min_actor_episodes=1)


@pytest.fixture
def matchup(straight_scene, tmp_path):
    return _matchup_fixture.__wrapped__(straight_scene, tmp_path)


def refresh(collector, report, update=0):
    return collector.refresh_opponent_sampling(
        report, optimizer_update=update, learner_version=report["learner_version"], **SETTINGS
    )


def test_refresh_changes_only_future_resets_and_restores_suffix(matchup):
    paths, _, _, manifest = matchup
    report = _evaluate(matchup)
    collector = Collector([paths["learner-train"]], num_worlds=2, seed=31, opponent_pool=manifest)
    model = ActorCritic(hidden_sizes=())
    try:
        before = collector.snapshot()
        state = refresh(collector, report)
        assert state["revision"] == 1
        after = collector.snapshot()
        assert after["worlds"] == before["worlds"]
        assert after["opponents"] == before["opponents"]
        assert after["rng"] == before["rng"]
        first = collector.collect(model, 4)
        assert np.all(first["opponent_sampling_revision"] == 0)
        refresh(collector, report, 2)
        snapshot = collector.snapshot()
        rng = torch.get_rng_state()
        expected = collector.collect(model, 35)
        assert 2 in expected["opponent_sampling_revision"]
        collector.restore(snapshot)
        torch.set_rng_state(rng)
        actual = collector.collect(model, 35)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for w in range(2):
            for episode in np.unique(actual["episode"][:, w * 32]):
                mask = actual["episode"][:, w * 32] == episode
                assert len(np.unique(actual["opponent_sampling_revision"][mask, w])) == 1
        assert np.all(actual["advantages"][~actual["learner_mask"]] == 0)
    finally:
        collector.close()


@pytest.mark.parametrize("kind", ["version", "roles", "sources", "evidence", "repeat"])
def test_invalid_refresh_never_changes_sampler_or_worlds(matchup, kind):
    paths, _, _, manifest = matchup
    report = _evaluate(matchup)
    collector = Collector([paths["learner-train"]], opponent_pool=manifest)
    try:
        refresh(collector, report)
        before = collector.snapshot()
        changed = copy.deepcopy(report)
        if kind == "version":
            changed["learner_version"] = "other"
        if kind == "roles":
            changed["learner_count"] = 2
        if kind == "sources":
            changed["learner_training_paths"] = [str(paths["opponent-train"])]
        if kind == "evidence":
            changed["matchups"][0]["episodes"][0]["learner"]["unsafe"] = 99
        with pytest.raises(ValueError):
            collector.refresh_opponent_sampling(
                changed,
                optimizer_update=0 if kind == "repeat" else 1,
                learner_version=report["learner_version"],
                **SETTINGS,
            )
        assert collector.snapshot() == before
    finally:
        collector.close()


@pytest.mark.parametrize("kind", ["probability", "revision", "episode_revision", "counter"])
def test_bad_adaptive_restore_is_transactional(matchup, kind):
    paths, _, _, manifest = matchup
    collector = Collector([paths["learner-train"]], opponent_pool=manifest)
    try:
        refresh(collector, _evaluate(matchup))
        before = collector.snapshot()
        bad = copy.deepcopy(before)
        state = bad["adaptive_opponents"]["state"]
        if kind == "probability":
            state["sampling"]["evidence"][0]["probability"] = 0.9
        if kind == "revision":
            state["revision"] += 1
        if kind == "episode_revision":
            bad["adaptive_opponents"]["assignment_revisions"][0] = 9
        if kind == "counter":
            state["optimizer_update"] = -1
        with pytest.raises(ValueError):
            collector.restore(bad)
        assert collector.snapshot() == before
    finally:
        collector.close()


def test_variance_pfsp_explores_and_favors_intermediate_safety_failure(matchup):
    report = _evaluate(matchup)
    # Controlled evidence mutation tests math only, not a real measured result.
    for row, failed in zip(report["matchups"], [0, 10]):
        row["episodes"][0]["learner"].update(actors=20, unsafe=failed, collision=failed, offroad=0)
    report["learner_count"] = 20
    pool = FrozenOpponentPool(matchup[-1])
    result = pfsp_from_report(report, pool.identity, **SETTINGS)
    probs = [x["probability"] for x in result["evidence"]]
    assert sum(probs) == pytest.approx(1)
    assert probs[1] > probs[0] >= 0.1
    assert result["metric"] == "collision_or_offroad_actor_episode_union"
    assert result["method"] == "pfsp_safety_variance_v1"


@pytest.fixture
def adaptive_run(matchup, tmp_path):
    import argparse
    import json

    from driversi.contracts.scene import Scene

    paths, _, agent, manifest = matchup
    scene = Scene.load(paths["evaluation"])
    scene.metadata["source_log"] = "independent-final-log"
    final = tmp_path / "final.npz"
    scene.save(final)
    suite = {"train": [{"path": str(paths["learner-train"])}],
             "curriculum": [{"path": str(paths["evaluation"])}],
             "evaluation": [{"path": str(final)}], "limits": ["synthetic correctness fixture"]}
    suite_path, config_path, pool_path, agent_path = [tmp_path / name for name in
                                                    ("suite.json", "config.json", "pool.json", "agent.json")]
    suite_path.write_text(json.dumps(suite))
    pool_path.write_text(json.dumps(manifest))
    agent_path.write_text(json.dumps(agent))
    config = {"schema_version": 1, "trainer": "ppo", "agent": str(agent_path),
              "opponent_pool": str(pool_path), "worlds": 2, "rollout_steps": 8,
              "seed": 37, "device": "cpu", "recording": "full",
              "optimization": {"epochs": 1, "minibatch_size": 16},
              "adaptive_opponents": {"interval_updates": 1, "min_actor_episodes": 1,
                                     "max_world_steps": 60, "seed": 43}}
    config_path.write_text(json.dumps(config))
    return argparse.Namespace(suite=suite_path, config=config_path, output_dir=tmp_path / "run",
                              budget=80, resume=None), suite, config


def test_training_refresh_checkpoint_and_full_resume_match(adaptive_run, tmp_path):
    import json

    from driversi.training.ppo import train

    args, _, _ = adaptive_run
    args.budget = 1
    train(args)
    partial = args.output_dir
    first = json.loads((partial / "report.json").read_text())
    assert first["adaptive_opponents"]["state"]["revision"] == 2
    args.resume = partial / "checkpoint.pt"
    args.output_dir = tmp_path / "resumed"
    args.budget = 150
    train(args)
    resumed = args.output_dir
    args.resume = None
    args.output_dir = tmp_path / "whole"
    train(args)
    whole = args.output_dir
    a, b = [json.loads((path / "report.json").read_text()) for path in (resumed, whole)]
    assert a["adaptive_opponents"]["state"] == b["adaptive_opponents"]["state"]
    assert a["evaluation"] == b["evaluation"]
    assert a["matchups"] == b["matchups"]
    assert a["adaptive_opponents"]["refreshes_this_run"][0]["revision"] == 3
    for path in (resumed, whole):
        for row in json.loads((path / "report.json").read_text())["adaptive_opponents"]["refreshes_this_run"]:
            assert row["physical_world_steps"] <= row["reserved_world_steps"] <= 60
    with np.load(resumed / "candidate.npz") as x, np.load(whole / "candidate.npz") as y:
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
    for path in (resumed / "rollouts").glob("batch-*.npz"):
        with np.load(path) as x, np.load(whole / "rollouts" / path.name) as y:
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key], err_msg=key)
            metadata = json.loads(str(x["metadata"]))
            assert metadata["sampling_mode"] == "adaptive_safety_pfsp"
            assert "opponent_sampling_revision" in x


@pytest.mark.parametrize("invalid", ["train_overlap", "final_overlap", "missing", "budget", "minimum"])
def test_bad_curriculum_rejected_before_training(adaptive_run, monkeypatch, invalid):
    import json

    from driversi.training.ppo import train

    args, suite, config = adaptive_run
    if invalid == "train_overlap":
        suite["curriculum"] = suite["train"]
    elif invalid == "final_overlap":
        suite["curriculum"] = suite["evaluation"]
    elif invalid == "missing":
        suite.pop("curriculum")
    elif invalid == "budget":
        config["adaptive_opponents"]["max_world_steps"] = 59
    else:
        config["adaptive_opponents"]["min_actor_episodes"] = 2
    args.suite.write_text(json.dumps(suite))
    args.config.write_text(json.dumps(config))
    monkeypatch.setattr("driversi.training.ppo.build_agent", lambda *a: pytest.fail("created model"))
    with pytest.raises(ValueError):
        train(args)
    assert not args.output_dir.exists()


def test_parallel_worlds_adaptive_distribution_matches_serial(matchup):
    paths, _, _, manifest = matchup
    report = _evaluate(matchup)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = ActorCritic(hidden_sizes=())
    a = Collector([paths["learner-train"]], num_worlds=2, seed=41, opponent_pool=manifest)
    b = Collector([paths["learner-train"]], num_worlds=2, seed=41, opponent_pool=manifest, workers=2)
    try:
        refresh(a, report)
        refresh(b, report)
        rng = torch.get_rng_state()
        x = a.collect(model, 45)
        torch.set_rng_state(rng)
        y = b.collect(model, 45)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key], err_msg=key)
    finally:
        a.close()
        b.close()


def test_failed_refresh_keeps_live_sampler_and_records_failure(adaptive_run, tmp_path, monkeypatch):
    import json

    from driversi.training.self_play.adaptive_opponents import refresh_for_training
    from driversi.training.legacy.config import resolve_training

    args, suite, _ = adaptive_run
    run = resolve_training(args)
    collector = Collector([r["path"] for r in suite["train"]], opponent_pool=run.opponent_pool)
    model = ActorCritic(hidden_sizes=())
    try:
        before = collector.snapshot()
        def fail(*a, **kw):
            raise RuntimeError("injected evaluator failure")
        monkeypatch.setattr("driversi.evaluation.benchmarks.matchup.evaluate_matchups", fail)
        with pytest.raises(RuntimeError, match="injected"):
            refresh_for_training(model, run.agent, collector,
                [r["path"] for r in suite["curriculum"]], [r["path"] for r in suite["evaluation"]],
                tmp_path / "failed", run.adaptive_opponents, 0, "test-v1")
        assert collector.snapshot() == before
        failure = json.loads((tmp_path / "failed/curriculum/revision-00001/failure.json").read_text())
        assert failure["status"] == "failed" and not failure["sampling_published"]
        assert failure["actual_world_steps"] is None
    finally:
        collector.close()
