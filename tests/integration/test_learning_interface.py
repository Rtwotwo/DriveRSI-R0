# ruff: noqa: E402
import copy
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")
from driversi.contracts.learning import END_REASONS, JointDecision
from driversi.training.sampling.environment import TeacherEnvironment


def decide(env, actions=(7, 7), probabilities=True):
    obs = env.observation()
    return JointDecision(obs.key, obs.actor_ids, np.asarray(actions), "test-policy-v1",
                         np.zeros(len(actions)) if probabilities else None)


def test_external_joint_step_matches_internal_and_exposes_reward_terms(straight_scene):
    a, b = TeacherEnvironment(straight_scene), TeacherEnvironment(straight_scene)
    try:
        for _ in range(7):
            result = a.advance(decide(a))
            obs, reward, raw = b.step(np.array([7, 7]))
            np.testing.assert_array_equal(result.observation.features, obs)
            np.testing.assert_array_equal(result.rewards, reward)
            np.testing.assert_array_equal(a.world.state(), b.world.state())
            np.testing.assert_array_equal(result.transition.valid, raw.valid)
            np.testing.assert_allclose(result.reward_terms.sum(axis=1), reward)
            assert result.on_policy_eligible.all()
        with pytest.raises(ValueError):
            result.observation.features[0, 0] = 99
        assert a.spec.to_dict()["goal_semantics"] == "endpoint_absorbing"
    finally:
        a.close()
        b.close()


def test_reset_restore_and_fork_reject_stale_foreign_actions(straight_scene):
    env = TeacherEnvironment(straight_scene)
    child = None
    try:
        initial = decide(env)
        snapshot = env.snapshot()
        env.reset()
        with pytest.raises(ValueError, match="identity"):
            env.advance(initial)
        current = decide(env)
        env.restore(snapshot)
        with pytest.raises(ValueError, match="identity"):
            env.advance(current)
        current = decide(env)
        child = env.fork()
        with pytest.raises(ValueError, match="identity"):
            child.advance(current)
        expected = env.advance(current)
        actual = child.advance(decide(child))
        np.testing.assert_array_equal(expected.observation.features, actual.observation.features)
        np.testing.assert_array_equal(env.world.state(), child.world.state())
        child.advance(decide(child, (1, 1)))
        assert env.observation().key.tick == 1
    finally:
        env.close()
        if child:
            child.close()


def test_wrong_actor_order_and_illegal_override_leave_world_unchanged(straight_scene):
    env = TeacherEnvironment(straight_scene)
    try:
        original = env.snapshot()
        with pytest.raises(ValueError, match="actor order"):
            env.advance(replace(decide(env), actor_ids=("b", "a")))
        with pytest.raises(ValueError, match="override"):
            env.advance(decide(env), {"a": 12})
        assert env.snapshot().native_state == original.native_state
        with pytest.raises(ValueError, match="unsupported"):
            replace(decide(env), action_space="continuous")
        with pytest.raises(ValueError, match="probabilities"):
            replace(decide(env), log_probs=np.array([np.nan, 0]))
    finally:
        env.close()


def test_override_and_missing_probability_do_not_enter_on_policy_loss(straight_scene):
    env = TeacherEnvironment(straight_scene)
    try:
        result = env.advance(decide(env), {"a": 1, "b": 7})
        np.testing.assert_array_equal(result.on_policy_eligible, [False, True])
        np.testing.assert_array_equal(result.decision.actions, [7, 7])
        np.testing.assert_array_equal(result.executed_actions, [1, 7])
        assert not env.advance(decide(env, probabilities=False)).on_policy_eligible.any()
    finally:
        env.close()


def test_source_budget_returns_final_state_and_explicit_end_reason(straight_scene):
    scene = copy.deepcopy(straight_scene)
    scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 8]}
    env = TeacherEnvironment(scene)
    try:
        for _ in range(30):
            result = env.advance(decide(env))
        assert result.observation.key.tick == 30
        assert result.observation.episode_done
        assert result.transition.truncated.all()
        np.testing.assert_array_equal(result.end_reasons, [END_REASONS["time_budget"]] * 2)
        assert not result.bootstrap_allowed.any()
        with pytest.raises(RuntimeError, match="done"):
            env.advance(decide(env))
    finally:
        env.close()


def test_goal_is_exposed_as_current_absorbing_semantics(straight_scene):
    env = TeacherEnvironment(straight_scene)
    try:
        reached = False
        while not env.world.episode_done:
            result = env.advance(decide(env))
            goal = result.transition.events["goal"]
            if goal.any():
                reached = True
                assert ((result.end_reasons[goal] & END_REASONS["goal"]) != 0).all()
                assert not result.bootstrap_allowed[goal].any()
        assert reached
    finally:
        env.close()


def test_model_reset_returns_new_identity_while_legacy_reset_stays_compatible(straight_scene):
    env = TeacherEnvironment(straight_scene)
    try:
        old = env.observation()
        observation = env.reset_episode()
        assert observation.key.generation == old.key.generation + 1
        result = env.advance(JointDecision(observation.key, observation.actor_ids,
                                          np.array([7, 7]), "model-reset-v1"))
        assert result.observation.key.tick == 1
        legacy = env.reset()
        assert isinstance(legacy, np.ndarray) and legacy.flags.writeable
        np.testing.assert_array_equal(legacy, old.features)
        assert env.observation().key.generation == observation.key.generation + 1
    finally:
        env.close()
