import copy

import numpy as np
import pytest

from driversi.contracts.driving_task import bind_driving_tasks, route_progress
from driversi.contracts.learning import END_REASONS, JointDecision
from driversi.contracts.road import Route
from driversi.data.scenes.scene_store import source_inventory
from driversi.rendering.bev.replay import export_replay
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi.training.sampling.environment import TeacherEnvironment


def tasks(scene, length=12, destinations=()):
    return bind_driving_tasks(
        scene,
        {a: Route([[0, y], [length, y]]) for a, y in [("a", 0), ("b", 8)]},
        destinations=destinations,
    )


def decision(env):
    obs = env.observation()
    return JointDecision(obs.key, obs.actor_ids, np.array([7, 7]), "test", np.zeros(2))


def test_support_end_is_not_rewarded_success_and_replay_preserves_it(straight_scene, tmp_path):
    scene = tasks(straight_scene, destinations=("b",))
    env = TeacherEnvironment(scene, progress_weight=0)
    try:
        snap = env.snapshot()
        record = RolloutRecord.start(env.world)
        while not env.world.episode_done:
            result = env.advance(decision(env))
            record.append(env.world, dict(zip(env.world.actor_ids, [7, 7])), result.transition)
        assert result.transition.truncated.tolist() == [True, False]
        assert result.transition.terminated.tolist() == [False, True]
        assert result.end_reasons.tolist() == [
            END_REASONS["route_support_end"],
            END_REASONS["goal"],
        ]
        assert result.rewards.tolist() == [0.0, 1.0]
        assert env.world.state()[:2, 12].tolist() == [0.0, 1.0]
        output = tmp_path / "rollout.npz"
        write_rollout(
            scene,
            output,
            record=record,
            snapshot=snap,
            controlled_ids=env.world.actor_ids,
            policy="test",
            controller_configuration={},
        )
        export_replay(scene, output, tmp_path / "viewer.html")
        with np.load(output) as data:
            assert data["route_support_end"][-1].tolist() == [True, False]
    finally:
        env.close()


def test_route_goal_outlives_logged_endpoint_and_snapshot_is_exact(straight_scene, tmp_path):
    scene = tasks(straight_scene, 50)
    env = TeacherEnvironment(scene)
    try:
        for _ in range(4):
            env.advance(decision(env))
        snap = env.snapshot()
        first = env.advance(decision(env))
        state = env.world.state()
        env.restore(snap)
        second = env.advance(decision(env))
        np.testing.assert_array_equal(first.rewards, second.rewards)
        np.testing.assert_array_equal(env.world.state(), state)
        while not env.world.episode_done:
            result = env.advance(decision(env))
        assert result.end_reasons.tolist() == [8, 8]
        assert not result.transition.events["goal"].any()
        assert env.last_task_diagnostics["a"]["status"] == "supported"
        path = tmp_path / "scene.npz"
        scene.save(path)
        spec = source_inventory([str(path)], 0.02)["environment_specs"][0]
        assert spec == env.spec.to_dict()
        altered = tasks(straight_scene, 50, ("a",))
        other = TeacherEnvironment(altered)
        try:
            with pytest.raises(ValueError, match="mismatch"):
                other.restore(snap)
        finally:
            other.close()
    finally:
        env.close()


def test_progress_rewards_forward_motion_even_when_endpoint_distance_grows():
    route = Route([[0, 0], [10, 0], [10, 10], [-10, 10]])
    before, after = np.array([0.0, 0]), np.array([1.0, 0])
    assert np.linalg.norm(after - route.points[-1]) > np.linalg.norm(before - route.points[-1])
    assert route_progress(route, before, after, corridor_m=1)["progress_m"] == 1
    assert route_progress(route, after, before, corridor_m=1)["progress_m"] == -1


def test_shortcut_jump_and_off_route_progress_are_rejected():
    route = Route([[0, 0], [10, 0], [10, 3], [0, 3]])
    result = route_progress(route, [1, 0], [1, 3.4], corridor_m=2)
    assert result["status"] == "projection_jump" and result["progress_m"] == 0
    assert route_progress(route, [1, 10], [2, 10], corridor_m=1)["status"] == "outside_corridor"
    assert (
        route_progress(route, [1, 1.5], [2, 1.5], corridor_m=2)["status"] == "ambiguous_projection"
    )


def test_contract_rejects_wrong_goal_missing_actor_and_implicit_destination(straight_scene):
    scene = tasks(straight_scene)
    altered = copy.deepcopy(scene)
    altered.metadata["task_goals_xy"]["a"] = [20, 0]
    with pytest.raises(ValueError, match="differ"):
        altered.validate()
    altered = copy.deepcopy(scene)
    del altered.metadata["driving_tasks"]["actors"]["b"]
    with pytest.raises(ValueError, match="cover"):
        altered.validate()
    altered = copy.deepcopy(scene)
    altered.metadata["driving_tasks"]["actors"]["a"]["endpoint"] = "destination"
    altered.metadata["driving_tasks"]["actors"]["a"]["route"]["end_reason"] = "branch"
    with pytest.raises(ValueError, match="explicitly"):
        altered.validate()


def test_legacy_reward_and_identity_are_unchanged(straight_scene):
    env = TeacherEnvironment(straight_scene)
    try:
        before = env.world.state()[:2]
        result = env.advance(decision(env))
        after = env.world.state()[:2]
        expected = 0.02 * (
            np.linalg.norm(before[:, :2] - before[:, 7:9], axis=1)
            - np.linalg.norm(after[:, :2] - after[:, 7:9], axis=1)
        )
        np.testing.assert_allclose(result.rewards, expected, rtol=1e-6)
        assert "route_support_end" not in result.transition.events
        from driversi.rendering.bev.replay import _scene_key

        assert _scene_key(straight_scene) == env.snapshot().scene_key
    finally:
        env.close()
