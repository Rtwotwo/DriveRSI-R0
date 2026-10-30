import copy

import numpy as np
import pytest

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.contracts.road import Route
from driversi.evaluation.search.augmentation import generate_pair
from driversi.evaluation.search.lateral_search import SearchBudget, SearchBudgetExceeded, StepAccount
from driversi.sim.backends.pufferdrive import PufferDriveWorld


def following_case(scene):
    scene = copy.deepcopy(scene)
    scene.poses[:, 0, 0] += 16
    scene.poses[:, 1, 1] = 0
    scene.metadata["task_goals_xy"] = {"a": [100, 0], "b": [100, 0]}
    routes = {a: Route([[-40, 0], [140, 0]]) for a in ("a", "b")}
    return scene, routes


def test_two_stages_restore_and_other_agents_respond(straight_scene):
    scene, routes = following_case(straight_scene)
    config = FollowingConfig(desired_speed=8.0)
    report, records, _, boundaries = generate_pair(
        scene, routes, actor="a", explore_steps=15, explore_speed=1.0, config=config
    )
    assert report["changed_actor_actions"]
    assert "b" in report["responding_actors"]
    np.testing.assert_array_equal(records[0].states[0], records[1].states[0])
    assert not np.array_equal(records[0].states[15], records[1].states[15])
    controller = RouteFollower(routes, config)
    with PufferDriveWorld(scene) as world:
        world.restore(boundaries[1])
        for tick in range(15, len(records[1].actions)):
            actions = {c["actor"]: c["action"] for c in controller.commands(world.observe())}
            assert [actions[a] for a in world.actor_ids] == records[1].actions[tick]
            world.step(actions)
            np.testing.assert_array_equal(world.state(), records[1].states[tick + 1])
    assert report["official_score"] is None and not report["trainable"]


def test_equal_speed_negative_control(straight_scene):
    scene, routes = following_case(straight_scene)
    report, records, _, _ = generate_pair(
        scene, routes, actor="a", explore_steps=15, explore_speed=8.0
    )
    np.testing.assert_array_equal(records[0].states, records[1].states)
    assert not report["responding_actors"] and not report["changed_actor_actions"]


def test_budget_and_test_split(straight_scene):
    scene, routes = following_case(straight_scene)
    account = StepAccount(SearchBudget(max_joint_steps=10))
    with pytest.raises(SearchBudgetExceeded):
        generate_pair(scene, routes, actor="a", explore_steps=15, step_guard=account.consume)
    assert account.steps == 10
    scene.metadata["split"] = "navtest"
    with pytest.raises(ValueError):
        generate_pair(scene, routes, actor="a", explore_steps=15)
