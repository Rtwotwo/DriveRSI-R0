"""Finite-jerk route tracking on independently constructed curved corridors."""

import json

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.contracts.road import RoadGraph, Route
from driversi.contracts.scene import LANE, ROAD_EDGE, Scene
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.execution.rollout import rollout

pytestmark = pytest.mark.native


def test_rollout_records_the_decision_actually_executed(straight_scene, tmp_path, monkeypatch):
    graph = RoadGraph(
        {"schema_version": 1, "map_name": "straight", "map_version": None,
         "origin_xyz": [0, 0, 0], "bounds_local": [-40, -20, 60, 20]},
        {f"lane:{actor}": {"kind": "lane", "points": [[-40, y], [60, y]],
                          "outgoing": [], "speed_limit_mps": None}
         for actor, y in (("a", 0), ("b", 8))},
    )
    decisions = []

    def non_idempotent_commands(self, observation):
        # Re-evaluating one observation would produce a different executed action.
        action = 10 if len(decisions) % 2 == 0 else 7
        commands = [{"actor": actor, "action": action} for actor in observation.actor_ids]
        decisions.append(commands)
        return commands

    monkeypatch.setattr(RouteFollower, "commands", non_idempotent_commands)
    output = tmp_path / "decisions.npz"
    rollout(straight_scene, output, policy="follow", road_graph=graph)
    with np.load(output, allow_pickle=False) as data:
        diagnostics = json.loads(str(data["controller_diagnostics"]))
        assert len(diagnostics) > 1
        assert diagnostics == decisions
        np.testing.assert_array_equal(data["actions"],
                                      [[command["action"] for command in frame]
                                       for frame in diagnostics])
        with PufferDriveWorld(straight_scene) as replay:
            for step, actions in enumerate(data["actions"], 1):
                replay.step(actions)
                np.testing.assert_array_equal(replay.state(), data["states"][step])
                np.testing.assert_array_equal(replay.motion(), data["motion"][step])


def curved_corridor(direction: int, radius: float, initial_speed: float = 8, half_width: float = 2.25):
    straight = np.column_stack((np.arange(-30, 11), np.zeros(41)))
    angle = np.linspace(0, 3.0, 240)
    arc = np.column_stack((10+radius*np.sin(angle), direction*radius*(1-np.cos(angle))))
    points = np.vstack((straight, arc[1:]))
    headings = np.r_[np.zeros(len(straight)), direction*angle[1:]]
    normals = np.column_stack((-np.sin(headings), np.cos(headings)))
    left, right = points+half_width*normals, points-half_width*normals
    lines = [points, left, right]
    road_points = np.column_stack((np.vstack(lines), np.zeros(sum(map(len, lines)))))
    times = np.arange(91)*.1
    poses = np.zeros((len(times), 1, 4))
    poses[:, 0, 0] = initial_speed*times
    velocities = np.zeros((len(times), 1, 3))
    velocities[:, 0, 0] = initial_speed
    scene = Scene(
        {"schema_version": 1, "scenario_id": "curved-corridor", "source_log": "synthetic",
         "split": "canonical", "map_name": "curve", "origin_xyz": [0, 0, 0],
         "map_complete": True, "task_goals_xy": {"car": points[-1].tolist()}},
        ("car",), np.ones(1, np.int32), np.asarray([[4.5, 1.8, 1.5]]), times, poses,
        velocities, np.ones(1, bool), road_points, np.r_[0, np.cumsum(list(map(len, lines)))],
        np.asarray([LANE, ROAD_EDGE, ROAD_EDGE]))
    return scene, Route(points)


@pytest.mark.parametrize("direction,radius", [(-1, 20), (1, 20), (-1, 35), (1, 35)])
def test_curved_corridor_progress_without_departure(direction, radius):
    scene, route = curved_corridor(direction, radius)
    follower = RouteFollower({"car": route})
    errors, speeds = [], []
    with PufferDriveWorld(scene) as world:
        while not world.episode_done:
            command = follower.commands(world.observe())[0]
            errors.append(command["route_distance_m"])
            speeds.append(command["actual_speed_mps"])
            transition = world.step({"car": command["action"]})
            assert not transition.events["collision"].any()
            assert not transition.events["offroad"].any()
        progress = route.project(world.state()[0, :2])[0]-route.project(scene.poses[0, 0, :2])[0]
    assert max(errors) < .8
    assert progress > 40  # Avoid passing merely by stopping before the curve.
    assert min(speeds) > 2
    if radius == 20:
        assert min(speeds) < 7.5  # Requires speed adaptation to the tight bend.


def test_misalignment_uses_vehicle_speed_and_limits_acceleration():
    scene, route = curved_corridor(1, 35, initial_speed=1.5)
    scene.poses[:, 0, 1] = .3
    scene.poses[:, 0, 3] = -.4
    scene.velocities[:, 0, :2] = 1.5*np.asarray([np.cos(-.4), np.sin(-.4)])
    follower = RouteFollower({"car": route}, FollowingConfig(desired_speed=8))
    with PufferDriveWorld(scene) as world:
        first = follower.commands(world.observe())[0]
        assert first["actual_speed_mps"] == pytest.approx(1.5, abs=1e-5)
        assert first["path_speed_mps"] < first["actual_speed_mps"]
        assert first["recovery_mode"]
        assert first["target_speed_mps"] <= first["actual_speed_mps"]
        while not world.episode_done:
            command = follower.commands(world.observe())[0]
            result = world.step({"car": command["action"]})
            assert not result.events["offroad"].any()
        final = world.state()[0]
        assert route.project(final[:2])[1] < .4
        assert route.project(final[:2])[0] - route.project(scene.poses[0, 0, :2])[0] > 10
        assert np.linalg.norm(final[4:6]) > .5


def test_low_speed_braking_releases_before_unintended_reverse():
    scene = following_scene(1.5, 2.5)
    scene.poses[:, 1, 0] = 6.5
    scene.velocities[:, 1] = 0
    routes = {actor: Route([[-30, 80 if actor == "far" else 0],
                            [250, 80 if actor == "far" else 0]]) for actor in scene.actor_ids}
    follower = RouteFollower(routes)
    with PufferDriveWorld(scene) as world:
        start = world.state()[0, 0]
        while not world.episode_done:
            command = follower.commands(world.observe())[0]
            assert command["forward_release_feasible"]
            result = world.step({"follower": command["action"], "lead": 7, "far": 7})
            state = world.state()
            assert state[0, 4] >= -1e-5
            assert state[1, 0]-state[0, 0]-4 > 0
            assert not result.events["collision"].any()
        assert world.state()[0, 0]-start > .2
        assert world.state()[0, 4] < .2


def test_low_speed_large_offset_recovers_without_parking():
    scene, route = curved_corridor(1, 35, initial_speed=1.5, half_width=4.5)
    scene.poses[:, 0, 1] = 2.3
    scene.poses[:, 0, 3] = .116
    scene.velocities[:, 0, :2] = 1.5*np.asarray([np.cos(.116), np.sin(.116)])
    follower = RouteFollower({"car": route})
    with PufferDriveWorld(scene) as world:
        initial_progress = route.project(world.state()[0, :2])[0]
        while not world.episode_done:
            command = follower.commands(world.observe())[0]
            transition = world.step({"car": command["action"]})
            assert not transition.events["offroad"].any()
            assert not transition.events["collision"].any()
        final = world.state()[0]
        assert route.project(final[:2])[1] < .5
        assert route.project(final[:2])[0]-initial_progress > 10
        assert np.linalg.norm(final[4:6]) > 1


def test_infeasible_external_braking_state_is_reported():
    scene, route = curved_corridor(1, 35)
    with PufferDriveWorld(scene) as world:
        observation = world.observe()
        observation.metric.ego[0, 3:5] = [.1, 0]
        observation.metric.ego[0, 7] = -5
        command = RouteFollower({"car": route}).commands(observation)[0]
        assert not command["forward_release_feasible"]
        assert command["predicted_minimum_release_speed_mps"] < 0
        assert command["action"]//3 == 3  # Maximum release jerk, with no state rewriting.


def test_exhausted_route_brakes_without_chasing_endpoint_behind():
    scene, _ = curved_corridor(1, 35, half_width=4.5)
    scene.poses[:, 0, 0] = 0
    scene.poses[:, 0, 1] = .4
    scene.velocities[:, 0, 0] = 4
    route = Route([[-30, 0], [-2, 0]], end_reason="branch")
    follower = RouteFollower({"car": route})
    with PufferDriveWorld(scene) as world:
        first = follower.commands(world.observe())[0]
        assert first["route_exhausted"]
        assert first["route_end_signed_distance_m"] == pytest.approx(2, abs=1e-5)
        assert first["tracking_reference"] == "terminal_tangent"
        assert first["lookahead_forward_m"] > 0
        assert first["desired_acceleration"] < 0
        for _ in range(30):
            command = follower.commands(world.observe())[0]
            transition = world.step({"car": command["action"]})
            assert command["route_exhausted"]
            assert not transition.events["offroad"].any()
            assert not transition.events["goal"].any()  # Exhaustion is not task success.
            assert abs(world.state()[0, 3]) < .3
            assert world.state()[0, 4] >= -1e-5
        assert np.linalg.norm(world.state()[0, 4:6]) < .3
