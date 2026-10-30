"""Vehicle interaction with physical replay actors on a straight, level road.

These cases verify obstacle observation and geometric collision participation.
The replay pedestrian/cyclist does not respond, and no crossing or right-of-way
policy is implied by the vehicle follower's generic in-path obstacle response.
"""

import numpy as np
import pytest

pytest.importorskip("driversi.sim._native")

from driversi.agents.traffic_control.following import RouteFollower
from driversi.contracts.road import Route
from driversi.contracts.scene import CYCLIST, LANE, PEDESTRIAN, ROAD_EDGE, VEHICLE, Scene
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


def replay_scene(kind: int, lateral_position: float) -> Scene:
    times = np.arange(31) * .1
    poses = np.zeros((len(times), 2, 4))
    poses[:, 0, 0] = 4 * times
    poses[:, 1, 0] = 8 + times
    poses[:, 1, 1] = lateral_position
    velocities = np.zeros((len(times), 2, 3))
    velocities[:, :, 0] = [4, 1]
    return Scene(
        metadata={"schema_version": 1, "scenario_id": "replay-counterpt",
                  "source_log": "synthetic", "split": "canonical", "map_name": "straight",
                  "origin_xyz": [0, 0, 0], "map_complete": True,
                  "task_goals_xy": {"car": [100, 0]}},
        actor_ids=("car", "counterpart"), kinds=np.asarray([VEHICLE, kind], np.int32),
        dimensions=np.asarray([[4, 2, 1.5], [.6 if kind == PEDESTRIAN else 1.8, .6, 1.7]]),
        times=times, poses=poses, velocities=velocities,
        controlled=np.asarray([True, False]),
        road_points=np.asarray([[-40, 0, 0], [120, 0, 0], [-40, -20, 0], [120, -20, 0],
                                [120, 20, 0], [-40, 20, 0], [-40, -20, 0]]),
        road_offsets=np.asarray([0, 2, 7]), road_types=np.asarray([LANE, ROAD_EDGE]),
    )


@pytest.mark.parametrize("kind", [PEDESTRIAN, CYCLIST], ids=["pedestrian", "cyclist"])
def test_replay_actor_is_observed_and_changes_vehicle_following(kind):
    policy = RouteFollower({"car": Route([[-40, 0], [120, 0]])})
    with PufferDriveWorld(replay_scene(kind, 0)) as ahead, \
            PufferDriveWorld(replay_scene(kind, 8)) as alongside:
        for world in (ahead, alongside):
            observation = world.observe()
            assert observation.actor_ids == ("car",)
            assert observation.metric.visible.tolist() == [[False, True]]
            assert observation.metric.neighbor_ids == (("", "counterpart"),)
            np.testing.assert_allclose(observation.metric.neighbors[0, 1, 3:5], [1, 0])
        ahead_command = policy.commands(ahead.observe())[0]
        alongside_command = policy.commands(alongside.observe())[0]
        assert ahead_command["lead_id"] == "counterpart"
        assert alongside_command["lead_id"] is None
        assert ahead_command["desired_acceleration"] < alongside_command["desired_acceleration"]
        assert ahead_command["action"] != alongside_command["action"]
        ahead.step({"car": ahead_command["action"]})
        alongside.step({"car": alongside_command["action"]})
        assert ahead.motion()[0, 0] < alongside.motion()[0, 0]


@pytest.mark.parametrize("kind", [PEDESTRIAN, CYCLIST], ids=["pedestrian", "cyclist"])
@pytest.mark.parametrize("lateral_position,expected_collision", [(0, True), (8, False)])
def test_vehicle_collision_with_replay_agrees_with_independent_geometry(
        kind, lateral_position, expected_collision):
    scene = replay_scene(kind, lateral_position)
    collided = False
    with PufferDriveWorld(scene) as world:
        assert not world.state()[:, 10].any()
        while not world.episode_done:
            transition = world.step({"car": 7})
            state = world.state()
            boxes = [rectangle_corners(*state[i, :2], state[i, 3], *scene.dimensions[i, :2])
                     for i in (0, 1)]
            overlap = rectangles_overlap(*boxes)
            assert bool(transition.events["collision"][0]) == overlap
            assert bool(transition.terminated[0]) == overlap
            assert not transition.events["offroad"].any()
            assert not transition.events["goal"].any()
            collided |= overlap
        assert collided == expected_collision


@pytest.mark.parametrize("kind", [PEDESTRIAN, CYCLIST], ids=["pedestrian", "cyclist"])
def test_replay_trajectory_does_not_react_to_vehicle_actions(kind):
    scene = replay_scene(kind, 8)
    with PufferDriveWorld(scene) as coast, PufferDriveWorld(scene) as brake:
        for step in range(1, len(scene.times)):
            coast_result = coast.step({"car": 7})
            brake_result = brake.step({"car": 1})
            coast_state, brake_state = coast.state(), brake.state()
            np.testing.assert_array_equal(coast_state[1], brake_state[1])
            for state in (coast_state, brake_state):
                np.testing.assert_allclose(state[1, :4], scene.poses[step, 1], atol=1e-4, rtol=0)
                np.testing.assert_allclose(state[1, 4:7], scene.velocities[step, 1], atol=1e-6, rtol=0)
            for result in (coast_result, brake_result):
                assert not result.terminated.any()
                assert result.episode_done == (step == len(scene.times) - 1)
        assert abs(coast_state[0, 0] - brake_state[0, 0]) > 1
