import numpy as np
import pytest

from driversi.agents.traffic_control.controllers import ConstantJerk, JointController
from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower, idm_acceleration
from driversi.contracts.road import Route
from driversi.contracts.transition import MetricObservation, Observation


def observation():
    return Observation(("follower", "lead"), np.zeros((2, 1124)), np.ones(2, bool), 0,
                       MetricObservation(np.asarray([[0, 0, 0, 8, 0, 4, 2, 0, 0, 0],
                                                      [25, 0, 0, 8, 0, 4, 2, 0, 0, 0]], float),
                                         np.asarray([[[25, 0, 0, 8, 0, 4, 2]],
                                                     [[0, 0, 0, 8, 0, 4, 2]]], float),
                                         np.ones((2, 1), bool), (("lead",), ("follower",))))


def test_idm_gap_and_closing_velocity_have_correct_direction():
    config = FollowingConfig()
    baseline = idm_acceleration(8, 8, 30, 0, config)
    assert idm_acceleration(8, 8, 20, 0, config) < baseline
    assert idm_acceleration(8, 8, 30, 3, config) < baseline
    assert idm_acceleration(4, 8, None, 0, config) > 0
    with pytest.raises(ValueError):
        FollowingConfig(headway=0)


def test_reactive_target_and_irrelevant_lane_negative_control():
    policy = RouteFollower({"follower": Route([[0, 0], [200, 0]]),
                            "lead": Route([[0, 0], [200, 0]])})
    obs = observation()
    original = policy.commands(obs)[0]
    obs.metric.neighbors[0, 0, 3] = 5
    assert policy.commands(obs)[0]["desired_acceleration"] < original["desired_acceleration"]
    obs.metric.neighbors[0, 0, 1] = 8
    assert policy.commands(obs)[0]["lead_id"] is None
    obs.metric.neighbors[0, 0, 3] = 2
    assert policy.commands(obs)[0]["lead_id"] is None


def test_joint_dispatch_keeps_current_metric_and_order():
    policy = RouteFollower({"follower": Route([[0, 0], [200, 0]])})
    first = JointController({"follower": policy, "lead": ConstantJerk()})
    second = JointController({"lead": ConstantJerk(), "follower": policy})
    assert first.act(observation()) == second.act(observation())


def test_ego_footprint_is_projected_when_heading_differs():
    policy = RouteFollower({"follower": Route([[0, 0], [200, 0]]),
                            "lead": Route([[0, 0], [200, 0]])})
    obs = observation()
    obs.metric.ego[0, 2] = np.pi/3
    command = policy.commands(obs)[0]
    ego_extent = .5*(4*.5 + 2*np.sqrt(3)/2)
    assert command["bumper_gap_m"] == pytest.approx(25-2-ego_extent)
