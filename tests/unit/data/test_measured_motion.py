import copy

import pytest

from driversi.data.participants.background import configure_background, propose_logged_tasks


@pytest.mark.parametrize("kind", [1, 2, 3])
def test_annotation_jitter_does_not_become_a_driving_intent(straight_scene, kind):
    s = copy.deepcopy(straight_scene)
    s.kinds[2] = kind
    s.metadata["first_observations"] = {"replay": {"velocity_world": [0.3, 0.0, 0.0]}}
    s.metadata["source_motion_evidence"] = dict(
        version=1,
        scope="first_contiguous_run_privileged_navigation",
        actors={
            "replay": dict(
                samples=40,
                finite_velocity_samples=40,
                first_timestamp_us=0,
                last_timestamp_us=3_000_000,
                reported_travel_m=0.02,
                position_path_length_m=37.0,
            )
        },
    )
    tasks, rejected = propose_logged_tasks(s)
    assert tasks["replay"]["navigation_source"] == "uncertain_motion_braking"
    assert tasks["replay"]["desired_speed"] == 0
    assert rejected["replay"] == "position_motion_contradicts_reported_travel"
    assert tasks["replay"]["points"][0] == s.poses[0, 2, :2].tolist()
    before = s.velocities.copy()
    configured = configure_background(s, tasks)
    assert configured.velocities[0, 2].tolist() == [0.3, 0.0, 0.0]
    assert (s.velocities == before).all()


def test_consistent_slow_motion_and_unknown_measurement_are_not_declared_jitter(straight_scene):
    for measured_travel, valid_samples in [(1.0, 40), (0.02, 39)]:
        s = copy.deepcopy(straight_scene)
        s.metadata["first_observations"] = {"replay": {"velocity_world": [0.3, 0.0, 0.0]}}
        s.metadata["source_motion_evidence"] = dict(
            version=1,
            scope="first_contiguous_run_privileged_navigation",
            actors={
                "replay": dict(
                    samples=40,
                    finite_velocity_samples=valid_samples,
                    first_timestamp_us=0,
                    last_timestamp_us=3_000_000,
                    reported_travel_m=measured_travel,
                    position_path_length_m=37.0,
                )
            },
        )
        _, rejected = propose_logged_tasks(s)
        assert rejected.get("replay") != "position_motion_contradicts_reported_travel"


@pytest.mark.parametrize("kind", [1, 2, 3, 10])
def test_continuous_fallback_uses_measured_motion_for_every_object(straight_scene, kind):
    import numpy as np

    from driversi.contracts.road import RoadGraph
    from driversi.data.participants.fleet import configure_vehicle_fleet
    from driversi.data.participants.traffic_continuity import configure_traffic_continuity

    source = copy.deepcopy(straight_scene)
    source.poses[:, 2] = source.poses[0, 2]
    source.velocities[:, 2, 0] = 3.0  # untrusted position-difference artefact
    source.metadata["first_observations"] = {"replay": {"velocity_world": [0.0, 0.0, 0.0]}}
    source.metadata["map_version"] = "fixture"
    source.metadata["map_bounds_global"] = [-100, -100, 100, 100]
    scene = configure_vehicle_fleet(source)
    scene.kinds[2] = kind
    scene.metadata["reactive_background"]["assignments"].pop("replay")
    if kind != 1:
        scene.metadata["vehicle_fleet"]["roles"].pop("replay")
    scene.velocities[:, 2, 0] = 3.0
    graph = RoadGraph(
        dict(
            schema_version=1,
            map_name=scene.metadata["map_name"],
            map_version="fixture",
            origin_xyz=scene.metadata["origin_xyz"],
            bounds_local=[-100, -100, 100, 100],
        ),
        {"lane:1": dict(kind="lane", points=[[0, 0], [20, 0]], outgoing=[], speed_limit_mps=None)},
    )
    result = configure_traffic_continuity(scene, graph)
    task = result.metadata["reactive_background"]["assignments"]["replay"]
    assert task["navigation_source"] == "observed_stationary_hold"
    assert result.velocities[0, 2].tolist() == [0.0, 0.0, 0.0]
    assert (
        result.metadata["traffic_continuity"]["navigation"]["replay"]["spawn_velocity_source"]
        == "measured"
    )
    assert np.array_equal(result.poses, scene.poses)
    assert scene.velocities[0, 2, 0] == 3.0
