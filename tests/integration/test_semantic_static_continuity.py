"""Native static categories are fixed objects, not low-speed moving agents."""

import copy

import numpy as np
import pytest

from driversi.contracts.scene import OTHER_OBJECT
from driversi.data.participants.traffic_continuity import configure_traffic_continuity
from driversi.sim.backends.pufferdrive import PufferDriveWorld

pytestmark = pytest.mark.native


class NoMovingRouteGraph:
    def validate_scene(self, scene):
        scene.validate()


def source_case(straight_scene, category, spawn):
    scene = copy.deepcopy(straight_scene)
    scene.kinds[2] = OTHER_OBJECT
    scene.poses[:, 2] = [10.0, -8.0, 0.0, 0.0]
    scene.velocities[:, 2] = [0.004, 0.003, 0.0]
    scene.presence = np.ones(scene.poses.shape[:2], bool)
    scene.presence[:spawn, 2] = False
    scene.metadata["source_actor_kinds"] = {"replay": category}
    scene.metadata["first_observations"] = {
        "replay": {"velocity_world": [0.004, 0.003, 0.0]}
    }
    scene.metadata["origin_xyz"] = [0.0, 0.0, 0.0]
    scene.metadata["map_bounds_global"] = [-100.0, -100.0, 100.0, 100.0]
    return scene


@pytest.mark.parametrize("category", ["traffic_cone", "barrier", "czone_sign", "generic_object"])
@pytest.mark.parametrize("spawn", [0, 5])
def test_native_static_semantics_hold_all_steps_and_restore(straight_scene, category, spawn):
    source = source_case(straight_scene, category, spawn)
    scene = configure_traffic_continuity(source, NoMovingRouteGraph())
    assert scene.actor_ids == source.actor_ids
    assert source.metadata["first_observations"]["replay"]["velocity_world"] == [0.004, 0.003, 0.0]
    evidence = scene.metadata["traffic_continuity"]["navigation"]["replay"]["semantic_static"]
    assert evidence["source_category"] == category
    assert evidence["measured_spawn_velocity_world"] == [0.004, 0.003, 0.0]
    with PufferDriveWorld(scene) as world:
        initial = world.snapshot()
        positions = []
        for _ in range(25):
            world.step([7, 7])
            if world.presence()[2]:
                positions.append(world.state()[2, :4].copy())
        assert positions
        np.testing.assert_array_equal(positions, np.broadcast_to(positions[0], np.shape(positions)))
        assert world.presence()[2]
        final = world.snapshot()
        world.restore(initial)
        for _ in range(25):
            world.step([7, 7])
        assert world.snapshot() == final


@pytest.mark.parametrize("category", ["unknown", "other_object", "movable_debris"])
def test_unknown_objects_keep_measured_motion(straight_scene, category):
    source = source_case(straight_scene, category, 0)
    scene = configure_traffic_continuity(source, NoMovingRouteGraph())
    evidence = scene.metadata["traffic_continuity"]["navigation"]["replay"]
    assert "semantic_static" not in evidence
    np.testing.assert_array_equal(scene.velocities[0, 2], source.velocities[0, 2])
    with PufferDriveWorld(scene) as world:
        before = world.state()[2, :2].copy()
        world.step([7, 7])
        assert np.linalg.norm(world.state()[2, :2] - before) > 0
