import copy

import numpy as np
import pytest

from driversi.contracts.scene import LANE, ROAD_EDGE
from driversi.sim.backends.pufferdrive import PufferDriveWorld


def boundary_scene(scene, *, profile=True, hole=False):
    s = copy.deepcopy(scene)
    s.controlled[:] = [True, False, False]
    if profile:
        s.metadata.update(
            road_boundary_policy="nuplan_strict_corners_v1",
            road_edge_model="clipped_official_nuplan_drivable_union_v2",
        )
    # Thin top indentation intersects the long vehicle side but no corner.
    ring = [
        [-40, -20],
        [60, -20],
        [60, 1.1],
        [0.1, 1.1],
        [0, 0.9],
        [-0.1, 1.1],
        [-40, 1.1],
        [-40, -20],
    ]
    lines = [np.array([[-40, 0, 0], [60, 0, 0]]), np.column_stack([ring, np.zeros(len(ring))])]
    if hole:
        ring2 = [[7, -3], [13, -3], [13, 3], [7, 3], [7, -3]]
        lines[1][:, 1] = np.where(lines[1][:, 1] > 0, 20, -20)
        lines.append(np.column_stack([ring2, np.zeros(5)]))
    s.road_points = np.concatenate(lines)
    s.road_offsets = np.r_[0, np.cumsum([len(line) for line in lines])]
    s.road_types = np.array([LANE] + [ROAD_EDGE] * (len(lines) - 1))
    return s


def step_target(world, x=0.4, y=0):
    return world.step_native(
        {"a": [x, y, 0, 0, 4, 0, 0, 0, 0, 0]}, observation_tick=world.observe().step
    )


def test_perimeter_contact_stays_visible_but_inside_corners_continue(straight_scene):
    with PufferDriveWorld(boundary_scene(straight_scene)) as world:
        snap = world.snapshot()
        result = step_target(world)
        assert result.events["road_boundary_contact"][0]
        assert not result.events["offroad"][0]
        assert result.observation.alive[0]
        state = world.state()
        world.restore(snap)
        other = step_target(world)
        np.testing.assert_array_equal(world.state(), state)
        np.testing.assert_array_equal(
            other.events["road_boundary_contact"], result.events["road_boundary_contact"]
        )


def test_legacy_profile_still_stops_perimeter_contact(straight_scene):
    with PufferDriveWorld(boundary_scene(straight_scene, profile=False)) as world:
        assert world.state()[0, 11] == 1
        assert not world.observe().alive[0]


@pytest.mark.parametrize("x,y,hole", [(0.4, 0.3, False), (0.4, 14, False), (10, 0, True)])
def test_corner_outside_or_inside_hole_stops_without_tolerance(straight_scene, x, y, hole):
    with PufferDriveWorld(boundary_scene(straight_scene, hole=hole)) as world:
        result = step_target(world, x, y)
        assert result.events["offroad"][0]
        assert not result.observation.alive[0]


def test_vehicle_collision_not_exempted_by_corner_profile(straight_scene):
    s = boundary_scene(straight_scene, hole=True)
    s.poses[:, 2, :2] = [5, 0]
    s.velocities[:, 2] = 0
    with PufferDriveWorld(s) as world:
        result = step_target(world, 1.2)
        assert result.events["collision"][0]
        assert not result.observation.alive[0]


def test_open_or_unknown_boundary_profile_is_rejected(straight_scene):
    s = boundary_scene(straight_scene)
    s.road_points[-1, 0] += 1
    with pytest.raises(ValueError, match="closed boundary"):
        PufferDriveWorld(s)
    s.metadata["road_boundary_policy"] = "ignore"
    with pytest.raises(ValueError, match="Unsupported road"):
        PufferDriveWorld(s)


def test_snapshot_profiles_cannot_be_mixed(straight_scene):
    with PufferDriveWorld(boundary_scene(straight_scene)) as a:
        with PufferDriveWorld(boundary_scene(straight_scene, profile=False)) as b:
            with pytest.raises(ValueError, match="Snapshot scene"):
                b.restore(a.snapshot())


def test_corner_profile_record_replays_contact_and_rejects_missing_evidence(
    straight_scene, tmp_path
):
    from driversi.rendering.bev.replay import _load_run, _scene_key, export_replay
    from driversi.sim.persistence.recording import RolloutRecord, write_rollout

    s = boundary_scene(straight_scene)
    with PufferDriveWorld(s) as world:
        record = RolloutRecord.start(world)
        initial = world.snapshot()
        tr = step_target(world)
        record.append(world, {"a": [0.0, 0.0]}, tr)
        target = tmp_path / "rollout.npz"
        write_rollout(
            s,
            target,
            record=record,
            snapshot=initial,
            controlled_ids=world.actor_ids,
            policy="test",
            controller_configuration={"action_space": "driverl_native_jerk_steering_rate"},
        )
    assert _scene_key(s) == initial.scene_key
    result = _load_run(s, target, _scene_key(s))
    assert "road_boundary_contact" in result["frames"][1]["arrivals"]["a"]
    export_replay(s, target, tmp_path / "viewer.html")
    with np.load(target, allow_pickle=False) as bundle:
        arrays = {k: bundle[k] for k in bundle.files if k != "road_boundary_contact"}
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError, match="missing perimeter contact"):
        _load_run(s, target, _scene_key(s))
