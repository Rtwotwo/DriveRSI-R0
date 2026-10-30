"""Sparse annotation changes must not become invented vehicle pirouettes."""

import numpy as np
import pytest

from driversi.data.participants.motion import source_vehicle_headings


def test_repeated_half_turn_annotations_preserve_axis_and_record_every_correction():
    poses = np.zeros((5, 1, 4))
    poses[:, 0, 0] = [0, .1, .2, .3, .4]
    poses[:, 0, 3] = np.deg2rad([179, 1, 2, -179, -178])
    original = poses.copy()
    times = np.arange(5) * .5
    yaw, audit = source_vehicle_headings(poses, times, ("parked",), np.array([1]))
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_allclose(np.rad2deg(yaw[:, 0]), [179, 181, 182, 181, 182])
    assert yaw[0, 0] == poses[0, 0, 3]
    assert len(audit["corrections"]) == 2
    first = audit["corrections"][0]
    assert first["actor_id"] == "parked"
    assert (first["source_start_index"], first["source_end_index"]) == (0, 1)
    assert (first["start_time_seconds"], first["end_time_seconds"]) == (0., .5)
    assert first["centre_speed_mps"] == pytest.approx(.2)
    assert first["raw_wrapped_delta_radians"] == pytest.approx(np.deg2rad(-178))
    assert first["corrected_delta_radians"] == pytest.approx(np.deg2rad(2))
    # Intermediate 10 Hz geometry does not rotate towards a perpendicular box.
    interpolated = np.interp(np.arange(21) * .1, times, yaw[:, 0])
    assert np.max(abs(np.diff(interpolated))) < np.deg2rad(1)


def test_real_slow_turn_and_numeric_pi_boundary_are_not_frozen():
    poses = np.zeros((4, 1, 4))
    poses[:, 0, 3] = np.deg2rad([170, 179, -175, -160])
    yaw, audit = source_vehicle_headings(poses, np.arange(4) * .5, ("car",), np.array([1]))
    np.testing.assert_allclose(np.rad2deg(yaw[:, 0]), [170, 179, 185, 200])
    assert audit["corrections"] == []


def test_fast_vehicle_pedestrian_cyclist_and_ego_do_not_use_box_ambiguity_rule():
    poses = np.zeros((2, 4, 4))
    poses[1, :, 3] = np.deg2rad(179)
    poses[1, 0, 0] = .501  # Just above the 1 m/s source-interval threshold.
    yaw, audit = source_vehicle_headings(
        poses, np.array([0., .5]), ("fast", "walker", "bike", "ego"), np.array([1, 2, 3, 1]))
    np.testing.assert_allclose(yaw, poses[..., 3])
    assert audit["corrections"] == []


def test_speed_uses_timestamp_difference_not_assumed_source_fps():
    poses = np.zeros((2, 1, 4))
    poses[1, 0, 0] = .8
    poses[1, 0, 3] = np.pi
    _, slower = source_vehicle_headings(poses, np.array([0., 1.]), ("car",), np.array([1]))
    _, faster = source_vehicle_headings(poses, np.array([0., .5]), ("car",), np.array([1]))
    assert slower["corrections"][0]["centre_speed_mps"] == pytest.approx(.8)
    assert faster["corrections"] == []


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, np.nan]])
def test_invalid_time_axis_is_rejected(times):
    with pytest.raises(ValueError, match="source times"):
        source_vehicle_headings(np.zeros((2, 1, 4)), np.array(times), ("car",), np.array([1]))
