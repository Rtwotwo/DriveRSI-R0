import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_multiview import (
    camera_input_for_target,
    novel_view_support,
    roof_front_camera,
    virtual_camera_owners,
)


def test_roof_mount_uses_box_centre_height_and_forward_optical_axis():
    pose, rig = roof_front_camera([10, 20, 1], [1, 0, 0, 0], [4, 2, 2])
    np.testing.assert_allclose(pose[:, 3], [11.4, 20, 2.2])
    np.testing.assert_allclose(-pose[:, 2], [1, 0, 0])
    assert rig["body_reference"] == "reconstructed_3d_box_center"
    assert rig["camera_to_body_optical"][2][3] == pytest.approx(1.2)
    rotated, _ = roof_front_camera([10, 20, 1], [2**-.5, 0, 0, 2**-.5], [4, 2, 2])
    np.testing.assert_allclose(rotated[:, 3], [10, 21.4, 2.2], atol=1e-12)
    np.testing.assert_allclose(-rotated[:, 2], [0, 1, 0], atol=1e-12)


def test_optimizer_inverse_is_world_axis_rotation_and_independent_translation():
    source = np.eye(4)
    source[:3, 3] = [12, 13, 14]
    adjustment = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    corrected = source.copy()
    corrected[:3, :3] = adjustment @ source[:3, :3]
    corrected[:3, 3] += [1, 2, 3]
    target, _ = roof_front_camera([8, 9, 10], [1, 0, 0, 0], [4, 2, 2])
    inverse = camera_input_for_target(source, corrected, target)
    np.testing.assert_allclose(adjustment @ inverse[:, :3], target[:, :3], atol=1e-12)
    np.testing.assert_allclose(inverse[:, 3]+[1, 2, 3], target[:, 3], atol=1e-12)
    np.testing.assert_allclose(source[:3, 3], [12, 13, 14])


@pytest.mark.parametrize("owners", [("ego",), ("unknown",), ("parked",), ("car", "car"), ("../car",)])
def test_virtual_owner_requires_controlled_bound_unique_safe_non_ego_actor(owners):
    with pytest.raises(ValueError):
        virtual_camera_owners(owners, ("ego", "car", "parked"), (True, True, False), {"car", "parked"})
    assert virtual_camera_owners(("car",), ("ego", "car"), (True, True), {"car"}) == ("car",)


def test_missing_asset_and_novel_view_quality_never_become_valid_from_lens_support():
    support = novel_view_support(np.eye(4), np.eye(4))
    assert support["distance_from_seed_camera_m"] == 0
    assert support["novel_view_quality"] == "unverified"
    assert support["ego_without_reconstruction_asset"]
    assert not support["trainable"]
    assert support["pedestrian_cyclist_appearance"] == "unverified"


@pytest.mark.parametrize("size,q", [([4, 2, -1], [1, 0, 0, 0]), ([4, 2, 2], [2, 0, 0, 0])])
def test_virtual_camera_rejects_bad_dimensions_and_nonunit_orientation(size, q):
    with pytest.raises(ValueError):
        roof_front_camera([0, 0, 0], q, size)


def test_source_ego_camera_supports_traffic_fleet_without_extra_learner():
    from driversi.rendering.gaussian.mtgs_multiview import validate_ego_camera_owner
    validate_ego_camera_owner(("ego", "traffic"), np.array([True, False]))
    for ids, flags in [(("ego", "traffic"), [False, True]), (("car",), [True])]:
        with pytest.raises(ValueError, match="controlled ego"):
            validate_ego_camera_owner(ids, np.array(flags))
