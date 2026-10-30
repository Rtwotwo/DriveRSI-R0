import numpy as np
import pytest

from driversi.data.assets.mtgs_coverage import audit_binding_assets
from driversi.rendering.gaussian.mtgs_registration import motion_in_source, move_camera, move_rigid


def source():
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    b[:3, 3] = [9, 4, 2]
    return dict(ego2global_original=a, ego2global=b)


def test_motion_is_transported_not_added_in_wrong_axes():
    s = source()
    p = np.c_[np.eye(3), [1, 0, 2]]
    out = move_camera(s, p, [0, 0, 2], [2, 0], 0)
    np.testing.assert_allclose(out[:, 3], [1, 2, 2], atol=1e-14)
    xyz, q = move_rigid(s, [1, 0, 2], [1, 0, 0, 0], [2, 0], 0)
    np.testing.assert_allclose(xyz, out[:, 3], atol=1e-14)
    np.testing.assert_array_equal(q, [1, 0, 0, 0])


def test_conjugation_preserves_relative_rigid_geometry_and_zero():
    s = source()
    theta = 0.2
    c, t = np.cos(theta), np.sin(theta)
    s["ego2global"][:3, :3] = [[1, 0, 0], [0, c, -t], [0, t, c]]
    p = np.c_[np.eye(3), [1, 2, 3]]
    np.testing.assert_array_equal(move_camera(s, p, [0, 0, 0], [0, 0], 0), p)
    delta, rot, _ = motion_in_source(s, [1, 2], 0.4)
    raw = np.array([[np.cos(0.4), -np.sin(0.4), 0], [np.sin(0.4), np.cos(0.4), 0], [0, 0, 1]])
    basis = s["ego2global"][:3, :3]
    x = np.array([3, 2, 1])
    np.testing.assert_allclose(rot @ (basis @ x) + delta, basis @ (raw @ x + [1, 2, 0]))


def test_bad_source_rejected():
    s = source()
    s["ego2global"][0, 0] = 2
    with pytest.raises(ValueError):
        move_camera(s, np.c_[np.eye(3), [0, 0, 0]], [0, 0, 0], [1, 0], 0)


def test_preflight_includes_stationary_and_missing_frame_assets():
    b = dict(
        travel_id=1,
        log_name="train-log",
        frames=[
            dict(
                mtgs_frame_index=1,
                token="frame",
                timestamp_us=4,
                source_objects=[
                    dict(track_token="a", kind="vehicle", physics_bound=True),
                    dict(track_token="b", kind="pedestrian", physics_bound=True),
                ],
            )
        ],
    )
    n = {
        token: dict(
            travel_id=1,
            gaussian_count=10,
            in_frame_mask=[True, False],
            runtime_frame_indices={"frame": dict(index=0, timestamp_us=4)},
        )
        for token in ("a", "b")
    }
    assert audit_binding_assets(b, n)["complete"]
    n["b"]["travel_id"] = 2
    assert audit_binding_assets(b, n)["frames"][0]["issues"][0]["reason"] == "asset_other_traversal"
    n["b"]["travel_id"] = 1
    n["b"]["in_frame_mask"] = [False]
    assert not audit_binding_assets(b, n)["complete"]
    del n["b"]
    assert audit_binding_assets(b, n)["missing_tokens"] == ["b"]


def test_forged_inventory_is_rejected():
    from types import SimpleNamespace

    from driversi.data.assets.mtgs_coverage import verify_source_inventory

    raw = [
        dict(
            token="frame",
            timestamp=1,
            log_name="train",
            anns=dict(track_tokens=["a"], gt_names=["pedestrian"]),
        )
    ]
    binding = dict(
        log_name="train",
        frames=[dict(navsim_index=0, token="frame", timestamp_us=1, source_objects=[])],
    )
    with pytest.raises(ValueError, match="immutable raw annotations"):
        verify_source_inventory(binding, SimpleNamespace(actor_ids=["ego", "a"]), raw)


def test_source_frame_height_is_not_added_twice():
    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    obj = MTGSWorldSession.__new__(MTGSWorldSession)
    obj.registration_mode = "source-frame"
    obj.height_mode = "reference-track"
    obj.ids = ["ego"]
    obj.controlled = [True]
    value = obj._height_corrections(None, None)["ego"]
    assert value["correction_m"] == 0 and not value["supported"]


def test_projected_rigid_points_match_camera_motion_under_tilt_and_turn():
    s = source()
    angle = 0.2
    c, t = np.cos(angle), np.sin(angle)
    s["ego2global"][:3, :3] = [[c, 0, t], [0, 1, 0], [-t, 0, c]]
    basis = s["ego2global"][:3, :3]
    camera = np.c_[basis, [1.0, 2.0, 3.0]]
    pivot = camera[:, 3].copy()
    point = pivot + basis @ np.array([4.0, 1.0, 2.0])
    moved = move_camera(s, camera, pivot, [2.0, 1.0], 0.3)
    d, rotation, _ = motion_in_source(s, [2.0, 1.0], 0.3)
    moved_point = pivot + rotation @ (point - pivot) + d
    np.testing.assert_allclose(
        moved[:, :3].T @ (moved_point - moved[:, 3]), camera[:, :3].T @ (point - pivot), atol=1e-12
    )


def test_body_relative_camera_extrinsics_survive_common_motion():
    s = source()
    body = np.eye(4)
    body[:3, 3] = [1, 2, 3]
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0.8, -0.3, 1.5]
    camera = body @ extrinsic
    moved_body = body.copy()
    moved_body[:3] = move_camera(s, body[:3], body[:3, 3], [2, 1], 0.4)
    moved_camera = camera.copy()
    moved_camera[:3] = move_camera(s, camera[:3], body[:3, 3], [2, 1], 0.4)
    np.testing.assert_allclose(np.linalg.inv(moved_body) @ moved_camera, extrinsic, atol=1e-12)


def test_runtime_catalog_and_audit_reject_empty_node_and_resolve_skipped_frame():
    from types import SimpleNamespace as NS

    from driversi.data.assets.mtgs_coverage import runtime_nodes

    sources = NS(travel_ids=[1, 1], frame_tokens=["kept", "kept"], frame_ids=[0, 0])
    node = NS(travel_id=1, num_points=0, in_frame_mask=np.array([True, False]))
    runtime = NS(
        pipeline=NS(datamanager=NS(train_dataset=NS(_dataparser_outputs=sources))),
        source_frames={(1, "kept"): {"timestamp": 9}},
        model=NS(gaussian_models={"rigid_object_a": node}),
    )
    binding = dict(
        travel_id=1,
        log_name="train",
        frames=[
            dict(
                mtgs_frame_index=1,
                token="kept",
                timestamp_us=9,
                source_objects=[dict(track_token="a", kind="vehicle", physics_bound=True)],
            )
        ],
    )
    assert not audit_binding_assets(binding, runtime_nodes(runtime))["complete"]
    node.num_points = 10
    report = audit_binding_assets(binding, runtime_nodes(runtime))
    assert report["complete"]
    assert report["frames"][0]["runtime_frame_indices"]["a"] == 0
    binding["frames"][0]["timestamp_us"] = 10
    assert not audit_binding_assets(binding, runtime_nodes(runtime))["complete"]


def test_source_frame_height_contract_can_be_exported_as_unsupported():
    from driversi.data.assets.mtgs_episode import _height_binding
    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    obj = MTGSWorldSession.__new__(MTGSWorldSession)
    obj.registration_mode = "source-frame"
    obj.height_mode = "source-frame"
    obj.ids = ["ego"]
    obj.controlled = [True]
    actor = obj._height_corrections(None, None)["ego"]
    camera = dict(
        camera_id="CAM_F0",
        source_timestamp_us=5,
        owner_actor_id="ego",
        height_binding=actor,
        scene_height_supported=False,
    )
    frame = dict(
        height_binding=dict(
            mode="source-frame",
            all_controlled_supported=False,
            per_camera=[
                dict(
                    camera_id="CAM_F0",
                    source_timestamp_us=5,
                    actors=[actor],
                    all_controlled_supported=False,
                )
            ],
        )
    )
    result = _height_binding(frame, [camera], ["ego"], "source-frame")
    assert result["all_controlled_supported"] is False
