"""Analytical camera/occlusion checks independent of the world controller."""

from dataclasses import replace

import numpy as np
import pytest

from driversi.contracts.visual import CameraCalibration, CameraRig, WorldFrame
from driversi.rendering.visual.primitive import PrimitiveRenderer, _polygon


def camera():
    transform = CameraCalibration.front(64, 48).camera_to_body.copy()
    transform[:3, 3] = [0, 0, 1]
    return CameraCalibration("front", 64, 48,
                             np.asarray([[40, 0, 32], [0, 40, 24], [0, 0, 1]]), transform)


def frame(scene, poses=None):
    return WorldFrame(scene.actor_ids, scene.poses[0] if poses is None else poses,
                      0, "episode", "root", 0)


def positioned(scene):
    scene.dimensions[:] = [4, 2, 2]
    scene.poses[:, 0] = [0, 0, 17, 0]
    scene.poses[:, 1] = [10, 0, -8, 0]
    scene.poses[:, 2] = [-10, 0, 50, 0]
    return scene


def test_projection_and_front_back_direction(straight_scene):
    scene = positioned(straight_scene)
    renderer = PrimitiveRenderer(scene, CameraRig((camera(),)))
    rendered = renderer.render_batch(frame(scene), ("a",))
    assert rendered.instances[0, 0, 24, 32] == 2
    assert rendered.depth[0, 0, 24, 32] == pytest.approx(8.)
    ys, xs = np.where(rendered.instances[0, 0] == 2)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (27, 36, 19, 28)
    assert not np.any(rendered.instances == 1)  # own body never rendered
    assert not np.any(rendered.instances == 3)  # behind camera
    pose = scene.poses[0].copy()
    pose[0, 3] = np.pi
    back = renderer.render_batch(frame(scene, pose), ("a",))
    assert back.instances[0, 0, 24, 32] == 3
    assert not np.any(back.instances == 2)


def test_rig_rotations_and_ground_optical_depth(straight_scene):
    scene = positioned(straight_scene)
    down_transform = np.eye(4)
    down_transform[:3, :3] = np.diag([1, -1, -1])
    down_transform[:3, 3] = [0, 0, 10]
    downward = replace(camera(), camera_id="down", camera_to_body=down_transform)
    renderer = PrimitiveRenderer(scene, CameraRig((camera(), downward)))
    result = renderer.render_batch(frame(scene), ("a",))
    # The 64x48 image sees x in [-8,8], y in [-6,6], with objects outside
    # its open footprint. Every ray intersects the plane at optical z=10.
    assert np.allclose(result.depth[0, 1], 10)
    assert not np.any(result.instances[0, 1])
    pose = scene.poses[0].copy()
    pose[:, 2] = 1000
    flat = renderer.render_batch(frame(scene, pose), ("a",))
    np.testing.assert_array_equal(result.rgb, flat.rgb)


def test_near_plane_crossing_and_perspective_depth():
    cal = camera()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.full((48, 64), np.inf)
    instances = np.zeros((48, 64), dtype=np.int32)
    # One vertex behind near plane; the surviving polygon must not disappear.
    crossing = np.asarray([[-.3, -.3, .1], [.5, -.3, 1], [0, .5, 1]])
    _polygon(crossing, cal, (1, 2, 3), 7, rgb, depth, instances)
    visible = instances == 7
    assert visible.sum() > 100
    assert np.all(depth[visible] >= cal.near)
    assert np.all(depth[visible] <= 1)

    # Independent known screen triangle and z values: the screen barycentric
    # weights at (20.5,16.5) yield inverse-z interpolation analytically.
    screen = np.asarray([[8., 8.], [56., 8.], [8., 40.]])
    zs = np.asarray([2., 8., 4.])
    vertices = np.column_stack(((screen - [32, 24]) / 40 * zs[:, None], zs))
    depth[:] = np.inf
    instances[:] = 0
    _polygon(vertices, cal, (1, 2, 3), 8, rgb, depth, instances)
    w1, w2 = (20.5-8)/48, (16.5-8)/32
    expected = 1 / ((1-w1-w2)/2 + w1/8 + w2/4)
    assert instances[16, 20] == 8
    assert depth[16, 20] == pytest.approx(expected)


def test_far_clip_rejects_and_clips_crossings():
    cal = replace(camera(), far=5)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.full((48, 64), np.inf)
    instances = np.zeros((48, 64), dtype=np.int32)
    vertices = np.asarray([[-1, -1, 3], [1, -1, 3], [0, 1, 7]])
    _polygon(vertices, cal, (1, 2, 3), 4, rgb, depth, instances)
    assert np.any(instances == 4)
    assert np.all(depth[instances == 4] <= 5)
    previous = depth.copy()
    _polygon(vertices + [0, 0, 10], cal, (9, 9, 9), 5, rgb, depth, instances)
    np.testing.assert_array_equal(previous, depth)


def test_occlusion_and_observer_batch_order(straight_scene):
    scene = positioned(straight_scene)
    pose = scene.poses[0].copy()
    pose[2] = [20, 0, 0, np.pi]
    renderer = PrimitiveRenderer(scene, CameraRig((camera(),)))
    observation = renderer.render_batch(frame(scene, pose), ("a", "replay"))
    assert observation.instances[0, 0, 24, 32] == 2
    assert not np.any(observation.instances[0] == 3)
    reverse = renderer.render_batch(frame(scene, pose), ("replay", "a"))
    np.testing.assert_array_equal(observation.rgb, reverse.rgb[::-1])
    np.testing.assert_array_equal(observation.depth, reverse.depth[::-1])
    np.testing.assert_array_equal(observation.instances, reverse.instances[::-1])
    pose[2, 0] = 5
    closer = renderer.render_batch(frame(scene, pose), ("a",))
    assert closer.instances[0, 0, 24, 32] == 3
    assert closer.depth[0, 0, 24, 32] == pytest.approx(3)


def test_renderer_and_result_isolation(straight_scene):
    scene = positioned(straight_scene)
    frozen = frame(scene)
    original_pose = frozen.poses.copy()
    renderer = PrimitiveRenderer(scene, CameraRig((camera(),)))
    first = renderer.render_batch(frozen, ("a",))
    scene.dimensions[:] = 80
    scene.poses[:] = 999
    scene.road_points[:] = 999
    second = renderer.render_batch(frozen, ("a",))
    np.testing.assert_array_equal(first.rgb, second.rgb)
    np.testing.assert_array_equal(original_pose, frozen.poses)
    for name in ("rgb", "depth", "instances", "valid"):
        assert not getattr(first, name).flags.writeable
        with pytest.raises(ValueError):
            getattr(first, name).flat[0] = 0


def test_map_segment_crossing_entire_frustum_is_not_culled(straight_scene):
    scene = positioned(straight_scene)
    scene.poses[:, 1, 0] = -20
    scene.road_points = np.asarray([[10., -1000., 0.], [10., 1000., 0.]])
    scene.road_offsets = np.asarray([0, 2])
    scene.road_types = np.asarray([4])
    transform = camera().camera_to_body.copy()
    transform[2, 3] = 1.125
    cal = replace(camera(), camera_to_body=transform)
    result = PrimitiveRenderer(scene, CameraRig((cal,))).render_batch(frame(scene), ("a",))
    # Both segment endpoints are offscreen, but the stripe crosses every ray
    # on row 28: x = fx * camera_height / (28.5 - cy) = 10 m.
    np.testing.assert_array_equal(result.rgb[0, 0, 28], np.tile([150, 151, 144], (64, 1)))
    np.testing.assert_allclose(result.depth[0, 0, 28], 10)


def test_invalid_observers_frame_and_ground_camera(straight_scene):
    renderer = PrimitiveRenderer(straight_scene, CameraRig((camera(),)))
    for actors in ((), ("missing",), ("a", "a")):
        with pytest.raises(ValueError):
            renderer.render_batch(frame(straight_scene), actors)
    with pytest.raises(ValueError):
        renderer.render_batch(replace(frame(straight_scene), actor_ids=("b", "a", "replay")), ("a",))
    transform = camera().camera_to_body.copy()
    transform[2, 3] = 0
    with pytest.raises(ValueError, match="above"):
        PrimitiveRenderer(straight_scene, CameraRig((replace(camera(), camera_to_body=transform),)))
