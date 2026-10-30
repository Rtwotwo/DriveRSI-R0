"""Independent geometry regressions for a world-moved source camera."""

from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_multiview import camera_input_for_target
from driversi.rendering.gaussian.mtgs_session import render_moved_source_camera
from driversi.sim.sensors.visual_sampling import body_camera_transform


class CameraTensor(np.ndarray):
    def new_tensor(self, value):
        return np.asarray(value).view(CameraTensor)


def rig():
    angle = .2
    correction = np.eye(4)
    correction[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]]
    correction[:3, 3] = [1., 2., .3]
    source = np.eye(4)
    source[:3, 3] = [11., 21., 3.]
    camera = SimpleNamespace(camera_to_worlds=source[None, :3].view(CameraTensor),
                             metadata={"cam_idx": 3})
    calls, queries = [], []

    def optimize(value):
        queries.append(1)
        result = value.camera_to_worlds.copy()
        result[0, :, :3] = correction[:3, :3] @ result[0, :, :3]
        result[0, :, 3] += correction[:3, 3]
        return result

    def render(value, *, rigid_poses, target_camera_to_world=None):
        calls.append((value, rigid_poses, target_camera_to_world))
        if target_camera_to_world is not None:
            corrected = optimize(value)[0]
            input_pose = camera_input_for_target(value.camera_to_worlds[0], corrected,
                                                  target_camera_to_world)
            supplied = SimpleNamespace(camera_to_worlds=input_pose[None].view(CameraTensor))
        else:
            supplied = value
        effective = optimize(supplied)[0]
        return {"camera_to_world": effective.tolist(), "rgb": np.zeros((2, 2, 3), np.uint8)}

    runtime = SimpleNamespace(model=SimpleNamespace(camera_optimizer=SimpleNamespace(
        apply_to_camera=optimize)), render_camera=render)
    optimized = source.copy()
    optimized[:3, :3] = correction[:3, :3] @ source[:3, :3]
    optimized[:3, 3] += correction[:3, 3]
    return runtime, camera, source, correction, optimized, calls, queries


def test_noncommuting_camera_correction_precedes_world_yaw():
    runtime, camera, source, correction, optimized, calls, _ = rig()
    world_rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    pivot, translation = np.array([10., 20., 1.]), np.array([2., -3., .4])
    expected = optimized[:3].copy()
    expected[:, :3] = world_rotation @ optimized[:3, :3]
    expected[:, 3] = pivot + world_rotation @ (optimized[:3, 3]-pivot) + translation
    result, target, seed = render_moved_source_camera(
        runtime, camera, pivot, translation[:2], np.pi/2, translation[2], {"car": "pose"})
    np.testing.assert_allclose(result["camera_to_world"], expected, atol=1e-12)
    np.testing.assert_allclose(target, expected, atol=1e-12)
    np.testing.assert_array_equal(seed, optimized[:3])
    wrong_rotation = correction[:3, :3] @ world_rotation @ source[:3, :3]
    assert not np.allclose(wrong_rotation, expected[:, :3])
    assert len(calls) == 1 and calls[0][2] is not None
    assert calls[0][1] == {"car": "pose"}
    np.testing.assert_array_equal(camera.camera_to_worlds, source[None, :3])
    assert camera.metadata == {"cam_idx": 3}


def test_stationary_source_applies_correction_once_without_target_roundtrip():
    runtime, camera, source, _, optimized, calls, queries = rig()
    result, target, seed = render_moved_source_camera(
        runtime, camera, [10, 20, 1], [0, 0], 0., 0., {})
    np.testing.assert_array_equal(result["camera_to_world"], optimized[:3])
    np.testing.assert_array_equal(target, source[:3])
    np.testing.assert_array_equal(seed, source[:3])
    assert len(queries) == 1 and calls[0][2] is None
    assert calls[0][0] is not camera
    np.testing.assert_array_equal(camera.camera_to_worlds, source[None, :3])


def test_pure_translation_and_height_preserve_commuting_source_path():
    runtime, camera, source, _, optimized, calls, queries = rig()
    result, _, _ = render_moved_source_camera(
        runtime, camera, [10, 20, 1], [2, -3], 0., .4, {})
    expected = optimized[:3].copy()
    expected[:, 3] += [2, -3, .4]
    np.testing.assert_allclose(result["camera_to_world"], expected, atol=1e-12)
    assert len(queries) == 1 and calls[0][2] is None
    np.testing.assert_array_equal(camera.camera_to_worlds, source[None, :3])


def test_effective_camera_to_body_rig_is_invariant_under_joint_world_motion():
    runtime, camera, _, _, optimized, _, _ = rig()
    body = np.eye(4)
    body[:3, 3] = [10, 20, 1]
    before = body_camera_transform(optimized[:3], body)
    result, _, _ = render_moved_source_camera(
        runtime, camera, body[:3, 3], [2, -3], np.pi/2, .4, {})
    after_body = body.copy()
    after_body[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    after_body[:3, 3] += [2, -3, .4]
    after = body_camera_transform(result["camera_to_world"], after_body)
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_invalid_height_fails_before_renderer_or_optimizer():
    runtime, camera, _, _, _, calls, queries = rig()
    with pytest.raises(ValueError, match="height correction"):
        render_moved_source_camera(runtime, camera, [0, 0, 0], [0, 0], .1, np.nan, {})
    assert calls == queries == []
