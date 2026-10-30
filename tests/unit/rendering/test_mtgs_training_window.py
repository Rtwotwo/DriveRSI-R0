"""Native full-checkpoint transport stays explicit and preserves source axes."""

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_training_window import asset_inputs, cv_pose


def test_source_axes_conversion_is_exact_involution():
    pose = np.c_[np.eye(3), [1.0, 2.0, 3.0]]
    converted = cv_pose(pose)
    np.testing.assert_array_equal(converted, np.c_[np.diag([1.0, -1.0, -1.0]), [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(cv_pose(converted), pose)
    np.testing.assert_array_equal(pose, np.c_[np.eye(3), [1.0, 2.0, 3.0]])


def test_native_pose_rejects_nonfinite_and_wrong_dimensions():
    for pose in (np.eye(4), np.full((3, 4), np.nan)):
        with pytest.raises(ValueError, match="Finite native"):
            cv_pose(pose)


def test_native_assets_require_actual_point_labels_and_same_traversal(tmp_path):
    base = tmp_path / "road_block-test"
    (base / "rgb_point_cloud").mkdir(parents=True)
    for name in ("config.yml", "step.ckpt"):
        (tmp_path / name).write_text(name)
    (base / "video_scene_dict.pkl").write_bytes(b"metadata")
    (base / "rgb_point_cloud/road_block-test-12.pcd").write_bytes(b"pcd")
    road = tmp_path / "road.yml"
    road.write_text(f"data_root: {tmp_path}\nroad_block_name: road_block-test\n")
    args = tmp_path / "config.yml", tmp_path / "step.ckpt", road, 12
    with pytest.raises(FileNotFoundError):
        asset_inputs(*args)
    labels = base / "rgb_point_cloud/road_block-test-12_sem_labels.npy"
    labels.write_bytes(b"labels")
    result = asset_inputs(*args)
    assert len(result) == 6 and str(labels) in result
    with pytest.raises(FileNotFoundError):
        asset_inputs(*args[:3], 13)
    with pytest.raises(ValueError):
        asset_inputs(*args[:3], True)


def test_native_source_evidence_cannot_use_novel_camera(tmp_path):
    from driversi.agents.backends.native import digest
    from driversi.rendering.diagnostics.worldengine_admission import _validate_source_frame

    source_image = tmp_path / "0000-CAM_F0-source-render.png"
    real = tmp_path / "photo.jpg"
    source_image.write_bytes(b"render")
    real.write_bytes(b"original")
    pose = np.c_[np.eye(3), [1.0, 2.0, 3.0]]
    row = dict(
        tick=0,
        rendered_source_image=str(source_image),
        real_source_image=str(real),
        source_motions={},
        source_token="frame",
        source_appearance_timestamp_us=12,
        source_camera_to_world=pose.tolist(),
    )
    files = {str(p): digest(p) for p in (source_image, real)}
    canonical = {"token": "frame", "timestamp": 12}
    _validate_source_frame(row, canonical, "CAM_F0", files, native_canonical_pose=pose)
    row["source_camera_to_world"][0][3] += 1
    with pytest.raises(ValueError, match="canonical"):
        _validate_source_frame(row, canonical, "CAM_F0", files, native_canonical_pose=pose)


def test_decoded_static_motion_preserves_shape_and_restores_after_exception():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from driversi.rendering.gaussian.mtgs_training_window import decoded_cloud_motion

    class Node:
        def __init__(self):
            self.parameter = torch.tensor([1.2, 0.0, 0.0, 0.0])
            self.native_means = torch.tensor([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
            self.native_quats = torch.tensor([[1.2, 0.0, 0.0, 0.0], [1.2, 0.0, 0.0, 0.0]])

        def get_gaussians(self, camera_to_worlds, **kwargs):
            return {
                "means": self.native_means,
                "quats": self.native_quats,
                "scales": torch.ones((2, 3)),
                "opacities": torch.ones(2),
            }

    node = Node()
    original = node.get_gaussians(None)
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = (
        [3.0, 0.0, 0.0],
        turn,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        [1.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0,
    )
    with pytest.raises(RuntimeError, match="render failed"):
        with decoded_cloud_motion(
            {"rigid_object_car": SimpleNamespace(node=node)}, {"car": transform}
        ):
            moved = node.get_gaussians(None)
            torch.testing.assert_close(
                moved["means"], torch.tensor([[3.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
            )
            torch.testing.assert_close(
                torch.linalg.norm(moved["quats"], dim=-1), torch.full((2,), 1.2)
            )
            torch.testing.assert_close(node.parameter, torch.tensor([1.2, 0.0, 0.0, 0.0]))
            raise RuntimeError("render failed")
    torch.testing.assert_close(node.get_gaussians(None)["means"], original["means"])
    assert "get_gaussians" not in node.__dict__
    assert node.cached_gaussians is None


@pytest.mark.parametrize("horizon", [8, 10])
def test_native_renderer_preserves_declared_actual_horizon(monkeypatch, horizon):
    from types import SimpleNamespace
    from driversi.rendering.gaussian import mtgs_training_window as m, worldengine_replay as w
    monkeypatch.setattr(m, "NativeTrainingRenderer", lambda *a: SimpleNamespace(asset_files={"asset": "sha"}))
    monkeypatch.setattr(w, "run", lambda *a, **kw: kw)
    result = m.run("scene", "rollout", "raw", "config", "weights", "dir/road", 2, "out", future_points=horizon)
    assert result["future_points"] == horizon
    assert result["asset_files"] == ["asset"]

@pytest.mark.parametrize("horizon", [True, 7, 9, 11])
def test_native_renderer_rejects_invented_horizon_before_gpu(monkeypatch, horizon):
    from driversi.rendering.gaussian import mtgs_training_window as m
    monkeypatch.setattr(m, "NativeTrainingRenderer", lambda *a: pytest.fail("No GPU initialization"))
    with pytest.raises(ValueError, match="eight or ten"):
        m.run("scene", "rollout", "raw", "config", "weights", "road", 2, "out", future_points=horizon)


def test_shared_export_accepts_native_size_name_without_mutating_calibration():
    from driversi.rendering.gaussian.worldengine_replay import bind_output_size

    pixels = np.zeros((540, 960, 3), np.uint8)
    native = {"output_calibration_size": [960, 540], "native_calibration_size": [1920, 1080]}
    result = bind_output_size(pixels, native, 960)
    assert result["output_image_size"] == [960, 540]
    assert result["native_calibration_size"] == [1920, 1080]
    assert "output_image_size" not in native
    for wrong in ({"output_calibration_size": [1920, 1080]}, {"output_image_size": [1920, 1080]}):
        with pytest.raises(ValueError, match="calibration output size"):
            bind_output_size(pixels, wrong, 960)
    with pytest.raises(ValueError, match="Actual renderer pixels"):
        bind_output_size(pixels, native, 1920)
