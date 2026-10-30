"""Source calibration semantics, asset boundaries and header-only dimensions."""

import copy
import json
import struct
import zlib

import numpy as np
import pytest

from driversi.data.assets.sensors import SOURCE_CAMERA_CONVENTION, extract_sensor_manifest


def source_frame():
    return {
        "timestamp": 123456789, "token": "frame-a", "log_name": "log-a",
        "map_location": "map-a", "ego2global_translation": [100, 200, 3],
        "ego2global_rotation": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        "lidar2ego_translation": [2, 3, 4],
        "lidar2ego_rotation": [np.sqrt(0.5), np.sqrt(0.5), 0, 0],
        "lidar_path": "log-a/lidar.bin",
        "cams": {"CAM_F0": {
            "data_path": "log-a/camera.png", "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": [5, 6, 7],
            "cam_intrinsic": [[10, 0, 3], [0, 11, 4], [0, 0, 1]],
            "distortion": [-0.3, 0.1, -0.002, 0.001, -0.05],
        }},
    }


def png_bytes(width, height):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x00\x00\x00" * width) * height))
            + chunk(b"IEND", b""))


def test_full_se3_direction_origin_and_center_reference(tmp_path):
    frame = source_frame()
    initial = copy.deepcopy(frame)
    manifest = extract_sensor_manifest(frame, tmp_path, origin_xyz=[90, 180, 1])
    camera = manifest["cameras"][0]
    # Producer field names say sensor2lidar, but values are camera-to-ego.
    # This nonidentity LiDAR pose catches accidental double application.
    np.testing.assert_allclose(np.asarray(camera["T_ego_from_camera"])[:3, 3], [5, 6, 7])
    np.testing.assert_allclose(np.asarray(camera["T_body_from_camera"])[:3, 3],
                               [3.539, 6, 7])
    np.testing.assert_allclose(np.asarray(camera["T_scene_from_camera"])[:3, 3],
                               [4, 25, 9])
    np.testing.assert_allclose(np.asarray(camera["T_scene_from_camera"])[:3, :3],
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(camera["T_lidar_from_camera"])[:3, 3],
                               [3, 3, -3], atol=1e-14)
    np.testing.assert_allclose(np.asarray(camera["T_lidar_from_camera"])[:3, :3],
                               [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-14)
    assert camera["source_extrinsics"] == {
        "convention": SOURCE_CAMERA_CONVENTION,
        "sensor2lidar_rotation": np.eye(3).tolist(),
        "sensor2lidar_translation": [5, 6, 7],
    }
    assert camera["distortion_coefficients"] == initial["cams"]["CAM_F0"]["distortion"]
    assert camera["distortion_model"] == "source_unspecified_not_rectified"
    assert manifest["coordinate_convention"]["camera_axes"] == "x_right_y_down_z_forward"
    assert json.loads(json.dumps(manifest, allow_nan=False))["schema_version"] == 2
    np.testing.assert_array_equal(frame["cams"]["CAM_F0"]["sensor2lidar_rotation"], np.eye(3))


def test_lidar_calibration_cannot_move_ego_camera(tmp_path):
    frame = source_frame()
    before = extract_sensor_manifest(frame, tmp_path)["cameras"][0]
    frame["lidar2ego_translation"] = [17, -9, 0.5]
    frame["lidar2ego_rotation"] = [1, 0, 0, 0]
    after = extract_sensor_manifest(frame, tmp_path)["cameras"][0]
    for field in ("T_ego_from_camera", "T_body_from_camera", "T_scene_from_camera"):
        np.testing.assert_array_equal(after[field], before[field])
    assert not np.allclose(after["T_lidar_from_camera"], before["T_lidar_from_camera"])


def test_default_origin_is_source_ego_and_files_are_checked(tmp_path):
    frame = source_frame()
    (tmp_path / "log-a").mkdir()
    (tmp_path / "log-a/camera.png").write_bytes(png_bytes(32, 17))
    (tmp_path / "log-a/lidar.bin").write_bytes(b"source lidar fixture")
    manifest = extract_sensor_manifest(frame, tmp_path)
    assert manifest["origin_xyz"] == [100, 200, 3]
    np.testing.assert_allclose(np.asarray(manifest["T_scene_from_ego"])[:3, 3], 0)
    asset = manifest["cameras"][0]["asset"]
    assert (asset["width"], asset["height"]) == (32, 17)  # Not twice principal point.
    assert asset["dimensions_status"] == "header_read_only"
    assert manifest["asset_counts"] == {
        "cameras": 1, "camera_files_present": 1, "camera_dimensions_read": 1,
    }
    assert manifest["lidar"]["file_exists"]


def test_jpeg_header_dimensions_and_missing_camera_remain_visible(tmp_path):
    frame = source_frame()
    (tmp_path / "log-a").mkdir()
    # Header-only JPEG fixture, deliberately not a fully decodable image.
    header = (b"\xff\xd8\xff\xe0\x00\x04ab\xff\xc0\x00\x11"
              + struct.pack(">BHHB", 8, 19, 41, 3) + bytes(9) + b"\xff\xd9")
    (tmp_path / "log-a/camera.png").write_bytes(header)
    frame["cams"]["CAM_B0"] = copy.deepcopy(frame["cams"]["CAM_F0"])
    frame["cams"]["CAM_B0"]["data_path"] = "log-a/missing.jpg"
    manifest = extract_sensor_manifest(frame, tmp_path)
    back, front = manifest["cameras"]
    assert back["camera_id"] == "CAM_B0"
    assert back["asset"]["dimensions_status"] == "missing_file"
    assert back["asset"]["width"] is None
    assert (front["asset"]["width"], front["asset"]["height"]) == (41, 19)
    assert manifest["asset_counts"]["camera_files_present"] == 1
    assert manifest["lidar"]["file_exists"] is False


def test_existing_unknown_image_header_is_not_silently_accepted(tmp_path):
    frame = source_frame()
    (tmp_path / "log-a").mkdir()
    (tmp_path / "log-a/camera.png").write_bytes(b"not an image")
    asset = extract_sensor_manifest(frame, tmp_path)["cameras"][0]["asset"]
    assert asset["file_exists"]
    assert asset["dimensions_status"] == "unavailable"
    assert asset["width"] is None and asset["height"] is None
    assert "invalid" in asset["dimensions_reason"]


@pytest.mark.parametrize("path", ["../escape.jpg", "/absolute.jpg", "C:/escape.jpg",
                                      "C:relative.jpg", "folder\\image.jpg", "."])
def test_escaping_paths_rejected_even_if_file_is_missing(tmp_path, path):
    frame = source_frame()
    frame["cams"]["CAM_F0"]["data_path"] = path
    with pytest.raises(ValueError, match="sensor_root"):
        extract_sensor_manifest(frame, tmp_path)


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "sensor"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(png_bytes(1, 1))
    try:
        (root / "linked.png").symlink_to(outside)
    except OSError:
        pytest.skip("Host does not allow unprivileged symlinks")
    frame = source_frame()
    frame["cams"]["CAM_F0"]["data_path"] = "linked.png"
    with pytest.raises(ValueError, match="sensor_root"):
        extract_sensor_manifest(frame, root)


@pytest.mark.parametrize("field,value", [
    ("sensor2lidar_rotation", np.diag([-1, 1, 1])),
    ("sensor2lidar_translation", [0, np.nan, 0]),
    ("cam_intrinsic", [[0, 0, 1], [0, 1, 1], [0, 0, 1]]),
    ("cam_intrinsic", [[1, 0, 1], [0, 1, 1], [0, 1, 1]]),
    ("distortion", [np.inf]),
])
def test_malformed_calibration_is_not_dropped(tmp_path, field, value):
    frame = source_frame()
    frame["cams"]["CAM_F0"][field] = value
    with pytest.raises(ValueError):
        extract_sensor_manifest(frame, tmp_path)


def test_invalid_quaternion_and_missing_transform_fail(tmp_path):
    frame = source_frame()
    frame["ego2global_rotation"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="unit wxyz"):
        extract_sensor_manifest(frame, tmp_path)
    frame = source_frame()
    del frame["lidar2ego_rotation"]
    with pytest.raises(ValueError, match="Missing lidar2ego_rotation"):
        extract_sensor_manifest(frame, tmp_path)


def test_camera_manifest_does_not_require_lidar_blob(tmp_path):
    frame = source_frame()
    del frame["lidar_path"]
    manifest = extract_sensor_manifest(frame, tmp_path)
    assert manifest["lidar"] is None
    assert len(manifest["cameras"]) == 1
