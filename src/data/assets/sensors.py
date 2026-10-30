"""Audit source NAVSIM camera assets and full calibration, without image decoding.

These manifests describe logged observations. They are not virtual camera rigs,
rendered observations, or evidence that counterfactual image assets are available.
All homogeneous transforms multiply column vectors: T_target_from_source.
This adapter follows the OpenScene/NAVSIM producer contract, not arbitrary
nuScenes-style dictionaries: its historically named ``sensor2lidar_*`` camera
fields contain camera-to-ego extrinsics copied from the nuPlan camera table.
"""

import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

SENSOR_MANIFEST_VERSION = 2
SOURCE_EGO_TO_CENTER_METRES = 1.461
SOURCE_CAMERA_CONVENTION = "openscene_navsim_sensor2lidar_fields_are_camera_to_ego"
SOURCE_CAMERA_PRODUCER = (
    "https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/"
    "process_data/helpers/nuplan_cameras_utils.py#L23-L39"
)


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite array of shape {shape}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return array


def _rotation(value: object, name: str) -> np.ndarray:
    rotation = _finite_array(value, (3, 3), name)
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
        raise ValueError(f"{name} must be a proper rotation, without scale or reflection")
    return rotation


def _quaternion_rotation(value: object, name: str) -> np.ndarray:
    quaternion = _finite_array(value, (4,), name)
    norm = np.linalg.norm(quaternion)
    if not np.isclose(norm, 1, rtol=0, atol=1e-3):
        raise ValueError(f"{name} must be a unit wxyz quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def _source_pose(frame: dict, prefix: str) -> np.ndarray:
    rotation_name, translation_name = f"{prefix}_rotation", f"{prefix}_translation"
    if rotation_name not in frame or translation_name not in frame:
        raise ValueError(f"Missing {rotation_name} or {translation_name}")
    return _transform(
        _quaternion_rotation(frame[rotation_name], rotation_name),
        _finite_array(frame[translation_name], (3,), translation_name),
    )


def _asset_path(root: Path, value: object, name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty relative asset path")
    # NAVSIM paths are POSIX-relative, even when this adapter runs on Windows.
    # Reject Windows drives and separators too, rather than changing meaning.
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or "\\" in value
            or ".." in path.parts or not path.parts):
        raise ValueError(f"{name} must remain inside sensor_root")
    resolved = root.joinpath(*path.parts).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"{name} must remain inside sensor_root")
    return path.as_posix(), resolved


def _image_size(path: Path) -> tuple[int, int]:
    """Read dimensions from bounded PNG/JPEG headers; do not claim decode validity."""
    with path.open("rb") as stream:
        header = stream.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            if len(header) != 24 or header[8:16] != b"\x00\x00\x00\x0dIHDR":
                raise ValueError("invalid_png_header")
            width, height = struct.unpack(">II", header[16:24])
        elif header[:2] == b"\xff\xd8":
            stream.seek(2)
            sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                   0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
            while True:
                if stream.tell() > 16 * 1024 * 1024:
                    raise ValueError("jpeg_header_exceeds_16_mib")
                if stream.read(1) != b"\xff":
                    raise ValueError("invalid_or_truncated_jpeg_header")
                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                if not marker or marker[0] in {0x00, 0xD8, 0xD9, 0xDA}:
                    raise ValueError("missing_jpeg_dimensions")
                if marker[0] == 0x01 or 0xD0 <= marker[0] <= 0xD7:
                    continue
                size_bytes = stream.read(2)
                if len(size_bytes) != 2:
                    raise ValueError("truncated_jpeg_segment")
                size = struct.unpack(">H", size_bytes)[0]
                if size < 2:
                    raise ValueError("invalid_jpeg_segment_length")
                if marker[0] in sof:
                    shape = stream.read(5)
                    if size < 8 or len(shape) != 5:
                        raise ValueError("invalid_jpeg_frame_header")
                    height, width = struct.unpack(">HH", shape[1:])
                    break
                stream.seek(size - 2, 1)
        else:
            raise ValueError("unsupported_or_invalid_image_header")
    if width <= 0 or height <= 0:
        raise ValueError("nonpositive_image_dimensions")
    return width, height


def _asset(root: Path, value: object, name: str, *, image: bool) -> dict:
    relative_path, path = _asset_path(root, value, name)
    record = {
        "relative_path": relative_path, "file_exists": path.is_file(),
        "width": None, "height": None,
        "dimensions_status": "not_applicable" if not image else "missing_file",
    }
    if not image or not record["file_exists"]:
        return record
    try:
        record["width"], record["height"] = _image_size(path)
        record["dimensions_status"] = "header_read_only"
    except (OSError, ValueError) as error:
        record["dimensions_status"] = "unavailable"
        record["dimensions_reason"] = str(error)
    return record


def extract_sensor_manifest(
    frame: dict, sensor_root: Path, *, origin_xyz: object = None,
) -> dict:
    """Extract JSON-safe camera calibration and asset presence for one trusted frame.

    ``sensor_root`` is the sensor blob partition root, not the dataset root.
    ``origin_xyz`` defaults to this frame's source ego position. For an existing
    Scene, supply Scene.metadata["origin_xyz"]. Only the OpenScene/NAVSIM
    camera-field convention is accepted by this adapter; the field names do not
    imply camera-to-LiDAR. No raw image is loaded as a
    simulated observation. Missing assets remain visible; malformed calibration
    or an escaping path rejects the manifest instead of silently dropping a camera.
    """
    if not isinstance(frame, dict):
        raise ValueError("Source frame must be a dictionary")
    timestamp = frame.get("timestamp")
    if (isinstance(timestamp, (bool, np.bool_))
            or not isinstance(timestamp, (int, np.integer))):
        raise ValueError("Source timestamp must be integer microseconds")
    for name in ("token", "log_name", "map_location"):
        if not isinstance(frame.get(name), str) or not frame[name]:
            raise ValueError(f"Missing source identity: {name}")
    cameras = frame.get("cams")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Source frame must contain a nonempty cams dictionary")
    if any(not isinstance(name, str) or not name for name in cameras):
        raise ValueError("Camera names must be nonempty strings")
    root = Path(sensor_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("sensor_root must be an existing directory")
    global_from_ego = _source_pose(frame, "ego2global")
    ego_from_lidar = _source_pose(frame, "lidar2ego")
    origin = (global_from_ego[:3, 3].copy() if origin_xyz is None
              else _finite_array(origin_xyz, (3,), "origin_xyz"))
    scene_from_global = _transform(np.eye(3), -origin)
    body_from_ego = _transform(np.eye(3), np.asarray([-SOURCE_EGO_TO_CENTER_METRES, 0, 0]))
    records = []
    for name, camera in sorted(cameras.items()):
        required = {"data_path", "cam_intrinsic", "distortion",
                    "sensor2lidar_rotation", "sensor2lidar_translation"}
        if not isinstance(camera, dict) or not required.issubset(camera):
            raise ValueError(f"Camera {name} is missing calibration or asset fields")
        intrinsic = _finite_array(camera["cam_intrinsic"], (3, 3), f"{name}.cam_intrinsic")
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-8)
                or abs(np.linalg.det(intrinsic)) < 1e-9):
            raise ValueError(f"{name}.cam_intrinsic must be an invertible camera matrix")
        distortion = np.asarray(camera["distortion"], dtype=np.float64)
        if distortion.ndim != 1 or not np.isfinite(distortion).all():
            raise ValueError(f"{name}.distortion must be a finite vector")
        # The OpenScene producer copies nuPlan camera table extrinsics into
        # these legacy names without a LiDAR transform. Multiplying by
        # ego_from_lidar here would incorrectly apply a second extrinsic.
        ego_from_camera = _transform(
            _rotation(camera["sensor2lidar_rotation"], f"{name}.sensor2lidar_rotation"),
            _finite_array(camera["sensor2lidar_translation"], (3,),
                          f"{name}.sensor2lidar_translation"),
        )
        lidar_from_camera = np.linalg.inv(ego_from_lidar) @ ego_from_camera
        records.append({
            "camera_id": name, "intrinsics": intrinsic.tolist(),
            "source_extrinsics": {
                "convention": SOURCE_CAMERA_CONVENTION,
                "sensor2lidar_rotation": ego_from_camera[:3, :3].tolist(),
                "sensor2lidar_translation": ego_from_camera[:3, 3].tolist(),
            },
            "distortion_coefficients": distortion.tolist(),
            "distortion_model": "source_unspecified_not_rectified",
            "asset": _asset(root, camera["data_path"], f"{name}.data_path", image=True),
            "T_lidar_from_camera": lidar_from_camera.tolist(),
            "T_ego_from_camera": ego_from_camera.tolist(),
            "T_body_from_camera": (body_from_ego @ ego_from_camera).tolist(),
            "T_scene_from_camera": (
                scene_from_global @ global_from_ego @ ego_from_camera).tolist(),
        })
    lidar = (_asset(root, frame["lidar_path"], "lidar_path", image=False)
             if frame.get("lidar_path") is not None else None)
    return {
        "schema_version": SENSOR_MANIFEST_VERSION,
        "kind": "source_sensor_assets_not_counterfactual_observation",
        "source": {"log_name": frame["log_name"], "token": frame["token"],
                   "timestamp_us": int(timestamp), "map_location": frame["map_location"]},
        "sensor_root": str(root), "origin_xyz": origin.tolist(),
        "coordinate_convention": {
            "matrix_direction": "T_target_from_source_column_vectors",
            "source_camera_fields": SOURCE_CAMERA_CONVENTION,
            "source_camera_producer": SOURCE_CAMERA_PRODUCER,
            "source_camera_assignment": (
                "https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/"
                "process_data/helpers/nuplan_cameras_utils.py#L80-L86"
            ),
            "camera_axes": "x_right_y_down_z_forward",
            "source_ego_reference": "rear_axle",
            "body_reference": "vehicle_center_same_height_as_source_ego",
            "source_ego_to_center_forward_metres": SOURCE_EGO_TO_CENTER_METRES,
            "world_axes": "map_axes_translated_by_origin_full_source_SE3",
            "quaternion_order": "wxyz",
        },
        "T_global_from_ego": global_from_ego.tolist(),
        "T_ego_from_lidar": ego_from_lidar.tolist(),
        "T_body_from_ego": body_from_ego.tolist(),
        "T_scene_from_ego": (scene_from_global @ global_from_ego).tolist(),
        "cameras": records, "lidar": lidar,
        "asset_counts": {
            "cameras": len(records),
            "camera_files_present": sum(item["asset"]["file_exists"] for item in records),
            "camera_dimensions_read": sum(
                item["asset"]["dimensions_status"] == "header_read_only" for item in records),
        },
        "limitations": [
            "one_source_frame_only_not_dataset_completeness",
            "image_headers_only_not_full_decode_validation",
            "raw_distortion_model_not_verified_or_rectified",
            "camera_capture_timestamps_not_present_in_standard_openscene_camera_dict",
            "full_source_pose_differs_from_planar_simulation",
            "no_reconstruction_asset_binding_or_counterfactual_rendering",
        ],
    }
