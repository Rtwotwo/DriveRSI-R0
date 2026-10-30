"""Persistent, sequential MTGS inference with transactional world-pose overrides.

No optional renderer dependency is imported until construction. Source cameras
retain the pinned loader's uncorrected pose and exact frame/time metadata; the
checkpoint camera correction is applied once inside the official renderer.
"""

from __future__ import annotations

import copy
import math
import pickle
import shutil
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from driversi.rendering.gaussian.mtgs import (
    BACKEND_REVISION,
    _array,
    _index_source_frames,
    _rgb8,
    _validate_checkpoint_topology,
    clear_rigid_cache,
)
from driversi.rendering.gaussian.mtgs_multiview import camera_input_for_target, rigid_matrix

RigidPoses = Mapping[str, tuple[Sequence[float], Sequence[float]]]


def source_pinhole_intrinsics(raw_intrinsics, distortion, raw_size, mode) -> np.ndarray:
    """Match the pinned loader's calibration without reading or remapping images.

    The loader retains the full image canvas and converts the pixel-centre
    convention by subtracting 0.5 before OpenCV and adding it afterward.
    This helper intentionally preserves the input float precision.
    """
    import cv2

    intrinsic = np.array(raw_intrinsics, copy=True)
    distortion = np.asarray(distortion)
    if (
        intrinsic.shape != (3, 3)
        or intrinsic.dtype.kind != "f"
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(distortion).all()
        or distortion.size not in (4, 5, 8, 12, 14)
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0, 0, 1])
    ):
        raise ValueError("Invalid source-lens calibration")
    if (
        len(raw_size) != 2
        or any(type(value) is not int or value <= 0 for value in raw_size)
        or mode not in ("optimal", "keep_focal_length")
    ):
        raise ValueError("Unsupported source camera canvas or undistortion mode")
    intrinsic[:2, 2] -= 0.5
    if mode == "optimal":
        intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, raw_size, 1)
    intrinsic[:2, 2] += 0.5
    return intrinsic


def _calibration_array_key(array) -> tuple:
    array = np.ascontiguousarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def source_lens_map(
    *,
    raw_intrinsics: np.ndarray,
    distortion: np.ndarray,
    pinhole_intrinsics: np.ndarray,
    raw_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical source-lens pixels to a rendered undistorted canvas.

    Intrinsics use Nerfstudio pixel-center coordinates (first pixel center is 0.5),
    matching the pinned loader's +/- 0.5 convention. The raw intrinsic is at raw
    resolution; pinhole intrinsic is already at render resolution. No FOV crop or
    content-based mask is performed. OpenCV is an optional renderer dependency.
    """
    import cv2

    raw_k = np.asarray(raw_intrinsics, dtype=np.float64)
    pinhole_k = np.asarray(pinhole_intrinsics, dtype=np.float64)
    dist = np.asarray(distortion, dtype=np.float64).reshape(-1)
    for matrix in (raw_k, pinhole_k):
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
        ):
            raise ValueError("Camera intrinsics require a finite positive pinhole calibration")
    if len(dist) not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError("Unsupported or invalid OpenCV source-lens distortion")
    if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in (*raw_size, *output_size)):
        raise ValueError("Camera dimensions must be positive integers")
    width, height = output_size
    y, x = np.mgrid[:height, :width]
    centers = np.stack(((x + 0.5) * raw_size[0] / width, (y + 0.5) * raw_size[1] / height), axis=-1)
    # All points and K share the half-pixel convention, so OpenCV's normalized
    # coordinates agree. Subtract 0.5 only when returning remap array indices.
    # OpenCV's default five inverse iterations leave visible subpixel errors
    # near strongly distorted source corners. Converge this geometric map once
    # and reuse it across ticks instead of paying the cost per frame.
    projected = (
        cv2.undistortPointsIter(
            centers.reshape(-1, 1, 2),
            raw_k,
            dist,
            None,
            pinhole_k,
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
        ).reshape(height, width, 2)
        - 0.5
    )
    if not np.isfinite(projected).all():
        raise ValueError("Lens inverse projection returned nonfinite coordinates")
    # Avoid floating-point errors invalidating exact identity boundary pixels.
    rounded = np.rint(projected)
    projected = np.where(np.abs(projected - rounded) < 1e-6, rounded, projected)
    return projected[:, :, 0].astype(np.float32), projected[:, :, 1].astype(np.float32)


def resample_source_lens(
    rgb: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    pinhole_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample calibrated source-lens RGB, returning explicit interpolation support.

    Invalid pixels are black and accompanied by a false mask; the original pinhole
    image remains the caller's artifact. This cannot improve reconstruction detail.
    """
    import cv2

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Lens sampling requires uint8 RGB")
    if (
        map_x.ndim != 2
        or map_x.shape != map_y.shape
        or not np.isfinite(map_x).all()
        or not np.isfinite(map_y).all()
    ):
        raise ValueError("Lens maps must have equal finite two-dimensional shape")
    if pinhole_valid_mask is None:
        support = np.ones(rgb.shape[:2], dtype=np.float32)
    else:
        if pinhole_valid_mask.dtype != bool or pinhole_valid_mask.shape != rgb.shape[:2]:
            raise ValueError("Pinhole support must be a bool mask matching RGB")
        support = pinhole_valid_mask.astype(np.float32)
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
    valid = (
        cv2.remap(
            support,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        >= 1.0 - 1e-6
    )
    sampled = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled[~valid] = 0
    return sampled, valid


def project_source_lens(result: Mapping[str, Any], projection: Mapping[str, Any]) -> dict:
    """One calibrated RGB/surface conversion for policy, preview and diagnostics.

    The raw pinhole result is untouched. Sampling support is not a geometry or
    visual-quality certificate. Future exporters must use this same conversion.
    """
    from driversi.rendering.gaussian.surface_quality import remap_surface

    rgb = np.asarray(result["rgb"])
    alpha, depth = result.get("accumulation"), result.get("depth")
    if any(v is not None and np.asarray(v).shape != rgb.shape[:2] for v in (alpha, depth)):
        raise ValueError("Raw surface channels must match pinhole RGB before lens projection")
    image, valid = resample_source_lens(rgb, projection["map_x"], projection["map_y"])
    alpha, depth = remap_surface(alpha, depth, projection["map_x"], projection["map_y"], valid)
    return {"rgb": image, "valid": valid, "accumulation": alpha, "depth": depth}


def _clear_all_rigid_caches(model: Any) -> None:
    for name, node in model.gaussian_models.items():
        if name.startswith("rigid_object_"):
            clear_rigid_cache(node)


@contextmanager
def temporary_rigid_poses(
    model: Any,
    travel_id: int,
    frame_index: int,
    poses: RigidPoses,
    *,
    source_frames: Mapping[str, int] | None = None,
) -> Iterator[None]:
    """Validate every override before mutation, then restore on every exit path.

    XYZ uses reconstruction metres and quaternion order is wxyz. Callers hold
    ``torch.no_grad()``. A retained world object needs an explicit earlier asset
    anchor; source presence and pose edits are restored together, even on failure.
    """
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("Source frame index must be a nonnegative integer")
    anchors = {} if source_frames is None else dict(source_frames)
    if set(anchors) - set(poses):
        raise ValueError("Asset anchors require matching world-pose overrides")
    replacements, presence_edits = [], []
    for token, pose in poses.items():
        if not isinstance(token, str) or not token or token.startswith("rigid_object_"):
            raise ValueError("Overrides require source track tokens, not model node names")
        name = "rigid_object_" + token
        if name not in model.gaussian_models:
            raise ValueError(f"No reconstructed rigid node for token {token}")
        node = model.gaussian_models[name]
        if int(node.travel_id) != travel_id:
            raise ValueError(f"Rigid token {token} belongs to another traversal")
        if frame_index >= int(node.num_frames):
            raise ValueError(f"Rigid token {token} is absent at the source frame")
        active = bool(node.in_frame_mask[frame_index])
        if token in anchors:
            anchor = anchors[token]
            if (
                type(anchor) is not int
                or not 0 <= anchor <= frame_index
                or not bool(node.in_frame_mask[anchor])
            ):
                raise ValueError("Asset anchor must be a supported current or earlier frame")
        if not active:
            if token not in anchors:
                raise ValueError(f"Rigid token {token} is absent at the source frame")
            presence_edits.append((node, active))
        if len(pose) != 2:
            raise ValueError("Each pose requires translation XYZ and quaternion wxyz")
        xyz, quat = (np.asarray(item, dtype=np.float64) for item in pose)
        if (
            xyz.shape != (3,)
            or quat.shape != (4,)
            or not np.isfinite(xyz).all()
            or not np.isfinite(quat).all()
        ):
            raise ValueError("Rigid poses require finite XYZ and quaternion wxyz")
        if not math.isclose(float(np.linalg.norm(quat)), 1.0, abs_tol=1e-5):
            raise ValueError("Rigid quaternion wxyz must have unit norm")
        trans_target = node.instance_trans if node.is_static else node.instance_trans[frame_index]
        quat_target = node.instance_quats if node.is_static else node.instance_quats[frame_index]
        replacements.extend(
            (
                (trans_target, trans_target.detach().clone(), xyz),
                (quat_target, quat_target.detach().clone(), quat),
            )
        )
    _clear_all_rigid_caches(model)
    try:
        for node, _ in presence_edits:
            node.in_frame_mask[frame_index] = True
        for target, _, replacement in replacements:
            target.copy_(target.new_tensor(replacement))
        yield
    finally:
        for target, saved, _ in replacements:
            target.copy_(saved)
        for node, saved in presence_edits:
            node.in_frame_mask[frame_index] = saved
        _clear_all_rigid_caches(model)


def select_source_dataset(manager, travel_id=None):
    """Route to an existing native dataset; never borrow another traversal's camera."""
    if travel_id is not None and (type(travel_id) is not int or travel_id < 0):
        raise ValueError("Source traversal must be a nonnegative integer")
    for partition in ("train", "eval"):
        dataset = getattr(manager, partition + "_dataset", None)
        if dataset is None:
            continue
        ids = {int(x) for x in dataset._dataparser_outputs.travel_ids}
        if (travel_id is None and partition == "train") or travel_id in ids:
            return dataset, partition
    raise ValueError(f"Checkpoint runtime has no native source camera traversal {travel_id}")


class MTGSRuntime:
    """One GPU-loaded checkpoint, shared setup for probes and world renderers.

    The runtime does not own a simulator or advance time. Methods reject concurrent
    use: callers batch/control scheduling outside this object. Unbound objects keep
    their source-frame poses. Changing a source camera pose is allowed, but its
    metadata must still identify that exact source frame and exposure reference.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        road_block_config: str | Path,
        scale: float = 0.5,
        trusted_assets: bool = False,
        source_travel_id: int | None = None,
    ) -> None:
        if not trusted_assets:
            raise ValueError("Only audited trusted MTGS assets may be deserialized")
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("Runtime scale must be within (0, 1]")
        self.paths = tuple(
            Path(path).resolve(strict=True)
            for path in (
                config_path,
                checkpoint_path,
                road_block_config,
            )
        )
        self.scale = float(scale)
        self.source_travel_id = source_travel_id
        self._operation = threading.Lock()
        self._pinhole_cache = OrderedDict()
        self._lens_cache = OrderedDict()
        self._load()

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("MTGS runtime is sequential; concurrent operation rejected")
        try:
            yield
        finally:
            self._operation.release()

    def _load(self) -> None:
        from driversi.agents.backends.sources import activate_packages, require_source

        from driversi.vendor import path as dependency_path
        owned = dependency_path("mtgs")
        state = require_source(owned, BACKEND_REVISION)
        activate_packages({"mtgs": owned / "mtgs", "nuplan_scripts": owned / "nuplan_scripts"})
        import mtgs
        import torch
        import yaml
        from mtgs.dataset.custom_datamanager import CustomFullImageDatamanager
        from mtgs.dataset.custom_dataset import CustomInputDataset

        class RuntimeDataManager(CustomFullImageDatamanager[CustomInputDataset]):
            DATALOADER_MAP = {
                **CustomFullImageDatamanager.DATALOADER_MAP,
                "prefetch": CustomFullImageDatamanager.DATALOADER_MAP["on_demand"],
            }

            @property
            def dataset_type(self):
                return CustomInputDataset

        if not torch.cuda.is_available():
            raise RuntimeError("MTGS runtime requires the isolated CUDA runtime")
        missing = [name for name in ("nvcc", "ninja") if shutil.which(name) is None]
        if missing:
            raise RuntimeError(
                f"Pinned GSplat runtime needs {missing} on PATH; prepend the isolated "
                "environment bin and CUDA toolkit bin directories"
            )
        source_root = Path(mtgs.__file__).resolve().parents[1]
        if source_root != owned:
            raise ValueError("Renderer must use owned MTGS source, not clones")
        self.revision = state["actual"]
        if self.revision != BACKEND_REVISION:
            raise ValueError(f"Unvalidated MTGS revision {self.revision}")
        config_path, checkpoint_path, road_block_config = self.paths
        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        parser_config = config.pipeline.datamanager.dataparser
        if not math.isclose(float(parser_config.scale_factor), 1.0, abs_tol=1e-12):
            raise ValueError(
                "Runtime requires reconstruction coordinates in metres (scale_factor=1)"
            )
        parser_config.road_block_config = road_block_config
        parser_config.load_cam_optim_from = None
        parser_config.eval_2hz = False
        manager = config.pipeline.datamanager
        manager.num_workers = 0
        manager.cache_strategy = "on_demand"
        manager.eval_cache_strategy = "on_demand"
        manager._target = RuntimeDataManager
        manager.camera_res_scale_factor = self.scale
        config.pipeline.model.lpips_metric = False
        self.training_exclusion_masks = list(manager.load_custom_masks)
        self.torch = torch
        self.device = torch.device("cuda")
        # Model initialization is deterministic without taking ownership of a
        # caller's actor or simulator random streams.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(0)
                np.random.seed(0)
                pipeline = config.pipeline.setup(device=self.device, test_mode="inference")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                _validate_checkpoint_topology(pipeline.model, checkpoint["pipeline"])
                pipeline.load_pipeline(checkpoint["pipeline"], checkpoint["step"])
        finally:
            np.random.set_state(numpy_state)
        pipeline.eval()
        self.pipeline = pipeline
        self.config = config
        self.model = pipeline.model
        self.dataset, self.source_partition = select_source_dataset(
            pipeline.datamanager, self.source_travel_id
        )
        self.parser = pipeline.datamanager.dataparser
        self.sources = self.dataset._dataparser_outputs
        self.metadata_path = Path(self.parser.video_scene.pickle_path)
        with self.metadata_path.open("rb") as stream:
            self.source_frames = _index_source_frames(pickle.load(stream))

    def photometric_context(self, index: int) -> dict:
        """Declare the checkpoint target domain without altering rendered pixels."""
        if type(index) is not int or not 0 <= index < len(self.sources.image_filenames):
            raise ValueError("Photometric reference requires a known source index")
        factors = getattr(self.sources, "v_adjust_factors", None)
        factor = 1.0 if factors is None else float(factors[index])
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Source exposure factor must be finite and positive")
        return {
            "source_partition": getattr(self, "source_partition", "train"),
            "source_travel_id": getattr(self, "source_travel_id", None),
            "render_domain": "checkpoint_exposure_aligned" if factors is not None else "source_rgb",
            "source_reference_domain": "source_rgb",
            "v_adjust_factor": factor,
            "alignment": "official_hsv_value_scale_and_clip" if factors is not None else "none",
            "raw_recovery_verified": factors is None,
            "quality_certificate": False,
        }

    def source_reference(self, index: int, output_size: tuple[int, int]) -> dict:
        """Read raw photo plus official exposure target, without label/depth I/O.

        Both use original lens coordinates. They only match source camera poses;
        a counterfactual camera cannot use either as pixel ground truth.
        """
        from PIL import Image

        context = self.photometric_context(index)
        if len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size):
            raise ValueError("Reference output size must contain positive integers")
        with self._exclusive():
            with Image.open(self.sources.image_filenames[index]) as original:
                raw = original.convert("RGB")
                target = Image.fromarray(self.dataset.get_numpy_image(index))
                if target.size != raw.size:
                    raise ValueError("Official reference must preserve the source lens canvas")
                raw = np.asarray(raw.resize(output_size, Image.Resampling.LANCZOS)).copy()
                target = np.asarray(target.resize(output_size, Image.Resampling.LANCZOS)).copy()
        return {"raw": raw, "checkpoint_target": target, "photometric": context}

    def source_sample(self, index: int) -> tuple[dict, Any]:
        """Return processed source RGB and an uncorrected official source camera."""
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.sources.image_filenames)
        ):
            raise ValueError("Source camera index is outside the dataset")
        with self._exclusive():
            previous = self.dataset.fake_data
            self.dataset.fake_data = False
            try:
                data, camera = self.dataset.get_data_and_camera(
                    index,
                    image_type="uint8",
                    scale_factor=self.scale,
                )
            finally:
                self.dataset.fake_data = previous
            camera.metadata["driversi_source_index"] = index
            camera = camera.to(self.device)
            self.validate_camera(camera)
            return data, camera

    def source_camera(self, index: int) -> Any:
        """Build the official seed camera without image, label, or depth I/O.

        Geometry follows ``CustomInputDataset.get_data_and_camera`` at the
        pinned revision. Metadata dimensions must describe the source canvas;
        the source-sample equivalence probe verifies that asset contract.
        Learned checkpoint corrections are still applied only by the renderer.
        """
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.sources.image_filenames)
        ):
            raise ValueError("Source camera index is outside the dataset")
        with self._exclusive():
            if self.dataset.crop_size is not None:
                raise ValueError("Camera-only loading does not support cropped source assets")
            camera = copy.deepcopy(self.dataset.cameras[index : index + 1])
            mode = self.dataset.undistort_images
            if mode is not False:
                raw_size = (int(camera.width.item()), int(camera.height.item()))
                intrinsic, distortion = self.sources.undistort_params[index]
                key = (
                    mode,
                    raw_size,
                    _calibration_array_key(intrinsic),
                    _calibration_array_key(distortion),
                )
                cache = self._pinhole_cache
                if key not in cache:
                    cache[key] = source_pinhole_intrinsics(intrinsic, distortion, raw_size, mode)
                    if len(cache) > 16:
                        cache.popitem(last=False)
                cache.move_to_end(key)
                intrinsic = cache[key]
                camera.fx[:] = intrinsic[0, 0]
                camera.fy[:] = intrinsic[1, 1]
                camera.cx[:] = intrinsic[0, 2]
                camera.cy[:] = intrinsic[1, 2]
            if self.scale != 1.0:
                camera.rescale_output_resolution(scaling_factor=self.scale)
            camera = self.dataset.push_cam_details_to_metadata(camera, self.sources, index)
            camera.metadata["driversi_source_index"] = index
            camera = camera.to(self.device)
            self.validate_camera(camera)
            return camera

    def source_lens_projection(self, camera: Any) -> dict:
        """Prepare a reusable map for the seed's original calibrated lens FOV."""
        with self._exclusive():
            self.validate_camera(camera)
            index = camera.metadata.get("driversi_source_index", camera.metadata.get("cam_idx"))
            raw_camera = self.sources.cameras[index]
            raw_size = (int(raw_camera.width.item()), int(raw_camera.height.item()))
            output_size = (int(camera.width.item()), int(camera.height.item()))
            raw_intrinsics, distortion = self.sources.undistort_params[index]
            pinhole = _array(camera.get_intrinsics_matrices()[0])
            key = (
                raw_size,
                output_size,
                _calibration_array_key(raw_intrinsics),
                _calibration_array_key(distortion),
                _calibration_array_key(pinhole),
            )
            cache = self._lens_cache
            if key not in cache:
                map_x, map_y = source_lens_map(
                    raw_intrinsics=raw_intrinsics,
                    distortion=distortion,
                    pinhole_intrinsics=pinhole,
                    raw_size=raw_size,
                    output_size=output_size,
                )
                map_x.flags.writeable = map_y.flags.writeable = False
                scaled = np.array(raw_intrinsics, dtype=np.float64, copy=True)
                scaled[0] *= output_size[0] / raw_size[0]
                scaled[1] *= output_size[1] / raw_size[1]
                cache[key] = {
                    "map_x": map_x,
                    "map_y": map_y,
                    "camera_model": "opencv_distorted_source_lens",
                    "raw_size": raw_size,
                    "output_size": output_size,
                    "K": scaled.tolist(),
                    "distortion": np.asarray(distortion).tolist(),
                    "semantics": "physical_source_lens_fov_resampled_from_pinhole; not_quality_repair",
                }
                if len(cache) > 16:
                    cache.popitem(last=False)
            cache.move_to_end(key)
            # The large immutable maps are shared; callers receive independent
            # mutable calibration descriptions and cannot alter cached metadata.
            projection = cache[key]
            return {
                **projection,
                "K": copy.deepcopy(projection["K"]),
                "distortion": copy.deepcopy(projection["distortion"]),
            }

    def validate_camera(self, camera: Any) -> tuple[int, int]:
        """Reject stale or mixed frame metadata before rendering replay objects."""
        meta = camera.metadata or {}
        index = meta.get("driversi_source_index", meta.get("cam_idx"))
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.sources.image_filenames)
        ):
            raise ValueError("Rendering requires a known source camera index")
        expected = {
            "cam_idx": None if getattr(self, "source_partition", "train") == "eval" else index,
            "cam_token": self.sources.cam_tokens[index],
            "frame_token": self.sources.frame_tokens[index],
            "travel_id": self.sources.travel_ids[index],
            "frame_idx": None
            if getattr(self, "source_partition", "train") == "eval"
            else self.sources.frame_ids[index],
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError("Source camera frame, traversal, or token mismatch")
        if meta.get("driversi_camera_correction_applied", False):
            raise ValueError("Source camera correction would be applied twice")
        time_value = _array(camera.times).reshape(-1)
        source_time = _array(self.sources.cameras.times[index]).reshape(-1)
        if (
            len(time_value) != 1
            or len(source_time) != 1
            or not np.isfinite(time_value).all()
            or not np.allclose(time_value, source_time, rtol=0, atol=1e-7)
        ):
            raise ValueError("Camera exposure time does not match its source frame seed")
        pose = _array(camera.camera_to_worlds)
        if pose.shape != (1, 3, 4) or not np.isfinite(pose).all():
            raise ValueError("Camera pose must be finite with shape (1, 3, 4)")
        rotation = pose[0, :3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not math.isclose(
            float(np.linalg.det(rotation)), 1, abs_tol=1e-4
        ):
            raise ValueError("Camera orientation must be a proper rotation")
        intrinsic = _array(camera.get_intrinsics_matrices())
        if (
            intrinsic.shape != (1, 3, 3)
            or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0, 0] <= 0
            or intrinsic[0, 1, 1] <= 0
            or not np.allclose(intrinsic[0, 2], [0, 0, 1])
            or int(camera.width.item()) <= 0
            or int(camera.height.item()) <= 0
        ):
            raise ValueError("Camera calibration or image dimensions are invalid")
        return int(expected["travel_id"]), int(self.sources.frame_ids[index])

    def render_camera(
        self,
        camera: Any,
        *,
        rigid_poses: RigidPoses | None = None,
        target_camera_to_world=None,
        exclude_tokens: Sequence[str] = (),
        rigid_source_frames: Mapping[str, int] | None = None,
    ) -> dict:
        """Render a frozen world pose set and roll all object edits back afterward.

        The returned timing includes pose validation, edits, GPU synchronization, and
        RGB transfer, but excludes model initialization, camera loading and disk writes.
        """
        with self._exclusive(), self.torch.no_grad():
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            travel_id, frame_index = self.validate_camera(camera)
            poses = {} if rigid_poses is None else rigid_poses
            rendered_camera = copy.deepcopy(camera)
            excluded = tuple(exclude_tokens)
            if len(set(excluded)) != len(excluded) or any(
                not isinstance(token, str)
                or "rigid_object_" + token not in self.model.gaussian_models
                for token in excluded
            ):
                raise ValueError(
                    "Excluded camera-owner tokens must identify unique reconstructed nodes"
                )
            for token in excluded:
                node = self.model.gaussian_models["rigid_object_" + token]
                if int(node.travel_id) != travel_id:
                    raise ValueError("Excluded camera-owner token belongs to another traversal")
            if excluded or poses:
                visible = list(
                    rendered_camera.metadata.get("visible_tokens", list(self.model.gaussian_models))
                )
                # Source-camera culling is stale after world motion. Let the
                # renderer's frustum/depth test evaluate every explicit binding.
                for token in poses:
                    name = "rigid_object_" + token
                    if name not in visible:
                        visible.append(name)
                if any(name not in self.model.gaussian_models for name in visible):
                    raise ValueError("Visible tokens must be explicit reconstruction node names")
                rendered_camera.metadata["visible_tokens"] = [
                    name
                    for name in visible
                    if name not in {"rigid_object_" + token for token in excluded}
                ]
                if not rendered_camera.metadata["visible_tokens"]:
                    raise ValueError("Camera exclusion cannot remove all reconstruction nodes")
            pose_semantics = "uncorrected_source_seed_then_checkpoint_correction_once"
            target_pose = None
            if target_camera_to_world is not None:
                target_pose = rigid_matrix(target_camera_to_world)[:3]
                adjusted_source = self.model.camera_optimizer.apply_to_camera(rendered_camera)
                input_pose = camera_input_for_target(
                    _array(rendered_camera.camera_to_worlds)[0],
                    _array(adjusted_source)[0],
                    target_pose,
                )
                rendered_camera.camera_to_worlds = rendered_camera.camera_to_worlds.new_tensor(
                    input_pose[None]
                )
                verified = _array(self.model.camera_optimizer.apply_to_camera(rendered_camera))[0]
                if not np.allclose(verified, target_pose, rtol=0, atol=2e-5):
                    raise ValueError(
                        "Virtual camera optimizer inversion does not reproduce target pose"
                    )
                pose_semantics = (
                    "target_pose_inverse_seed_adjustment_then_checkpoint_correction_once"
                )
                effective_pose = rigid_matrix(verified)[:3]
            else:
                # Observe the same pure checkpoint transform used downstream.
                # Keep rendered_camera untouched: rasterization still receives
                # the original input and applies its correction exactly once.
                effective_pose = rigid_matrix(
                    _array(self.model.camera_optimizer.apply_to_camera(rendered_camera))[0]
                )[:3]
            with temporary_rigid_poses(
                self.model, travel_id, frame_index, poses, source_frames=rigid_source_frames
            ):
                outputs = self.model.get_outputs_for_camera(rendered_camera)
                rgb = _rgb8(outputs["rgb_appearance"])
                # Preserve GS evidence before releasing the temporary world poses.
                # Alpha is coverage, not semantic validity; expected depth is not
                # an independently measured physical surface. Legacy test/plugin
                # renderers that omit both channels remain explicitly unknown.
                accumulation, depth = None, None
                if outputs.get("accumulation") is not None or outputs.get("depth") is not None:
                    if outputs.get("accumulation") is None or outputs.get("depth") is None:
                        raise ValueError("Renderer must provide both accumulation and depth")
                    accumulation = np.array(
                        _array(outputs["accumulation"]), dtype=np.float32, copy=True
                    )
                    depth = np.array(_array(outputs["depth"]), dtype=np.float32, copy=True)
                    if accumulation.shape == (*rgb.shape[:2], 1):
                        accumulation = accumulation[:, :, 0]
                    if depth.shape == (*rgb.shape[:2], 1):
                        depth = depth[:, :, 0]
                    if (
                        accumulation.shape != rgb.shape[:2]
                        or depth.shape != rgb.shape[:2]
                        or not np.isfinite(accumulation).all()
                        or not np.isfinite(depth).all()
                        or np.any(accumulation < -1e-6)
                        or np.any(accumulation > 1 + 1e-6)
                        or np.any(depth < 0)
                    ):
                        raise ValueError("Renderer returned invalid accumulation or expected depth")
                    accumulation = np.clip(accumulation, 0, 1)
                self.torch.cuda.synchronize()
            return {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "photometric": self.photometric_context(
                    camera.metadata.get("driversi_source_index", camera.metadata.get("cam_idx"))
                ),
                "surface_semantics": "gs_alpha_and_expected_depth_not_measured_geometry",
                "render_seconds": time.perf_counter() - started,
                "override_tokens": sorted(poses),
                "reused_asset_frames": dict(rigid_source_frames or {}),
                "unbound_object_mode": "source_frame_reconstruction",
                "camera_pose_semantics": pose_semantics,
                "camera_to_world": effective_pose.tolist(),
                "camera_to_world_semantics": "effective_after_world_motion_and_checkpoint_camera_optimizer",
                "camera_coordinate_frame": "mtgs_reconstruction_metres",
                "camera_pose_axes": "nerfstudio_right_up_back",
                "excluded_tokens": list(excluded),
                "target_camera_to_world": None if target_pose is None else target_pose.tolist(),
            }
