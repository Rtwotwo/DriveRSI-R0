"""Bounded MTGS checkpoint probes, separate from the authoritative driving world.

The optional MTGS/torch stack is imported only by ``render_mtgs_probe``.  This
module does not turn reconstruction instances into simulation actors.  Outputs
are explicitly ineligible as closed-loop training observations.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

SCHEMA_VERSION = 1
BACKEND_REVISION = "7ab67a3e386e5a4830017819324922a7bb9f26f7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output: Path, manifest: dict, *, compact: bool = False) -> None:
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=None if compact else 2,
                   separators=(",", ":") if compact else None, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "manifest.json")


@contextmanager
def _probe_evidence(output: Path, manifest: dict) -> Iterator[None]:
    """A partial image directory is always accompanied by explicit failure evidence."""
    manifest["status"] = "running"
    _write_manifest(output, manifest)
    try:
        yield
        expected = manifest["selection"]["frame_count"]
        if len(manifest["frames"]) != expected:
            raise RuntimeError(f"Only {len(manifest['frames'])}/{expected} visible probes found")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "reason": str(error)}
        for attempt in manifest["attempts"]:
            if attempt["status"] == "running":
                attempt.update(status="failed", reason=str(error))
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest["selection"]["candidates_examined"] = len(manifest["attempts"])
        _write_manifest(output, manifest)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb8(value: Any) -> np.ndarray:
    rgb = _array(value)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Renderer returned invalid RGB")
    if rgb.dtype == np.uint8:
        return rgb.copy()
    if rgb.min() < -1e-5 or rgb.max() > 1.00001:
        raise ValueError("Renderer RGB must be within [0, 1]")
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def compare_probe_images(
    baseline: np.ndarray, intervention: np.ndarray, restored: np.ndarray,
    camera_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure intervention support and restoration, not photorealism."""
    arrays = (baseline, intervention, restored)
    if any(a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3 for a in arrays):
        raise ValueError("Probe comparisons require uint8 RGB images")
    if len({a.shape for a in arrays}) != 1 or not baseline.size:
        raise ValueError("Probe images must share a nonempty shape")
    changed = np.any(baseline != intervention, axis=-1)
    restore_error = np.abs(baseline.astype(np.int16) - restored.astype(np.int16))
    result = {
        "changed_pixel_count": int(changed.sum()),
        "changed_pixel_fraction": float(changed.mean()),
        "restore_equal": bool(np.array_equal(baseline, restored)),
        "max_restore_pixel_error": int(restore_error.max()),
        "baseline_rgb_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        "intervention_rgb_sha256": hashlib.sha256(intervention.tobytes()).hexdigest(),
        "restored_rgb_sha256": hashlib.sha256(restored.tobytes()).hexdigest(),
    }
    if camera_valid_mask is not None:
        if (camera_valid_mask.dtype != np.bool_ or camera_valid_mask.shape != changed.shape
                or not camera_valid_mask.any()):
            raise ValueError("Camera validity must be a nonempty bool mask matching RGB")
        valid_count = int(camera_valid_mask.sum())
        valid_changed = int((changed & camera_valid_mask).sum())
        result.update(
            camera_valid_pixel_count=valid_count,
            camera_valid_pixel_fraction=float(camera_valid_mask.mean()),
            valid_changed_pixel_count=valid_changed,
            valid_changed_pixel_fraction=valid_changed / valid_count,
            changed_outside_camera_valid_count=int((changed & ~camera_valid_mask).sum()),
        )
    return result


def source_camera_valid_mask(
    dataset: Any, image_index: int, raw_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Lens remap support only, independent of ego/semantic training exclusions.

    Sizes are (width, height). The pinned loader retains the full remapped canvas
    (its ROI is not cropped) and resizes masks with nearest interpolation.
    This is not a validity certificate for reconstruction or novel viewpoints.
    """
    width, height = raw_size
    if min(width, height, *output_size) < 1:
        raise ValueError("Camera dimensions must be positive")
    if dataset.undistort_images:
        blank = np.full((height, width, 3), 255, dtype=np.uint8)
        remapped = dataset._undistort_image(
            blank, image_index, return_mask=True, interpolation="linear",
        )
        valid = np.asarray(remapped["valid_mask"], dtype=bool)
        if valid.shape != (height, width, 1):
            raise ValueError("Official lens remap changed the expected canvas")
        valid = valid[:, :, 0]
    else:
        return np.ones((output_size[1], output_size[0]), dtype=bool)
    if raw_size != output_size:
        valid = dataset._resize_numpy_image(
            valid.astype(np.uint8), output_size, resample=dataset.RESAMPLE_MAP["mask"],
        )
    return np.asarray(valid, dtype=bool)


def clear_rigid_cache(node: Any) -> None:
    """MTGS caches transformed Gaussians by frame index and training step."""
    node.last_step = None
    node.last_frame_idx = None
    node.cached_gaussians = None


@contextmanager
def temporary_translation(
    node: Any, frame_index: int, translation: Sequence[float],
) -> Iterator[None]:
    """Replace one pose, restore even on errors, invalidate both branch caches.

    The caller holds torch.no_grad().  No Parameter object or checkpoint is
    replaced; only its selected translation is temporarily overwritten.
    """
    replacement = np.asarray(translation, dtype=np.float64)
    if replacement.shape != (3,) or not np.isfinite(replacement).all():
        raise ValueError("Translation must be three finite reconstruction coordinates")
    parameter = node.instance_trans
    if node.is_static:
        target = parameter
    else:
        if frame_index < 0 or frame_index >= len(parameter):
            raise ValueError("Object frame index is outside its source lifetime")
        if not bool(node.in_frame_mask[frame_index]):
            raise ValueError("Object is absent at this source frame")
        target = parameter[frame_index]
    saved = target.detach().clone()
    clear_rigid_cache(node)
    try:
        target.copy_(target.new_tensor(replacement))
        yield
    finally:
        target.copy_(saved)
        clear_rigid_cache(node)


def _validate_checkpoint_topology(model: Any, state: dict[str, Any]) -> None:
    """Reject official loader's silent missing-node or pose-shape fallback."""
    prefix = "_model.gaussian_models."
    nodes = {key[len(prefix):].split(".", 1)[0] for key in state if key.startswith(prefix)}
    expected = set(model.gaussian_models)
    if nodes != expected:
        raise ValueError(
            "Checkpoint/data node mismatch: "
            f"missing={sorted(expected - nodes)}, unexpected={sorted(nodes - expected)}"
        )
    for name, node in model.gaussian_models.items():
        if not hasattr(node, "instance_trans"):
            continue
        for field in ("instance_trans", "instance_quats"):
            key = prefix + name + "." + field
            if key not in state or tuple(state[key].shape) != tuple(getattr(node, field).shape):
                raise ValueError(f"Checkpoint/data pose shape mismatch: {key}")
    camera_key = "_model.camera_optimizer.pose_adjustment"
    if camera_key in state and state[camera_key].shape[0] != model.num_train_data:
        raise ValueError("Checkpoint/data camera count mismatch")


def _index_source_frames(scene: dict) -> dict[tuple[int, str], dict]:
    """Bind by stable tokens, never the parser's mutable train/eval frame indices."""
    indexed = {}
    for video_token, video in scene.items():
        travel_id = int(video_token.split("-")[-1])
        for frame in video["frame_infos"]:
            key = travel_id, str(frame["token"])
            if key in indexed:
                raise ValueError("Duplicate source traversal/frame token")
            indexed[key] = frame
    return indexed


def _source_inventory(frame: dict, node_tokens: set) -> list:
    return [
        {"token": str(token), "class": str(kind),
         "reconstructed_rigid_node": str(token) in node_tokens,
         "physics_bound": False,
         "appearance_coverage": "rigid_node" if str(token) in node_tokens else "unverified"}
        for token, kind in zip(frame["track_tokens"], frame["gt_names"])
    ]


def _candidate_node(model: Any, camera: Any) -> tuple[str, Any] | None:
    """Deterministic frustum candidate; changed pixels subsequently test visibility."""
    meta = camera.metadata
    travel_id, frame_index = int(meta["travel_id"]), int(meta["frame_idx"])
    pose = _array(model.camera_optimizer.apply_to_camera(camera)[0])
    intrinsic = _array(camera.get_intrinsics_matrices()[0])
    width, height = int(camera.width.item()), int(camera.height.item())
    candidates = []
    for name, node in model.gaussian_models.items():
        if not name.startswith("rigid_object_") or int(node.travel_id) != travel_id:
            continue
        quat, trans = node.get_object_pose(frame_index, None)
        if quat is None:
            continue
        local = pose[:3, :3].T @ (_array(trans) - pose[:3, 3])
        # Nerfstudio stores OpenGL camera axes: right, up, backwards.
        local *= [1, -1, -1]
        if local[2] <= 2:
            continue
        pixel = intrinsic @ local
        pixel = pixel[:2] / pixel[2]
        if not (0.05 * width < pixel[0] < 0.95 * width
                and 0.05 * height < pixel[1] < 0.95 * height):
            continue
        size = _array(node.instance_size)
        score = float(size[0] * size[2] / local[2] ** 2)
        candidates.append((-score, name, node))
    if not candidates:
        return None
    _, name, node = min(candidates, key=lambda item: (item[0], item[1]))
    return name, node


def render_mtgs_probe(
    *, config_path: str | Path, checkpoint_path: str | Path,
    road_block_config: str | Path, output: str | Path,
    camera_name: str = "CAM_F0", frame_count: int = 3,
    displacement: Sequence[float] = (0.0, 2.0, 0.0), scale: float = 0.5,
    start_index: int = 0, frame_stride: int = 5, trusted_assets: bool = False,
) -> dict[str, Any]:
    """Load a pinned official model and write source/baseline/pose-probe PNGs.

    ``trusted_assets`` is required because official YAML and torch checkpoint
    formats deserialize Python objects.  The road-block config must already bind
    the downloaded data root.  Rendering uses source training cameras (with the
    checkpoint camera correction), not an independently held-out evaluation set.
    """
    if not trusted_assets:
        raise ValueError("Only audited trusted MTGS assets may be deserialized")
    delta = np.asarray(displacement, dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.any(delta):
        raise ValueError("Displacement must be a nonzero finite XYZ vector")
    if not (1 <= frame_count <= 32) or not (0 < scale <= 1) or not math.isfinite(scale):
        raise ValueError("Probe budget requires 1..32 frames and scale in (0, 1]")
    if start_index < 0 or frame_stride < 1:
        raise ValueError("Invalid frame selection")
    paths = [Path(p).resolve(strict=True)
             for p in (config_path, checkpoint_path, road_block_config)]
    config_path, checkpoint_path, road_block_config = paths
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Probe output must be absent or empty; old evidence is not overwritten")

    from PIL import Image

    from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime

    runtime = MTGSRuntime(
        config_path=config_path, checkpoint_path=checkpoint_path,
        road_block_config=road_block_config, scale=scale, trusted_assets=True,
    )
    torch = runtime.torch
    revision = runtime.revision
    parser_config = runtime.config.pipeline.datamanager.dataparser
    training_exclusion_masks = runtime.training_exclusion_masks
    model, dataset, sources = runtime.model, runtime.dataset, runtime.sources
    metadata_path, source_frames = runtime.metadata_path, runtime.source_frames
    matches = [i for i, p in enumerate(sources.image_filenames)
               if Path(p).parent.name == camera_name and i >= start_index]
    if not matches:
        raise ValueError(f"No source camera {camera_name}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": "mtgs_renderer_probe", "trainable": False,
        "capabilities": ["source_camera_checkpoint_render", "isolated_rigid_translation_probe"],
        "limits": ["not_bound_to_driving_world", "not_visual_self_play",
                   "no_pedestrian_or_cyclist_coverage_guarantee", "no_controlled_signal_states",
                   "novel_view_support_unverified", "source_training_camera_not_heldout_quality"],
        "backend": {"name": "MTGS", "revision": revision,
                    "adapter_source_sha256": _sha256(Path(__file__)),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                    "gsplat": importlib.metadata.version("gsplat")},
        "assets": {label: {"path": str(path), "sha256": _sha256(path)} for label, path in
                   zip(("config", "checkpoint", "road_block_config"), paths)},
        "selection": {"camera": camera_name, "frame_count": frame_count, "scale": scale,
                      "start_index": start_index, "frame_stride": frame_stride,
                      "rule": "frustum_size_rank_then_nonzero_intervention_in_lens_support"},
        "coordinate_frame": "MTGS reconstruction frame; units metres when parser scale=1",
        "scene_scale_factor": float(sources.scene_scale_factor),
        "source_processing": {"undistort_images": parser_config.undistort_images,
                              "exposure_alignment": parser_config.use_exposure_alignment,
                              "resolution_scale": scale},
        "training_exclusion_masks": training_exclusion_masks,
        "rgb_output": "rgb_appearance: checkpoint source-camera exposure correction",
        "image_loading": "on_demand via isolated datamanager; full scene metadata/model loaded",
        "camera_valid_mask_semantics": (
            "255=source lens remap support; official full canvas with nearest mask resize; "
            "excludes no semantic class or ego region; not novel-view or world-coverage validity"
        ),
        "frames": [], "attempts": [],
    }
    manifest["assets"]["source_metadata"] = {
        "path": str(metadata_path.resolve()), "sha256": _sha256(metadata_path),
    }
    node_tokens = {name.removeprefix("rigid_object_") for name in model.gaussian_models
                   if name.startswith("rigid_object_")}
    manifest["reconstruction_nodes"] = [
        {"token": name.removeprefix("rigid_object_"), "node_name": name,
         "travel_id": int(node.travel_id), "is_static": bool(node.is_static),
         "source_frame_count": int(node.num_frames), "physics_bound": False}
        for name, node in model.gaussian_models.items() if name.startswith("rigid_object_")
    ]
    attempts = 0
    base_timestamp = None
    selected_travel = None
    with _probe_evidence(output, manifest), torch.no_grad():
        for source_index in matches[::frame_stride]:
            if len(manifest["frames"]) >= frame_count:
                break
            attempts += 1
            if attempts > 80:
                break
            if selected_travel is not None and sources.travel_ids[source_index] != selected_travel:
                continue
            attempt = {"dataset_index": source_index, "status": "running"}
            manifest["attempts"].append(attempt)
            _write_manifest(output, manifest)
            data, camera = runtime.source_sample(source_index)
            candidate = _candidate_node(model, camera)
            if candidate is None:
                attempt.update(status="skipped", reason="no_rigid_center_in_candidate_frustum")
                continue
            name, node = candidate
            attempt["candidate_node"] = name
            frame_index = int(camera.metadata["frame_idx"])
            travel_id = int(camera.metadata["travel_id"])
            _, translation = node.get_object_pose(frame_index, None)
            original = _array(translation).copy()
            replacement = original + delta * float(sources.scene_scale_factor)

            def render() -> tuple[np.ndarray, float]:
                # Never reuse transformed rigid geometry across camera/branch queries.
                for rigid_name, rigid in model.gaussian_models.items():
                    if rigid_name.startswith("rigid_object_"):
                        clear_rigid_cache(rigid)
                torch.cuda.synchronize()
                started = time.perf_counter()
                rgb = _rgb8(model.get_outputs_for_camera(copy.deepcopy(camera))["rgb_appearance"])
                torch.cuda.synchronize()
                return rgb, time.perf_counter() - started

            baseline, baseline_seconds = render()
            with temporary_translation(node, frame_index, replacement):
                intervention, intervention_seconds = render()
            _, restored_translation = node.get_object_pose(frame_index, None)
            if not np.array_equal(_array(restored_translation), original):
                raise RuntimeError("MTGS object translation was not restored exactly")
            restored, restore_seconds = render()
            raw_path = Path(sources.image_filenames[source_index])
            with Image.open(raw_path) as raw:
                valid_mask = source_camera_valid_mask(
                    dataset, source_index, raw.size, (baseline.shape[1], baseline.shape[0]),
                )
            diagnostics = compare_probe_images(baseline, intervention, restored, valid_mask)
            diagnostics["pose_restore_equal"] = True
            if not diagnostics["restore_equal"]:
                raise RuntimeError("MTGS branch restoration did not reproduce baseline pixels")
            if diagnostics["valid_changed_pixel_count"] == 0:
                attempt.update(status="skipped", reason="no_changed_pixels_in_camera_support",
                               diagnostics=diagnostics)
                continue
            item_index = len(manifest["frames"])
            names = {key: f"{item_index:03d}-{key}.png"
                     for key in ("baseline", "intervention", "restored", "source", "source_raw",
                                 "camera_valid_mask")}
            image_arrays = {"baseline": baseline, "intervention": intervention,
                            "restored": restored, "source": _rgb8(data["image"]),
                            "camera_valid_mask": valid_mask.astype(np.uint8) * 255}
            for key, array in image_arrays.items():
                Image.fromarray(array).save(output / names[key])
            with Image.open(raw_path) as raw:
                raw.convert("RGB").save(output / names["source_raw"])
            raw_times = model.data_frame_dict[travel_id]["raw_timestamps"]
            timestamp = int(_array(raw_times[frame_index]).item())
            if base_timestamp is None:
                base_timestamp = timestamp
                selected_travel = travel_id
            optimized_pose = _array(model.camera_optimizer.apply_to_camera(camera)[0])
            source_frame = source_frames[(travel_id, str(camera.metadata["frame_token"]))]
            if int(source_frame["timestamp"]) != timestamp:
                raise ValueError("Source frame token/timestamp mismatch")
            source_camera = source_frame["cams"][camera_name]
            if Path(source_camera["data_path"]).stem != camera.metadata["cam_token"]:
                raise ValueError("Source camera token mismatch")
            diagnostics["render_seconds"] = {
                "baseline": baseline_seconds, "intervention": intervention_seconds,
                "restored": restore_seconds,
            }
            manifest["frames"].append({
                **names, "camera_name": camera_name, "dataset_index": source_index,
                "frame_token": camera.metadata["frame_token"],
                "cam_token": camera.metadata["cam_token"], "travel_id": travel_id,
                "frame_index": frame_index, "normalized_timestamp": float(camera.times.item()),
                "source_timestamp_us": timestamp,
                "source_time_semantics": "LiDAR frame time; not camera exposure timestamp",
                "camera_timestamp_us": int(source_camera["timestamp"]),
                "time_seconds": (timestamp - base_timestamp) / 1e6,
                "camera": {"T_recon_from_camera_opengl": optimized_pose.tolist(),
                           "K": _array(camera.get_intrinsics_matrices()[0]).tolist(),
                           "width": int(camera.width.item()), "height": int(camera.height.item()),
                           "distortion": "source image undistorted by official dataset adapter",
                           "pose": "source camera plus learned checkpoint correction, once"},
                "object": {"token": name.removeprefix("rigid_object_"), "node_name": name,
                           "original_translation": original.tolist(),
                           "override_translation": replacement.tolist(),
                           "delta_reconstruction_m": delta.tolist(),
                           "is_static_reconstruction_node": bool(node.is_static)},
                "source_inventory": _source_inventory(source_frame, node_tokens),
                "source_original_sha256": _sha256(raw_path),
                "source_original_path": str(raw_path),
                "image_file_sha256": {key: _sha256(output / name) for key, name in names.items()},
                "diagnostics": diagnostics,
            })
            attempt.update(status="accepted", frame_output_index=item_index)
    return manifest
