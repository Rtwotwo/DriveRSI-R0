"""One real native clip for inspection; never a complete video/admission receipt.

Run with the StreetCrafter environment. The CLI supervises a separate worker so
the wall-time limit also covers a native call that stops returning to Python.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    import types

    root = Path(__file__).resolve().parents[4]
    for name, folder in (("driversi", "src"),):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(root / folder)]
        sys.modules[name] = module

import numpy as np

from driversi.agents.backends.native import digest
from driversi.rendering.video.bundle import validate_bundle
from driversi.rsi.plugins.synthesis import check_files
from driversi.rendering.video.streetcrafter import runtime

SIZE = (1024, 576)
TICKS = list(range(25))
POSITION_TOLERANCE_M = 0.002
ROTATION_TOLERANCE_RAD = 0.0001


def _save(output, report):
    temporary = Path(output) / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(Path(output) / "report.json")


def _sealed(files, path):
    if files.get(str(path)) != digest(path):
        raise ValueError("Input is not sealed in its receipt: " + str(path))


def _pose_error(target, source):
    target, source = np.asarray(target, float), np.asarray(source, float)
    for matrix in (target, source):
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)
        ):
            raise ValueError("Finite rigid camera poses required")
    cosine = (np.trace(target[:3, :3].T @ source[:3, :3]) - 1) / 2
    return dict(
        translation_m=float(np.linalg.norm(target[:3, 3] - source[:3, 3])),
        rotation_rad=float(np.arccos(np.clip(cosine, -1, 1))),
    )


def validate_inputs(prepared, camera):
    """Check the sealed physical chain, including the actual tick-zero pose."""
    spec = json.loads(Path(prepared).read_text())
    if spec.get("kind") != "streetcrafter_native_conditions" or spec.get("schema_version") != 1:
        raise ValueError("Native StreetCrafter conditions required")
    check_files(spec["files"])
    _sealed(spec["files"], spec["geometry"])
    geometry = json.loads(Path(spec["geometry"]).read_text())
    if geometry.get("kind") != "calibrated_video_conditions" or geometry.get("source_control"):
        raise ValueError("Executed dense geometry required; no invented source control")
    check_files(geometry["files"])
    _sealed(geometry["files"], geometry["bundle"])
    bundle = validate_bundle(spec["bundle"])
    if (
        geometry["bundle"] != spec["bundle"]
        or any(v["branch_identity"] != bundle["branch_identity"] for v in (spec, geometry))
        or any(v["camera_order"] != bundle["camera_order"] for v in (spec, geometry))
        or camera not in spec["camera_order"]
    ):
        raise ValueError("Same branch, bundle and selected camera required")
    for value in (spec, geometry):
        if [row["tick"] for row in value["frames"]] != list(range(71)):
            raise ValueError("All 71 actual dense ticks required; no padding")
        for row in value["frames"]:
            if row["timestamp_us"] != bundle["source_start_timestamp_us"] + row["tick"] * 100000:
                raise ValueError("Actual 10Hz timestamps required")
    if spec["point_cloud_recipe"].get("resolution") != list(reversed(SIZE)):
        raise ValueError("Native guide resolution required")
    first = geometry["frames"][0]
    initial = spec["initial_reference_images"][camera]
    if (
        initial != geometry["initial_reference_images"][camera]
        or initial != first["cameras"][camera]["source_image"]
        or first["source_timestamp_us"] != first["timestamp_us"]
        or not np.allclose(first["delta_xy_yaw"], 0, atol=1e-5, rtol=0)
        or not np.allclose(
            first["source_ego_to_world"], geometry["initial_reference_ego_to_world"],
            atol=1e-8, rtol=0,
        )
    ):
        raise ValueError("Real tick-zero reference must match initial executed pose/time")
    _sealed(spec["files"], initial)
    _sealed(geometry["files"], initial)
    cam = first["cameras"][camera]
    error = _pose_error(
        cam["camera_to_world"],
        np.asarray(first["source_ego_to_world"]) @ np.asarray(cam["camera_to_ego"]),
    )
    if (
        error["translation_m"] > POSITION_TOLERANCE_M
        or error["rotation_rad"] > ROTATION_TOLERANCE_RAD
    ):
        raise ValueError("Initial reference camera pose is stale or differs")
    for tick in TICKS:
        _sealed(spec["files"], spec["frames"][tick]["cameras"][camera]["guide"])
    return spec, geometry, bundle, error


def validate_gs_inputs(path, prepared, camera, geometry, bundle):
    """Admit a bounded input receipt, never certify the resulting image geometry."""
    spec = json.loads(Path(path).read_text())
    if (
        spec.get("kind") != "streetcrafter_gs_matched_conditions"
        or spec.get("schema_version") != 1
        or spec.get("diagnostic_only") is not True
        or spec.get("trainable") is not False
        or spec.get("branch_identity") != bundle["branch_identity"]
        or spec.get("camera_order") != [camera]
        or Path(spec.get("source_conditions", "")).resolve() != Path(prepared).resolve()
    ):
        raise ValueError("Sealed same-branch, same-camera diagnostic GS receipt required")
    original = json.loads(Path(prepared).read_text())
    if spec["geometry"] != original["geometry"] or spec["bundle"] != original["bundle"]:
        raise ValueError("GS receipt geometry/bundle differs from actual executed conditions")
    check_files(spec["files"])
    for source in (spec["geometry"], spec["bundle"], spec["source_conditions"]):
        _sealed(spec["files"], source)
    if [row["tick"] for row in spec["frames"]] != TICKS:
        raise ValueError("GS refinement needs exactly 25 actual ticks 0..24; no padding")
    if len({row["source_timestamp_us"] for row in spec["frames"]}) != 25:
        raise ValueError("GS sequence must use 25 distinct actual native source captures")
    for row in spec["frames"]:
        tick = row["tick"]
        native_tick = int(round(
            (int(row["source_timestamp_us"]) - bundle["source_start_timestamp_us"]) / 100000
        ))
        if native_tick != tick:
            raise ValueError("Native GS source capture rounds to another physical tick")
        if (
            row["timestamp_us"] != bundle["source_start_timestamp_us"] + tick * 100000
            or abs(row["source_timestamp_us"] - row["timestamp_us"]) > 60000
            or row.get("geometry_tick") != tick
            or row.get("geometry_frame_source_timestamp_us")
            != geometry["frames"][tick]["source_timestamp_us"]
        ):
            raise ValueError("GS timestamps do not match actual executed geometry")
        cam = row["cameras"][camera]
        for key in ("gs", "guide", "render_mask", "guide_mask"):
            _sealed(spec["files"], cam[key])
        calibration = cam["calibration"]
        pose = np.asarray(calibration["camera_to_world"], float)
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0, 0, 0, 1]])
        _pose_error(pose, pose)
        intrinsic = np.asarray(calibration["distorted_intrinsic"], float)
        distortion = np.asarray(calibration["distortion"], float)
        size = calibration["output_calibration_size"]
        if (
            intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-8, rtol=0)
            or distortion.shape != (5,) or not np.isfinite(distortion).all()
            or len(size) != 2 or min(size) <= 0
            or size[0] * SIZE[1] != size[1] * SIZE[0]
        ):
            raise ValueError("Finite same-aspect optimized GS calibration required")
    if spec["initial_reference_images"][camera] != spec["frames"][0]["cameras"][camera]["gs"]:
        raise ValueError("GS refinement must initialize from matched GS frame zero")
    return spec


def pixel_statistics(pixels):
    """Descriptive RGB statistics, without a learned quality/admission claim."""
    value = np.asarray(pixels, np.float64) / 255
    return dict(
        mean_rgb=value.mean(axis=(0, 1)).tolist(),
        std_rgb=value.std(axis=(0, 1)).tolist(),
        black_fraction=float(np.all(value <= 1 / 255, axis=2).mean()),
        white_fraction=float(np.all(value >= 254 / 255, axis=2).mean()),
    )


def reference_statistics(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError("Reference comparison requires equal native RGB dimensions")
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference**2))
    return dict(
        mae_rgb_255=float(np.abs(difference).mean()),
        mse_rgb_255=mse,
        psnr_db=None if mse == 0 else float(10 * math.log10(255**2 / mse)),
        exact_match=mse == 0,
    )


def run(prepared, checkpoint, output, *, camera="CAM_F0", max_clips=1,
        expected_checkpoint_sha256=None, gs_sequence=None):
    """Run one clip. Use the supervised CLI when a hard deadline is required."""
    from PIL import Image

    if max_clips != 1:
        raise ValueError("Bounded diagnostic supports exactly one clip")
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[4]
    if repository / "exp" not in output.parents:
        raise ValueError("Diagnostic output must be a new directory under repository exp/")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(
        kind="streetcrafter_partial_clip_diagnostic", schema_version=1,
        backend="streetcrafter", status="validating", diagnostic_only=True,
        complete_conditioned_video=False, trainable=False, visual_admitted=False,
        source_control=False, gs_refinement_performed=False,
        selected_camera=camera, requested_ticks=TICKS, generated_ticks=[],
        frames=[], stage_seconds={},
        native_recipe=dict(
            steps=50, guidance_scale=2.5, seed=23, fps=10, frames_per_clip=25,
            resolution=list(reversed(SIZE)), training_free_guidance=False,
            scale=1.0, cond_indices=[0], joint_multiview=False,
        ),
        causal_history_verified=False, geometry_registration_certified=False,
        multiview_consistency_certified=False, temporal_consistency_certified=False,
        appearance_asset_scope="offline_full_source_traversal_including_future_observations",
        limitations=[
            "One camera and ticks 0-24 only; this is incomplete for the full video contract",
            "Aggregated MTGS colored clouds differ from native colored LiDAR conditions",
            "Generation uses point-cloud conditions, not a paired GS latent-refinement sequence",
            "Native guide masking does not certify preservation of generated scene geometry",
            "Offline full-traversal appearance and video conditioning are not history-causal",
            "Tick-zero score is conditioning-reference reconstruction, not novel-view quality",
            "No geometry, terrain, cross-camera, temporal or training-admission certification",
        ],
    )

    @contextmanager
    def stage(name):
        report["status"] = name
        report["active_stage_started_elapsed_seconds"] = time.monotonic() - started
        _save(output, report)
        print("StreetCrafter diagnostic stage:", name, flush=True)
        start = time.monotonic()
        try:
            yield
        finally:
            report["stage_seconds"][name] = time.monotonic() - start
            _save(output, report)
            print("StreetCrafter diagnostic completed:", name,
                  round(report["stage_seconds"][name], 3), "seconds", flush=True)

    _save(output, report)
    try:
        with stage("validate_conditions"):
            spec, geometry, bundle, error = validate_inputs(prepared, camera)
            report.update(
                branch_identity=bundle["branch_identity"],
                conditioning_report=str(Path(prepared).resolve()),
                conditioning_sha256=digest(prepared),
                geometry_report=spec["geometry"], geometry_sha256=digest(spec["geometry"]),
                input_camera_order=spec["camera_order"],
                omitted_cameras=[c for c in spec["camera_order"] if c != camera],
                omitted_ticks=list(range(25, 71)),
                history_ticks=bundle["history_ticks"], future_ticks=bundle["future_ticks"],
                initial_reference_pairing=dict(
                    tick=0, camera=camera, **error,
                    translation_tolerance_m=POSITION_TOLERANCE_M,
                    rotation_tolerance_rad=ROTATION_TOLERANCE_RAD,
                    exact_capture_timestamp=True,
                ),
            )
            gs_spec = None
            if gs_sequence is not None:
                gs_spec = validate_gs_inputs(gs_sequence, prepared, camera, geometry, bundle)
                report.update(
                    gs_sequence_report=str(Path(gs_sequence).resolve()),
                    gs_sequence_sha256=digest(gs_sequence),
                    initial_reference_pairing=dict(
                        tick=0, camera=camera, kind="same_camera_GS_render_not_raw_photo",
                        calibration=gs_spec["frames"][0]["cameras"][camera]["calibration"],
                    ),
                    raw_photo_reference_comparison=False,
                    native_mask_blending_active=False,
                )
                report["native_recipe"].update(
                    training_free_guidance=True, scale=0.3, active_denoising_steps=15,
                )
                report["limitations"][2] = (
                    "Same-camera GS latent refinement; generated pixels need independent geometry review"
                )
                report["limitations"][5] = (
                    "Output-vs-GS metrics measure image change, not quality; raw photos are unpaired"
                )
        with stage("prepare_inputs"):
            inputs = output / "inputs"
            inputs.mkdir()
            chosen = gs_spec or spec
            original = Path(chosen["initial_reference_images"][camera])
            original_copy = inputs / ("reference-original" + original.suffix.lower())
            shutil.copyfile(original, original_copy)
            with Image.open(original) as image:
                reference = np.asarray(image.convert("RGB").resize(SIZE, Image.Resampling.BICUBIC))
            reference_path = inputs / "reference-native.png"
            Image.fromarray(reference).save(reference_path)
            report["initial_reference"] = dict(
                source=str(original), source_sha256=digest(original),
                original_copy=str(original_copy), original_copy_sha256=digest(original_copy),
                native_copy=str(reference_path), native_copy_sha256=digest(reference_path),
                resize="Pillow RGB bicubic to native 1024x576; same pixels used for metrics",
            )
            guides = []
            gs_images, render_masks, guide_masks = [], [], []
            report["guide_inputs"] = []
            if gs_spec:
                report["gs_inputs"] = []
            for tick in TICKS:
                source = Path(chosen["frames"][tick]["cameras"][camera]["guide"])
                destination = inputs / f"{tick:04d}-{camera}-guide.png"
                shutil.copyfile(source, destination)
                with Image.open(destination) as image:
                    if image.mode != "RGB" or image.size != SIZE:
                        raise ValueError("Native RGB guide required; no resizing or padding")
                    pixels = np.asarray(image)
                guides.append(pixels.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)
                report["guide_inputs"].append(dict(
                    tick=tick, source=str(source), path=str(destination),
                    sha256=digest(destination),
                    nonzero_fraction=float(np.any(pixels != 0, axis=2).mean()),
                ))
                if gs_spec:
                    frame = gs_spec["frames"][tick]["cameras"][camera]
                    copied = {}
                    for key in ("gs", "render_mask", "guide_mask"):
                        destination = inputs / f"{tick:04d}-{camera}-{key}.png"
                        shutil.copyfile(frame[key], destination)
                        with Image.open(destination) as image:
                            if image.size != SIZE or image.mode != ("RGB" if key == "gs" else "L"):
                                raise ValueError("Native RGB GS / L mask required; no implicit resizing")
                            value = np.asarray(image)
                        if key == "gs":
                            gs_images.append(value)
                        else:
                            (render_masks if key == "render_mask" else guide_masks).append(
                                value[None].astype(np.float32) / 255
                            )
                        copied[key] = dict(path=str(destination), sha256=digest(destination))
                    report["gs_inputs"].append(dict(
                        tick=tick, calibration=frame["calibration"], **copied,
                    ))
            images = np.zeros((25, 3, SIZE[1], SIZE[0]), np.float32)
            images[0] = reference.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
            if gs_spec:
                images = np.stack(gs_images).transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1
        with stage("verify_checkpoint"):
            checkpoint = Path(checkpoint).resolve(strict=True)
            actual_sha = digest(checkpoint)
            if expected_checkpoint_sha256 is not None and actual_sha != expected_checkpoint_sha256:
                raise ValueError("Checkpoint checksum differs from verified download")
            report.update(checkpoint=str(checkpoint), checkpoint_sha256=actual_sha)
        with stage("load_native"):
            import torch

            model = runtime.load_native(checkpoint)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with stage("native_forward"):
            batch = dict(
                img_seq=torch.from_numpy(images), guide_seq=torch.from_numpy(np.stack(guides)),
                training_free_guidance=gs_spec is not None,
            )
            if gs_spec:
                batch.update(img_mask_seq=torch.from_numpy(np.stack(render_masks)),
                             guide_mask_seq=torch.from_numpy(np.stack(guide_masks)))
            samples = model.forward(
                batch, scale=0.3 if gs_spec else 1.0, cond_indices=[0],
            ).detach().float().cpu().permute(0, 2, 3, 1).numpy()
            if (
                samples.shape != (25, SIZE[1], SIZE[0], 3)
                or not np.isfinite(samples).all()
                or samples.min() < 0 or samples.max() > 1
            ):
                raise ValueError("Native clip output malformed or outside RGB [0,1]")
        with stage("save_rgb_and_metrics"):
            for tick, sample in zip(TICKS, samples):
                pixels = (sample * 255).round().astype(np.uint8)
                path = output / f"{tick:04d}-{camera}.png"
                Image.fromarray(pixels).save(path)
                row = dict(
                    tick=tick, timestamp_us=spec["frames"][tick]["timestamp_us"],
                    camera=camera, path=str(path), sha256=digest(path), size=list(SIZE),
                    pixel_statistics=pixel_statistics(pixels),
                )
                if gs_spec:
                    row["reference_comparison"] = dict(
                        evaluated=False, reason="Raw photograph is unpaired with optimized GS optics",
                    )
                    row["gs_input_comparison"] = dict(
                        **reference_statistics(pixels, gs_images[tick]),
                        kind="same_camera_GS_image_change_not_quality",
                        gs_sha256=report["gs_inputs"][tick]["gs"]["sha256"],
                        quality_score=False, novel_view_quality=False,
                    )
                elif tick == 0:
                    row["reference_comparison"] = dict(
                        **reference_statistics(pixels, reference),
                        kind="conditioning_reference_reconstruction_only",
                        source_image=spec["initial_reference_images"][camera],
                        source_sha256=report["initial_reference"]["source_sha256"],
                        novel_view_quality=False,
                    )
                else:
                    row["reference_comparison"] = dict(
                        evaluated=False,
                        reason="No independently validated paired source scene at this executed tick",
                    )
                report["frames"].append(row)
                report["generated_ticks"].append(tick)
            report["reference_comparison_count"] = 0 if gs_spec else 1
            report["gs_change_comparison_count"] = 25 if gs_spec else 0
            report["gs_refinement_performed"] = gs_spec is not None
        report["status"] = "generated_diagnostic_partial"
    except BaseException as exc:
        report.update(status="failed", failure=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--camera", default="CAM_F0")
    parser.add_argument("--max-clips", type=int, choices=[1], default=1)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--gs-sequence", help="Sealed actual same-camera GS+guide receipt; native scale=.3")
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.max_wall_seconds <= 1800:
        parser.error("wall-time budget must be 1..1800 seconds")
    if args.worker:
        run(args.prepared, args.checkpoint, args.output, camera=args.camera,
            max_clips=args.max_clips, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            gs_sequence=args.gs_sequence)
        return
    if Path(args.output).exists():
        parser.error("output directory already exists; choose a fresh exp/ diagnostic directory")
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"])
    try:
        returncode = process.wait(timeout=args.max_wall_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        report_path = Path(args.output) / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            active = report.get("status")
            if "active_stage_started_elapsed_seconds" in report:
                report.setdefault("stage_seconds", {})[active] = max(
                    0, time.monotonic() - started - report["active_stage_started_elapsed_seconds"],
                )
                report["interrupted_stage"] = active
            report.update(status="failed", failure=type(exc).__name__,
                          elapsed_seconds=time.monotonic() - started,
                          wall_time_budget_seconds=args.max_wall_seconds,
                          terminated_worker_pid=process.pid)
            _save(report_path.parent, report)
        raise SystemExit(124 if isinstance(exc, subprocess.TimeoutExpired) else 130) from exc
    if returncode:
        raise SystemExit(returncode)
    report_path = Path(args.output) / "report.json"
    report = json.loads(report_path.read_text())
    report.update(wall_time_budget_seconds=args.max_wall_seconds,
                  supervised_elapsed_seconds=time.monotonic() - started,
                  worker_pid=process.pid, visible_cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    _save(report_path.parent, report)
    print(json.dumps({"report": str(report_path), "status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()

