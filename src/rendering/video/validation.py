"""Independent output integrity checks; rendering is never visual admission."""

import struct
import zlib
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rendering.video.bundle import CAMERAS, FUTURE, HISTORY, validate_bundle
from driversi.rsi.plugins.synthesis import check_files


def png_size(path):
    """Read the hashed native RGB PNG header without importing GPU-env Pillow."""
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if (
        len(header) != 33
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[8:16] != b"\x00\x00\x00\x0dIHDR"
    ):
        raise ValueError("Native RGB PNG required")
    if zlib.crc32(header[12:29]) != struct.unpack(">I", header[29:33])[0]:
        raise ValueError("Corrupt PNG header")
    width, height, depth, colour, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    if (
        not width
        or not height
        or (depth, colour, compression, filtering, interlace) != (8, 2, 0, 0, 0)
    ):
        raise ValueError("Native non-interlaced 8bit RGB PNG required")
    return [width, height]


def validate_output(path):
    report = read_mapping(path)
    if (
        report.get("kind") != "conditioned_video_output"
        or report.get("status") != "generated_diagnostic"
    ):
        raise ValueError("Incomplete native video output")
    condition_path = report["conditioning_report"]
    if digest(condition_path) != report["conditioning_sha256"]:
        raise ValueError("Condition report changed")
    conditions = read_mapping(condition_path)
    check_files(conditions["files"])
    geometry = read_mapping(conditions["geometry"])
    check_files(geometry["files"])
    bundle = validate_bundle(geometry["bundle"])
    if any(
        value["branch_identity"] != bundle["branch_identity"]
        for value in (report, conditions, geometry)
    ):
        raise ValueError("Different physical branch in video chain")
    if (
        report["camera_order"] != list(CAMERAS)
        or report["history_ticks"] != list(HISTORY)
        or report["future_ticks"] != list(FUTURE)
    ):
        raise ValueError("Camera/time contract changed")
    if report["source_control"] != geometry["source_control"]:
        raise ValueError("Source and novel-view modes confused")
    required = list(HISTORY + FUTURE)
    if [frame["tick"] for frame in report["frames"]] != required:
        raise ValueError("Four actual history and ten actual future frames required")
    images = {}
    for frame in report["frames"]:
        if frame["timestamp_us"] != bundle["source_start_timestamp_us"] + frame["tick"] * 100000:
            raise ValueError("Video timestamp differs from physics")
        if set(frame["cameras"]) != set(CAMERAS):
            raise ValueError("Incomplete camera rig")
        for image in frame["cameras"].values():
            p = Path(image["path"])
            if digest(p) != image["sha256"]:
                raise ValueError("Video pixels changed")
            if png_size(p) != image["size"]:
                raise ValueError("Unexpected video format")
            images[str(p)] = image["sha256"]
    if report.get("trainable") is not False or report.get("visual_admitted") is not False:
        raise ValueError("Generator cannot authorize its own training pixels")
    return report, geometry, images


def source_comparison(novel_path, source_path):
    """Compare matched real source captures, never novel views, with generated control."""
    from PIL import Image

    novel, novel_geometry, _ = validate_output(novel_path)
    source, source_geometry, _ = validate_output(source_path)
    if (
        novel["source_control"]
        or not source["source_control"]
        or novel["backend"] != source["backend"]
        or novel["branch_identity"] != source["branch_identity"]
    ):
        raise ValueError("Paired native source reconstruction control required")
    if novel["backend"] != "worlddreamer":
        raise ValueError("Source image preprocessing for this backend not yet verified")
    if novel["native_recipe"] != source["native_recipe"]:
        raise ValueError("Source and novel generation recipes differ")
    geometric = {frame["tick"]: frame for frame in source_geometry["frames"]}
    scores = []
    for frame in source["frames"]:
        for cam, image in frame["cameras"].items():
            raw = Path(geometric[frame["tick"]]["cameras"][cam]["source_image"])
            with Image.open(raw) as captured:
                width, height = captured.size
                nw, nh = int(width * 0.21), int(height * 0.21)
                left, top = int((nw - 400) / 2), nh - 224
                target = np.asarray(
                    captured.convert("RGB")
                    .resize((nw, nh), Image.Resampling.BILINEAR)
                    .crop((left, top, left + 400, top + 224)),
                    np.float64,
                )
            with Image.open(image["path"]) as generated:
                prediction = np.asarray(generated, np.float64)
            if prediction.shape != target.shape:
                raise ValueError("No upsampling may conceal native output resolution")
            mse = float(np.mean((prediction - target) ** 2))
            psnr = None if mse == 0 else float(10 * np.log10(255**2 / mse))
            scores.append(
                dict(
                    tick=frame["tick"],
                    camera=cam,
                    source_image=str(raw),
                    source_sha256=digest(raw),
                    generated_sha256=image["sha256"],
                    mse=mse,
                    psnr_db=psnr,
                )
            )
    values = [row["psnr_db"] for row in scores if row["psnr_db"] is not None]
    mean = float(np.mean(values)) if len(values) == len(scores) else None
    reasons = [
        "independent_generated_actor_geometry_unverified",
        "generated_multiview_and_temporal_consistency_unverified",
        "terrain_registration_unverified",
    ]
    if mean is not None and mean < 27:
        reasons.insert(0, "source_reconstruction_below_unchanged_27db_gate")
    return dict(
        schema_version=1,
        kind="conditioned_video_quality_diagnostic",
        branch_identity=novel["branch_identity"],
        novel_report=str(Path(novel_path).resolve()),
        source_report=str(Path(source_path).resolve()),
        files={str(Path(p).resolve()): digest(p) for p in (novel_path, source_path)},
        source_psnr_mean_db=mean,
        source_psnr_threshold_db=27.0,
        source_image_count=len(scores),
        source_scores=scores,
        comparison_domain="native224x400_exact_reference_resize_crop; not equivalent to full_resolution_GS_metrics",
        novel_view_psnr_used=False,
        admitted=False,
        trainable=False,
        rejection_reasons=reasons,
        independent_visual_review_required=True,
        caveat="Image fidelity, exact actor geometry and learnability are distinct. Passing source PSNR alone cannot admit video.",
    )
