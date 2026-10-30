"""Canonical source reconstruction checks before spending a physical rollout.

This produces an asset diagnostic, never synthetic-data admission. A subsequent
physical branch must still pass the normal rendering and trajectory contracts.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def source_window(frames, token):
    ordered = sorted(frames, key=lambda f: int(f["timestamp"]))
    matches = [i for i, f in enumerate(ordered) if f["token"] == token]
    if len(matches) != 1:
        raise ValueError("A unique source anchor is required")
    anchor = int(ordered[matches[0]]["timestamp"])
    stamps = np.asarray([int(f["timestamp"]) for f in ordered])
    # Reconstruction metadata can be10Hz; select real captures at2Hz by time,
    # never by assuming that four metadata rows mean two seconds.
    targets = anchor + np.arange(-4, 11) * 500000
    indices = [int(np.abs(stamps-t).argmin()) for t in targets]
    if len(set(indices)) != 15 or np.any(np.abs(stamps[indices]-targets) > 50000):
        raise ValueError("Asset lacks the complete source7-second window")
    selected = [ordered[i] for i in indices]
    dt = np.diff([int(f["timestamp"]) for f in selected]) / 1e6
    if len(selected) != 15 or np.any(np.abs(dt-.5) > .05):
        raise ValueError("Continuous2Hz source captures required")
    if any(f.get("skipped") or not all(c in f["cams"] for c in CAMERAS)
           for f in selected):
        raise ValueError("Every source time requires all four calibrated cameras")
    return selected


def run(asset, sensor_root, output, *, minimum_source_psnr_db=25.):
    from PIL import Image
    from driversi.rendering.gaussian.worldengine_portable import PortableRenderer
    from driversi.rendering.diagnostics.worldengine_admission import source_view_psnr
    if minimum_source_psnr_db not in (25.,27.):
        raise ValueError('Explicit versioned source threshold25 or27 required')

    out = Path(output).resolve()
    if not out.is_relative_to(project_root() / "exp") or out.exists():
        raise ValueError("A new owned diagnostic output is required")
    bundle = Path(asset["bundle_root"]).resolve(strict=True)
    metadata = bundle / "video_scene_dict.pkl"
    checkpoint = bundle / "background" / (asset["asset_id"] + ".ckpt")
    if digest(metadata) != asset["metadata_sha256"]:
        raise ValueError("Asset metadata differs from its registered source")
    with metadata.open("rb") as stream:
        videos = pickle.load(stream)
    if len(videos) != 1:
        raise ValueError("One registered source video is required")
    frames = source_window(next(iter(videos.values()))["frame_infos"], asset["token"])
    sensor_root = Path(sensor_root).resolve(strict=True)
    for frame in frames:
        for camera in CAMERAS:
            real = (sensor_root / frame["cams"][camera]["data_path"]).resolve(strict=True)
            if not real.is_relative_to(sensor_root / asset["source_log"]):
                raise ValueError("Source image is outside the selected training log")
    renderer = PortableRenderer(checkpoint, metadata=metadata)
    out.mkdir(parents=True)
    write_json(out / "request.json", dict(asset=asset, sensor_root=str(sensor_root),
        minimum_source_psnr_db=minimum_source_psnr_db,
        inputs={str(p): digest(p) for p in (metadata, checkpoint)},
        canonical_token_window=[f["token"] for f in frames], trainable=False))
    rows, files = [], {}
    for index, frame in enumerate(frames):
        for camera in CAMERAS:
            pixels, evidence = renderer.render(frame, camera, width=960)
            path = out / f"{index*5:04d}-{camera}-source-render.png"
            Image.fromarray(pixels).save(path)
            real = sensor_root / frame["cams"][camera]["data_path"]
            factor = float(frame["cams"][camera].get("v_adjust", 1.))
            files[str(path)], files[str(real)] = digest(path), digest(real)
            rows.append(dict(tick=index*5, camera=camera, source_token=frame["token"],
                rendered_source_image=str(path), real_source_image=str(real),
                exposure_factor=factor, measured_psnr_db=source_view_psnr(path, real, factor),
                source_camera_to_world=evidence["camera_to_world"], source_motions={},
                source_appearance_timestamp_us=int(frame["timestamp"])))
        write_json(out / "progress.json", dict(completed_images=len(rows), total_images=60))
    scores = [row["measured_psnr_db"] for row in rows]
    result = dict(kind="canonical_source_asset_preflight", schema_version=1,
        source_mode="canonical_source_pose_without_motion_v1", asset=asset,
        inputs={str(p): digest(p) for p in (metadata, checkpoint)}, files=files,
        producer_code={str(Path(__file__).resolve()): digest(__file__)}, frames=rows,
        mean_psnr_db=float(np.mean(scores)), passed_preflight=bool(np.mean(scores)>minimum_source_psnr_db),
        threshold=minimum_source_psnr_db, operator='>', output_image_size=[960, 540], trainable=False,
        scope="source_reconstruction_only_not_novel_view_or_synthetic_admission")
    write_json(out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result = run(**read_mapping(args.job))
    print(json.dumps({k: result[k] for k in ("mean_psnr_db", "passed_preflight", "trainable")}))


if __name__ == "__main__":
    main()
