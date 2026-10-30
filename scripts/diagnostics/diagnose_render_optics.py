"""Fixed-source MTGS optics diagnosis; no world edits or image completion."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime, project_source_lens, resample_source_lens

METRIC_CONVENTIONS = {
    "metric_pixels": "post-quantization uint8 RGB, range 0..255",
    "reference_resize": "PIL LANCZOS",
    "metric_support": "intersection of normal source-lens validity and area-downsampled native validity (>=1-1e-6)",
    "blue_signature": {
        "rows": [0, 80],
        "predicate": "B>150 and B>R+60 and B>G+40",
        "warning": "raw and lens images cover different rays; counts are not matched-FOV artifact reduction",
    },
    "temporal": "unwarped reference residual change on consecutive common support, not optical-flow-compensated flicker",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=1752)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    runtime = MTGSRuntime(
        **{k: Path(cfg[k]) for k in ("config_path", "checkpoint_path", "road_block_config")},
        scale=0.5,
        trusted_assets=True,
    )
    camera = runtime.source_camera(args.index)
    projection = runtime.source_lens_projection(camera)
    result = runtime.render_camera(camera)
    lens, valid = resample_source_lens(result["rgb"], projection["map_x"], projection["map_y"])
    refs = runtime.source_reference(args.index, (lens.shape[1], lens.shape[0]))
    source = refs["raw"]
    Image.fromarray(refs["checkpoint_target"]).save(args.output / "reference-checkpoint.png")
    for name, pixels in [
        ("raw", result["rgb"]),
        ("lens", lens),
        ("reference", source),
        ("valid", valid.astype(np.uint8) * 255),
    ]:
        Image.fromarray(pixels).save(args.output / (name + ".png"))
    np.savez_compressed(
        args.output / "projection.npz", map_x=projection["map_x"], map_y=projection["map_y"]
    )
    old = np.asarray(Image.open(args.config.parent / "gs-release/fault-reference.png"))
    old_lens, old_valid = resample_source_lens(old, projection["map_x"], projection["map_y"])
    Image.fromarray(old_lens).save(args.output / "old-projected.png")

    # Diagnostic color signature for the reported top fragment, never a repair mask.
    def blue(rgb):
        rgb = rgb.astype(float)
        return (
            (rgb[..., 2] > 150)
            & (rgb[..., 2] > rgb[..., 0] + 60)
            & (rgb[..., 2] > rgb[..., 1] + 40)
        )

    report = {
        "measurement_conventions": METRIC_CONVENTIONS,
        "photometric": refs["photometric"],
        "index": args.index,
        "frame": int(camera.metadata["frame_idx"]),
        "travel_id": int(camera.metadata["travel_id"]),
        "camera_metadata_keys": list(camera.metadata),
        "source_file": str(runtime.sources.image_filenames[args.index]),
        "projection": {k: v for k, v in projection.items() if k not in ("map_x", "map_y")},
        "raw_equals_old": bool(np.array_equal(result["rgb"], old)),
        "old_difference_pixels": int(np.any(result["rgb"] != old, axis=-1).sum()),
        "top_blue_pixels": {
            k: int(blue(v)[:80].sum())
            for k, v in [
                ("raw", result["rgb"]),
                ("lens", lens),
                ("reference", source),
                ("old_projected", old_lens),
            ]
        },
        "valid_pixels": int(valid.sum()),
        "pixel_count": int(valid.size),
        "seconds": time.perf_counter() - start,
        "trainable": False,
        "visible_nodes": camera.metadata.get("visible_tokens"),
        "all_node_names": list(runtime.model.gaussian_models),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("all_node_names", "visible_nodes")},
            default=str,
        ),
        flush=True,
    )
    if args.ablate:
        layers = {}
        for group, nodes in {
            "background": ["background"],
            "skybox": ["skybox"],
            "rigid": [n for n in runtime.model.gaussian_models if n.startswith("rigid_object_")],
        }.items():
            probe = copy.deepcopy(camera)
            probe.metadata["visible_tokens"] = nodes
            layer = runtime.render_camera(probe)
            projected = project_source_lens(layer, projection)
            for mode, pixels in [("raw", layer["rgb"]), ("lens", projected["rgb"])]:
                Image.fromarray(pixels).save(args.output / (group + "-" + mode + ".png"))
            layers[group] = {
                "top_blue_raw": int(blue(layer["rgb"])[:80].sum()),
                "top_blue_lens": int(blue(projected["rgb"])[:80].sum()),
            }
        report["node_ablation"] = layers
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print("LAYERS", json.dumps(layers), flush=True)
    if args.episode:
        import cv2

        episode = json.loads(args.episode.read_text())
        indices = [
            next(c["source_seed"]["index"] for c in f["cameras"] if c["camera_id"] == "CAM_F0")
            for f in episode["frames"]
        ]
        entries = []
        previous = None
        for index in indices:
            cam = runtime.source_camera(index)
            native = copy.deepcopy(cam)
            native.rescale_output_resolution(scaling_factor=2.0)
            lo = runtime.render_camera(cam)
            hi = runtime.render_camera(native)
            np.testing.assert_array_equal(lo["camera_to_world"], hi["camera_to_world"])
            low = project_source_lens(lo, runtime.source_lens_projection(cam))
            high = project_source_lens(hi, runtime.source_lens_projection(native))
            size = (low["rgb"].shape[1], low["rgb"].shape[0])
            down = cv2.resize(high["rgb"], size, interpolation=cv2.INTER_AREA)
            high_valid = (
                cv2.resize(high["valid"].astype(np.float32), size, interpolation=cv2.INTER_AREA)
                >= 1 - 1e-6
            )
            common = low["valid"] & high_valid
            refs = runtime.source_reference(index, size)
            real = refs["checkpoint_target"]
            folder = args.output / ("frame-%04d" % index)
            folder.mkdir()
            for name, pixels in [
                ("raw", lo["rgb"]),
                ("lens", low["rgb"]),
                ("native", down),
                ("reference", refs["raw"]),
                ("reference-checkpoint", real),
                ("valid", common.astype(np.uint8) * 255),
            ]:
                Image.fromarray(pixels).save(folder / (name + ".png"))

            def measure(pixels, mask):
                error = pixels.astype(float) - real.astype(float)
                mse = float(np.mean(error[mask] ** 2))
                return {
                    "psnr_db": None if mse == 0 else float(10 * np.log10(255**2 / mse)),
                    "mae_uint8": float(np.mean(abs(error[mask]))),
                }

            roi = np.zeros_like(common)
            roi[240:500, 600:960] = True
            item = {
                "index": index,
                "frame": int(cam.metadata["frame_idx"]),
                "timestamp_us": int(
                    runtime.source_frames[
                        (int(cam.metadata["travel_id"]), cam.metadata["frame_token"])
                    ]["timestamp"]
                ),
                "photometric": refs["photometric"],
                "path": folder.name,
                "low": measure(low["rgb"], common),
                "native": measure(down, common),
                "vehicle_roi_low": measure(low["rgb"], common & roi),
                "vehicle_roi_native": measure(down, common & roi),
                "lens_blue_top": int(blue(low["rgb"])[:80].sum()),
                "native_blue_top": int(blue(down)[:80].sum()),
                "common_valid_fraction": float(common.mean()),
                "render_low_seconds": lo["render_seconds"],
                "render_native_seconds": hi["render_seconds"],
            }
            errors = [
                low["rgb"].astype(float) - real.astype(float),
                down.astype(float) - real.astype(float),
            ]
            if previous is not None:
                mask = common & previous[1]
                item["unwarped_residual_change_mae"] = {
                    k: float(np.mean(abs(e - prev)[mask]))
                    for k, e, prev in zip(["low", "native"], errors, previous[0])
                }
            previous = errors, common
            entries.append(item)
            (args.output / "sequence.json").write_text(
                json.dumps(
                    {
                        "scope": "source-trajectory same-pose comparison, not counterfactual rollout; unwarped temporal residual is diagnostic, not motion-compensated flicker",
                        "reference_domain": "official_checkpoint_exposure_aligned",
                        "measurement_conventions": METRIC_CONVENTIONS,
                        "status": "completed" if len(entries) == len(indices) else "running",
                        "expected_frames": len(indices),
                        "trainable": False,
                        "vehicle_roi_xyxy": [600, 240, 960, 500],
                        "frames": entries,
                    },
                    indent=2,
                )
            )
            print("FRAME", json.dumps(item), flush=True)

        from driversi.rendering.gaussian.mtgs_viewer import export_optics_viewer

        export_optics_viewer(args.output, args.output / "index.html")


if __name__ == "__main__":
    main()
