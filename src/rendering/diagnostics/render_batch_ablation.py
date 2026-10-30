"""Warm resident camera-batch ablation; excludes image writes and replay setup."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from driversi.rendering.diagnostics.render_benchmark import CAMERAS, branch_inputs, write


def main(config_path, output):
    import torch
    from PIL import Image

    from driversi.rendering.gaussian.worldengine_batch import BatchedPortableRenderer
    from driversi.rendering.gaussian.worldengine_portable import PortableRenderer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.25, 0)
    config = json.loads(Path(config_path).read_text())
    branches = [branch_inputs(x["render_report"]) for x in config["branches"]]
    checkpoint, metadata = branches[0][1:3]
    kwargs = dict(metadata=metadata, terrain=metadata.parent / "road_height_map")
    scalar, batched = (
        PortableRenderer(checkpoint, **kwargs),
        BatchedPortableRenderer(checkpoint, **kwargs),
    )
    # Same persistent objects survive A -> B -> A. No trajectory rebasing.
    rows = []
    modes = [("scalar", 1), ("cached_optics_c1", 1), ("batch_c2", 2), ("batch_c4", 4)]
    for repeat, branch_index in enumerate((0, 1, 0)):
        original = branches[branch_index][3]
        archived = Path(config["branches"][branch_index]["render_report"]).parent
        for mode, size in modes if repeat % 2 == 0 else list(reversed(modes)):
            # One timepoint warmup per mode excludes first-allocation effects.
            f = original["frames"][0]
            options = dict(
                motions=f["motions"], absent_physical=f["absent_physical_with_asset"], width=960
            )
            if mode == "scalar":
                for camera in CAMERAS:
                    scalar.render(f["source_token"], camera, **options)
            else:
                for start in range(0, 4, size):
                    batched.render_many(f["source_token"], CAMERAS[start : start + size], **options)
            elapsed, max_error, evidence_exact = 0.0, 0, True
            torch.cuda.reset_peak_memory_stats()
            for frame in original["frames"]:
                options = dict(
                    motions=frame["motions"],
                    absent_physical=frame["absent_physical_with_asset"],
                    width=960,
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                if mode == "scalar":
                    result = {
                        c: scalar.render(frame["source_token"], c, **options) for c in CAMERAS
                    }
                else:
                    result = {}
                    for first in range(0, 4, size):
                        result.update(
                            batched.render_many(
                                frame["source_token"], CAMERAS[first : first + size], **options
                            )
                        )
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                for camera, (pixels, evidence) in result.items():
                    expected = frame["cameras"][camera]
                    ref = np.asarray(Image.open(archived / expected["file"]))
                    max_error = max(
                        max_error, int(np.abs(pixels.astype(np.int16) - ref.astype(np.int16)).max())
                    )
                    evidence_exact = evidence_exact and all(
                        expected.get(k) == v for k, v in evidence.items()
                    )
            row = dict(
                repetition=repeat,
                branch=branch_index,
                mode=mode,
                seconds=elapsed,
                images=60,
                max_pixel_error=max_error,
                evidence_exact=evidence_exact,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            )
            rows.append(row)
            write(output / "progress.json", dict(rows=rows))
            print(json.dumps(row), flush=True)
    exact = all(r["max_pixel_error"] == 0 and r["evidence_exact"] for r in rows)
    write(
        output / "summary.json",
        dict(
            status="verified" if exact else "failed",
            rows=rows,
            scope="resident render API only: CPU preparation, rasterization, transfer and distortion; excludes PNG, replay diagnostics, source PSNR, export and features",
            branch_order=[0, 1, 0],
            new_production_samples=0,
        ),
    )
    if not exact:
        raise ValueError("Camera batching or branch-reset parity failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.config, args.output)
