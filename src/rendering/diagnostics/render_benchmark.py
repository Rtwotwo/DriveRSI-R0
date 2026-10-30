"""Bounded scalar/batched WorldEngine comparison on frozen executed branches.

This diagnostic writes no production dataset or vector index.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from driversi.rendering.gaussian.source_session import SourceSession, replay_adapter

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def branch_inputs(path):
    from driversi.agents.backends.native import digest

    path = Path(path)
    report = json.loads(path.read_text())
    for name, sha in report["inputs"].items():
        if digest(name) != sha:
            raise ValueError("Frozen branch input changed: " + name)
    paths = list(map(Path, report["inputs"]))
    checkpoint = next(p for p in paths if p.suffix == ".ckpt")
    metadata = next(p for p in paths if p.name == "video_scene_dict.pkl")
    return (
        dict(
            scene_path=next(p for p in paths if p.name == "scene.npz"),
            rollout_path=next(p for p in paths if p.name == "teacher.npz"),
            bundle=metadata.parent,
            raw_log=next(p for p in paths if p.suffix == ".pkl" and p != metadata),
        ),
        checkpoint,
        metadata,
        report,
    )


def image_parity(reference, actual):
    from PIL import Image

    reference, actual = Path(reference), Path(actual)
    old, new = json.loads(reference.read_text()), json.loads(actual.read_text())
    if len(old["frames"]) != len(new["frames"]):
        raise ValueError("Different render coverage")
    rows = []
    for a, b in zip(old["frames"], new["frames"]):
        if a["tick"] != b["tick"] or a["motions"] != b["motions"]:
            raise ValueError("Different physical render requests")
        for cam in CAMERAS:
            ae, be = a["cameras"][cam], b["cameras"][cam]
            x = np.asarray(Image.open(reference.parent / ae["file"]))
            y = np.asarray(Image.open(actual.parent / be["file"]))
            diff = np.abs(x.astype(np.int16) - y.astype(np.int16))
            evidence_differences = [k for k in ae if k != "file" and ae[k] != be.get(k)]
            rows.append(
                dict(
                    tick=a["tick"],
                    camera=cam,
                    pixels_exact=bool(np.array_equal(x, y)),
                    max_abs=int(diff.max()),
                    mean_abs=float(diff.mean()),
                    evidence_differences=evidence_differences,
                )
            )
    return dict(
        images=len(rows),
        pixels_exact=all(r["pixels_exact"] for r in rows),
        evidence_exact=all(not r["evidence_differences"] for r in rows),
        max_abs=max(r["max_abs"] for r in rows),
        rows=rows,
    )


def compare_source(reference, actual):
    from PIL import Image

    old, new = json.loads(Path(reference).read_text()), json.loads(Path(actual).read_text())
    if len(old["frames"]) != len(new["frames"]):
        raise ValueError("Different source coverage")
    rows = []
    for a, b in zip(old["frames"], new["frames"]):
        if (a["tick"], a["camera"], a["source_token"]) != (
            b["tick"],
            b["camera"],
            b["source_token"],
        ):
            raise ValueError("Different canonical source request")
        x, y = (np.asarray(Image.open(z["rendered_source_image"])) for z in (a, b))
        fields = (set(a) | set(b)) - {"rendered_source_image", "measured_psnr_db"}
        differences = sorted(k for k in fields if a.get(k) != b.get(k))
        rows.append(
            dict(
                tick=a["tick"],
                camera=a["camera"],
                pixels_exact=bool(np.array_equal(x, y)),
                psnr_delta=b["measured_psnr_db"] - a["measured_psnr_db"],
                evidence_differences=differences,
            )
        )
    return dict(
        images=len(rows),
        pixels_exact=all(r["pixels_exact"] for r in rows),
        evidence_exact=all(not r["evidence_differences"] for r in rows),
        max_psnr_delta=max(abs(r["psnr_delta"]) for r in rows),
        mean_psnr=float(np.mean([f["measured_psnr_db"] for f in new["frames"]])),
        rows=rows,
    )


def main(config_path, output, *, repeats=3):
    import torch

    from driversi.rendering.gaussian.worldengine_batch import BatchedPortableRenderer
    from driversi.rendering.gaussian.worldengine_portable import PortableRenderer
    from driversi.rendering.gaussian.worldengine_replay import run as replay
    from driversi.rendering.gaussian.worldengine_source import run as source_run

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.25, 0)
    branches = [branch_inputs(x["render_report"]) for x in config["branches"]]
    checkpoint, metadata = branches[0][1:3]
    if any(x[1:3] != (checkpoint, metadata) for x in branches):
        raise ValueError("This benchmark is explicitly a single-asset two-branch comparison")
    terrain = metadata.parent / "road_height_map"
    # Same process imports/JIT/cache for both arms; first-ever startup is reported separately.
    start = time.perf_counter()
    warm = PortableRenderer(checkpoint, metadata=metadata, terrain=terrain)
    f = branches[0][3]["frames"][0]
    warm.render(
        f["source_token"],
        CAMERAS[0],
        motions=f["motions"],
        absent_physical=f["absent_physical_with_asset"],
        width=960,
    )
    del warm
    gc.collect()
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start
    results = []
    modes = ["scalar", "optimized"]
    for repetition in range(repeats):
        for mode in modes if repetition % 2 == 0 else list(reversed(modes)):
            root = output / f"r{repetition}-{mode}"
            root.mkdir()
            write(
                output / "status.json",
                dict(status="running", repetition=repetition, mode=mode, completed=len(results)),
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            asset_seconds = 0.0
            if mode == "optimized":
                t = time.perf_counter()
                resident = BatchedPortableRenderer(checkpoint, metadata=metadata, terrain=terrain)
                canonical = BatchedPortableRenderer(checkpoint, metadata=metadata)
                import inspect

                session = SourceSession(
                    canonical,
                    checkpoint,
                    metadata,
                    [
                        Path(inspect.getfile(PortableRenderer)),
                        Path(inspect.getfile(BatchedPortableRenderer)),
                        Path(__file__),
                    ],
                )
                torch.cuda.synchronize()
                asset_seconds = time.perf_counter() - t
            timings = []
            for index, (kwargs, _, _, original) in enumerate(branches):
                target = root / str(index)
                target.mkdir()
                t = time.perf_counter()
                if mode == "scalar":
                    renderer = PortableRenderer(checkpoint, metadata=metadata, terrain=terrain)
                else:
                    renderer = replay_adapter(resident)
                replay(
                    **kwargs,
                    output=target / "render",
                    future_points=10,
                    width=960,
                    renderer=renderer,
                )
                torch.cuda.synchronize()
                render_seconds = time.perf_counter() - t
                if mode == "scalar":
                    del renderer
                    gc.collect()
                t = time.perf_counter()
                if mode == "scalar":
                    source_run(target / "render/report.json", target / "source")
                    source_path, hit = target / "source/report.json", False
                else:
                    source_path, hit = session.run(target / "render/report.json", target / "source")
                torch.cuda.synchronize()
                timings.append(
                    dict(
                        branch=index,
                        render_seconds=render_seconds,
                        source_seconds=time.perf_counter() - t,
                        cache_hit=hit,
                        render_report=str(target / "render/report.json"),
                        source_report=str(source_path),
                    )
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            peak = torch.cuda.max_memory_allocated()
            if mode == "optimized":
                del renderer, resident, canonical, session
            parity = []
            for row in timings:
                i = row["branch"]
                parity.append(
                    dict(
                        branch=i,
                        render=image_parity(
                            config["branches"][i]["render_report"], row["render_report"]
                        ),
                        source=compare_source(
                            config["branches"][i]["source_report"], row["source_report"]
                        ),
                    )
                )
            value = dict(
                repetition=repetition,
                mode=mode,
                seconds=elapsed,
                asset_setup_seconds=asset_seconds,
                peak_gpu_bytes=peak,
                stages=timings,
                parity=parity,
            )
            results.append(value)
            write(root / "result.json", value)
            write(
                output / "results.json",
                dict(results=results, warmup_seconds=warmup, new_production_samples=0),
            )
            print(
                json.dumps(
                    {k: value[k] for k in ("repetition", "mode", "seconds", "peak_gpu_bytes")}
                ),
                flush=True,
            )
    exact = all(
        p["render"]["pixels_exact"]
        and p["render"]["evidence_exact"]
        and p["source"]["pixels_exact"]
        and p["source"]["evidence_exact"]
        and p["source"]["max_psnr_delta"] == 0
        for r in results
        for p in r["parity"]
    )
    summary = dict(
        status="verified" if exact else "parity_failed",
        results=results,
        warmup_seconds=warmup,
        new_production_samples=0,
        scope="two frozen 7s branches: actual replay diagnostics, new images, source images/PSNR and cache validation; no native export or feature timing",
    )
    write(output / "summary.json", summary)
    write(output / "status.json", dict(status=summary["status"], completed=len(results)))
    if not exact:
        raise RuntimeError("Pixel/evidence/source parity failed; not production eligible")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    main(args.config, args.output, repeats=args.repeats)
