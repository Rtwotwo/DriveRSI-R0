"""Rank exact NAVSIM training windows by actual loaded GS asset coverage.

No partial window is selected for synthesis. The catalog must come from the
native replay runtime; its checkpoint/metadata hashes are rechecked here.
"""

import argparse
import json
import pickle
from pathlib import Path

import yaml

from driversi.agents.backends.native import digest
from driversi.data.assets.mtgs_binding import match_navsim_traversal
from driversi.data.assets.mtgs_coverage import audit_binding_assets, verify_source_inventory
from driversi.data.sources.navsim import compile_frames


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def complete_window_starts(indices, *, future_points=10):
    """Require all real 2 Hz source frames through the model's actual horizon."""
    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Eight or ten native future points required")
    indices = sorted(set(indices))
    support = set(indices)
    last_offset = 4 + future_points
    return [start for start in indices
            if all(start + offset in support for offset in range(last_offset + 1))]


def run(catalog_path, metadata_path, checkpoint_path, log_root, split_path, output):
    catalog = json.loads(catalog_path.read_text())
    if catalog["checkpoint_sha256"] != digest(checkpoint_path) or catalog[
        "metadata_sha256"
    ] != digest(metadata_path):
        raise ValueError("Asset catalog checkpoint or source metadata changed")
    allowed = set(yaml.safe_load(split_path.read_text())["log_names"])
    with metadata_path.open("rb") as stream:
        videos = pickle.load(stream)
    rows = []
    source_logs = {}
    for video in videos.values():
        log_name = video["log_name"]
        if log_name not in allowed:
            rows.append(dict(source_log=log_name, status="excluded_not_navtrain"))
            continue
        log = log_root / (log_name + ".pkl")
        source_logs[str(log)] = digest(log)
        with log.open("rb") as stream:
            frames = pickle.load(stream)
        nav_indices = {f["token"]: i for i, f in enumerate(frames)}
        indices = sorted(
            nav_indices[f["token"]]
            for f in video["frame_infos"]
            if not f.get("skipped") and f["token"] in nav_indices
        )
        # Four changed histories end at2s; ten future points end at7s.
        # Native4s PDM is shorter than the image/trajectory consumer contract.
        starts = complete_window_starts(indices)
        for start in starts:
            row = dict(source_log=log_name, start_frame=start, source_token=frames[start]["token"])
            try:
                scene = compile_frames(
                    frames,
                    start=start,
                    duration=7.0,
                    split="train",
                    max_actors=1,
                    population_mode="logged",
                )
                bindings = {camera: match_navsim_traversal(
                    frames, video, scene=scene, camera_name=camera, render_end_tick=70)
                    for camera in CAMERAS}
                binding = bindings["CAM_F0"]
                verify_source_inventory(binding, scene, frames)
                row["verified_cameras"] = list(CAMERAS)
                row["render_end_tick"] = 70
                audit = audit_binding_assets(binding, catalog["nodes"])
                row.update(
                    status="complete_assets" if audit["complete"] else "asset_incomplete",
                    asset_preflight=audit,
                )
            except (ValueError, IndexError, KeyError) as exc:
                row.update(status="source_support_rejected", reason=str(exc))
            rows.append(row)
    rows.sort(
        key=lambda x: (
            x["status"] != "complete_assets",
            len(x.get("asset_preflight", {}).get("missing_tokens", [])),
            x["source_log"],
            x.get("start_frame", -1),
        )
    )
    report = dict(
        kind="navtrain_asset_window_selection",
        window_contract=dict(history_frames=4, future_points=10, render_end_tick=70,
                             minimum_source_frames=15, cameras=list(CAMERAS)),
        rows=rows,
        selected=[
            dict(source_log=x["source_log"], start_frame=x["start_frame"])
            for x in rows
            if x["status"] == "complete_assets"
        ],
        trainable=False,
        scope="source_asset_readiness_not_visual_or_driving_admission",
        sources={
            **{
                str(p): digest(p)
                for p in (catalog_path, metadata_path, checkpoint_path, split_path)
            },
            **source_logs,
        },
    )
    output.write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ("catalog", "metadata", "checkpoint", "log-root", "split", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--trusted-pickle", action="store_true")
    a = p.parse_args()
    if not a.trusted_pickle:
        p.error("Explicit trust of source metadata required")
    report = run(a.catalog, a.metadata, a.checkpoint, a.log_root, a.split, a.output)
    print(
        json.dumps(
            {"windows": len(report["rows"]), "complete_asset_windows": len(report["selected"])}
        )
    )


if __name__ == "__main__":
    main()
