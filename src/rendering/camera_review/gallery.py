"""Build a lazy, camera-bound review case inside the existing whitebox gallery.

Reuses the verified media/video bundle, without copying image assets or changing
calibration, object geometry, timestamps, or the original downloadable records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

STATIC = (
    "gallery.html",
    "gallery.css",
    "gallery.js",
    "wireframe.js",
    "projection.js",
    "semantic.js",
    "structure.js",
    "structure-controls.js",
    "structure-geometry.js",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}


def partition(data, chunk_size=8):
    """Losslessly split camera arrays, leaving all state/trajectory fields intact."""
    if not isinstance(chunk_size, int) or not 1 <= chunk_size <= 32:
        raise ValueError("Invalid chunk size")
    frames = data["frames"]
    ids = set(data["camera_owners"])
    if not frames or not ids or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", x) for x in ids):
        raise ValueError("Missing frames or unsafe camera identity")
    scene = {
        k: data[k]
        for k in (
            "session_id",
            "scenario_id",
            "source_log",
            "registration_mode",
            "map_layers",
            "source_frames",
        )
    }
    scene.update(
        schema_version=1,
        kind="lazy_camera_whitebox",
        trainable=False,
        chunk_size=chunk_size,
        cameras={},
        frames=[],
    )
    chunks = {}
    for i, frame in enumerate(frames):
        if i and (
            frame["tick"] <= frames[i - 1]["tick"]
            or frame["time_seconds"] <= frames[i - 1]["time_seconds"]
            or frame["generation"] != frames[i - 1]["generation"]
        ):
            raise ValueError("Invalid frame sequence")
        if len(frame["cameras"]) != len(ids) or {c["camera_id"] for c in frame["cameras"]} != ids:
            raise ValueError("Camera coverage differs by frame")
        if not 0 <= frame["source_index"] < len(data["source_frames"]):
            raise ValueError("Missing causal source frame")
        if (
            data["source_frames"][frame["source_index"]]["timestamp_us"]
            > frame["simulation_timestamp_us"]
        ):
            raise ValueError("Source frame comes from the future")
        scene["frames"].append({k: v for k, v in frame.items() if k != "cameras"})
    for cid in sorted(ids):
        cameras = [next(c for c in f["cameras"] if c["camera_id"] == cid) for f in frames]
        owner = data["camera_owners"][cid]
        for frame, cam in zip(frames, cameras, strict=True):
            if cam["owner_actor_id"] != owner or cam["virtual"] != cameras[0]["virtual"]:
                raise ValueError("Camera identity changed")
            if owner not in {a["id"] for a in frame["actors"] if a["present"]}:
                raise ValueError("Camera owner missing from world")
        scene["cameras"][cid] = {
            "owner_actor_id": owner,
            "virtual": cameras[0]["virtual"],
            "chunks": [],
        }
        for start in range(0, len(frames), chunk_size):
            name = f"chunks/{cid}-{start:04}.json"
            chunks[name] = {
                "camera_id": cid,
                "start": start,
                "ticks": [f["tick"] for f in frames[start : start + chunk_size]],
                "frames": cameras[start : start + chunk_size],
            }
            scene["cameras"][cid]["chunks"].append({"path": name})
    return scene, chunks


def export(bundle, destination):
    from .publication import bundle_files

    bundle, destination = Path(bundle).resolve(), Path(destination)
    bundle_files(bundle)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Use a new gallery case or explicitly stage a verified replacement")
    data = json.loads((bundle / "data.json").read_text())
    scene, chunks = partition(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".camera-gallery-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "case"
        stage.mkdir()
        scene["map"] = dump(stage / "map.json", {"features": data["map_features"]})
        for cid, camera in scene["cameras"].items():
            for entry in camera["chunks"]:
                path = entry["path"]
                info = dump(stage / path, chunks[path])
                entry.update(sha256=info["sha256"], bytes=info["bytes"])
        dump(stage / "scene.json", scene)
        for name in STATIC:
            shutil.copyfile(
                Path(__file__).with_name(name),
                stage / ("index.html" if name == "gallery.html" else name),
            )
        from .transfer import optimize

        optimize(stage)
        info = {
            "schema_version": 1,
            "kind": "camera_gallery_case",
            "title": "NAVSIM Boston · 全要素结构化 / 自由视角 / 相机对照",
            "entrypoint": "index.html",
            "camera_count": len(scene["cameras"]),
            "duration_seconds": data["frames"][-1]["time_seconds"]
            - data["frames"][0]["time_seconds"],
            "frame_count": len(data["frames"]),
            "source_frame_count": len(data["source_frames"]),
            "source_data_sha256": digest(bundle / "data.json"),
            "asset_bundle": "../../camera-review/",
            "video_count": 0,
        }
        if (bundle / "videos.json").exists():
            info["video_count"] = len(json.loads((bundle / "videos.json").read_text())["videos"])
        info["files"] = {
            p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()
        }
        dump(stage / "gallery-entry.json", info)
        stage.rename(destination)
    return info


def verify(destination, bundle):
    destination, bundle = Path(destination), Path(bundle)
    entry = json.loads((destination / "gallery-entry.json").read_text())
    if entry["source_data_sha256"] != digest(bundle / "data.json"):
        raise ValueError("Gallery source data changed")
    for name, checksum in entry["files"].items():
        p = destination / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(destination.resolve())
            or digest(p) != checksum
        ):
            raise ValueError("Gallery content changed")
    source = json.loads((bundle / "data.json").read_text())
    scene = json.loads((destination / "scene.json").read_text())
    for cid, camera in scene["cameras"].items():
        reconstructed = []
        for ref in camera["chunks"]:
            chunk = json.loads((destination / ref["path"]).read_text())
            if digest(destination / ref["path"]) != ref["sha256"]:
                raise ValueError("Chunk hash differs")
            reconstructed.extend(chunk["frames"])
        expected = [
            next(c for c in f["cameras"] if c["camera_id"] == cid) for f in source["frames"]
        ]
        if reconstructed != expected:
            raise ValueError("Camera data differs from original")
    if scene["frames"] != [
        {k: v for k, v in f.items() if k != "cameras"} for f in source["frames"]
    ]:
        raise ValueError("World states changed")
    if scene["source_frames"] != source["source_frames"]:
        raise ValueError("Source logs changed")
    if json.loads((destination / "map.json").read_text())["features"] != source["map_features"]:
        raise ValueError("Map changed")
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export(args.bundle, args.output)
    print(json.dumps(verify(args.output, args.bundle), ensure_ascii=False))


if __name__ == "__main__":
    main()
