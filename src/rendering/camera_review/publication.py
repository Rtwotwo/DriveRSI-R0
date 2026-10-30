"""Validate the self-contained camera review before public static publication."""

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path


def bundle_files(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Review bundle must not be a symlink")
    data = json.loads((root / "data.json").read_text())
    if data.get("kind") != "camera_bound_scene_inspection" or data.get("trainable") is not False:
        raise ValueError("Unsupported review contract")

    def private(value):
        if isinstance(value, str) and (
            value.startswith(("/data/", "/root/", "/opt/")) or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            raise ValueError("Private server path in public bundle")
        if isinstance(value, dict):
            for child in value.values():
                private(child)
        elif isinstance(value, list):
            for child in value:
                private(child)

    private(data)
    files = {"index.html", "viewer.js", "projection.js", "semantic.js", "downloads.js", "data.json"}
    used = set()
    identities = set()
    for frame in data["frames"]:
        identity = (frame["generation"], frame["tick"])
        if identity in identities or not frame["cameras"]:
            raise ValueError("Unbound or duplicate demo frame")
        identities.add(identity)
        if {c["camera_id"] for c in frame["cameras"]} != set(data["camera_owners"]):
            raise ValueError("Incomplete per-frame camera coverage")
        actors = {a["id"] for a in frame["actors"] if a["present"]}
        for camera in frame["cameras"]:
            if (
                camera["owner_actor_id"] not in actors
                or data["camera_owners"].get(camera["camera_id"]) != camera["owner_actor_id"]
            ):
                raise ValueError("Camera/actor binding mismatch")
            if not camera.get("image") or (camera["virtual"] and camera.get("source")):
                raise ValueError("Invalid camera media semantics")
            source = camera.get("source_whitebox")
            if source:
                photo = camera.get("source_photo", {})
                if (
                    camera["virtual"]
                    or not camera.get("source")
                    or source["camera_id"] != camera["camera_id"]
                    or source["owner_actor_id"] != "ego"
                    or source["camera_token"] != photo.get("camera_token")
                    or source["source_timestamp_us"] != photo.get("frame_timestamp_us")
                ):
                    raise ValueError("Original photo and whitebox mismatch")
            for key in ("image", "source", "valid"):
                if camera.get(key):
                    used.add(camera[key])
    if used != set(data["media_sha256"]):
        raise ValueError("Media ledger does not match camera references")
    for name, checksum in data["media_sha256"].items():
        if not re.fullmatch(r"media/[0-9a-f]{64}\.png", name) or name != f"media/{checksum}.png":
            raise ValueError("Unsafe media name")
        files.add(name)
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Media changed since binding")
    if (root / "videos.json").exists():
        video = json.loads((root / "videos.json").read_text())
        private(video)
        if (
            video.get("kind") != "camera_whitebox_videos"
            or video.get("trainable") is not False
            or video.get("data_sha256")
            != hashlib.sha256((root / "data.json").read_bytes()).hexdigest()
            or video.get("session_id") != data.get("session_id")
        ):
            raise ValueError("Video provenance differs from camera scene")
        files.add("videos.json")
        for name, checksum in video.get("renderer_sha256", {}).items():
            if (
                name not in ("semantic.js", "projection.js")
                or hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum
            ):
                raise ValueError("Video renderer differs from published whitebox")
        if set(video.get("renderer_sha256", {})) != {"semantic.js", "projection.js"}:
            raise ValueError("Missing video renderer provenance")
        entries = video.get("videos", [])
        expected = set()
        for camera in data["frames"][0]["cameras"]:
            variants = ["branch"] if camera["virtual"] else ["source", "branch", "comparison"]
            expected.update((camera["camera_id"], v) for v in variants)
        if (
            len(entries) != len(expected)
            or {(e["camera_id"], e["variant"]) for e in entries} != expected
        ):
            raise ValueError("Incomplete or duplicate video camera/variant coverage")
        for entry in entries:
            if entry["owner_actor_id"] != data["camera_owners"][entry["camera_id"]]:
                raise ValueError("Wrong video camera owner")
            if not math.isfinite(entry["duration_seconds"]) or entry["duration_seconds"] <= 0:
                raise ValueError("Invalid video duration")
            if (
                entry["encoded_frame_count"] != video["frame_count"]
                or abs(entry["duration_seconds"] - video["frame_count"] / video["output_fps"])
                > 1e-7
            ):
                raise ValueError("Video timebase mismatch")
            if entry["path"] != f"videos/{entry['camera_id']}--{entry['variant']}.mp4":
                raise ValueError("Video filename/binding mismatch")
        for entry in entries + [video["archive"]]:
            name = entry["path"]
            if not re.fullmatch(r"videos/[A-Za-z0-9_-]+\.(mp4|zip)", name):
                raise ValueError("Unsafe video path")
            file = root / name
            if (
                file.is_symlink()
                or file.parent.is_symlink()
                or file.stat().st_size != entry["bytes"]
                or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("Video changed since export")
            files.add(name)
        details = root / "videos/manifest.json"
        base = json.loads(details.read_text())
        if base != {k: v for k, v in video.items() if k != "archive"}:
            raise ValueError("Archive manifest differs from downloadable clips")
        files.add("videos/manifest.json")
        with zipfile.ZipFile(root / video["archive"]["path"]) as z:
            names = [Path(e["path"]).name for e in entries] + ["manifest.json"]
            if sorted(z.namelist()) != sorted(names):
                raise ValueError("Unexpected archive contents")
            for entry in entries:
                if hashlib.sha256(z.read(Path(entry["path"]).name)).hexdigest() != entry["sha256"]:
                    raise ValueError("Archive clip differs from download")
            if z.read("manifest.json") != details.read_bytes():
                raise ValueError("Archive provenance differs")
    result = []
    for name in sorted(files):
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size >= 100_000_000:
            raise ValueError("Missing, linked or oversized bundle file")
        result.append((name, path))
    if sum(p.stat().st_size for _, p in result) >= 500_000_000:
        raise ValueError("Review bundle is too large")
    return result
