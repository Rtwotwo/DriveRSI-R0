"""Pinned MTGS pilot assets: integrity, bounded extraction and source inspection.

This module does not download, import MTGS/torch, execute checkpoints, or claim
that a reconstruction is bound to the simulator. Pickle inspection requires an
explicit trust decision even after an archive has passed its checksum.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
import tarfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

REPOSITORY = "OpenDriveLab/MTGS"
REVISION = "0b9a9b6543e1444665ee2e36fdb95857447af62a"
BLOCK_ID = "road_block-331220_4690660_331190_4690710"
MANIFEST_VERSION = 1
_RECEIPT = "driversi-extraction.json"
_ASSETS = (
    (f"MTGS_paper_data/{BLOCK_ID}.tar.gz", 4330948694,
     "1d80c5564c33fa63667adc56753cb2aadb1f397ef43ff2dd8f2a7b85a52a5771", "scene"),
    ("MTGS_paper_ckpts/mtgs_ckpt_main_mt.tar", 6175631360,
     "36acb125a73cb3137db12789129cc654defaf007bfb8a361b48eed01c07bec27", "checkpoint"),
    ("MTGS_paper_data/ego_masks.tar.gz", 10075,
     "6b3bb873fce713532a031bae466efad0d5531c6e7a378b919dd5998131d8ef40", "ego_masks"),
    ("MTGS_paper_data/nuplan_log_infos.jsonl", 35935479,
     "fbef3a4dc2ea273ac2cd35da8f3c8077d6bd56de8454142055eb8f12e3c51c30", "log_index"),
)


def default_manifest(*, transport: str = "https://hf-mirror.com") -> dict:
    """Describe immutable official LFS metadata and the chosen download transport."""
    if transport not in {"https://hf-mirror.com", "https://huggingface.co"}:
        raise ValueError("transport must be the official host or hf-mirror.com")
    assets = []
    for repository_path, size, digest, role in _ASSETS:
        name = PurePosixPath(repository_path).name
        archive = name.endswith((".tar", ".tar.gz"))
        folder = name.removesuffix(".gz").removesuffix(".tar")
        assets.append({
            "repository_path": repository_path, "name": name, "role": role,
            "size_bytes": size, "sha256": digest,
            "archive_path": f"archives/{name}", "is_archive": archive,
            "extraction_path": f"extracted/{folder}" if archive else None,
            "official_url": f"https://huggingface.co/datasets/{REPOSITORY}"
                            f"/resolve/{REVISION}/{repository_path}",
            "download_url": f"{transport}/datasets/{REPOSITORY}"
                            f"/resolve/{REVISION}/{repository_path}",
            "metadata_url": f"https://huggingface.co/api/datasets/{REPOSITORY}"
                            f"/tree/{REVISION}/{PurePosixPath(repository_path).parent}",
        })
    return {
        "schema_version": MANIFEST_VERSION, "repository": REPOSITORY,
        "revision": REVISION, "transport": transport, "block_id": BLOCK_ID,
        "metadata_authority": "official_repository_lfs_size_and_sha256",
        "assets": assets, "simulator_ready": False, "trainable": False,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(path: Path, *, expected_size: int, expected_sha256: str) -> dict:
    """Report presence and integrity separately; a partial file is not downloaded."""
    path = Path(path)
    if expected_size < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected size and SHA-256 must be valid")
    present = path.exists()
    result = {"path": str(path), "present": present, "downloaded": False,
              "verified": False, "size_bytes": None, "sha256": None,
              "expected_size_bytes": expected_size, "expected_sha256": expected_sha256,
              "error": "missing"}
    if not present:
        return result
    if path.is_symlink() or not path.is_file():
        result["error"] = "not_regular_file"
        return result
    before = path.stat()
    result["size_bytes"] = before.st_size
    if before.st_size != expected_size:
        result["error"] = "size_mismatch"
        return result
    result["downloaded"] = True
    digest = file_sha256(path)
    after = path.stat()
    result["sha256"] = digest
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        result["error"] = "file_changed_during_verification"
    elif digest != expected_sha256:
        result["error"] = "sha256_mismatch"
    else:
        result.update(verified=True, error=None)
    return result


@dataclass(frozen=True)
class ExtractionLimits:
    """Resource limits apply to the entire archive, including unselected members."""

    max_members: int = 250_000
    max_total_bytes: int = 100 * 1024**3
    max_file_bytes: int = 20 * 1024**3

    def validate(self) -> None:
        if min(self.max_members, self.max_total_bytes, self.max_file_bytes) <= 0:
            raise ValueError("extraction limits must be positive")


def _relative_member(name: str) -> Path:
    """Reject cross-platform unsafe names instead of normalizing them silently."""
    posix = PurePosixPath(name)
    if (not name or "\\" in name or "\x00" in name or posix.is_absolute()
            or PureWindowsPath(name).drive or ".." in posix.parts
            or any(":" in part for part in posix.parts)):
        raise ValueError(f"unsafe archive member: {name!r}")
    parts = tuple(part for part in posix.parts if part != ".")
    if not parts:
        return Path(".")
    if any(part.rstrip(". ") != part or PureWindowsPath(part).is_reserved() for part in parts):
        raise ValueError(f"non-portable archive member: {name!r}")
    return Path(*parts)


def _extracted_status(destination: Path, digest: str) -> dict:
    receipt_path = destination / _RECEIPT
    result = {"extracted": False, "extraction_path": str(destination),
              "extraction_error": "not_extracted", "extracted_files": 0}
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return result
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["archive_sha256"] != digest or receipt["schema_version"] != 1:
            raise ValueError("receipt does not match archive")
        files = receipt["files"]
        if not files:
            raise ValueError("empty extraction")
        for item in files:
            relative = _relative_member(item["path"])
            path = destination / relative
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("extracted file escapes root")
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size_bytes"]:
                raise ValueError(f"extracted file missing or resized: {relative}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["extraction_error"] = str(error)
        return result
    return {"extracted": True, "extraction_path": str(destination),
            "extraction_error": None, "extracted_files": len(files),
            "extracted_bytes": sum(item["size_bytes"] for item in files),
            "extracted_integrity": "receipt_and_sizes_checked",
            "selection": receipt["selection"]}


def safe_extract_tar(
    archive: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    include_prefixes: Sequence[str] | None = None,
    required_component: str | None = None,
    limits: ExtractionLimits = ExtractionLimits(),
) -> dict:
    """Verify then extract regular files into a new, isolated destination.

    All members are checked before writing any file. Links, devices, duplicates,
    traversal, and excessive expansion are rejected, even outside a selection.
    ``required_component`` selects a directory component, not a substring.
    """
    archive, destination = Path(archive), Path(destination)
    limits.validate()
    verification = verify_asset(archive, expected_size=expected_size,
                                expected_sha256=expected_sha256)
    if not verification["verified"]:
        raise ValueError(f"archive integrity failed: {verification['error']}")
    prefixes = None if include_prefixes is None else [
        _relative_member(value).as_posix() for value in include_prefixes]
    if prefixes == []:
        raise ValueError("include_prefixes must select at least one prefix")
    if required_component is not None:
        if len(_relative_member(required_component).parts) != 1:
            raise ValueError("required_component must name one directory component")
    selection = {"include_prefixes": prefixes, "required_component": required_component}
    existing = _extracted_status(destination, expected_sha256)
    if existing["extracted"]:
        if existing["selection"] != selection:
            raise ValueError("existing extraction has a different selection")
        return {**verification, **existing}
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing extraction: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("extraction parent must not be a symlink")
    parent = destination.parent.resolve()
    staging = parent / f".{destination.name}.extract-{uuid.uuid4().hex}"
    records = []
    archive_stat = archive.stat()
    with tarfile.open(archive, "r:*") as bundle:
        members = []
        seen: set[str] = set()
        total = 0
        count = 0
        for member in bundle:
            count += 1
            relative = _relative_member(member.name)
            if count > limits.max_members:
                raise ValueError("archive member count exceeds limit")
            if not (member.isfile() or member.isdir()) or member.issparse():
                raise ValueError(f"unsupported archive member type: {member.name}")
            if relative == Path("."):
                if not member.isdir():
                    raise ValueError("archive root must be a directory")
                continue
            key = relative.as_posix().casefold()
            if key == _RECEIPT or key in seen:
                raise ValueError(f"duplicate or reserved archive member: {member.name}")
            seen.add(key)
            if member.size < 0 or member.size > limits.max_file_bytes:
                raise ValueError("archive file size exceeds limit")
            if member.isfile():
                total += member.size
            if total > limits.max_total_bytes:
                raise ValueError("archive expanded size exceeds limit")
            selected = (prefixes is None or any(
                relative.as_posix() == prefix or relative.as_posix().startswith(prefix + "/")
                for prefix in prefixes))
            selected &= required_component is None or required_component in relative.parts
            if member.isfile() and selected:
                members.append((member, relative))
        if not members:
            raise ValueError("archive selection contains no regular files")
        staging.mkdir()
        try:
            for member, relative in members:
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                digest = hashlib.sha256()
                written = 0
                with source, target.open("xb") as stream:
                    while block := source.read(4 * 1024 * 1024):
                        written += len(block)
                        if written > member.size:
                            raise ValueError("archive member exceeds declared size")
                        digest.update(block)
                        stream.write(block)
                if written != member.size:
                    raise ValueError("truncated archive member")
                records.append({"path": relative.as_posix(), "size_bytes": written,
                                "sha256": digest.hexdigest()})
            receipt = {"schema_version": 1, "archive_sha256": expected_sha256,
                       "archive_size_bytes": expected_size, "selection": selection,
                       "archive_members": count, "archive_expanded_bytes": total,
                       "files": records}
            after = archive.stat()
            if (after.st_size, after.st_mtime_ns) != (
                    archive_stat.st_size, archive_stat.st_mtime_ns):
                raise ValueError("archive changed during extraction")
            (staging / _RECEIPT).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            staging.rename(destination)
        finally:
            # Only this invocation's unique staging directory may be removed.
            if staging.exists() and staging.resolve().parent == parent:
                shutil.rmtree(staging)
    return {**verification, **_extracted_status(destination, expected_sha256)}


def inspect_assets(asset_root: Path, *, verify: bool = True) -> dict:
    """Audit downloaded/verified/extracted states independently, without unpickling."""
    root = Path(asset_root).resolve()
    manifest = default_manifest()
    result = {**manifest, "asset_root": str(root), "assets": []}
    for asset in manifest["assets"]:
        path = root / asset["archive_path"]
        if verify:
            status = verify_asset(path, expected_size=asset["size_bytes"],
                                  expected_sha256=asset["sha256"])
        else:
            regular = path.is_file() and not path.is_symlink()
            status = {"path": str(path), "present": path.exists(),
                      "downloaded": regular and path.stat().st_size == asset["size_bytes"],
                      "verified": False, "verification_skipped": True,
                      "size_bytes": path.stat().st_size if regular else None}
        extraction = (_extracted_status(root / asset["extraction_path"], asset["sha256"])
                      if asset["is_archive"] else {"extracted": None})
        result["assets"].append({**asset, "status": {**status, **extraction}})
    result["all_downloaded"] = all(a["status"]["downloaded"] for a in result["assets"])
    result["all_verified"] = all(a["status"]["verified"] for a in result["assets"])
    return result


def extract_assets(
    asset_root: Path, *, names: Sequence[str] | None = None,
    checkpoint_prefixes: Sequence[str] | None = None,
    limits: ExtractionLimits = ExtractionLimits(),
) -> dict:
    """Extract chosen pilot archives, retaining the original verified archives.

    Checkpoints default to the selected Boston block directory. Explicit prefixes
    are useful when an upstream archive uses a different directory layout.
    """
    root = Path(asset_root).resolve()
    assets = default_manifest()["assets"]
    known = {asset["name"] for asset in assets if asset["is_archive"]}
    selected = known if names is None else set(names)
    if not selected or not selected <= known:
        raise ValueError(f"names must select known archives: {sorted(known)}")
    outputs = []
    for asset in assets:
        if asset["name"] not in selected:
            continue
        is_checkpoint = asset["role"] == "checkpoint"
        destination = root / asset["extraction_path"]
        if not destination.resolve().is_relative_to(root):
            raise ValueError("extraction path escapes asset root")
        status = safe_extract_tar(
            root / asset["archive_path"], destination,
            expected_size=asset["size_bytes"], expected_sha256=asset["sha256"],
            include_prefixes=checkpoint_prefixes if is_checkpoint else None,
            required_component=(BLOCK_ID if is_checkpoint and checkpoint_prefixes is None else None),
            limits=limits)
        outputs.append({"name": asset["name"], "status": status})
    return {"schema_version": 1, "asset_root": str(root), "assets": outputs,
            "simulator_ready": False, "trainable": False}


def inspect_block_metadata(path: Path, *, trusted: bool = False,
                           max_file_bytes: int = 512 * 1024**2) -> dict:
    """Inspect the documented MTGS ``video_scene_dict.pkl`` source inventory.

    Source annotations are not reconstructed instances. This function deliberately
    leaves renderer/world inventories unknown instead of assuming they match.
    """
    if not trusted:
        raise ValueError("pickle inspection requires trusted=True for this official asset")
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or max_file_bytes <= 0
            or path.stat().st_size > max_file_bytes):
        raise ValueError("metadata must be a bounded regular file")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not data:
        raise ValueError("MTGS metadata must contain a non-empty video mapping")
    observations: Counter = Counter()
    tracks: dict[str, set[str]] = defaultdict(set)
    videos = []
    for token, video in data.items():
        if not isinstance(token, str) or not isinstance(video, dict):
            raise ValueError("invalid MTGS video entry")
        frames = video.get("frame_infos")
        if not isinstance(frames, list) or not frames:
            raise ValueError("each MTGS video requires frame_infos")
        timestamps = []
        camera_names: set[str] = set()
        camera_frames = 0
        for frame in frames:
            timestamp = frame.get("timestamp")
            if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
                raise ValueError("MTGS frame timestamps must be integer microseconds")
            timestamps.append(int(timestamp))
            names, tokens = frame.get("gt_names", []), frame.get("track_tokens", [])
            if len(names) != len(tokens):
                raise ValueError("source object names and track tokens must align")
            for name, track in zip(names, tokens):
                if not isinstance(name, str) or not isinstance(track, str):
                    raise ValueError("source object classes and IDs must be strings")
                observations[name] += 1
                tracks[name].add(track)
            cams = frame.get("cams", {})
            if not isinstance(cams, dict):
                raise ValueError("frame cams must be a mapping")
            camera_names.update(cams)
            camera_frames += len(cams)
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("video timestamps must increase strictly")
        videos.append({
            "video_token": token, "log_name": video.get("log_name"),
            "map_location": video.get("map_location"), "frames": len(frames),
            "usable_frames": sum(not frame.get("skipped", False) for frame in frames),
            "first_timestamp_us": timestamps[0], "last_timestamp_us": timestamps[-1],
            "duration_seconds": (timestamps[-1] - timestamps[0]) / 1_000_000,
            "camera_names": sorted(camera_names), "camera_frame_records": camera_frames,
        })
    return {"schema_version": 1, "source_path": str(path.resolve()),
            "source_sha256": file_sha256(path), "videos": videos,
            "source_object_observations": dict(sorted(observations.items())),
            "source_unique_tracks_by_class": {k: len(v) for k, v in sorted(tracks.items())},
            "renderer_object_inventory": None, "world_object_inventory": None,
            "simulator_ready": False, "trainable": False}
