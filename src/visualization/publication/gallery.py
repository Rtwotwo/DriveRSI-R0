"""Validate and atomically publish the read-only presentation gallery below viz/."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

SCOPE = (
    "Recorded multi-agent rollout, candidate correction and rendering; "
    "not evolving learned self-play."
)


def checked_path(root, name, *, required=True, hidden=False):
    """Reject ambiguous URL paths and symlinks, including symlinked parents."""
    if (not isinstance(name, str) or not name or name.startswith("/")
            or any(c in name for c in "\\:?#%")
            or any(ord(c) < 32 for c in name)):
        raise ValueError("Invalid gallery path")
    parts = name.split("/")
    if any(not p or p in (".", "..") or (p.startswith(".") and not hidden) for p in parts):
        raise ValueError("Unsafe gallery path")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink is not a publication path: {name}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path escapes gallery")
    if path.exists() and not path.is_file():
        raise ValueError(f"Expected a regular file: {name}")
    if required and not path.is_file():
        raise ValueError(f"Missing gallery file: {name}")
    return path


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys,
                      parse_constant=reject_constant)


def case_identity(card):
    if not isinstance(card, dict):
        raise ValueError("Case must be an object")
    cid = card.get("id")
    if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cid):
        raise ValueError("Invalid case ID")
    for field in ("source_log", "scenario_id"):
        if not isinstance(card.get(field), str) or not card[field]:
            raise ValueError(f"Missing case origin: {field}")
    return cid


def same_origin(left, right):
    if any(left.get(key) != right.get(key) for key in ("source_log", "scenario_id")):
        raise ValueError(f"Case origin conflict: {left.get('id')}")


def load_catalog(out):
    catalog = read_json(checked_path(out, "data/cases.json"))
    if (not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int
            or catalog["schema_version"] != 1 or not isinstance(catalog.get("cases"), list)):
        raise ValueError("Unsupported or malformed catalog")
    seen = set()
    for card in catalog["cases"]:
        cid = case_identity(card)
        if cid in seen:
            raise ValueError("Duplicate case identity")
        seen.add(cid)
    optional = checked_path(out, "data/multiview-card.json", required=False)
    if optional.exists():
        update = read_json(optional)
        cid = case_identity(update)
        cards = []
        for card in catalog["cases"]:
            if card["id"] == cid:
                same_origin(card, update)
                cards.append({**card, **update})
            else:
                cards.append(card)
        if cid not in seen:
            cards.append(update)
        catalog = {**catalog, "cases": cards}
    return catalog


@contextmanager
def publication_lock(out):
    path = checked_path(out, ".export.lock", required=False, hidden=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Publication lock must be a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def atomic_write(root, name, raw):
    """A failed write/replace leaves the prior complete file, never a partial one."""
    path = checked_path(root, name, required=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + "-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        checked_path(root, name, required=False)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_case(out, card, assets):
    cid = case_identity(card)
    name = card.get("data")
    if not isinstance(name, str) or not name.startswith("data/") or not name.endswith(".json"):
        raise ValueError("Case data must name a data/*.json file")
    data_path = checked_path(out, name)
    data = read_json(data_path)
    if not isinstance(data, dict) or data.get("id") != cid:
        raise ValueError("Case data identity mismatch")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or case_identity(metadata) != cid:
        raise ValueError("Case metadata identity mismatch")
    same_origin(card, metadata)
    for entry in (card, metadata):
        if entry.get("trainable") is not False or entry.get("learned_self_play") is not False:
            raise ValueError("This diagnostic gallery requires explicit false qualification flags")
    frames = data.get("frames")
    if (not isinstance(frames, list) or not frames
            or type(card.get("frame_count")) is not int or len(frames) != card["frame_count"]):
        raise ValueError("Invalid frame list/count")
    previous_tick = previous_time = -math.inf
    camera_frames = 0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Frame must be an object")
        tick, time = frame.get("tick"), frame.get("time_seconds")
        if (type(tick) is not int or tick < 0 or not finite_number(time) or time < 0
                or tick <= previous_tick or time <= previous_time):
            raise ValueError("Frame timestamps/ticks must be finite and strictly increasing")
        previous_tick, previous_time = tick, time
        actors = frame.get("actors")
        cameras = frame.get("cameras", [])
        for entries, label in ((actors, "actors"), (cameras, "cameras")):
            if not isinstance(entries, list):
                raise ValueError(f"Invalid {label} list")
            ids = []
            for entry in entries:
                if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                        or not entry["id"]):
                    raise ValueError(f"Invalid {label} identity")
                ids.append(entry["id"])
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {label} identity")
        for actor in actors:
            if not all(finite_number(actor.get(key)) for key in ("x", "y", "yaw", "length", "width")):
                raise ValueError("Invalid actor geometry")
        for camera in cameras:
            camera_frames += 1
            if camera.get("virtual") and camera.get("source"):
                raise ValueError("Virtual camera cannot borrow a real source image")
            for field in ("image", "source", "comparison", "valid_mask"):
                value = camera.get(field)
                if value:
                    if not isinstance(value, str) or not value.startswith("media/"):
                        raise ValueError("Camera must reference media/")
                    checked_path(out, value)
                    assets.add(value)
            if not camera.get("image"):
                raise ValueError("Camera requires a rendered image")
    if card.get("thumbnail"):
        thumbnail = card["thumbnail"]
        if not isinstance(thumbnail, str) or not thumbnail.startswith("media/"):
            raise ValueError("Thumbnail must reference media/")
        checked_path(out, thumbnail)
    return {
        "id": cid, "frames": len(frames), "rendered_images": camera_frames,
        "seconds": frames[-1]["time_seconds"] - frames[0]["time_seconds"],
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }


def publish(root: Path):
    root = root.absolute()
    if root.is_symlink():
        raise ValueError("Project root may not be a symlink")
    root = root.resolve()
    out = root / "viz"
    if out.is_symlink() or not out.is_dir():
        raise ValueError("Gallery must be a real directory")
    with publication_lock(out):
        catalog = load_catalog(out)
        cards = catalog["cases"]
        assets = set()
        data_paths = [card.get("data") for card in cards]
        if any(not isinstance(path, str) for path in data_paths):
            raise ValueError("Invalid case data path")
        if len(set(data_paths)) != len(data_paths):
            raise ValueError("Case IDs may not share a data file")
        results = [validate_case(out, card, assets) for card in cards]
        template = checked_path(root, "src/rendering/inspection/showcase.html").read_bytes()
        catalog.setdefault("presentation_scope", SCOPE)
        receipt = {
            "schema_version": 1, "case_count": len(cards),
            "distinct_source_logs": len({card["source_log"] for card in cards}),
            "unique_media_files": len(assets),
            "media_logical_bytes": sum(checked_path(out, name).stat().st_size for name in assets),
            "cases": results,
            "checks": {
                "strict_timestamp_order": True, "actor_identity_unique": True,
                "all_referenced_media_exist": True, "virtual_cameras_have_no_source_image": True,
                "gallery_paths_contained": True, "symlink_paths_rejected": True,
            },
            "template_sha256": hashlib.sha256(template).hexdigest(), "trainable": False,
        }
        publications = [
            ("data/cases.json", json_bytes(catalog)),
            ("index.html", template),
            ("data/publication-checks.json", json_bytes(receipt)),
        ]
        # Validate every destination before changing any public file.
        for name, _ in publications:
            checked_path(out, name, required=False)
        for name, raw in publications:
            atomic_write(out, name, raw)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    publish(parser.parse_args().root)


if __name__ == "__main__":
    main()
