"""Read released reconstruction identities without executing upstream YAML tags.

Asset availability is a scheduling prerequisite, never a visual certificate or
model-failure observation. Source-log isolation is retained across all tokens.
"""

import re
import tarfile
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.training.sampling.environment import source_log_group

_FIELDS = ("central_log", "city", "road_block_name", "split")


def read_config_index(archive, train_logs, protected_logs=()):
    train = {source_log_group(x) for x in train_logs}
    protected = {source_log_group(x) for x in protected_logs}
    if not train or train & protected:
        raise ValueError("Disjoint explicit training and protected source groups required")
    rows, seen = [], set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if not member.isfile() or not member.name.endswith(".yaml"):
                continue
            if member.size > 65536:
                raise ValueError("Unexpectedly large reconstruction config")
            text = stream.extractfile(member).read().decode("utf-8")
            fields = {}
            for key in _FIELDS:
                matches = re.findall(r"^" + key + r": ([a-zA-Z0-9_.-]+)\s*$", text, re.M)
                if len(matches) != 1:
                    raise ValueError("Missing/ambiguous reconstruction identity: " + key)
                fields[key] = matches[0]
            identity = fields["road_block_name"]
            log = fields["central_log"]
            if identity in seen or not identity.startswith(log + "-"):
                raise ValueError("Duplicate or conflicting reconstruction identity")
            seen.add(identity)
            group = source_log_group(log)
            eligible = group in train and group not in protected and fields["split"] == "trainval"
            rows.append(
                {
                    "asset_id": identity,
                    "source_log": log,
                    "source_group": group,
                    "city": fields["city"],
                    "config_member": member.name,
                    "source_eligible": eligible,
                    "source_status": "navtrain" if eligible else "excluded_source",
                }
            )
    return {
        "schema_version": 1,
        "kind": "worldengine_reconstruction_index",
        "archive": str(Path(archive).resolve()),
        "archive_sha256": digest(archive),
        "rows": sorted(rows, key=lambda row: row["asset_id"]),
        "scope": "released_config_identity_not_local_asset_or_quality",
        "trainable": False,
    }


def bundle_files(root, asset_id):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", asset_id):
        raise ValueError("Unsafe reconstruction identity")
    base = Path(root).resolve() / asset_id
    files = {
        "checkpoint": base / "background" / (asset_id + ".ckpt"),
        "metadata": base / "video_scene_dict.pkl",
        "height_map": base / "road_height_map" / "road_height_map.npy",
        "height_transform": base / "road_height_map" / "sim2.json",
    }
    result = {}
    for key, path in files.items():
        if path.is_file() and not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("Asset file escapes its managed root")
        result[key] = {
            "path": str(path),
            "present": path.is_file(),
            "sha256": digest(path) if path.is_file() else None,
        }
    return result


class SourceBinding:
    """Canonical sources in one released checkpoint coordinate system."""

    def __init__(self, asset_id, videos, checkpoint_origin):
        import numpy as np

        expected = asset_id + "-0"
        if set(videos) != {expected}:
            raise ValueError("Checkpoint and metadata reconstruction identities differ")
        video = videos[expected]
        if not asset_id.startswith(video["log_name"] + "-"):
            raise ValueError("Checkpoint and metadata source logs differ")
        origin = np.asarray(video["global2world_translation"], dtype=float)
        other = np.asarray(checkpoint_origin, dtype=float)
        if (
            origin.shape != (3,)
            or other.shape != (3,)
            or not np.isfinite(origin).all()
            or not np.allclose(origin, other, atol=1e-8, rtol=0)
        ):
            raise ValueError("Checkpoint and metadata reconstruction origins differ")
        self.asset_id, self.source_log = asset_id, video["log_name"]
        self.frames = {f["token"]: f for f in video["frame_infos"]}
        if len(self.frames) != len(video["frame_infos"]):
            raise ValueError("Ambiguous metadata source token")

    def resolve(self, source):
        import numpy as np

        token = source if isinstance(source, str) else source["token"]
        if token not in self.frames:
            raise ValueError("Source token does not belong to this reconstruction")
        canonical = self.frames[token]
        if canonical.get("skipped"):
            raise ValueError("Skipped source frame cannot support rendering")
        if not isinstance(source, str):
            if int(source["timestamp"]) != int(canonical["timestamp"]):
                raise ValueError("Source timestamp does not match reconstruction")
            for key in ("ego2global", "ego2global_original"):
                if not np.array_equal(np.asarray(source[key]), np.asarray(canonical[key])):
                    raise ValueError("Source pose does not match reconstruction")
        # Camera fields always come from the bound metadata, never a caller's copy.
        return canonical
