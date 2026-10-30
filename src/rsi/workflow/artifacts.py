"""Content-checked, typed stage outputs; paths alone are not evidence."""
import hashlib
import json
from pathlib import Path

from driversi.data.assets.mtgs_assets import file_sha256


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seal(result, root, task_id, stage):
    root = Path(root).resolve()
    files = {}
    for name, raw in result["artifacts"].items():
        path = Path(raw).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("Provider artifact is outside its task stage")
        files[name] = {"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size}
    ref = {"task_id": task_id, "stage_id": stage["id"], "kind": stage["output_kind"],
           "provider": stage["provider"], "provider_version": stage["provider_version"], "files": files}
    if "provider_contract" in stage:
        ref["contract"] = stage["provider_contract"]
    return {**ref, "identity": fingerprint(ref)}


def verify(result, task_id, stage):
    ref = result["summary"].get("artifact_reference")
    if not ref:
        raise ValueError("Unsealed stage dependency; rerun under the current Harness")
    body = {k: v for k, v in ref.items() if k != "identity"}
    if ref["identity"] != fingerprint(body):
        raise ValueError("Artifact reference identity changed")
    expected = (task_id, stage["id"], stage["output_kind"], stage["provider"], stage["provider_version"])
    if tuple(ref[k] for k in ("task_id", "stage_id", "kind", "provider", "provider_version")) != expected:
        raise ValueError("Foreign task, stage, kind or provider artifact")
    if ref.get("contract") != stage.get("provider_contract"):
        raise ValueError("Artifact provider contract changed")
    if set(result["artifacts"]) != set(ref["files"]):
        raise ValueError("Artifact bundle changed")
    for name, item in ref["files"].items():
        path = Path(result["artifacts"][name]).resolve(strict=True)
        if str(path) != item["path"] or path.stat().st_size != item["size"] or file_sha256(path) != item["sha256"]:
            raise ValueError("Stage artifact changed: " + name)
    return ref
