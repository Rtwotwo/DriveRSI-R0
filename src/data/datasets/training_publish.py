"""Publish native training samples without requiring retrieval annotations."""
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.data.datasets import synthetic_publish

KIND = "admitted_synthetic_training_reference"
DEFERRED = dict(features="deferred", scene_caption="deferred", classification="deferred")


def validate_binding(sample, export_path, root):
    """Native validation alone does not bind an arbitrary export to that sample."""
    export_path, root = Path(export_path).resolve(strict=True), Path(root).resolve()
    export = json.loads(export_path.read_text())
    package = export.get("package_id", "")
    if not package or export_path != root / "samples" / package / "export.json":
        raise ValueError("Training reference requires its canonical package")
    synthetic_publish._verify(export)
    row = sample["row"]
    if (sample["sample_id"] != export["token"] or row["token"] != export["token"]
            or row["anchor_index"] != export["anchor_index"]
            or set(row["image_files"]) != set(export["image_files"])):
        raise ValueError("Training sample and export identity differ")
    for key in ("log_path", "sensor_root"):
        if Path(row[key]).resolve(strict=True) != Path(export[key]).resolve(strict=True):
            raise ValueError("Training sample and export paths differ")
    # Check all stored frames, not just the current anchor's policy inputs.
    if not export["files"] or any(sample["files"].get(p) != h for p, h in export["files"].items()):
        raise ValueError("Export files are not sealed by the native training admission")
    return export


def validate_reference(ref, root):
    from driversi.data.datasets.native_admission import validate_sample

    if (ref.get("kind") != KIND or ref.get("schema_version") != 2
            or ref.get("postprocessing") != DEFERRED or ref.get("trainable") is not True
            or ref.get("indexable") is not False or "feature_report" in ref):
        raise ValueError("Explicit training-only reference required")
    value = dict(ref)
    seal = value.pop("admission_id", None)
    if seal != hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest():
        raise ValueError("Training reference identity changed")
    for key in ("export", "sample_report"):
        if digest(ref[key]) != ref[key + "_sha256"]:
            raise ValueError("Training reference source changed")
    sample = validate_sample(ref["sample_report"])
    export = validate_binding(sample, ref["export"], root)
    if ref["sample_id"] != sample["sample_id"] or ref["package_id"] != export["package_id"]:
        raise ValueError("Training reference identity differs from its sample")
    return sample


def index_training_package(sample_report, export_path):
    from driversi.data.datasets.native_admission import validate_sample
    from driversi.data.datasets.simscale_collection import require_campaign, publish, PDMS_POLICY, validate_pdms
    from driversi.rsi.correction.epdms_admission import POLICY, validate_sample as validate_epdms
    from driversi.data.datasets.postprocessing import training_only

    root = synthetic_publish.dataset_root().resolve()
    if not training_only(root.parent):
        raise ValueError("Activate the campaign's training-only processing policy first")
    sample_path, export_path = Path(sample_report).resolve(strict=True), Path(export_path).resolve(strict=True)
    sample = validate_sample(sample_path)
    export = validate_binding(sample, export_path, root)
    config = require_campaign(root.parent, sample_path)
    if config.get("trajectory_quality") == POLICY:
        validate_epdms(sample)
    if config.get("trajectory_admission") == PDMS_POLICY:
        validate_pdms(sample)
    value = dict(schema_version=2, kind=KIND, package_id=export["package_id"],
                 export=str(export_path), export_sha256=digest(export_path),
                 sample_report=str(sample_path), sample_report_sha256=digest(sample_path),
                 sample_id=sample["sample_id"], trainable=True, indexable=False,
                 postprocessing=DEFERRED)
    value["admission_id"] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = root / "admitted" / export["package_id"] / (value["admission_id"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".locks" / (export["package_id"] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("Training admission reference is immutable")
        else:
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
            os.replace(temporary, path)
    result = dict(path=str(path), sha256=digest(path), package_id=export["package_id"])
    result["simscale_collection"] = publish(path, root.parent)
    return result
