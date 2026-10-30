"""Atomic canonical dataset packages; experiment receipts remain separate."""

import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path

from driversi.agents.backends.native import digest, project_root


def dataset_root():
    return project_root() / "dataset" / "synthetic_navsim"


def _verify(export):
    for name, sha in export["files"].items():
        if digest(name) != sha:
            raise ValueError("Canonical synthetic package or its source changed")


def _relocate(value, source, destination):
    if isinstance(value, str):
        return (
            str(destination) + value[len(str(source)) :]
            if value.startswith(str(source) + "/")
            else value
        )
    if isinstance(value, list):
        return [_relocate(v, source, destination) for v in value]
    if isinstance(value, dict):
        return {
            _relocate(k, source, destination): _relocate(v, source, destination)
            for k, v in value.items()
        }
    return value


def publish_package(identity, producer, alias):
    """Finish a new package on the dataset filesystem before exposing its path.

    producer(staging) must return a non-admitting native export with relative
    per-frame sensor paths. Failed staging is retained; retries get a new path.
    """
    root = dataset_root().resolve()
    alias = Path(alias)
    from driversi.data.datasets.simscale_collection import require_campaign

    require_campaign(root.parent, alias, link_location=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    target = root / "samples" / key
    lock_path = root / ".locks" / (key + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            export = json.loads((target / "export.json").read_text())
            if export.get("package_identity") != identity or export.get("package_id") != key:
                raise ValueError("Canonical package identity changed")
            _verify(export)
        else:
            staging = root / ".staging" / (key + "-" + uuid.uuid4().hex)
            staging.parent.mkdir(parents=True, exist_ok=True)
            export = producer(staging)
            if export.get("trainable") is not False:
                raise ValueError("Storage publication cannot grant training admission")
            from driversi.data.datasets.synthetic_storage import HISTORY_ONLY_LAYOUT, ANCHOR_ONLY_LAYOUT, image_storage_layout
            layout = export.get("storage_layout")
            if layout not in ("logs_sensors_history_future_v2", HISTORY_ONLY_LAYOUT, ANCHOR_ONLY_LAYOUT):
                raise ValueError("Canonical dataset requires a declared partitioned export")
            if layout != image_storage_layout(export.get("image_contract")):
                raise ValueError("History-only storage requires its explicit image contract")
            _verify(export)
            export = _relocate(export, staging, target)
            export.update(package_id=key, package_identity=identity, storage_root=str(target))
            (staging / "export.json").write_text(json.dumps(export, indent=2, allow_nan=False))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, target)
            _verify(export)
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            if alias.resolve(strict=True) != target:
                raise ValueError("Experiment export alias points to another package")
        elif alias.exists():
            raise ValueError("Do not overwrite an existing experiment export")
        else:
            alias.symlink_to(target, target_is_directory=True)
        from driversi.data.datasets.postprocessing import training_only

        if not training_only(root.parent):
            from driversi.data.catalog.synthetic_classification import enqueue
            enqueue(target / "export.json")
        return export


def index_admitted_package(sample_report, feature_report):
    """Index only after native supervision and fresh history features pass."""
    from driversi.data.datasets.native_admission import validate_sample

    sample_path, feature_path = (
        Path(sample_report).resolve(strict=True),
        Path(feature_report).resolve(strict=True),
    )
    sample = validate_sample(sample_path)
    feature = json.loads(feature_path.read_text())
    if feature.get("trainable") is not True or feature.get("parent_vectors_copied") is not False:
        raise ValueError("Fresh admitted generated image features required")
    proof = feature.get("admission", {})
    if Path(proof.get("path", "")).resolve() != sample_path or proof.get("sha256") != digest(
        sample_path
    ):
        raise ValueError("Feature and native sample admission differ")
    export_path = Path(feature["export"]).resolve(strict=True)
    if digest(export_path) != feature["export_sha256"]:
        raise ValueError("Feature export changed")
    root = dataset_root().resolve()
    if not export_path.is_relative_to(root / "samples"):
        raise ValueError("New canonical samples belong in dataset/synthetic_navsim/samples")
    export = json.loads(export_path.read_text())
    _verify(export)
    if export_path.parent.name != export["package_id"]:
        raise ValueError("Canonical package ID mismatch")
    if any(sample["files"].get(image["path"]) != image["sha256"] for image in feature["images"]):
        raise ValueError("Indexed feature pixels differ from admitted native sample")
    from driversi.data.datasets.simscale_collection import require_campaign
    from driversi.rsi.correction.epdms_admission import POLICY, validate_sample as validate_epdms

    collection_config = require_campaign(root.parent, sample_path)
    if collection_config and collection_config.get("trajectory_quality") == POLICY:
        validate_epdms(sample)  # Fail before writing any trainable canonical index.
    if collection_config:
        from driversi.data.datasets.simscale_collection import PDMS_POLICY, validate_pdms
        if collection_config.get("trajectory_admission") == PDMS_POLICY:
            validate_pdms(sample)

    value = dict(
        schema_version=1,
        kind="admitted_synthetic_dataset_reference",
        package_id=export["package_id"],
        export=str(export_path),
        export_sha256=digest(export_path),
        sample_report=str(sample_path),
        sample_report_sha256=digest(sample_path),
        feature_report=str(feature_path),
        feature_report_sha256=digest(feature_path),
        sample_id=sample["sample_id"],
        trainable=True,
    )
    reference_id = hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    value["admission_id"] = reference_id
    path = root / "admitted" / export["package_id"] / (reference_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".locks" / (export["package_id"] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Dataset admission reference is immutable")
        if not path.exists():
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
            os.replace(temporary, path)
    from driversi.data.datasets.simscale_collection import publish

    collection = publish(path, root.parent) if (root.parent / "collection.json").exists() else None
    result = dict(path=str(path), sha256=digest(path), package_id=export["package_id"])
    if collection is not None:
        result["simscale_collection"] = collection
    return result
