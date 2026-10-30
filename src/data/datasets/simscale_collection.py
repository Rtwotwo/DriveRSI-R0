"""Admitted OpenScene views over the current generated dataset.

Canonical pixels and native receipts stay immutable. Only camera path strings
change in the small compatibility PKL; targets, geometry and times do not.
"""
import argparse
import copy
import fcntl
import json
import os
import pickle
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.correction.epdms_admission import POLICY, validate_sample as validate_epdms

PDMS_POLICY = dict(profile="pdms_gt95_v1", metric="PDMS", operator=">", threshold=0.95,
                   score_unit="0_to_1", scope="each_generated_trajectory",
                   ec="optional_diagnostic", protocol="navsim-v1-native-math-executed-counterfactual-v1")


def active(root):
    path = Path(root) / "collection.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    from driversi.data.datasets.synthetic_storage import HISTORY_ONLY_IMAGE_CONTRACT, ANCHOR_ONLY_IMAGE_CONTRACT
    contract_version = {3: (HISTORY_ONLY_IMAGE_CONTRACT, "synthetic_driverl_history_v3"), 4: (ANCHOR_ONLY_IMAGE_CONTRACT, "synthetic_driverl_anchor_v4")}
    expected = contract_version.get(value.get("schema_version"))
    history = expected is not None and (value.get("image_contract"), value.get("split")) == expected
    pdms = (value.get("schema_version") in (2, 3, 4) and value.get("trajectory_quality") is None
            and value.get("trajectory_admission") == PDMS_POLICY
            and (history or (value.get("schema_version") == 2 and value.get("split") == "synthetic_driverl_pdms_v2")))
    if value.get("trajectory_quality") != POLICY and not pdms:
        raise ValueError("Active collection requires the declared per-trajectory EPDMS policy")
    return value


def activate_pdms(root, campaign):
    """Explicit operator-authorized new split; preserve all previous receipts."""
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    if root != project_root().resolve() / "dataset" or not campaign.is_relative_to(project_root().resolve() / "exp"):
        raise ValueError("Use the project dataset and an exp campaign")
    campaign.mkdir(parents=True, exist_ok=True)
    value = dict(schema_version=2, campaign=str(campaign), split="synthetic_driverl_pdms_v2",
                 storage="openscene_simscale_views", history_frames=4, future_frames=10,
                 native_target_unchanged=True, source_split="navtrain", original_data_included=False,
                 admission_required=True, trajectory_quality=None, trajectory_admission=PDMS_POLICY)
    with (root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = active(root)
        if old == value:
            return value
        if old is None or old.get("schema_version") != 1 or old.get("trajectory_quality") != POLICY:
            raise ValueError("Only the explicit complete-EPDMS to PDMS transition is supported")
        archived = root / "manifests" / old["split"] / "collection.json"
        if archived.exists() and json.loads(archived.read_text()) != old:
            raise ValueError("Prior collection contract differs from its archive")
        destination = root / "manifests" / value["split"]
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("New admission version already contains data")
        write_json(archived, old)
        for folder in ("navsim_logs", "sensor_blobs", "sensor_blobs_hist", "sensor_blobs_fut", "manifests"):
            (root / folder / value["split"]).mkdir(parents=True, exist_ok=True)
        write_json(destination / "index.json", dict(schema_version=1, split=value["split"],
                   samples=[], count=0, qualification="native_admission_revalidation_required"))
        write_json(root / "collection.json", value)
    return value



def activate_history_only(root, campaign):
    """Activate the common four-history RGB collection, preserving old samples."""
    from driversi.data.datasets.synthetic_storage import HISTORY_ONLY_IMAGE_CONTRACT
    return _activate_image_collection(root, campaign, HISTORY_ONLY_IMAGE_CONTRACT, 3, "synthetic_driverl_history_v3", {2})


def activate_anchor_only(root, campaign):
    """Activate the released ChainFlow's four-anchor-camera image contract."""
    from driversi.data.datasets.synthetic_storage import ANCHOR_ONLY_IMAGE_CONTRACT
    return _activate_image_collection(root, campaign, ANCHOR_ONLY_IMAGE_CONTRACT, 4, "synthetic_driverl_anchor_v4", {2, 3})


def _activate_image_collection(root, campaign, contract, version, split, previous_versions):
    """Separate new image contracts without re-signing prior sample evidence."""

    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    project = project_root().resolve()
    if root != project / "dataset" or not campaign.is_relative_to(project / "exp"):
        raise ValueError("Use the project dataset and an exp campaign")
    root.mkdir(parents=True, exist_ok=True)
    campaign.mkdir(parents=True, exist_ok=True)
    value = dict(schema_version=version, campaign=str(campaign), split=split,
                 storage="openscene_simscale_views", history_frames=4, future_frames=10,
                 native_target_unchanged=True, source_split="navtrain", original_data_included=False,
                 admission_required=True, trajectory_quality=None, trajectory_admission=PDMS_POLICY,
                 image_contract=contract, future_images_included=False,
                 allowed_anchor_indices=[3])
    with (root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = active(root)
        if old == value:
            return value
        if old is not None and old.get("schema_version") not in previous_versions:
            raise ValueError("Only a new or existing strict-PDMS collection can activate history-only export")
        destination = root / "manifests" / value["split"]
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("History-only collection already contains a different campaign")
        if old is not None:
            archived = root / "manifests" / old["split"] / "collection.json"
            if archived.exists() and json.loads(archived.read_text()) != old:
                raise ValueError("Prior collection contract differs from its archive")
            write_json(archived, old)
        for folder in ("navsim_logs", "sensor_blobs", "sensor_blobs_hist", "manifests"):
            (root / folder / value["split"]).mkdir(parents=True, exist_ok=True)
        write_json(destination / "index.json", dict(schema_version=1, split=value["split"],
                   samples=[], count=0, qualification="native_admission_revalidation_required"))
        write_json(root / "collection.json", value)
    return value

def validate_pdms(sample):
    from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence

    native = sample["row"]["admission"]
    if digest(native["path"]) != native["sha256"]:
        raise ValueError("Native admission changed")
    proof = json.loads(Path(native["path"]).read_text())
    report_path = proof["branch"]["report"]
    if digest(report_path) != proof["files"][report_path]:
        raise ValueError("Executed branch evidence changed")
    evidence = json.loads(Path(report_path).read_text())["trajectory_evidence"]
    if evidence.get("admission_profile") != PDMS_POLICY["profile"]:
        raise ValueError("Explicit strict PDMS profile required")
    validate_trajectory_evidence(evidence)
    return dict(PDMS_POLICY, score=evidence["score"]["score"])


def initialize(root, campaign):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    project = project_root().resolve()
    if root != project / "dataset" or not campaign.is_relative_to(project / "exp"):
        raise ValueError("Use the project dataset and an exp campaign")
    campaign.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    value = dict(schema_version=1, campaign=str(campaign), split="synthetic_driverl_v1",
                 storage="openscene_simscale_views", history_frames=4, future_frames=10,
                 native_target_unchanged=True, source_split="navtrain",
                 original_data_included=False, admission_required=True, trajectory_quality=POLICY)
    with (root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = active(root)
        if old is not None:
            if old != value:
                raise ValueError("Dataset already belongs to another campaign")
            return old
        for name in ("synthetic_navsim/samples", "synthetic_navsim/admitted",
                     "synthetic_navsim/.staging", "navsim_logs", "sensor_blobs",
                     "sensor_blobs_hist", "sensor_blobs_fut", "manifests"):
            path = root / name
            if path.exists() and any(path.iterdir()):
                raise ValueError("Initialize only the cleared dataset; never delete existing data")
        for name in ("navsim_logs", "sensor_blobs", "sensor_blobs_hist", "sensor_blobs_fut", "manifests"):
            (root / name / value["split"]).mkdir(parents=True, exist_ok=True)
        write_json(root / "collection.json", value)
        write_json(root / "manifests" / value["split"] / "index.json",
                   dict(schema_version=1, split=value["split"], samples=[], count=0,
                        qualification="native_admission_revalidation_required"))
    return value


def require_campaign(root, evidence, *, link_location=False):
    config = active(root)
    path = Path(evidence)
    owned = path.parent.resolve() / path.name if link_location else path.resolve()
    if config and not owned.is_relative_to(Path(config["campaign"]).resolve()):
        raise ValueError("Current dataset accepts only the active campaign")
    return config


def _link(path, target):
    path, target = Path(path), Path(target).resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve(strict=True) != target:
            raise ValueError("Existing sensor link belongs to another sample")
    elif path.exists():
        raise ValueError("Refusing to replace existing dataset payload")
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def publish(reference_path, root):
    """Publish verified native supervision; retrieval processing may be deferred."""
    from driversi.data.datasets.native_admission import validate_sample
    from driversi.data.datasets.synthetic_publish import _verify

    root = Path(root).resolve()
    reference_path = Path(reference_path).resolve(strict=True)
    ref = json.loads(reference_path.read_text())
    config = require_campaign(root, ref["sample_report"])
    if config is None:
        return None
    from driversi.data.datasets.training_publish import KIND, validate_reference
    training_only = ref.get("kind") == KIND
    pairs = [("export", "export_sha256"), ("sample_report", "sample_report_sha256")]
    if not training_only:
        pairs.append(("feature_report", "feature_report_sha256"))
    for path_key, sha_key in pairs:
        if digest(ref[path_key]) != ref[sha_key]:
            raise ValueError("Admitted collection source changed")
    sample = (validate_reference(ref, root / "synthetic_navsim") if training_only
              else validate_sample(ref["sample_report"]))
    metric = validate_epdms(sample) if config.get("trajectory_quality") == POLICY else None
    pdms = validate_pdms(sample) if config.get("trajectory_admission") == PDMS_POLICY else None
    export_path = Path(ref["export"]).resolve(strict=True)
    package = ref["package_id"]
    if export_path != root / "synthetic_navsim/samples" / package / "export.json":
        raise ValueError("Collection requires its canonical generated package")
    export = json.loads(export_path.read_text())
    _verify(export)
    from driversi.data.datasets.synthetic_storage import history_only_images, image_frame_indices
    history_only = history_only_images(config.get("image_contract"))
    rendered_indices = image_frame_indices(config.get("image_contract"))
    if export.get("image_contract") != config.get("image_contract"):
        raise ValueError("Native export and active collection image contracts differ")
    if sample["sample_id"] != export["token"] or ref["sample_id"] != sample["sample_id"]:
        raise ValueError("Native sample and export identity differ")
    if not export["token"].startswith("synthetic_") or export.get("anchor_index") != 3:
        raise ValueError("Executed synthetic history required")
    with Path(export["log_path"]).open("rb") as stream:
        original = pickle.load(stream)  # Trusted, sealed native producer only.
    if len(original) != config["history_frames"] + config["future_frames"]:
        raise ValueError("Current collection requires four history and ten actual future frames")
    if any(b["timestamp"] - a["timestamp"] != 500000 for a, b in zip(original, original[1:])):
        raise ValueError("Continuous 2 Hz native frames required")
    frames, links, counts = copy.deepcopy(original), [], dict(history=0, future=0)
    branch = original[3]["log_name"]
    if not branch or Path(branch).name != branch or branch in (".", ".."):
        raise ValueError("Safe synthetic log identifier required")
    cameras = None
    for i, frame in enumerate(frames):
        role = "history" if i < 4 else "future"
        observed = set()
        if frame["log_name"] != branch:
            raise ValueError("One branch per native log required")
        for camera, info in frame["cams"].items():
            if info.get("data_path") is None:
                continue
            rel = Path(info["data_path"])
            if len(rel.parts) != 4 or rel.parts[:3] != (role, branch, camera):
                raise ValueError("Native sensor partition differs from its frame role")
            pixel = (Path(export["sensor_root"]) / rel).resolve(strict=True)
            if not pixel.is_relative_to(export_path.parent / "sensors"):
                raise ValueError("Sensor must belong to canonical package")
            # Native image_files is the anchor-only policy input, not all 56
            # stored images. The sealed files map covers history and future.
            if export["files"].get(str(pixel)) != digest(pixel):
                raise ValueError("Sensor pixels are not sealed by export")
            relative = Path(*rel.parts[1:])
            info["data_path"] = relative.as_posix()
            links.append((root / "sensor_blobs" / config["split"] / relative, pixel))
            folder = "sensor_blobs_hist" if i < 4 else "sensor_blobs_fut"
            links.append((root / folder / config["split"] / relative, pixel))
            observed.add(camera)
            counts[role] += 1
        if history_only and i not in rendered_indices:
            if observed or any(any(v is not None for v in info.values()) for info in frame["cams"].values()):
                raise ValueError("Unrendered numerical frames must not contain camera payload")
        else:
            if len(observed) != 4 or (cameras is not None and observed != cameras):
                raise ValueError("Four consistent native cameras required")
            cameras = observed
    split = config["split"]
    log = root / "navsim_logs" / split / (branch + ".pkl")
    manifest = root / "manifests" / split / (package + ".json")
    payload = pickle.dumps(frames, protocol=4)
    with (root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if active(root) != config:
            raise ValueError("Active dataset changed during publication")
        if log.exists() and log.read_bytes() != payload:
            raise ValueError("Existing collection log differs")
        for link, pixel in links:
            _link(link, pixel)
        if not log.exists():
            temporary = log.with_suffix(".pkl.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, log)
        item = dict(package_id=package, token=export["token"], log_path=str(log),
                    log_sha256=digest(log), sensor_root=str(root / "sensor_blobs" / split),
                    canonical_export=str(export_path), canonical_export_sha256=digest(export_path),
                    admission_reference=str(reference_path), admission_reference_sha256=digest(reference_path),
                    history_frames=4, future_frames=10, image_counts=counts,
                    compatibility_changes=["camera_data_path_only"],
                    native_target_unchanged=True, admission_recheck_required=True, epdms=metric)
        if pdms is not None:
            item["pdms"] = pdms
        if history_only:
            item.update(image_contract=config["image_contract"], future_images_included=False,
                        allowed_anchor_indices=[3])
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old != item:
                raise ValueError("Collection publication differs from prior receipt")
        else:
            write_json(manifest, item)
        # Only indexed receipts are discoverable; interrupted links alone do not admit data.
        index_path = manifest.parent / "index.json"
        index = json.loads(index_path.read_text())
        if package not in index["samples"]:
            index["samples"].append(package)
            index["samples"].sort()
            index["count"] = len(index["samples"])
            write_json(index_path, index)
    return dict(manifest=str(manifest), **item)


def validate_members(root):
    """Read the committed collection index and recheck native admission, not flags."""
    from driversi.data.datasets.native_admission import validate_sample
    from driversi.data.datasets.synthetic_publish import _verify

    root = Path(root).resolve()
    config = active(root)
    if config is None:
        raise ValueError("No active collection")
    index_path = root / "manifests" / config["split"] / "index.json"
    index = json.loads(index_path.read_text())
    if index["count"] != len(index["samples"]) or len(set(index["samples"])) != index["count"]:
        raise ValueError("Collection member count differs")
    rows = []
    for package in index["samples"]:
        if Path(package).name != package or package in (".", ".."):
            raise ValueError("Unsafe package ID")
        item = json.loads((index_path.parent / (package + ".json")).read_text())
        for path_key, sha_key in (("log_path", "log_sha256"), ("canonical_export", "canonical_export_sha256"),
                                 ("admission_reference", "admission_reference_sha256")):
            if digest(item[path_key]) != item[sha_key]:
                raise ValueError("Collection evidence changed")
        ref = json.loads(Path(item['admission_reference']).read_text())
        if digest(ref['sample_report']) != ref['sample_report_sha256']:
            raise ValueError("Native admission receipt changed")
        sample = validate_sample(ref['sample_report'])
        if config.get('trajectory_quality') == POLICY and validate_epdms(sample) != item.get('epdms'):
            raise ValueError('Collection EPDMS admission differs')
        if config.get('trajectory_admission') == PDMS_POLICY and validate_pdms(sample) != item.get('pdms'):
            raise ValueError('Collection strict PDMS admission differs')
        export = json.loads(Path(item['canonical_export']).read_text())
        _verify(export)
        if sample['sample_id'] != item['token']:
            raise ValueError("Native admission identity differs")
        with Path(item['log_path']).open('rb') as stream:
            frames = pickle.load(stream)
        for frame in frames:
            for camera in frame['cams'].values():
                if camera.get('data_path') is None:
                    continue
                pixel = (Path(item['sensor_root']) / camera['data_path']).resolve(strict=True)
                if export['files'].get(str(pixel)) != digest(pixel):
                    raise ValueError("Collection sensor link differs from admitted pixels")
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--activate-pdms", action="store_true")
    modes.add_argument("--history-only", action="store_true")
    modes.add_argument("--anchor-only", action="store_true")
    args = parser.parse_args()
    operation = activate_anchor_only if args.anchor_only else activate_history_only if args.history_only else (activate_pdms if args.activate_pdms else initialize)
    print(json.dumps(operation(project_root() / "dataset", args.campaign)))


if __name__ == "__main__":
    main()
