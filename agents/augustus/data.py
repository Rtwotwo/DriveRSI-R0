"""Native HF cache relabeling. No image preprocessing, language generation or GRPO replay."""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.runtime.budget import write_json
from driversi.training.sampling.environment import source_log_group

SPLITS = ("train", "validation", "development")


def manifest(path, *, allow_smoke=False):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    joint = value.get("kind") == "augustus_joint_cache"
    if joint:
        from driversi_agents.augustus.branch_supervision import normalize_joint_manifest

        value = normalize_joint_manifest(value, path, allow_smoke=allow_smoke)
    required = {
        "schema_version",
        "kind",
        "dataset_version",
        "splits",
        "protected_logs",
        "parent_training_logs",
    }
    benchmark = value.get("kind") in (
        "augustus_locked_navtest_cache",
        "augustus_full_navtest_cache",
    )
    smoke = value.get("kind") == "augustus_cache_smoke" or benchmark
    if benchmark:
        required.add("benchmark")
    if smoke:
        required |= {"evaluation_scope", "parent_provenance"}
        if not allow_smoke:
            raise ValueError("Cache smoke only: parent-disjoint evaluation is not established")
        if (
            value.get("evaluation_scope")
            != (value["benchmark"]["protocol"] if benchmark else "cache_and_loader_only")
            or value.get("parent_provenance") != "unknown"
        ):
            raise ValueError("Invalid smoke provenance")
    if (
        set(value) != required
        or value["schema_version"] != 1
        or value["kind"]
        not in (
            "augustus_original_cache",
            "augustus_cache_smoke",
            "augustus_locked_navtest_cache",
            "augustus_full_navtest_cache",
        )
    ):
        raise ValueError("Unsupported AUGUstus dataset manifest")
    if not value["dataset_version"] or set(value["splits"]) != set(SPLITS):
        raise ValueError("Explicit train, native validation, and independent development required")
    groups = {}
    for key in ("protected_logs", "parent_training_logs"):
        if not isinstance(value[key], list) or any(
            not isinstance(x, str) or not x for x in value[key]
        ):
            raise ValueError("Explicit source log declarations required")
    for split in SPLITS:
        block = value["splits"][split]
        block_keys = (
            {"cache", "records", "joint_source_blocks"}
            if joint and split == "train"
            else {"cache", "records"}
        )
        if set(block) != block_keys or not block["records"]:
            raise ValueError("Nonempty native cache with source identities required")
        block["cache"] = str((path.parent / block["cache"]).resolve())
        seen, logs = set(), set()
        for row in block["records"]:
            if set(row) != {"frame_id", "log_name", "timestamp_us", "ego_pose"}:
                raise ValueError("Incomplete original anchor identity")
            if (
                not isinstance(row["frame_id"], str)
                or not row["frame_id"]
                or row["frame_id"] in seen
                or not isinstance(row["log_name"], str)
                or not row["log_name"]
                or type(row["timestamp_us"]) is not int
            ):
                raise ValueError("Invalid or duplicate anchor")
            pose = np.asarray(row["ego_pose"], dtype=float)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError("Invalid anchor pose")
            seen.add(row["frame_id"])
            logs.add(source_log_group(row["log_name"]))
        groups[split] = logs
        if (not benchmark or split != "development") and logs & {
            source_log_group(x) for x in value["protected_logs"]
        }:
            raise ValueError("Protected logs cannot enter iterative RSI")
    if any(groups[a] & groups[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1 :]):
        raise ValueError("Train, validation and development source logs overlap")
    if {source_log_group(x) for x in value["parent_training_logs"]} & (
        groups["validation"] | groups["development"]
    ):
        raise ValueError("Parent training logs overlap evaluation")
    if benchmark:
        from driversi_agents.augustus.benchmark import validate_membership

        validate_membership(value)
    return value


def cache_files(value):
    files = []
    for block in value["splits"].values():
        root = Path(block["cache"])
        if not root.is_dir():
            files.append(str(root))
            continue
        found = [p for p in root.glob("batch_*/*") if p.is_file()]
        if not found:
            files.append(str(root / "batch_0/state.json"))
        for p in found:
            if not p.resolve().is_relative_to(root.resolve()):
                raise ValueError("External cache file reference")
            files.append(str(p))
    return files


def read_cache(block):
    from datasets import concatenate_datasets, load_from_disk

    batches = sorted(Path(block["cache"]).glob("batch_*"))
    if not batches:
        raise ValueError("No native HF cache batches")
    ds = concatenate_datasets([load_from_disk(str(p)) for p in batches])
    tokens = list(ds["frame_id"])
    expected = [x["frame_id"] for x in block["records"]]
    if len(set(tokens)) != len(tokens) or set(tokens) != set(expected):
        raise ValueError("Native cache coverage differs from source manifest")
    if "joint_source_blocks" in block:
        source_rows = {}
        for source in block["joint_source_blocks"]:
            if "joint_source_blocks" in source:
                raise ValueError("Nested synthetic composition is unsupported")
            for row in read_cache(source):
                if row["frame_id"] in source_rows:
                    raise ValueError("Duplicate mixed source identity")
                source_rows[row["frame_id"]] = row_digest(row)
        if set(source_rows) != set(tokens) or any(
            row_digest(row) != source_rows[row["frame_id"]] for row in ds
        ):
            raise ValueError(
                "Mixed native cache rows differ from admitted original/synthetic sources"
            )
    return ds


def row_digest(row):
    """Fingerprint every native payload field, including actual video bytes."""
    value = {
        k: {"bytes_sha256": hashlib.sha256(v).hexdigest()} if isinstance(v, bytes) else v
        for k, v in row.items()
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def validate_row(row):
    import av

    video = row["video"]
    with av.open(io.BytesIO(video)) as container:
        frames = list(container.decode(video=0))
    if len(frames) != 4 or any((f.width, f.height) != (2048, 512) for f in frames):
        raise ValueError("Expected native four-frame 2048x512 video")
    hist = np.asarray(json.loads(row["historical_x_y_angle"]), dtype=float)
    future = np.asarray(json.loads(row["future_x_y_angle"]), dtype=float)
    if (
        hist.shape != (4, 3)
        or future.shape != (8, 3)
        or not np.isfinite(hist).all()
        or not np.isfinite(future).all()
    ):
        raise ValueError("Malformed native history/future trajectory")
    if np.max(np.abs(hist[-1])) > 1e-3:
        raise ValueError("History does not end at current ego origin")
    if not row.get("solution_prediction"):
        raise ValueError("Native prediction CoT supervision missing")
    return future


def relabel(source, proposals, output, label_source="drivetto", *, allow_smoke=False):
    if label_source not in ("original", "drivetto"):
        raise ValueError("Unknown label source")
    if json.loads(Path(source).read_text()).get("kind") == "augustus_joint_cache":
        raise ValueError("Original-anchor relabel cannot strip same-branch synthetic admission")
    value = manifest(source, allow_smoke=allow_smoke)
    report = json.loads(Path(proposals).read_text())
    if report.get("kind") != "original_anchor_proposals" or report.get("threshold") != 0.95:
        raise ValueError("Expected verified DriveTTO original-state proposals")
    selected = {x["token"]: x for x in report["accepted"]}
    if (label_source == "drivetto" and not selected) or len(selected) != len(report["accepted"]):
        raise ValueError("No unique eligible DriveTTO proposals")
    source_files = {p: digest(p) for p in cache_files(value)}
    train = read_cache(value["splits"]["train"])
    records = {x["frame_id"]: x for x in value["splits"]["train"]["records"]}
    if not set(selected) <= records.keys():
        raise ValueError("Candidate anchor absent from native training cache")
    for split in ("validation", "development"):
        read_cache(value["splits"][split])
    changed = []

    def replace(row):
        token = row["frame_id"]
        gt = validate_row(row)
        if token not in selected or label_source == "original":
            return row
        proposal, origin = selected[token], records[token]
        if (
            proposal["log_name"] != origin["log_name"]
            or proposal["timestamp_us"] != origin["timestamp_us"]
            or not np.allclose(proposal["ego_pose"], origin["ego_pose"], atol=1e-7, rtol=0)
            or not (
                np.allclose(proposal["expert_gt"], gt, atol=1e-5, rtol=0)
                or (
                    report.get("source_label_precision") == "official_p3_two_decimals"
                    and np.allclose(np.round(proposal["expert_gt"], 2), gt, atol=1e-5, rtol=0)
                )
            )
        ):
            raise ValueError("Teacher proposal and native cache anchor/GT mismatch")
        for field in ("table", "contract", "publication"):
            if digest(proposal[field]) != proposal[field + "_sha256"]:
                raise ValueError("DriveTTO evidence changed")
        if proposal["labels"]["score"] < 0.95 or proposal["reason"] is not None:
            raise ValueError("Ineligible proposal")
        row = dict(row)
        row["future_x_y_angle"] = json.dumps(proposal["trajectory"], allow_nan=False)
        # Ego planning/reasoning can contradict a relabel. Native prediction-only
        # CoT remains valid solely in this original logged-world task.
        row["solution_planning"] = ""
        row["reasoning_trace"] = ""
        changed.append(
            {
                "token": token,
                "candidate_row": proposal["candidate_row"],
                "video_sha256": hashlib.sha256(row["video"]).hexdigest(),
            }
        )
        return row

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = train.map(replace, load_from_cache_file=False, keep_in_memory=True)
    if label_source == "drivetto" and len(changed) != len(selected):
        raise ValueError("Not all teacher labels were consumed")
    result.save_to_disk(str(output / "train/batch_0"))
    updated = copy.deepcopy(value)
    updated["splits"]["train"]["cache"] = str(output / "train")
    updated["dataset_version"] += ":" + label_source + ":" + digest(proposals)[:16]
    write_json(output / "dataset.json", updated)
    if source_files != {p: digest(p) for p in cache_files(value)}:
        raise ValueError("Source cache changed while constructing dataset")
    summary = {
        "kind": "augustus_relabel_dataset",
        "manifest": str(output / "dataset.json"),
        "manifest_sha256": digest(output / "dataset.json"),
        "source_manifest_sha256": digest(source),
        "source_cache_files": source_files,
        "cache_files": {p: digest(p) for p in cache_files(updated)},
        "proposals_sha256": digest(proposals),
        "replacements": changed,
        "original_rows": len(train),
        "label_source": label_source,
        "supervision": {
            "history": "original_logged_four_frames",
            "target": "proposed_future_ego_trajectory"
            if label_source == "drivetto"
            else "logged_future_ego_trajectory",
            "frame": "current_ego_rear_axle_x_forward_y_left",
            "poses": 8,
            "interval_seconds": 0.5,
            "on_policy": False,
        },
        "replaced_rows": len(changed),
        "source_groups": sorted({r["log_name"] for r in records.values()}),
        "allowed_cot_components": ["prediction"],
        "new_images": 0,
        "scope": "original_anchor_offline_supervision",
        "on_policy_rl": False,
        "reactive_or_visual_admission": False,
    }
    write_json(output / "report.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--proposals", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label-source", choices=("original", "drivetto"), default="drivetto")
    p.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Retain unknown parent provenance for bounded engineering pilots",
    )
    a = p.parse_args()
    relabel(a.source, a.proposals, a.output, a.label_source, allow_smoke=a.allow_smoke)
