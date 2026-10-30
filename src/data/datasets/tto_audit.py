"""Read-only audit of committed DriveTTO endpoint-search tables, not new scoring."""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

COMPONENTS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score")
ROW_DTYPE = np.dtype([("keys", "<i4", (2,)), ("trajectories", "<f4", (8, 3)),
                      ("labels", "<f8", (7,)), ("phase", "<i2")], align=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def scene_path(root, split, token):
    if split not in {"train", "val"} or not re.fullmatch(r"[0-9a-f]{16}", token):
        raise ValueError("Invalid source scene identity")
    root = Path(root).resolve()
    result = (root / "scenes" / split / token).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Scene escapes configured bank")
    return result


def verified_rows(directory):
    directory = Path(directory)
    publication = json_read(directory / "ALL_SCORES.json")
    state = json_read(directory / "state.json")
    if (publication.get("format") != "all_scored_candidates_v1" or
            publication.get("passed") is not True or publication.get("token") != directory.name
            or publication.get("split") != directory.parent.name):
        raise ValueError("Unpublished or mismatched candidate table")
    path = directory / "candidates.npy"
    digest = sha256(path)
    if digest != publication["data_digest"]:
        raise ValueError("Candidate table differs from its publication")
    rows = np.load(path, allow_pickle=False)
    if (rows.dtype != ROW_DTYPE or rows.ndim != 1 or not len(rows)
            or len(rows) != publication["candidates"] or state["count"] != len(rows)):
        raise ValueError("Candidate schema or committed count mismatch")
    labels, trajectories = rows["labels"], rows["trajectories"]
    if (not np.isfinite(labels).all() or not np.isfinite(trajectories).all()
            or np.any((labels < 0) | (labels > 1))):
        raise ValueError("Invalid candidate numbers")
    expected = labels[:, 0] * labels[:, 1] * (5*labels[:, 2]+5*labels[:, 3]+2*labels[:, 4])/12
    if np.max(np.abs(expected-labels[:, 6])) > 1e-9:
        raise ValueError("Saved score does not match declared v1 score components")
    return rows, digest


def distinct_qualified(rows, gt):
    # Same stored-float32, signed-zero-normalized equality as the bank producer.
    def key(value):
        value = np.array(value, dtype="<f4", copy=True)
        value[value == 0] = 0
        return value.tobytes()
    seen = {key(gt)}
    count = 0
    for index in np.flatnonzero(rows["labels"][:, -1] >= .95):
        k = key(rows["trajectories"][index])
        if k not in seen:
            seen.add(k)
            count += 1
    return count


def zero_offset_heading(gt):
    """Unscored diagnostic: current constructor recomputes headings even at zero offset."""
    gt = np.asarray(gt, dtype=float)
    full = np.concatenate([np.zeros((1, 2)), gt[:, :2]])
    tangent = np.concatenate([full[2:]-full[:-2], full[-1:]-full[-2:-1]])
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    return np.where(np.linalg.norm(tangent, axis=1) > .02, heading, gt[:, 2])


def category(best, qualified):
    if qualified:
        return "partial_high_score"
    if best == 0:
        return "all_zero"
    if best < .8:
        return "below_80"
    if best < .9:
        return "80_to_90"
    return "90_to_95"


def audit_scene(bank, record):
    directory = scene_path(bank, record["split"], record["token"])
    rows, digest = verified_rows(directory)
    contract = json_read(directory / "contract.json")
    identity = contract["identity"]
    if any(str(identity[k]) != record[k] for k in ("token", "split", "log_name")):
        raise ValueError("Snapshot identity mismatch")
    if tuple(identity["component_names"]) != COMPONENTS:
        raise ValueError("Unknown score component order")
    gt = np.asarray(identity["expert_gt"], dtype=float)
    expert = json_read(directory / "expert.json")
    if gt.shape != (8, 3) or not np.allclose(expert["trajectory"], gt, rtol=0, atol=1e-9):
        raise ValueError("Expert anchor mismatch")
    labels = rows["labels"]
    best = int(np.argmax(labels[:, -1]))
    n = distinct_qualified(rows, gt)
    delta = (zero_offset_heading(gt)-gt[:, 2]+np.pi) % (2*np.pi)-np.pi
    snapshot_best = float(record["best_pdms"])/100.0  # Export CSV uses 0..100; bank labels use 0..1.
    return {"token": record["token"], "split": record["split"], "log_name": record["log_name"],
            "category": category(float(labels[best, -1]), n), "queries": len(rows),
            "qualified_unique": n, "missing": max(0, 256-n), "best_row": best,
            "best_pdms": float(labels[best, -1]), "best_labels": labels[best].tolist(),
            "expert_labels": expert["labels"], "expert_gt": gt.tolist(),
            "ego_pose": identity["ego_pose"], "timestamp_us": identity["timestamp_us"],
            "candidate_sha256": digest, "contract_sha256": sha256(directory / "contract.json"),
            "expert_sha256": sha256(directory / "expert.json"),
            "snapshot": {"qualified_unique": int(record["qualified_unique"]), "best_pdms": snapshot_best},
            "snapshot_changed": int(record["qualified_unique"]) != min(256, n) or
                                abs(snapshot_best-float(labels[best, -1])) > 1e-10,
            "zero_offset_queries": int(np.all(rows["keys"] == 0, axis=1).sum()),
            "component_zero_counts": dict(zip(COMPONENTS, (labels == 0).sum(axis=0).tolist())),
            "gt_high_but_candidates_low": bool(expert["labels"][-1] >= .95 and n == 0),
            "zero_offset_heading_delta_max_deg": float(np.max(np.abs(delta))*180/np.pi),
            "zero_offset_heading_is_unscored": True,
            "nearest_zero_row": int(np.argmin(np.linalg.norm(rows["keys"].astype(float), axis=1))),
            "best_endpoint_offset_m": (rows["keys"][best]*.001).tolist(),
            "constructor": contract["protocol"]["constructor"],
            "score_context": "saved official NAVSIM v1 reference-plus-one; not reactive multi-agent execution",
            "trainable": False}


def build_audit(report_csv, bank, output):
    with Path(report_csv).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len({r["token"] for r in records}) != len(records):
        raise ValueError("Duplicate snapshot tokens")
    rows, errors = [], []
    for record in records:
        try:
            rows.append(audit_scene(bank, record))
        except (ValueError, OSError, KeyError) as error:
            errors.append({"token": record.get("token"), "error": str(error)})
    counts = {key: sum(r["category"] == key for r in rows) for key in
              ("all_zero", "below_80", "80_to_90", "90_to_95", "partial_high_score")}
    result = {"kind": "tto_shortfall_audit", "schema_version": 1,
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "snapshot_csv": str(Path(report_csv).resolve()), "snapshot_sha256": sha256(report_csv),
              "bank_root": str(Path(bank).resolve()), "components": list(COMPONENTS),
              "summary": {"requested": len(records), "verified": len(rows), "errors": len(errors),
                          "categories": counts, "snapshot_changed": sum(r["snapshot_changed"] for r in rows),
                          "gt_high_candidates_low": sum(r["gt_high_but_candidates_low"] for r in rows)},
              "scenes": rows, "errors": errors, "trainable": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_audit(args.report, args.bank, args.output)["summary"]))
