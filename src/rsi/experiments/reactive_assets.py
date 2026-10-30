"""Exact source-time coverage for native GS production, before costly rollout."""

import json
import pickle
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.data.catalog.store import Catalog, identity
from driversi.rsi.experiments.navval_retrieval import training_members
from driversi.runtime.budget import write_json

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def supported_window(raw, anchor, canonical):
    # The 2 s arrival starts four source frames earlier; native DriveRL also
    # needs three real source-history frames before that physical start.
    if anchor < 7 or anchor + 10 >= len(raw):
        return False
    window = raw[anchor - 4 : anchor + 11]
    for frame in window:
        source = canonical.get(frame["token"])
        if source is None or source.get("skipped") or set(CAMERAS) - set(source.get("cams", {})):
            return False
        if abs(int(source["timestamp"]) - int(frame["timestamp"])) > 50000:
            return False
    times = [int(f["timestamp"]) for f in window]
    return all(450000 <= b - a <= 550000 for a, b in zip(times, times[1:]))


def inventory(database, roots, output, *, population="navval"):
    """No rendering/admission: each token has all 15 source poses, four cameras."""
    output = Path(output)
    cat = Catalog(database, "navtrain", readonly=True)
    try:
        members, membership_files = training_members(cat, population)
        logs = sorted(set(members.values()))
        selected = cat.db.execute(
            "SELECT token,record,source_group FROM scenes WHERE source_log IN ("
            + ",".join("?" for _ in logs)
            + ")",
            logs,
        ).fetchall()
        records = {r["token"]: json.loads(r["record"]) for r in selected if r["token"] in members}
        groups = {r["token"]: r["source_group"] for r in selected}
        metadata = sorted(
            {p.resolve() for base in roots for p in Path(base).glob("*/video_scene_dict.pkl")}
        )
        inputs = {**membership_files, **{str(p): digest(p) for p in metadata}}
        signature = identity(dict(files=inputs, members=members, population=population, version=3))
        if output.exists():
            old = json.loads(output.read_text())
            if old["input_id"] == signature:
                return old
            raise ValueError("Asset inventory inputs changed; use a new revision")
        rows, rejected = [], []
        for path in metadata:
            with path.open("rb") as f:
                videos = pickle.load(f)
            for video_id, video in videos.items():
                log = video.get("log_name")
                candidates = [r for r in records.values() if r["source_log"] == log]
                if not candidates:
                    continue
                asset, travel = video_id.rsplit("-", 1)
                bundle = path.parent
                files = [
                    bundle / "background" / (asset + ".ckpt"),
                    path,
                    bundle / "road_height_map/road_height_map.npy",
                    bundle / "road_height_map/sim2.json",
                ]
                if not all(p.is_file() for p in files):
                    rejected.append(dict(bundle=str(bundle), reason="incomplete_asset_files"))
                    continue
                with Path(candidates[0]["log_path"]).open("rb") as f:
                    raw = pickle.load(f)
                canonical = {f["token"]: f for f in video["frame_infos"]}
                supported = [
                    r for r in candidates if supported_window(raw, r["anchor_index"], canonical)
                ]
                rows.extend(
                    dict(
                        token=r["token"],
                        source_log=log,
                        anchor_index=r["anchor_index"],
                        source_group=groups[r["token"]],
                        bundle_root=str(bundle),
                        asset_id=asset,
                        travel_id=int(travel),
                        metadata_sha256=inputs[str(path)],
                        cameras=list(CAMERAS),
                        history_frames=4,
                        future_frames=10,
                        render_times=15,
                        trainable=False,
                    )
                    for r in supported
                )
                rejected.append(
                    dict(
                        bundle=str(bundle),
                        source_log=log,
                        source_candidates=len(candidates),
                        supported=len(supported),
                        unsupported=len(candidates) - len(supported),
                        reason="source_time_support_is_not_visual_admission",
                    )
                )
        value = dict(
            kind="training_gs_source_coverage",
            population=population,
            schema_version=1,
            input_id=signature,
            files=inputs,
            rows=sorted(rows, key=lambda r: (r["source_log"], r["anchor_index"], r["asset_id"])),
            diagnostics=rejected,
            training_members=len(members),
            trainable=False,
            claim="source_pose_coverage_not_counterfactual_quality",
        )
        write_json(output, value)
        return value
    finally:
        cat.close()


def measured_source_coverage(database, coverage_path, source_reports, output, *, minimum_db=25.0):
    """Select windows with measured source fit; this never grants visual admission.

    A repaired checkpoint's score is local to its actually compared timestamps.
    Metadata coverage alone cannot transfer that score to another log interval.
    """
    import math
    import statistics

    from driversi.configuration import read_mapping

    if minimum_db != 25.0:
        raise ValueError("This pilot uses the declared 25 dB source-fit experiment")
    coverage = read_mapping(coverage_path)
    files = {str(Path(coverage_path).resolve()): digest(coverage_path), **coverage["files"]}
    verified = {}
    evidence = []
    for report_path in source_reports:
        report_path = Path(report_path).resolve(strict=True)
        report = read_mapping(report_path)
        if report.get("source_mode") != "canonical_source_pose_without_motion_v1":
            raise ValueError("Canonical measured source comparison required")
        dependencies = {**report["inputs"], **report["files"]}
        for name, sha in dependencies.items():
            if verified.get(name) != sha:
                if digest(name) != sha:
                    raise ValueError("Measured reconstruction evidence changed")
                verified[name] = sha
        files.update(dependencies)
        files[str(report_path)] = digest(report_path)
        scores = {}
        for row in report["frames"]:
            key = (row["source_token"], row["camera"])
            value = row["measured_psnr_db"]
            if key in scores or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Ambiguous or invalid source quality observation")
            scores[key] = value
        evidence.append((str(report_path), report["inputs"], scores))
    cat = Catalog(database, "navtrain", readonly=True)
    raw_logs = {}
    selected, missing = [], []
    try:
        for row in coverage["rows"]:
            log = row["source_log"]
            if log not in raw_logs:
                record = json.loads(
                    cat.db.execute(
                        "SELECT record FROM scenes WHERE token=?", (row["token"],)
                    ).fetchone()[0]
                )
                with Path(record["log_path"]).open("rb") as stream:
                    raw_logs[log] = pickle.load(stream)
            anchor = row["anchor_index"]
            expected = {
                (f["token"], c) for f in raw_logs[log][anchor - 4 : anchor + 11] for c in CAMERAS
            }
            checkpoint = str(Path(row["bundle_root"]) / "background" / (row["asset_id"] + ".ckpt"))
            found = None
            for report_path, inputs, scores in evidence:
                if checkpoint not in inputs or len(expected) != 60 or not expected <= scores.keys():
                    continue
                mean = statistics.mean(scores[k] for k in expected)
                if mean >= minimum_db:
                    found = {
                        **row,
                        "source_quality_mean_db": mean,
                        "source_quality_evidence": report_path,
                    }
                    break
            if found:
                selected.append(found)
            else:
                missing.append(
                    dict(
                        token=row["token"],
                        bundle=row["bundle_root"],
                        reason="no_matching_full_window_source_quality_evidence",
                    )
                )
    finally:
        cat.close()
    result = {
        **coverage,
        "kind": "navval_gs_measured_source_coverage",
        "input_id": identity(dict(files=files, minimum_db=minimum_db)),
        "files": files,
        "rows": selected,
        "diagnostics": missing,
        "claim": "source_fit_prioritization_only_not_visual_admission",
        "minimum_db": minimum_db,
    }
    write_json(output, result)
    return result
