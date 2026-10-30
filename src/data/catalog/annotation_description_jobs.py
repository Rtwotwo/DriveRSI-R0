"""Prepare bounded training-only annotation-reuse pilots; optional short text requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import annotation_descriptions as descriptors
from . import bailian_caption as client
from .store import identity
from driversi.data.sources.navsim import load_log, yaw_from_quaternion
from driversi.data.scenes.context import read_map_context
from driversi.rendering.whitebox.exporter import sha256

CONFIG = Path(__file__).resolve().parents[2] / "config/data/scene-labels/reuse-prompt.md"


def log_facts(record, frames, maps):
    import shapely
    a = record["anchor_index"]
    if a < 4 or a + 10 >= len(frames):
        raise ValueError("Require two seconds of prior context and five future seconds")
    window = frames[a-4:a+11]
    if frames[a]["token"] != record["token"]:
        raise ValueError("Anchor mismatch")
    if any(f["log_name"] != record["source_log"] for f in window):
        raise ValueError("Cross-log window")
    times = np.asarray([f["timestamp"] for f in window], dtype=np.int64) / 1e6
    if not np.allclose(np.diff(times), .5, atol=.03, rtol=0):
        raise ValueError("Discontinuous source; no interpolation or imputation in this pilot")
    pose = np.asarray([f["ego2global_translation"] for f in window], dtype=float)
    states = np.asarray([f["ego_dynamic_state"] for f in window], dtype=float)
    yaw = np.unwrap([yaw_from_quaternion(f["ego2global_rotation"]) for f in window])
    if (not np.isfinite(pose).all() or not np.isfinite(states).all()
            or not np.isfinite(yaw).all()):
        raise ValueError("Invalid motion")
    speed = np.linalg.norm(states[:, :2], axis=1) * 3.6
    origin = pose[4]
    xy = pose[:, :2] - origin[:2]
    bounds = np.r_[xy.min(axis=0)-40, xy.max(axis=0)+40]
    features, layers, provenance = read_map_context(
        frames[a]["map_location"], Path(maps), origin, bounds)
    points = shapely.points(xy)
    support = {}
    for name in ("intersections", "lanes_polygons", "crosswalks"):
        if name not in layers or layers[name]["availability"] == "unavailable":
            support[name] = None
            continue
        geoms = [shapely.geometry.shape(f["geometry"]) for f in features if f["layer"] == name]
        support[name] = int(shapely.covers(shapely.union_all(geoms), points).sum())
    road = {
        "map_name": frames[a]["map_location"],
        "map_version": provenance["version"],
        "relation_basis": "source rear-axle XY at native sample times inside map polygons",
        "intersection_samples": support["intersections"],
        "lane_polygon_samples": support["lanes_polygons"],
        "crosswalk_samples": support["crosswalks"],
        "sample_count": len(window),
        "lane_change": "not_computed",
        "junction_shape": "unknown",
        "ego_signal_phase": "unknown",
    }
    return {
        "window_relative_anchor_s": [round(float(times[0]-times[4]), 3),
                                     round(float(times[-1]-times[4]), 3)],
        "motion": {
            "basis": "source ego_dynamic_state velocity; source pose yaw",
            "start_kmh": round(float(speed[0]), 2),
            "end_kmh": round(float(speed[-1]), 2),
            "range_kmh": np.round([speed.min(), speed.max()], 2).tolist(),
            "net_heading_change_deg": round(float(np.rad2deg(yaw[-1]-yaw[0])), 2),
        },
        "road": road,
    }


def prepare(index_path, annotations_path, maps, output, limit=6):
    if not 1 <= limit <= 6:
        raise ValueError("Pilot limit must be 1..6")
    index = json.loads(Path(index_path).read_text())
    if index["split"] != "navtrain" or not index["complete"]:
        raise ValueError("Require complete training member index")
    annotations = json.loads(Path(annotations_path).read_text())
    output = Path(output)
    inputs = dict(index_sha256=sha256(index_path), annotations_sha256=sha256(annotations_path),
                  source_sha256=sha256(__file__), descriptor_sha256=sha256(descriptors.__file__),
                  limit=limit, maps=str(Path(maps).resolve()))
    if (output/"preparation.json").exists():
        old = json.loads((output/"preparation.json").read_text())
        if old["inputs"] != inputs:
            raise ValueError("Changed inputs: use a new pilot output directory")
        for f, digest in old["files"].items():
            if sha256(output/f) != digest:
                raise ValueError("Prepared file changed")
        return old
    groups = {}
    members = set(index["expected_tokens"])
    for row in index["records"]:
        if row["kind"] == "original_navtrain" and row["token"] in members:
            key = row["source_log"]+"_"+row["token"]
            if key in annotations:
                groups.setdefault(row["source_log"], []).append(row)
    selected, rejected, commands = [], [], {}
    for log in sorted(groups, key=identity):
        rows = groups[log]
        # Mid-log member avoids deliberately selecting only the first easy frame.
        record = rows[len(rows)//2]
        row = annotations[log+"_"+record["token"]]
        facts = descriptors.annotation_facts(row)
        command = facts["navigation_command"]
        if commands.get(command, 0) >= 2:
            continue
        try:
            frames = load_log(Path(record["log_path"]), trusted=True)
            facts.update(log_facts(record, frames, maps))
        except (ValueError, OSError) as exc:
            rejected.append(dict(token=record["token"], error_type=type(exc).__name__,
                                 reason=str(exc)[:200]))
            continue
        result = descriptors.describe(record["token"], log, facts)
        result["provenance"] = dict(source_log_sha256=sha256(record["log_path"]),
                                    annotation_key=log+"_"+record["token"],
                                    annotation_source_sha256=inputs["annotations_sha256"])
        client.write(output/"scenes"/(record["token"]+".json"), result)
        selected.append(record)
        commands[command] = commands.get(command, 0)+1
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError("No eligible original training samples")
    client.write(output/"records.json", selected)
    files = {str(p.relative_to(output)):sha256(p) for p in (output/"scenes").glob("*.json")}
    files["records.json"] = sha256(output/"records.json")
    report = dict(inputs=inputs, scene_count=len(selected), source_logs=len(selected),
                  command_counts=commands, files=files, rejected=rejected,
                  api_calls=0, student_scoring=False, training=False,
                  full_topology=False, factual_accuracy_measured=False)
    client.write(output/"preparation.json", report)
    return report


def text_jobs(output, limit=3):
    if not 1 <= limit <= 3:
        raise ValueError("Text pilot limit must be 1..3")
    prompt = CONFIG.read_text()
    jobs = []
    for path in sorted((Path(output)/"scenes").glob("*.json"))[:limit]:
        scene = json.loads(path.read_text())
        # IDs and private filesystem paths are not needed by the language service.
        facts = {k:v for k,v in scene["facts"].items()}
        if "road" in facts:
            facts["road"] = {k:v for k,v in facts["road"].items() if k not in ("map_name","map_version")}
        job = dict(token=scene["token"], scope="retrospective_annotation_summary",
                   prompt=prompt, facts=facts, evidence=[],
                   input_identity=identity(dict(prompt=prompt, facts=facts)))
        # Store the exact request ingredients before any paid execution. Receipts
        # alone cannot recover an older prompt after the maintained file changes.
        client.write(Path(output)/"api/prepared"/
                     (scene["token"]+"-"+job["input_identity"]+".json"), job)
        jobs.append(job)
    return jobs


def decode(job, model, response):
    choice = response["choices"][0]
    raw = choice["message"]["content"]
    if choice.get("finish_reason") != "stop" or not isinstance(raw, str):
        raise ValueError("Incomplete response")
    value = json.loads(raw)
    if set(value) != {"summary_zh", "evidence_fields"}:
        raise ValueError("Unexpected response fields")
    if not isinstance(value["summary_zh"], str) or not 1 <= len(value["summary_zh"]) <= 200:
        raise ValueError("Summary must be 1..200 characters")
    refs = value["evidence_fields"]
    if (not isinstance(refs, list) or not refs
            or not all(isinstance(ref, str) for ref in refs)
            or len(refs) != len(set(refs))):
        raise ValueError("Unique evidence fields required")
    for ref in refs:
        if not isinstance(ref, str) or ref not in job["facts"]:
            raise ValueError("Unknown evidence field")
    return dict(status="format_valid_unreviewed", model=model, raw=raw,
                returned_model=response.get("model"), request_id=response.get("id"),
                usage=response.get("usage"), description=value,
                factual_status="language_paraphrase_requires_review",
                indexable=False, trainable=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("index", "annotations", "maps", "output"):
        parser.add_argument("--"+arg, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--private-config")
    parser.add_argument("--credential-file")
    args = parser.parse_args()
    report = prepare(args.index, args.annotations, args.maps, args.output, args.limit)
    if args.execute:
        config = client.private_config(args.private_config)
        if len(config["models"]) != 1:
            raise ValueError("One explicitly configured model required")
        config.update(max_requests=3, max_attempts=1, max_tokens=600, concurrency=1)
        result = client.run(text_jobs(args.output), config, Path(args.output)/"api",
                            key=client.credential(args.credential_file), decode=decode)
        print(json.dumps({k:v for k,v in result.items() if k != "rows"}))
        if not result["complete"]:
            raise SystemExit(2)
    else:
        print(json.dumps({k:v for k,v in report.items() if k not in ("inputs","files","rejected")}))


if __name__ == "__main__":
    main()

