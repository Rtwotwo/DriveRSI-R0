"""Current-anchor multi-camera description pilot, reusing validated train members."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import bailian_caption as client
from .store import identity
from driversi.data.sources.navsim import load_log
from driversi.rendering.whitebox.conditioning import CAMERAS
from driversi.rendering.whitebox.exporter import sha256

PROMPT = Path(__file__).resolve().parents[2]/"config/data/scene-labels/anchor-view-prompt.md"
FIELDS = {"road_layout", "road_paint", "visible_actors", "roadside_obstruction", "visibility_signal"}


def prepare(prepared, token, output):
    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared/"preparation.json").read_text())
    for name in ("records.json", "scenes/"+token+".json"):
        if sha256(prepared/name) != manifest["files"][name]:
            raise ValueError("Changed prepared training evidence")
    records = json.loads((prepared/"records.json").read_text())
    matches = [r for r in records if r["token"] == token]
    if len(matches) != 1 or matches[0]["kind"] != "original_navtrain":
        raise ValueError("One validated original training member required")
    record = matches[0]
    scene = json.loads((prepared/"scenes"/(token+".json")).read_text())
    if scene["split"] != "navtrain" or sha256(record["log_path"]) != scene["provenance"]["source_log_sha256"]:
        raise ValueError("Source log or split changed")
    frame = load_log(Path(record["log_path"]), trusted=True)[record["anchor_index"]]
    if frame["token"] != token or frame["log_name"] != record["source_log"]:
        raise ValueError("Source anchor mismatch")
    velocity = np.asarray(frame["ego_dynamic_state"][:2], dtype=float)
    if not np.isfinite(velocity).all():
        raise ValueError("Invalid current velocity")
    facts = {"scope": "current_anchor_only", "future_included": False,
             "navigation_command": scene["facts"]["navigation_command"],
             "anchor_speed_kmh": round(float(np.linalg.norm(velocity)*3.6), 2)}
    evidence = []
    for camera in CAMERAS:
        path = Path(record["sensor_root"])/frame["cams"][camera]["data_path"]
        with Image.open(path) as source:
            picture = source.convert("RGB")
            picture.thumbnail((768, 768))
            stream = io.BytesIO()
            picture.save(stream, format="JPEG", quality=80)
        data = stream.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        target = output/"images"/(digest+".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        if sha256(target) != digest:
            raise ValueError("Changed compressed image")
        evidence.append(dict(id=camera+"_anchor", camera=camera, relative_time_s=0,
                             source_file=str(path), source_sha256=sha256(path),
                             file=str(target.resolve()), sha256=digest, bytes=len(data)))
    profile = dict(prompt=PROMPT.read_text(), facts=facts, evidence=evidence,
                   source_log_sha256=scene["provenance"]["source_log_sha256"],
                   image_max_side=768, jpeg_quality=80)
    job = dict(token=token, scope="current_anchor_only", input_identity=identity(profile), **profile)
    if (output/"job.json").exists() and json.loads((output/"job.json").read_text()) != job:
        raise ValueError("Changed request: use a new output directory")
    client.write(output/"job.json", job)
    return job


def decode(job, model, response):
    choice = response["choices"][0]
    raw = choice["message"]["content"]
    if choice.get("finish_reason") != "stop" or not isinstance(raw, str):
        raise ValueError("Incomplete anchor response")
    value = json.loads(raw)
    if set(value) != {"summary_zh", "features"}:
        raise ValueError("Unexpected response fields")
    if not isinstance(value["summary_zh"], str) or not 1 <= len(value["summary_zh"]) <= 120:
        raise ValueError("Summary length must be1..120")
    features = value["features"]
    if not isinstance(features, list) or not 1 <= len(features) <= 5:
        raise ValueError("One to five features required")
    allowed = {e["id"] for e in job["evidence"]}
    for item in features:
        if not isinstance(item, dict) or set(item) != {"field", "text", "evidence_ids"}:
            raise ValueError("Invalid feature")
        field = item["field"]
        if not isinstance(field, str) or field not in FIELDS:
            raise ValueError("Invalid feature field")
        if not isinstance(item["text"], str) or not 1 <= len(item["text"]) <= 80:
            raise ValueError("Invalid feature text")
        refs = item["evidence_ids"]
        if (not isinstance(refs, list) or not refs or
                not all(isinstance(ref, str) and ref in allowed for ref in refs)
                or len(refs) != len(set(refs))):
            raise ValueError("Invalid camera evidence")
    return dict(status="format_valid_unreviewed", model=model, raw=raw,
                usage=response.get("usage"), returned_model=response.get("model"),
                request_id=response.get("id"), description=value,
                scope="current_anchor_only", factual_status="visual_review_required",
                indexable=False, trainable=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prepared", "token", "output"):
        parser.add_argument("--"+arg, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--private-config")
    parser.add_argument("--credential-file")
    args = parser.parse_args()
    job = prepare(args.prepared, args.token, args.output)
    if not args.execute:
        print(json.dumps(dict(prepared=True, images=len(job["evidence"]), api_calls=0)))
        return
    config = client.private_config(args.private_config)
    if len(config["models"]) != 1:
        raise ValueError("One explicit model required")
    config.update(max_requests=1, max_attempts=1, max_tokens=900, concurrency=1)
    report = client.run([job], config, Path(args.output)/"api",
                        key=client.credential(args.credential_file), decode=decode)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
