"""Independent AutoDrive-P3 copies augmented with current-view retrieval drafts."""
from __future__ import annotations

import json
import shutil
import sqlite3
import io
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .bailian_caption import write
from .store import identity
from driversi.data.sources.navsim import load_log
from driversi.rendering.whitebox.conditioning import CAMERAS
from driversi.rendering.whitebox.exporter import sha256

VERSION = "driversi.cluster_caption.v1"
CONFIG = Path(__file__).resolve().parents[2]/"config/data/scene-labels"
DIMENSIONS = {"road_layout", "lane_markings", "traffic_control", "vehicle_configuration",
              "vulnerable_users", "roadside_obstacles", "visibility_context"}


def connect(library):
    db = sqlite3.connect(Path(library)/"captions.sqlite3", timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db


def initialize(index_file, annotation_file, library):
    library = Path(library)
    library.mkdir(parents=True, exist_ok=True)
    inputs = dict(index_sha256=sha256(index_file), annotation_sha256=sha256(annotation_file))
    if (library/"manifest.json").exists():
        old = json.loads((library/"manifest.json").read_text())
        if old["inputs"] != inputs or sha256(library/old["original_file"]) != inputs["annotation_sha256"]:
            raise ValueError("Changed original corpus: use a new library directory")
        return old
    if (library/"captions.sqlite3").exists():
        raise ValueError("Incomplete initialization requires inspection before resuming")
    index = json.loads(Path(index_file).read_text())
    annotations = json.loads(Path(annotation_file).read_text())
    split = index["split"]
    if split not in ("navtrain", "navtest") or not index["complete"]:
        raise ValueError("Complete original navtrain or navtest index required")
    records = index["records"]
    members = set(index["expected_tokens"])
    if len(records) != len(members) or {r["token"] for r in records} != members:
        raise ValueError("Index member coverage differs")
    keyed = {r["source_log"]+"_"+r["token"]:r for r in records}
    if set(keyed) != set(annotations) or any(r["kind"] != "original_"+split for r in records):
        raise ValueError("Annotation and declared split member populations differ")
    if any("cluster_caption" in row for row in annotations.values()):
        raise ValueError("Expected unextended original annotations")
    prefix = "autodrive-p3" if split == "navtrain" else "navsim"
    original_file, extended_file = prefix+".original.json", prefix+".with-cluster-caption.json"
    shutil.copyfile(annotation_file, library/original_file)
    if sha256(library/original_file) != inputs["annotation_sha256"]:
        raise ValueError("Original copy differs")
    db = connect(library)
    try:
        db.execute("CREATE TABLE captions(token TEXT PRIMARY KEY, annotation_key TEXT UNIQUE NOT NULL, record TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', caption TEXT, receipt TEXT, error TEXT)")
        keep = ("token", "source_log", "kind", "log_path", "sensor_root", "anchor_index", "image_files")
        with db:
            db.executemany("INSERT INTO captions(token,annotation_key,record) VALUES(?,?,?)",
                ((r["token"], key, json.dumps({k:r[k] for k in keep})) for key,r in keyed.items()))
    finally:
        db.close()
    # Different logs for the pilot, deterministic and unrelated to student scores.
    logs = {}
    for row in sorted(records, key=lambda r:(r["source_log"], r["anchor_index"])):
        logs.setdefault(row["source_log"], []).append(row)
    pilot = [logs[log][len(logs[log])//2]["token"] for log in sorted(logs,key=identity)[:16]]
    rest = [r["token"] for r in sorted(records,key=lambda r:(r["source_log"],r["anchor_index"]))
            if r["token"] not in set(pilot)]
    batches = [pilot]+[rest[i:i+64] for i in range(0,len(rest),64)]
    write(library/"batches.json", batches)
    manifest = dict(schema_version=VERSION, split=split, scene_count=len(records), inputs=inputs,
                    original_file=original_file, extended_file=extended_file,
                    database="captions.sqlite3", batch_count=len(batches), native_fields_unchanged=True,
                    status="initialized", annotation_source=str(Path(annotation_file).resolve()),
                    no_student_scoring=True, no_production_index_write=True)
    if split == "navtest":
        manifest.update(annotation_origin="navsim_current_anchor_metadata",
                        native_training_labels_available=False, private_test_partition=True)
    write(library/"manifest.json", manifest)
    export(library)
    return manifest


def export(library):
    """Materialize a compatible independent copy; never rewrite native fields."""
    library = Path(library)
    manifest = json.loads((library/"manifest.json").read_text())
    original = json.loads((library/manifest["original_file"]).read_text())
    db = connect(library)
    try:
        rows = {r["annotation_key"]:dict(r) for r in db.execute(
            "SELECT annotation_key,status,caption,receipt,error FROM captions")}
    finally:
        db.close()
    destination = library/manifest["extended_file"]
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        stream.write("{\n")
        for i,(key,value) in enumerate(original.items()):
            if i: stream.write(",\n")
            row = rows[key]
            extension = json.loads(row["caption"]) if row["caption"] else dict(
                schema_version=VERSION,status=row["status"],error=row["error"],indexable=False,
                trainable=False,scope="current_anchor_four_cameras")
            stream.write(json.dumps(key)+": "+json.dumps({**value,"cluster_caption":extension},ensure_ascii=False))
        stream.write("\n}\n")
    temporary.replace(destination)
    write(library/"export-status.json",dict(rows=len(rows),
        counts={state:sum(r["status"]==state for r in rows.values()) for state in {r["status"] for r in rows.values()}},
        file=destination.name, native_fields_unchanged=True))
    return destination


@lru_cache(maxsize=4)
def source_log(path):
    return load_log(Path(path), trusted=True), sha256(path)


def prepare_job(record, annotation, output, prompt, config):
    """Load only current images; retain exact local evidence separately from API text."""
    output = Path(output)
    frames, log_hash = source_log(record["log_path"])
    frame = frames[record["anchor_index"]]
    if (record["kind"] not in ("original_navtrain", "original_navtest") or frame["token"] != record["token"]
            or frame["log_name"] != record["source_log"]):
        raise ValueError("Original indexed anchor mismatch")
    velocity = np.asarray(frame["ego_dynamic_state"][:2], dtype=float)
    if not np.isfinite(velocity).all():
        raise ValueError("Nonfinite current speed")
    facts = dict(scope="current_anchor_only", future_included=False,
                 anchor_speed_kmh=round(float(np.linalg.norm(velocity)*3.6),2),
                 navigation_command=annotation["driving_command"])
    evidence=[]
    for camera in CAMERAS:
        source = Path(record["sensor_root"])/frame["cams"][camera]["data_path"]
        if str(source) not in record["image_files"]:
            raise ValueError("Indexed image does not correspond to current anchor")
        raw = source.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            image.thumbnail((config["image_max_side"], config["image_max_side"]))
            buffer=io.BytesIO();image.save(buffer,format="JPEG",quality=config["jpeg_quality"])
        data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
        target=output/"images"/digest[:2]/(digest+".jpg")
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)
        elif sha256(target)!=digest:raise ValueError("Compressed evidence changed")
        evidence.append(dict(id=camera+"_anchor",camera=camera,relative_time_s=0,
            source_timestamp_us=int(frame["timestamp"]),source_file=str(source),
            source_sha256=hashlib.sha256(raw).hexdigest(),file=str(target.resolve()),
            sha256=digest,bytes=len(data)))
    profile=dict(prompt=prompt,facts=facts,evidence=evidence,source_log_sha256=log_hash)
    return dict(token=record["token"],split=record["kind"].removeprefix("original_"),
                scope="current_anchor_only",input_identity=identity(profile),**profile)


def validate_description(job,value):
    """Same description contract for provider replies and explicitly authored reviews."""
    if set(value)!={"scene_description_zh","attributes","retrieval_tags","uncertainties"}:
        raise ValueError("Unexpected caption fields")
    def prose(value,limit):
        return isinstance(value,str) and 0<len(value)<=limit
    if not prose(value["scene_description_zh"],400) or len(value["scene_description_zh"])<100:
        raise ValueError("Scene description must contain100..400 characters")
    attributes=value["attributes"]
    if not isinstance(attributes,list) or not 4<=len(attributes)<=10:raise ValueError("Invalid attributes")
    allowed={e["id"] for e in job["evidence"]}
    for item in attributes:
        if not isinstance(item,dict) or set(item)!={"dimension","description","evidence_ids","certainty"}:
            raise ValueError("Invalid attribute fields")
        if not isinstance(item["dimension"],str) or item["dimension"] not in DIMENSIONS:
            raise ValueError("Unknown attribute dimension")
        if not prose(item["description"],120) or item["certainty"] not in ("visible","uncertain"):
            raise ValueError("Invalid attribute description")
        refs=item["evidence_ids"]
        if (not isinstance(refs,list) or not refs or
                not all(isinstance(x,str) and x in allowed for x in refs) or len(refs)!=len(set(refs))):
            raise ValueError("Unknown or duplicate camera reference")
    tags=value["retrieval_tags"];unknown=value["uncertainties"]
    if not isinstance(tags,list) or not 3<=len(tags)<=12 or not all(prose(x,24) for x in tags):
        raise ValueError("Invalid retrieval tags")
    if not isinstance(unknown,list) or len(unknown)>4 or not all(prose(x,80) for x in unknown):
        raise ValueError("Invalid uncertainty notes")
    combined=json.dumps(value,ensure_ascii=False)
    flags=[term for term in ("正在变道","正在超车","即将碰撞","为了避让","导致事故") if term in combined]
    return flags


def decode(job,model,response):
    choice=response["choices"][0];raw=choice["message"]["content"]
    if choice.get("finish_reason")!="stop" or not isinstance(raw,str):
        raise ValueError("Incomplete caption response")
    value=json.loads(raw)
    flags=validate_description(job,value)
    return dict(status="format_valid_unreviewed",model=model,raw=raw,usage=response.get("usage"),
        returned_model=response.get("model"),request_id=response.get("id"),description=value,
        content_review_flags=flags,indexable=False,trainable=False)


def extension(token,job,receipt,receipt_file):
    split = job.get("split", "navtrain")
    if split not in ("navtrain", "navtest"):raise ValueError("Unknown caption split")
    return dict(schema_version=VERSION,token=token,split=split,status="machine_draft",
        observation_scope="current_anchor_four_cameras",future_images_included=False,
        **{k:v for k,v in job["facts"].items() if k in ("navigation_command","anchor_speed_kmh")},
        **receipt["description"],model=receipt["model"],
        evidence_file=str(Path(receipt_file).parents[2]/"jobs.json"),receipt_file=str(receipt_file),
        review_status="unreviewed_machine_annotation",content_review_flags=receipt["content_review_flags"],
        native_prediction_target="unchanged" if split == "navtrain" else "not_available_in_test_metadata",
        indexable=False,trainable=False)
