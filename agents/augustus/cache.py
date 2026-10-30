"""Audited original-NAVSIM + official AutoDrive-P3 ingestion for native SFT.

Prediction language is copied unchanged; absent reasoning is not synthesized.
Only the native prediction supervision mode is admitted by this entry point.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

from driversi.training.sampling.environment import source_log_group
from driversi_agents.augustus.data import manifest, read_cache, validate_row


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_block(text, key, count=None):
    match = re.fullmatch(r"\s*```json\s*(\[.*\])\s*```\s*", text, re.S)
    if not match:
        raise ValueError("invalid official annotation block")
    values = json.loads(match.group(1))
    if count is not None and len(values) != count:
        raise ValueError("annotation horizon mismatch")
    return [value[key] for value in values] if key else values


def original_window(frames, entry, *, require_uniform_timing=True):
    indices = [i for i, x in enumerate(frames) if x["token"] == entry["frame_id"]]
    if len(indices) != 1:
        raise ValueError("ambiguous or missing original token")
    i = indices[0]
    window = frames[i - 3 : i + 9]
    if i < 3 or len(window) != 12:
        raise ValueError("full original history/future unavailable")
    if any(x["log_name"] != entry["scene_name"] for x in window):
        raise ValueError("source log mismatch")
    times = np.array([x["timestamp"] for x in window], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must increase")
    if require_uniform_timing and not np.all(np.abs(np.diff(times) - 500000) < 50000):
        raise ValueError("source sampling not 2 Hz")
    history = np.array(parse_block(entry["historical_trajectory"], "x_y_radian", 4), dtype=float)
    future = np.array(parse_block(entry["solution"]["planning"], "x_y_radian", 8), dtype=float)
    origin = np.asarray(window[3]["ego2global_translation"])[:2]
    quat = np.asarray(window[3]["ego2global_rotation"])
    from pyquaternion import Quaternion
    yaw = Quaternion(quat).yaw_pitch_roll[0]
    transform = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    actual = []
    for frame in window:
        q = np.asarray(frame["ego2global_rotation"])
        h = Quaternion(q).yaw_pitch_roll[0]
        delta = (h - yaw + np.pi) % (2 * np.pi) - np.pi
        actual.append(
            [*((np.asarray(frame["ego2global_translation"])[:2] - origin) @ transform), delta]
        )
    labels = np.concatenate([history, future])
    actual = np.asarray(actual)
    error = np.abs(labels - actual)
    error[:, 2] = np.abs((labels[:, 2] - actual[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    # Published P3 labels are rounded to two decimal places.
    if not np.isfinite(labels).all() or error.max() > 0.011:
        raise ValueError("annotation does not match original anchor trajectory")
    if not entry["solution"].get("prediction"):
        raise ValueError("prediction annotation missing")
    parse_block(entry["solution"]["prediction"], None)
    return window, history, future, [float(origin[0]), float(origin[1]), float(yaw)]


def build_video(history, sensors, path):
    import cv2
    from moviepy.editor import ImageSequenceClip
    from PIL import Image

    frames = []
    evidence = []
    for frame in history:
        images = []
        for name in ("CAM_L0", "CAM_F0", "CAM_R0"):
            source = (Path(sensors) / frame["cams"][name]["data_path"]).resolve(strict=True)
            if not source.is_relative_to(Path(sensors).resolve()):
                raise ValueError("camera path escapes sensor root")
            image = cv2.imread(str(source))
            if image is None or image.shape != (1080, 1920, 3):
                raise ValueError("native camera shape mismatch")
            images.append(image[:-56] if name == "CAM_F0" else image[28:-28, 416:-416])
            evidence.append(
                dict(token=frame["token"], camera=name, path=str(source), sha256=sha(source))
            )
        resized = cv2.resize(np.concatenate(images, axis=1), (2048, 512))
        # Match upstream's intermediate default-quality JPEG then RGB moviepy/H264.
        ok, encoded = cv2.imencode(".jpg", resized)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        import io

        frames.append(np.asarray(Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")))
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(frames, fps=2.0)
    try:
        clip.write_videofile(str(path), codec="libx264", audio=False, logger=None, threads=2)
    finally:
        clip.close()
    return evidence


def build(annotation, selection, output, repository, *, resume=False):
    annotation, selection, output, repository = map(
        Path, (annotation, selection, output, repository)
    )
    from driversi.agents.backends.sources import require_model_root

    require_model_root(repository, "augustus")
    declaration = json.loads(selection.read_text())
    if (
        set(declaration)
        != {
            "schema_version",
            "splits",
            "protected_logs",
            "parent_training_logs",
            "evaluation_scope",
        }
        or declaration["schema_version"] != 1
    ):
        raise ValueError("invalid cache selection")
    seal = {
        "annotation_sha256": sha(annotation),
        "selection_sha256": sha(selection),
        "builder_sha256": sha(__file__),
        "supervision": ["prediction"],
        "future_images_in_input": False,
    }
    output = output.resolve()
    seal_path = output / "input-seal.json"
    if output.exists():
        if not resume or not seal_path.is_file() or json.loads(seal_path.read_text()) != seal:
            raise ValueError("cache recovery input changed")
    else:
        output.mkdir(parents=True)
        seal_path.write_text(json.dumps(seal, indent=2))
    annotations = json.loads(annotation.read_text())
    result = dict(
        schema_version=1,
        kind="augustus_cache_smoke"
        if declaration["evaluation_scope"] == "cache_and_loader_only"
        else "augustus_original_cache",
        dataset_version="official-p3-" + seal["selection_sha256"][:16],
        splits={},
        protected_logs=declaration["protected_logs"],
        parent_training_logs=declaration["parent_training_logs"],
    )
    if declaration["evaluation_scope"] not in ("cache_and_loader_only", "parent_disjoint"):
        raise ValueError("Explicit evaluation scope required")
    if declaration["evaluation_scope"] == "cache_and_loader_only":
        result.update(evaluation_scope="cache_and_loader_only", parent_provenance="unknown")
    groups = {}
    for split, block in declaration["splits"].items():
        logs = [Path(item["log"]).stem for item in block]
        groups[split] = {source_log_group(x) for x in logs}
    if set(groups) != {"train", "validation", "development"}:
        raise ValueError("three isolated splits required")
    protected = {source_log_group(x) for x in declaration["protected_logs"]}
    if any(g & protected for g in groups.values()):
        raise ValueError("protected source in cache")
    if any(groups[a] & groups[b] for i, a in enumerate(groups) for b in list(groups)[i + 1 :]):
        raise ValueError("source leakage")
    if {source_log_group(x) for x in declaration["parent_training_logs"]} & (
        groups["validation"] | groups["development"]
    ):
        raise ValueError("parent source leakage")
    native = repository / "AUGUstus_SFT/data_process/4_4_build_hf_cache.py"
    sys.path.insert(0, str(native.parent))
    spec = importlib.util.spec_from_file_location("augustus_original_cache_builder", native)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = []
    for split, blocks in declaration["splits"].items():
        rows = []
        records = []
        for block in blocks:
            log = Path(block["log"]).resolve(strict=True)
            with log.open("rb") as f:
                frames = pickle.load(f)
            for token in block["tokens"]:
                key = log.stem + "_" + token
                if key not in annotations:
                    raise ValueError("official annotation absent: " + key)
                entry = annotations[key]
                window, history, future, pose = original_window(frames, entry)
                video = output / "videos" / log.stem / (token + ".mp4")
                image_evidence = build_video(window[:4], block["sensors"], video)
                row = dict(
                    id=key,
                    video=video.read_bytes(),
                    scene_token=log.stem,
                    frame_id=token,
                    driving_command=entry["driving_command"],
                    ego_dynamic_state=entry["ego_dynamic_state"],
                    historical_x_y_angle=json.dumps(history.tolist()),
                    future_x_y_angle=json.dumps(future.tolist()),
                    historical_vx_vy=json.dumps(
                        [
                            [round(float(x), 2) for x in frame["ego_dynamic_state"][:2]]
                            for frame in window[:4]
                        ]
                    ),
                    reasoning_trace=None,
                    solution_perception=entry["solution"]["perception"],
                    solution_prediction=entry["solution"]["prediction"],
                    solution_planning=entry["solution"]["planning"],
                )
                validate_row(row)
                rows.append(row)
                records.append(
                    dict(
                        frame_id=token,
                        log_name=log.stem,
                        timestamp_us=int(window[3]["timestamp"]),
                        ego_pose=pose,
                    )
                )
                evidence.append(
                    dict(
                        frame_id=token,
                        split=split,
                        log_sha256=sha(log),
                        video_sha256=sha(video),
                        images=image_evidence,
                        annotation_key=key,
                        annotation_sha256=hashlib.sha256(
                            json.dumps(entry, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                )
        cache = output / split
        # A complete split is published atomically. Recovery checks rows before reuse.
        if cache.exists():
            ds = read_cache({"cache": str(cache), "records": records})
            if list(ds) != rows:
                raise ValueError("resumed cache differs from regenerated rows")
        else:
            temporary = output / (split + ".building")
            if temporary.exists():
                raise ValueError("uncommitted cache fragment requires inspection")
            module.create_dataset(rows).save_to_disk(str(temporary / "batch_0"))
            temporary.rename(cache)
        result["splits"][split] = dict(cache=str(cache), records=records)
    manifest_path = output / "dataset.json"
    manifest_path.write_text(json.dumps(result, indent=2))
    manifest(manifest_path, allow_smoke=True)
    (output / "provenance.json").write_text(
        json.dumps(dict(**seal, native_builder_sha256=sha(native), samples=evidence), indent=2)
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    print(build(a.annotations, a.selection, a.output, a.repository, resume=a.resume))


if __name__ == "__main__":
    main()
