"""Publish admitted synthetic branches through the existing scene replay schema.

This is a read-only projection of signed production receipts. It never changes
admission, renders new images, or counts replay copies as new training samples.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from driversi.visualization.publication.pages import public_card, public_case


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Replay provenance digest mismatch: {Path(path).name}")
    return Path(path)


def ready_rows(outcomes):
    rows = [row for row in outcomes["rows"] if row.get("status") == "training_sample_ready"]
    packages = set()
    for row in rows:
        if not row.get("trainable") or not row.get("distinct") or not row.get("images_changed"):
            raise ValueError("Replay requires an admitted, distinct branch with new images")
        ref = row["dataset_reference"]
        admission = read(verified(ref["path"], ref["sha256"]))
        if admission.get("kind") != "admitted_synthetic_dataset_reference" or not admission.get("trainable"):
            raise ValueError("Missing admitted dataset reference")
        if admission["package_id"] in packages or admission["package_id"] != ref["package_id"]:
            raise ValueError("Duplicate or mismatched sample package")
        packages.add(admission["package_id"])
        for field in ("export", "sample_report", "feature_report"):
            verified(admission[field], admission[field + "_sha256"])
        for path, sha in row["files"].items():
            verified(path, sha)
        sample = read(row["sample_report"])
        if not sample.get("trainable") or sample["row"].get("kind") != "synthetic_navtrain":
            raise ValueError("Only admitted training-source samples may be published")
        features = read(row["feature_report"])
        if not features.get("trainable") or features.get("parent_vectors_copied") is not False:
            raise ValueError("Fresh image feature admission is required")
        if features.get("future_images_encoded") is not False:
            raise ValueError("Future observations must not enter history retrieval features")
    return rows


class Media:
    """Content-addressed copies; lossless WebP is checked against decoded PNG pixels."""

    def __init__(self, output, prefix):
        self.output = output
        self.prefix = prefix
        self.items = {}

    def add(self, path):
        path = Path(path)
        sha = digest(path)
        suffix = ".webp" if path.suffix.lower() == ".png" else path.suffix.lower()
        if suffix not in {".webp", ".jpg", ".jpeg"}:
            raise ValueError("Unexpected replay image type")
        relative = f"media/{self.prefix}/{sha}{suffix}"
        self.items.setdefault(relative, {"source": str(path), "source_sha256": sha})
        return relative

    def copy(self, item):
        relative, evidence = item
        source = Path(evidence["source"])
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        verified(source, evidence["source_sha256"])
        if source.suffix.lower() == ".png":
            with Image.open(source) as original:
                if original.mode not in {"RGB", "RGBA"}:
                    raise ValueError("Unexpected PNG mode; do not silently change pixels")
                if not target.exists():
                    original.save(target, format="WEBP", lossless=True, method=6, exact=True)
                with Image.open(target) as replay:
                    if replay.size != original.size or not np.array_equal(
                        np.asarray(original), np.asarray(replay.convert(original.mode))
                    ):
                        raise ValueError("Lossless replay encoding changed pixels")
        else:
            if not target.exists():
                shutil.copyfile(source, target)
            verified(target, evidence["source_sha256"])
        evidence.update(public_path=relative, public_sha256=digest(target), bytes=target.stat().st_size,
                        pixel_equivalent=True)
        return evidence


def segments(points, presence):
    start = None
    for i in range(len(points) + 1):
        exists = i < len(points) and bool(presence[i])
        if exists and start is None:
            start = i
        if not exists and start is not None:
            yield points[start:i].tolist()
            start = None


def export(outcomes, output, prefix, receipt, source_cameras=None):
    source_cameras = set(source_cameras or ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"])
    output = Path(output)
    rows = ready_rows(read(outcomes))
    if not rows:
        raise ValueError("No admitted branches to publish")
    media = Media(output, prefix)
    cards, audit, parents = [], [], set()
    for index, row in enumerate(rows, 1):
        generation = read(row["generation"])
        for path, sha in generation["files"].items():
            verified(path, sha)
        branch = read(generation["branch_report"])
        execution = Path(branch["execution"])
        for name in ("scene.npz", "teacher.npz"):
            path = execution / name
            verified(path, branch["files"][str(path)])
        with np.load(execution / "scene.npz", allow_pickle=False) as z:
            scene = {key: z[key].copy() for key in z.files}
        with np.load(execution / "teacher.npz", allow_pickle=False) as z:
            rollout = {key: z[key].copy() for key in z.files}
        metadata = json.loads(str(scene["metadata"]))
        control = json.loads(str(rollout["metadata"]))
        states, actions = rollout["states"], rollout["actions"]
        dt = float(control["dt_seconds"])
        if len(states) != 71 or len(actions) != 70 or not np.isclose(dt, .1):
            raise ValueError("This replay contract requires complete 7-second execution at 10 Hz")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite physical execution")
        actor_ids = rollout["actor_ids"].tolist()
        if actor_ids != scene["actor_ids"].tolist() or "ego" not in actor_ids:
            raise ValueError("Actor identity mismatch")
        render = read(generation["render_payload"])
        source = read(generation["source_report"])
        if render["source_log"] != branch["source_log"]:
            raise ValueError("Render/source log mismatch")
        if [f["tick"] for f in render["frames"]] != list(range(0, 71, 5)):
            raise ValueError("Missing rendered replay times")
        sources = {(f["tick"], f["camera"]): f for f in source["frames"]}
        if len(sources) != 60:
            raise ValueError("Source comparison must contain all four cameras at 15 times")
        score = generation["teacher"]["same_anchor_score"]
        psnr = float(np.mean([f["measured_psnr_db"] for f in source["frames"]]))
        cid = f"{prefix}-{index:02d}"
        parent = (branch["source_log"], generation["token"])
        parents.add(parent)
        card = dict(
            id=cid, title=f"Pittsburgh · 合成分支 {index:02d}",
            subtitle="主动扰动新历史 → 同锚点原生 TTS 未来 · 数值试验准入",
            city=metadata["map_name"], source_log=parent[0], scenario_id=parent[1],
            kind="admitted_synthetic", duration_seconds=7.0, camera_count=4, frame_count=15,
            trainable=True, learned_self_play=False, data=f"data/{cid}.json",
            source_type="admitted_synthetic_training_branch", executed_branch="teacher",
            generator=control["policy"],
            summary=(f"本批 {len(rows)} 条不同分支，共用同一源场景；本分支 PDMS {100 * score['score']:.2f}，"
                     f"源位姿平均 PSNR {psnr:.2f} dB。四相机各 4 帧新历史与 10 帧未来，另含起始诊断帧。"),
            warnings=["数值试验准入，不代表独立视觉审核或官方基准成绩",
                      "工程源场景，未确认 ChainFlow 在新锚点存在短板",
                      "0–2 秒 LQR 跟踪扰动历史；2–7 秒 DriveRL 原生 TTS8，非全程 DriveRL",
                      "源位姿 PSNR 不等于新视角 PSNR；原日志照片按时间对照，视点可不同",
                      "网页原日志照片对照：" + "、".join(sorted(source_cameras)) + "；合成图像四路完整保留"],
            score=dict(label="反事实 PDMS", before=None, after=100 * score["score"],
                       protocol=score["protocol"], official_benchmark=False,
                       scored_seconds=score["scored_seconds"], actual_world_seconds=score["actual_world_seconds"]),
        )
        frames = []
        for frame in render["frames"]:
            tick = frame["tick"]
            if set(frame["cameras"]) != {"CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"}:
                raise ValueError("Four-camera coverage mismatch")
            cameras = []
            for name, camera in frame["cameras"].items():
                comparison = sources[tick, name]
                if comparison["source_token"] != frame["source_token"]:
                    raise ValueError("Image/source timestamp identity mismatch")
                cameras.append(dict(id=name, owner_actor_id="ego", virtual=False, tick=tick,
                                    time_seconds=tick * dt,
                                    image=media.add(Path(generation["render_payload"]).parent / camera["file"]),
                                    source=media.add(comparison["real_source_image"]) if name in source_cameras else None))
            actors = []
            for j, aid in enumerate(actor_ids):
                state = states[tick, j]
                present = bool(rollout["presence"][tick, j])
                actors.append(dict(id=aid, x=float(state[0]), y=float(state[1]), z=float(state[2]),
                                   yaw=float(state[3]), speed=float(np.hypot(state[4], state[5])),
                                   length=float(scene["dimensions"][j, 0]), width=float(scene["dimensions"][j, 1]),
                                   kind={1: "vehicle", 2: "pedestrian", 3: "cyclist"}.get(int(scene["kinds"][j]), "other_object"),
                                   controlled=aid in rollout["controlled_ids"], present=present, alive=present,
                                   collision=present and bool(state[10]), offroad=present and bool(state[11])))
            frames.append(dict(tick=tick, time_seconds=tick * dt, phase="initial" if tick == 0 else
                               "history" if tick <= 20 else "future", actors=actors, cameras=cameras))
        card["thumbnail"] = next(c["image"] for c in frames[4]["cameras"] if c["id"] == "CAM_F0")
        roads = [scene["road_points"][a:b, :2].tolist() for a, b in
                 zip(scene["road_offsets"][:-1], scene["road_offsets"][1:])]
        trajectories = [dict(actor_id=aid, role="executed", points=points)
                        for j, aid in enumerate(actor_ids)
                        for points in segments(states[:, j, :2], rollout["presence"][:, j])]
        ego = actor_ids.index("ego")
        trajectories.append(dict(actor_id="ego", role="reference", points=scene["poses"][:, ego, :2].tolist()))
        data = dict(id=cid, metadata=card, roads=roads, routes=[], trajectories=trajectories, frames=frames,
                    evidence=dict(physics_dt_seconds=dt, coordinate_frame=metadata["coordinate_frame"],
                                  join="Images and physical states joined by recorded tick and camera ID",
                                  image_policy="Lossless web copies; original training media unchanged",
                                  trainable=True, logged_benchmark=False, learned_self_play=False,
                                  counts=dict(frames=15, rendered_rgb=60, source_reference_rgb=15 * len(source_cameras)),
                                  report_fields=dict(arrival_tick=20, history_ticks=[5,10,15,20],
                                                     label_ticks=list(range(25,71,5)), restore_exact=bool(generation["teacher"]["restore_exact"])),
                                  branches=dict(teacher=dict(completed_steps=70, trajectory_admitted=True, score=score))))
        write(output / card["data"], public_case(data))
        cards.append(public_card(card))
        audit.append(dict(id=cid, candidate=row["candidate"], package_id=row["dataset_reference"]["package_id"],
                          dataset_reference=row["dataset_reference"], pdms=score["score"] * 100, source_psnr_db=psnr,
                          generation=row["generation"], frames=15, training_images=56, physical_states=71, physical_actions=70))
    with ThreadPoolExecutor(max_workers=4) as pool:
        exported = list(pool.map(media.copy, media.items.items()))
    write(output / "data/cases.json", dict(schema_version=1, presentation_scope="admitted synthetic replay copies", cases=cards))
    result = dict(status="exported", source_scenes=len(parents), replay_branches=len(cards), new_training_samples=0,
                  rendered_images=len(cards)*60, training_images=len(cards)*56,
                  media_files=len(exported), media_bytes=sum(m["bytes"] for m in exported),
                  pixels_verified=True, rows=audit, media=exported)
    write(receipt, result)
    return {key: value for key, value in result.items() if key not in {"rows", "media"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--source-cameras", nargs="+", choices=["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"])
    args = parser.parse_args()
    if not args.prefix.replace("-", "").isalnum():
        parser.error("prefix must contain only letters, digits and hyphens")
    print(json.dumps(export(args.outcomes, args.output, args.prefix, args.receipt, args.source_cameras), ensure_ascii=False))


if __name__ == "__main__":
    main()
