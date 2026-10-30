"""Curate one sealed multi-owner camera case without rerendering or altering evidence."""

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path = source / "manifest.json"
    manifest_sha = digest(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (data.get("kind") != "mtgs_world_session" or data.get("status") != "completed"
            or data.get("trainable") is not False or not data.get("frames")):
        raise ValueError("Expected a complete diagnostic MTGS world session")
    frames = data["frames"]
    if len({str(f.get("generation", 0)) for f in frames}) != 1:
        raise ValueError("This exporter accepts one episode, never concatenated resets")
    controlled = set(data["controlled_ids"])
    camera_ids = tuple(c["camera_id"] for c in frames[0]["cameras"])
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Duplicate camera identity")
    expected_owners = {c["camera_id"]: (c["owner_actor_id"], bool(c["virtual"]))
                       for c in frames[0]["cameras"]}
    virtual_owners = {owner for owner, virtual in expected_owners.values() if virtual}
    if len(virtual_owners) < 2 or "ego" in virtual_owners or not virtual_owners <= controlled:
        raise ValueError("Need at least two distinct controlled non-ego virtual camera owners")
    media = output / "media/multiview"
    media.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    assets, methods, asset_cache = [], {}, {}

    def asset(value):
        if (not isinstance(value, str) or not value or "\\" in value or ":" in value
                or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts):
            raise ValueError("Expected safe portable source asset")
        original = (source / value).resolve()
        if not original.is_relative_to(source) or not original.is_file():
            raise ValueError("Missing or escaping source asset")
        if value in asset_cache:
            return asset_cache[value]
        name = "--".join(PurePosixPath(value).parts)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Unexpected asset name")
        destination = media / name
        checksum = digest(original)
        if destination.exists():
            if destination.is_symlink() or digest(destination) != checksum:
                raise ValueError("Existing curated asset differs")
            method = "existing_verified"
        else:
            try:
                os.link(original, destination)
                method = "hardlink"
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                    raise
                shutil.copyfile(original, destination)
                method = "copy"
            if digest(destination) != checksum:
                raise ValueError("Curated media checksum differs")
        relative = destination.relative_to(output).as_posix()
        methods[method] = methods.get(method, 0) + 1
        assets.append({"source": str(original), "image": relative, "sha256": checksum,
                       "bytes": original.stat().st_size, "transfer": method})
        asset_cache[value] = relative
        return relative

    results, per_frame = [], []
    previous_tick, previous_time = -1, -1.0
    roster = set()
    for frame in frames:
        tick, seconds = frame["tick"], frame["time_seconds"]
        if (not isinstance(tick, int) or tick <= previous_tick or not finite(seconds)
                or seconds <= previous_time or frame.get("render_tick", tick) != tick):
            raise ValueError("World and rendering timestamps are not aligned and monotone")
        if abs(seconds - tick * data["dt_seconds"]) > 1e-6:
            raise ValueError("Source simulation clock is inconsistent")
        previous_tick, previous_time = tick, seconds
        actor_lookup = {a["id"]: a for a in frame["actors"]}
        if len(actor_lookup) != len(frame["actors"]):
            raise ValueError("Duplicate actor identity")
        roster.update(actor_lookup)
        actors = []
        for row in frame["actors"]:
            xy, dims = row["position"], row.get("dimensions", [])
            if len(xy) < 2 or not all(finite(v) for v in xy[:2] + [row["yaw"], row["speed"]]):
                raise ValueError("Non-finite BEV state")
            if len(dims) < 2 or not all(finite(v) and v > 0 for v in dims[:2]):
                raise ValueError("Invalid actor dimensions")
            actors.append({"id": row["id"], "x": xy[0], "y": xy[1], "yaw": row["yaw"],
                           "speed": row["speed"], "length": dims[0], "width": dims[1],
                           "kind": {1: "vehicle", 2: "pedestrian", 3: "cyclist", 10: "other_object"}.get(row.get("kind"), "unknown"), "controlled": bool(row.get("controlled")),
                           "present": row.get("present") is not False,
                           "alive": row.get("alive") is not False,
                           "collision": bool(row.get("collision")),
                           "offroad": bool(row.get("offroad")), "role": row.get("role")})
        if tuple(c["camera_id"] for c in frame["cameras"]) != camera_ids:
            raise ValueError("Camera ordering or population changed")
        cameras, camera_evidence = [], []
        for cam in frame["cameras"]:
            ident, owner, virtual = cam["camera_id"], cam["owner_actor_id"], bool(cam["virtual"])
            if (owner, virtual) != expected_owners[ident] or owner not in actor_lookup:
                raise ValueError("Camera changed or lost its physical owner")
            if actor_lookup[owner].get("present") is False:
                raise ValueError("Camera owner is not present at capture tick")
            if cam.get("render_tick", tick) != tick:
                raise ValueError("Camera and world tick mismatch")
            pose = cam.get("camera_to_world")
            if not (isinstance(pose, list) and len(pose) in (3, 4)
                    and all(len(v) == 4 and all(finite(x) for x in v) for v in pose)):
                raise ValueError("Missing finite camera extrinsics")
            if virtual and cam.get("source") is not None:
                raise ValueError("Virtual camera must not claim an original reference image")
            warnings = ["新位姿画质及物理配准未验收；保留原始渲染伪影。"]
            if virtual:
                warnings += ["其他受控车辆的虚拟车顶机位；不是实拍传感器。",
                             "未提供该车同位姿真实图像；ego外观与行人资产仍有缺口。"]
            else:
                warnings += ["原始照片仅作来源参考；偏离日志后不是像素真值。"]
            camera = {"id": ident, "owner_actor_id": owner, "virtual": virtual,
                      "image": asset(cam["image"]), "warnings": warnings}
            if cam.get("source") is not None:
                camera["source"] = asset(cam["source"])
            if cam.get("valid") is not None:
                camera["valid_mask"] = asset(cam["valid"])
            cameras.append(camera)
            camera_evidence.append({"camera_id": ident, "owner_actor_id": owner,
                                    "virtual": virtual, "camera_to_world": pose,
                                    "camera_to_body": cam.get("camera_to_body"),
                                    "camera_model": cam.get("camera_model"), "K": cam.get("K"),
                                    "support": cam.get("support"), "quality": cam.get("quality")})
        results.append({"tick": tick, "time_seconds": seconds, "actors": actors,
                        "cameras": cameras})
        per_frame.append({"tick": tick, "simulation_timestamp_us": frame.get("simulation_timestamp_us"),
                          "present_count": sum(a["present"] for a in actors),
                          "coverage": frame.get("coverage"), "cameras": camera_evidence})
    if digest(manifest_path) != manifest_sha:
        raise ValueError("Source manifest changed during export")
    trajectories = [{"actor_id": ident, "role": "executed",
                     "points": [[a["x"], a["y"]] for f in results for a in f["actors"]
                                if a["id"] == ident and a["present"]]}
                    for ident in sorted(controlled)]
    card = {"id": "multiview", "title": "七车共同交互 · 三辆车的独立视角",
            "subtitle": "完整人口生命周期版本：7 辆受控车＋2 个非 ego 虚拟机位",
            "city": "Boston", "source_log": data["source_log"],
            "scenario_id": data["scenario_id"], "kind": "规则群体交互 / 多主体虚拟相机",
            "duration_seconds": results[-1]["time_seconds"] - results[0]["time_seconds"],
            "camera_count": len(camera_ids), "frame_count": len(results),
            "camera_coverage": {"controlled_actors": len(controlled),
                                "camera_owners": len({owner for owner, _ in expected_owners.values()}),
                                "configured_cameras": len(camera_ids)},
            "playback": {"mode": "precomputed", "performance": data.get("performance", {})},
            "trainable": False, "learned_self_play": False,
            "summary": "同一物理时刻联动显示七车轨迹、全对象 BEV 与四路 GS。两路机位属于不同非 ego 受控车辆，可检查相机随主体运动及对象出现/离场。",
            "warnings": ["规则策略群体，不是学习型对抗 Self-Play；未在本次展示中更新模型。",
                         "67 个对象是全回合名册，不等于每帧同时在场；回放背景与受控主体分开标记。",
                         "新视角、行人外观、ego外观和高程配准仍未通过视觉数据准入；trainable=false。"],
            "data": "data/multiview.json",
            "thumbnail": next(c["image"] for c in results[15]["cameras"] if c["virtual"])}
    case = {"id": "multiview", "metadata": card, "roads": data.get("map", {}).get("roads", []),
            "routes": data.get("map", {}).get("routes", []), "frames": results,
            "trajectories": trajectories,
            "evidence": {"provenance": {"source_manifest": str(manifest_path),
                         "manifest_sha256": manifest_sha, "source_inputs": data.get("inputs"),
                         "population": data.get("population_configuration"), "source_limits": data.get("limits")},
                         "checks": {"single_episode": True, "timestamps_monotone": True,
                                    "world_render_ticks_equal": True, "camera_owner_present_each_frame": True,
                                    "camera_id_owner_mapping_stable": True,
                                    "virtual_cameras_have_no_source_groundtruth": True,
                                    "media_hashes_verified": True,
                                    "quality_certified": False, "new_rendering_performed": False},
                         "counts": {"controlled_actors": len(controlled), "roster": len(roster),
                                    "virtual_owners": sorted(virtual_owners), "frames": len(results),
                                    "rendered_rgb": sum(len(f["cameras"]) for f in results),
                                    "source_reference_rgb": sum("source" in c for f in results for c in f["cameras"]),
                                    "media_files": len(assets), "media_transfer_methods": methods},
                         "timeline": per_frame, "media": assets,
                         "training_use": "diagnostic_only", "rendered_rgb_used_for_actions": False}}
    for name, document in [("multiview.json", case), ("multiview-card.json", card)]:
        destination = output / "data" / name
        if destination.is_symlink():
            raise ValueError("Refuse to write through symlink")
        destination.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")), encoding="utf-8")
    return case["evidence"]["counts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.session, args.output), ensure_ascii=False))
