"""Publish a verified, path-redacted whitebox bundle into a static demo site."""

import argparse
import copy
import html
import json
import re
import shutil
import tempfile
from pathlib import Path

from driversi.rendering.whitebox import exporter
from driversi.rendering.whitebox.schema import validate_pair

FILES = ("source-structure.json", "interaction-structure.json", "map.json")


def public_data(manifest, source, branch, map_data):
    validate_pair(source, branch)
    manifest, source, branch, map_data = copy.deepcopy((manifest, source, branch, map_data))
    source["provenance"] = {
        k: source["provenance"][k]
        for k in ("family", "anchor_timestamp_us", "pose_scope")
        if k in source["provenance"]
    }
    branch["provenance"] = {
        k: branch["provenance"][k]
        for k in ("anchor_timestamp_us", "branch_id", "scene_sha256", "scene_key", "validation")
        if k in branch["provenance"]
    }
    map_data["source"] = {
        k: map_data["source"][k]
        for k in ("version", "projected_crs", "geometry_model")
        if k in map_data["source"]
    }
    manifest["publication"] = "inspection_data_only; full server provenance retained in exp"
    return manifest, source, branch, map_data


def publish(bundle, gallery, name, title):
    bundle, gallery = Path(bundle).resolve(), Path(gallery)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,70}", name):
        raise ValueError("Invalid publication case name")
    if gallery.is_symlink():
        raise ValueError("Gallery must be a real directory")
    gallery.mkdir(parents=True, exist_ok=True)
    gallery = gallery.resolve()
    target = gallery / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for filename in FILES:
        if (bundle / filename).is_symlink() or exporter.sha256(bundle / filename) != manifest[
            "sha256"
        ][filename]:
            raise ValueError("Source bundle hash mismatch or symlink")
    rows = [json.loads((bundle / f).read_text()) for f in FILES]
    manifest, source, branch, map_data = public_data(manifest, *rows)
    manifest["title"] = title
    with tempfile.TemporaryDirectory(prefix=".publish-whitebox-", dir=gallery) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for filename, value in zip(FILES, (source, branch, map_data), strict=True):
            exporter.dump(stage / filename, value)
        manifest["sha256"] = {f: exporter.sha256(stage / f) for f in FILES}
        exporter.dump(stage / "manifest.json", manifest)
        for filename in ("viewer.html", "geometry.js", "viewer.js"):
            shutil.copyfile(Path(exporter.__file__).with_name(filename), stage / filename)
        # Never copy arbitrary source folders, logs, checkpoints or credentials.
        stage.rename(target)
    return manifest


def index(gallery):
    gallery = Path(gallery)
    cards = []
    for path in sorted(gallery.glob("*/gallery-entry.json")):
        data = json.loads(path.read_text())
        if data.get("kind") != "camera_gallery_case":
            continue
        cards.append(
            f'<a class="card" href="{html.escape(path.parent.name)}/index.html">'
            f"<b>{html.escape(data['title'])}</b>"
            f"<p>{data['duration_seconds']:.1f} 秒 · {data['camera_count']} 路绑定相机</p>"
            "<p>原日志 / 新轨迹 · 语义白膜 / RGB / 黑底 BEV</p>"
            f"<small>自由环绕 / 全要素检查 · {data['video_count']} 个白膜视频可下载 →</small></a>"
        )
    for path in sorted(gallery.glob("*/manifest.json")):
        data = json.loads(path.read_text())
        counts = data["counts"]
        cards.append(
            f'<a class="card" href="{html.escape(path.parent.name)}/viewer.html">'
            f"<b>{html.escape(data.get('title', path.parent.name))}</b>"
            f"<p>{data['timeline_end_seconds']:.1f} 秒 · 原日志 {counts['source_frames']} 帧"
            f" / 交互 {counts['interaction_frames']} 帧</p>"
            f"<p>地图 {counts['map_features']} 个要素 · 原日志 "
            f"{counts['source_unique_actors']} / 进入世界 "
            f"{counts['physical_unique_actors']} 个对象</p><small>播放并切换两种状态 →</small></a>"
        )
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DriveRSI · 白膜回放</title>
<style>body{margin:0;background:#eef3f6;color:#143345;font:16px/1.6 system-ui}main{max-width:1100px;margin:45px auto;padding:20px}h1{font-size:32px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}.card{display:block;background:white;color:#18425a;border-radius:12px;padding:24px;text-decoration:none;border:1px solid #ccdae3}.card:hover{border-color:#00818b}.card b{font-size:21px}small{color:#00818b}.note{padding:18px;border-left:4px solid #d3a347;background:#fff9ec;margin:24px 0}a{color:#007f88}</style>
<main><a href="../index.html">← DriveRSI 展示平台</a><h1>语义白膜 · 两种结构化状态</h1>
<p>道路面与三维线框，直接由真实地图和状态记录驱动。切换原始日志与实际交互分支，查看车辆、行人、骑行者的进入、保留和离场。</p>
<div class="cards">"""
        + "".join(cards)
        + """</div><div class="note">相机绑定案例提供 MP4 / ZIP 视频下载；各案例的结构化状态也可下载。交互分支中的碰撞、停车和失败仍按记录展示；显示完整不代表驾驶合格。平面道路不表达立交高程，透视线框不代表真实可见性，白膜不是视觉训练准入证明。</div>
<p>自车前视 / 跟随 / 自由环绕 · 时间轴 / 逐帧 / 变速播放 · 对象选择 · 原始采样，无插值</p></main></html>"""
    )
    temporary = gallery / ".index-next.html"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(gallery / "index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--gallery", help="Explicit static gallery destination; defaults to release config"
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "src/config/visualization/release.json",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    if args.gallery is None:
        config = json.loads(args.release_config.read_text())
        root = Path(config["release_root"]).resolve()
        gallery = (
            root / config["pages_checkout"] / config["site_directory"] / config["demo_directory"]
        )
        if not gallery.resolve().is_relative_to(root):
            raise ValueError("Release destination escapes configured root")
        args.gallery = str(gallery)
    print(json.dumps(publish(args.bundle, args.gallery, args.name, args.title)))
    index(args.gallery)


if __name__ == "__main__":
    main()
