"""Read-only blocker evidence for a continuous nuPlan diagnostic run."""

import argparse
import hashlib
import html
import json
import sqlite3
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.evaluation.metrics.source_support import source_entry_overlaps
from driversi.rendering.inspection.continuous_gallery import shell
from driversi.rendering.bev.replay import export_replay, export_source_replay


def dependency_intervals(life, diagnostics, valid, controlled):
    """Recorded current-world targets; lead selection alone is not causal blame."""
    intervals, active = [], {}
    for tick, commands in enumerate(diagnostics):
        current = {(c["actor"], c["conflict_id"], c.get("conflict_reason"))
                   for c in life[tick].get("executed_traffic_controls", []) if c.get("conflict_id")}
        for c in commands:
            if not valid[tick, controlled.index(c["actor"])] or c.get("controller") == "baseline_action_replay":
                continue
            for h in c.get("traffic_hazards", []):
                current.add((c["actor"], h["actor"], h.get("reason")))
            if c.get("lead_id"):
                current.add((c["actor"], c["lead_id"], "reported_lead"))
        for key in list(active):
            if key not in current:
                intervals.append(active.pop(key))
        for a, b, reason in current:
            item = active.setdefault((a, b, reason), {"actor": a, "target": b, "reason": reason,
                                                      "start_tick": tick, "end_tick_exclusive": tick})
            item["end_tick_exclusive"] = tick + 1
    intervals.extend(active.values())
    return sorted(intervals, key=lambda x: (x["start_tick"], x["actor"], x["target"], str(x["reason"])))


def diagnose(case, db, pair):
    scene = Scene.load(case / "scene.npz")
    source = Scene.load(case / "source.npz")
    if db.stem != source.metadata["source_log"]:
        raise ValueError("DB does not match the source log")
    ids = scene.actor_ids
    if len(set(pair)) != 2 or not all(a in ids for a in pair):
        raise ValueError("Require two distinct scene actor IDs")
    audit = source_entry_overlaps(source)
    start_us = source.metadata["nuplan_source"]["anchor_timestamp_us"]
    end_us = start_us + round(source.times[-1] * 1e6)
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT p.timestamp,a.x,a.y,a.yaw,a.length,a.width,a.vx,a.vy,"
            "b.x,b.y,b.yaw,b.length,b.width,b.vx,b.vy "
            "FROM lidar_box a JOIN lidar_box b ON a.lidar_pc_token=b.lidar_pc_token "
            "JOIN lidar_pc p ON a.lidar_pc_token=p.token "
            "WHERE a.track_token=? AND b.track_token=? AND p.timestamp BETWEEN ? AND ? ORDER BY p.timestamp",
            (*tuple(bytes.fromhex(a) for a in pair), start_us, end_us),
        ).fetchall()
    if not rows:
        raise ValueError("No simultaneous raw annotations in this scene window")
    raw = []
    for row in rows:
        a, b = row[1:8], row[8:15]
        raw.append({"timestamp_us": row[0],
                    "overlap": bool(rectangles_overlap(rectangle_corners(*a[:5]), rectangle_corners(*b[:5]))),
                    "centers_m": [list(a[:2]), list(b[:2])],
                    "length_width_m": [list(a[3:5]), list(b[3:5])],
                    "planar_speeds_mps": [float(np.hypot(*a[5:])), float(np.hypot(*b[5:]))]})
    archive = case / "rollout.npz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with np.load(archive, allow_pickle=False) as data:
        life = json.loads(str(data["lifecycle"]))
        diagnostics = json.loads(str(data["controller_diagnostics"]))
        valid = data["valid"]
        controlled = data["controlled_ids"].tolist()
    intervals = dependency_intervals(life, diagnostics, valid, controlled)
    report = {"source_log": source.metadata["source_log"], "source_db": str(db),
              "raw_time_window_us": [start_us, end_us],
              "raw_pair": pair, "raw_pair_numbers": [ids.index(a)+1 for a in pair],
              "raw_common_frames": len(raw), "raw_overlap_frames": sum(x["overlap"] for x in raw),
              "raw_frames": raw, "source_integrity": audit,
              "controller_dependency_intervals": intervals,
              "interpretation": "Recorded conflict/lead dependencies are not established causal fault. Raw overlaps do not prove duplicate IDs.",
              "simulated_motion_changed": False, "trainable": False, "rollout_sha256": digest}
    output = case / "blockage-report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    inspection = {"kind": "continuous_nuplan", "gallery": True,
                  "note": "此处是控制器仿真记录。请结合源框入场检查与控制器关系检查，未通过交通真实性验收。",
                  "paired_view": {"path": "source.html", "label": "查看同对象、同时刻的源轨迹"}}
    export_replay(scene, archive, case / "viewer.html", inspection=inspection, compress_payload=True)
    export_replay(scene, case / "baseline.npz", case / "anchor.html",
                  compare_path=case / "brake.npz", inspection=inspection, compress_payload=True)
    export_source_replay(source, case / "source.html", inspection={
        "note": "同时间范围源轨迹，10 Hz 编译插值；不运行驾驶控制器。源框重叠来自标注几何，不直接判定事故。",
        "paired_view": {"path": "viewer.html", "label": "查看同对象、同时刻的控制器仿真"}})
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    source_events = [e for e in audit["events"] if set(e["actors"]) == set(pair)]
    first = source_events[0]["tick"] if source_events else 0
    late = min(first + 50, len(source.times)-1)
    final = len(source.times)-1
    def link(page, tick, actor, label):
        return f'<a href="{page}?actor={actor}&tick={tick}">{label}</a>'
    links = " · ".join([link("source.html", first, pair[0], "源框首次重叠" if source_events else "起始源轨迹（未检出入场重叠）"),
                        link("viewer.html", first, pair[0], "同一时刻的仿真"),
                        link("source.html", late, pair[0], "稍后源轨迹"),
                        link("viewer.html", late, pair[0], "稍后仿真状态"),
                        link("viewer.html", max(0, final-10), "ego", "末段控制关系"),
                        link("anchor.html", max(0, final-10), "ego", "末段干预对照")])
    rows_html = "".join(f'<tr><td>{ids.index(e["actor"])+1} → {ids.index(e["target"])+1}</td>'
                        f'<td>{link("viewer.html", e["start_tick"], e["actor"], str(e["start_tick"]/10)+"s")}</td>'
                        f'<td>{e["end_tick_exclusive"]/10:g}s（不含）</td><td>{html.escape(str(e["reason"]))}</td></tr>'
                        for e in intervals if e["end_tick_exclusive"]-e["start_tick"]>=10)
    page = f"""<h1>这次不是原始日志回放：源框与控制器阻塞诊断</h1>
<p>原始日志只显示记录的位置；控制器仿真重新计算车辆运动。请分别检查两种模式，不以记录可重复证明驾驶安全。</p>
<nav>{links} · <a href="blockage-report.json">逐帧证据</a></nav>
<h2>原始标注核查</h2><p>第{ids.index(pair[0])+1}号和第{ids.index(pair[1])+1}号对象在原始数据库中同时出现{len(raw)}帧，其中{sum(x['overlap'] for x in raw)}帧的逐帧包围盒重叠。使用原始逐帧尺寸独立核对，不依赖仿真的固定框。重叠仍不等于重复身份或真实事故；对象未被删除或合并。</p>
<h2>核对控制器关系与生命周期</h2><p>记录保留每辆车报告的前车与冲突目标，用于检查互相制动和等待链。源标注结束后，平台仍保留已接管的对象。仿真堵塞不能直接当作真实日志堵塞，也不能直接标作驾驶模型失败。</p>
<p>下面展示控制器实际报告的前车/冲突关系，累计不足1秒的关系仍保存在JSON中。关系记录不是责任归因；控制器和背景路线是否合理需单独验收。本次没有改动既有物理轨迹、碰撞规则或模型。</p>
<table><tr><th>对象编号</th><th>开始</th><th>结束</th><th>记录类型</th></tr>{rows_html}</table>
<p>trainable=false。未计算官方PDM。没有通过忽略碰撞或删除对象制造通行结果。</p>"""
    (case / "blockage.html").write_text(shell(page), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("raw_frames", "controller_dependency_intervals", "source_integrity")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--pair", nargs=2, required=True)
    a = p.parse_args()
    diagnose(a.case, a.db, a.pair)
