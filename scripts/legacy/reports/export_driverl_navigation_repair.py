"""Publish same-anchor correction evidence, never merge it with native benchmark scores."""
import html
import json
import os
from pathlib import Path

import numpy as np
from driversi.rendering.inspection.driverl import overview
from shapely.geometry import Polygon

from driversi.contracts.scene import Scene
from driversi.evaluation.metrics.geometry import rectangle_corners

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/demonstrations/driverl-navigation-repair-v2"
BASE = ROOT / "outputs/demonstrations/driverl-multicity-v1"
BLOCKER = "3722656fb17e585a"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def measure(folder, parent, start):
    scene = Scene.load(folder / "scene.npz")
    previous_scene = Scene.load(parent / "scene.npz")
    assert scene.actor_ids == previous_scene.actor_ids
    with np.load(folder / "rollout.npz", allow_pickle=False) as z, np.load(parent / "rollout.npz", allow_pickle=False) as old:
        prefix = {}
        for key in ("states", "motion", "presence", "actions", "rewards", "valid", "terminated", "truncated", "collision", "offroad", "goal", "route_support_end"):
            count = start + 1 if key in ("states", "motion", "presence") else start
            prefix[key] = bool(np.array_equal(z[key][:count], old[key][:count]))
        assert all(prefix.values()), "Correction prefix differs from the fixed parent"
        states, presence = z["states"], z["presence"]
        i = scene.actor_ids.index("ego")
        result = dict(prefix_exact=prefix, same_actor_ids=True, states=len(states),
            terminal_speed_mps=float(np.linalg.norm(states[-1, i, 4:6])),
            path_length_m=float(np.linalg.norm(np.diff(states[:, i, :2], axis=0), axis=1).sum()),
            correction_path_length_m=float(np.linalg.norm(np.diff(states[start:, i, :2], axis=0), axis=1).sum()))
        if BLOCKER in scene.actor_ids:
            j = scene.actor_ids.index(BLOCKER)
            ticks = np.flatnonzero(presence[:, i] & presence[:, j])
            axis = np.array([np.cos(states[start, i, 3]), np.sin(states[start, i, 3])])
            signed = (states[ticks, j, :2] - states[ticks, i, :2]) @ axis
            gaps = [Polygon(rectangle_corners(*states[t, i, :2], states[t, i, 3], *scene.dimensions[i, :2])).distance(
                    Polygon(rectangle_corners(*states[t, j, :2], states[t, j, 3], *scene.dimensions[j, :2]))) for t in ticks]
            result["blocker"] = dict(actor=BLOCKER, present_at_end=bool(presence[-1, j]),
                presence_identical_to_parent=bool(np.array_equal(presence[:, j], old["presence"][:, j])),
                first_tick=int(ticks[0]), last_tick=int(ticks[-1]), min_rectangle_gap_m=float(min(gaps)),
                forward_projection_at_end_m=float(signed[-1]),
                projection_definition="blocker minus ego, projected on ego heading at intervention; negative means behind this fixed axis, not independent driving qualification")
    decisions = read(folder / "decisions.json")
    repairs = [d for d in decisions if d.get("repair_active")]
    result["repair_attempts"] = len(repairs)
    result["no_admissible_proposal_ticks"] = [d["tick"] for d in repairs if d["repair_evidence"]["selected_proposal"] is None]
    result["commands_finite"] = bool(np.isfinite([d["executed_action"] for d in repairs]).all())
    history = read(folder / "native-history.json")
    result["scored_history_seconds"] = (history[-1]["ego"][-1] - history[0]["ego"][-1]) / 1e6
    result["actual_acceleration_range"] = [float(x) for x in (min(v["ego"][5] for v in history), max(v["ego"][5] for v in history))]
    return result


def main():
    cases, sections = [], []
    for log, title, version, start in [(4, "Pittsburgh · 方向零分", "route-v4", 280), (5, "Singapore · 持续障碍", "map-v5", 260)]:
        parent = BASE / f"log-{log:02}" / version
        folder = OUT / f"log-{log:02}-guarded-v4"
        if not (folder / "official-score.json").exists():
            raise RuntimeError(f"Incomplete correction evidence: {folder}")
        report, score = read(folder / "report.json"), read(folder / "official-score.json")
        evidence = measure(folder, parent, start)
        case = dict(log=log, title=title, correction=folder.name, intervention_tick=start,
            report=report, official=score, execution_checks=evidence, trainable=False,
            protocol="custom_60s_hybrid_correction_not_native_15s_benchmark")
        case["outcome"] = ("direction_correction_passed_on_this_case_only" if log == 4 else "unresolved_bypass_stopped_before_obstacle")
        cases.append(case)
        official_dir = BASE / f"log-{log:02}" / ("official-v2" if log == 4 else "official")
        source_score = read(ROOT / f"outputs/demonstrations/driverl-failure-repair-v1/log-{log:02}/source-score.json")
        rows = [("源日志专家 / 原始交通", source_score, None, None),
                ("官方 DriveRL / IDM 交通", read(official_dir / "official-score.json"), None, official_dir),
                ("DriveRSI / 修复前", read(parent / "official-score.json"), read(parent / "report.json"), parent),
                ("DriveRSI / 原生 64 候选搜索", read(BASE / f"log-{log:02}/native-search64-v2/official-score.json"), read(BASE / f"log-{log:02}/native-search64-v2/report.json"), BASE / f"log-{log:02}/native-search64-v2"),
                ("DriveRSI / 独立 Harness 纠正", score, report, folder)]
        table = []
        for label, sc, rp, path in rows:
            m = sc.get("metric_scores", {}) if sc else {}
            cells = [label, "待评分" if not sc or sc.get("score") is None else f"{sc['score']*100:.4f}"]
            cells += ["—" if m.get(k) is None else f"{m[k]*100:.2f}" for k in ("driving_direction_compliance", "drivable_area_compliance", "no_ego_at_fault_collisions", "time_to_collision_within_bound", "ego_progress_along_expert_route")]
            cells += ["—" if rp is None else str(rp["contacts"]["owner_contact_ticks"])]
            link = "—" if path is None or not (path / "viewer.html").exists() else '<a href="' + os.path.relpath(path / 'viewer.html', OUT) + f'?actor=ego&amp;tick={start}">事件回放</a>'
            table.append("<tr>" + "".join("<td>"+html.escape(c)+"</td>" for c in cells) + "<td>"+link+"</td></tr>")
        scene = Scene.load(folder / "scene.npz")
        paths = []
        for label, color, path in [("修复前", "#ff7276", parent), ("Harness纠正", "#3de0bb", folder)]:
            with np.load(path / "rollout.npz", allow_pickle=False) as z:
                paths.append((label, color, z["states"][:, scene.actor_ids.index("ego"), :2]))
        note = ("交叉口路径与反向连接线关联导致官方方向扣分；评分器未修改。该分支测试地图约束纠正，不是 DriveRL 自身已学会。" if log == 4 else "绕行尚未解决：障碍车辆持续存在，自车最终停在其前方，车身最小间距约 0.75 米。此分支避免了碰撞，但不是绕行成功。官方 IDM 未接管该障碍，因此官方运行时的满分不能直接证明本场景的绕障能力。")
        sections.append(f'<section><h2>{title}</h2><p>{note}</p><table><tr><th>执行来源</th><th>总分</th><th>方向</th><th>道路</th><th>无责任碰撞</th><th>TTC</th><th>进度</th><th>ego 接触帧</th><th>查看</th></tr>'+''.join(table)+'</table><p>红：修复前；绿：独立纠正。其他物体也随当前世界响应。</p>'+overview(scene,paths)+f'<details><summary>同锚点、障碍和实际执行核验</summary><pre>{html.escape(json.dumps(evidence,ensure_ascii=False,indent=2))}</pre></details><p><a href="{folder.name}/index.html">完整执行证据</a> · <a href="{folder.name}/official-score.json">官方原始评分</a></p></section>')
    (OUT / "comparison.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2, allow_nan=False))
    (OUT / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>DriveRL 同锚点纠正验收</title><style>body{font:16px/1.7 system-ui;background:#edf3f6;color:#173849;max-width:1400px;margin:30px auto;padding:20px}section{background:white;padding:24px;margin:24px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}aside{background:#fff1c8;padding:20px}svg{width:100%;max-height:480px}a{color:#006978}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>DriveRL · 同锚点地图约束纠正</h1><aside>独立 Harness 规划与跟踪器接管明确时间段；DriveRL 原生影子输出仍逐步核对。自定义 60 秒世界，不是论文 15 秒原生基准；官方评分器、碰撞与障碍不改。手工选择干预时机，未训练，trainable=false。评分记录覆盖至 59.9 秒，物理接触另检查完整 60 秒。</aside><p><a href="../driverl-benchmark-v1/index.html">正式基准：论文逐项比较</a> · <a href="comparison.json">完整证据</a></p>'+''.join(sections)+'</html>',encoding='utf-8')
    print(json.dumps([dict(log=c["log"],score=c["official"].get("score"),checks=c["execution_checks"]) for c in cases],ensure_ascii=False))


if __name__ == "__main__":
    main()
