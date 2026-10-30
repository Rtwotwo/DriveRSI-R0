"""Original scores, changed worlds and correction attempts remain separate."""

import html
import json
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.rendering.inspection.driverl import overview

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/demonstrations/driverl-multicity-v1"
OUT = ROOT / "outputs/demonstrations/driverl-failure-repair-v1"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def number(value):
    return "待完成" if value is None else f"{100 * value:.2f}"




def main():
    OUT.mkdir(parents=True, exist_ok=True)
    definitions = [
        (
            4,
            "Pittsburgh · 行驶方向",
            "route-v4",
            "native-search-v1",
            320,
            "官方方向指标在交叉口选择并持续关联反向连接线18486；实际驶入相邻连接线。零分保留，不能只凭负向投影认定自车倒车，也不能据此撤销官方扣分。8和64候选搜索均未解决方向扣分。",
        ),
        (
            5,
            "Singapore · 静止车辆接触",
            "map-v5",
            "native-search-v1",
            290,
            "3722656fb17e585a仅有5条/约0.2秒零速标注，平台持续保留，官方IDM没有该对象。场景状态不同，不能将分差全部归为观测适配损伤；不通过删车消除碰撞。本轮64候选已避免接触，但约29秒后停在障碍前，未完成绕行；官方进度81.73%。",
        ),
        (
            6,
            "Singapore · 道路边界",
            "map-v5",
            "corners-v6",
            203,
            "地图改用官方道路表面。新显式物理模式保留边缘接触事件，但按四角零容差判定越界；车辆碰撞仍终止。官方评分及阈值不改。此前整车边缘触线停车与官方角点检查不等价。本次完成60秒但仍有3帧周界接触，官方TTC未通过，尚未成为合格数据。",
        ),
    ]
    cases, sections = [], []
    for idx, title, before, after, tick, reason in definitions:
        d = BASE / f"log-{idx:02}"
        source_score = read(OUT / f"log-{idx:02}/source-score.json")
        official_name = (
            "official-v2" if (d / "official-v2/official-score.json").exists() else "official"
        )
        official = read(d / official_name / "official-score.json")
        runs = []
        for label, version in [
            ("最初平台", "final" if idx == 4 else "closed-loop"),
            ("修复前固定分支", before),
            ("本轮8候选" if idx != 6 else "本轮边界修复", after),
            *(([("本轮64候选", "native-search64-v2")]) if idx != 6 else []),
        ]:
            score = read(d / version / "official-score.json")
            report = read(d / version / "report.json")
            runs.append(
                dict(
                    label=label,
                    version=version,
                    score=None if score is None else score.get("score"),
                    status="pending"
                    if report is None
                    else "scoring_or_export_pending"
                    if score is None
                    else score.get("status"),
                    inference_steps=None if report is None else report.get("inference_steps"),
                    metric_scores=None if score is None else score.get("metric_scores"),
                    contacts=None if report is None else report["contacts"]["owner_contact_ticks"],
                    terminals=None if report is None else report.get("terminal_events"),
                    boundary_contacts=None
                    if report is None
                    else report.get("road_boundary_contact_steps"),
                    search=None if report is None else report.get("correction_search"),
                )
            )
        case = dict(
            log=idx,
            reason=reason,
            source_score=None if source_score is None else source_score["score"],
            native_score=None if official is None else official["score"],
            runs=runs,
            protocol="custom_60s_official_nuPlan_metrics_not_paper_benchmark",
            trainable=False,
        )
        cases.append(case)
        source_folder = d / before
        if not (source_folder / "scene.npz").exists():
            source_folder = d / ("route-v4" if idx == 4 else "motion-v3")
        scene = Scene.load(source_folder / "scene.npz")
        source = Scene.load(source_folder / "source.npz")
        paths = [("原始日志", "#c6d0d5", source.poses[:, source.actor_ids.index("ego"), :2])]
        origin = np.asarray(scene.metadata["origin_xyz"][:2])
        for label, color, folder in [
            ("官方DriveRL+IDM", "#f5c14b", d / official_name),
            ("平台修复前", "#ff7276", d / before),
            ("8候选" if idx != 6 else "边界修复", "#3de0bb", d / after),
            *(([("64候选", "#c797ff", d / "native-search64-v2")]) if idx != 6 else []),
        ]:
            rows = read(folder / "native-history.json")
            if rows:
                xy = (
                    np.array(
                        [
                            [
                                v["ego"][0] + 1.461 * np.cos(v["ego"][2]),
                                v["ego"][1] + 1.461 * np.sin(v["ego"][2]),
                            ]
                            for v in rows
                        ]
                    )
                    - origin
                )
                paths.append((label, color, xy))
        table = f"<tr><td>原始日志专家轨迹</td><td>{number(case['source_score'])}</td><td>原始交通</td></tr><tr><td>官方DriveRL＋IDM</td><td>{number(case['native_score'])}</td><td>官方原生交通</td></tr>"
        links = []
        for run in runs:
            link = f"../driverl-multicity-v1/log-{idx:02}/{run['version']}"
            if (d / run["version"] / "viewer.html").exists():
                links.append(
                    f'<a href="{link}/viewer.html?actor=ego&amp;tick={tick}">{run["label"]}回放</a>'
                )
            table += f"<tr><td>{run['label']} ({run['version']})</td><td>{number(run['score'])}</td><td>{html.escape(json.dumps({k: run[k] for k in ['status', 'inference_steps', 'contacts', 'terminals', 'boundary_contacts']}, ensure_ascii=False))}</td></tr>"
        sections.append(
            f"<section><h2>{title}</h2><p>{reason}</p><p>{' · '.join(links)}</p><table><tr><th>来源</th><th>分数 /100</th><th>执行证据（接触为状态数）</th></tr>{table}</table><p>灰：原始日志；黄：官方IDM；红：平台修复前；绿：8候选／边界修复；紫：64候选。只叠加自车路径；交通世界不同，详见各分支回放。</p>{overview(scene, paths)}</section>"
        )
    (OUT / "comparison.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False, allow_nan=False)
    )
    (OUT / "index.html").write_text(
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>DriveRL 失败修复与原始分数</title><style>
body{font:16px/1.7 system-ui;background:#edf3f6;color:#173849;max-width:1300px;margin:30px auto;padding:20px}
section{background:white;padding:24px;margin:24px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}td:last-child{font-size:13px;overflow-wrap:anywhere}aside{background:#fff1c8;padding:20px}svg{width:100%;max-height:480px}a{color:#006978}
</style><h1>DriveRL · 失败修复与原始分数</h1><aside>全部为固定60秒诊断，评分使用未修改的官方nuPlan指标。原始日志、官方IDM、DriveRSI响应式世界分别列出。TTS候选单独标注，不作为“无TTS适配性能保持”的证据；未完成不填零。未训练，样本未准入。</aside><p><a href="../driverl-benchmark-v1/index.html">正式15秒基准与论文逐项比较</a> · <a href="comparison.json">完整分数与失败记录</a></p>"""
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )
    print(json.dumps(cases, ensure_ascii=False))


if __name__ == "__main__":
    main()
