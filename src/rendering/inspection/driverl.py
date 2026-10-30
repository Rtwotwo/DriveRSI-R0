"""Evidence index for a completed native DriveRL run; no synthetic performance claim."""

import html
import json
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene


def export_driverl_gallery(output: Path):
    report = json.loads((output / "report.json").read_text())
    scene = Scene.load(output / "scene.npz")
    score_path = output / "official-score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    score_text = (
        f"nuPlan官方指标（自定义长日志）: {score['score'] * 100:.2f}"
        if score.get("status") == "scored"
        else "官方评分未完成：" + str(score.get("error", "未运行"))
    )
    parent_link = (
        '<a href="../../index.html">四城市总览</a>' if output.parent.name.startswith("log-") else ""
    )
    repair = report.get("correction_backend")
    executed = report.get("native_executed_steps", report["inference_steps"])
    title = "DriveRL + Harness 地图约束纠正" if repair else "DriveRL 已实际控制自车"
    subtitle = ("显式纠正分支；纠正期间原生模型仅作影子对照，不计作原生成绩。"
                if repair else "官方观测 → 官方策略 → 官方连续动力学 → 同一多车物理世界")
    boundary = report.get("road_boundary_contact_steps")
    boundary_text = "未记录" if boundary is None else str(boundary)
    end = report["terminal_events"]["route_support_end"]
    reason = (
        f"自车于 {end[0] / 10:.1f} 秒到达路线支持边界，随后停车；背景交通继续。"
        if end
        else "请查看终止与碰撞记录；运行完成不等于驾驶合格。"
    )
    links = " ".join(
        f'<a href="viewer.html?actor=ego&tick={k}">{k / 10:.1f}s</a>'
        for k in sorted(set([0, 180, 348, 450, *end, report["steps"]]))
    )
    (output / "index.html").write_text(
        f"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 连续闭环实测</title><style>
body{{margin:0;background:#eef3f7;color:#173245;font:16px/1.7 system-ui}}main{{max-width:1100px;margin:40px auto;padding:0 24px}}
header{{padding:32px;background:#102c3e;color:white;border-radius:20px}}h1{{font-size:32px;margin:10px 0}}.tag{{color:#7ee0d3;letter-spacing:2px;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:22px 0}}article{{background:white;padding:24px;border-radius:14px;border:1px solid #d9e2e9}}b.metric{{display:block;font-size:32px;color:#007c79}}a{{display:inline-block;padding:7px 15px;margin:4px;border-radius:7px;background:#e1efef;color:#075d60;text-decoration:none}}.note{{border-left:4px solid #c17d10;background:#fff7e8;padding:18px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dbe4eb}}code{{overflow-wrap:anywhere;font-size:13px}}
</style><main><header><div class="tag">{"HARNESS CORRECTION" if repair else "NATIVE MODEL"} / REAL NUPLAN / NO TRAINING</div><h1>{title}</h1><p>{subtitle}</p><code>{html.escape(scene.metadata["source_log"])}</code></header>
<div class="grid"><article>物理世界持续推进<b class="metric">{report["seconds"]:.1f} s</b>{report["steps"]} 次联合推进</article><article>有效模型控制<b class="metric">{executed / 10:.1f} s</b>原生执行 {executed} 步；纠正执行 {report.get("repair_steps", 0)} 步</article><article>同输入原生输出误差<b class="metric">{report["native_output_max_abs_error"]:.0f}</b>原生影子动作 + 原生单步状态；不替代纠正分支验收</article><article>几何接触状态数<b class="metric">{report["contacts"]["owner_contact_ticks"]} / {report["contacts"]["other_contact_ticks"]}</b>ego / 其他对象；不是官方PDM</article></div>
<p class="note">{reason} 模型计算对照通过，不代表完整基准性能已复现。没有训练、晋级或放宽图像准入。</p>
<article><h2>实际播放与事件定位</h2><a href="viewer.html?actor=ego&tick=0">播放 DriveRL 闭环</a><a href="source.html">源日志对照</a>{parent_link}<p>{links}</p><p>车辆及行人来自本日志；自车停止后不再显示待执行的模型动作。此处为 BEV，不是 GS 渲染。</p></article>
<article style="margin-top:20px"><h2>核验边界</h2><table><tr><th>检查</th><th>实测</th></tr>
<tr><td>原始 nuPlan 对象 → 适配输入</td><td>{len(report["source_input_parity"]["compared_tensor_fields"])} 个特征张量 / {report["source_input_parity"]["compared_values"]:,} 数值完全一致</td></tr>
<tr><td>初态位置/航向数值差上限</td><td>{report["initial_conversion"]["max_pose_error"]:.3g}（坐标米 / 航向弧度）</td></tr>
<tr><td>初态速度转换误差</td><td>{report["initial_conversion"]["max_velocity_error"]:.3g} m/s</td></tr>
<tr><td>执行后位置最大舍入</td><td>{report["maximum_world_position_rounding_m"]:.3g} m</td></tr>
<tr><td>恢复后重放</td><td>{report["restore_steps"]} 步状态、动作、历史一致：{report["restore_exact"]}</td></tr>
<tr><td>道路边界配置 / 周界接触帧数</td><td>{html.escape(report.get("road_boundary_policy", "perimeter_stop_legacy"))} / {boundary_text}；与越界终止、物体碰撞分别记录</td></tr>
<tr><td>官方指标 / 完整基准</td><td>{html.escape(score_text)}；不是 NAVSIM PDMS / 官方榜单；trainable=false</td></tr></table>
<p><a href="report.json">完整结果</a><a href="official-score.json">官方评分结果</a><a href="decisions.json">逐步原生对照</a><a href="execution-evidence.json">接触与执行证据</a></p></article></main></html>""",
        encoding="utf-8",
    )


def overview(scene, paths):
    all_xy = np.concatenate([xy for _, _, xy in paths])
    low, high = all_xy.min(0) - 20, all_xy.max(0) + 20
    scale = min(900 / (high[0] - low[0]), 430 / (high[1] - low[1]))

    def points(xy):
        values = (xy - low) * scale
        return " ".join(f"{x + 20:.2f},{450 - y:.2f}" for x, y in values)

    bits = [
        '<svg viewBox="0 0 960 480" role="img" aria-label="同一来源的自车路径对照"><rect width="960" height="480" fill="#102333"/>'
    ]
    for a, b, kind in zip(scene.road_offsets[:-1], scene.road_offsets[1:], scene.road_types):
        xy = scene.road_points[a:b, :2]
        if np.any(np.all((xy >= low) & (xy <= high), axis=1)):
            bits.append(
                f'<polyline points="{points(xy)}" fill="none" stroke="#39515f" stroke-width="0.7"/>'
            )
    for label, color, xy in paths:
        bits.append(
            f'<polyline points="{points(xy)}" fill="none" stroke="{color}" stroke-width="2.5"><title>{html.escape(label)}</title></polyline>'
        )
    bits.append("</svg>")
    return "".join(bits)
