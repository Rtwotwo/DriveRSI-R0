"""Publish evidence and existing BEV players for a completed collection task."""

import argparse
import html
import json
from pathlib import Path


def export(root, task):
    folder = root / "tasks" / task / "collect"
    report = json.loads((folder / "report.json").read_text())
    reached = json.loads((folder / "candidate/recovery-task.json").read_text())
    objects = report["branches"]["candidate"]["all_world_objects"]
    history = "、".join(f"{tick * 0.1:.1f}" for tick in reached["history_ticks"])
    future = reached["available_future_ticks"] * 0.1
    rows = []
    for branch, label in (("parent", "父模型"), ("candidate", "更新后模型（未晋级）")):
        row = report["branches"][branch]
        href = f"tasks/{html.escape(task)}/collect/{branch}"
        rows.append(
            f"<tr><td>{label}</td><td>{row['steps']}</td><td>{len(row['contacts'])}</td>"
            f'<td><a href="{href}/viewer.html?actor=ego&tick=20">播放实际执行</a> · '
            f'<a href="{href}/recovery-task.json">到达状态与历史</a></td></tr>'
        )
    comparison = report["comparison"]
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>DriveRSI：学习模型回到采集环节</title><style>
    body{{max-width:1050px;margin:45px auto;padding:0 22px;font:17px/1.7 system-ui;background:#f3f6fa;color:#183047}}
    table{{border-collapse:collapse;width:100%;background:white}}td,th{{text-align:left;padding:14px;border-bottom:1px solid #dce4eb}}
    a{{color:#087d87}}.notice{{background:#fff2d8;padding:18px;border-radius:10px}}code{{overflow-wrap:anywhere}}
    </style><h1>更新后的模型 → 实际到达状态</h1>
    <p>真实 nuPlan Pittsburgh，同一初态，{objects} 个世界对象。此页验证采集和恢复，不评价驾驶性能收益。</p>
    <p class="notice">候选未晋级 · 规则交通，非学习型 Self-Play · 无新训练 · 无教师或视觉准入。trainable=false。</p>
    <table><tr><th>策略</th><th>执行步数（10 Hz）</th><th>接触状态数</th><th>证据</th></tr>{"".join(rows)}</table>
    <p>父模型与更新后模型有 {comparison["different_action_steps"]} 步动作不同。另一进程恢复第 {reached["anchor_tick"]} 步完整世界与模型记忆，后续 {report["restore"]["steps"]} 步精确一致。</p>
    <p>四帧历史：{history} 秒；待纠正锚点：{reached["anchor_tick"] * 0.1:.1f} 秒。已有 {future:.1f} 秒实际后续用于诊断，不自动作为高分教师标签。</p>
    <p>本轮是“训练产物重新参与采集”的连接验证。尚需：失败锚点自动选择、同状态可靠教师纠正、分支评分、按需渲染与质量验收、独立收益判断及通过后的策略晋级。</p>
    <p><a href="tasks/{html.escape(task)}/collect/report.json">完整采集结果</a> · <a href="tasks/{html.escape(task)}/collect/restore.json">跨进程恢复</a> · <a href="reactive-check/recovery-task.json">Boston 扰动到达状态</a> · <a href="../harness-skills-v1/index.html">已有搜索、GS 与配对训练</a></p>
    </html>"""
    (root / "index.html").write_text(page)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--task", default="learned-recovery-v2")
    args = p.parse_args()
    export(args.root.resolve(), args.task)
