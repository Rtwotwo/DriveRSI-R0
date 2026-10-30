"""Publish a read-only evidence view of synthesis and matched learning tasks."""

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.rsi.workflow.artifacts import verify
from driversi.rsi.workflow.store import HarnessStore
from driversi.runtime.budget import write_json


def publish(registry, learning_task, reactive_task, output):
    output = Path(output).resolve()
    output.relative_to((project_root() / "exp").resolve())
    output.mkdir(parents=True, exist_ok=True)
    store = HarnessStore(Path(registry))
    states = [store.get(name) for name in (learning_task, reactive_task)]
    reports = {}
    for state in states:
        declarations = {x["id"]: x for x in state["task"]["steps"]}
        for stage in state["stages"]:
            if stage["status"] != "completed":
                continue
            verify(stage["result"], state["id"], declarations[stage["id"]])
            reports[(state["id"], stage["id"])] = stage["result"]

    def link(path, label):
        path = Path(path).resolve()
        path.relative_to((project_root() / "exp").resolve())
        return (
            '<a href="'
            + quote(os.path.relpath(path, output), safe="/")
            + '">'
            + html.escape(label)
            + "</a>"
        )

    sections = [
        "<p>表内 agent steps 只计仿真与评分步骤，训练并非零成本。优化器更新、加载和检查点耗时见各阶段回执；复用搜索的历史成本仍保留。</p>"
    ]
    if (output / "cost-accounting.json").is_file():
        sections.append(
            "<p>" + link(output / "cost-accounting.json", "查看全部尝试、训练与失败费用") + "</p>"
        )
    for state in states:
        rows = []
        for stage in state["stages"]:
            ref = reports.get((state["id"], stage["id"]))
            evidence = (
                link(ref["artifacts"]["report"], "原始证据")
                if ref
                else html.escape(str(stage.get("reason") or ""))
            )
            rows.append(
                "<tr><td>"
                + html.escape(stage["id"])
                + "</td><td>"
                + html.escape(stage["status"])
                + "</td><td>"
                + str(stage.get("charged_agent_steps", 0))
                + "</td><td>"
                + evidence
                + "</td></tr>"
            )
        sections.append(
            "<h2>"
            + html.escape(state["id"])
            + " · "
            + html.escape(state["status"])
            + "</h2><table><tr><th>阶段</th><th>状态</th><th>计费 agent steps</th><th>证据</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    search = reports.get((learning_task, "search"))
    if search:
        data = read_mapping(search["artifacts"]["diagnostics"])
        rows = []
        for row in data["cases"]:
            values = [
                row["controls"][name]["score"]
                for name in (
                    "expert_gt",
                    "zero_offset_reconstruction",
                    "zero_offset_gt_heading_diagnostic_only",
                )
            ]
            cells = [
                html.escape(row["token"]),
                *[f"{v * 100:.4f}" for v in values],
                f"{row['best_score'] * 100:.4f}",
            ]
            rows.append("<tr>" + "".join("<td>" + c + "</td>" for c in cells) + "</tr>")
        sections.append(
            "<h2>同锚点新搜索与诊断对照</h2><p>原生 NAVSIM v1 单候选评分，百分制展示。GT 航向仅作离线诊断，不能当作在线策略。复用搜索的任务本次新增查询为 0，原始搜索成本仍保留。</p><table><tr><th>训练 token</th><th>原 GT</th><th>零偏移构造</th><th>恢复 GT 航向</th><th>新候选最佳</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    feedback = reports.get((learning_task, "feedback"))
    if feedback:
        data = read_mapping(feedback["artifacts"]["report"])
        rows = []
        control = read_mapping(reports[(learning_task, "control-eval")]["artifacts"]["report"])[
            "metrics"
        ]
        treatment = read_mapping(reports[(learning_task, "treatment-eval")]["artifacts"]["report"])[
            "metrics"
        ]
        for metric in ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C"):
            values = [
                control["parent"][metric],
                control["candidate"][metric],
                treatment["candidate"][metric],
            ]
            rows.append(
                "<tr><td>"
                + metric
                + "</td>"
                + "".join(f"<td>{v * 100:.5f}</td>" for v in values)
                + "</tr>"
            )
        sections.append(
            "<h2>配对训练与独立开发评测</h2><p>同父模型、原生损失与更新预算；开发样本与本轮训练原始日志隔离。父模型训练来源未知，微型试验不能证明泛化或授权晋级。</p><table><tr><th>指标</th><th>父模型</th><th>普通续训</th><th>新标签续训</th></tr>"
            + "".join(rows)
            + "</table><pre>"
            + html.escape(
                json.dumps(
                    {
                        k: data[k]
                        for k in (
                            "observed_pilot_benefit",
                            "promoted",
                            "resume_check",
                            "next_round",
                        )
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre>"
        )

    rendered = reports.get((reactive_task, "render"))
    package = reports.get((reactive_task, "package"))
    if rendered:
        sections.append(
            "<h2>真实多车执行分支 → GS → NAVSIM 检查包</h2><p>"
            + link(rendered["artifacts"]["viewer"], "播放选中分支的自车三相机")
            + "</p><p>重新执行记录动作并逐状态核对。未来画面供时序检查；训练检查包只导出四帧历史。该分支没有独立教师资格，GS 质量门槛未放宽。</p>"
        )
        for image in sorted(
            Path(rendered["artifacts"]["session"]).parent.glob("generation-*/0020-*-image.png")
        ):
            if "virtual" in image.name:
                continue
            rel = quote(os.path.relpath(image, output), safe="/")
            sections.append(
                '<figure><img src="'
                + rel
                + '"><figcaption>'
                + html.escape(image.name)
                + "</figcaption></figure>"
            )
    if package:
        data = read_mapping(package["artifacts"]["report"])
        sections.append(
            "<pre>"
            + html.escape(
                json.dumps(
                    {k: data.get(k) for k in ("trainable", "image_counts", "rejection_reasons")},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre>"
        )
    summary = {
        "tasks": [
            {k: s[k] for k in ("id", "status", "sim_agent_steps", "failure_reason")} for s in states
        ],
        "official_model_performance_claim": False,
        "automatic_training": False,
        "promoted": False,
    }
    write_json(output / "summary.json", summary)
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DriveRSI · 技能与合成训练证据</title><style>body{font:16px/1.6 system-ui;background:#edf2f7;color:#152b40;max-width:1160px;margin:30px auto;padding:0 22px}h1{font-size:30px}h2{margin-top:36px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;border-bottom:1px solid #dae1e8;text-align:left;overflow-wrap:anywhere}a{color:#006c80}pre{white-space:pre-wrap;background:#fff;padding:18px}figure{display:inline-block;width:30%;margin:1%}img{width:100%;display:block}.notice{border-left:5px solid #e0a21b;padding:15px;background:#fff5db}small{color:#506475}</style><h1>DriveRSI · 可执行技能与合成训练</h1><p>新搜索 → 独立复核 → 历史/数据版本 → 普通续训对照 → 独立评测 → 收益反馈</p><p class="notice">这是有上限的真实样本验证。任务完成、轨迹高分、视觉合格、学生获益是不同结论；不自动晋级、不触发下一轮大规模训练。</p>"""
    (output / "index.html").write_text(
        document
        + "".join(sections)
        + "<p><small>所有记录均来自上述任务的封存产物；失败尝试保留于同一实验目录。</small></p></html>",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--learning-task", required=True)
    parser.add_argument("--reactive-task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.registry, args.learning_task, args.reactive_task, args.output)))


if __name__ == "__main__":
    main()
