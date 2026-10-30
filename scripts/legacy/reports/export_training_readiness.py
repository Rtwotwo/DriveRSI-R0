"""Publish bounded ChainFlow readiness evidence without turning probes into benchmarks."""

import argparse
import html
import json
from pathlib import Path


def export(run, output):
    run, output = Path(run), Path(output)
    task = run / "tasks/chainflow-incremental-readiness-v1"
    reports = {}
    for phase in ("train", "resume", "replay", "evaluation", "replay_evaluation"):
        path = task / phase / "training-result.json"
        if path.is_file():
            reports[phase] = json.loads(path.read_text())
    feedback_path = task / "paired_feedback/report.json"
    if not feedback_path.exists():
        feedback_path = task / "paired_feedback/feedback.json"
    feedback = json.loads(feedback_path.read_text()) if feedback_path.is_file() else None
    data = json.loads((run / "data-admission-audit.json").read_text())
    render_path = run / "render/production_writer_report.json"
    render = json.loads(render_path.read_text()) if render_path.exists() else None
    summary = {
        "kind": "chainflow_readiness_probe",
        "model": "ChainFlow stage1 navtrain_9308",
        "official_benchmark": False,
        "synthetic_training": False,
        "training_sample_count": data["derived_manifest"]["split_counts"]["train"],
        "development_sample_count": data["derived_manifest"]["split_counts"]["development"],
        "parent_pretraining_overlap": "unknown",
        "temporal_contract": data["temporal_contract"],
        "excluded_rows": data["excluded_original_rows"],
        "phases": {
            name: {
                key: value.get(key)
                for key in (
                    "status",
                    "updates",
                    "total_updates",
                    "metrics",
                    "consumed",
                    "effective_training_recipe",
                    "parameter_update_evidence",
                    "worker_pid",
                    "state_fingerprint",
                    "losses",
                    "sample_trace",
                )
            }
            for name, value in reports.items()
        },
        "resume_state_equal": bool(
            reports.get("resume")
            and reports.get("replay")
            and reports["resume"].get("state_fingerprint")
            == reports["replay"].get("state_fingerprint")
        ),
        "paired_feedback": feedback,
        "render_writer": render,
        "stable_improvement_demonstrated": False,
        "limitations": [
            "Identical-data arms verify execution and recovery, not augmentation benefit.",
            "One heldout recording for this update does not establish generalization or rare-event safety.",
            "The explicit incremental schedule and FP32 run are not full native training-recipe reproduction.",
            "Synthetic visual admission remains required; no failed branch entered training.",
            "Rendering measurements distinguish CPU save path from full GS and multi-GPU throughput.",
            "Other model families are not validated by this ChainFlow experiment.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    rows = []
    for name, record in reports.items():
        metrics = record.get("metrics") or {}
        scores = " / ".join(
            f"{arm}: {values['score'] * 100:.6f}" for arm, values in metrics.items()
        )
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(x)) + "</td>"
                for x in (
                    name,
                    record["status"],
                    record.get("total_updates", "—"),
                    record.get("parameter_update_evidence", {}).get(
                        "trainable_parameters_changed", "—"
                    ),
                    scores or "—",
                )
            )
            + "</tr>"
        )
    body = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ChainFlow stage1 · RSI readiness</title><style>
body{font-family:system-ui,sans-serif;background:#f3f6f8;color:#172b3a;margin:auto;padding:32px;max-width:1100px;line-height:1.6}
section{background:white;padding:24px;margin:20px 0;border-radius:12px}h1{font-size:28px}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #dce4ea;padding:10px;text-align:left}a{color:#086c97}.note{border-left:5px solid #d89d27;padding:16px;background:#fff7e6}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}code{background:#edf3f7;padding:2px 5px}</style>
<h1>ChainFlow stage1 · RSI 训练可靠性验证</h1>
<p class="note">这是小样本训练与工程验证，不是官方 navtest 成绩，也不是合成数据收益实验。原生模型、loss 和 PDM 保持不变；显式增量调度使用 2 步预热，FP32 与官方默认混精存在差异。</p>
<section><h2>真实训练与恢复</h2><p>官方 navtrain 精确 token：1 个训练样本、1 个本轮隔离开发样本。父模型预训练覆盖仍未完全核实。</p>
<table><tr><th>阶段</th><th>状态</th><th>累计更新</th><th>参数改变</th><th>开发 PDMS /100</th></tr>"""
        + "".join(rows)
        + "</table>"
    )
    body += (
        "<p>独立进程续训与重放最终状态一致：<b>"
        + str(summary["resume_state_equal"])
        + "</b>。同数据两组不获得增强收益认定。</p></section>"
    )
    body += """<section><h2>已修复的根本问题</h2><ol>
<li>训练入口同时检查官方日志与场景 token；旧清单两条不合规样本保留诊断并排除。</li>
<li>完整日志缓存默认最多 1 个；4 帧历史和 10 个真实未来点供原生双时域目标使用，主轨迹仍为 8 点。</li>
<li>原配方 212,500 步预热不适合数步验证；显式增量配方封存调度、精度和恢复状态。</li>
<li>按原始日志汇总配对收益，保留局部安全退化；少量相关窗口不能支持稳定晋级。</li>
<li>有界并行保存，所有文件完成且哈希可用后才发布帧清单；不减少视觉证据。</li></ol></section>
<section><h2>尚未解决的规模化验收</h2><p>同分支 GS 视觉准入与 10 点完整未来、有效训练干预、多独立日志与多种子收益、自动晋级回采、分布式吞吐和故障恢复仍需实际验证。不能从这些修复推断四类模型都已稳定变强。</p></section>"""
    if render:
        medians = render["median_wall_seconds"]
        body += ("<section><h2>生产保存阶段实测</h2><p>4 个时刻 × 4 相机，共 104 个文件；3 次重复中位数。PNG / 压缩 NPZ / SHA 逐项一致。</p>"
                 "<table><tr><th>保存线程</th><th>耗时</th></tr>"
                 f"<tr><td>1</td><td>{medians['1']:.3f} s</td></tr><tr><td>4</td><td>{medians['4']:.3f} s</td></tr></table>"
                 f"<p>保存阶段 {render['speedup_four_threads']:.2f}×；包含不可变快照拷贝与逐 tick flush，不含 GPU 渲染，不是多 GPU 吞吐承诺。</p></section>")
    body += '<p><a href="evidence.json">完整机器可读证据</a> · <a href="../cross-split-mining-v1/index.html">跨 split 挖掘验证</a></p>'
    body += (
        "<details><summary>当前配对反馈</summary><pre>"
        + html.escape(json.dumps(feedback, ensure_ascii=False, indent=2))
        + "</pre></details></html>"
    )
    (output / "index.html").write_text(body)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.run, args.output)
