"""Publish exact-denominator official benchmark progress, never success-only scores."""

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from driversi.evaluation.metrics.benchmark_comparison import compare_official_results, component_comparison


def main():
    root = Path(__file__).resolve().parents[2]
    q = root / "outputs/quality/driverl-benchmark-v1"
    launch = json.loads((q / "launch.json").read_text())
    coverage = json.loads((q / "coverage.json").read_text())
    run_id = launch["environment"]["DRIVERL_EVAL_RUN_ID"]
    base = root / "outputs/demonstrations/driverl-benchmark-v1"
    paper_metrics = json.loads((root / "src/config/evaluation/driverl_paper_metrics.json").read_text())
    rows = []
    for task, paper in launch["paper_no_tts"].items():
        split, mode = task.rsplit("_", 1)
        key = {
            "val14": "driverl_val14",
            "test14hard": "driverl_test14_hard",
            "test14random": "driverl_test14_random",
        }[split]
        expected = set(coverage[key]["matches"])
        simulation = (
            "closed_loop_reactive_agents_driverl"
            if mode == "r"
            else "closed_loop_nonreactive_agents_driverl"
        )
        folder = base / run_id / (simulation + "_" + task)
        aggregation_name = (
            "closed_loop_reactive_agents_weighted_average"
            if mode == "r"
            else "closed_loop_nonreactive_agents_weighted_average"
        )
        aggregation_path = (
            root
            / "agents/driverl/native/DriveRL/nuplan-devkit/nuplan/planning/script/config/simulation/metric_aggregator"
            / (aggregation_name + ".yaml")
        )
        aggregation = yaml.safe_load(aggregation_path.read_text())[aggregation_name]
        required_metrics = set(aggregation["multiple_metrics"]) | (
            set(aggregation["metric_weights"]) - {"default"}
        )
        record = dict(
            task=task,
            expected=len(expected),
            paper_score=paper,
            status="queued",
            succeeded=0,
            failed=0,
            score=None,
            delta=None,
            problems=[],
            path=str(folder.relative_to(base)),
        )
        replay_index = {}
        if folder.exists():
            for saved in (folder / "simulation_log").rglob("*.msgpack.xz"):
                replay_index.setdefault(saved.name.removesuffix(".msgpack.xz"), []).append(str(saved.relative_to(base)))
            record["status"] = "running"
            record["saved_simulations"] = sum(len(v) for v in replay_index.values())
        report = folder / "runner_report.parquet"
        aggregates = list((folder / "aggregator_metric").glob("*weighted_average_metrics*.parquet"))
        if report.exists():
            runner = pd.read_parquet(report)
            record.update(
                succeeded=int(runner["succeeded"].eq(True).sum()),
                failed=int(runner["succeeded"].ne(True).sum()),
            )
            record["failure_cases"] = runner.loc[
                runner["succeeded"].ne(True), ["scenario_name", "log_name", "error_message"]
            ].to_dict("records")
            if len(aggregates) != 1:
                record["problems"].append("missing_or_ambiguous_aggregator")
                record["status"] = "incomplete"
            else:
                scores = pd.read_parquet(aggregates[0])
                # Pandas nullable cells become None; preserve invalid numeric scores for rejection.
                records = scores.astype(object).where(pd.notna(scores), None).to_dict("records")
                record.update(
                    compare_official_results(
                        expected,
                        runner.to_dict("records"),
                        records,
                        paper,
                        required_metrics=required_metrics,
                    )
                )
                record["metric_comparison"] = component_comparison(
                    expected, runner.to_dict("records"), records,
                    paper_metrics["results"][task], paper)
                # Preserve every imperfect driving result, separately from runner exceptions.
                record["driving_failures"] = [
                    {"token": v["scenario"], "log_name": v["log_name"], "score": v["score"],
                     "scenario_type": v["scenario_type"],
                     "components": {k: v.get(k) for k in required_metrics},
                     "replay_paths": replay_index.get(str(v["scenario"]), [])}
                    for v in records if v.get("num_scenarios") is None
                    and (v.get("score") is None or v["score"] < 1.0)
                ]
        rows.append(record)
    result = dict(
        updated_utc=datetime.now(timezone.utc).isoformat(),
        release="driverl-teacher-u2400",
        tts=False,
        scenario_duration_seconds=15,
        protocol="official_nuplan_Val14_Test14-hard_Test14-random_NR_R",
        paper="https://arxiv.org/html/2609.06055v1#S3.SS1.SSS1",
        results=rows,
        complete=all(x["status"] == "complete" for x in rows),
        train=False,
    )
    if result["complete"]:
        result["mean_score"] = sum(x["score"] for x in rows) / len(rows)
        result["paper_mean"] = 93.01
    (base / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    table = []
    for row in rows:
        score = "—" if row["score"] is None else f"{row['score']:.2f}"
        delta = "—" if row["delta"] is None else f"{row['delta']:+.2f}"
        status = html.escape(row["status"] + " " + ", ".join(row["problems"]))
        table.append(
            f"<tr><td>{row['task']}</td><td>{row['expected']}</td><td>{row['saved_simulations'] if 'saved_simulations' in row else 0}</td><td>{row['succeeded']} / {row['failed']}</td><td>{row['paper_score']:.2f}</td><td>{score}</td><td>{delta}</td><td>{status}</td></tr>"
        )
    details = []
    for row in rows:
        if not row.get("metric_comparison"):
            continue
        cells = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['paper']:.2f}</td><td>{v['actual']:.2f}</td><td>{v['delta']:+.2f}</td></tr>" for k,v in row["metric_comparison"].items())
        details.append(f"<h2>{row['task']} · 逐指标</h2><table><tr><th>指标</th><th>论文 A5</th><th>本次</th><th>差值</th></tr>{cells}</table><p>全部非满分驾驶案例：{len(row.get('driving_failures', []))}；与执行异常分开保存在 comparison.json，原始官方回放保留。</p>")
        zero_cases = [case for case in row.get("driving_failures", []) if case.get("score") == 0.0]
        zero_rows = []
        for case in zero_cases:
            failed_metrics = [key for key, value in case["components"].items() if value == 0.0]
            replays = " ".join('<a href="' + html.escape(path, quote=True) + '">官方回放文件</a>' for path in case["replay_paths"])
            zero_rows.append("<tr><td>" + html.escape(case["token"]) + "</td><td>" + html.escape(case["scenario_type"]) + "</td><td>" + html.escape(", ".join(failed_metrics)) + "</td><td>" + replays + "</td></tr>")
        details.append(f"<details><summary>零分驾驶案例：{len(zero_cases)}（运行成功也可能驾驶失败）</summary><p>列出原始指标为零的项目；这不是因果责任诊断。文件为官方 nuPlan 原始回放。</p><table><tr><th>Token</th><th>场景</th><th>零分指标</th><th>证据</th></tr>" + "".join(zero_rows) + "</table></details>")
    (base / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>DriveRL 官方 nuPlan 评测</title><style>body{font:16px/1.7 system-ui;background:#edf3f6;color:#173849;margin:40px}table{border-collapse:collapse;background:white;width:100%}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}aside{background:#fff2cf;padding:20px}a{color:#006c78}</style><h1>DriveRL · 官方 nuPlan 基准对照</h1><p>u2400 · 不启用 TTS · 15 秒 · 1,651 个场景 × NR / R</p><aside>这是官方运行时基线复现。DriveRSI 自定义 60 秒交通世界另行展示；本表不能单独证明平台交通控制或适配性能等价。只有 runner 与评分结果完整覆盖固定 token 且没有执行失败，才显示可比总分。未完成不填零，也不计算成功子集平均分。</aside><p>结果快照 '
        + html.escape(result["updated_utc"])
        + "</p><table><tr><th>任务</th><th>应测场景</th><th>已保存回放</th><th>已结算成功 / 失败</th><th>论文</th><th>本次</th><th>差值</th><th>状态</th></tr>"
        + "".join(table)
        + "</table>" + "".join(details)
        + '</table><p><a href="comparison.json">完整状态及失败明细</a> · <a href="../driverl-multicity-v1/index.html">四城市长日志与修复对照</a> · <a href="https://arxiv.org/html/2609.06055v1#S3.SS1.SSS1">论文来源</a></p></html>',
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "results": [
                    {k: v for k, v in x.items() if k not in {"failure_cases", "driving_failures", "path"}} for x in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
