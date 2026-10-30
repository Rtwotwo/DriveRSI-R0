"""Evidence-only multicity index, including failed attempts and official reference."""

import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    root = Path(__file__).resolve().parents[3]
    base = root / "outputs/demonstrations/driverl-multicity-v1"
    rows = []
    evidence = []
    for i in range(1, 9):
        log = f"log-{i:02}"
        p = base / log
        run_dir = next(
            (
                name
                for name in [
                    "route-v4",
                    "motion-v3",
                    "motion-v2",
                    "final-v2",
                    "final",
                    "closed-loop",
                ]
                if (p / name / "report.json").exists()
            ),
            "motion-v3" if (p / "prepared-motion-v3").exists() else "closed-loop",
        )
        source_dir = {
            "route-v4": "prepared-route-v4",
            "motion-v3": "prepared-motion-v3",
            "motion-v2": "prepared-motion",
            "final-v2": "prepared-final",
            "final": "prepared-final",
        }.get(run_dir, "prepared-v2" if (p / "prepared-v2").exists() else "prepared")
        prep = read(p / source_dir / "preparation.json")
        report = read(p / run_dir / "report.json")
        score = read(p / run_dir / "official-score.json")
        official_dir = "official-v2" if (p / "official-v2").exists() else "official"
        official = read(p / official_dir / "official-score.json")
        failure = read(p / run_dir / "job-result.json")
        failure_log = root / "outputs/quality/driverl-multicity-v1" / (log + "-" + run_dir + ".log")
        failure_reason = failure.get("error") or (
            failure_log.read_text().strip().splitlines()[-1]
            if failure.get("exit_code") and failure_log.exists()
            else "执行中 / 未运行"
        )
        newest = "route-v4" if (p / "prepared-route-v4").exists() else "motion-v3"
        newest_prep = "prepared-route-v4" if newest == "route-v4" else "prepared-motion-v3"
        pending = (p / newest_prep).exists() and not (p / newest / "report.json").exists()
        contacts = report.get("contacts", {})
        initial_contacts = (
            contacts.get("onset_counts", {}).get("initial_overlap", 0) if report else "—"
        )
        time_text = (
            f"{report['inference_steps'] / 10:.1f} / {report['seconds']:.1f}s"
            if report
            else "待完成"
        )
        contact_text = f"自车 {contacts.get('owner_contact_ticks', '—')} / 背景 {contacts.get('other_contact_ticks', '—')} 状态；初态 {initial_contacts} 对"
        route = prep.get("route", {})
        event = {k: v for k, v in report.get("terminal_events", {}).items() if v}
        city = prep.get("city", log)

        def value(s):
            return (
                f"{s['score'] * 100:.2f}"
                if s.get("status") == "scored"
                else html.escape(str(s.get("error", "未完成")))
            )

        links = []
        for relative, title in [
            (f"{run_dir}/index.html", "当前验收"),
            (f"{run_dir}/viewer.html?actor=ego&tick=180", "当前闭环播放"),
            ("closed-loop/viewer.html?actor=ego&tick=180", "早期对照"),
            ("final/viewer.html?actor=ego&tick=180", "运动修复前"),
            ("source.html", "完整源日志"),
            (f"{official_dir}/official-score.json", "官方世界评分"),
            (f"{official_dir}/parity.json", "官方同输入对照"),
            (f"{run_dir}/official-score.json", "平台评分"),
            ("prepared/failure.json", "原始准备失败"),
            ("closed-loop/job-result.json", "早期执行记录"),
            ("motion-v2/report.json", "上版运动修复记录"),
            (f"{run_dir}/job-result.json", "执行记录"),
        ]:
            if (p / relative.split("?")[0]).exists():
                links.append(f'<a href="{log}/{relative}">{title}</a>')
        rows.append(
            f'<tr data-city="{html.escape(city)}"><td><b>{log}</b><br>{html.escape(city)}<br><small>版本 {html.escape(run_dir)}{f"；{newest} 复测中" if pending else ""}</small><br><small>{html.escape(prep.get("source_log", ""))}</small></td><td>{prep.get("actors", "—")}<br>{route.get("original_length_m", 0):.0f} → {route.get("extended_length_m", route.get("original_length_m", 0)):.0f} m<br>{html.escape(route.get("end_reason", route.get("reason", "")))}</td><td>{time_text}<br>{html.escape(str(event or "无终止事件" if report else failure_reason))}<br><small>{html.escape(contact_text)}</small></td><td>{value(score)}</td><td>{value(official)}</td><td>{" ".join(links)}</td></tr>'
        )
        evidence.append(
            dict(
                log=log,
                run=run_dir,
                preparation=prep,
                execution=report,
                score=score,
                official_world=official,
                job=failure,
                newer_attempt_pending=pending,
                attempts={
                    name: {
                        "job": read(p / name / "job-result.json"),
                        "report": read(p / name / "report.json"),
                        "score": read(p / name / "official-score.json"),
                    }
                    for name in [
                        "closed-loop",
                        "final",
                        "final-v2",
                        "motion-v2",
                        "motion-v3",
                        "route-v4",
                    ]
                    if (p / name).exists()
                },
            )
        )
    (base / "summary.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    (base / "index.html").write_text(
        """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DriveRL · 四城市连续闭环</title><style>body{font:15px/1.6 system-ui;background:#edf3f6;color:#18384a;margin:0}main{max-width:1400px;margin:auto;padding:30px}header{padding:30px;background:#123346;color:white;border-radius:16px}h1{margin:5px 0}.note{padding:20px;background:#fff4da;border-left:4px solid #bc7b18;margin:20px 0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:15px;text-align:left;border-bottom:1px solid #d6e2e6;vertical-align:top}a{display:inline-block;color:#006b70;margin:4px}small{font-size:11px;overflow-wrap:anywhere}select{padding:10px;margin:20px 0}td:first-child{max-width:260px}code{overflow-wrap:anywhere}</style><main><header><div>NATIVE DRIVE RL · FIXED 8 LOGS · NO TRAINING</div><h1>四城市连续闭环与官方指标</h1><p>路线支持延续 / 完整失败记录 / 官方运行时独立参照</p></header><div class="note">平台闭环与官方 nuPlan IDM 使用不同背景交通及终止规则，分数差不能直接归因于模型适配。两边独立运行同一权重；同输入输出校验单独记录。初始来源重叠被保留并计入接触，不意味着无碰撞；短暂标注消失也不被直接当作物理离场。评分调用官方 nuPlan 指标和权重，但这是自定义长日志协议，不是 NAVSIM PDMS 或官方基准复现。未启动训练，全部保持 trainable=false。未评分和失败场景仍列出。</div><p><a href="../driverl-benchmark-v1/index.html">正式 nuPlan 三划分 × NR/R 与论文对照</a></p><label>城市 <select id="city"><option value="">全部城市</option><option>us-ma-boston</option><option>us-pa-pittsburgh-hazelwood</option><option>sg-one-north</option><option>us-nv-las-vegas-strip</option></select></label><table><thead><tr><th>日志</th><th>对象 / 路线支持</th><th>模型有效 / 世界时长</th><th>平台世界官方指标<br>0–100</th><th>官方 IDM 世界指标<br>0–100</th><th>证据与播放</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table><p><a href="../driverl-closed-loop-v1/release/index.html">修复前 34.8 秒案例</a> · <a href="summary.json">完整汇总</a></p></main><script>document.getElementById('city').onchange=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!!e.target.value&&r.dataset.city!==e.target.value);</script></html>""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
