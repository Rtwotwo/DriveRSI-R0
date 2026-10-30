"""Export a read-only gallery from actual bounded NAVSIM teacher task receipts."""

import argparse
import html
import json
from pathlib import Path


def export(root):
    root = Path(root).resolve()
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    costs = 0
    for report in sorted(root.glob("tasks/*/teacher/execution/report.json")):
        if "gs-export" in report.parts[-4]:
            continue
        data = json.loads(report.read_text())
        costs += data.get("charged_joint_steps", 0)
        for branch, evidence in data.get("branches", {}).items():
            score = evidence.get("score") or {}
            trajectory = report.parent / (branch + ".html")
            link = esc(trajectory.relative_to(root))
            label = esc(report.parts[-4] + " / " + branch)
            role = (
                data.get("teacher", "unknown")
                if branch == "teacher"
                else "bounded control candidate"
                if branch.startswith("candidate-")
                else "arrival continuation; not student"
            )
            contact = len(data.get("arrival_contacts", [])) + len(
                evidence.get("object_contacts", [])
            )
            points = "unknown" if "score" not in score else f"{100 * score['score']:.3f}"
            passed = evidence.get("trajectory_admitted", False)
            rows.append(
                f'<tr><td><a href="{link}">{label}</a></td><td>{esc(role)}</td><td>{points}</td><td>{contact}</td><td>{"trajectory pass" if passed else "rejected"}</td></tr>'
            )
    artifacts = []
    for task in sorted(root.glob("tasks/*gs-export*")):
        for relative, label in [
            ("select/trajectory-window.json", "4-history / 8-future executed states"),
            ("render/render/session/viewer.html", "GS sequential camera viewer"),
            ("dataset/manifest.json", "dataset and admission report"),
            ("render/render/report.json", "exact physics replay / rendering status"),
        ]:
            path = task / relative
            if path.exists():
                artifacts.append(
                    f'<li>{esc(task.name)}: <a href="{esc(path.relative_to(root))}">{label}</a></li>'
                )
    summary_html = ""
    summary_path = root / "result-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        result_rows = "".join(
            f"<tr><td>{esc(row['source_log'])}</td><td>{100 * row['baseline_score']:.3f}</td><td>{100 * row['teacher_score']:.3f}</td><td>{esc(row['qualified_trajectory'])}</td></tr>"
            for row in summary["trajectory_results"]
        )
        summary_html = f"""<h2>本轮实际结果</h2>
<p><a href="{esc(summary["viewer"])}">最新连续回放</a> · <a href="window.html">4历史＋8未来逐帧查看</a> · <a href="{esc(summary["package"])}">数据准入</a> · <a href="visual-admission-audit.json">视觉资产失败证据</a></p>
<table><tr><th>训练来源日志</th><th>扰动延续对照</th><th>控制搜索分支</th><th>轨迹通过</th></tr>{result_rows}</table>
<p class="note">已渲染 {summary["rendered_images"]} 张图，导出历史 {summary["image_counts"]["history"]} 张、未来 {summary["image_counts"]["future"]} 张；原生 NAVSIM 加载通过。视觉准入仍拒绝：当前日志对象与重建检查点节点匹配数为 {summary["checkpoint_object_matches"]}。相机移动不等于交通对象外观随世界移动；车辆残影、高程代理与配准问题未解决。训练未启动。</p>"""
        session = root / summary["viewer"]
        manifest = json.loads(session.with_name("manifest.json").read_text())
        tiles = []
        for frame in manifest["frames"]:
            if frame["tick"] not in range(5, 61, 5):
                continue
            camera = next(c for c in frame["cameras"] if c["camera_id"] == "CAM_F0")
            source = (session.parent / camera["image"]).relative_to(root)
            role = "历史输入" if frame["tick"] <= 20 else "未来监督"
            tiles.append(
                f'<figure><img src="{esc(source)}"><figcaption>{role} · {frame["time_seconds"]:.1f}s · tick {frame["tick"]}</figcaption></figure>'
            )
        (root / "window.html").write_text(
            f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>实际执行的4历史＋8未来</title><style>body{{font:16px system-ui;background:#f4f6fa;margin:24px}}main{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}figure{{margin:0;background:white}}img{{width:100%}}figcaption{{padding:10px}}</style><h1>同一物理分支：新历史与纠正未来</h1><p><a href="index.html">返回结果</a> · <a href="{esc(summary["viewer"])}">播放／切换四相机</a></p><p>2.0秒为决策锚点；第一行是四帧输入，后两行是未来监督。车辆资产缺失，图像仍为诊断数据，trainable=false。此页没有把原日志图片替代为合成图片。</p><main>{"".join(tiles)}</main></html>"""
        )
    body = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>NAVSIM 4+8 trajectory-first synthesis</title>
<style>body{{font:16px system-ui;margin:32px;max-width:1550px;background:#f6f8fb;color:#172b3a}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border:1px solid #d6dee7;text-align:left}}a{{color:#1762a5}}.note{{background:#fff4d7;padding:16px;border-radius:8px}}li{{margin:10px}}</style>
<h1>NAVSIM：先执行新历史与纠正未来，再渲染</h1>
<p>训练来源内的有界小样本验证。4 帧历史：0.5、1.0、1.5、2.0 秒；8 帧未来：2.5～6.0 秒。评分另执行到7.0秒以提供 TTC 上下文。</p>
<p class="note">轨迹门槛保持95分，方向合规独立检查，历史及未来接触均保留。分数采用原生 NAVSIM v1 PDM 数学、实际反事实分支环境，属于自定义分支协议，不是官方原日志基准成绩。规则交通与固定扰动不是学习型 Self-Play；扰动对照不是学生模型。</p>
<p>候选搜索、原生 DriveRL、PDM 教师分别标记。当前所有导出均保持 trainable=false；轨迹通过不代表图像、参与者覆盖或训练收益通过。未启动优化器更新。</p>
{summary_html}
<h2>真实分支与失败记录</h2><p>表中独立生成任务实际计费 {costs} 个联合物理步；复用产物的渲染任务不重复计作搜索。接触数量为成对接触的状态记录，不是事故次数。</p>
<table><thead><tr><th>回放</th><th>生成方式</th><th>反事实 PDM ×100</th><th>接触记录</th><th>轨迹准入</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>完整窗口、渲染与数据工件</h2><ul>{"".join(artifacts)}</ul>
<p>原始失败工件保留。早期 exploratory PDM 文件顶层旧 policy 字段有误，应以 report 与逐步 controller diagnostics 识别 PDM；后续记录已修复。GS v3 及以前的 native_replay 原始输入标签残留误标，v4 从封存生成器协议派生，未改写旧工件。</p>
</html>"""
    (root / "index.html").write_text(body)
    return dict(rows=len(rows), joint_steps=costs, artifacts=len(artifacts), trainable=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    print(json.dumps(export(parser.parse_args().experiment)))
