"""Small server-side inspection page for immutable recipe evidence."""

import html
import json
import time
from pathlib import Path


def write_viewer(output, state):
    from driversi.contracts.scene import Scene
    from driversi.rendering.bev.replay import export_replay
    from driversi.runtime.budget import write_json

    started = time.monotonic()
    esc = html.escape
    output = Path(output)
    scene = Scene.load(output / "scene.npz") if (output / "scene.npz").exists() else None
    exports = []
    rows = []
    for a in state["attempts"]:
        m = a.get("metrics") or {}
        link = (
            '<a href="' + esc(a["rollout"], quote=True) + '">完整动作记录</a>'
            if a.get("rollout")
            else ""
        )
        if scene is not None and a.get("rollout"):
            name = a["recipe"]["id"] + ".html"
            try:
                baseline = output / (state["spec"]["baseline"]["id"] + ".npz")
                export_replay(
                    scene,
                    output / a["rollout"],
                    output / name,
                    compare_path=baseline
                    if baseline.exists() and baseline.name != a["rollout"]
                    else None,
                )
                link += ' · <a href="' + esc(name, quote=True) + '">交互回放</a>'
                exports.append({"recipe": a["recipe"]["id"], "page": name, "status": "completed"})
            except Exception as exc:
                exports.append(
                    {"recipe": a["recipe"]["id"], "status": "failed", "reason": str(exc)}
                )
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + esc(str(x)) + "</td>"
                for x in (
                    a["recipe"]["id"],
                    a["status"],
                    m.get("engineering_pass"),
                    m.get("route_progress_m"),
                    m.get("reasons", a.get("failure_reason")),
                    a.get("charged_agent_steps"),
                )
            )
            + "<td>"
            + link
            + "</td></tr>"
        )
    body = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>DriveRSI 纠正配方</title>
<style>body{font:16px system-ui;background:#10202d;color:#e4eef5;padding:30px;max-width:1200px;margin:auto}a{color:#70c9ff}td,th{border-bottom:1px solid #526879;padding:12px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>同状态纠正配方</h1><p>工程代理验收，不是 PDM 评分。所有产物 trainable=false；历史保留、其他主体逐步响应。</p>"""
    body += "<p>运行：" + esc(state["spec"]["id"]) + " · " + esc(state["status"]) + "</p>"
    body += (
        "<p>角色："
        + esc(state["role"])
        + "；已计费物理 agent steps："
        + str(state.get("admitted_agent_steps", 0))
        + "</p>"
    )
    body += (
        "<table><tr><th>配方</th><th>执行</th><th>工程通过</th><th>路线进度 m</th><th>拒绝原因</th><th>计费步数</th><th>记录</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    body += (
        '<p><a href="report.json">完整报告和检索证据</a></p><h2>跨日志检索依据</h2><pre>'
        + esc(json.dumps(state.get("selection_evidence", []), ensure_ascii=False, indent=2))
        + "</pre></html>"
    )
    (Path(output) / "index.html").write_text(body)
    write_json(
        output / "presentation.json",
        {
            "exports": exports,
            "export_wall_seconds": time.monotonic() - started,
            "included_in_execution_budget": False,
        },
    )
