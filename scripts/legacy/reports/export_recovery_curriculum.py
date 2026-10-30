"""Publish actual Harness evidence, including rejected and interrupted tasks."""
import argparse
import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def esc(value):
    return html.escape(str(value))


def stages(directory):
    records = {}
    report = read(directory / "report.json") or {}
    for value in report.get("stages", []):
        records[value["id"]] = dict(stage=value["id"], status=value["status"],
            reason=value.get("reason"), cost=value.get("charged_agent_steps"),
            cost_basis=value.get("cost_basis"))
    for path in sorted(directory.glob("*.result.json")):
        value = read(path)
        name = path.name.removesuffix(".result.json")
        records.setdefault(name, dict(stage=name, status=value["status"],
            reason=value.get("reason"), cost=value.get("sim_agent_steps")))
    for path in sorted(directory.glob("*.receipt.json")):
        value = read(path)
        name = path.name.removesuffix(".receipt.json")
        records.setdefault(name, dict(stage=name, status=value.get("phase", "unknown"),
            reason=value.get("reason"), cost=None, evidence="receipt_only"))
    return list(records.values())


def available_link(root, path, label):
    return f"<a href='{esc(path.relative_to(root).as_posix())}'>{esc(label)}</a>" if path.is_file() else ""


def export(root, task):
    t = root / "tasks" / task
    selection = read(t / "select/selection.json") or {}
    correction = read(t / "correct/correction.json") or {}
    feedback = read(t / "feedback/report.json")
    records = stages(t)
    training = {}
    for name in ("control-train", "control-resume", "treatment-train", "treatment-resume", "replay"):
        value = read(t / name / "training-result.json")
        if value:
            training[name] = {k: value.get(k) for k in
                ("status", "updates", "total_updates", "curriculum", "metrics", "consumed", "reason", "worker_pid")}
    secondary = []
    for directory in sorted((root / "tasks").iterdir()):
        if not directory.is_dir() or directory.name == task:
            continue
        s = read(directory / "select/selection.json") or {}
        task_report = read(directory / "report.json") or {}
        secondary.append(dict(task=directory.name, status=task_report.get("status", "unknown"),
            selection_status=s.get("status", "not_executed"), stages=stages(directory),
            eligible=sum(bool(x.get("eligible")) for x in s.get("candidates", [])),
            candidates=len(s.get("candidates", [])), contact_states=len(s.get("contacts", [])),
            physics_steps=s.get("physics_steps")))
    manifest = dict(task=task, selection=selection, correction=correction, training=training,
        feedback=feedback, stages=records, secondary=secondary, visual_trainable=False, promoted=False)
    (root / "summary.json").write_text(json.dumps(manifest, indent=2))
    body = ["<h1>从实际交互到在线训练课程</h1><p>真实 nuPlan · 自动风险锚点 · 同状态教师 · 原生独立评分 · 配对在线更新</p>",
        "<div class='notice'>这是小规模链路验收。风险排序不是驾驶错误定论；背景为规则交通。教师动作没有充当 PPO 样本，视觉数据尚未准入，模型未晋级。</div>"]
    chosen = selection.get("selection") or {}
    body.append("<h2>1　自动选择锚点</h2>")
    if chosen:
        body.append(f"<p>选中时刻 {chosen['tick']/10:.1f} s；保存完整世界、策略记忆和四帧到达历史。径向接近指标用于排序，不等同于官方 TTC。</p>")
    else:
        body.append(f"<p>未选出合格锚点：{esc(selection.get('reason', selection.get('status', '尚未执行')))}。后续阶段不得进入训练。</p>")
    body.append(available_link(root, t / "select/scout/viewer.html", "查看完整采集回放，包括被拒绝的状态"))
    body.append("<table><tr><th>时刻 s</th><th>对象</th><th>车身间距 m</th><th>接近时间代理 s</th><th>可选</th></tr>")
    for row in selection.get("candidates", []):
        values = (row["tick"]/10, row.get("actor"), row.get("clearance_m"), row.get("closing_ttc_s"), row.get("eligible"))
        body.append("<tr>" + "".join("<td>" + esc(round(x, 3) if isinstance(x, float) else x) + "</td>" for x in values) + "</tr>")
    body.append("</table><h2>2　同锚点实际分支</h2><p>评分使用共同前缀和分叉后实际执行世界，原生 nuPlan 指标保持不变。这里是自定义短窗口，不是正式基准或 NAVSIM PDMS。</p><div class='cards'>")
    playable = []
    for name in ("student", "teacher"):
        value = correction.get("branches", {}).get(name)
        if not value:
            continue
        score, exe, gate = value.get("score", {}), value.get("execution", {}), value.get("admission", {})
        label = "学生续驶" if name == "student" else "教师候选"
        raw = score.get("score")
        display = f"{100*raw:.4f} / 100" if isinstance(raw, (int, float)) else "未取得评分"
        viewer = t / "correct" / name / "viewer.html"
        if viewer.is_file():
            playable.append(name)
        body.append(f"<article><h3>{label}</h3><strong>{display}</strong><p>接触状态：{esc(exe.get('contact_states', '未知'))} · 完整后续：{esc(exe.get('complete', '未知'))}</p><p>" +
            (f"课程准入：{esc(gate.get('accepted'))}；{esc(gate.get('reasons', []))}" if name == "teacher" else "同状态对照；原分支已安全，不能据此宣称难例被纠正") +
            "</p>" + available_link(root, viewer, "实际回放") + " · " +
            available_link(root, t / "correct" / name / "report.json", "完整原始评分与失败原因") + "</article>")
    body.append("</div>")
    if playable:
        options = "".join(f"<option value='{name}'>{'学生续驶' if name == 'student' else '教师候选'}</option>" for name in playable)
        body.append("<label>播放分支 <select id='branch'>" + options + "</select></label>")
        body.append(f"<iframe id='replay' title='实际分支回放' src='tasks/{esc(task)}/correct/{playable[0]}/viewer.html?actor=ego&tick={chosen.get('tick', 0)}'></iframe>")
    else:
        body.append("<p>没有已执行的纠正分支。请查看采集记录及下方阶段失败原因。</p>")
    body.append("<h2>3　合格分支进入在线课程</h2><p>控制组从预定第 2 秒状态开始，实验组从自动选择状态开始。共用父模型、原生损失、奖励、优化器和更新预算；两组均重新采样当前策略动作与概率。教师分支提供可恢复性依据，未直接作为 RL 动作标签。</p>")
    body.append("<table><tr><th>阶段</th><th>状态</th><th>本阶段更新</th><th>累计更新</th></tr>")
    for name, value in training.items():
        body.append("<tr>" + "".join("<td>" + esc(x) + "</td>" for x in
            (name, value["status"], value["updates"], value["total_updates"])) + "</tr>")
    body.append("</table><h2>4　独立评测与收益反馈</h2>")
    if feedback:
        body.append("<p>开发日志与训练来源隔离。当前仅为 2 秒工程探针，不能推断完整驾驶能力；即使微小正收益也不晋级。</p>")
        body.append("<table><tr><th>策略</th><th>前进距离 m</th><th>无接触状态比例</th></tr>")
        metrics = feedback["arms"]["control"]["evaluation"]["metrics"]
        for label, value in (("父模型", metrics["parent"]), ("普通续训", metrics["candidate"]),
                             ("课程训练", feedback["arms"]["treatment"]["evaluation"]["metrics"]["candidate"])):
            body.append(f"<tr><td>{label}</td><td>{value['travelled_m']:.6f}</td><td>{value['contact_free_fraction']}</td></tr>")
        body.append("</table><pre>" + esc(json.dumps({k: feedback[k] for k in
            ("resume", "treatment_minus_control", "next_action", "promoted")}, indent=2, ensure_ascii=False)) + "</pre>")
        body.append(available_link(root, t / "feedback/report.json", "完整反馈、成本及薄弱点分析"))
    else:
        body.append("<p>训练或评测尚未完成，不展示推测收益。</p>")
    body.append("<h2>其他真实窗口与失败保留</h2><p>下面的第二窗口来自同一 Pittsburgh 原始训练日志的另一分片，不是独立城市或独立样本。没有合格锚点时，停止教师调用及训练；保留实际交通接触。</p>")
    for value in secondary:
        directory = root / "tasks" / value["task"]
        body.append(f"<article><h3>{esc(value['task'])}</h3><p>任务：{esc(value['status'])}；选择：{esc(value['selection_status'])}；可选锚点 {value['eligible']}/{value['candidates']}；物体接触状态 {value['contact_states']}。</p>" +
            available_link(root, directory / "select/scout/viewer.html", "失败采集回放") + " · " +
            available_link(root, directory / "select/selection.json", "全部候选与接触对象") +
            "<pre>" + esc(json.dumps(value["stages"], indent=2, ensure_ascii=False)) + "</pre></article>")
    body.append("<h2>主任务全部阶段与失败记录</h2><pre>" + esc(json.dumps(records, indent=2, ensure_ascii=False)) + "</pre>")
    body.append("<p><a href='summary.json'>机器可读汇总</a> · <a href='implementation-ledger.json'>实现与验收记录</a></p>")
    style = "body{margin:32px auto;max-width:1180px;padding:0 20px;background:#f3f6f8;color:#173244;font:16px/1.6 system-ui}h1{margin-bottom:4px}h2{margin-top:38px}.notice,article{padding:20px;border:1px solid #cfdee5;border-radius:10px;background:white}.notice{border-left:5px solid #db9b37}.cards{display:flex;gap:20px}.cards article{flex:1}strong{font-size:24px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd;word-break:break-word}pre{white-space:pre-wrap;padding:18px;background:#e5edf2}iframe{width:100%;height:780px;border:1px solid #bbb;margin-top:12px}a{color:#007f8d}select{padding:8px;margin-top:16px}"
    script = f"const branch=document.getElementById('branch'); if(branch) branch.onchange=e=>document.getElementById('replay').src='tasks/{task}/correct/'+e.target.value+'/viewer.html?actor=ego&tick={chosen.get('tick', 0)}';"
    (root / "index.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>DriveRSI · 纠正课程闭环</title><style>" + style + "</style><body>" + "".join(body) + "<script>" + script + "</script></body></html>")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--task", required=True)
    args = p.parse_args()
    export(args.root, args.task)
