"""Publish actual active-generation attempts and conditional learning evidence."""
import argparse,html,json
from pathlib import Path


def read(p):return json.loads(p.read_text()) if p.is_file() else {}
def esc(x):return html.escape(str(x))
def score(x):return f"{100*x:.4f}" if isinstance(x,(int,float)) else "--"
def link(root,p,label,suffix=""):
    return f"<a href='{esc(p.relative_to(root).as_posix()+suffix)}'>{esc(label)}</a>" if p.is_file() else ""


METRICS={"no_ego_at_fault_collisions":"无责任碰撞","drivable_area_compliance":"道路合规","driving_direction_compliance":"行驶方向","ego_is_making_progress":"有效前进","time_to_collision_within_bound":"TTC","ego_progress_along_expert_route":"专家路线进度","speed_limit_compliance":"限速","ego_is_comfortable":"舒适性"}


def export(root):
    body=["<h1>DriveRSI · 主动难例与连续恢复</h1>","<p>训练来源的真实 nuPlan 日志。先实际执行受约束扰动，再从相同状态比较原生学生与教师。保持原生模型、loss 和官方评分；规则响应交通，不称为学习型 Self-Play。下表为自定义长窗口 nuPlan 指标，非正式基准或 NAVSIM PDMS。</p>"]
    nav=read(root/"navigation-comparison.json")
    if nav:
        body.append("<section><h2>Boston 59.4 秒分岔：同源修复对照</h2><table><tr><th>执行量</th><th>之前</th><th>之后</th></tr>")
        for key,label in (("steps","记录步数"),("valid_actions","有效控制动作"),("support_first_s","路线支持首次耗尽 / s"),("travelled_m","实际行驶 / m"),("contacts","接触状态数"),("score_100","官方分数 / 100")):
            vals=["未触发" if nav[role][key] is None else (f"{nav[role][key]:.2f}" if isinstance(nav[role][key],float) else str(nav[role][key])) for role in ("before","after")]
            body.append(f"<tr><td>{label}</td><td>{esc(vals[0])}</td><td>{esc(vals[1])}</td></tr>")
        body.append("</table>"+link(root,root/"previous-boston/viewer.html","修复前 59.4 秒","?actor=ego&tick=594")+link(root,root/nav["after_viewer"],"修复后 59.4 秒","?actor=ego&tick=594")+link(root,root/"navigation-comparison.json","原始对照")+"</section>")
    rows=[]
    for task in sorted((root/"tasks").iterdir()):
        if not task.is_dir():continue
        run=read(task/"report.json");sel=read(task/"select/selection.json");gen=read(task/"select/generation.json");corr=read(task/"correct/correction.json");progress=read(task/"select/progress.json")
        row=dict(task=task.name,status=run.get("status","running"),selection=sel.get("status"),reason=sel.get("reason"),correction=corr.get("genuine"));rows.append(row)
        body.append(f"<section><h2>{esc(task.name)}</h2><p>任务：{esc(row['status'])}；筛选：{esc(row['selection'])}；原因：{esc(row['reason'])}；实际推进：{sel.get('physics_steps',progress.get('tick',0))/10:.1f} 秒。</p>")
        if gen:
            cfg=gen["configuration"];anchor=gen["anchor_tick"];reached=sel.get("reached",{}).get("arrival_verified",False)
            body.append(f"<p>扰动 {cfg['start_tick']/10:.1f}–{anchor/10:.1f} 秒，原生动作增量 {esc([cfg['jerk_delta'],cfg['lateral_delta']])}。计划锚点：{anchor/10:.1f} 秒；完整四帧历史核验：{esc(sel.get('reached',{}).get('history_ticks','未发布训练锚点'))}（10 Hz tick）。物理作用验证：{esc(gen.get('physical_effect_verified','旧尝试未记录独立核查'))}。</p>")
            body.append(link(root,task/"select/scout/viewer.html","播放到达历史","?actor=ego&tick="+str(cfg['start_tick']))+(link(root,task/"select/scout/viewer.html","查看已验证新锚点","?actor=ego&tick="+str(anchor)) if reached else "")+link(root,task/"select/generation.json","全部命令与约束证据"))
            failures=[x for x in gen["bounds"] if not x["accepted"]]
            if failures:body.append(f"<p>到达约束拒绝：{esc(failures[-1])}</p>")
            if gen.get("execution_error"):body.append(f"<p>执行异常：{esc(gen['execution_error'])}；已保留实际前缀。</p>")
        body.append(link(root,task/"select/scout/viewer.html","完整学生回放")+link(root,task/"select/selection.json","选择/拒绝记录"))
        if corr:
            body.append(f"<p>真实纠正结论：{esc(corr.get('genuine'))}</p><table><tr><th>指标（乘 100）</th><th>学生</th><th>教师</th></tr>")
            for key,label in [("score","总分"),*METRICS.items()]:
                values=[]
                for role in ("student","teacher"):
                    sc=corr["branches"][role]["score"];values.append(score(sc.get("score") if key=="score" else sc.get("metric_scores",{}).get(key)))
                body.append(f"<tr><td>{label}</td><td>{values[0]}</td><td>{values[1]}</td></tr>")
            body.append("</table>")
            for role,b in corr["branches"].items():
                x=b["execution"];body.append(f"<p>{role}：相同锚点 {esc(x['same_anchor'])}；后续 {x['suffix_steps']/10:.1f} 秒；接触 {x['contact_states']} 个状态；准入原因 {esc(b['admission']['reasons'])}。</p>"+link(root,task/f"correct/{role}/viewer.html",role+"同锚点播放","?actor=ego&tick="+str(x["anchor_tick"])))
        diagnosis=read(task/"attribution.json")
        if diagnosis:
            native=diagnosis["branches"]["student"];teacher=diagnosis["branches"]["teacher"]
            body.append(f"<p>时序诊断：锚点 {diagnosis['anchor_tick']/10:.1f} 秒；官方方向累计进度首次为负在 {native['first_negative_direction_tick']/10:.1f} 秒。学生恢复重放数组一致；教师共享历史数组一致。教师锚点合格候选 {teacher['first_teacher_admissible']}，仍未证明该状态可纠正。</p>"+link(root,task/"attribution.json","同锚点与扣分时序证据"))
        body.append("<h3>配对训练、长测与反馈</h3>")
        for stage in ("control-train","control-resume","treatment-train","treatment-resume","replay"):
            v=read(task/stage/"training-result.json")
            if v:body.append(f"<p>{stage}：{esc(v.get('status'))}；累计更新 {esc(v.get('total_updates'))}；新进程 {esc(v.get('worker_pid'))}。</p>"+link(root,task/stage/"training-result.json","训练回执"))
        for stage in ("control-evaluate","treatment-evaluate"):
            v=read(task/stage/"training-result.json")
            if v:body.append(f"<p>{stage}：{esc(v.get('status'))}；原因 {esc(v.get('reason'))}。</p>"+link(root,task/stage/"training-result.json","长测回执"))
            for c in sorted((task/stage).glob("case-*")):
                for role in ("parent","candidate"):
                    x=read(c/role/"execution.json")
                    if x:body.append(f"<p>{stage} / {c.name} / {role}：{x.get('recorded_steps',0)/10:.1f} 秒，得分 {score(x.get('score',{}).get('score'))}，接触 {len(x.get('contacts',[]))}。</p>"+link(root,c/role/"viewer.html","独立实际回放"))
        feed=read(task/"feedback/report.json")
        body.append(f"<p>晋级：{esc(feed.get('promoted',False))}；收益：{esc(feed.get('benefit','尚无合格完整证据'))}。重新采集只在固定长测通过后执行。</p>")
        body.append(link(root,task/"report.json","完整任务证据")+"</section>")
    body.append("<p>教师轨迹只证明可恢复性，不能冒充当前策略的 on-policy 样本。结构化 RL 合格不代表 GS 视觉监督合格。失败、未纠正和未完成尝试均保留；不会用安全原场景填充难例。</p>")
    (root/"summary.json").write_text(json.dumps(dict(tasks=rows,visual_trainable=False),indent=2))
    (root/"index.html").write_text("<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>DriveRSI 主动难例恢复</title><style>body{font:16px system-ui;max-width:1180px;margin:32px auto;padding:0 20px;background:#eef3f7;color:#163041}section{background:white;padding:22px;margin:18px 0;border-radius:12px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd6dd;padding:8px;text-align:left}a{display:inline-block;color:#006c91;margin:6px 14px 6px 0}p{line-height:1.65;overflow-wrap:anywhere}</style>"+"".join(body)+"</html>")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);export(p.parse_args().root)
