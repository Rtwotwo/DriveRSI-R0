"""Evidence gallery for fixed long recovery and heldout evaluation attempts."""
import argparse,html,json,math
from pathlib import Path

def read(p):return json.loads(p.read_text()) if p.is_file() else {}
def esc(x):return html.escape(str(x))
def points(x):return f"{100*x:.4f} / 100" if isinstance(x,(int,float)) else "未评分"
def link(root,p,text):return f"<a href='{esc(p.relative_to(root).as_posix())}'>{esc(text)}</a>" if p.is_file() else ""

def source_diagram(root, raw):
    objects={o["actor"]:o["raw_box"] for pair in raw["pairs"] for o in pair["objects"]}
    x0=min(b[0] for b in objects.values());y0=min(b[1] for b in objects.values())
    polygons=[]
    for actor,b in objects.items():
        x,y,width,length,_,yaw,_=b
        pts=[((x-x0)+dx*math.cos(yaw)-dy*math.sin(yaw),(y-y0)+dx*math.sin(yaw)+dy*math.cos(yaw))
            for dx,dy in ((length/2,width/2),(-length/2,width/2),(-length/2,-width/2),(length/2,-width/2))]
        text=" ".join(f"{a:.4f},{c:.4f}" for a,c in pts)
        polygons.append(f"<polygon points='{text}' fill='#e76d52' fill-opacity='.32' stroke='#d93f28' stroke-width='.04'><title>{actor}</title></polygon>")
    width=max(b[0]-x0 for b in objects.values())+6;height=max(b[1]-y0 for b in objects.values())+6
    (root/"source-contacts.svg").write_text(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='-3 -3 {width} {height}' width='500' height='420' role='img' aria-label='原始数据库的路障重叠框，单位米'>"+"".join(polygons)+"</svg>")


def export(root):
    rows=[];body=["<h1>真实难例纠正与长时收益验证</h1>","<p>固定原生模型、loss、官方评分。源标注接触、驾驶失败、教师失败分别保留；未通过不得训练或晋级。分数按原始 0–1 乘 100 展示，原始值完整保留；接触状态数不是独立事故数。此处为自定义长日志，不是 nuPlan 正式基准或 NAVSIM PDMS。</p>"]
    raw=read(root/"raw-db-contact-verification.json")
    if raw:
        source_diagram(root,raw)
        body.append("<img src='source-contacts.svg' alt='原始路障框重叠；不是控制后的结果'><p>原始路障框俯视图，重叠区域叠加加深；悬停查看来源 ID。</p>")
        body.append("<h2>第二窗口：原始数据库接触核验</h2><p>六对隔离栏在任何模型动作之前已经重叠。原始 DB 的尺寸、航向及位置与场景一致；没有删改物体，没有豁免碰撞。该窗口保持隔离。</p>")
        body.append(link(root,root/"raw-db-contact-verification.json","原始 DB 对照证据"))
        body.append("<table><tr><th>对象</th><th>原始重叠面积 m²</th><th>编译后面积 m²</th></tr>")
        for row in raw["pairs"]:body.append(f"<tr><td>{esc(' / '.join(row['actors']))}</td><td>{row['raw_overlap_area']:.6f}</td><td>{row['compiled_overlap_area']:.6f}</td></tr>")
        body.append("</table>")
    route=read(root/"prepared-boston-navigation-exact/preparation.json")
    if route:
        a,b=route["before"],route["after"]
        body.append("<h2>Boston 路线支持修复</h2>")
        body.append(f"<p>已加载地图中的路线支持：{a['extended_length_m']:.2f} → {b['extended_length_m']:.2f} 米。仅当一个分支能沿完整有序导航继续、其他分支已证实不兼容时延续；多解和缺图仍保留边界。原始位姿、尺寸、时间、参与者及背景配置保持一致。</p>")
        body.append(link(root,root/"prepared-boston-navigation-exact/preparation.json","地图连接依据"))
        body.append(link(root,root/"tasks/hard-recovery-log-01-v4/select/scout/viewer.html","修复前实际播放"))
        body.append(link(root,root/"tasks/hard-recovery-boston-navigation-v1/select/scout/viewer.html","修复后实际播放"))
        parity=read(root/"navigation-execution-comparison.json")
        if parity:
            body.append("<table><tr><th>实际执行</th><th>修复前</th><th>修复后</th></tr>")
            for key,label in (("duration_s","完整世界时长（秒）"),("route_support_first_s","首次路线支持耗尽（秒；空值为未触发）"),("travelled_m","自车累计行驶（米）"),("contact_states","物体接触状态数"),("policy_actions","自车有效动作步数"),("score_100","官方自定义长日志得分")):
                body.append(f"<tr><td>{label}</td><td>{esc(parity['before'].get(key))}</td><td>{esc(parity['after'].get(key))}</td></tr>")
            body.append("</table>"+link(root,root/"navigation-execution-comparison.json","完整执行对照"))
    preflight=read(root/"source-preflight/training-result.json")
    if preflight:
        body.append("<h2>独立评测的源数据检查</h2><p>Las Vegas 预定窗口存在初始行人框重叠，现已在模型推理前拦截，实际执行 0 步，保持未评分；没有将该案例换成更容易的日志，也没有计作安全通过。两份 Singapore 窗口属于同一原始日志，不能替代多个独立来源。</p>")
        body.append(link(root,root/"source-preflight/training-result.json","源数据拒绝证据"))
        body.append(link(root,root/"long-baseline-reconciliation.json","已完成分支与未完成来源核对"))
    body.append("<h2>难例筛选与同锚点纠正</h2>")
    for task in sorted((root/"tasks").glob("*")):
        if not task.is_dir():continue
        run=read(task/"report.json");sel=read(task/"select/selection.json");corr=read(task/"correct/correction.json")
        ev=read(task/"evaluate/training-result.json")
        if not ev:
            partial=[]
            for case_path in sorted((task/"evaluate").glob("case-*")):
                branches={role:read(case_path/role/"execution.json") for role in ("parent","candidate")}
                if not any(branches.values()):continue
                partial.append((case_path,branches))
            if partial:
                body.append(f"<section><h3>{esc(task.name)}：部分长测结果</h3><p>整个评测未完成；不能据此晋级。</p>")
                for path,branches in partial:
                    for role,value in branches.items():
                        if value:body.append(f"<p>{esc(path.name)} / {role}：{value['recorded_steps']/10:.1f} 秒，官方分数 {points(value['score'].get('score'))}，接触状态 {len(value['contacts'])}。"+link(root,path/role/"viewer.html","实际播放")+"</p>")
                body.append("</section>")
        row=dict(task=task.name,status=run.get("status","running"),selection=sel,correction=corr,evaluation=ev)
        rows.append(row)
        body.append(f"<section><h3>{esc(task.name)}</h3><p>任务：{esc(row['status'])}；筛选：{esc(sel.get('status','未执行'))}；原因：{esc(sel.get('reason',run.get('failure_reason','')))}</p>")
        if sel:
            if task.name == "hard-recovery-log-01-v4":
                support=read(root/"baseline-support-diagnostic.json")
                if support:
                    body.append("<p>补充执行核查：旧版报告生成后发现第 491 步（49.1 秒）路线支持耗尽，不能以官方 100 分认定完整驾驶合格。保留旧报告和分数，另存诊断，不覆盖历史证据。</p>"+link(root,root/"baseline-support-diagnostic.json","路线支持诊断"))
            if sel.get("support_diagnosis",{}).get("blocked"):
                body.append("<p>路线支持不足："+esc(sel["support_diagnosis"])+"；不能当作模型能力失败。</p>")
            if sel.get("source_contacts",{}).get("quarantined"):body.append(f"<p>初始重叠 {len(sel['source_contacts']['contacts'])} 对；尚未执行动作，因此不是零接触驾驶结果。</p>")
            body.append(f"<p>实际推进 {sel.get('physics_steps',0)/10:.1f} 秒；官方总分 {points(sel.get('score',{}).get('score'))}；接触状态数 {len(sel.get('contacts',[]))}。</p>")
            body.append(link(root,task/"select/scout/viewer.html","播放完整学生轨迹"))
            body.append(' · '+link(root,task/"select/selection.json","选择与拒绝证据"))
        if corr:
            body.append(f"<p>真实纠正验收：{esc(corr.get('genuine'))}</p>")
            for role,b in corr["branches"].items():
                body.append(f"<p>{esc(role)}：分数 {points(b['score'].get('score'))}；后续 {b['execution']['suffix_steps']/10:.1f}s；接触状态 {b['execution']['contact_states']}；准入 {esc(b['admission'])}</p>")
                body.append(link(root,task/"correct"/role/"viewer.html",role+"同锚点回放"))
        if ev:
            body.append("<h4>独立长时评测（无教师接管）</h4><table><tr><th>来源日志</th><th>时长</th><th>父模型</th><th>候选</th><th>接触状态：父／候选</th></tr>")
            for i,c in enumerate(ev["cases"]):
                a,b=c['branches']['parent'],c['branches']['candidate']
                body.append(f"<tr><td>{esc(c['source_log'])}</td><td>{c['steps']/10}s</td><td>{points(a['score'].get('score'))}</td><td>{points(b['score'].get('score'))}</td><td>{len(a['contacts'])} / {len(b['contacts'])}</td></tr>")
                body.append("<tr><td colspan='5'>"+link(root,task/f"evaluate/case-{i:02}/parent/viewer.html","父模型回放")+" · "+link(root,task/f"evaluate/case-{i:02}/candidate/viewer.html","候选回放")+"</td></tr>")
            body.append("</table><p>复用既有两次更新的候选检查点建立长时基线；不代表本轮已完成难例训练。</p>")
        for stage in run.get("stages",[]):
            if stage.get("status") in ("failed","outcome_unknown"):
                body.append(f"<p>阶段 {esc(stage['id'])}：{esc(stage.get('reason'))}</p>")
        body.append(link(root,task/"report.json","完整任务与产物证据")+"</section>")
    body.append("<p>训练只消费经教师验收的实际状态课程；教师动作不进入 on-policy 批次。合格新分支不足时，配对训练、晋级和重新采集保持未执行。既有短测不替代本页长时证据。</p>")
    (root/"summary.json").write_text(json.dumps(dict(tasks=rows,visual_trainable=False),indent=2))
    (root/"index.html").write_text("<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>DriveRSI 真实难例与长时评测</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#152c3b}section{background:white;padding:20px;margin:18px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd6dd;padding:8px;text-align:left}a{color:#066e9d;margin-right:12px}p{line-height:1.7}</style>"+"".join(body)+"</html>")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args();export(a.root)
