"""Read-only evidence gallery: actual lane association, forecasts and workflow gates."""

import argparse
import json
from pathlib import Path


def export(root):
    data = json.loads((root / "direction-view-data.json").read_text())
    tasks = []
    for path in sorted((root / "tasks").glob("*")):
        if not (path / "report.json").is_file():
            continue
        report = json.loads((path / "report.json").read_text())
        correction = (
            json.loads((path / "correct/correction.json").read_text())
            if (path / "correct/correction.json").is_file()
            else {}
        )
        decisions = path / "correct/teacher/decisions.json"
        evidence = json.loads(decisions.read_text()) if decisions.is_file() else []
        search = evidence[0].get("repair_evidence", {}) if evidence else {}
        tasks.append(
            dict(
                search={k: search.get(k) for k in ("attempted_schedules", "proposals", "admissible_proposals", "search_wall_seconds")},
                id=path.name,
                status=report["status"],
                stages=[
                    {k: x.get(k) for k in ("id", "status", "reason")} for x in report["stages"]
                ],
                correction=correction,
            )
        )
    data["tasks"] = tasks
    data["searches"] = []
    for case in data["cases"]:
        for family in ("lattice", "damped", "quintic", "frontaxle"):
            p = root / (case["id"] + "-" + family + ".json")
            if p.is_file():
                d = json.loads(p.read_text())
                data["searches"].append(
                    dict(
                        case=case["id"],
                        family=family,
                        supported=d["proposals"],
                        rejected=len(d.get("rejected_schedules", [])),
                        admissible=d["admissible_proposals"],
                        file=p.name,
                    )
                )
    (root / "gallery-data.json").write_text(json.dumps(data, ensure_ascii=False))
    (root / "index.html").write_text(PAGE)


PAGE = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DriveRSI · 方向零分教师纠正</title>
<style>body{font:16px system-ui;margin:0;background:#eef3f7;color:#163041}main{max-width:1240px;margin:28px auto;padding:0 20px}section{padding:22px;background:white;border-radius:12px;margin:18px 0}h1{font-size:28px}p{line-height:1.65}a{color:#08769b;margin-right:16px}select,button{font:inherit;padding:8px;margin:6px}canvas{background:#101f2d;border-radius:8px;width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cad6df;padding:8px;text-align:left}.warn{border-left:5px solid #cf8233;padding-left:14px}.chips{display:flex;flex-wrap:wrap;gap:8px}.chips span{background:#edf1f5;padding:6px;border-radius:5px}input[type=range]{width:65%}code{overflow-wrap:anywhere}.legend{font-size:14px}.status{font-weight:600}</style>
<main><h1>方向零分：同锚点教师纠正与失败证据</h1><p class="warn">同一 Pittsburgh 训练日志的两个扰动锚点，学生对照各30秒；不是正式基准成绩。地图、原生模型、loss 与官方评分保持不变。候选预测不等于联合执行；不合格分支不能用于训练。短于预定窗口的教师分数不与完整学生分数作收益比较。本轮两个原锚点均未找到合格纠正，未启动配对训练、独立长测或晋级。</p>
<section><h2>道路关联与候选构造</h2><label>锚点 <select id="case"></select></label><label>预测候选 <select id="forecast"><option value="proposals">原 PDM＋LQR</option><option value="frontaxle">前轴反馈候选</option><option value="quintic">连续起点参考候选</option></select></label><label><input type="checkbox" id="show" checked>显示预测</label>
<canvas id="map" width="1100" height="580"></canvas><p class="legend">绿色区域：路线车道 19820；红色区域：对向车道 19814。黄色：实际学生轨迹与车身；青色虚线：未准入的 4 秒预测。坐标来自同一 nuPlan 地图与世界状态。</p>
<button id="play">播放</button><input id="tick" aria-label="实际学生时间轴" type="range" min="75" max="140" value="95"><output id="time"></output><p id="details"></p><p id="construction"></p><a href="direction-view-data.json">原始地图与轨迹证据</a></section>
<section><h2>有限候选搜索</h2><p>历史候选集逐版保留；正式后端采用 24 个前轴反馈＋54 个连续起点参考候选。动态约束不支持的候选明确拒绝；所有候选继续要求原有方向、道路、碰撞与 TTC 条件。</p><table id="searches"><thead><tr><th>原锚点</th><th>候选族</th><th>可完成预测／构造拒绝</th><th>全部检查通过</th><th>证据</th></tr></thead><tbody></tbody></table></section>
<section><h2>执行记录身份核查</h2><p>v1 保留接口白名单拒绝记录；v2 已实际调用新搜索教师，但旧回放标签仍显示 PDM。真实后端以 decisions.json 的 repair_evidence.backend 为准。标签已修复，以下逐步重放验证不改写原任务产物。</p><a href="execution-audit.json">修复后逐步一致性核查</a><a href="execution-audit-95/viewer.html?actor=ego&tick=95">9.5 秒锚点身份修复后回放</a><a href="execution-audit-100/viewer.html?actor=ego&tick=100">10 秒锚点身份修复后回放</a></section><div id="runs"></div><section><h2>解释边界</h2><p>地图核查确认实际学生进入了方向相反的车道。原居中参考轨迹与实际后轴起点有偏差，跟踪器需要消化误差；扩展搜索从实际状态推进，保留各候选失败原因。当前没有证据证明这两个状态已可靠纠正，也不能据有限搜索宣布物理无解。</p><p>合格后才会执行同父模型、同预算配对训练，跨进程续训、固定独立长测与条件晋级。页面按实际任务回执展示，未执行阶段不会写成已完成。</p><a href="attribution-report.json">同锚点对照与原因分类</a><a href="implementation-ledger.json">实现与验收记录</a><a href="../active-recovery-v3/index.html">上一轮完整失败回放</a></section></main>
<script>
let data,chosen=0,playing=false;const by=id=>document.getElementById(id);const ctx=by('map').getContext('2d');
function line(points,color,width=1,dash=[]){ctx.beginPath();points.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])}
let bounds;function xy(p){return [550+(p[0]-bounds[0])*bounds[2],290-(p[1]-bounds[1])*bounds[2]]}
function draw(){const c=data.cases[chosen],tick=+by('tick').value,a=c.actual.find(x=>x.tick===tick),anchor=c.actual.find(x=>x.tick===c.anchor);bounds=[anchor.center[0],anchor.center[1],10];ctx.clearRect(0,0,1100,580);
for(const l of c.lanes){const pts=l.polygon.map(xy);ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle=l.id==='19820'?'#204d43':l.id==='19814'?'#503136':'#243441';ctx.fill();line(l.polygon,'#51616e');line(l.baseline,l.id==='19820'?'#62d8a1':l.id==='19814'?'#f09494':'#637987',1,[4,4]);let n=0;for(let j=1;j<l.baseline.length;j++)if(Math.hypot(l.baseline[j][0]-bounds[0],l.baseline[j][1]-bounds[1])<Math.hypot(l.baseline[n][0]-bounds[0],l.baseline[n][1]-bounds[1]))n=j;const a=l.baseline[Math.max(0,n-1)],b=l.baseline[Math.min(l.baseline.length-1,n+1)],p=l.baseline[n],theta=Math.atan2(b[1]-a[1],b[0]-a[0]);const mid=xy(p);ctx.fillStyle='#e1eaf2';ctx.font='13px system-ui';ctx.fillText(l.id,mid[0]+12,mid[1]);line([[p[0]-2*Math.cos(theta-.45),p[1]-2*Math.sin(theta-.45)],p,[p[0]-2*Math.cos(theta+.45),p[1]-2*Math.sin(theta+.45)]],'#d9e5ec',2)}
line(c.actual.map(x=>x.center),'#ddba63',2);if(by('show').checked){const f=c.forecasts.find(x=>x.family===by('forecast').value);line(f.centers,'#5be2e4',3,[7,5])}
line(a.corners,'#fff2aa',3);const q=xy(a.center);line([a.center,[a.center[0]+3*Math.cos(a.center[2]),a.center[1]+3*Math.sin(a.center[2])]],'#fff',3);
ctx.fillStyle='#e0eef5';ctx.font='17px system-ui';ctx.fillText('实际学生 '+(tick/10).toFixed(1)+' s · 预测起点 '+(c.anchor/10).toFixed(1)+' s',20,28);by('time').textContent=(tick/10).toFixed(1)+' 秒';const t=c.diagnosis.trace.find(x=>x.tick===tick);by('details').textContent=t?'官方道路关联：'+t.associated.map(x=>x.id).join(', ')+'；实际速度 '+t.speed.toFixed(2)+' m/s。':'该时刻请通过完整实际回放检查。';by('construction').textContent='原居中参考起点与实际后轴相差 '+c.proposal_start_gap_m.toFixed(2)+' m。前轴反馈候选的最大 1 秒对向位移约 '+c.detailed.max_oncoming_in_1s.toFixed(2)+' m；不是合格轨迹。';}
function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}function link(parent,url,text){const a=el('a',text);a.href=url;parent.append(a)}
fetch('gallery-data.json').then(r=>r.json()).then(d=>{data=d;d.cases.forEach((c,i)=>{const o=el('option',(c.anchor/10).toFixed(1)+'秒 · '+c.id);o.value=i;by('case').append(o)});for(const x of d.searches){const tr=el('tr');for(const v of [x.case,x.family,x.supported+' / '+x.rejected,x.admissible])tr.append(el('td',String(v)));const td=el('td');link(td,x.file,'全部评分');tr.append(td);by('searches').tBodies[0].append(tr)}
for(const t of d.tasks){const s=el('section');s.append(el('h2',t.id));s.append(el('p','任务状态：'+t.status));if(t.correction.branches?.teacher?.teacher?.backend)s.append(el('p','教师实际后端：'+t.correction.branches.teacher.teacher.backend));if(t.search?.attempted_schedules){s.append(el('p','本次实际搜索：'+t.search.attempted_schedules+' 个候选；完成预测 '+t.search.proposals+'；通过全部预测检查 '+t.search.admissible_proposals+'；搜索耗时 '+t.search.search_wall_seconds.toFixed(2)+' 秒。'));link(s,'tasks/'+t.id+'/correct/teacher/decisions.json','搜索与实际动作记录')}const chips=el('div');chips.className='chips';t.stages.forEach(x=>chips.append(el('span',x.id+'：'+x.status)));s.append(chips);for(const [role,b] of Object.entries(t.correction.branches||{})){const e=b.execution;const p=el('p',role+'：共同历史 '+e.anchor_tick/10+' s；实际后续 '+e.suffix_steps/10+' s；完整窗口 '+e.complete+'；准入 '+b.admission.accepted+'；原因 '+b.admission.reasons.join('、')+'；执行信息 '+(e.error||'无异常'));s.append(p);link(s,'tasks/'+t.id+'/correct/'+role+'/viewer.html?actor=ego&tick='+e.anchor_tick,role+' 实际回放');link(s,'tasks/'+t.id+'/correct/'+role+'/report.json','原始回执')}link(s,'tasks/'+t.id+'/report.json','完整流程证据');by('runs').append(s)}draw()}).catch(e=>{by('details').textContent='证据加载失败：'+e;});
by('case').onchange=()=>{chosen=+by('case').value;by('tick').value=data.cases[chosen].anchor;draw()};by('forecast').onchange=draw;by('tick').oninput=draw;by('show').onchange=draw;by('play').onclick=()=>{playing=!playing;by('play').textContent=playing?'暂停':'播放'};setInterval(()=>{if(playing&&data){let v=+by('tick').value;by('tick').value=v>=140?75:v+1;draw()}},100);
</script></html>"""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    export(p.parse_args().root)
