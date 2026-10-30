"""Export only training-side expansion evidence; does not grant admission."""
import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np


def run(source, output):
    source, output = source.resolve(), output.resolve()
    plan = json.loads((source / 'plan.json').read_text())
    result_path = source / 'report.json'
    final = result_path.exists()
    report = json.loads((result_path if final else source / 'progress.json').read_text())
    if plan.get('scored_student') or plan.get('source', {}).get('source_log', '').startswith('navtest'):
        raise ValueError('Only the score-free training expansion report is supported')
    output.mkdir(parents=True, exist_ok=True)
    rows, origin = [], None
    for item in report['rows']:
        i = item['index']
        task = source / 'tasks' / item['task']
        execution = task / 'teacher/execution'
        row = {k: item.get(k) for k in ('index', 'pdms', 'status', 'reason', 'teacher_qualified',
                                      'distinct', 'rendered', 'visual_admitted', 'trainable')}
        row.update(plan['candidates'][i])
        if (execution / 'report.json').exists():
            physical = json.loads((execution / 'report.json').read_text())
            row['branches'] = physical.get('branches', {})
            row['wall_seconds'] = physical.get('wall_seconds')
            row['reason'] = physical.get('branches', {}).get('teacher', {}).get('failure') or row['reason']
        if (execution / 'teacher.npz').exists():
            with np.load(execution / 'teacher.npz', allow_pickle=False) as z:
                xy = np.asarray(z['states'][:, list(z['actor_ids']).index('ego'), :2])
                if origin is None:
                    origin = xy[0]
                row['xy'] = (xy - origin).round(4).tolist()
        if (execution / 'teacher.html').exists():
            name = f'branch-{i:02d}.html'
            shutil.copyfile(execution / 'teacher.html', output / name)
            row['replay'] = name
        inspection = source.parent / 'reviews' / f'{i:02d}' / 'inspection'
        if (inspection / 'viewer.html').exists():
            shutil.copytree(inspection, output / f'images-{i:02d}', dirs_exist_ok=True)
            row['images'] = f'images-{i:02d}/viewer.html'
        rows.append(row)
    evidence = dict(source_log=plan['source']['source_log'], target=plan['target_qualified_samples'],
                    status=report['status'], final=final, completed=len(rows), planned=len(plan['candidates']),
                    native_future_points=plan['future_frames'], rows=rows,
                    counts={k: report.get(k) for k in ('teacher_qualified', 'distinct_qualified_trajectories',
                                                      'rendered', 'visually_admitted', 'trainable_samples')})
    (output / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    payload = json.dumps(evidence, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>每源 20 条 · DriveRSI 扩充验证</title><style>
body{font-family:system-ui;margin:0;background:#f3f6fa;color:#182a3c}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:30px;margin-bottom:8px}.muted{color:#536879;line-height:1.7}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.card{background:white;padding:18px 24px;border-radius:12px;min-width:125px}.card b{display:block;font-size:26px;color:#087b7c}.note{background:#fff3d8;padding:16px;border-left:4px solid #d69b25;line-height:1.7}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:24px 0}.panel{background:white;border-radius:12px;padding:20px}canvas{width:100%;height:350px;border:1px solid #d5dfe7}select{font-size:16px;padding:8px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #e1e8ed}th{background:#dfebef}a{color:#006d7a}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.table{overflow:auto}.pass{color:#087b7c}.fail{color:#a45c1c}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:16px}}
</style><main><p>DriveRSI · navtrain 来源 · 有限预算试验</p><h1>每源 20 条，按合格样本计数</h1><p class="muted">冻结 DriveRL 教师生成新历史与纠正未来；原评分、物理与视觉准入保持不变。这里没有 navtrain 学生评分。</p><div id="cards" class="cards"></div>
<div class="note">生成尝试、PDMS 合格、轨迹去重、GS 渲染、联合监督准入分别计数。没有达到目标会保留缺口；相似分支不等于独立场景。本轮是固定受约束扰动与原生教师，不是学习型自博弈。</div>
<div class="grid"><div class="panel"><h2>实际执行轨迹</h2><select id="select"></select><p class="muted">局部坐标，单位米；蓝色为新历史，绿色为教师未来。此图只展示轨迹差异，不代表道路几何。</p><canvas id="plot" width="600" height="350"></canvas></div><div class="panel"><h2>选中分支证据</h2><div id="links"></div><pre id="detail"></pre></div></div>
<h2>全部尝试与失败</h2><div class="table"><table><thead><tr><th>分支</th><th>到达方式</th><th>PDMS</th><th>轨迹准入</th><th>不重复</th><th>GS 完成</th><th>训练准入</th><th>记录</th></tr></thead><tbody id="rows"></tbody></table></div><p class="muted" id="scope"></p></main><script>
const D=__DATA__;const el=id=>document.getElementById(id);const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const flag=x=>x===true?'通过':x===false?'未通过':'待处理';
const pairs=[['已处理',D.completed+'/'+D.planned],['每源目标',D.target],['轨迹准入',D.counts.teacher_qualified??D.rows.filter(x=>x.teacher_qualified).length],['不重复',D.counts.distinct_qualified_trajectories??'待汇总'],['已渲染',D.counts.rendered??'待汇总'],['训练准入',D.counts.trainable_samples??0]];el('cards').innerHTML=pairs.map(([k,v])=>`<div class="card">${esc(k)}<b>${esc(v)}</b></div>`).join('');
el('rows').innerHTML=D.rows.map(r=>`<tr><td>${r.index}</td><td>${esc(r.arrival_backend)}<br>${esc(JSON.stringify(r.arrival_action))}</td><td>${r.pdms==null?'未完成':r.pdms.toFixed(3)}</td><td>${flag(r.teacher_qualified)}</td><td>${flag(r.distinct)}</td><td>${flag(r.rendered)}</td><td>${flag(r.trainable)}</td><td>${r.replay?`<a href="${r.replay}">物理回放</a>`:''}<br>${esc(r.reason??'保留原始回执')}</td></tr>`).join('');
el('select').innerHTML=D.rows.map(r=>`<option value="${r.index}">分支 ${r.index} · ${r.pdms==null?'未完成':r.pdms.toFixed(2)+' 分'}</option>`).join('');
function show(){let r=D.rows.find(x=>x.index==el('select').value);if(!r)return;el('links').innerHTML=(r.replay?`<a href="${r.replay}">完整物理回放</a>　`:'')+(r.images?`<a href="${r.images}">四相机像素与物体框核查</a>`:'');el('detail').textContent=JSON.stringify(Object.fromEntries(Object.entries(r).filter(([k])=>!['xy','replay','images'].includes(k))),null,2);const c=el('plot'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const points=r.xy||[];if(!points.length)return;const xs=points.map(x=>x[0]),ys=points.map(x=>x[1]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const scale=Math.min(540/Math.max(1,xmax-xmin),290/Math.max(1,ymax-ymin));const map=p=>[30+(p[0]-xmin)*scale,320-(p[1]-ymin)*scale];ctx.strokeStyle='#e2e8ed';for(let x=0;x<600;x+=30){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,350);ctx.stroke()}for(let y=0;y<350;y+=30){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(600,y);ctx.stroke()}for(let i=1;i<points.length;i++){ctx.strokeStyle=i<=20?'#2778c4':'#087b7c';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...map(points[i-1]));ctx.lineTo(...map(points[i]));ctx.stroke()}ctx.fillStyle='#182a3c';ctx.fillText(`x ${xmin.toFixed(1)}–${xmax.toFixed(1)} m · y ${ymin.toFixed(1)}–${ymax.toFixed(1)} m`,15,18)}el('select').onchange=show;show();el('scope').textContent=`来源：${D.source_log}。状态：${D.status}。AUGUstus 原生 ${D.native_future_points} 个未来点；完整物理未来仍验证 5 秒。未进行新学生训练，未作技能记忆收益结论。`;
</script></html>'''.replace('__DATA__', payload)
    (output / 'index.html').write_text(page)
    return {'output': str(output), 'completed': len(rows), 'final': final}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output)))
