"""Publish bounded public evidence for a full NAVSIM evaluation/mining campaign."""
import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from driversi.runtime.budget import write_json


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChainFlow 9405 · 正式 navtest 与训练来源挖掘</title>
<style>body{font:16px/1.65 system-ui;margin:0;background:#101b2a;color:#e8edf5}main{max-width:1120px;margin:auto;padding:32px}h1{font-size:30px}a{color:#8ed8ff}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}section{background:#19283b;border:1px solid #30435b;border-radius:12px;padding:20px;margin:18px 0}.big{font-size:28px;font-weight:700;color:#78dfd0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #34465c;overflow-wrap:anywhere}.muted{color:#acbcd0}.warn{color:#ffc675}progress{width:100%;height:18px}small{word-break:break-all}</style>
<main><p class="muted">DriveRSI · 9405 官方 stage1 · 数据分区校正实验</p><h1>正式 navtest → 能力汇总 → navtrain 候选确认</h1>
<p>原生模型、loss、PDM 保持不变。测试结果用于适应性开发，测试图像、轨迹标签和场景身份不进入训练侧检索条件。</p>
<div class="cards"><section><div>正式 navtest 评分</div><div class="big" id="done">—</div><progress id="bar"></progress><div id="errors"></div></section><section><div>完整 navtrain 输入盘点</div><div class="big" id="train">—</div><div id="trainstatus"></div></section><section><div>相同 9405 权重基准</div><div class="big" id="score">等待完整覆盖</div><div id="delta"></div></section></div>
<section><h2>当前执行状态</h2><div id="state"></div><p id="scope" class="warn"></p><small id="updated" class="muted"></small></section>
<section><h2>原生评分子项</h2><table><thead><tr><th>指标</th><th>完整 navtest 结果（×100）</th></tr></thead><tbody id="metrics"></tbody></table><p class="muted">运行异常不当作驾驶零分；覆盖不全时不展示正式总分。总分参考 94.05，不为未公布的子项编造参考值。</p></section>
<section><h2>能力与条件汇总</h2><p id="shortfalls"></p><table><thead><tr><th>能力子项</th><th>未满分场景数</th><th>所列条件中未满分比例最高的一组</th></tr></thead><tbody id="weaknesses"></tbody></table><p class="muted">子项小于 1 是筛查信号，不等于总分低于 95，也不直接证明模型原因。条件分组互有重叠；训练侧只收到类型与统计，不收到测试场景身份或标签。</p></section>
<section><h2>训练来源中的实际候选</h2><p>最多 16 个候选，每个原始日志组最多一个。检索匹配只是场景条件相似；必须由同一检查点实际运行确认短板，再做同锚点归因和纠正。</p><div style="overflow:auto"><table><thead><tr><th>navtrain token / 来源</th><th>学生 PDMS</th><th>实测不足子项</th><th>处理</th></tr></thead><tbody id="candidates"></tbody></table></div></section>
<section><h2>针对已确认短板的真实物理分支</h2><p id="teacher-state">尚未执行</p><div style="overflow:auto"><table><thead><tr><th>训练 token</th><th>教师</th><th>反事实 PDMS</th><th>准入与回放</th></tr></thead><tbody id="teachers"></tbody></table></div><p class="muted">保留固定控制、原生 DriveZero 和走廊恢复的尝试。最新原生到达方式先实际执行 0.8 秒预热，再由 DriveZero 执行 1.2 秒，形成连续历史；不插值伪造策略观测。从相同锚点推演五秒未来，对照延续 DriveZero，走廊恢复使用训练来源未来作特权教师参考。这些对照不代表 ChainFlow 的反事实表现；规则交通不称为学习型自博弈。原生 PDMS 评分四秒，完整五秒未来另行保留执行和接触检查。</p></section>
<section><h2>后续数据与训练约定</h2><p>DriveZero 教师或受约束扰动生成连续新历史，保存 4 帧历史＋10 个真实未来点。纠正分支 PDMS ≥ 95，再由现有 GS 渲染并检查物理对齐，进入同父模型、同更新预算的配对训练。完整训练来源池不等于已经完整训练一遍。</p><p class="warn">本页不会把高分教师轨迹、任务执行完成或候选匹配当作学生收益或模型晋级。</p><a href="../chainflow-9405-rsi-v1/index.html">此前小样本工程试验（非 navtest 成绩）</a></section></main>
<script>
const names={score:'PDMS',no_at_fault_collisions:'无责任碰撞',drivable_area_compliance:'可行驶区域合规',driving_direction_compliance:'行驶方向合规',ego_progress:'路线进度',time_to_collision_within_bound:'TTC',comfort:'舒适性'};
const e=id=>document.getElementById(id);function td(tr,value){let d=document.createElement('td');d.textContent=value;tr.append(d)}
async function refresh(){try{const d=await(await fetch('data.json?t='+Date.now(),{cache:'no-store'})).json();e('done').textContent=d.scored+' / '+d.expected;e('bar').max=d.expected;e('bar').value=d.scored;e('errors').textContent='运行异常：'+d.failed;e('train').textContent=d.train_available+' / '+d.train_expected;e('trainstatus').textContent=d.train_complete?'官方训练来源完整覆盖':'仍在盘点日志、图像与缓存';e('score').textContent=d.complete?d.pdms.toFixed(4):'等待完整覆盖';e('delta').textContent=d.complete?'相对发布参考94.05：'+(d.pdms-94.05).toFixed(4):'不会使用部分样本均值冒充正式结果';e('state').textContent=d.status+' · '+d.phase+(d.error?' · '+d.error:'');e('scope').textContent='本轮新增训练更新：0；尚未生成新的合成准入数据或晋级模型。';e('updated').textContent='更新：'+d.updated;e('metrics').replaceChildren();for(const [key,name] of Object.entries(names)){let tr=document.createElement('tr');td(tr,name);td(tr,d.complete&&d.metrics[key]!=null?(d.metrics[key]*100).toFixed(4):'待完成');e('metrics').append(tr)}e('candidates').replaceChildren();for(const r of d.candidates){let tr=document.createElement('tr');td(tr,r.token+' / '+r.source_log);td(tr,r.score==null?'未完成':(r.score*100).toFixed(2));td(tr,(r.confirmed_capability_shortfalls||[]).join('、')||'尚无确认短板');td(tr,r.action);e('candidates').append(tr)}if(!d.candidates.length){let tr=document.createElement('tr');td(tr,'候选确认尚未完成，暂不展示推测结果');e('candidates').append(tr)}}catch(err){e('state').textContent='状态读取失败，请稍后刷新'}}refresh();setInterval(refresh,15000);
</script></html>'''

HTML = HTML.replace("e('metrics').replaceChildren();", """e('shortfalls').textContent='全量零分：'+d.zero_score_count+'；PDMS 低于 95：'+d.below_95_count;e('weaknesses').replaceChildren();const caps={collision:'碰撞',comfort:'舒适性',direction_compliance:'方向',interaction:'TTC',road_compliance:'道路',route_progress:'进度'};for(const r of d.capability_summary){let tr=document.createElement('tr');td(tr,caps[r.capability]||r.capability);td(tr,r.failures+' / '+r.count);td(tr,r.condition+'：'+r.condition_failures+' / '+r.condition_count+' ('+(100*r.condition_failures/r.condition_count).toFixed(2)+'%)');e('weaknesses').append(tr)}e('metrics').replaceChildren();""")

HTML = HTML.replace("e('candidates').replaceChildren();", """e('teacher-state').textContent='状态：'+d.teacher_status+'；通过轨迹准入：'+d.qualified_trajectories+'；视觉及训练准入仍需单独完成';e('teachers').replaceChildren();for(const t of d.teachers){let tr=document.createElement('tr');td(tr,t.token);td(tr,t.backend);td(tr,t.score==null?'未完成':(t.score*100).toFixed(2));let cell=document.createElement('td');cell.textContent=(t.trajectory_admitted?'轨迹通过':t.status==='completed'?'执行完成，轨迹未通过':t.status)+(t.failure?' · '+t.failure:'');for(const [label,href] of Object.entries(t.views)){let a=document.createElement('a');a.textContent=' · '+label;a.href=href;cell.append(a)}tr.append(cell);e('teachers').append(tr)}e('candidates').replaceChildren();""")

HTML = HTML.replace('<p id="shortfalls"></p>', '<p id="shortfalls"></p><p><a href="navtest-failures.html">查看全部零分／低分测试场景及子项</a> · <a href="navtest-low-scores.csv" download>下载低分清单 CSV</a></p>')
HTML = HTML.replace('最多 16 个候选，每个原始日志组最多一个。', '<span id="retrieval-coverage">读取候选覆盖</span> 每个原始日志组最多一个。')
HTML = HTML.replace('<th>实测不足子项</th>', '<th>检索目标 → 实测不足子项</th>')
HTML = HTML.replace("td(tr,(r.confirmed_capability_shortfalls||[]).join('、')||'尚无确认短板');", "td(tr,(r.retrieval_target_capability||'综合排序')+' → '+((r.confirmed_capability_shortfalls||[]).join('、')||'尚无确认短板'));")
HTML = HTML.replace("e('state').textContent=d.status", "e('retrieval-coverage').textContent='当前批次选中 '+d.retrieval_count+' 个候选；实测完成 '+d.confirmation_count+' 个。';e('state').textContent=d.status")

FAILURES_HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>navtest 零分与低分完整清单</title><style>body{font:16px/1.6 system-ui;background:#101b2a;color:#e8edf5;margin:0}main{max-width:1250px;margin:auto;padding:28px}a{color:#8ed8ff}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #34465c;text-align:left;overflow-wrap:anywhere}select,input,button{font:inherit;padding:8px;margin:8px 8px 8px 0}button:disabled{opacity:.4}.warn{color:#ffc675}.scroll{overflow:auto}small{color:#acbcd0}</style>
<main><a href="index.html">返回实验总览</a><h1>navtest 零分与低分完整清单</h1>
<p class="warn">仅供评测核查。测试 token、图像和轨迹标签不作为训练候选；跨划分检索只使用能力与条件汇总。</p>
<p id="summary">正在读取完整清单</p><a href="navtest-low-scores.csv" download>下载全部低于 95 分场景 CSV</a>
<div><label>得分区间 <select id="band"><option value="zero">零分</option><option value="0-80">大于 0、低于 80</option><option value="80-95">80 至低于 95</option><option value="all">所有低于 95</option></select></label>
<label>子项不足 <select id="metric"><option value="all">全部</option><option value="no_at_fault_collisions">碰撞</option><option value="drivable_area_compliance">道路</option><option value="driving_direction_compliance">方向</option><option value="ego_progress">进度</option><option value="time_to_collision_within_bound">TTC</option><option value="comfort">舒适性</option></select></label><input id="query" placeholder="筛选 token 或来源日志" aria-label="筛选 token 或来源日志"></div>
<p id="count"></p><button id="prev">上一页</button><button id="next">下一页</button>
<div class="scroll"><table><thead><tr><th>测试 token／来源日志</th><th>PDMS</th><th>未满分子项（×100）</th><th>可观察条件</th></tr></thead><tbody id="rows"></tbody></table></div>
<small>每页 50 条。子项不足可以重叠；该表记录评分证据，不自动判定失败根因。</small></main>
<script>const $=id=>document.getElementById(id);let data,page=0;const names={no_at_fault_collisions:'碰撞',drivable_area_compliance:'道路',driving_direction_compliance:'方向',ego_progress:'进度',time_to_collision_within_bound:'TTC',comfort:'舒适性'};function render(){if(!data)return;const band=$('band').value,metric=$('metric').value,q=$('query').value.trim().toLowerCase();const rows=data.rows.filter(r=>(band==='all'||r.band===band)&&(metric==='all'||r.metrics[metric]<1)&&(!q||(r.token+' '+r.source_log).toLowerCase().includes(q)));page=Math.min(page,Math.max(0,Math.ceil(rows.length/50)-1));$('count').textContent='筛选命中 '+rows.length+' 条 · 第 '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/50))+' 页';$('prev').disabled=page===0;$('next').disabled=(page+1)*50>=rows.length;$('rows').replaceChildren();for(const r of rows.slice(page*50,(page+1)*50)){const tr=document.createElement('tr');for(const v of [r.token+' / '+r.source_log,(r.score*100).toFixed(4),Object.entries(names).filter(([k])=>r.metrics[k]<1).map(([k,n])=>n+': '+(r.metrics[k]*100).toFixed(2)).join(' · '),r.conditions.join(' · ')]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('rows').append(tr)}}for(const id of ['band','metric','query'])$(id).addEventListener('input',()=>{page=0;render()});$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};fetch('navtest-failures.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{data=d;$('summary').textContent='全量 '+d.expected+' 个场景；零分 '+d.bins.zero+'；0～80 分 '+d.bins['0-80']+'；80～95 分 '+d.bins['80-95']+'；低于 95 合计 '+d.rows.length+'。区间均不含上界。';render()}).catch(err=>{$('summary').textContent='清单尚未生成或读取失败：'+err.message});</script></html>'''


def publish_failures(campaign, destination, summary):
    """Keep evaluator-only failures in the internal audit directory."""
    if not summary.get('complete'):
        return None
    source = campaign / 'baseline9405/per-scene.csv'
    descriptor_path = campaign / 'navtest-descriptors.json'
    fingerprint = dict(inputs=[dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                              for p in (source, descriptor_path)], identity=summary['identity'])
    if read(destination / 'source-identity.json', None) == fingerprint:
        cached = read(destination / 'navtest-failures.json', None)
        if cached is not None:
            return cached
    descriptors = {r['token']: r for r in read(campaign / 'navtest-descriptors.json', {})['rows']}
    metrics = ('no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
               'ego_progress', 'time_to_collision_within_bound', 'comfort')
    with (campaign / 'baseline9405/per-scene.csv').open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != summary['expected'] or len({r['token'] for r in raw}) != len(raw):
        raise ValueError('Full unique evaluation inventory required for failure page')
    rows, bins = [], dict(zero=0, **{'0-80': 0, '80-95': 0, '95+': 0})
    for r in raw:
        if r['valid'].lower() != 'true':
            raise ValueError('Runtime errors must not be published as driving zero scores')
        meta = descriptors[r['token']]
        if r['log_name'] != meta['source_log']:
            raise ValueError('Evaluation descriptor/source mismatch')
        values = {k: float(r[k]) for k in ('score', *metrics)}
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values.values()):
            raise ValueError('Invalid native metric in failure inventory')
        score = values.pop('score')
        band = 'zero' if score == 0 else '0-80' if score < .8 else '80-95' if score < .95 else '95+'
        bins[band] += 1
        if score < .95:
            rows.append(dict(token=r['token'], source_log=r['log_name'], score=score,
                             metrics=values, conditions=meta['conditions'], band=band))
    rows.sort(key=lambda r: (r['score'], r['token']))
    value = dict(expected=summary['expected'], bins=bins, rows=rows,
                 purpose='evaluator_only_failure_audit_not_training_candidates',
                 checkpoint_sha256=summary['identity']['checkpoint_sha256'])
    write_json(destination / 'navtest-failures.json', value)
    (destination / 'navtest-failures.html').write_text(FAILURES_HTML, encoding='utf-8')
    with (destination / 'navtest-low-scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['token', 'source_log', 'score', *metrics])
        writer.writeheader()
        writer.writerows(dict(token=r['token'], source_log=r['source_log'], score=r['score'],
                              **r['metrics']) for r in rows)
    write_json(destination / 'source-identity.json', fingerprint)
    return value


def read(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


TRAIN_HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全量 navtrain 原生挖掘</title>
<style>body{font:16px/1.7 system-ui;background:#101b2a;color:#e8edf5;margin:0}main{max-width:1180px;margin:auto;padding:28px}a{color:#8ed8ff}section{background:#19283b;padding:20px;border-radius:12px;margin:20px 0}progress{width:100%}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #34465c;overflow-wrap:anywhere}input,select,button{padding:8px;font:inherit;margin:8px}.warn{color:#ffc675}.big{font-size:28px;color:#78dfd0}</style>
<main><a href="index.html">返回平台总览</a><h1>完整 navtrain → 实测短板 → 待扩充 token</h1>
<p>冻结 ChainFlow stage1 9405，直接使用训练池自身的预测和原生 PDM 子项。此流程不读取 navtest 逐样本结果或测试反馈优先级。</p>
<section><div class="big" id="coverage">读取进度</div><progress id="bar"></progress><div id="status"></div><p id="counts"></p><p class="warn">这些是训练来源的挖掘分数，不是测试成绩。低分样本仍需同锚点归因、教师 PDMS ≥95、物理及视觉准入，才可进入合成训练。</p></section>
<section><h2>待扩充训练样本</h2><a href="navtrain-augmentation-tokens.csv" download>下载当前全部候选 token CSV</a><p>完整原始日志和历史保持追溯；历史缺口先修复，运行异常另行记录。</p>
<label>检索 <input id="q" placeholder="训练 token 或来源日志"></label><label>能力 <select id="cap"><option value="all">全部</option><option value="collision">碰撞</option><option value="road_compliance">道路</option><option value="direction_compliance">方向</option><option value="interaction">TTC</option><option value="route_progress">进度</option><option value="comfort">舒适性</option></select></label>
<p id="shown"></p><button id="prev">上一页</button><button id="next">下一页</button><div style="overflow:auto"><table><thead><tr><th>训练 token／来源</th><th>PDMS</th><th>实测子项不足</th><th>下一步</th></tr></thead><tbody id="rows"></tbody></table></div></section></main>
<script>const $=id=>document.getElementById(id),names={collision:'碰撞',road_compliance:'道路',direction_compliance:'方向',interaction:'TTC',route_progress:'进度',comfort:'舒适性'};let d,page=0;function render(){if(!d)return;const c=d.plan.coverage,q=$('q').value.toLowerCase(),cap=$('cap').value;const rows=d.plan.augmentation_candidates.filter(r=>(cap==='all'||r.observed_shortfalls.includes(cap))&&(!q||(r.token+' '+r.source_log).toLowerCase().includes(q)));page=Math.min(page,Math.max(0,Math.ceil(rows.length/50)-1));$('coverage').textContent=c.scored+' / '+c.expected+' 场景完成原生评分';$('bar').max=c.expected;$('bar').value=c.scored;$('status').textContent='执行状态：'+d.state.status+'；运行异常：'+c.runtime_failures+'；尚未记录：'+c.missing;$('counts').textContent='待归因与扩充候选：'+d.plan.augmentation_candidates.length+'；总分低于95：'+(d.plan.bins.zero+(d.plan.bins['0-80']||0)+(d.plan.bins['80-95']||0))+'；其中零分：'+d.plan.bins.zero+'。未完成全量时不外推总体失败比例。';$('shown').textContent='筛选 '+rows.length+' 条，第 '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/50))+' 页';$('prev').disabled=page===0;$('next').disabled=(page+1)*50>=rows.length;$('rows').replaceChildren();for(const r of rows.slice(page*50,(page+1)*50)){let tr=document.createElement('tr');for(const v of [r.token+' / '+r.source_log,r.pdms.toFixed(2),r.observed_shortfalls.map(c=>names[c]||c).join('、'),r.history_continuous?'同锚点教师／平台归因':'修复历史支持']){let td=document.createElement('td');td.textContent=v;tr.append(td)}$('rows').append(tr)}}for(const id of ['q','cap'])$(id).oninput=()=>{page=0;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};async function refresh(){try{d=await(await fetch('navtrain-mining.json?t='+Date.now(),{cache:'no-store'})).json();render()}catch(e){$('status').textContent='进度暂不可用：'+e.message}}refresh();setInterval(refresh,20000);</script></html>'''


def publish_train_mining(folder, destination):
    state = read(folder / 'state.json', {'status': 'not_started'})
    plan = read(folder / 'augmentation-plan.json', dict(coverage=dict(
        expected=103288,scored=0,missing=103288,runtime_failures=0),
        augmentation_candidates=[],bins=dict(zero=0),test_feedback_used=False))
    if plan.get('test_feedback_used') is not False:
        raise ValueError('Train-only publication requires a train-only plan')
    write_json(destination / 'navtrain-mining.json', dict(state=state,plan=plan))
    (destination / 'navtrain-mining.html').write_text(TRAIN_HTML, encoding='utf-8')
    with (destination / 'navtrain-augmentation-tokens.csv').open('w', newline='') as stream:
        keys=['token','source_log','pdms','history_continuous','next_action']
        writer=csv.DictWriter(stream, fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in plan['augmentation_candidates'])
    return state


def publish(campaign, destination):
    campaign, destination = Path(campaign), Path(destination)
    state = read(campaign / "postprocess-state.json", {})
    train = read(campaign / "navtrain/coverage.json", {})
    train_progress = read(campaign / "navtrain/progress.json", {})
    summary = read(campaign / "baseline9405/summary.json", {})
    progress = [read(p, {}) for p in sorted((campaign / "baseline9405").glob("shard-*/progress.json"))]
    # A terminal shard may have up to nine more rows than its last progress tick.
    for i, p in enumerate(sorted((campaign / "baseline9405").glob("shard-*/progress.json"))):
        terminal = read(p.parent / "summary.json", {})
        if terminal:
            progress[i] = dict(recorded=terminal["succeeded"] + terminal["failed"], failed=terminal["failed"])
    task = Path(state.get("task", "full-navtest-to-navtrain-v1")).name
    decision = read(campaign / "tasks" / task / "decision/report.json", {})
    retrieval = read(campaign / "tasks" / task / "retrieve/report.json", {})
    receipt = read(campaign / "tasks" / task / "confirm/native/receipt.json", {})
    observations = {r['token']: r for r in receipt.get('rows', [])}
    candidates = decision.get('decisions', [])
    if not candidates:
        metric_caps = dict(no_at_fault_collisions='collision', drivable_area_compliance='road_compliance',
                           driving_direction_compliance='direction_compliance', ego_progress='route_progress',
                           time_to_collision_within_bound='interaction', comfort='comfort')
        for row in retrieval.get('candidates', []):
            observed = observations.get(row['token'], {})
            values = observed.get('metrics', {})
            candidates.append(dict(token=row['token'], source_log=row['source_log'],
                retrieval_target_capability=row['match'].get('target_capability'),
                score=values.get('score'), confirmed_capability_shortfalls=[
                    cap for metric,cap in metric_caps.items() if values.get(metric, 1) < 1],
                action=observed.get('classification') or observed.get('status') or '等待模型确认'))
    analysis = read(campaign / "evaluator-analysis-preview.json", {})
    capability_summary = []
    for item in analysis.get("capability_feedback", {}).get("slices", []):
        cohorts = [c for c in analysis.get("cohorts", [])
                   if c["capability"] == item["capability"] and c["count"] > 0]
        if cohorts:
            strongest = max(cohorts, key=lambda c: c["failures"] / c["count"])
            capability_summary.append(dict(**item, condition=" + ".join(strongest["conditions"]),
                                           condition_count=strongest["count"],
                                           condition_failures=strongest["failures"]))
    teacher_reports = [(p.parent, read(p, {})) for p in sorted(campaign.glob("targeted-teachers*/report.json"))]
    teacher_status = "running" if any(d.get("status") == "running" for _,d in teacher_reports) else "completed" if teacher_reports else "not_started"
    teachers = []
    for teacher_root, teacher_report, case in [(root, d, c) for root,d in teacher_reports for c in d.get("rows", [])]:
        attempts = case.get("attempts", []) or [dict(backend="preparing", status=case.get("status", "running"), failure=case.get("error"))]
        for attempt in attempts:
            views = {}
            task_id = attempt.get("task")
            failure = attempt.get("failure")
            if task_id:
                if Path(task_id).name != task_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task_id):
                    raise ValueError("Unsafe teacher task identity")
                raw = read(teacher_root / "tasks" / task_id / "teacher/execution/report.json", {})
                # Partial worker evidence explains failure; it never upgrades the
                # sealed task's admission or provides an accepted score.
                if raw.get("reason"):
                    failure = (failure + " · " if failure else "") + raw["reason"]
                for branch in ("baseline", "teacher", "arrival-failure", "partial-failure"):
                    source = teacher_root / "tasks" / task_id / "teacher/execution" / (branch + ".html")
                    if source.is_file():
                        relative = Path("replays") / teacher_root.name / task_id / (branch + ".html")
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists() or target.stat().st_size != source.stat().st_size:
                            shutil.copyfile(source, target)
                        views[branch] = relative.as_posix()
            score = attempt.get("score") or {}
            teachers.append(dict(token=case["token"],backend=teacher_report.get("arrival_backend", "bounded_control") + " → " + attempt["backend"],score=score.get("score"),
                trajectory_admitted=attempt.get("trajectory_admitted", False),
                status=attempt.get("status", "pending"),failure=failure,views=views))
    payload = dict(updated=datetime.now(timezone.utc).isoformat(),
                   expected=12146, scored=sum(p.get("recorded", 0) for p in progress),
                   failed=sum(p.get("failed", 0) for p in progress),
                   train_available=train.get("available", train_progress.get("available", 0)),
                   train_expected=103288, train_complete=train.get("complete", False),
                   complete=summary.get("complete", False), pdms=summary.get("pdms"),
                   metrics=summary.get("metrics", {}) if summary.get("complete") else {},
                   status=state.get("status", "running"), phase=state.get("phase", "baseline_evaluation"),
                   error=state.get("error", state.get("failure_reason")),
                   candidates=candidates, optimizer_updates=0, promoted=False,
                   retrieval_count=len(retrieval.get('candidates', [])), confirmation_count=len(observations),
                   zero_score_count=analysis.get("zero_score_count", "未完成"),
                   below_95_count=analysis.get("below_95_count", "未完成"),
                   capability_summary=capability_summary,
                   teachers=teachers,teacher_status=teacher_status,
                   qualified_trajectories=sum(d.get("qualified_trajectories",0) for _,d in teacher_reports))
    destination.mkdir(parents=True, exist_ok=True)
    train_mining_state = publish_train_mining(campaign.parent / 'chainflow-navtrain-mining-v1', destination)
    payload['train_mining_status'] = train_mining_state['status']
    audit = campaign / 'evaluator-only-audit'
    audit.mkdir(parents=True, exist_ok=True)
    failures = publish_failures(campaign, audit, summary)
    if failures is not None:
        payload.update(zero_score_count=failures['bins']['zero'], below_95_count=len(failures['rows']))
    public_html = HTML.replace('<p><a href="navtest-failures.html">查看全部零分／低分测试场景及子项</a> · <a href="navtest-low-scores.csv" download>下载低分清单 CSV</a></p>',
        '<p>测试逐场景记录仅保留在服务器内部审计目录。<a href="navtrain-mining.html">查看全量 navtrain 挖掘进度与待扩充清单</a></p>')
    (destination / "index.html").write_text(public_html, encoding="utf-8")
    # Previously published evaluator rows must not remain reachable after the
    # user's explicit boundary change. Archive them outside the web root.
    archive = campaign / 'withdrawn-evaluator-publication'
    archive.mkdir(exist_ok=True)
    for name in ('navtest-failures.json', 'navtest-low-scores.csv', 'navtest-failures.html'):
        old = destination / name
        if old.exists() and not (archive / name).exists():
            shutil.move(str(old), str(archive / name))
    (destination / 'navtest-failures.html').write_text(
        '<!doctype html><meta charset="utf-8"><h1>测试明细已收回至内部审计</h1>'
        '<p>展示平台不再提供测试 token、场景条件或逐样本分数。训练挖掘直接分析 navtrain 自身。</p>'
        '<a href="navtrain-mining.html">查看 navtrain 挖掘</a> · <a href="index.html">返回总览</a>', encoding='utf-8')
    write_json(destination / "data.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--until-terminal", action="store_true")
    args = parser.parse_args()
    import time
    started = time.monotonic()
    while True:
        data = publish(args.campaign, args.destination)
        if not args.until_terminal or (data["status"] != "running" and data["teacher_status"] != "running"
                                      and data['train_mining_status'] not in ('preparing','running')) or time.monotonic() - started >= 172800:
            break
        time.sleep(15)
    print(json.dumps(data, ensure_ascii=False))
