'use strict';

let lastSystemStep = null;
const maturity = {pilot:'小规模已验证', partial:'部分接通 / 待验收', planned:'计划接入', active:'当前推进'};
function architectureEvidence(key, label) {
  if (!key) return '<span class="note">架构说明 · 执行状态见当前任务</span>';
  return '<a href="'+evidenceLink(key)+'">'+esc(label)+' →</a>';
}
function evidenceLink(key) {
  const [kind,id] = key.split(':');
  return kind === 'experiment' ? '#view=experiments&run='+encodeURIComponent(id) : '#view=production&item='+encodeURIComponent(id);
}
function evidenceStatus(key) {
  const [kind,id] = key.split(':');
  const row = (kind === 'experiment' ? data.experiments : data.production).find(r=>r.id===id);
  return row ? labels[row.status] || row.status : '缺少回执';
}
function aggregate(row,label) { return row?.fields.find(f=>f.label===label); }
function renderSystem() {
  const system = data.system;
  if (!system) { $('systemFlow').hidden=true; for(const id of ['systemDate','systemSummary','systemBoundary'])$(id).textContent=''; $('systemContent').textContent='尚未登记架构说明。'; return; }
  $('systemSummary').textContent=system.summary;
  $('systemBoundary').textContent=system.boundary;
  $('systemDate').textContent='架构核对日期 '+system.reviewed_on;
  renderSystemFlow(system);
  $('systemContent').innerHTML = '<details class="module-details"><summary>按源码职责查看六层模块</summary>'+ '<div class="architecture-layers">'+system.layers.map(r=>
    '<article class="panel architecture-layer"><div class="task-top"><h2>'+esc(r.title)+'</h2><span class="badge warn">'+esc(maturity[r.status])+'</span></div><code>'+esc(r.owner)+'</code><p>'+esc(r.description)+'</p><div class="layer-output">产物 · '+esc(r.outputs)+'</div>'+architectureEvidence(r.evidence,'查看关联证据 · '+evidenceStatus(r.evidence))+'</article>'
  ).join('')+'</div><p class="architecture-return">评测收益与成本 → 技能记忆 → 下一轮实验选择</p>'+
  '</details><div class="section-heading"><h2>模型职责与接入边界</h2><span>接入 ≠ 已证明稳定收益</span></div><div class="model-grid">'+system.models.map(r=>
    '<article class="panel"><span class="badge warn">'+esc(maturity[r.status])+'</span><h3>'+esc(r.title)+'</h3><b>'+esc(r.role)+'</b><p>'+esc(r.detail)+'</p></article>'
  ).join('')+'</div><div class="section-heading"><h2>当前要补齐的能力</h2></div><div class="priority-grid">'+system.priorities.map(r=>
    '<article class="panel"><h3>'+esc(r.title)+'</h3><p>'+esc(r.detail)+'</p>'+architectureEvidence(r.evidence,'查看进度与证据')+'</article>'
  ).join('')+'</div><div class="section-heading"><h2>实验与发布约束</h2></div><ul class="contract-list">'+system.contracts.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>';
}
function renderProgress() {
  const baseline=data.experiments.find(r=>r.id==='augustus-navval-auto');
  const done=aggregate(baseline,'已完成')?.value, expected=aggregate(baseline,'预定测试场景')?.value;
  const valid=Number.isFinite(done)&&Number.isFinite(expected)&&expected>0;
  const fraction=valid?Math.min(100,Math.max(0,100*done/expected)):null;
  const formal=aggregate(baseline,'全量得分可用')?.value===true;
  const score=aggregate(baseline,'全量PDMS');
  const scoreText=formal&&score?.value!=null?format(score)+'/100':'完整成绩待覆盖核对';
  $('currentProgress').innerHTML='<div class="current-progress panel"><div><span class="eyebrow">CURRENT CHECKPOINT</span><h2>AUGUstus · 官方 navtest 基线</h2><p>只显示聚合进度；不发布测试场景明细。</p></div><div class="coverage"><strong>'+(valid?done.toLocaleString('zh-CN')+' / '+expected.toLocaleString('zh-CN'):'尚无完整进度记录')+'</strong><span>'+esc(scoreText)+'</span><div class="metric-bar"><i style="width:'+(fraction??0)+'%"></i></div><small>'+(valid?fraction.toFixed(1)+'% 覆盖 · ':'')+'回执状态：'+esc(baseline?labels[baseline.status]||baseline.status:'未知')+'</small></div><a href="#view=experiments&run=augustus-navval-auto">查看评测 →</a></div>';
  $('overviewBoundary').textContent=data.system?.boundary||'独立查看执行完成、质量准入与实际收益。';
  const records=[
    ['vectors','navtrain计划','训练来源规模','计划覆盖 · 有效向量见数据生产'],
    ['caption64','场景数','Caption 试点','两类描述 · 事实待审核'],
    ['h3-prompts','验证场景','H3 条件接口','仅提示词 · 未运行视频模型']
  ];
  const cards=records.map(([id,label,title,note])=>{
    const row=data.production.find(r=>r.id===id),field=aggregate(row,label);
    return '<article class="stat"><span>'+esc(title)+'</span><strong>'+(field?format(field):'未记录')+'</strong><small>'+esc(note)+'</small></article>';
  });
  cards.push('<article class="stat"><span>训练、回放与实验</span><strong>'+data.experiments.length+' <small>组已登记</small></strong><small>原生训练 / 恢复 / 反馈分别留证</small></article>');
  $('stats').innerHTML=cards.join('');
}

function stageFields(row) {
  return [['输入',row.inputs],['执行与机制',row.operation],['产物',row.outputs],['验收与边界',row.acceptance]].map(([name,value])=>'<div><h4>'+name+'</h4><p>'+esc(value)+'</p></div>').join('');
}
function renderSystemFlow(system) {
  const rows=system.steps||[];
  $('systemFlow').hidden=!rows.length;
  if(!rows.length)return;
  const id=new URLSearchParams(location.hash.slice(1)).get('step');
  const row=rows.find(r=>r.id===id)||rows[0];
  $('stageNavigation').innerHTML=rows.map((r,i)=>'<a class="stage-tab '+(r.id===row.id?'selected':'')+'" '+(r.id===row.id?'aria-current="step"':'')+' href="#view=architecture&step='+encodeURIComponent(r.id)+'"><span>'+String(i+1).padStart(2,'0')+'</span>'+esc(r.short)+'</a>').join('');
  $('stageDetail').innerHTML='<div class="task-top"><div><span class="eyebrow">STAGE '+String(rows.indexOf(row)+1).padStart(2,'0')+'</span><h2>'+esc(row.title)+'</h2><p>'+esc(row.subtitle)+'</p></div><span class="badge warn">'+esc(maturity[row.status])+'</span></div><div class="stage-fields">'+stageFields(row)+'</div><div class="stage-current"><b>当前实现</b><p>'+esc(row.current)+'</p>'+architectureEvidence(row.evidence,'查看关联执行证据')+'</div>';
  $('allStageDetails').innerHTML=rows.map((r,i)=>'<article class="panel full-stage"><h3>'+String(i+1).padStart(2,'0')+' · '+esc(r.title)+'</h3><div class="stage-fields">'+stageFields(r)+'</div><p class="note">当前实现：'+esc(r.current)+'</p>'+architectureEvidence(r.evidence,'查看关联证据')+'</article>').join('');
  if(id && row.id===id && lastSystemStep!==id)requestAnimationFrame(()=>$('stageNavigation').scrollIntoView({block:'start',behavior:'smooth'}));
  lastSystemStep=id;
}
