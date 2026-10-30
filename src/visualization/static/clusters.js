'use strict';
let clusterPayload=null, clusterPage=0, clusterSelected=0, clusterScope=0;
const clusterPageSize=36;
async function renderClusters(){
  if(!data.clustering){$('clusterContent').innerHTML='<p class="notice">尚未登记完成的聚类结果。</p>';return;}
  if(!clusterPayload){
    $('clusterLoading').textContent='读取全库分层快照…';
    try{
      const response=await fetch('clusters.json',{cache:'no-cache'});
      if(!response.ok)throw Error('聚类快照读取失败');
      const value=await response.json();
      if(value.schema_version!==1||value.publication_scope!=='training_examples_and_aggregate_test_counts_only')throw Error('聚类数据范围不匹配');
      clusterPayload=value;
    }catch(e){$('clusterLoading').textContent=e.message;return;}
  }
  $('clusterLoading').textContent='训练簇 / 冻结映射 / 仅公开测试汇总';
  const s=clusterPayload.summary;
  $('clusterStats').innerHTML=[['训练成员',s.members.navtrain,'完整覆盖'],['测试成员',s.members.navtest,'仅冻结映射，无逐场景明细'],['最细叶簇',s.leaf_clusters,'可向上合并查看'],['每叶簇中位数',s.leaf_size.median,'最小 '+s.leaf_size.min+' / 最大 '+s.leaf_size.max]].map(([label,v,note])=>'<article class="stat"><span>'+label+'</span><strong>'+Number(v).toLocaleString()+'</strong><small>'+note+'</small></article>').join('');
  $('clusterBoundary').textContent=s.scope+' 细簇是统计分组；导航指令不是已执行转向。测试侧超出训练簇半径的 '+s.out_of_train_radius.navtest+' 个场景仍保留为不确定匹配。';
  const f=data.current?null:clusterPayload.failure_mapping;
  if(data.current){$('clusterFailureSummary').hidden=true;$('retrievalComparison').hidden=true;$('clusterFocus').value='all';if($('clusterSort').value==='failure')$('clusterSort').value='large';}
  if(f){
    $('clusterFailureSummary').hidden=false;
    $('clusterFailureSummary').innerHTML='<div class="eyebrow">CHAINFLOW STAGE1 9405 / 既有 NAVTEST 评分映射</div><h2>'+f.queries+' 个重点查询 → '+f.primary_leaf_clusters+' 个主叶簇</h2><p>零分 '+f.zero_queries+' · 非零且低于0.6 '+f.nonzero_below_06+' · 实测父模型 '+f.baseline_pdms.toFixed(4)+' PDMS</p><p>主簇训练池 '+f.primary_training_pool.toLocaleString()+' 个；限制来源后候选 '+f.ranked_training_candidates.toLocaleString()+' 个；含相邻扩展共 '+f.expanded_leaf_clusters+' 簇。</p><p>加载与核验 '+f.load_seconds.toFixed(3)+' s · 全部查询路由、扩展及簇内排序 '+f.query_seconds.toFixed(3)+' s（热运行中位数） · 本次完整运行 '+f.total_seconds.toFixed(2)+' s。</p><p class="note">'+f.uncertain_primary_queries+' 个查询超过训练簇经验半径，仍标为不确定。命中仅说明场景条件相似，不是同失败原因、Recall或训练准入。未对navtrain学生评分。这里只显示测试汇总。</p>';
    if(f.overlap){
      const names={pure_visual_top16:'纯视觉相似性 Top16',historical_hybrid_top16:'旧视觉8＋结构文本8融合'};
      $('clusterFailureSummary').innerHTML+='<h3>与已有检索候选的重叠（同父模型、同527查询）</h3><table><thead><tr><th>已有方法</th><th>候选数</th><th>共同token</th><th>占旧集合</th><th>并集交比</th><th>同一查询有交集</th></tr></thead><tbody>'+f.overlap.map(r=>'<tr><td>'+esc(names[r.name]||r.name)+'</td><td>'+r.comparison_count+'</td><td>'+r.shared+'</td><td>'+(r.fraction_of_comparison*100).toFixed(2)+'%</td><td>'+(r.jaccard*100).toFixed(2)+'%</td><td>'+r.same_query_with_overlap+' / '+f.queries+'</td></tr>').join('')+'</tbody></table><p class="note">旧混合队列核实为5,086条，不是5,806条。聚类每查询最多20条，对照最多16条；集合重叠不是语义精度或Recall，低重叠也不自动证明互补有效。</p>';
    }
    if(!renderClusters.focusInitialized){$('clusterFocus').value='primary_total';$('clusterSort').value='failure';renderClusters.focusInitialized=true;}
  }
  updateClusterList();
  if(!data.current&&typeof renderRetrievalComparison==='function')renderRetrievalComparison();
}
function clusterAncestor(nodeId,depth){let n=clusterPayload.nodes[nodeId];while(n.depth>depth)n=clusterPayload.nodes[n.parent];return n.id;}
function inClusterScope(n){if(!clusterScope)return true;let x=n;while(x.depth>clusterPayload.nodes[clusterScope].depth)x=clusterPayload.nodes[x.parent];return x.id===clusterScope;}
function clusterRows(){
 const focus=$('clusterFocus').value;
 const level=$('clusterLevel').value, text=$('clusterSearch').value.trim().toLowerCase();
 const nodes=clusterPayload.nodes, depth=clusterPayload.cuts[level];
 let ids=new Set(nodes.filter(n=>!n.children.length).map(n=>level==='leaf'?n.id:clusterAncestor(n.id,depth)));
 let rows=[...ids].map(id=>nodes[id]).filter(inClusterScope).filter(n=>focus==='all'||(n.failure_focus?.[focus]||0)>0).filter(n=>(!$('clusterTestOnly').checked||n.test_count>0)&&(!text||(n.label+' '+n.tags.concat(n.map_tags||[]).map(t=>t.text).join(' ')+' '+n.id).toLowerCase().includes(text)));
 const order=$('clusterSort').value;
 rows.sort((a,b)=>order==='failure'?((b.failure_focus?.[focus==='all'?'primary_total':focus]||0)-(a.failure_focus?.[focus==='all'?'primary_total':focus]||0)||a.id-b.id):order==='small'?a.train_count-b.train_count:order==='test'?b.test_count-a.test_count:b.train_count-a.train_count);
 return rows;
}
function updateClusterList(){
 if(!clusterPayload)return;
 const rows=clusterRows(),pages=Math.max(1,Math.ceil(rows.length/clusterPageSize));clusterPage=Math.min(clusterPage,pages-1);
 $('clusterCount').textContent=rows.length+' 个组 · 第 '+(clusterPage+1)+' / '+pages+' 页';
 $('clusterPrev').disabled=clusterPage===0;$('clusterNext').disabled=clusterPage===pages-1;
 const path=[];let n=clusterPayload.nodes[clusterScope];while(n){path.unshift(n);n=n.parent>=0?clusterPayload.nodes[n.parent]:null;}
 $('clusterBreadcrumb').innerHTML=path.map(n=>'<button data-scope="'+n.id+'">'+(n.id===0?'全库':'C'+String(n.id).padStart(4,'0'))+'</button>').join('<span> / </span>');
 $('clusterCards').innerHTML=rows.slice(clusterPage*clusterPageSize,(clusterPage+1)*clusterPageSize).map(n=>'<button class="cluster-card '+(n.id===clusterSelected?'selected':'')+'" data-cluster="'+n.id+'"><div><code>C'+String(n.id).padStart(4,'0')+'</code><small>第 '+n.depth+' 层</small></div><h3>'+esc(n.label)+'</h3><p><b>'+n.train_count+'</b> 训练 · <b>'+n.test_count+'</b> 测试</p>'+focusBadge(n)+'<small>'+n.source_groups+' 个来源组 · '+(n.speed_mean_kmh===null?'速度未知':n.speed_mean_kmh.toFixed(1)+' km/h 均速')+'</small></button>').join('')||'<p>当前筛选下没有场景组。</p>';
 drawClusterMap(rows);
 showCluster(clusterSelected);
}
function focusBadge(n){if(data.current)return '';const f=n.failure_focus;return f?'<p class="note">9405主簇：'+f.primary_zero+' 零分 / '+f.primary_low+' 低分 · 扩展命中 '+f.expanded_total+'</p>':'';}
function drawClusterMap(rows){
 const all=clusterPayload.nodes.filter(n=>!n.children.length);
 const xs=all.map(n=>n.xy[0]),ys=all.map(n=>n.xy[1]);
 const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const x=v=>30+740*(v-xmin)/(xmax-xmin||1),y=v=>320-290*(v-ymin)/(ymax-ymin||1);
 $('clusterMap').innerHTML='<svg viewBox="0 0 800 350" role="img" aria-label="训练簇中心二维投影；每个点代表一个分组">'+rows.map(n=>'<circle tabindex="0" role="button" data-cluster="'+n.id+'" aria-label="场景组 '+n.id+'，训练 '+n.train_count+'，测试 '+n.test_count+'" cx="'+x(n.xy[0])+'" cy="'+y(n.xy[1])+'" r="'+Math.min(12,2+Math.log1p(n.train_count)/2)+'" fill="'+(n.failure_focus?.primary_total?'#c56d27':n.test_count?'#12857d':'#a1b5c2')+'" opacity="0.7" stroke="'+(n.id===clusterSelected?'#e5a640':'#fff')+'" stroke-width="'+(n.id===clusterSelected?3:.5)+'"><title>C'+n.id+' '+esc(n.label)+' · '+n.train_count+' / '+n.test_count+'</title></circle>').join('')+'</svg>';
}
function showCluster(id){
 const n=clusterPayload.nodes[id];if(!n)return;clusterSelected=id;
 const stop={target_size:'达到细分目标',small_child_preserved:'继续拆分会产生过小子簇',identical_representation:'表示相同，未强行拆分',depth_limit:'达到深度上限'};
 $('clusterDetail').innerHTML='<div class="eyebrow">CLUSTER C'+String(n.id).padStart(4,'0')+'</div><h2>'+esc(n.label)+'</h2><p>训练 '+n.train_count+' · 测试 '+n.test_count+' · 来源组 '+n.source_groups+'</p>'+focusBadge(n)+'<div class="cluster-tags">'+n.tags.map(t=>'<button data-tag="'+esc(t.text)+'">'+esc(t.text)+' <small>'+t.count+'</small></button>').join('')+'</div><h3>地图确认的条件</h3><p>'+((n.map_tags||[]).map(t=>esc(t.text)+' ('+t.count+')').join(' · ')||'无已确认的地图类别')+'</p><p class="note">训练内平均距离 '+n.mean_distance.toFixed(3)+'；99%半径 '+n.radius_p99.toFixed(3)+'。它们是表示空间距离，不是语义准确率。</p><div class="cluster-children">'+(n.children.length?n.children.map(id=>{const c=clusterPayload.nodes[id];return '<button data-drill="'+id+'">展开 C'+id+' · '+c.train_count+' / '+c.test_count+' →</button>'}).join(''):'<span class="badge">叶簇 · '+esc(stop[n.stop_reason]||n.stop_reason)+'</span>')+'</div><h3>训练侧代表场景</h3>'+n.representatives.map(r=>'<article class="cluster-example"><code>'+esc(r.token)+'</code><p>'+esc(r.description)+'</p><small>速度 '+(r.speed_kmh===null?'未知':Number(r.speed_kmh).toFixed(1)+' km/h')+' · 导航指令 '+esc(r.navigation_command)+'</small><div class="badge warn">机器描述草稿，尚未逐条事实复核</div></article>').join('');
 document.querySelectorAll('.cluster-card').forEach(c=>c.classList.toggle('selected',Number(c.dataset.cluster)===id));
}
document.addEventListener('click',event=>{
 const t=event.target.closest('[data-cluster],[data-drill],[data-scope],[data-tag]');if(!t||!clusterPayload)return;
 if(t.dataset.cluster!==undefined){showCluster(Number(t.dataset.cluster));return;}
 if(t.dataset.tag!==undefined){$('clusterSearch').value=t.dataset.tag;clusterPage=0;updateClusterList();return;}
 const id=Number(t.dataset.drill??t.dataset.scope);clusterScope=id;clusterSelected=id;clusterPage=0;
 if(t.dataset.drill!==undefined){$('clusterLevel').value='leaf';$('clusterSearch').value='';}
 updateClusterList();
});
document.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.matches('circle[data-cluster]')){e.preventDefault();showCluster(Number(e.target.dataset.cluster));}});
for(const id of ['clusterLevel','clusterSort','clusterTestOnly','clusterFocus'])document.getElementById(id).addEventListener('change',()=>{clusterPage=0;updateClusterList()});
$('clusterSearch').addEventListener('input',()=>{clusterPage=0;updateClusterList()});
$('clusterPrev').addEventListener('click',()=>{clusterPage--;updateClusterList()});
$('clusterNext').addEventListener('click',()=>{clusterPage++;updateClusterList()});
