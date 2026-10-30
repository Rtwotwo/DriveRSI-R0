'use strict';
let retrievalReview=null, retrievalRecall=null;
const retrievalRoot='/private-retrieval-geography/';
const retrievalNames={visual:'视觉相似性',hybrid:'旧融合',cluster:'文本＋状态＋拓扑聚类'};
async function renderRetrievalComparison(){
 if(!['127.0.0.1','localhost','[::1]'].includes(location.hostname))return;
 const panel=document.getElementById('retrievalComparison');if(!panel)return;
 if(!retrievalReview){
  try{const response=await fetch(retrievalRoot+'browser.json',{cache:'no-cache'});if(!response.ok)return;
   const value=await response.json();if(value.scope!=='private_loopback_retrieval_review')return;retrievalReview=value;
  }catch(e){return;}
 }
 panel.hidden=false;
 const s=retrievalReview.summary;
 panel.innerHTML='<div class="eyebrow">PRIVATE / 原始日志位置与同相机对照</div><h2>检索找到的是同一地点，还是相似场景？</h2><p>地图坐标以米为单位；跨地图距离不比较。下表均使用原排名前16条，分母为527个查询。地理接近不等于同道路、同交互或检索语义准确。</p><table><thead><tr><th>方法</th><th>Top1同地图</th><th>Top16任一≤50m</th><th>≤100m</th><th>≤500m</th><th>≤100m且朝向差≤30°</th></tr></thead><tbody>'+Object.entries(s.methods).map(([k,v])=>'<tr><td>'+retrievalNames[k]+'</td><td>'+v.top1.same_map_pairs+' / 527</td>'+['50','100','500'].map(r=>'<td>'+v.top16.within_radius[r].queries+' / 527</td>').join('')+'<td>'+v.top16.within_radius['100'].queries_heading_30deg+' / 527</td></tr>').join('')+'</tbody></table><p class="note">训练全库本来存在100m内场景的查询：'+s.geographic_availability['100']+' / 527。距离是锚点平面直线距离，未验证沿路距离与上下层道路。均截取原排名前16；旧融合有16个查询仅15条，来源多样性约束沿用各自原配置。旧融合 '+retrievalReview.ranking_audit.hybrid_top_score_ties+' 个查询的最高分并列，Top1受并列排序规则影响。</p><div class="filter-row"><label>对照案例 <select id="retrievalExample">'+retrievalReview.examples.map((e,i)=>'<option value="'+i+'">'+e.id+' · '+esc(e.query.map)+' · '+(e.score_group==='zero'?'零分':'低分')+'</option>').join('')+'</select></label><label>相机 <select id="retrievalCamera"><option>CAM_F0</option><option>CAM_L0</option><option>CAM_R0</option><option>CAM_B0</option><option value="all">全部四相机</option></select></label><label>展示候选 <select id="retrievalRanking"><option value="top1">原始排名第1</option><option value="nearest16">前16中地理最近（标出原排名）</option></select></label></div><p id="retrievalSelection" class="notice"></p><div id="retrievalImages" class="retrieval-grid"></div><p class="note">'+esc(retrievalReview.caption_status)+' 不同场景的单帧无法单独证明切入、让行等时序动作。图像只在本机私有入口提供，没有写入公开发布包。</p>';
 for(const id of ['retrievalExample','retrievalCamera','retrievalRanking'])document.getElementById(id).addEventListener('change',showRetrievalExample);
 showRetrievalExample();
 renderGeographicRecall();
 renderCompactRetrieval();
}
async function renderCompactRetrieval(){
 if(!['127.0.0.1','localhost','[::1]'].includes(location.hostname))return;
 let data;try{const r=await fetch(retrievalRoot+'compact-retrieval.json',{cache:'no-cache'});if(!r.ok)return;data=await r.json();}catch(e){return;}
 const s=data.summary;if(s?.status!=='complete')return;
 let box=document.getElementById('compactRetrieval');if(!box){box=document.createElement('section');box.id='compactRetrieval';document.getElementById('retrievalComparison').prepend(box);}
 const labels={visual_top1:'全库视觉Top1（既有排序）',cluster_visual_top1:'主簇内视觉Top1（本轮实算）',visual_geo25_top1:'视觉Top16中取一条：25m／30°筛选',cluster_geo25_top1:'主簇内取一条：25m／30°筛选',cluster_geo50_top1:'主簇内取一条：50m／30°筛选',cluster_geo25_asset_top1:'主簇＋25m／30°＋登记资产'};
 box.innerHTML='<div class="eyebrow">COMPACT / 只处理239个零分查询</div><h2>先精确缩小种子，再投入合成预算</h2><p>冻结文本＋状态＋拓扑主簇，使用同一编码器的12张历史图像逐视角余弦均值排序。未重拟合聚类、未给navtrain学生评分。位置筛选是选题条件，不是独立语义准确率。</p><table><thead><tr><th>方案</th><th>有候选查询</th><th>去重训练种子</th><th>来源日志</th><th>有登记资产种子</th></tr></thead><tbody>'+Object.entries(labels).map(([key,label])=>{const v=s.methods[key];return '<tr><td>'+label+'</td><td>'+v.queries_returned+'/239</td><td>'+v.unique_training_tokens+'</td><td>'+v.source_logs+'</td><td>'+v.registered_asset_tokens+'</td></tr>';}).join('')+'</tbody></table><p>首批预算最多24个种子、每来源日志1个：实际 '+s.compact.seeds+' 个种子覆盖 '+s.compact.covered_queries+' 个查询的检索条件。资产优先的独立小队列有 '+s.asset_compact.seeds+' 个种子，可关联 '+s.asset_compact.covered_queries+' 个查询。其余查询明确未覆盖。</p><p class="note">资产登记不等于视觉准入。只使用连续历史且未被簇半径标为不确定的查询关系；不以放宽位置或画质条件来补足数量。完整加载＋核验 '+s.timings.load_and_verify_seconds.toFixed(2)+'秒，簇内排序 '+s.timings.exact_aligned_ranking_seconds.toFixed(3)+'秒。前者包括原始逐图特征读取，不能与旧簇编号查表耗时混称。</p><p>历史成本：7秒教师推演中位 '+data.cost.teacher.median_seconds.toFixed(0)+'秒；四相机60图渲染 '+data.cost.render.median_seconds.toFixed(0)+'秒；源位姿核验 '+data.cost.source_reconstruction.median_seconds.toFixed(0)+'秒。历史 '+data.cost.attempts+' 次尝试仅产出 '+data.cost.qualified+' 条24dB试验组数据，不能按每个种子保证20条外推。</p><p class="notice">本轮新推演：'+esc(data.pilot?.message||'结果待核验')+'</p><hr>';
 const heading=box.querySelector('h2');if(heading)heading.textContent='历史定位筛选实验（已退出生产）';
 await renderLocalizationFreePlan();
}
async function renderLocalizationFreePlan(){
 if(!['127.0.0.1','localhost','[::1]'].includes(location.hostname))return;
 let d;try{const r=await fetch(retrievalRoot+'localization-free-retrieval.json',{cache:'no-cache'});if(!r.ok)return;d=await r.json();}catch(e){return;}
 const s=d.summary,b=d.budget;if(s?.kind!=='localization_free_visual_queue'||s.localization_used_for_selection!==false)return;
 let box=document.getElementById('localizationFreeRetrieval');if(!box){box=document.createElement('section');box.id='localizationFreeRetrieval';}document.getElementById('retrievalComparison').prepend(box);
 box.innerHTML='<div class="eyebrow">ACTIVE / 无定位信息的生产检索</div><h2>239个零分查询 → 562个训练场景 → 目标5,620条新样本</h2><p>当前生产使用纯视觉Top3。坐标、地理距离、绝对航向不参与排序或筛选；聚类只作辅助分组与对照。零分信息仅用于选取测试查询，训练数据仍只来自navtrain。</p><table><thead><tr><th>方法</th><th>覆盖查询</th><th>去重训练场景</th><th>独立来源日志</th></tr></thead><tbody>'+[['visual_top2','纯视觉Top2'],['visual_top3','纯视觉Top3 · 当前采用'],['cluster_visual_top2','簇内视觉Top2 · 对照']].map(([k,label])=>{const v=s.comparison[k];return '<tr><td>'+label+'</td><td>'+v.query_count+'/239</td><td>'+v.unique_seeds+'</td><td>'+v.source_logs+'</td></tr>';}).join('')+'</tbody></table><p>同一查询最多1条来自同一来源组；全局去重，不用定位或GT未来标签复筛。当前候选的语义准确率仍未核验。读取既有排名并编译队列耗时 '+s.elapsed_seconds.toFixed(2)+'秒，不含原始特征提取。</p><h3>Harness全局预算与执行状态</h3><p>每场景目标 '+b.target_per_anchor+' 条合格且不同的新轨迹，最多 '+b.attempts_per_anchor+' 次尝试；每批最多 '+b.wave_size+' 个场景。连续3次轨迹重复、1次视觉失败或2次执行失败后停止对应场景。未达到目标如实记录缺额，未知在途结果不自动重发。</p><p class="notice">本次真实入口检查：'+esc(d.status.reason)+'。待补资产 '+(b.state_counts.needs_asset_build||0)+'，待修复 '+(b.state_counts.needs_asset_repair||0)+'。本轮没有启动新的推演、渲染或训练。</p><table><thead><tr><th>顺利通过且资产完备时</th><th>条件性耗时</th></tr></thead><tbody><tr><td>单场景10条</td><td>40～60分钟</td></tr><tr><td>562场景×10条，串行</td><td>374.7～562小时，约15.6～23.4天</td></tr><tr><td>理想4条独立流水线</td><td>93.7～140.5小时，约3.9～5.9天</td></tr><tr><td>理想8条独立流水线</td><td>46.8～70.3小时，约2.0～2.9天</td></tr></tbody></table><p class="note">并行时间是理想外推，当前控制器按单个有界波次执行；不是已经验证的多机吞吐。未计失败、重复、资产建设、排队、导出和训练。上一批259次尝试仅4条24dB试验样本，不能据此承诺本轮5,620条的交付日期。使用navtest失败反馈仍属于自适应开发。</p><hr>';
}
async function renderGeographicRecall(){
 if(!retrievalRecall){try{const r=await fetch(retrievalRoot+'geographic-recall.json',{cache:'no-cache'});if(!r.ok)return;retrievalRecall=await r.json();}catch(e){return;}}
 if(retrievalRecall.status!=='complete')return;
 let box=document.getElementById('retrievalRecall');
 if(!box){box=document.createElement('section');box.id='retrievalRecall';document.getElementById('retrievalComparison').insertBefore(box,document.getElementById('retrievalExample').closest('.filter-row'));}
 const previous=document.getElementById('retrievalGeoRule')?.value||retrievalRecall.primary_protocol;
 const options=Object.entries(retrievalRecall.protocols).map(([key,p])=>'<option value="'+key+'"'+(key===previous?' selected':'')+'>同地图 ≤'+p.radius_m+'m'+(p.max_heading_difference_deg?'，朝向差 ≤'+p.max_heading_difference_deg+'°':'，不限朝向')+'</option>').join('');
 const protocol=retrievalRecall.protocols[previous],percent=v=>v===null?'不适用':(v*100).toFixed(2)+'%';
 box.innerHTML='<h3>完整训练库作分母：单路、交集与并集的地理 Recall</h3><label>位置判定标准 <select id="retrievalGeoRule">'+options+'</select></label><p>逐查询从103,288个训练锚点中穷尽满足条件的成员。相关集合大小：最少 '+protocol.relevant_candidates.min+'，中位 '+protocol.relevant_candidates.median+'，最多 '+protocol.relevant_candidates.max+'。只衡量地点/朝向，不证明相同交通交互。</p><table><thead><tr><th>集合</th><th>平均候选/查询</th><th>返回对中地理合格比例</th><th>地理Recall（逐查询均值）</th><th>至少命中一个（全部527查询）</th></tr></thead><tbody>'+[['visual','视觉Top16'],['cluster','聚类Top16'],['intersection','两路逐查询交集'],['union','两路逐查询并集，最多32']].map(([k,title])=>{const v=protocol.methods[k];return '<tr><td>'+title+'</td><td>'+v.mean_candidates_per_query.toFixed(2)+'</td><td>'+percent(v.micro_precision)+'</td><td>'+percent(v.macro_recall)+'</td><td>'+v.queries_with_any_match+'/527 · '+percent(v.hit_fraction_all_queries)+'</td></tr>';}).join('')+'</tbody></table><p class="note">交集仅 '+(527-protocol.methods.intersection.empty_retrieval_queries)+' 个查询有候选，不能用全局候选交集替代。并集保留更多候选，未按16条重排。无地理真值查询数 '+protocol.methods.visual.queries_without_ground_truth+'，从Recall均值排除并单独报告。返回对中合格比例按总命中/总返回计算；完整结果另含逐查询Precision。</p>';
 document.getElementById('retrievalGeoRule').addEventListener('change',renderGeographicRecall);
}
function showRetrievalExample(){
 const e=retrievalReview.examples[Number(document.getElementById('retrievalExample').value)],mode=document.getElementById('retrievalRanking').value,camera=document.getElementById('retrievalCamera').value;
 document.getElementById('retrievalSelection').textContent=e.id+' · '+e.selection+'。'+retrievalReview.selection_policy;
 const cards=[{title:'测试查询 '+e.id,scene:e.query,detail:'原始PDMS '+(e.score*100).toFixed(2)+'；仅作私有对照'}];
 for(const name of ['visual','hybrid','cluster']){const p=e.methods[name][mode];cards.push({title:retrievalNames[name],scene:p?.scene,detail:p?'原排名 #'+p.rank+' · '+(p.same_map?p.distance_m.toFixed(1)+' m · 朝向差 '+p.heading_difference_deg.toFixed(1)+'°':'不同地图，距离不适用'):'前16没有同地图候选'});}
 document.getElementById('retrievalImages').innerHTML=cards.map(c=>'<article class="retrieval-card"><h3>'+esc(c.title)+'</h3><p>'+esc(c.detail)+'</p>'+(!c.scene?'<p>无可比较图像</p>':(camera==='all'?['CAM_F0','CAM_L0','CAM_R0','CAM_B0']:[camera]).map(cam=>'<a href="'+retrievalRoot+c.scene.images[cam]+'" target="_blank" rel="noopener"><img loading="lazy" alt="'+esc(c.title+' '+cam)+'" src="'+retrievalRoot+c.scene.images[cam]+'"></a><small>'+cam+'</small>').join('')+'<p><b>'+esc(c.scene.map)+'</b> · C'+c.scene.leaf+'</p><p>速度 '+Number(c.scene.speed_kmh).toFixed(1)+' km/h · 导航 '+esc(c.scene.navigation_command)+'</p><p>'+esc(c.scene.map_labels.join(' · '))+'</p><p>'+esc(c.scene.description)+'</p>'+(c.scene.token?'<code>'+esc(c.scene.token)+'</code>':''))+'</article>').join('');
}
