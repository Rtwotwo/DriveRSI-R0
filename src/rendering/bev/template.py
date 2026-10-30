"""Bundled, dependency-free HTML/CSS/JS for the offline replay inspector."""

HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>DriveRSI · 交互回放</title>
<style>
:root{--ink:#203244;--muted:#6c7c89;--border:#dde5ea;--paper:#fff;--teal:#007c80;--blue:#3478e5;--red:#d85550;--amber:#c98b34;--violet:#9b77d5}
*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
button,select,input{font:inherit}button{cursor:pointer;white-space:nowrap}button,select{border:1px solid var(--border);border-radius:7px;background:#fff;color:var(--ink);padding:7px 11px}button:hover{background:#f1f6f8;border-color:#b4c6d1}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #52babc;outline-offset:2px}button.primary{background:var(--teal);border-color:var(--teal);color:white;min-width:68px}button:disabled{opacity:.45;cursor:default}input[type=checkbox]{accent-color:var(--teal)}
header{display:flex;justify-content:space-between;gap:18px;align-items:center;padding:20px 28px;background:var(--paper);border-bottom:1px solid var(--border)}.brand{font-weight:750;letter-spacing:.035em;font-size:22px}.brand small{color:var(--muted);font-size:15px;font-weight:450;letter-spacing:0;margin-left:14px}.subtitle{margin-top:4px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}.badge{display:inline-block;border:1px solid #bce0da;border-radius:30px;background:#edf9f5;padding:5px 12px;color:#257266;font-size:12px;white-space:nowrap}.layout{max-width:1800px;margin:auto;padding:20px 24px;display:grid;grid-template-columns:minmax(420px,1fr) 328px;gap:18px}.stage,.card{background:var(--paper);border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:0 2px 5px #1b344404}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:12px 15px;border-bottom:1px solid var(--border);flex-wrap:wrap}.toolbar strong{font-size:13px}.tools{display:flex;align-items:center;gap:8px;font-size:12px}.viewport{height:min(62vh,680px);min-height:390px;position:relative;background:#101d2a;overflow:hidden}#map{display:block;width:100%;height:100%;touch-action:none;cursor:grab}#map:active{cursor:grabbing}.map-caption{pointer-events:none;position:absolute;top:16px;left:18px;color:#d5e2ed;font-size:12px;background:#101d2ad9;border:1px solid #ffffff16;border-radius:7px;padding:8px 11px;max-width:calc(100% - 36px)}.map-caption strong{font-variant-numeric:tabular-nums;color:#fff;font-weight:600}.map-legend{pointer-events:none;position:absolute;bottom:14px;left:18px;right:18px;display:flex;align-items:center;justify-content:space-between;gap:12px;color:#c5d3df;font-size:11px}.legend-items{display:flex;gap:12px;flex-wrap:wrap}.key{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;background:#45bdba}.key.replay{background:#d9a65d}.key.traffic{background:#a6b5e3}.key.compare{background:#b497de}.key.selected{background:#56a5ff}.layers{display:flex;flex-wrap:wrap;gap:14px;padding:10px 16px;border-bottom:1px solid var(--border);font-size:12px;color:#526574}.layers label{cursor:pointer;display:flex;align-items:center;gap:5px}.playback{padding:14px 16px}.time-row{display:flex;align-items:center;gap:12px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}.time-row input{width:100%;accent-color:var(--teal);cursor:pointer}.time-stamp{min-width:58px;color:var(--ink);font-weight:650}.event-strip{height:10px;position:relative;margin:0 68px 9px}.event-dot{position:absolute;top:1px;width:5px;height:7px;padding:0;border:0;border-radius:2px;background:var(--red)}.event-dot.goal{background:#299886}.event-dot.truncated{background:#7d8da0}.transport{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.transport .hint{margin-left:auto;color:var(--muted);font-size:11px}.chart-area{margin:0 16px 14px;border-top:1px solid var(--border);padding-top:10px}.chart-head{display:flex;justify-content:space-between;align-items:center;gap:10px;color:var(--muted);font-size:11px}.chart-labels{display:flex;gap:14px;flex-wrap:wrap}.chart-labels span:before{content:"";display:inline-block;width:15px;height:2px;margin:0 5px 3px 0;background:var(--blue)}.chart-labels span:nth-child(2):before{background:var(--teal)}.chart-labels span:nth-child(3):before{background:var(--amber)}#chart{display:block;width:100%;height:138px;cursor:crosshair}.chart-note{font-size:11px;color:var(--muted)}aside{display:flex;flex-direction:column;gap:14px;min-width:0}.card{padding:16px}.card h2{font-size:14px;margin:0 0 12px;font-weight:680}.actor-select{width:100%;font-size:12px}.actor-id{font-family:ui-monospace,Consolas,monospace;color:var(--muted);font-size:11px;overflow-wrap:anywhere;margin:8px 0 11px}.source-row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:13px;font-size:12px}.source-pill{background:#e9f5f4;color:#257b79;border-radius:5px;padding:3px 7px;font-size:11px;max-width:230px;overflow-wrap:anywhere}.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px 10px}.metric .label{font-size:11px;color:var(--muted)}.metric .value{font-size:23px;font-weight:650;letter-spacing:-.03em;line-height:1.5;font-variant-numeric:tabular-nums}.metric .unit{font-size:11px;color:var(--muted);font-weight:400;margin-left:4px;letter-spacing:0}.detail-list{margin:13px 0 0;border-top:1px solid var(--border);padding-top:10px}.detail-list>div{display:flex;justify-content:space-between;gap:12px;font-size:12px;margin:6px 0}.detail-list .label{color:var(--muted);flex-shrink:0}.detail-list .value{text-align:right;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.action-box{padding:10px 11px;background:#f4f7f9;border:1px solid #e6ecef;border-radius:7px;margin-top:12px;font-size:12px}.action-box .label{color:var(--muted);font-size:11px;margin-bottom:4px}.status{font-size:11px;padding:3px 7px;border-radius:5px;background:#f0f4f6;color:#657886}.status.warning{color:#b3463f;background:#fff0ee}.status.success{color:#257c68;background:#e9f7f0}.events{max-height:210px;overflow:auto;margin:0 -5px;padding:0 5px}.event-row{display:flex;width:100%;align-items:start;text-align:left;gap:10px;padding:8px 0;border:0;border-bottom:1px solid #edf1f4;border-radius:0;background:transparent;font-size:12px;white-space:normal}.event-row:hover{background:#f6f9fa}.event-row .time{font-size:11px;color:var(--muted);min-width:40px;font-variant-numeric:tabular-nums}.event-row .body{min-width:0}.event-row .who{color:var(--muted);font-size:10px;overflow-wrap:anywhere}.empty{font-size:12px;color:var(--muted);padding:8px 0}.limits{font-size:11px;color:var(--muted);line-height:1.75}.limits p{margin:6px 0}.limits code{font-size:10px;overflow-wrap:anywhere}.compare-card{border-left:3px solid var(--violet);background:#fbfaff;margin-top:12px;padding:9px 10px;font-size:12px}.compare-card strong{color:#8061a9;font-size:11px}.compare-card div{margin-top:4px}.footer{padding:0 28px 20px;font-size:11px;color:#80909e}.hidden{display:none!important}
@media(max-width:1050px){.layout{grid-template-columns:minmax(360px,1fr) 292px;padding:16px;gap:12px}.card{padding:13px}.transport .hint{width:100%;margin-left:0}.brand{font-size:20px}}
@media(max-width:760px){header{padding:16px}.brand small{display:block;margin:1px 0 0;font-size:13px}.layout{display:flex;flex-direction:column;padding:12px}.viewport{height:52vh;min-height:340px}aside{display:grid;grid-template-columns:1fr 1fr}.actor-card{grid-row:span 2}.card{min-width:0}.footer{padding:0 16px 16px}.badge{font-size:10px;padding:4px 8px}}
@media(max-width:480px){aside{display:flex}.toolbar{padding:10px}.tools{gap:5px}.layers{gap:9px}.map-legend{align-items:end}.legend-items{gap:7px}.chart-head{align-items:start}.chart-labels{gap:5px}.transport button{padding:6px 8px}}
.context-note{padding:9px 16px;background:#fff8e9;color:#735930;border-bottom:1px solid var(--border);font-size:11px}.key.context{background:transparent;border:1px dashed #e8adcf}.context-list{font-size:11px;line-height:1.7;overflow-wrap:anywhere;max-height:160px;overflow:auto}.context-list p{margin:5px 0}.layer-list{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:11px}.layer-list label{display:flex;align-items:center;gap:4px}.source-select-label{display:block;color:var(--muted);font-size:11px;margin:9px 0 4px}.source-details{font-size:12px;line-height:1.75;overflow-wrap:anywhere}.kind-key{color:#a3b4c1;font-size:10px}.context-active{border-color:#d9a8c3}.availability{color:#687e8c}.context-warning{color:#946039;background:#fff7e8;border-radius:6px;padding:8px;margin:8px 0}
</style>
</head>
<body>
<header><div><div class="brand">DriveRSI <small>多车交互 · 回放检查器</small></div><div id="subtitle" class="subtitle"></div></div><span id="modeBadge" class="badge">离线记录 / 状态可追溯</span></header>
<div class="context-warning" style="margin:14px 24px;padding:14px;border-left:5px solid #b76b16"><strong id="modeTitle"></strong><p id="modeExplanation"></p><a id="pairedView" class="hidden"></a></div>
<div id="inspectionNote" class="context-warning hidden" style="margin:14px 24px"><a id="galleryHome" href="../index.html">← nuPlan 场景目录</a> · <span id="inspectionText"></span></div>
<main class="layout">
<section class="stage" aria-label="回放画布和时间控制">
<div class="toolbar"><strong>共同世界 · 俯视图</strong><div class="tools"><button id="focusEgo">聚焦 Ego</button><a id="permalink" title="保留当前对象和帧">定位链接</a><button id="fitActors">适配对象</button><button id="fitMap">全地图</button><label><input id="follow" type="checkbox"> 跟随选中对象</label></div></div>
<div class="viewport"><canvas id="map" aria-label="交通对象俯视图，点击对象查看来源"></canvas><div class="map-caption"><strong id="frameCaption"></strong><br><span id="frameMeaning"></span><br><span id="populationCaption"></span></div><div class="map-legend"><div class="legend-items"><span><i class="key selected"></i>选中对象</span><span><i class="key"></i>策略动作车辆</span><span><i class="key traffic"></i>交通反馈（含停车 / 制动）</span><span><i class="key replay"></i>日志回放对象</span><span><i class="key context"></i>源日志上下文</span><span id="compareLegend" class="hidden"><i class="key compare"></i>对比分支（虚线）</span></div><span id="scaleText"></span></div></div>
<div class="layers"><label>地图视图 <select id="mapPreset" aria-label="地图显示预设"><option value="interaction" selected>交互视图</option><option value="all">全部图层</option><option value="custom" disabled>自定义图层</option></select></label><label title="用于核对仿真使用的道路折线；与语义地图同时打开可能重叠"><input type="checkbox" id="roads" checked> 仿真道路（诊断）</label><label><input type="checkbox" id="graph"> 车道图（诊断）</label><label><input type="checkbox" id="routes" checked> 选车路线</label><label><input type="checkbox" id="trails" checked> 运动尾迹</label><label><input type="checkbox" id="ids"> 车辆编号</label><label><input type="checkbox" id="reference"> 日志参考（非仿真）</label><label id="compareLayer" class="hidden"><input type="checkbox" id="comparison" checked> 对比分支</label></div>
<div class="layers"><label><input type="checkbox" id="worldObjects" checked> 世界对象</label><label><input type="checkbox" id="contextObjects" checked> 源日志对象</label><label><input type="checkbox" id="vehicles" checked> 车辆</label><label><input type="checkbox" id="pedestrians" checked> 行人</label><label><input type="checkbox" id="cyclists" checked> 骑行者</label><label><input type="checkbox" id="otherObjects" checked> 其他 / 未知</label><label><input type="checkbox" id="signals" checked> 信号记录</label><span class="kind-key">矩形：车辆 · 圆/椭圆：行人 · 菱形：骑行者 · 三角：锥桶 · ×：未知对象</span><span class="kind-key">+：地图参考节点 · 小方块：其他地图点 · 灯形：交通灯几何（状态未绑定） · 彩色方框：连接器源信号 · 空心圆环：选车日志参考位置</span></div>
<div class="context-note" id="contextNotice"></div>
<div class="playback"><div class="time-row"><span id="currentTime" class="time-stamp">0.00 s</span><input id="timeline" type="range" min="0" step="1" value="0" aria-label="回放时间轴"><span id="duration" class="time-stamp"></span></div><div id="eventStrip" class="event-strip" aria-label="事件时间标记"></div><div class="transport"><button id="start" title="回到初始状态">回到开始</button><button id="previous" title="上一帧（左方向键）">上一帧</button><button id="play" class="primary">播放</button><button id="next" title="下一帧（右方向键）">下一帧</button><select id="speed" aria-label="播放倍速"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><span class="hint">滚轮缩放 · 拖动平移 · 点击选车 · 空格播放</span></div></div>
<div class="chart-area"><div class="chart-head"><strong>选车时序</strong><div class="chart-labels"><span id="chartSpeedLabel">纵向速度 m/s</span><span id="chartAccelLabel">实际加速度 m/s²</span><span>反馈目标 / 干预加速度 m/s²</span></div></div><canvas id="chart" aria-label="速度与加速度曲线，点击跳转时间"></canvas><div id="chartNote" class="chart-note"></div></div>
</section>
<aside>
<section class="card actor-card" id="actorCard"><h2>对象与控制</h2><label class="source-select-label" for="actor">世界中的对象</label><select id="actor" class="actor-select" aria-label="选择世界对象"></select><label class="source-select-label" for="contextActor">源日志上下文（无独立控制）</label><select id="contextActor" class="actor-select" aria-label="选择源日志上下文对象"></select><div id="contextDetails" class="source-details hidden"></div><div id="worldDetails"><div id="actorId" class="actor-id"></div><div class="source-row"><span id="controlSource" class="source-pill"></span><span id="actorStatus" class="status"></span></div><div class="metric-grid"><div class="metric"><div id="speedLabel" class="label">纵向速度（负值为倒车）</div><span id="metricSpeed" class="value"></span><span class="unit">m/s</span></div><div class="metric"><div id="accelLabel" class="label">实际纵向加速度</div><span id="metricAccel" class="value"></span><span class="unit">m/s²</span></div><div class="metric"><div id="targetLabel" class="label">本帧目标加速度</div><span id="metricTarget" class="value"></span><span class="unit">m/s²</span></div><div class="metric"><div class="label">当前位置距路线</div><span id="metricDeviation" class="value"></span><span class="unit">m</span></div></div>
<div class="action-box"><div id="actionInterval" class="label"></div><div id="actionValue"></div></div>
<p id="diagnosticSource" class="context-warning"></p><div class="detail-list"><div><span class="label">实际 / 日志参考速率</span><span id="speedComparison" class="value"></span></div><div><span class="label">配置期望速度</span><span id="configuredSpeed" class="value"></span></div><div><span class="label">停车来源证据</span><span id="stationaryEvidence" class="value"></span></div><div><span class="label">朝向数据质量</span><span id="headingQuality" class="value"></span></div><div><span class="label">当前位置 x / y</span><span id="position" class="value"></span></div><div><span class="label">朝向 / 转角</span><span id="heading" class="value"></span></div><div><span class="label">本帧到达事件</span><span id="arrivals" class="value"></span></div><div><span id="recoveryLabel" class="label">目标速度 / 恢复模式</span><span id="recovery" class="value"></span></div><div><span id="headingErrorLabel" class="label">路线朝向误差</span><span id="headingError" class="value"></span></div><div><span id="leadLabel" class="label">感知前车</span><span id="lead" class="value"></span></div><div><span id="gapLabel" class="label">车身间距 / TTC</span><span id="gap" class="value"></span></div><div><span class="label">路线起始节点</span><span id="routeStart" class="value"></span></div><div><span class="label">匹配来源 / 候选</span><span id="routeMatch" class="value"></span></div><div><span id="routeSafetyLabel" class="label">路线 / 前向制动状态</span><span id="routeSafety" class="value"></span></div><div><span class="label">路线终点原因</span><span id="routeEnd" class="value"></span></div><div><span class="label">已知路线限速</span><span id="speedLimit" class="value"></span></div></div>
<div id="compareMetrics" class="compare-card hidden"><strong>对比分支 · 同一时刻</strong><div id="compareValues"></div><div id="compareEvents"></div></div></div></section>
<section class="card"><h2>避让关系与源数据问题</h2><p id="dependencyMeaning" class="availability"></p><div id="dependencyGraph"></div><p class="availability">选中对象此前报告的关系（点击回到首次记录；累计帧数不表示连续）。</p><div id="dependencyHistory"></div><p id="sourceIntegrityMeaning" class="context-warning"></p><div id="sourceIntegrityEvents"></div></section>
<section class="card"><h2>地图语义与数据覆盖</h2><p id="mapViewStatus" class="availability"></p><div id="semanticLayers" class="layer-list"></div><p class="availability">地图填色：灰色道路 · 黄绿步行区 · 绿色停车区 · 紫色横道。控制来源与对象类别沿用画布图例。</p><div id="mapAvailability" class="context-list availability"></div><div id="signalDetails" class="context-list"></div><p class="context-warning">仅源日志参考对象与地图信号图层用于对照，不自动参与物理交互。进入世界的对象按在场状态参与观测和几何碰撞；日志回放不具备反应式避让；标为反应式背景的对象按当前世界反馈控制，源标注中断不等于仿真离场。</p></section>
<section class="card"><h2>事件记录 <span id="eventCount" class="subtitle"></span></h2><div class="tools" style="flex-wrap:wrap;margin-bottom:10px"><select id="eventFilter" aria-label="事件类型"><option value="all">全部事件</option><option value="safety">碰撞 / 越界 / 接触</option><option value="lifecycle">接管 / 离场 / 标注变化</option><option value="contact">物体接触对</option></select><label><input id="selectedEvents" type="checkbox">仅选中对象</label></div><div id="events" class="events"></div></section>
<section class="card limits"><h2>记录与边界</h2><p id="recordDetails"></p><p id="recordTiming"></p><p id="playbackClock"></p><p id="recordController"></p><p id="recordMap"></p><p>日志参考轨迹与仿真运动分层展示。记录中的道路边线不自动等同于真实路缘；本视图不补全缺失的路权或信号状态。</p><p>这是结构化状态检查器，不生成相机观测，也不以轨迹差异代替交互因果验收。旧版场景按日志末端或显式 xy 目标判断到达；route-task-v1 仅将明确目的地记为完成，路线支持边界记为独立截断。两者均不是交通规则合格证。</p><p>动作 t 由状态 t 决定，作用至状态 t+1。碰撞、越界和终止显示在结果到达帧；最终帧没有待执行动作。</p><p id="buildId"></p></section>
</aside></main><footer class="footer">显示状态与来源嵌入本文件 · 无外部资源请求 · 坐标采用场景局部地图轴，车身以中心位置表示</footer>
<script id="replay-data" type="application/json">__REPLAY_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("replay-data").textContent);
const S = DATA.scene, RUN = DATA.run, OTHER = DATA.comparison;
const SOURCE = DATA.display_mode === "source_reference";
if(SOURCE){for(const p of document.querySelectorAll('p')){if(p.textContent.includes('进入世界的对象')||p.textContent.includes('动作 t 由状态 t'))p.textContent='源标注参考回放：不运行控制器或碰撞检测；标注出现与中断不能直接解释为物理入场与离场。';}}

let eventEntries = [];
const CONTEXT = DATA.context, worldIds = new Set(S.actor_ids);
const $ = id => document.getElementById(id), ids = S.actor_ids, last = S.times.length - 1;
const backgroundTasks = S.metadata.reactive_background?.assignments || {};
const isReactive = i => Object.prototype.hasOwnProperty.call(backgroundTasks,ids[i]);
const headingCorrections = S.metadata.heading_audit?.corrections || [];
const axisAmbiguous = new Set(headingCorrections.map(item=>item.actor_id));
const uncertainReplay = i => !S.controlled[i] && axisAmbiguous.has(ids[i]);
const controlIndex = new Map(ids.filter((_, i) => S.controlled[i]).map((id, i) => [id, i]));
const EVENT = {collision:"碰撞", offroad:"越界", road_boundary_contact:"车体周界接触道路边界（独立记录）", goal:"到达目标", terminated:"终止", truncated:"时域截断", route_support_end:"路线支持耗尽", initial_stop:"初始化时已停止", entered:"源标注出现", exited:"源标注中断", admitted:"物理接管", departed:"有据物理离场", pending:"重叠 / 间距不足，等待接管", contact:"几何接触开始（非责任判定）"};
const END = {branch:"分岔处停止", dead_end:"道路末端", unloaded_exit:"出口未加载", cycle:"检测到环路", geometry_gap:"几何断点", node_budget:"节点预算", explicit_endpoint:"显式目标", unspecified:"未声明"};
const MATCH = {explicit_initial_node:"显式任务指定",unique_best_score:"唯一最佳匹配",continuous_topology_seam:"连续拓扑接缝"};
const color = {selected:"#62acff", controlled:"#4bc1bd", replay:"#dfaf6c", traffic:"#a6b5e3", compare:"#b49add", danger:"#fa817a"};
let frame = 0, selected = ids.includes("ego") ? ids.indexOf("ego") : Math.max(0,S.controlled.findIndex(Boolean)), playing = false, lastClock = 0, accumulator = 0;
let contextSelection = null;
const mapLayerControls = new Map();
let camera = {x:0,y:0,scale:5}, mapSize = {w:1,h:1}, pointer = null;
const canvas = $("map"), ctx = canvas.getContext("2d"), chart = $("chart"), chartCtx = chart.getContext("2d");
const number = (value, digits=2) => Number.isFinite(value) ? value.toFixed(digits) : "—";
const text = (id, value) => {$(id).textContent = value;};
const own = (object,key) => object&&Object.prototype.hasOwnProperty.call(object,key)?object[key]:undefined;
function compact(id) { return id.length > 16 ? id.slice(0,8)+"…"+id.slice(-5) : id; }
const KIND = {vehicle:"车辆",pedestrian:"行人",bicycle:"骑行者",cyclist:"骑行者",traffic_cone:"锥桶",barrier:"障碍物",czone_sign:"施工标志",generic_object:"其他物体"};
const worldKind = i => (S.metadata.source_actor_kinds?.[ids[i]] || {1:"vehicle",2:"pedestrian",3:"cyclist"}[S.kinds[i]] || "unknown");
function actorName(i) { return `${own(KIND,worldKind(i))||"对象"} ${i+1} · ${compact(ids[i])}`; }
function kindGroup(kind){return kind==="vehicle"?"vehicles":kind==="pedestrian"?"pedestrians":["bicycle","cyclist"].includes(kind)?"cyclists":"otherObjects";}
function kindVisible(kind){return $(kindGroup(kind)).checked;}
function worldVisible(i){return RUN.presence[frame][i] && $("worldObjects").checked&&kindVisible(worldKind(i));}
function contextFrame(){const i=DATA.context_frame_indices?.[frame];return CONTEXT&&Number.isInteger(i)?CONTEXT.frames[i]:null;}
function contextObjects(){return (contextFrame()?.objects||[]).filter(o=>!worldIds.has(o.id));}
function selectedContextObject(){return contextSelection?.role==="object"?contextObjects().find(o=>o.id===contextSelection.id):null;}
function geometryPoints(geometry){if(geometry.type==="GeometryCollection")return geometry.geometries.flatMap(geometryPoints);const points=[];function visit(v){if(typeof v[0]==="number")points.push(v);else v.forEach(visit);}visit(geometry.coordinates);return points;}
function selectedMapFeature(){return contextSelection?.role==="map"?CONTEXT?.map_features.find(f=>f.id===contextSelection.id):null;}
function featurePosition(feature){const p=geometryPoints(feature.geometry);return p.length?[p.reduce((s,v)=>s+v[0],0)/p.length,p.reduce((s,v)=>s+v[1],0)/p.length]:null;}
function selectedPosition(){return contextSelection?(selectedContextObject()?.pose|| (selectedMapFeature()?featurePosition(selectedMapFeature()):null)):(RUN.presence[frame][selected]?RUN.states[frame][selected]:(RUN.lifecycle?.[frame]?.continuity?.pending?.find(p=>p.actor===ids[selected])?.source_pose||null));}
function speed(run,t,i) { if(!run.presence[t][i])return null; const s=run.states[t][i]; return uncertainReplay(i)?Math.hypot(s[4],s[5]):s[4]*Math.cos(s[3])+s[5]*Math.sin(s[3]); }
function actualAcceleration(run,t,i) {
  if(S.controlled[i]) return run.states[t][i][9]?null:run.motion[t][i][0];
  if(t===0 || !run.presence[t][i] || !run.presence[t-1][i]) return null;
  const state=run.states[t][i], prev=run.states[t-1][i], dt=S.times[t]-S.times[t-1];
  if(uncertainReplay(i))return (Math.hypot(state[4],state[5])-Math.hypot(prev[4],prev[5]))/dt;
  return ((state[4]-prev[4])*Math.cos(state[3])+(state[5]-prev[5])*Math.sin(state[3]))/dt;
}
function outgoing(run,t,i) { return own(run.frames[t].outgoing,ids[i]) || null; }
function controlInfo(run,t,i){return own(run.frames[t].control_sources,ids[i])||{controller:run.metadata.controller_configuration.controller||run.metadata.policy,phase:"configured_without_action",diagnostic_origin:"unavailable"};}
function controlName(info){return info.controller==="baseline_action_replay"?"基准动作开环回放":info.controller;}
function controlCaption(info){const label=info.intervention==="declared_lateral_route_intent"?"横向路线意图干预":info.intervention==="longitudinal_brake_target"?"纵向制动干预":"已记录干预";return `${info.phase==="last_executed"?"最后执行 · ":info.phase==="configured_without_action"?"仅配置 · ":""}${controlName(info)}${info.intervention?" + "+label:""}`;}
function arrivals(run,t,i) { return own(run.frames[t].arrivals,ids[i]) || []; }
function worldToScreen(x,y) { return [(x-camera.x)*camera.scale+mapSize.w/2, mapSize.h/2-(y-camera.y)*camera.scale]; }
function screenToWorld(x,y) { return [(x-mapSize.w/2)/camera.scale+camera.x, (mapSize.h/2-y)/camera.scale+camera.y]; }
function dimensions(element,context) {
  const box=element.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  element.width=Math.max(1,Math.round(box.width*dpr)); element.height=Math.max(1,Math.round(box.height*dpr));
  context.setTransform(dpr,0,0,dpr,0,0); return {w:box.width,h:box.height};
}
function fit(points,padding=25) {
  if(!points.length)return;
  const bounds=points.reduce((b,p)=>[Math.min(b[0],p[0]),Math.min(b[1],p[1]),Math.max(b[2],p[0]),Math.max(b[3],p[1])],[Infinity,Infinity,-Infinity,-Infinity]);
  const xmin=bounds[0]-padding, xmax=bounds[2]+padding;
  const ymin=bounds[1]-padding, ymax=bounds[3]+padding;
  camera.x=(xmin+xmax)/2;camera.y=(ymin+ymax)/2;
  camera.scale=Math.min(mapSize.w/Math.max(xmax-xmin,60),mapSize.h/Math.max(ymax-ymin,60))*.87;
  $("follow").checked=false;drawMap();
}
function fitActors() { fit([...RUN.states[frame].filter((_,i)=>worldVisible(i)).map(s=>s.slice(0,2)),...($("contextObjects").checked?contextObjects().filter(o=>kindVisible(o.kind)).map(o=>o.pose.slice(0,2)):[])],20); }
function currentTrail(run,i){let start=frame;while(start>0&&run.presence[start-1][i])start--;return run.states.slice(start,frame+1).map(row=>row[i]);}
function path(points,stroke,width=1,dash=[]) {
  if(!points.length)return;ctx.beginPath();
  points.forEach((p,i)=>{const q=worldToScreen(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q);});
  ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);
}
function drawGrid() {
  const corners=[screenToWorld(0,0),screenToWorld(mapSize.w,mapSize.h)];
  let interval=10;while(interval*camera.scale<35)interval*=2;while(interval*camera.scale>150)interval/=2;
  const xmin=Math.floor(corners[0][0]/interval)*interval,xmax=corners[1][0];
  const ymin=Math.floor(corners[1][1]/interval)*interval,ymax=corners[0][1];
  ctx.strokeStyle="#ffffff06";ctx.lineWidth=1;ctx.beginPath();
  for(let x=xmin;x<xmax;x+=interval){const p=worldToScreen(x,0);ctx.moveTo(p[0],0);ctx.lineTo(p[0],mapSize.h);}
  for(let y=ymin;y<ymax;y+=interval){const p=worldToScreen(0,y);ctx.moveTo(0,p[1]);ctx.lineTo(mapSize.w,p[1]);}ctx.stroke();
  const pixels=interval*camera.scale;text("scaleText",`${number(interval,0)} m ≈ ${number(pixels,0)} px · 地图 y ↑`);
}
function drawCar(run,t,i,ghost=false) {
  if(!worldVisible(i)||!run.presence[t][i])return;
  const state=run.states[t][i], p=worldToScreen(state[0],state[1]), dims=S.dimensions[i];
  const isSelected=!contextSelection&&i===selected, danger=state[10]||state[11];
  const shade=ghost?color.compare:danger?color.danger:isSelected?color.selected:S.controlled[i]?color.controlled:isReactive(i)?color.traffic:color.replay;
  ctx.save();ctx.translate(...p);ctx.rotate(-state[3]);
  const length=Math.max(dims[0]*camera.scale,4),width=Math.max(dims[1]*camera.scale,3);
  ctx.fillStyle=shade+((state[9]&&!danger)?"66":"b8");ctx.strokeStyle=shade;ctx.lineWidth=isSelected?2:1.2;ctx.setLineDash(ghost?[4,3]:[]);
  entityShape(worldKind(i),length,width);if(!ghost)ctx.fill();ctx.stroke();ctx.setLineDash([]);
  if(!uncertainReplay(i)){ctx.beginPath();ctx.moveTo(length*.12,-width*.28);ctx.lineTo(length*.35,0);ctx.lineTo(length*.12,width*.28);ctx.stroke();}
  if(isSelected&&!ghost){ctx.strokeStyle="#ffffff8c";ctx.lineWidth=1;ctx.strokeRect(-length/2-3,-width/2-3,length+6,width+6);}
  ctx.restore();
  if($("ids").checked&&!ghost){ctx.font="11px sans-serif";ctx.textAlign="center";ctx.fillStyle="#e8f0f6";ctx.fillText(String(i+1),p[0],p[1]-Math.max(width,length)/2-7);}
}
function entityShape(kind,length,width){ctx.beginPath();if(kind==="pedestrian")ctx.ellipse(0,0,Math.max(length/2,3),Math.max(width/2,3),0,0,2*Math.PI);else if(["bicycle","cyclist"].includes(kind)){ctx.moveTo(length/2,0);ctx.lineTo(0,width/2);ctx.lineTo(-length/2,0);ctx.lineTo(0,-width/2);ctx.closePath();}else if(kind==="traffic_cone"){ctx.moveTo(0,-width/2);ctx.lineTo(length/2,width/2);ctx.lineTo(-length/2,width/2);ctx.closePath();}else ctx.rect(-length/2,-width/2,length,width);}
function drawContextObject(object){
  if(!$("contextObjects").checked||!kindVisible(object.kind))return;
  const p=worldToScreen(...object.pose),length=Math.max(object.dimensions[0]*camera.scale,6),width=Math.max(object.dimensions[1]*camera.scale,6),isSelected=contextSelection?.role==="object"&&contextSelection.id===object.id;
  ctx.save();ctx.translate(...p);ctx.rotate(-object.pose[3]);ctx.strokeStyle=isSelected?color.selected:"#e8adcf";ctx.fillStyle="#e8adcf22";ctx.lineWidth=isSelected?2:1.4;ctx.setLineDash([3,3]);entityShape(object.kind,length,width);ctx.fill();ctx.stroke();ctx.setLineDash([]);
  if(!Object.prototype.hasOwnProperty.call(KIND,object.kind)){ctx.beginPath();ctx.moveTo(-length/2,-width/2);ctx.lineTo(length/2,width/2);ctx.moveTo(-length/2,width/2);ctx.lineTo(length/2,-width/2);ctx.stroke();}
  ctx.restore();if($("ids").checked||isSelected){ctx.font="10px sans-serif";ctx.textAlign="center";ctx.fillStyle="#edcce0";ctx.fillText(`${own(KIND,object.kind)||object.kind} · ${compact(object.id)}`,p[0],p[1]-Math.max(length,width)/2-5);}
}
const LAYER = {lanes_polygons:"车道面",lane_connectors:"车道连接",lane_connectors_polygons:"连接区域",baseline_paths:"车道中心线",intersections:"路口",crosswalks:"人行横道",walkways:"步行区域",stop_lines:"停止线",carpark_areas:"停车区域",generic_drivable_areas:"可行驶区域",road_segments:"道路区段",traffic_lights:"交通灯几何",traffic_signs:"交通标志",boundaries:"地图边界线",stop_polygons:"停止区域",dubins_nodes:"地图参考节点",lane_group_connectors:"车道组连接",lane_groups_polygons:"车道组区域",gen_lane_connectors_scaled_width_polygons:"连接器辅助区域"};
function layerName(layer){return own(LAYER,layer)||layer;}
function diagnosticLayer(layer){return layer==="road_segments"||layer==="dubins_nodes"||layer==="lane_connectors"||layer.startsWith("lane_group")||layer.startsWith("gen_");}
function geometryRank(geometry){return ["Point","MultiPoint"].includes(geometry.type)?2:["LineString","MultiLineString"].includes(geometry.type)?1:0;}
function groundRank(layer){return own({generic_drivable_areas:0,road_segments:0,intersections:1,lanes_polygons:2,lane_connectors:3,lane_connectors_polygons:3,carpark_areas:4,walkways:5,crosswalks:6,stop_polygons:7},layer)??1;}
function geometryParts(feature,geometry=feature.geometry){return geometry.type==="GeometryCollection"?geometry.geometries.flatMap(g=>geometryParts(feature,g)):[{feature,geometry}];}
const semanticDrawables=(CONTEXT?.map_features||[]).flatMap(feature=>geometryParts(feature)).sort((a,b)=>geometryRank(a.geometry)-geometryRank(b.geometry)||groundRank(a.feature.layer)-groundRank(b.feature.layer));
function mapStyle(layer,active){
  if(active)return {stroke:"#62acff",fill:"#62acff33",width:1.3,dash:[]};
  if(layer==="baseline_paths")return {stroke:"#98aab16e",fill:"#00000000",width:.8,dash:[5,7]};
  if(layer==="boundaries")return {stroke:"#84949d75",fill:"#00000000",width:.8,dash:[]};
  if(layer==="walkways")return {stroke:"#a6ae6966",fill:"#a6ae692d",width:.6,dash:[]};
  if(layer==="crosswalks")return {stroke:"#a68cc68c",fill:"#a68cc642",width:.8,dash:[]};
  if(layer==="carpark_areas")return {stroke:"#7caa8570",fill:"#7caa8534",width:.6,dash:[]};
  if(layer==="traffic_lights")return {stroke:"#9bafb6",fill:"#00000000",width:1.1,dash:[]};
  if(["stop_lines","stop_polygons"].includes(layer))return {stroke:"#d6ba8899",fill:"#d6ba8834",width:1.1,dash:[]};
  return {stroke:diagnosticLayer(layer)?"#71848b45":"#8b979b48",fill:diagnosticLayer(layer)?"#8892970d":"#88929722",width:.6,dash:[]};
}
function drawGeometry(geometry,stroke,fill,pointSymbol,lineStyle){
  const point=p=>{const q=worldToScreen(...p);ctx.save();ctx.translate(...q);ctx.strokeStyle=stroke;ctx.fillStyle=stroke;ctx.lineWidth=1.1;if(pointSymbol==="reference_node"){ctx.beginPath();ctx.moveTo(-3,0);ctx.lineTo(3,0);ctx.moveTo(0,-3);ctx.lineTo(0,3);ctx.stroke();}else if(pointSymbol==="traffic_light_geometry"){ctx.fillStyle="#101d2a";ctx.fillRect(-3.5,-6.5,7,13);ctx.strokeRect(-3.5,-6.5,7,13);ctx.fillStyle=stroke;for(const y of [-4,0,4])ctx.fillRect(-1.5,y-1,3,2);}else ctx.fillRect(-2,-2,4,4);ctx.restore();};
  const polygon=rings=>{ctx.beginPath();for(const ring of rings){ring.forEach((p,i)=>{const q=worldToScreen(...p);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();}ctx.fillStyle=fill;ctx.fill("evenodd");ctx.strokeStyle=stroke;ctx.lineWidth=lineStyle.width;ctx.stroke();};
  if(geometry.type==="Point")point(geometry.coordinates);else if(geometry.type==="MultiPoint")geometry.coordinates.forEach(point);else if(geometry.type==="LineString")path(geometry.coordinates,stroke,lineStyle.width,lineStyle.dash);else if(geometry.type==="MultiLineString")geometry.coordinates.forEach(p=>path(p,stroke,lineStyle.width,lineStyle.dash));else if(geometry.type==="Polygon")polygon(geometry.coordinates);else if(geometry.type==="MultiPolygon")geometry.coordinates.forEach(polygon);else if(geometry.type==="GeometryCollection")geometry.geometries.forEach(g=>drawGeometry(g,stroke,fill,pointSymbol,lineStyle));
}
function signalPosition(record){
  const node=own(DATA.graph,"connector:"+record.connector_id);if(node?.points?.length)return node.points[0];
  const feature=CONTEXT?.map_features.find(f=>["lane_connectors","lane_connectors_polygons"].includes(f.layer)&&f.id===`${f.layer}:${record.connector_id}`);return feature?featurePosition(feature):null;
}
function drawSemanticMap(phase){
  for(const {feature,geometry}of semanticDrawables){if(geometryRank(geometry)!==phase||!mapLayerControls.get(feature.layer)?.checked)continue;const active=contextSelection?.role==="map"&&contextSelection.id===feature.id,style=mapStyle(feature.layer,active),pointSymbol=feature.layer==="dubins_nodes"?"reference_node":feature.layer==="traffic_lights"?"traffic_light_geometry":"map_point";drawGeometry(geometry,style.stroke,style.fill,pointSymbol,style);}
  if(phase===2&&$("signals").checked)for(const record of contextFrame()?.signals.records||[]){const anchor=signalPosition(record);if(!anchor)continue;const p=worldToScreen(...anchor);ctx.fillStyle=record.is_red?"#f66d6d":"#c4b2e9";ctx.fillRect(p[0]-5,p[1]-5,10,10);ctx.strokeStyle="#182b3e";ctx.lineWidth=1.2;ctx.strokeRect(p[0]-5,p[1]-5,10,10);ctx.beginPath();ctx.moveTo(p[0]-2.5,p[1]);ctx.lineTo(p[0]+2.5,p[1]);ctx.stroke();if($("ids").checked){ctx.font="10px sans-serif";ctx.fillStyle="#f1dee0";ctx.textAlign="left";ctx.fillText(record.is_red?"连接器红灯":"连接器非红（绿/黄未知）",p[0]+8,p[1]-6);}}
}
function drawPending() {
  const life=RUN.lifecycle?.[frame]?.continuity;
  for(const row of life?.pending||[]){
    const i=ids.indexOf(row.actor);if(i<0||!kindVisible(worldKind(i)))continue;
    const pose=row.source_pose,p=worldToScreen(pose[0],pose[1]),dims=S.dimensions[i];
    ctx.save();ctx.translate(...p);ctx.rotate(-pose[3]);ctx.strokeStyle="#f6b75b";ctx.lineWidth=1.5;ctx.setLineDash([3,4]);
    entityShape(worldKind(i),Math.max(dims[0]*camera.scale,5),Math.max(dims[1]*camera.scale,4));ctx.stroke();ctx.restore();
    if($("ids").checked){ctx.fillStyle="#f6b75b";ctx.font="11px sans-serif";ctx.fillText(String(i+1)+" 待接管",p[0],p[1]-8);}
  }
}
function drawMap() {
  if($("follow").checked){const p=selectedPosition();if(p){camera.x=p[0];camera.y=p[1];}}
  ctx.clearRect(0,0,mapSize.w,mapSize.h);drawGrid();
  drawSemanticMap(0);drawSemanticMap(1);
  if($("roads").checked)S.roads.forEach(road=>path(road.points,road.type===6?"#698193":"#40586c",road.type===6?1.5:1,road.type===4?[4,7]:[]));
  if($("graph").checked&&DATA.graph)Object.values(DATA.graph).forEach(node=>path(node.points,node.kind==="connector"?"#8d839d":"#618680",1));
  drawSemanticMap(2);
  const route=contextSelection?null:own(RUN.routes,ids[selected]);
  if($("routes").checked&&route){path(route.points,"#619eff66",5);path(route.points,"#8bb8ff",1.4);}
  if(!contextSelection&&$("reference").checked&&worldVisible(selected)){let segment=[];for(let t=0;t<S.times.length;t++){if(RUN.presence[t][selected])segment.push(S.reference_poses[t][selected]);else{path(segment,"#e5bd73",1.5,[7,5]);segment=[];}}path(segment,"#e5bd73",1.5,[7,5]);
    const ref=S.reference_poses[frame][selected],p=worldToScreen(ref[0],ref[1]);ctx.beginPath();ctx.arc(...p,5,0,Math.PI*2);ctx.strokeStyle="#e5bd73";ctx.lineWidth=1.5;ctx.stroke();}
  if($("trails").checked){for(let i=0;i<ids.length;i++)if(worldVisible(i))path(currentTrail(RUN,i),i===selected&&!contextSelection?"#68afffcc":"#5c9c995e",i===selected&&!contextSelection?2:1);
    if(!contextSelection&&worldVisible(selected)&&OTHER&&$("comparison").checked)path(currentTrail(OTHER,selected),"#b59bdd",1.5,[5,4]);}
  contextObjects().forEach(drawContextObject);drawPending();
  if(OTHER&&$("comparison").checked)ids.forEach((_,i)=>drawCar(OTHER,frame,i,true));
  ids.forEach((_,i)=>{if(i!==selected)drawCar(RUN,frame,i);});drawCar(RUN,frame,selected);
  if(SOURCE||contextSelection||!worldVisible(selected))return;
  const goal=RUN.states[frame][selected].slice(7,9),p=worldToScreen(...goal);
  ctx.strokeStyle="#80a9c788";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0]-4,p[1]);ctx.lineTo(p[0]+4,p[1]);ctx.moveTo(p[0],p[1]-4);ctx.lineTo(p[0],p[1]+4);ctx.stroke();
}
function drawChart() {
  const {w,h}=dimensions(chart,chartCtx),margin={l:35,r:35,t:16,b:22},width=w-margin.l-margin.r,height=h-margin.t-margin.b;
  if(contextSelection){chartCtx.clearRect(0,0,w,h);text("chartNote","当前选择源上下文或地图要素：没有仿真动作、执行器状态或反事实运动曲线。请选择世界对象查看控制诊断。");return;}
  text("chartSpeedLabel",uncertainReplay(selected)?"平面速率 m/s":"纵向速度 m/s");
  text("chartAccelLabel",SOURCE?"源速率变化 m/s²":uncertainReplay(selected)?"速率变化 m/s²":"实际加速度 m/s²");
  const speeds=S.times.map((_,t)=>speed(RUN,t,selected)),actual=S.times.map((_,t)=>actualAcceleration(RUN,t,selected));
  const targets=S.times.map((_,t)=>controlInfo(RUN,t,selected).diagnostic_origin==="baseline_reference"?null:outgoing(RUN,t,selected)?.diagnostics.desired_acceleration??null);
  const maxSpeed=Math.max(5,...speeds)*1.12,minSpeed=Math.min(0,...speeds)*1.12,allAccel=[...actual,...targets].filter(Number.isFinite);
  const minAccel=Math.min(-1,...allAccel)-.4,maxAccel=Math.max(1,...allAccel)+.4;
  const x=t=>margin.l+t/Math.max(last,1)*width,ySpeed=v=>margin.t+height-(v-minSpeed)/(maxSpeed-minSpeed)*height,yAccel=v=>margin.t+height-(v-minAccel)/(maxAccel-minAccel)*height;
  chartCtx.clearRect(0,0,w,h);chartCtx.font="10px sans-serif";chartCtx.lineWidth=1;
  for(let n=0;n<=2;n++){const y=margin.t+n*height/2;chartCtx.strokeStyle="#e8eef2";chartCtx.beginPath();chartCtx.moveTo(margin.l,y);chartCtx.lineTo(w-margin.r,y);chartCtx.stroke();chartCtx.fillStyle="#71879b";chartCtx.textAlign="right";chartCtx.fillText(number(maxSpeed-n*(maxSpeed-minSpeed)/2,1),margin.l-7,y+3);chartCtx.textAlign="left";chartCtx.fillText(number(maxAccel-n*(maxAccel-minAccel)/2,1),w-margin.r+6,y+3);}
  function line(values,transform,stroke,dashed=false){chartCtx.beginPath();let started=false;values.forEach((v,t)=>{if(!Number.isFinite(v)){started=false;return;}started?chartCtx.lineTo(x(t),transform(v)):chartCtx.moveTo(x(t),transform(v));started=true;});chartCtx.strokeStyle=stroke;chartCtx.lineWidth=1.6;chartCtx.setLineDash(dashed?[4,3]:[]);chartCtx.stroke();chartCtx.setLineDash([]);}
  line(speeds,ySpeed,"#3478e5");line(actual,yAccel,"#008488");line(targets,yAccel,"#c8943c",true);
  const cursor=x(frame);chartCtx.strokeStyle="#50687988";chartCtx.beginPath();chartCtx.moveTo(cursor,margin.t-4);chartCtx.lineTo(cursor,h-margin.b);chartCtx.stroke();
  chartCtx.fillStyle="#7b8b97";chartCtx.textAlign="left";chartCtx.fillText("0 s",margin.l,h-5);chartCtx.textAlign="right";chartCtx.fillText(number(S.times[last],1)+" s",w-margin.r,h-5);
  text("chartNote",uncertainReplay(selected)?"左轴：平面速率；右轴：速率差分。源框存在轴向歧义，不能由校正后的框轴判断车头或倒车。":S.controlled[selected]?"左轴：有符号纵向速度；右轴：执行器加速度。终止后不显示遗留加速度；目标线仅显示本世界的反馈决策或干预目标；动作开环回放段的基准参考诊断不画成当前目标，末帧无目标值。":isReactive(selected)?"左轴：平面速率；右轴：差分加速度。交通控制按当前世界反馈，目标与执行记录分开保存。":"左轴：有符号纵向速度；右轴：由相邻速度差分得到的区间加速度。回放车辆没有控制动作或目标加速度。");
}
function fleetMode(i) {return S.metadata.vehicle_fleet?.roles?.[ids[i]]?.mode;}
function fleetCaption(i) {return ({policy_action:"策略动作车辆",route_feedback:"交通控制 · 路线反馈",bounded_motion_feedback:"交通控制 · 有界推演",stationary_hold:"交通控制 · 停车保持",uncertain_brake:"交通控制 · 不确定运动制动",outside_window:"窗口外 · 未入场"})[fleetMode(i)]||"";}
const dependencyFrames = RUN.frames.map((f,t)=>{
  const links=new Map();
  const add=(a,b,reason)=>{if(!worldIds.has(a)||!worldIds.has(b)||a===b||!RUN.presence[t][ids.indexOf(a)]||!RUN.presence[t][ids.indexOf(b)])return;const row=links.get(a)||[];if(!row.some(e=>e.target===b))row.push({target:b,reason});links.set(a,row);};
  if(SOURCE)return links;
  for(const c of RUN.lifecycle?.[t]?.executed_traffic_controls||[]){add(c.actor,c.conflict_id,c.conflict_reason||"交通冲突预测");if(c.target_speed_mps<.2)add(c.actor,c.following_id,"跟车保持");}
  for(const [a,o] of Object.entries(f.outgoing||{})){
    if(o.control_source?.diagnostic_origin!=="current_world")continue;
    for(const h of o.diagnostics?.traffic_hazards||[])add(a,h.actor,h.reason);
    if(o.diagnostics?.lead_id)add(a,o.diagnostics.lead_id,"跟驰前车");
  }
  return links;
});
function jumpEvidence(actor,t){pause();selectActor(ids.indexOf(actor));setFrame(t);}
function updateDependencies(){
  const graph=$("dependencyGraph"),history=$("dependencyHistory"),events=$("sourceIntegrityEvents");graph.replaceChildren();history.replaceChildren();events.replaceChildren();
  text("dependencyMeaning",SOURCE?"源轨迹没有控制器，不生成避让关系。":"主分支当前记录的前车/冲突关系；循环依赖不是交通路权或事故责任判定。末帧没有新决策。");
  const label=a=>a==="ego"?"Ego":`${ids.indexOf(a)+1}号车`;
  const selectedId=ids[selected],queue=[selectedId],visited=new Set(),related=new Set([selectedId]);let count=0;
  while(queue.length&&count<32){const a=queue.shift();if(visited.has(a))continue;visited.add(a);
    for(const edge of dependencyFrames[frame].get(a)||[]){
      if(count++>=32)break;related.add(edge.target);const b=document.createElement("button");b.textContent=`${label(a)} → ${label(edge.target)} · ${edge.reason}`;b.style="display:block;margin:6px 0;max-width:100%;white-space:normal;text-align:left";b.onclick=()=>jumpEvidence(edge.target,frame);graph.append(b);queue.push(edge.target);
    }
  }
  if(count>=32){const p=document.createElement("p");p.textContent="关系较多，当前仅显示前32条；可选车继续检查。";graph.append(p);}
  if(!graph.childNodes.length)graph.textContent=SOURCE?"无控制器关系":frame===last?"最终记录状态：无待执行决策":"选中对象没有当前关系记录；不代表场景安全。";
  const past=new Map();for(let t=0;t<frame;t++)for(const e of dependencyFrames[t].get(selectedId)||[]){const row=past.get(e.target)||{first:t,count:0};row.count++;past.set(e.target,row);}
  for(const [a,row]of past){related.add(a);const b=document.createElement("button");b.textContent=`${label(selectedId)} → ${label(a)} · 首次 ${number(S.times[row.first],1)}s / 累计${row.count}帧`;b.style="display:block;margin:6px 0";b.onclick=()=>jumpEvidence(selectedId,row.first);history.append(b);}
  if(!past.size)history.textContent="无此前关系记录";
  const audit=DATA.source_integrity;
  text("sourceIntegrityMeaning",!audit?"未附加入场源几何检查。":`源框入场检查共 ${audit.events.length} 条重叠记录。下列仅展示与选中/关联对象有关且不晚于本帧的记录；不合并身份、不忽略碰撞。没有记录不等于通过安全验收。`);
  for(const event of audit?.events||[]){if(event.tick>frame||!event.actors.some(a=>related.has(a)))continue;const b=document.createElement("button");b.textContent=`${number(event.time_seconds,1)}s：${event.actors.map(label).join(" / ")} 源框重叠`;b.style="display:block;margin:6px 0";b.onclick=()=>jumpEvidence(event.entering_actors[0],event.tick);events.append(b);}
  if(!events.childNodes.length)events.textContent="当前筛选下没有源入场重叠记录。";
  const pair=DATA.inspection?.paired_view;
  if(pair&&typeof pair.path==="string"){
    const url=new URL(pair.path,location.href);if(url.origin===location.origin){url.searchParams.set("actor",selectedId);url.searchParams.set("tick",String(frame));$("pairedView").href=url.href;$("pairedView").textContent=pair.label;$("pairedView").classList.remove("hidden");}
  }
}
function updateSidebar() {
  updateContext();
  updateDependencies();
  $("worldDetails").classList.toggle("hidden",Boolean(contextSelection));$("contextDetails").classList.toggle("hidden",!contextSelection);$("actorCard").classList.toggle("context-active",Boolean(contextSelection));
  if(contextSelection){
    const box=$("contextDetails");box.replaceChildren();const detail=(label,value)=>{const p=document.createElement("p");p.textContent=label+"："+value;box.append(p);};
    detail("身份",contextSelection.id);
    if(contextSelection.role==="map"){const feature=selectedMapFeature();detail("角色","静态地图要素 · 仅显示");detail("图层",layerName(feature.layer));detail("几何",feature.geometry.type);detail("图层状态",mapLayerControls.get(feature.layer)?.checked?"显示":"已隐藏");detail("属性",JSON.stringify(feature.properties));detail("规则与交互","显示地图语义不表示当前已实现相应碰撞、信号约束或路权");}
    else {const source=contextFrame(),object=selectedContextObject();detail("角色","源日志上下文 · 不参与本次世界推进");detail("源日志",CONTEXT.metadata.source_log);detail("源采样",source?`${number(source.time_seconds,3)} s / ${source.timestamp_us} μs；年龄 ${number(S.times[frame]-source.time_seconds,3)} s`:"当前时刻没有有效源采样");detail("当前可见性",!object?"此采样未标注该对象；不沿用旧位置":$("contextObjects").checked&&kindVisible(object.kind)?"显示源记录位置":"已被显示过滤器隐藏");if(object){detail("原始类别",(own(KIND,object.kind)||"未定义类别")+" / "+object.kind);detail("位置 x / y / z",object.pose.slice(0,3).map(v=>number(v)).join(" / ")+" m");detail("朝向",number(object.pose[3]*180/Math.PI,1)+"°");detail("尺寸 长 / 宽 / 高",object.dimensions.map(v=>number(v)).join(" / ")+" m");}detail("控制与运动","无策略动作、无执行器状态；不推断速度，不作为反事实运动真值");}
    return;
  }
  const id=ids[selected],state=RUN.states[frame][selected],action=outgoing(RUN,frame,selected),diag=action?.diagnostics||{},events=arrivals(RUN,frame,selected),route=own(RUN.routes,id);
  const executedTraffic = RUN.lifecycle?.[frame]?.executed_traffic_controls?.find(d=>d.actor===id);
  const background = executedTraffic || RUN.lifecycle?.[frame]?.background_decisions?.find(d=>d.actor===id);
  const control=controlInfo(RUN,frame,selected),reference=control.diagnostic_origin==="baseline_reference",openLoop=control.controller==="baseline_action_replay";
  text("actorId",id);text("controlSource",fleetCaption(selected)||(S.controlled[selected]?controlCaption(control):isReactive(selected)?"反应式背景 · 当前世界反馈":"世界日志回放 · 无策略动作"));
  text("diagnosticSource",!S.controlled[selected]?(isReactive(selected)?`当前状态冲突预测：${background?.conflict_id||"无"}；源标注${RUN.lifecycle?.[frame]?.source_supported?.[selected]?"支持":"中断，仿真仍持有对象"}。规则控制，不代表人类行为模型。${S.metadata.observed_motion_inference?.[id]?" 本对象使用源速度构造有限导航意图，缺标后的运动属于仿真推演。":""}`:"世界回放对象没有策略诊断。"):!action?(control.phase==="last_executed"?`本帧无待执行动作或当前诊断；最后执行来源对应 t = ${number(S.times[control.command_tick],2)} s。`:"没有已执行动作或策略诊断。"):reference?"以下目标、前车、间距和路线跟踪诊断来自基准分支，仅作参考；当前车按已记录动作开环执行，没有重新感知这些值。":control.diagnostic_origin==="unavailable"?"本帧动作已记录，但没有对应策略诊断。":control.intervention==="declared_lateral_route_intent"?"当前世界反馈诊断；控制器参考路线已被替换，世界任务终点保持不变。实际执行动作以记录为准。":control.intervention?"当前世界反馈诊断；干预已覆盖控制器输出。实际执行动作以记录为准。":"当前世界反馈决策；诊断对应本帧待执行动作。");
  text("targetLabel",reference?"基准分支参考加速度":"本帧目标加速度");
  text("leadLabel",reference?"基准分支前车（参考）":"感知前车");
  text("gapLabel",reference?"基准间距 / TTC":"车身间距 / TTC");
  text("recoveryLabel",reference?"基准目标速度 / 模式":"目标速度 / 恢复模式");
  text("headingErrorLabel",reference?"基准路线朝向误差":"路线朝向误差");
  text("routeSafetyLabel",reference?"初始路线 / 基准制动诊断":"路线 / 前向制动状态");
  text("speedLabel",uncertainReplay(selected)?"平面速率（车头方向未确认）":"纵向速度（负值为倒车）");
  text("metricSpeed",number(speed(RUN,frame,selected)));text("metricAccel",number(actualAcceleration(RUN,frame,selected)));text("accelLabel",S.controlled[selected]?"实际纵向加速度":uncertainReplay(selected)?"速率变化（差分）":"区间纵向加速度（差分）");
  text("speedComparison",`${number(Math.hypot(state[4],state[5]))} / ${number(S.reference_planar_speeds_mps?.[frame]?.[selected])} m/s（均为平面速率；参考来自插值日志）`);
  text("configuredSpeed",S.controlled[selected]?!action?"本帧无待执行策略":`${number(diag.configured_desired_speed_mps??RUN.metadata.controller_configuration.parameters?.desired_speed)} m/s；${openLoop?"基准分支配置，当前来源为动作开环回放":"每帧目标见下方，不等于原司机速度"}`:isReactive(selected)?`${number(backgroundTasks[id].desired_speed)} m/s；向路径终点有界制动，可能越过终点；保留碰撞实体`:"世界回放，无控制器期望速度");
  const stationary = S.metadata.vehicle_fleet?.roles?.[id]?.stationary_evidence;
  const measured = stationary?.source;
  text("stationaryEvidence",stationary?.accepted?`首连续源段 ${measured.finite_velocity_samples}/${measured.samples} 条速度；最大 ${number(measured.max_reported_speed_mps)} m/s，位置漂移 ${number(measured.max_position_excursion_m)} m。离线配置停车保持，非在线预测。`:"未通过该证据规则 / 未提供；不能据此推断正在行驶。");
  const corrected=headingCorrections.filter(item=>item.actor_id===id);
  text("headingQuality",corrected.length?`${corrected.length} 处源框轴向校正（${corrected.map(item=>number(item.start_time_seconds,2)+"–"+number(item.end_time_seconds,2)+" s").join("、")}）；${S.controlled[selected]?"当前朝向由动力学推进，日志仅参考":"车头方向未确认，不推断倒车"}；原始 Context 未改`:(S.metadata.heading_audit?"未触发低速 π 跳变规则（不代表标注无误）":"旧资产未执行朝向歧义审计"));
  text("metricTarget",number(frame===last?null:(diag.desired_acceleration??background?.acceleration_mps2)));text("metricDeviation",number(own(RUN.route_distance,id)?.[frame]));
  text("position",`${number(state[0])} / ${number(state[1])} m`);text("heading",`${number(state[3]*180/Math.PI,1)}° / ${S.controlled[selected]?number(RUN.motion[frame][selected][2]*180/Math.PI,1)+"°":"未测量"}`);
  text("arrivals",events.length?events.map(e=>EVENT[e]).join(" · "):"无");text("lead",background?.conflict_id?compact(background.conflict_id):diag.lead_id?compact(diag.lead_id):action?"未选择":"—");
  text("recovery",`${number(diag.target_speed_mps)} m/s / ${diag.recovery_mode??"未记录"}`);text("headingError",Number.isFinite(diag.heading_error_radians)?number(diag.heading_error_radians*180/Math.PI,1)+"°":"—");
  text("gap",`${number(diag.bumper_gap_m)} m / ${number(diag.ttc_seconds)} s`);text("routeStart",route?.node_ids?.[0]||"未提供节点");text("routeEnd",route?(own(END,route.end_reason)||route.end_reason):"未提供路线");text("speedLimit",route?.speed_limit_mps?number(route.speed_limit_mps)+" m/s":"未知 / 未声明");
  const matching=own(RUN.metadata.controller_configuration.routing?.actors,id);text("routeMatch",matching?`${own(MATCH,matching.reason)||matching.reason} / ${matching.candidates?.length??"未知"} 个${Object.keys(matching.requested_successors||{}).length?" / 已指定下游分岔":""}`:"未记录");
  text("routeSafety",`${matching?.initial_route_feasibility?.insufficient_stopping_distance?"初始路线长度不足 / ":""}${diag.route_exhausted?"已越过路线末端 / ":""}${diag.forward_release_feasible===false?"当前状态无法保证前向释放制动":diag.forward_release_feasible===true?"前向制动释放可行":"无本帧诊断"}${diag.traffic_hazards?.length?" / 冲突处理："+diag.traffic_hazards.map(h=>compact(h.actor)+" "+(h.reason==="crossing_yield"?"交叉让行（仿真优先级）":"包围盒预测冲突")+(Number.isFinite(h.time_seconds)?" "+number(h.time_seconds)+" s":"")).join("；"):""}`);
  if(action && Array.isArray(action.action)){text("actionInterval",`${diag.controller==="Harness-PDM-route-repair"?"Harness PDM 纠正控制":diag.controller==="Harness-native-recovery-search-v1"?"Harness 候选搜索纠正（非模型原生动作）":diag.controller==="Harness-bounded-perturbation"?"Harness 有界扰动（非模型原生动作）":"原生连续控制"} · ${number(S.times[frame],1)} → ${number(S.times[frame+1],1)} s`);text("actionValue",`纵向 jerk ${number(action.action[0],4)} m/s³ / 转向角速度 ${number(action.action[1],4)} rad/s`);}
  else if(action){const index=action.action,long=[-15,-4,0,4][Math.floor(index/3)],lat=[-4,0,4][index%3];text("actionInterval",`待执行${openLoop?"基准回放":""}动作 · ${number(S.times[frame],1)} → ${number(S.times[frame+1],1)} s`);text("actionValue",`#${index} · 纵向 jerk ${long} / 横向 jerk ${lat} m/s³`);}
  else {text("actionInterval",frame===last?"最终状态 · 没有下一步动作":isReactive(selected)?(executedTraffic?`交通反馈已执行 · tick ${executedTraffic.observation_tick} → ${executedTraffic.execution_tick}`:"反应式背景预测 · 当前世界反馈"):"本帧无有效控制动作");text("actionValue",S.controlled[selected]?(frame===last?"回合记录已结束":"车辆已终止 / 失效"):frame===last?"回合记录已结束":isReactive(selected)?`背景目标速度 ${number(background?.target_speed_mps)} m/s；预测冲突 ${number(background?.predicted_conflict_seconds)} s`:"位置由日志回放推进");}
  let status="交互中",statusClass="status";
  if(state[10]||state[11]){status=state[10]?"碰撞后停止":"越界后停止";statusClass+=" warning";}
  else if(state[12]){status="已到达目标";statusClass+=" success";}
  else if(state[9])status=frame===0?"初始化已停止":"已停止";else if(frame===last)status="记录结束";else if(!S.controlled[selected])status=isReactive(selected)?"反应式交互中":"回放中";
  if(!RUN.presence[frame][selected]){status="未在场（源日志未标注）";for(const key of ["metricSpeed","metricAccel","position","heading","speedComparison"])text(key,"—");text("actionValue","没有物理实体，不参与当前碰撞或决策");text("diagnosticSource","当前未在场，没有控制执行或冲突预测；分配的控制模式在实际入场后生效。");}
  const continuity=RUN.lifecycle?.[frame]?.continuity;
  if(continuity){
    const phase=continuity.status[selected],pending=continuity.pending.find(p=>p.actor===id);
    if(phase==="pending"){
      status="等待安全接管 · 身份保留";
      text("diagnosticSource","源标注入场框不满足几何/时距条件；延迟 "+number((pending?.delay_steps||0)*.1,1)+" s。阻挡对象："+(pending?.blocking_actors.join("、")||"待核查")+"。橙色虚框为入口位置，不是已进入世界的实体。");
      text("actionValue","等待接管；不删除身份，不把标注重叠注入物理世界");
    }else if(phase==="departed"){
      status="已沿地图路线越过边界离场";
      text("diagnosticSource","完整物体包围盒已越过声明的仿真边界，且速度朝外；不是按标注消失或超时删除。");
    }else if(phase==="unseen")status="尚未到源入场时刻";
  }
  text("actorStatus",status);$("actorStatus").className=statusClass;
  if(SOURCE){
    text("controlSource","源日志编译回放 · 无控制器");
    text("diagnosticSource","显示日志插值位置与速度；原始标注可能中断，不据此判断物理离场。未执行碰撞或官方 PDM 评分。");
    text("actionInterval","源轨迹参考 · 无动作执行");text("actionValue","本页面不运行策略或物理控制");
    text("arrivals","未进行驾驶事件评估");text("metricTarget","—");
  }
  if(OTHER){const alternate=OTHER.states[frame][selected],delta=Math.hypot(alternate[0]-state[0],alternate[1]-state[1]);text("compareValues",`速度 ${number(speed(OTHER,frame,selected))} m/s · 位置差 ${number(delta)} m`);const e=arrivals(OTHER,frame,selected);text("compareEvents",`${S.controlled[selected]?controlCaption(controlInfo(OTHER,frame,selected)):"世界日志回放"} · 本帧事件：${e.length?e.map(v=>EVENT[v]).join(" / "):"无"}`);}
}
function updateContext(){
  const source=contextFrame(),objects=contextObjects(),worldShown=ids.filter((_,i)=>worldVisible(i)).length,contextShown=$("contextObjects").checked?objects.filter(o=>kindVisible(o.kind)).length:0;
  text("populationCaption",`世界在场 ${RUN.presence[frame].filter(Boolean).length} / ${ids.length} 槽位（显示 ${worldShown}） · 仅源日志 ${objects.length} 个（显示 ${contextShown}）${source?" · 源采样 "+number(source.time_seconds,2)+" s":" · 源采样不可用"}`);
  const lifecycle=RUN.lifecycle?.[frame]?.continuity;
  if(lifecycle){
    const pending=lifecycle.status.filter(s=>s==="pending").length,departed=lifecycle.status.filter(s=>s==="departed").length;
    $("populationCaption").textContent+=" · 等待接管 "+pending+" · 有据离场 "+departed+"（橙色虚框为等待身份）";
  }
  const loaded=CONTEXT?.map_features.length||0,shown=(CONTEXT?.map_features||[]).filter(f=>mapLayerControls.get(f.layer)?.checked).length;
  text("mapViewStatus",CONTEXT?`已加载 ${loaded} 个地图要素；当前显示 ${shown} 个。下列覆盖数是已加载数，不等于当前显示数。${$("mapPreset").value==="all"?"全部图层用于几何审计，多套几何可能重叠。":"诊断图层可以逐项开启；显示开关不删除源数据。"}`:"未附加语义地图；使用现有仿真道路几何。");
  const box=$("signalDetails");box.replaceChildren();const p=document.createElement("p");p.textContent=!CONTEXT?"信号状态：未提供上下文":!source?"信号状态：当前时刻无有效源采样":!source.signals.available?"信号状态：此源帧未提供":!source.signals.records.length?"信号状态：源帧已提供，记录为空":`信号状态：${source.signals.records.length} 条源记录；红色=红灯，紫色=非红（绿/黄未知）；均为源时间状态。`;box.append(p);
  for(const record of source?.signals.records||[]){const row=document.createElement("p");row.textContent=`连接器 ${record.connector_id} · ${record.is_red?"红灯":"非红（绿/黄未知）"} · ${signalPosition(record)?"连接器有空间位置":"没有可匹配的连接器几何"}`;box.append(row);}
}
function applyMapPreset(value,redraw=true){
  const showAll=value==="all";for(const [layer,input]of mapLayerControls)input.checked=!input.disabled&&(showAll||!diagnosticLayer(layer));
  const hasSemanticRoads=(CONTEXT?.map_features||[]).some(f=>mapLayerControls.get(f.layer)?.checked&&["baseline_paths","lanes_polygons","lane_connectors","lane_connectors_polygons","generic_drivable_areas","intersections"].includes(f.layer));
  $("roads").checked=showAll||!hasSemanticRoads;$("graph").checked=showAll&&Boolean(DATA.graph);$("mapPreset").value=value;
  if(redraw){updateSidebar();drawMap();}
}
function setupContext(){
  const chooser=$("contextActor"),empty=document.createElement("option");empty.value="";empty.textContent="选择仅显示的上下文对象 / 地图要素";chooser.append(empty);
  const kinds=new Map();for(const source of CONTEXT?.frames||[])for(const object of source.objects)if(!worldIds.has(object.id)&&!kinds.has(object.id))kinds.set(object.id,object.kind);
  const group=document.createElement("optgroup");group.label="源日志对象 · 无仿真动作";for(const [id,kind]of kinds){const option=document.createElement("option");option.value=JSON.stringify({role:"object",id});option.textContent=`${own(KIND,kind)||kind} · ${compact(id)}`;group.append(option);}if(group.children.length)chooser.append(group);
  const maps=document.createElement("optgroup");maps.label="地图语义 · 静态显示";for(const feature of CONTEXT?.map_features||[]){const option=document.createElement("option");option.value=JSON.stringify({role:"map",id:feature.id});option.textContent=layerName(feature.layer)+" · "+feature.id;maps.append(option);}if(maps.children.length)chooser.append(maps);chooser.disabled=!kinds.size&&!maps.children.length;
  for(const [layer,info]of Object.entries(CONTEXT?.map_layers||{})){const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.checked=info.availability==="available";input.disabled=info.availability!=="available";input.setAttribute("aria-label",layerName(layer));mapLayerControls.set(layer,input);input.onchange=()=>{$("mapPreset").value="custom";updateSidebar();drawMap();};label.append(input,document.createTextNode(layerName(layer)+(diagnosticLayer(layer)?"（诊断）":"")));$("semanticLayers").append(label);const status=document.createElement("p");status.textContent=`${layerName(layer)}：${info.availability==="unavailable"?"源地图未提供":info.availability==="loaded_empty"?"已读取；范围内为空":`${info.feature_count} 个要素`}${layer==="traffic_lights"?"；灯杆几何与连接器状态未自动绑定":""}`;$("mapAvailability").append(status);}
  if(!CONTEXT)text("mapAvailability","未附加源上下文与语义地图；无法判断未显示对象是否不存在。");
  else {const p=document.createElement("p"),bounds=CONTEXT.metadata.map_bounds_local;const extent=bounds?`x ${number(bounds[0],1)} 至 ${number(bounds[2],1)} m，y ${number(bounds[1],1)} 至 ${number(bounds[3],1)} m`:"范围未声明";p.textContent=`源人口：${kinds.size} 个仅显示对象（整个窗口的 ID 并集）；地图只覆盖已加载区域：${extent}。`;$("mapAvailability").append(p);}
  text("contextNotice",CONTEXT?"虚线上下文来自原始日志，和当前仿真分支可能不一致；无独立交互、碰撞或反事实真值。源记录采用最近过去采样，最长保留 "+CONTEXT.metadata.max_sample_age_seconds+" s；对象缺失即隐藏。":"本文件未附加完整源标注，当前只展示进入世界的对象；空白不能解释为没有行人、障碍物或信号。");
  if(!CONTEXT){$("contextObjects").disabled=true;$("signals").disabled=true;}
  applyMapPreset("interaction",false);$("mapPreset").onchange=e=>applyMapPreset(e.target.value);
  chooser.onchange=e=>{if(!e.target.value){selectActor(selected);return;}contextSelection=JSON.parse(e.target.value);$("actor").value="";updateSidebar();drawMap();drawChart();};
}
function setFrame(value) {
  frame=Math.max(0,Math.min(last,Math.round(value)));$("timeline").value=frame;
  text("playbackClock",`播放按墙钟 ${number(Number($("speed").value),1)}×；绘图繁忙可跳过显示帧，记录和物理步长保持不变。`);
  text("currentTime",number(S.times[frame])+" s");text("frameCaption",`状态 ${frame} / ${last} · t = ${number(S.times[frame])} s`);
  text("frameMeaning",SOURCE?"源日志编译轨迹 · 插值参考，不执行动作、不计算碰撞评分":(frame===last?"最终记录状态 · 不再执行动作":`本帧所有受控车辆共同决策，推进至 t = ${number(S.times[frame+1])} s`)+(Object.keys(S.metadata.observed_motion_inference||{}).length?" · 含源速度推演对象；缺标后不是日志真值":""));
  $("previous").disabled=frame===0;$("next").disabled=frame===last;
  updateSidebar();drawMap();drawChart();updatePermalink();
}
function pause(){playing=false;text("play","播放");}
function togglePlay(){if(last===0)return;if(playing){pause();return;}if(frame===last)setFrame(0);playing=true;lastClock=performance.now();accumulator=0;text("play","暂停");}
function selectActor(i){contextSelection=null;selected=i;$("actor").value=i;$("contextActor").value="";updateSidebar();drawMap();drawChart();updatePermalink();if($("selectedEvents").checked)renderEvents();}
function animate(now){if(playing){accumulator+=(now-lastClock)/1000*Number($("speed").value);const dt=RUN.metadata.dt_seconds;let next=frame;while(accumulator>=dt&&next<last){next++;accumulator-=dt;}if(next!==frame)setFrame(next);if(frame===last)pause();}lastClock=now;requestAnimationFrame(animate);}
function setupEvents(){
  const entries=[];
  for(const [branch,run]of [["主分支",RUN],["对比分支",OTHER]])if(run){
    ids.forEach((actor,i)=>{if(S.controlled[i]&&run.states[0][i][9])entries.push({t:0,actor,events:["initial_stop"],branch});});
    let previousContacts=new Set();
    for(let t=0;t<=last;t++){
      for(const [actor,events]of Object.entries(run.frames[t].arrivals))entries.push({t,actor,events,branch});
      if(t)ids.forEach((actor,i)=>{
        const continuity=run.lifecycle?.[t]?.continuity;
        if(run.presence[t][i]!==run.presence[t-1][i])entries.push({t,actor,events:[continuity?(run.presence[t][i]?"admitted":"departed"):(run.presence[t][i]?"entered":"exited")],branch});
        if(continuity?.status[i]==="pending"&&run.lifecycle?.[t-1]?.continuity?.status[i]!=="pending")entries.push({t,actor,events:["pending"],branch});
      });
      const current=new Set();
      for(const pair of run.lifecycle?.[t]?.contacts||[]){
        if(pair.length!==2||!worldIds.has(pair[0])||!worldIds.has(pair[1]))continue;
        const key=JSON.stringify([...pair].sort());current.add(key);
        if(!previousContacts.has(key))entries.push({t,actor:pair[0],partner:pair[1],events:["contact"],branch});
      }
      previousContacts=current;
    }
  }
  eventEntries=entries.sort((a,b)=>a.t-b.t);renderEvents();
}
function renderEvents(){
  $("events").replaceChildren();$("eventStrip").replaceChildren();
  const type=$("eventFilter").value, only=$("selectedEvents").checked;
  const entries=eventEntries.filter(e=>(!only||e.actor===ids[selected]||e.partner===ids[selected])&&(type==="all"||e.events.some(v=>type==="safety"?["collision","offroad","contact","road_boundary_contact"].includes(v):type==="lifecycle"?["entered","exited","admitted","departed","pending"].includes(v):v==="contact")));
  text("eventCount",`· ${entries.length} / ${eventEntries.length} 条`);
  if(!entries.length){const empty=document.createElement("div");empty.className="empty";empty.textContent="当前筛选没有事件。源轨迹没有驾驶碰撞评估；空列表不代表安全通过。";$("events").append(empty);}
  for(const entry of entries){const button=document.createElement("button");button.className="event-row";button.dataset.tick=entry.t;button.dataset.actor=entry.actor;
    const time=document.createElement("span");time.className="time";time.textContent=number(S.times[entry.t],1)+" s";
    const body=document.createElement("span");body.className="body";const label=document.createElement("span");label.textContent=entry.events.map(e=>EVENT[e]||e).join(" · ");
    const who=document.createElement("div");who.className="who";who.textContent=(OTHER?entry.branch+" · ":"")+actorName(ids.indexOf(entry.actor))+(entry.partner?" ↔ "+compact(entry.partner):"");
    body.append(label,who);button.append(time,body);button.onclick=()=>{pause();selectActor(ids.indexOf(entry.actor));setFrame(entry.t);};$("events").append(button);
    const marker=document.createElement("button");marker.className="event-dot "+(entry.events.some(e=>["collision","offroad","contact","road_boundary_contact"].includes(e))?"":entry.events.includes("goal")?"goal":"truncated");marker.style.left=(100*entry.t/Math.max(last,1))+"%";if(entry.branch==="对比分支")marker.style.top="5px";marker.title=`${entry.branch} · ${time.textContent} · ${label.textContent}`;marker.setAttribute("aria-label",marker.title);marker.onclick=button.onclick;$("eventStrip").append(marker);
  }
}
function updatePermalink(){
  const url=new URL(location.href);url.searchParams.set("actor",ids[selected]);url.searchParams.set("tick",String(frame));$("permalink").href=url.href;
}
const worldPlaceholder=document.createElement("option");worldPlaceholder.value="";worldPlaceholder.textContent="选择世界中的对象";worldPlaceholder.disabled=true;$("actor").append(worldPlaceholder);
ids.forEach((id,i)=>{const option=document.createElement("option");option.value=i;option.textContent=actorName(i)+(fleetCaption(i)?" · "+fleetCaption(i):S.controlled[i]?" · 受控":isReactive(i)?" · 反应式背景":" · 世界回放");$("actor").append(option);});$("actor").value=selected;
const fleetRoles = Object.values(S.metadata.vehicle_fleet?.roles||{});
text("subtitle",`${S.metadata.map_name} / ${S.metadata.scenario_id} · ${ids.length} 个世界对象 · ${RUN.metadata.controlled_actors??controlIndex.size} 辆策略动作车辆`+(fleetRoles.length?` · ${fleetRoles.filter(r=>r.mode!=="outside_window").length} 辆窗口内车辆均有控制分配（含停车 / 制动）`:""));
text("duration",number(S.times[last])+" s");$("timeline").max=last;
if(last===0){["timeline","play","speed","start"].forEach(id=>$(id).disabled=true);text("modeBadge","单帧诊断 / 初始化即停止");}
text("recordDetails",`${RUN.name} · ${RUN.metadata.dt_seconds} s / step · ${last} 次联合推进 · split: ${S.metadata.split}`);
const sampling=S.source_timing,intervals=sampling?.interval_seconds;
text("recordTiming",intervals?`源采样 ${sampling.sample_count} 帧 · 间隔中位数 ${number(intervals.median,6)} s（约 ${number(1/intervals.median,2)} Hz），范围 ${number(intervals.min,6)}–${number(intervals.max,6)} s。${SOURCE?"源轨迹插值":"物理记录"} ${number(1/RUN.metadata.dt_seconds,2)} Hz；插值不增加原始观测。`:"源采样间隔未提供；物理记录步长不能当作原始日志帧率。");
const controllerName=m=>typeof m.controller_configuration.controller==="string"?m.controller_configuration.controller:m.policy;
text("recordController",`主分支默认控制器：${controllerName(RUN.metadata)}；逐车实际来源以本帧对象信息为准，探针可覆盖控制。${OTHER?"对比分支默认："+controllerName(OTHER.metadata)+"；严格使用同一场景、时间轴和车辆身份。":"该视图仅展示已记录轨迹，不在浏览器内重新仿真。"}`);
text("recordMap",`地图边界模型：${S.metadata.road_edge_model||"记录未声明"}。路线仅展示记录中明确提供的任务；未知信息保持未知。`);
text("buildId",`主分支 build: ${RUN.metadata.backend_build||"未记录"}${OTHER?"；对比分支 build: "+(OTHER.metadata.backend_build||"未记录"):""}${OTHER&&RUN.metadata.backend_build!==OTHER.metadata.backend_build?"。动力学构建不同，轨迹差异不能只归因控制器。":""}`);
if(OTHER){["compareLegend","compareLayer","compareMetrics"].forEach(id=>$(id).classList.remove("hidden"));text("modeBadge","仿真双分支 · 不是原始日志");}
if(!DATA.graph){$("graph").disabled=true;$("graph").parentElement.title="本文件未附加车道图";}
$("eventFilter").onchange=renderEvents;$("selectedEvents").onchange=renderEvents;
$("focusEgo").disabled=!ids.includes("ego");
$("focusEgo").onclick=()=>{selectActor(ids.indexOf("ego"));camera.scale=5;$("follow").checked=true;drawMap();};
text("modeTitle",SOURCE?"源日志轨迹参考 · 未运行驾驶控制器":"控制器仿真记录 · 不是原始日志回放");
text("modeExplanation",SOURCE?"显示同一时间范围的源位置与速度（10 Hz 编译插值）。标注中断不等于物理离场；包围盒重叠也不能直接判定真实事故。":"日志用于初始化和路线参考，随后车辆按当前世界重新决策。页面播放保存的仿真结果；停车、拥堵和碰撞可能由控制器或接入问题造成，不能当作真实日志事件。");
if(!SOURCE&&!OTHER)text("modeBadge","已记录仿真 · 非源日志");
if(DATA.inspection){$("inspectionNote").classList.remove("hidden");text("inspectionText",DATA.inspection.note||"");if(!DATA.inspection.gallery)$("galleryHome").classList.add("hidden");if(DATA.inspection.kind==="driverl_native"){$("galleryHome").href="index.html";text("galleryHome","← DriveRL 原生验证");}}
if(SOURCE){text("modeBadge","源日志参考 · 非仿真 / 未评分");text("recordDetails",`${RUN.name} · 10 Hz 插值参考 · split: ${S.metadata.split}`);text("recordController","无控制器；所有对象均按源日志编译轨迹显示。不会把参考轨迹包装成物理执行。");text("buildId","源轨迹展示，不使用仿真构建或碰撞事件");}
$("actor").onchange=e=>{if(e.target.value!=="")selectActor(Number(e.target.value));};$("timeline").oninput=e=>{pause();setFrame(Number(e.target.value));};
$("start").onclick=()=>{pause();setFrame(0);};$("previous").onclick=()=>{pause();setFrame(frame-1);};$("next").onclick=()=>{pause();setFrame(frame+1);};$("play").onclick=togglePlay;
$("fitActors").onclick=fitActors;$("fitMap").onclick=()=>fit([...S.roads.flatMap(road=>road.points),...(CONTEXT?.map_features||[]).flatMap(feature=>geometryPoints(feature.geometry))],5);
["roads","graph","routes","trails","ids","reference","comparison","follow","worldObjects","contextObjects","vehicles","pedestrians","cyclists","otherObjects","signals"].forEach(id=>$(id).onchange=()=>{if(["roads","graph"].includes(id))$("mapPreset").value="custom";updateSidebar();drawMap();});
canvas.addEventListener("wheel",event=>{event.preventDefault();const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top,before=screenToWorld(x,y);camera.scale=Math.max(.2,Math.min(100,camera.scale*Math.exp(-event.deltaY*.001)));const after=screenToWorld(x,y);camera.x+=before[0]-after[0];camera.y+=before[1]-after[1];$("follow").checked=false;drawMap();},{passive:false});
canvas.addEventListener("pointerdown",event=>{pointer={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener("pointermove",event=>{if(!pointer)return;const dx=event.clientX-pointer.x,dy=event.clientY-pointer.y;if(Math.hypot(event.clientX-pointer.startX,event.clientY-pointer.startY)>4)pointer.moved=true;if(pointer.moved){camera.x-=dx/camera.scale;camera.y+=dy/camera.scale;$("follow").checked=false;drawMap();}pointer.x=event.clientX;pointer.y=event.clientY;});
canvas.addEventListener("pointerup",event=>{if(pointer&&!pointer.moved){const box=canvas.getBoundingClientRect(),p=screenToWorld(event.clientX-box.left,event.clientY-box.top);let chosen=null,distance=Infinity;function hit(pose,dims,value){const dx=p[0]-pose[0],dy=p[1]-pose[1],localX=dx*Math.cos(pose[3])+dy*Math.sin(pose[3]),localY=-dx*Math.sin(pose[3])+dy*Math.cos(pose[3]),extra=5/camera.scale;if(Math.abs(localX)<=dims[0]/2+extra&&Math.abs(localY)<=dims[1]/2+extra&&Math.hypot(dx,dy)<distance){chosen=value;distance=Math.hypot(dx,dy);}}RUN.states[frame].forEach((s,i)=>{if(worldVisible(i))hit(s,S.dimensions[i],{role:"world",index:i});});if($("contextObjects").checked)contextObjects().forEach(o=>{if(kindVisible(o.kind))hit(o.pose,o.dimensions,{role:"object",id:o.id});});if(chosen?.role==="world")selectActor(chosen.index);else if(chosen){contextSelection=chosen;$("actor").value="";$("contextActor").value=JSON.stringify(chosen);updateSidebar();drawMap();drawChart();}}pointer=null;});
canvas.addEventListener("pointercancel",()=>{pointer=null;});
chart.onclick=event=>{const box=chart.getBoundingClientRect();pause();setFrame(last?((event.clientX-box.left-35)/(box.width-70)*last):0);};
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","TEXTAREA","BUTTON"].includes(document.activeElement.tagName))return;if(event.code==="Space"){event.preventDefault();togglePlay();}if(event.code==="ArrowLeft"){event.preventDefault();pause();setFrame(frame-1);}if(event.code==="ArrowRight"){event.preventDefault();pause();setFrame(frame+1);}});
new ResizeObserver(()=>{mapSize=dimensions(canvas,ctx);drawMap();drawChart();}).observe(canvas.parentElement);
$("speed").onchange=()=>setFrame(frame);
mapSize=dimensions(canvas,ctx);setupContext();setupEvents();fitActors();
const params=new URLSearchParams(location.search),requestedActor=ids.indexOf(params.get("actor"));
if(requestedActor>=0)selectActor(requestedActor);
const requestedTick=Number(params.get("tick"));setFrame(Number.isFinite(requestedTick)?requestedTick:0);
requestAnimationFrame(animate);
</script>
</body>
</html>'''
