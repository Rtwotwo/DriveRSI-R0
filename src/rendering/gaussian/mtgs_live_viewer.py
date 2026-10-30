"""Inspect world-driven MTGS sessions without presenting rules as learned policies."""

import json
import math
from pathlib import Path, PurePosixPath
from urllib.parse import quote


def _payload(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def live_html(token: str) -> str:
    """Serve alongside the authenticated local ``/state`` and ``/control`` API."""
    if not isinstance(token, str) or not token or len(token) > 512:
        raise ValueError("Expected a nonempty bounded session token")
    return _HTML.replace("__CONFIG__", _payload({"mode": "live", "token": token}))


def _asset(root: Path, value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or value.startswith("/") or ".." in PurePosixPath(value).parts):
        raise ValueError("Session image must be a portable relative file")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Session image is missing or outside its artifact directory")
    return quote(value, safe="/")


def write_session_viewer(manifest_path: Path) -> Path:
    """Create a portable recorded-session viewer; it cannot control a world."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("kind") != "mtgs_world_session"
            or data.get("trainable") is not False or not isinstance(data.get("frames"), list)
            or not data["frames"] or data.get("status") != "completed"):
        raise ValueError("Expected a completed non-training MTGS world session")
    clocks, generations = {}, []
    for index, frame in enumerate(data["frames"]):
        if not isinstance(frame, dict):
            raise ValueError("Recorded frame must be an object")
        generation = frame.get("generation", 0)
        if type(generation) not in (str, int) or generation == "":
            raise ValueError("Recorded generation requires a nonempty string or integer identity")
        if generation not in clocks:
            clocks[generation] = (-1, -1.0)
            generations.append({"id": generation, "frame_indices": []})
        previous_tick, previous_time = clocks[generation]
        tick, time = frame.get("tick"), frame.get("time_seconds")
        if (type(tick) is not int or tick < 0 or tick <= previous_tick
                or type(time) not in (int, float) or not math.isfinite(time)
                or time < 0 or time <= previous_time):
            raise ValueError("Recorded frame ticks and times must strictly increase")
        clocks[generation] = tick, time
        next(group for group in generations if group["id"] == generation)["frame_indices"].append(index)
        cameras = frame.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            raise ValueError("Every recorded frame requires camera images")
        identities = set()
        for camera in cameras:
            if not isinstance(camera, dict):
                raise ValueError("Camera record must be an object")
            identity = camera.get("camera_id")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("Camera IDs must be unique within a frame")
            identities.add(identity)
            camera["url"] = _asset(root, camera.get("image"))
            for key in ("source", "pinhole", "valid", "surface", "alpha", "depth"):
                if camera.get(key) is not None:
                    camera[key + "_url"] = _asset(root, camera[key])
    data["viewer_generations"] = generations
    output = root / "viewer.html"
    html = _HTML.replace("__CONFIG__", _payload({"mode": "recorded", "manifest": data}))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(html)
    return output


_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · 写实多车交互检查</title><style>
:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#e7eff8;background:#09131d}
*{box-sizing:border-box}body{max-width:1720px;margin:auto;padding:24px}h1{font-size:29px;margin:8px 0 14px}h2{font-size:18px;margin:0 0 12px}p{line-height:1.65}a{color:#80c8ff}
.eyebrow,.muted{color:#9ab0c3}.notice{padding:14px 18px;background:#2b261c;border:1px solid #716139;border-radius:9px}.error{color:#ffc1ba;white-space:pre-wrap}.timing{padding:12px;background:#152c3d;border:1px solid #33566d;border-radius:8px}
.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:16px 0}button,select{padding:9px 13px;border:1px solid #496a80;background:#20394b;color:#f2f7fb;border-radius:7px;font:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}#brake{background:#512d28;border-color:#8d4b40}label{display:flex;align-items:center;gap:8px}input[type=range]{flex:1;min-width:180px}
.cameras{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{min-width:0;padding:16px;border:1px solid #2b465b;border-radius:10px;background:#122332}.camera-image{width:100%;display:block;object-fit:contain;min-height:120px;background:#060c11}.camera-links{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin:10px 0}.caption{font-size:13px;margin:8px 0;color:#a8bfd0}.layout{display:grid;grid-template-columns:minmax(300px,1fr) minmax(350px,1fr);gap:16px;margin-top:16px}#bev{width:100%;height:380px;background:#0a1722;border-radius:6px;display:block}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #2b4356;word-break:break-all}tr.active{background:#24435a}tr[data-id]{cursor:pointer}.table-scroll{max-height:270px;overflow:auto}.selection{padding-top:12px}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto}details{border:1px solid #2b465b;border-radius:8px;padding:15px;margin-top:16px}summary{cursor:pointer}#identity{overflow-wrap:anywhere}.recorded-only{display:none}body.recorded .recorded-only{display:flex}body.recorded .live-only{display:none}#connection{font-weight:600}#camera-selection{font-size:13px}#camera-selection label{display:inline-flex;margin-right:15px}#limits li{margin:8px 0;line-height:1.6}
@media(max-width:900px){body{padding:14px}.cameras,.layout{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><p class="eyebrow">DriveRSI / 世界驱动的场景重建</p>
<h1>写实多车交互检查</h1>
<p class="notice"><strong>多智能体交互检查 · trainable=false</strong><br>每辆车的控制来源与版本见逐车策略身份。规则策略使用结构化观测；外部策略输入若未记录，图像是否参与决策应视为未知。当前记录尚不能作为学习型视觉 Self-Play 的训练批次。</p>
<p id="trajectory-source" class="notice" hidden></p>
<p id="identity" class="muted"></p><p id="connection">连接中…</p>
<div class="toolbar"><button id="play">播放</button><button id="pause">暂停</button><button id="step">推进一步</button><button id="reset">重置</button><label>对象 <select id="actor"></select></label><button id="brake" class="live-only">对选中受控车施加制动</button></div>
<div class="toolbar recorded-only"><label>独立回合 <select id="generation"></select></label><button id="previous">上一帧</button><input id="scrub" type="range" min="0" step="1" value="0" aria-label="记录帧"><span id="frame-index"></span></div>
<p id="timing" class="timing"></p><p id="error" class="error" role="alert"></p>
<p class="muted">默认图像按源镜头标定与畸变重投影；不是裁剪或修图。针孔画布保留原镜头视域外区域，便于检查蓝边与伪影。采集照片是原日志参考，干预后不是真值。</p>
<div id="camera-selection"></div><section id="cameras" class="cameras"></section>
<section class="layout"><article class="card"><h2>共同世界 · BEV</h2><canvas id="bev" aria-label="共同世界对象位置"></canvas><p class="caption" id="map-status"></p><p class="caption">蓝色：受控对象；橙色：其他世界对象；白色：当前选中。无尺寸数据时，箭头仅表示对象中心与朝向。</p></article>
<article class="card"><h2>对象与控制状态</h2><div class="table-scroll"><table><thead><tr><th>对象</th><th>角色</th><th>速度 m/s</th><th>状态</th></tr></thead><tbody id="actors"></tbody></table></div><div class="selection"><strong id="selected-title">未选对象</strong><pre id="selected"></pre></div></article></section>
<details open><summary>对象覆盖与能力边界</summary><ul id="limits"></ul><pre id="coverage"></pre></details>
<details><summary>采样与渲染性能</summary><p class="muted">物理频率、相机频率与墙钟吞吐是不同指标；浏览器刷新频率不代表仿真或渲染达到实时。</p><pre id="performance"></pre></details>
<details><summary>逐车策略身份（角色不代表已训练）</summary><pre id="population"></pre></details>
<details open><summary>决策输入与合成采样</summary><pre id="sampling"></pre></details>
<details><summary>场景与会话身份</summary><pre id="provenance"></pre></details>
<script>
const CONFIG=__CONFIG__,$=id=>document.getElementById(id),LIVE=CONFIG.mode==='live';
const LIMIT_LABELS={visual_diagnostic_not_driving_teacher:'RGB已实际输入像素依赖诊断策略；该策略不具备已验证驾驶能力，样本不作为教师监督或RL批次。',external_policy_inputs_unrecorded:'外部策略输入与记忆未记录；无法确认其是否使用图像，不能据此声明视觉闭环或完整训练经验。',rule_driven_interaction_not_learned_visual_self_play:'规则控制器读取结构化世界；尚未开展学习型视觉 Self-Play。',reference_relative_display_not_registered_collision_geometry:'画面使用相对参考轨迹的位姿增量；重建坐标与碰撞几何的完整配准尚未验证。',reconstruction_only_nodes_are_not_physics_actors:'部分重建对象只有外观，没有对应的物理世界主体；可见不代表能参与碰撞或响应。',pedestrian_and_cyclist_appearance_unverified:'行人和骑行者的独立外观资产尚未验证，不能声称全对象覆盖。',novel_view_quality_unverified:'偏离原采集机位后的图像质量尚未验收。',source_training_camera_appearance:'当前采用参与重建训练的采集相机外观；不是持留新视角质量评测。'};
let state=null,selectedActor='',visibleCameras=null,cameraSignature='',pollTimer=null,playTimer=null,polling=false,commandBusy=false,activeFetch=null,activeControl=null,closed=false,controlEpoch=0,recordIndex=0,recordGeneration=0,recordPlaying=false,playDeadline=0;
if(!LIVE){document.body.classList.add('recorded');$('step').textContent='下一记录帧';$('reset').textContent='回到首帧';}
const number=(v,d=2)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'未提供';
function element(tag,value,cls){const e=document.createElement(tag);if(value!==undefined)e.textContent=value;if(cls)e.className=cls;return e;}
function safeURL(value){if(typeof value!=='string'||!value)return null;try{const url=new URL(value,location.href);if(!['http:','https:','file:'].includes(url.protocol))return null;if(LIVE&&url.origin!==location.origin)return null;return url.href;}catch{return null;}}
function busy(){const done=state?.status==='completed'||state?.done===true,failed=Boolean(state?.error)||state?.status==='failed';for(const id of ['play','step'])$(id).disabled=commandBusy||(LIVE?(done||failed||!state):false);$('pause').disabled=commandBusy;$('reset').disabled=commandBusy;const actor=(state?.actors??[]).find(a=>a.id===selectedActor);$('brake').disabled=commandBusy||done||failed||!actor?.controlled||actor.alive===false;}
function chooseActor(value){selectedActor=value;$('actor').value=value;renderActors();drawBEV();busy();}
function renderActors(){const actors=state?.actors??[];const rows=[];for(const a of actors){const row=document.createElement('tr');row.dataset.id=a.id;if(a.id===selectedActor)row.className='active';const flags=[a.present===false?'未在场':null,state.lifecycle?.contacts?.some(pair=>pair.includes(a.id))?'几何接触':null,a.alive===false?'已停止':null,a.collision?'碰撞':null,a.offroad?'越界':null].filter(Boolean);for(const value of [a.id,a.controlled?'受控':a.role==='reactive_background'?'反应式背景':'回放 / 其他',(a.present===false?'—':number(a.speed)),flags.join(' / ')||'有效'])row.append(element('td',value));row.onclick=()=>chooseActor(a.id);rows.push(row);}$('actors').replaceChildren(...rows);const a=actors.find(x=>x.id===selectedActor);$('selected-title').textContent=a?'对象 '+a.id:'未选对象';if(!a){$('selected').textContent='';return;}if(a.present===false){$('selected').textContent='未在场：当前没有物理实体或有效位置，不显示策略动作。';return;}if(!a.controlled){$('selected').textContent=(a.role==='reactive_background'?'反应式背景，按当前世界反馈；不属于学习主体。\n':'日志回放对象；无学习策略动作。\n')+JSON.stringify({present:a.present??true,position:a.position,speed:a.speed,motion_inference:a.motion_origin??null,background_forecast:state.lifecycle?.background_decisions?.find(d=>d.actor===a.id)??null,current_geometric_contacts:state.lifecycle?.contacts?.filter(pair=>pair.includes(a.id))??"未记录",termination_events:"背景对象不计算学习策略终止事件；接触见上方几何记录"},null,2);return;}const diagnostics=Array.isArray(state.diagnostics)?state.diagnostics.find(x=>x.actor===a.id):state.diagnostics?.[a.id];$('selected').textContent=JSON.stringify({position:a.position,speed:a.speed,alive:a.alive,policy_binding:state.behavior_policies?.[a.id]??'未记录',action_origin:state.transition?.action_origins?.[a.id]??'未记录',policy_action:state.policy_actions?.[a.id]??'未提供',executed_action:state.executed_actions?.[a.id]??'未提供',diagnostics:diagnostics??'未提供'},null,2);}
function heightBindingCard(camera){
 const block=element('div',undefined,'height-binding');
 const binding=camera.height_binding;
 if(!binding||typeof binding!=='object'||Array.isArray(binding)){
  block.append(element('p','高程支持信息未提供；不能据此判断车辆是否贴地。','caption'));
 }else{
  const mode=binding.mode==='reference-track'?'有限参考轨迹高程代理':binding.mode==='source-time'?'源时间高度（未按新位置修正）':'未识别的高程绑定方式';
  block.append(element('p',mode+' · 高程修正 '+number(typeof binding.correction_m==='number'?binding.correction_m*100:undefined,2)+' cm','caption'));
  if(binding.mode==='reference-track')block.append(element('p','这是有限轨迹代理，不是地形网格；不能证明真实横坡、物理配准或新视角质量。','caption'));
  if(binding.supported===false){
   const warning=element('p','高程查询不支持：保留源高度。原因：'+(typeof binding.reason==='string'&&binding.reason?binding.reason:'未提供')+'。不能作为高程有效或图像质量通过的依据。','notice error');
   warning.setAttribute('role','alert');block.append(warning);
  }else if(binding.supported===true){
   block.append(element('p',binding.mode==='reference-track'?'高程查询在代理支持范围内；这不是图像质量验收通过。':'已记录高程绑定状态；源时间高度不保证反事实车辆贴地。','caption'));
  }else block.append(element('p','高程查询结果未提供；不默认认定支持。','caption'));
  const detail=element('details');detail.append(element('summary','高程绑定详情'),element('pre',JSON.stringify(binding,null,2)));block.append(detail);
 }
 const sceneSupport=camera.scene_height_supported===false||state?.scene_height_supported===false?false:camera.scene_height_supported??state?.scene_height_supported;
 if(sceneSupport===false){
  const warning=element('p','世界车辆高程未全部支持：至少一个世界车辆的高程查询失败；即使当前相机主体查询成功，画面仍可能包含高度失配对象。','notice error');
  warning.setAttribute('role','alert');block.append(warning);
 }else if(sceneSupport!==true)block.append(element('p','其他世界车辆的高程支持状态未提供。','caption'));
 return block;
}
function surfaceCard(camera){
 const block=element('div',undefined,'surface-diagnostics'),report=camera.surface_diagnostics;
 if(!report||report.status==='unavailable'){
  block.append(element('p','透明度 / 深度诊断：未提供；旧记录不据此认定有空洞或已通过质量检查。','caption'));return block;
 }
 const percent=value=>typeof value==='number'?number(value*100,2)+'%':'未提供';
 if(report.coverage_status==='low_coverage_warning'){
  const warning=element('p','低覆盖警告：镜头有效区域内 '+percent(report.low_opacity_fraction)+' 的像素 α<0.5，超过 '+percent(report.thresholds?.low_coverage_warning_fraction)+' 工程告警阈值。该画面可能包含未充分重建的区域；这不是对真实场景的判断，也不代表其余像素质量通过。','notice error');warning.setAttribute('role','alert');block.append(warning);
 }
 block.append(element('p','表面诊断 · 低透明度 α<0.5：'+percent(report.low_opacity_fraction)+' · α<0.95：'+percent(report.below_opaque_fraction)+' · 近面<0.5m且α≥0.95：'+percent(report.opaque_near_surface_fraction)+' · 深色不透明：'+percent(report.black_opaque_fraction),'caption'));
 block.append(element('p','比例分母为镜头有效像素。黑色可能是真实深色物体或渲染伪影；期望深度不是物理首个碰撞面。这些统计不认证图像质量，原始 RGB 未修改。','caption'));
 const detail=element('details');detail.append(element('summary','透明度 / 期望深度统计与阈值'),element('pre',JSON.stringify(report,null,2)));block.append(detail);return block;
}
function cameraCard(camera){const capturedTick=camera.render_tick??state.render_tick??state.tick;const c=element('article',undefined,'card');c.append(element('h2',camera.camera_id));if(camera.virtual){c.append(element('p','虚拟车顶视角 · 不是该车的实拍标定 · 新视角质量未认证','caption'));c.append(element('p','缺少采集车 ego 的重建外观；行人资产未完整覆盖，不能作为合格训练观测','error'));}const owner=camera.owner_actor_id??camera.actor_id??'ego';c.append(element('p','相机所属主体：'+owner+(owner==='ego'?' · 原采集车机位；不同相机不代表第二辆策略车的视角':''),'caption'));c.append(heightBindingCard(camera));c.append(surfaceCard(camera));const photo=camera.photometric;if(photo?.render_domain==='checkpoint_exposure_aligned')c.append(element('p','渲染沿用重建训练的曝光归一化（系数 '+number(photo.v_adjust_factor,3)+'）；采集照片保留原曝光，二者不能直接作像素误差比较。','caption'));else if(!photo||photo.render_domain==='unknown')c.append(element('p','曝光域未记录；采集照片仅供定性参考。','caption'));const quality=camera.quality;if(quality){c.append(element('p','质量门禁：'+(quality.diagnostic_usable?'元数据可供诊断':'元数据不足或不一致')+' · 高质量监督：'+(quality.high_quality_supervision_eligible?'通过':'未通过'),'caption'));c.append(element('p','未满足项：'+(quality.reasons??[]).join(' / '),'caption'));}else c.append(element('p','质量门禁：未提供；镜头有效像素不代表重建质量','caption'));const valid=camera.source_lens_valid_fraction;if(typeof valid==='number')c.append(element('p','源镜头重投影有效像素：'+number(valid*100,2)+'% · 该比例不衡量新视角重建质量','caption'));if(camera.source_seed_error_us!==undefined)c.append(element('p','重建参考时刻偏差：'+number(camera.source_seed_error_us/1000,3)+' ms · 采集照片仅供对照','caption'));const image=document.createElement('img');image.className='camera-image';image.alt=camera.camera_id+' 世界渲染';const status=element('p','加载中…','caption');image.onload=()=>{status.textContent='渲染 tick '+capturedTick+' · '+image.naturalWidth+' × '+image.naturalHeight;};image.onerror=()=>{status.textContent='图像加载失败，不能作为有效相机观测';status.className='error';};const path=safeURL(camera.url);if(path)image.src=path;else status.textContent='未提供有效相机图像';c.append(image,status);const links=element('div',undefined,'camera-links');for(const [label,key] of [['查看渲染图','url'],['采集照片（参考）','source_url'],['针孔画布（含原镜头视域外区域）','pinhole_url'],['重投影有效区域掩码','valid_url'],['透明度 α（诊断）','alpha_url'],['期望深度（诊断，0–80m对数灰度）','depth_url'],['无损表面通道 NPZ','surface_url']]){const href=safeURL(camera[key]??(LIVE&&['valid_url','alpha_url','depth_url','surface_url'].includes(key)&&camera[key.slice(0,-4)]?'/artifacts/'+camera[key.slice(0,-4)]:null));if(href){const a=element('a',label);a.href=href;a.target='_blank';a.rel='noopener';links.append(a);}else if(key==='source_url')links.append(element('span','采集照片未提供','muted'));}c.append(links);return c;}
function renderCameras(){const cameras=state?.cameras??[];if(visibleCameras===null){const preferred=['CAM_F0','CAM_R0'].filter(id=>cameras.some(c=>c.camera_id===id));const virtual=cameras.find(c=>c.virtual&&c.owner_actor_id!=='ego');visibleCameras=new Set(virtual?[preferred[0]??cameras[0].camera_id,virtual.camera_id]:(preferred.length?preferred:cameras.slice(0,2).map(c=>c.camera_id)));}const signature=JSON.stringify({images:cameras.map(c=>[c.camera_id,c.url,c.source_url,c.pinhole_url,c.render_tick,c.height_binding,c.scene_height_supported,c.quality,c.surface_diagnostics,c.photometric]),sceneHeightSupport:state.scene_height_supported,shown:Array.from(visibleCameras),tick:state.render_tick??state.tick});if(signature===cameraSignature)return;cameraSignature=signature;const choices=[];for(const c of cameras){const label=element('label');const input=document.createElement('input');input.type='checkbox';input.checked=visibleCameras.has(c.camera_id);input.onchange=()=>{input.checked?visibleCameras.add(c.camera_id):visibleCameras.delete(c.camera_id);renderCameras();};label.append(input,document.createTextNode(c.camera_id));choices.push(label);}$('camera-selection').replaceChildren(...choices);$('cameras').replaceChildren(...cameras.filter(c=>visibleCameras.has(c.camera_id)).map(cameraCard));if(!cameras.length)$('cameras').append(element('p','当前时刻未提供相机渲染。','error'));}
function drawBEV(){const canvas=$('bev'),ratio=window.devicePixelRatio||1,width=Math.max(canvas.clientWidth,100),height=380;canvas.width=width*ratio;canvas.height=height*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);const map=state?.map,actors=(state?.actors??[]).filter(a=>a.present!==false),roads=Array.isArray(map?.roads)?map.roads:[],routes=Array.isArray(map?.routes)?map.routes:[];const valid=p=>Array.isArray(p)&&Number.isFinite(p[0])&&Number.isFinite(p[1]);const points=[...roads.flat(),...routes.flatMap(r=>r.points??[]),...actors.map(a=>a.position)].filter(valid);$('map-status').textContent=roads.length?'地图线与对象位于同一世界坐标。':'地图几何未提供；当前仅显示世界对象，不能据此判断道路覆盖。';if(!points.length){ctx.fillStyle='#a8bfd0';ctx.fillText('没有有效世界坐标',15,30);return;}let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;for(const p of points){x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1]);}const scale=Math.min((width-40)/Math.max(20,x1-x0),(height-40)/Math.max(20,y1-y0)),cx=(x0+x1)/2,cy=(y0+y1)/2;const xy=p=>[width/2+(p[0]-cx)*scale,height/2-(p[1]-cy)*scale];function line(points,color){ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.beginPath();let begun=false;for(const p of points??[]){if(!valid(p))continue;const [x,y]=xy(p);begun?ctx.lineTo(x,y):ctx.moveTo(x,y);begun=true;}ctx.stroke();}for(const road of roads)line(road,'#4d687a');for(const route of routes)line(route.points,route.actor_id===selectedActor?'#d5edff':'#2e83b0');for(const a of actors){if(!valid(a.position))continue;const [x,y]=xy(a.position);ctx.save();ctx.translate(x,y);ctx.rotate(-(Number.isFinite(a.yaw)?a.yaw:0));ctx.fillStyle=a.controlled?'#54b7ff':'#e9ad54';ctx.strokeStyle=a.id===selectedActor?'white':'#122332';ctx.lineWidth=a.id===selectedActor?2:1;const dimensions=a.dimensions;if(Array.isArray(dimensions)&&dimensions[0]>0&&dimensions[1]>0){ctx.beginPath();ctx.rect(-dimensions[0]*scale/2,-dimensions[1]*scale/2,dimensions[0]*scale,dimensions[1]*scale);}else{ctx.beginPath();ctx.moveTo(7,0);ctx.lineTo(-5,-4);ctx.lineTo(-3,0);ctx.lineTo(-5,4);ctx.closePath();}ctx.fill();ctx.stroke();ctx.restore();}}
function apply(next){if(!next||typeof next!=='object')throw new Error('无效状态响应');if(LIVE&&state&&typeof next.generation==='number'&&typeof state.generation==='number'&&next.generation<state.generation)return;if(LIVE&&state&&next.generation===state.generation&&typeof next.sequence==='number'&&typeof state.sequence==='number'&&next.sequence<state.sequence)return;state=next;const source=state.challenge_replay;$('trajectory-source').hidden=!source;if(source){$('trajectory-source').textContent='已记录联合动作复现 · '+source.candidate_id+' · '+(source.diagnostic_failure?'机动未通过验收：仅诊断渲染':source.maneuver_validation_pass===true?'机动通过有界工程检查；仍无教师/图像质量资格':'原始轨迹检查状态见来源报告')+'。此处 RGB 是输出，不用于重新决策。';}$('sampling').textContent=JSON.stringify({mode:state.sampling?.decision_mode??'structured',configuration:state.sampling??null,decision:state.transition?.policy_decision??state.latest_transition?.policy_decision??null},null,2);$('identity').textContent='来源日志：'+(state.source_log??'未提供')+' · 场景：'+(state.scenario_id??'未提供');$('connection').textContent=LIVE?'在线世界推进与渲染 · '+(state.status??'未知')+' · '+(state.playing?'正在推进':'暂停'):'离线记录回放 · '+(recordPlaying?'播放中':'暂停')+' · 不连接模拟世界';const renderTick=state.render_tick??state.tick,worldTick=state.world_tick??state.tick;const dt=state.dt??state.dt_seconds;const stride=state.sensor_stride??CONFIG.manifest?.sensor_stride;const cameraHz=state.render_hz??(stride?1/(dt*stride):undefined);$('timing').textContent='最新世界 tick '+worldTick+' · 图像与 BEV tick '+renderTick+' · 渲染快照时间 '+number(state.time_seconds,3)+' s · 世界时间 '+number(state.world_time_seconds??worldTick*dt,3)+' s · 物理 '+number(1/dt,1)+' Hz · 相机 '+number(cameraHz,1)+' Hz'+(renderTick!==worldTick||(state.world_generation!==undefined&&state.world_generation!==state.generation)?' · 注意：图像与当前世界不在同一快照':'');const failure=state.error??state.failure;$('error').textContent=failure?typeof failure==='string'?failure:JSON.stringify(failure):'';const actors=state.actors??[];if(!actors.some(a=>a.id===selectedActor))selectedActor=actors.find(a=>a.controlled)?.id??actors[0]?.id??'';$('actor').replaceChildren(...actors.map(a=>{const o=element('option',a.id+(a.controlled?' · 受控':a.role==='reactive_background'?' · 反应式背景':' · 回放 / 其他'));o.value=a.id;return o;}));$('actor').value=selectedActor;renderActors();renderCameras();drawBEV();$('limits').replaceChildren(...(state.limits??[]).map(item=>element('li',typeof item==='string'?(item==='reconstruction_only_nodes_are_not_physics_actors'&&state.coverage?.lifecycle_binding==='world_presence_authoritative'?'未绑定当前世界的独立重建节点已排除；静态背景中的残留外观仍未验证。':LIMIT_LABELS[item]??item):JSON.stringify(item))));$('coverage').textContent=JSON.stringify(state.coverage??'对象覆盖信息未提供',null,2);$('population').textContent=JSON.stringify(state.behavior_policies??'策略群体未记录',null,2);$('performance').textContent=JSON.stringify(state.performance??'尚无采样性能数据',null,2);$('provenance').textContent=JSON.stringify({source_log:state.source_log,scenario_id:state.scenario_id,render_generation:state.generation,world_generation:state.world_generation??state.generation,sequence:state.sequence,controlled_ids:state.controlled_ids,trainable:false},null,2);busy();}
function schedulePoll(delay=200){clearTimeout(pollTimer);if(!closed&&LIVE)pollTimer=setTimeout(poll,delay);}
async function poll(){if(closed||!LIVE)return;if(polling||commandBusy){schedulePoll();return;}polling=true;const epoch=controlEpoch,abort=new AbortController();activeFetch=abort;try{const response=await fetch('/state',{cache:'no-store',headers:{'X-DriveRSI-Token':CONFIG.token},signal:abort.signal});if(!response.ok)throw new Error('状态请求 HTTP '+response.status);const data=await response.json();if(!closed&&epoch===controlEpoch)apply(data);}catch(error){if(error.name!=='AbortError'){$('connection').textContent='连接中断';$('error').textContent=error.message;}}finally{polling=false;if(activeFetch===abort)activeFetch=null;schedulePoll();}}
async function control(operation){if(!LIVE||commandBusy||closed)return;commandBusy=true;controlEpoch++;activeFetch?.abort();const abort=new AbortController();activeControl=abort;busy();try{const payload={operation,expected_generation:state?.world_generation??state?.generation};if(operation==='brake')payload.actor_id=selectedActor;const response=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json','X-DriveRSI-Token':CONFIG.token},body:JSON.stringify(payload),signal:abort.signal});const data=await response.json();if(!response.ok)throw new Error('控制请求 HTTP '+response.status+' · '+(data.error??'未提供原因'));if(data.error)throw new Error(typeof data.error==='string'?data.error:JSON.stringify(data.error));if(Array.isArray(data.actors))apply(data);}catch(error){$('error').textContent=error.message;}finally{if(activeControl===abort)activeControl=null;commandBusy=false;busy();schedulePoll(0);}}
function recordedFrames(){const manifest=CONFIG.manifest;return manifest.viewer_generations[recordGeneration].frame_indices.map(i=>manifest.frames[i]);}
function recordedFrame(){const manifest=CONFIG.manifest,frames=recordedFrames(),group=manifest.viewer_generations[recordGeneration],frame=frames[recordIndex];apply({...manifest,...frame,frames:undefined,playing:recordPlaying,sequence:recordIndex,generation:group.id,render_tick:frame.render_tick??frame.tick});$('scrub').max=frames.length-1;$('scrub').value=recordIndex;$('frame-index').textContent=(recordIndex+1)+' / '+frames.length;$('previous').disabled=recordIndex===0;$('step').disabled=recordIndex===frames.length-1;$('play').disabled=recordIndex===frames.length-1;}
function stopRecorded(){recordPlaying=false;clearTimeout(playTimer);if(!LIVE&&state)recordedFrame();}
function advanceRecorded(){if(!recordPlaying||closed)return;const frames=recordedFrames();if(recordIndex>=frames.length-1){stopRecorded();return;}playDeadline+=(frames[recordIndex+1].time_seconds-frames[recordIndex].time_seconds)*1000;const delay=Math.max(0,playDeadline-performance.now());playTimer=setTimeout(()=>{recordIndex++;recordedFrame();advanceRecorded();},delay);}
$('play').onclick=()=>{if(LIVE)control('play');else if(!recordPlaying){recordPlaying=true;playDeadline=performance.now();recordedFrame();advanceRecorded();}};$('pause').onclick=()=>LIVE?control('pause'):stopRecorded();$('step').onclick=()=>{if(LIVE)control('step');else{stopRecorded();recordIndex=Math.min(recordedFrames().length-1,recordIndex+1);recordedFrame();}};$('reset').onclick=()=>{if(LIVE)control('reset');else{stopRecorded();recordIndex=0;recordedFrame();}};$('brake').onclick=()=>control('brake');$('actor').onchange=()=>chooseActor($('actor').value);$('previous').onclick=()=>{stopRecorded();recordIndex=Math.max(0,recordIndex-1);recordedFrame();};$('scrub').oninput=()=>{const requestedIndex=Number($('scrub').value);stopRecorded();recordIndex=requestedIndex;recordedFrame();};
window.addEventListener('resize',drawBEV);window.addEventListener('pagehide',()=>{closed=true;clearTimeout(pollTimer);clearTimeout(playTimer);activeFetch?.abort();activeControl?.abort();});if(LIVE){busy();poll();}else{const groups=CONFIG.manifest.viewer_generations;$('generation').replaceChildren(...groups.map((g,i)=>{const frames=g.frame_indices.map(j=>CONFIG.manifest.frames[j]);const o=element('option','回合 '+g.id+' · '+frames.length+' 帧 · '+number(frames[0].time_seconds,1)+'–'+number(frames[frames.length-1].time_seconds,1)+' s');o.value=i;return o;}));$('generation').onchange=()=>{stopRecorded();recordGeneration=Number($('generation').value);recordIndex=0;recordedFrame();};const query=new URLSearchParams(location.search);let linkError='';if(query.has('generation')){const requested=groups.findIndex(g=>String(g.id)===query.get('generation'));if(requested<0)linkError='候选链接的回合不存在';else recordGeneration=requested;}$('generation').value=recordGeneration;if(query.has('tick')){const raw=query.get('tick'),tick=Number(raw),frames=recordedFrames();if(!/^\d+$/.test(raw)||!Number.isSafeInteger(tick)||tick<frames[0].tick||tick>frames.at(-1).tick)linkError='候选链接的tick不在记录范围';else{recordIndex=0;for(let i=0;i<frames.length;i++)if(frames[i].tick<=tick)recordIndex=i;}}recordedFrame();if(linkError)$('error').textContent=linkError;}
</script></body></html>'''
