'use strict';
const G=window.WhiteboxGeometry, $=id=>document.getElementById(id), canvas=$('view'), ctx=canvas.getContext('2d');
let manifest,map,streams,t=0,playing=false,last=0,selected='ego',hits=[],lastEgo=null;
let orbit={yaw:0,pitch:.55,distance:30,x:0,y:0},drag=null;
const colors={ego:'#007e92',vehicle:'#347ce5',pedestrian:'#e19a19',bicycle:'#9a58cc',cyclist:'#9a58cc'};
const surfaces={generic_drivable_areas:['#c5deee',0],road_segments:['#b9d5e9',1],lane_groups_polygons:['#b7d5e9',2],intersections:['#b5d4e9',3],lanes_polygons:['#acd0e8',4],gen_lane_connectors_scaled_width_polygons:['#acd0e8',4],walkways:['#e3e6e9',5],carpark_areas:['#d5e1e7',1],crosswalks:['#f0f3f5',6],stop_polygons:['#e6edf2',6]};
let polygons=[],lines=[],far=200;
function attachBounds(p){const pts=p.rings?p.rings.flat():p.points;let x=Infinity,y=Infinity,X=-Infinity,Y=-Infinity;for(const q of pts){x=Math.min(x,q[0]);y=Math.min(y,q[1]);X=Math.max(X,q[0]);Y=Math.max(Y,q[1]);}p.bounds=[x,y,X,Y];return p;}
function nearby(p,c){const [x,y,X,Y]=p.bounds,dx=Math.max(x-c.x,0,c.x-X),dy=Math.max(y-c.y,0,c.y-Y);return dx*dx+dy*dy<160000;}

function buildMap(){
  const laneBounds=new Set();
  for(const feature of map.features){if(feature.layer==='lanes_polygons')for(const key of ['left_boundary_fid','right_boundary_fid']){const id=feature.properties?.[key];if(id!==null&&id!==undefined)laneBounds.add(String(id));}}
  for(const feature of map.features){
    for(const primitive of G.primitives(feature.geometry)){
      if(primitive.type==='polygon'&&surfaces[feature.layer])polygons.push(attachBounds({...primitive,layer:feature.layer}));
      if(primitive.type==='line'&&['boundaries','baseline_paths'].includes(feature.layer))lines.push(attachBounds({...primitive,layer:feature.layer,laneBoundary:laneBounds.has(feature.id.split(':')[1])}));
    }
  }
  polygons.sort((a,b)=>surfaces[a.layer][1]-surfaces[b.layer][1]);
}
function camera(ego){
  const [x,y,,yaw]=ego.pose_xyz_yaw, mode=$('camera').value;
  if(mode==='front')return {x:x+ego.dimensions_lwh[0]*.27*Math.cos(yaw),y:y+ego.dimensions_lwh[0]*.27*Math.sin(yaw),z:1.65,yaw,pitch:.055};
  if(mode==='chase')return {x:x-12*Math.cos(yaw),y:y-12*Math.sin(yaw),z:7,yaw,pitch:.35};
  const a=yaw+orbit.yaw,d=orbit.distance,cp=Math.cos(orbit.pitch);
  return {x:x+orbit.x-d*cp*Math.cos(a),y:y+orbit.y-d*cp*Math.sin(a),z:1+d*Math.sin(orbit.pitch),yaw:a,pitch:orbit.pitch};
}
function frame(){const seq=streams[$('stream').value];return {...G.frameAt(seq.frames,t,seq.max_sample_age_seconds),seq};}
function line3(a,b,c,w,h,f,color,width=1,dash=[]){
  const cut=G.clipLine(G.cameraPoint(a,c),G.cameraPoint(b,c),.3,far);if(!cut)return;
  const p=G.project(cut[0],w,h,f),q=G.project(cut[1],w,h,f);
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([]);
}
function render(){
  if(!streams)return;
  const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(window.devicePixelRatio||1,2);
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#f7f9fb';ctx.fillRect(0,0,w,h);
  const data=frame(),objects=data.available?data.frame.objects:[],record=data.frame;
  const ego=record.objects.find(o=>o.actor_id==='ego')||lastEgo;
  if(!ego){$('hud').textContent='当前无自车位姿；不推测相机位置';return;}lastEgo=ego;
  const c=camera(ego),f=w*.65;far=G.safeFar(c,map.bounds_local,w,h,f);
  if($('roads').checked){
    for(const poly of polygons){
      if(!nearby(poly,c))continue;
      ctx.beginPath();let valid=false;
      for(const ring of poly.rings){
        const clipped=G.clipRing(ring.map(p=>G.cameraPoint([p[0],p[1],0],c)),.3,far);
        if(clipped.length<3)continue;valid=true;
        clipped.forEach((p,i)=>{const q=G.project(p,w,h,f);if(i===0)ctx.moveTo(...q);else ctx.lineTo(...q);});ctx.closePath();
      }
      if(valid){ctx.fillStyle=surfaces[poly.layer][0];ctx.fill('evenodd');}
    }
  }
  for(const ln of lines){
    if(!nearby(ln,c))continue;
    const base=ln.layer==='baseline_paths';if(!(base?$('baseline').checked:ln.laneBoundary?$('lines').checked:$('otherLines').checked))continue;
    for(let i=1;i<ln.points.length;i++)line3([...ln.points[i-1].slice(0,2),.015],[...ln.points[i].slice(0,2),.015],c,w,h,f,base?'#a67c28':ln.laneBoundary?'#ffffff':'#7898a9',base||!ln.laneBoundary?1:1.7,base||!ln.laneBoundary?[5,5]:[]);
  }
  hits=[];
  if($('boxes').checked){
    const sorted=[...objects].sort((a,b)=>G.cameraPoint(b.pose_xyz_yaw,c)[2]-G.cameraPoint(a.pose_xyz_yaw,c)[2]);
    for(const obj of sorted){
      if(obj.actor_id==='ego'&&$('camera').value==='front')continue;
      const selectedNow=obj.actor_id===selected,cs=G.corners(obj),color=selectedNow?'#e53b43':colors[obj.kind]||'#df654c';
      for(const [i,j] of G.edges)line3(cs[i],cs[j],c,w,h,f,color,selectedNow?2.8:1.65);
      const center=G.cameraPoint([obj.pose_xyz_yaw[0],obj.pose_xyz_yaw[1],obj.dimensions_lwh[2]+.3],c);
      if(center[2]>.3&&center[2]<far){const p=G.project(center,w,h,f);
        if(p[0]>0&&p[0]<w&&p[1]>0&&p[1]<h){ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText(obj.actor_id==='ego'?'ego':obj.actor_id.slice(0,6),p[0]+4,p[1]);hits.push({x:p[0],y:p[1],id:obj.actor_id});}
      }
    }
  }
  const status=data.available?'有效记录':data.ended?'该流已结束，未外推':data.index<0?'该流尚未开始':'采样间隔过大，隐藏对象';
  $('hud').textContent=`${$('stream').selectedOptions[0].text} · ${status}\n时间轴 ${t.toFixed(2)} s · 实际样本 ${record.time_seconds.toFixed(3)} s · 样本年龄 ${Math.max(0,data.age).toFixed(3)} s\n在场 ${objects.length} 个 · 接触记录 ${record.lifecycle?.contacts?.length??'未计算'} · ${$('camera').selectedOptions[0].text} · 视距 ${far.toFixed(0)} m · 虚拟诊断相机`;
  $('clock').textContent=`${t.toFixed(2)} / ${manifest.timeline_end_seconds.toFixed(2)} s`;$('time').value=t;
  const obj=objects.find(o=>o.actor_id===selected);
  $('detail').textContent=obj?`ID: ${obj.actor_id}\n类别: ${obj.kind} · 控制来源: ${typeof obj.control_source==='string'?obj.control_source:JSON.stringify(obj.control_source)}\n记录 xyz / yaw: ${obj.pose_xyz_yaw.map(x=>x.toFixed(3)).join(' / ')}\n尺寸 l / w / h: ${obj.dimensions_lwh.map(x=>x.toFixed(2)).join(' / ')} m\n真实时间戳: ${record.timestamp_us}\n显示地面 z=0；记录坐标未修改。`:`${selected} 在当前记录中不在场；状态：${record.lifecycle?.status_by_actor?.[selected]||'无当前标注'}；未从另一条流补入。`;
  window.whiteboxStatus={role:$('stream').value,time:t,recordTime:record.time_seconds,available:data.available,objectCount:objects.length,camera:$('camera').value,selected};
}
function actorOptions(){
  const ids=Object.keys(streams[$('stream').value].actors);
  $('actor').replaceChildren(...ids.map(id=>{const o=document.createElement('option');o.value=id;o.textContent=id;return o;}));
  if(!ids.includes(selected))selected='ego';$('actor').value=selected;
}
function animate(now){
  if(playing&&last){
    t=Math.min(manifest.timeline_end_seconds,t+(now-last)/1000*Number($('speed').value));
    render();
    if(t>=manifest.timeline_end_seconds){playing=false;$('play').textContent='播放';}
  }
  last=now;requestAnimationFrame(animate);
}
function setPlay(){if(t>=manifest.timeline_end_seconds)t=0;playing=!playing;$('play').textContent=playing?'暂停':'播放';render();}
async function read(name,hash){
  const r=await fetch(name);if(!r.ok)throw Error(`${name}: HTTP ${r.status}`);const bytes=await r.arrayBuffer();
  if(hash&&crypto.subtle){const sum=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');if(sum!==hash)throw Error(`${name}: 内容校验失败`);}
  return JSON.parse(new TextDecoder().decode(bytes));
}
async function init(){
  manifest=await read('manifest.json');
  const [m,s,b]=await Promise.all(['map.json','source-structure.json','interaction-structure.json'].map(n=>read(n,manifest.sha256?.[n])));
  map=m;streams={source_log:s,executed_branch:b};buildMap();actorOptions();
  $('time').max=manifest.timeline_end_seconds;
  const query=new URLSearchParams(location.search);
  if(['source_log','executed_branch'].includes(query.get('stream')))$('stream').value=query.get('stream');
  if(['front','chase','orbit'].includes(query.get('camera')))$('camera').value=query.get('camera');
  const requested=Number(query.get('t'));if(Number.isFinite(requested))t=Math.max(0,Math.min(manifest.timeline_end_seconds,requested));
  if(query.get('actor') in streams[$('stream').value].actors)selected=query.get('actor');
  actorOptions();
  for(const name of ['manifest.json','map.json','source-structure.json','interaction-structure.json']){const a=document.createElement('a');a.href=name;a.textContent=name;a.download=name;$('downloads').append(a,document.createElement('br'));}
  $('summary').textContent=`${manifest.source_family.toUpperCase()} · ${manifest.source_log}\n原日志 ${s.frames.length} 帧 / 交互 ${b.frames.length} 帧；地图 ${map.features.length} 个要素。来源划分：${s.source_split}。`;
  $('stream').onchange=()=>{actorOptions();render();};$('actor').onchange=()=>{selected=$('actor').value;render();};
  for(const id of ['camera','roads','lines','otherLines','baseline','boxes'])$(id).onchange=render;
  $('play').onclick=setPlay;$('reset').onclick=()=>{playing=false;$('play').textContent='播放';t=0;orbit={yaw:0,pitch:.55,distance:30,x:0,y:0};render();};
  $('time').oninput=()=>{t=Number($('time').value);render();};
  for(const [id,d] of [['prev',-1],['next',1]])$(id).onclick=()=>{const {seq,index}=frame();t=seq.frames[Math.max(0,Math.min(seq.frames.length-1,index+d))].time_seconds;render();};
  canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,moved:0};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag||$('camera').value!=='orbit')return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved+=Math.abs(dx)+Math.abs(dy);drag.x=e.clientX;drag.y=e.clientY;if(e.shiftKey){orbit.x-=dy*.12;orbit.y-=dx*.12;const radius=Math.hypot(orbit.x,orbit.y);if(radius>50){orbit.x*=50/radius;orbit.y*=50/radius;}}else{orbit.yaw-=dx*.006;orbit.pitch=Math.max(.08,Math.min(1.5,orbit.pitch+dy*.004));}render();};
  canvas.onpointerup=e=>{if(drag&&drag.moved<5){const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=hits.map(o=>({...o,d:Math.hypot(o.x-x,o.y-y)})).sort((a,b)=>a.d-b.d)[0];if(best&&best.d<45){selected=best.id;$('actor').value=selected;render();}}drag=null;};
  canvas.onwheel=e=>{if($('camera').value==='orbit'){e.preventDefault();orbit.distance=Math.max(5,Math.min(150,orbit.distance*Math.exp(e.deltaY*.001)));render();}};
  window.onresize=render;window.onkeydown=e=>{if(e.code==='Space'&&!['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName)){e.preventDefault();setPlay();}};
  render();requestAnimationFrame(animate);
}
init().catch(error=>{$('error').textContent=error.stack||String(error);$('hud').textContent='加载失败；不显示不一致的数据';});
