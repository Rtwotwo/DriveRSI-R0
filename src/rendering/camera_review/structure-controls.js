/* UI controller for structured worlds, independent of sensor frame downloads. */
(function(root){
const $=id=>document.getElementById(id),S=root.StructuralWhitebox;
const names={vehicle:"车辆",pedestrian:"行人",bicycle:"自行车",cyclist:"骑行者",generic_object:"其他物体",traffic_cone:"锥桶",barrier:"障碍物"};
class Inspector{
 constructor(scene,features,onChange){this.scene=scene;this.prepared=S.build(features);this.change=onChange;this.focus="ego";this.orbit={yaw:0,pitch:.5,distance:35,x:0,y:0};this.hits={};this.kinds=new Set(scene.source_frames.flatMap(f=>f.objects.map(o=>o.kind)));this.kinds.add("generic_object");this.raf=null;
 const catalog=new Map(scene.source_frames.flatMap(f=>f.objects.map(o=>[o.id,o.kind])));
 for(const f of scene.frames)for(const o of f.actors)if(!catalog.has(o.id))catalog.set(o.id,"generic_object");
 this.catalog=catalog;
 for(const [id,kind]of catalog){const o=document.createElement("option");o.value=id;o.textContent=(id==="ego"?"ego":id.slice(0,8))+" · "+(names[kind]||kind);$("focusActor").append(o);}$("focusActor").value="ego";
 for(const kind of this.kinds){const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.checked=true;input.setAttribute("aria-label","显示"+(names[kind]||kind));input.onchange=()=>{input.checked?this.kinds.add(kind):this.kinds.delete(kind);this.schedule();};label.append(input,document.createTextNode(names[kind]||kind));$("kinds").append(label);}
 $("viewpoint").onchange=()=>this.schedule();$("focusActor").onchange=()=>{this.focus=$("focusActor").value;this.schedule();};
 for(const id of ["showTrajectory","showSignals","sourceGhosts","showObjectLabels"])$(id).onchange=()=>this.schedule();
 for(const [id,amount]of [["rotateLeft",.25],["rotateRight",-.25]])$(id).onclick=()=>{$("viewpoint").value="orbit";this.orbit.yaw+=amount;this.schedule();};
 for(const [id,factor]of [["zoomIn",.8],["zoomOut",1.25]])$(id).onclick=()=>{$("viewpoint").value="orbit";this.orbit.distance=Math.max(5,Math.min(180,this.orbit.distance*factor));this.schedule();};
 $("resetView").onclick=()=>{this.orbit={yaw:0,pitch:.5,distance:35,x:0,y:0};$("viewpoint").value="chase";this.schedule();};
 for(const id of ["sourceCanvas","branchCanvas"]){const canvas=$(id);let drag=null;
 canvas.onpointerdown=e=>{if($("mode").value!=="structure")return;drag={x:e.clientX,y:e.clientY,moved:0};canvas.setPointerCapture(e.pointerId);};
 canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;drag.moved+=Math.abs(dx)+Math.abs(dy);if(drag.moved<3)return;$("viewpoint").value="orbit";
 if(e.shiftKey){this.orbit.x-=dy*.12;this.orbit.y-=dx*.12;this.orbit.x=Math.max(-100,Math.min(100,this.orbit.x));this.orbit.y=Math.max(-100,Math.min(100,this.orbit.y));}
 else{this.orbit.yaw-=dx*.006;this.orbit.pitch=Math.max(.08,Math.min(1.5,this.orbit.pitch+dy*.004));}this.schedule();};
 canvas.onpointerup=e=>{if(drag&&drag.moved<3){const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;const hit=(this.hits[id]||[]).map(h=>({...h,d:Math.hypot(h.x-x,h.y-y)})).sort((a,b)=>a.d-b.d)[0];if(hit&&hit.d<40){this.focus=hit.id;$("focusActor").value=hit.id;this.schedule();}}drag=null;};
 canvas.onpointercancel=()=>{drag=null;};
 canvas.addEventListener("wheel",e=>{if($("mode").value!=="structure")return;e.preventDefault();$("viewpoint").value="orbit";this.orbit.distance=Math.max(5,Math.min(180,this.orbit.distance*Math.exp(e.deltaY*.001)));this.schedule();},{passive:false});
 }
 }
 schedule(){if(this.raf!==null)return;this.raf=requestAnimationFrame(()=>{this.raf=null;this.change();});}
 summary(f,source){const objects=S.frameObjects(this.scene,f,source),srcKinds=S.countKinds(objects.source),worldKinds=S.countKinds(objects.world);
 const fmt=c=>Object.entries(c).map(([k,n])=>(names[k]||k)+" "+n).join(" / ");
 $("objectCoverage").textContent="原日志："+fmt(srcKinds)+"；当前世界："+fmt(worldKinds)+"。共 "+f.actors.length+" 个对象槽位，只有 present=true 的对象属于当前世界。";
 $("signalRecords").replaceChildren();const signals=source.signals;if(signals?.available){if(!signals.records.length)$("signalRecords").textContent="本日志帧无信号记录";for(const s of signals.records){const row=document.createElement("div");row.textContent="连接车道 "+s.connector_id+"："+(s.is_red?"红灯":"非红（并不等于确认绿灯）");row.style.color=s.is_red?"#bd3540":"#8a6513";$("signalRecords").append(row);}}else $("signalRecords").textContent="源日志未提供信号状态";
 const foot=document.createElement("small");foot.textContent="以上为源日志信号参考；不能当作分支信号灯的模拟输出。";$("signalRecords").append(foot);
 $("executionState").textContent="当前在场 "+objects.world.length+"；受控 "+objects.world.filter(o=>o.controlled).length+"；回放 "+objects.world.filter(o=>!o.controlled).length+"；碰撞标记 "+objects.world.filter(o=>o.collision).length+"；越界标记 "+objects.world.filter(o=>o.offroad).length+"。路线任务、路权裁定和建筑网格未包含在本展示记录中。";
 return objects;
 }
 draw(f,source,layers){const objects=this.summary(f,source),candidate=objects.world.find(o=>o.id===this.focus)||objects.source.find(o=>o.id===this.focus),anchor=candidate||objects.world.find(o=>o.id==="ego")||objects.source.find(o=>o.id==="ego");if(!anchor)throw Error("没有可用的观察中心");
 const mode=$("viewpoint").value,camera=S.camera(anchor,mode,this.orbit),tracked=this.focus,sourceIds=new Set(objects.world.map(o=>o.id));
 const shared={camera,hideActor:mode==="front"?anchor.id:null,sourceSignals:source.signals,layers,selected:this.focus,kinds:this.kinds,showLabels:$("showObjectLabels").checked,showTrajectory:$("showTrajectory").checked,showSignals:$("showSignals").checked};
 const sourceTrace=this.scene.source_frames.map(f=>f.objects.find(o=>o.id===tracked)?.pose||null),branchTrace=this.scene.frames.map(f=>f.actors.find(o=>o.id===tracked&&o.present)?.position||null);
 const original=S.render($("sourceCanvas"),this.prepared,{...shared,objects:objects.source,trajectory:sourceTrace});
 const branch=S.render($("branchCanvas"),this.prepared,{...shared,objects:objects.world,trajectory:branchTrace,ghosts:$("sourceGhosts").checked?objects.source.filter(o=>!sourceIds.has(o.id)):[]});
 this.hits.sourceCanvas=original.hits;this.hits.branchCanvas=branch.hits;
 $("coverage").textContent="已加载全部 "+this.prepared.count+" 个地图要素 · 原日志 "+objects.source.length+" / 世界 "+objects.world.length+" 个在场对象 · 类别筛选后 "+branch.filteredObjects+" 个 · "+$("viewpoint").selectedOptions[0].text+" · 跟踪 "+tracked+(candidate?"":"（本帧不在场，观察中心回退 ego）");
 return{original,branch};
 }
}
root.StructuredInspector=Inspector;
})(globalThis);