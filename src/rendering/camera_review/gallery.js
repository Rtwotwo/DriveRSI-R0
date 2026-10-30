"use strict";
const $=id=>document.getElementById(id),P=CalibratedProjection;
const q=new URLSearchParams(location.search),assetBase="../../camera-review/";
let scene,mapFeatures,layers=new Set(),index=0,cameraId=q.get("camera")||"CAM_F0",selected=null;
let inspector=null,calibratedFeatures=null,ready=null,version=0,playing=false,timer=null,downloads=null,wireDownloads=null,lastElapsed=0;
const chunks=new Map(),imageCache=new Map();
function trim(cache,n){while(cache.size>n)cache.delete(cache.keys().next().value);}
async function read(path,checksum,compressed=null){
 const packed=compressed&&typeof DecompressionStream!=="undefined",url=packed?compressed.path:path;
 const r=await fetch(url);if(!r.ok)throw Error("数据读取失败 "+r.status+": "+url);
 let b=await r.arrayBuffer();
 if(packed&&new Uint8Array(b)[0]===31&&new Uint8Array(b)[1]===139)b=await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
 if(checksum&&crypto.subtle){const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",b)),x=>x.toString(16).padStart(2,"0")).join("");if(hash!==checksum)throw Error("数据版本校验失败，请刷新页面");}
 return JSON.parse(new TextDecoder().decode(b));
}
function chunkPath(i,id){return scene.cameras[id].chunks[Math.floor(i/scene.chunk_size)];}
function getChunk(i,id){const entry=chunkPath(i,id);if(!chunks.has(entry.path)){const p=read(entry.path,entry.sha256,entry.gzip).catch(e=>{chunks.delete(entry.path);throw e;});chunks.set(entry.path,p);trim(chunks,6);}return chunks.get(entry.path);}
function image(path){if(!path)return Promise.resolve(null);if(!imageCache.has(path)){const p=new Promise((resolve,reject)=>{const i=new Image();i.decoding="async";i.onload=()=>resolve(i);i.onerror=()=>reject(Error("当前图像加载失败"));i.src=assetBase+path;}).catch(e=>{imageCache.delete(path);throw e;});imageCache.set(path,p);trim(imageCache,10);}return imageCache.get(path);}
function stop(){playing=false;clearTimeout(timer);$("play").textContent="播放";}
function message(canvas,text){const c=canvas.getContext("2d");c.fillStyle="#f7f9fb";c.fillRect(0,0,canvas.width,canvas.height);c.fillStyle="#567484";c.font="20px sans-serif";c.textAlign="center";c.fillText(text,canvas.width/2,canvas.height/2);c.textAlign="left";}
function white(canvas,cam,overlay=false){if(!cam){message(canvas,"虚拟机位没有原日志白膜");return;}(($("style").value==="wire")?WireWhitebox:SemanticWhitebox).draw(canvas,cam,{features:calibratedFeatures||mapFeatures,layers,selected,overlay});}
function geometry(g,fn){if(g.type==="Polygon")g.coordinates.forEach(x=>fn(x,true));else if(g.type==="MultiPolygon")g.coordinates.forEach(x=>x.forEach(y=>fn(y,true)));else if(g.type==="LineString")fn(g.coordinates,false);else if(g.type==="MultiLineString")g.coordinates.forEach(x=>fn(x,false));else if(g.type==="Point")fn([g.coordinates],false);else if(g.type==="MultiPoint")g.coordinates.forEach(p=>fn([p],false));else if(g.type==="GeometryCollection")g.geometries.forEach(x=>geometry(x,fn));}
function path(ctx,ps){ctx.beginPath();ps.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));}
function box(pose,size){const [x,y,,a]=pose,c=Math.cos(a),s=Math.sin(a);return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([u,v])=>[x+c*u*size[0]/2-s*v*size[1]/2,y+s*u*size[0]/2+c*v*size[1]/2]);}
function bev(canvas,f,cam,source,original){const ctx=canvas.getContext("2d"),owner=f.actors.find(a=>a.id===cam.owner_actor_id),center=owner.position,scale=5.1,project=p=>[canvas.width/2+(p[0]-center[0])*scale,canvas.height/2-(p[1]-center[1])*scale];ctx.fillStyle="#050b13";ctx.fillRect(0,0,canvas.width,canvas.height);
 for(const feat of mapFeatures){if(!layers.has(feat.layer))continue;geometry(feat.geometry,(ps,closed)=>{if(ps.length===1){const [x,y]=project(ps[0]);ctx.strokeStyle=feat.layer==="traffic_lights"?"#ffc65c":"#809dac";ctx.lineWidth=1.5;ctx.beginPath();if(feat.layer==="traffic_lights")ctx.rect(x-3,y-6,6,10);else{ctx.moveTo(x-3,y);ctx.lineTo(x+3,y);ctx.moveTo(x,y-3);ctx.lineTo(x,y+3);}ctx.stroke();return;}path(ctx,ps.map(project));ctx.strokeStyle="#71818d";ctx.lineWidth=1;if(closed){ctx.closePath();ctx.fillStyle=feat.layer==="walkways"?"#295346":"#374355";ctx.fill();}else ctx.stroke();});}
 const objects=original?source.objects:f.actors.filter(a=>a.present).map(a=>({id:a.id,pose:[...a.position,0,a.yaw],dimensions:a.dimensions,controlled:a.controlled,reconstructed:a.reconstructed}));
 for(const o of objects){path(ctx,box(o.pose,o.dimensions).map(project));ctx.closePath();ctx.fillStyle=o.id===selected?"#42e0c4":o.id===cam.owner_actor_id?"#4da7ff":original?"#c0cfde":!o.reconstructed?"#ed8ebf":o.controlled?"#eda54c":"#dbe4f0";ctx.fill();}
 const id=selected||cam.owner_actor_id,points=(original?scene.source_frames:scene.frames).map(f=>(original?f.objects:f.actors).find(a=>a.id===id&&(original||a.present))).map(a=>a?(original?a.pose:a.position):null);
 ctx.strokeStyle=original?"#ffd268":"#42e0c4";ctx.lineWidth=3;ctx.beginPath();let connected=false;for(const p of points){if(!p){connected=false;continue;}const v=project(p);connected?ctx.lineTo(...v):ctx.moveTo(...v);connected=true;}ctx.stroke();
}
function details(){if(!ready)return;const {f,c,source}=ready;$("diagnostics").textContent=JSON.stringify(ready.mode==="structure"?{tick:f.tick,type:"virtual_inspection_view_not_sensor",view:$("viewpoint").value,focus:inspector.focus,orbit:inspector.orbit,geometry:"flat ground for inspection; raw coordinates preserved",trainable:false}:{tick:f.tick,owner:c.owner_actor_id,camera:c.camera_id,K:c.K,camera_to_world:c.camera_to_world,distortion:c.distortion,quality:c.quality,trainable:false},null,2);
 if($("logDetails").open)$("log").textContent=JSON.stringify(source,null,2);
 if(!$("objectDetails").open)return;$("objects").replaceChildren();const src=new Map(source.objects.map(a=>[a.id,a])),world=new Map(f.actors.filter(a=>a.present).map(a=>[a.id,a]));for(const id of new Set([...src.keys(),...world.keys()])){const s=src.get(id),w=world.get(id),tr=document.createElement("tr");if(id===selected)tr.className="active";const vals=[id,s?.kind||c.objects.find(a=>a.id===id)?.kind||"—",s?s.pose.slice(0,2).map(x=>x.toFixed(1)).join(" / "):"未出现",w?w.position.map(x=>x.toFixed(1)).join(" / ")+" · "+w.speed.toFixed(1)+" m/s":"未进入/已离场",w?(w.controlled?"受控":"回放")+" / "+(w.reconstructed?"GS 实例":"缺少 GS"):"仅日志"];for(const v of vals){const td=document.createElement("td");td.textContent=v;tr.append(td);}tr.onclick=()=>{selected=id===selected?null:id;if(ready.mode==="structure"){inspector.focus=id;$("focusActor").value=id;}draw();};$("objects").append(tr);}}
function draw(){if(!ready)return;const {f,c,source,rgb,photo,mode}=ready;
 // Never draw images from a previous request after a tab/camera change.
 if(mode!==$("mode").value)return;
 const structural=mode==="structure",original=structural||mode==="bev"||!c.virtual;
 $("structureControls").hidden=!structural;$("cameraControl").hidden=structural;$("styleControl").hidden=structural;$("overlayControl").hidden=structural;
 $("sourcePanel").hidden=!original;$("sourceExtra").hidden=mode!=="quad"||c.virtual;$("branchExtra").hidden=mode!=="quad";$("stage").classList.toggle("single",!original);
 for(const id of ["sourceCanvas","branchCanvas","sourceWhitebox","branchWhitebox"]){if($(id).width!==c.width||$(id).height!==c.height){$(id).width=c.width;$(id).height=c.height;}}
 if(mode==="structure"){inspector.draw(f,source,layers);}
 else if(mode==="whitebox"){white($("branchCanvas"),c);if(!c.virtual)white($("sourceCanvas"),c.source_whitebox);}
 else if(mode==="bev"){bev($("branchCanvas"),f,c,source,false);bev($("sourceCanvas"),f,c,source,true);}
 else {if(rgb)$("branchCanvas").getContext("2d").drawImage(rgb,0,0,c.width,c.height);if(photo)$("sourceCanvas").getContext("2d").drawImage(photo,0,0,c.width,c.height);else message($("sourceCanvas"),"无同机位原始实拍");
 if($("overlay").checked){white($("branchCanvas"),c,true);if(photo&&c.source_whitebox)white($("sourceCanvas"),c.source_whitebox,true);}
 if(mode==="quad"){white($("branchWhitebox"),c);if(!c.virtual)white($("sourceWhitebox"),c.source_whitebox);}}
 $("sourceStamp").textContent="源样本 "+source.time_seconds.toFixed(3)+" s";$("branchStamp").textContent=f.time_seconds.toFixed(2)+" s · tick "+f.tick;
 $("clock").textContent=f.time_seconds.toFixed(2)+" / "+scene.frames.at(-1).time_seconds.toFixed(2)+" s";$("time").value=index;
 $("binding").textContent=c.camera_id+" · "+(c.virtual?"虚拟车辆 "+c.owner_actor_id.slice(0,6):"采集车 ego");
 $("limits").textContent=(c.virtual?"此机位无原日志实拍，仅展示新轨迹；BEV 仍可对照两条状态流。 ":"原图与原白膜采用前一真实日志帧，不插值伪造新观测。 ")+ "粉色表示缺少 GS 实例；线框是标注几何，RGB 中的模糊与残影仍按原记录保留。";
 if(structural){
 $("binding").textContent="结构化诊断视角 · 非 RGB 相机";
 $("limits").textContent="两侧共用观察机位；拖动旋转、Shift 拖动平移、滚轮缩放。完整对象按在场掩码展示；道路与对象贴地显示不表达坡道/立交。地图仅显示副本简化 ≤2 cm，完整原数据仍可下载。信号颜色为日志参考，非分支模拟灯态。";
 }else{inspector.summary(f,source);$("coverage").textContent="已加载 "+mapFeatures.length+" 个地图要素；当前为固定标定机位，切到「结构化三维白膜」可自由环绕。";}
 details();const url=new URL(location.href);url.searchParams.set("tick",f.tick);url.searchParams.set("camera",c.camera_id);url.searchParams.set("mode",mode);if(structural){url.searchParams.set("view",$("viewpoint").value);url.searchParams.set("actor",inspector.focus);}history.replaceState(null,"",url);
}
function link(file,label,base=assetBase){const a=document.createElement("a");a.href=base+file.path;a.download=file.path.split("/").at(-1);a.className="download";a.textContent=label+" · "+(file.bytes/1048576).toFixed(1)+" MB";return a;}
function videoLinks(){
 const vid=$("downloadCamera").value||cameraId,wire=$("downloadStyle").value==="wire",d=wire?wireDownloads:downloads,base=wire?"":assetBase;
 $("videoLinks").replaceChildren();$("videoArchive").replaceChildren();$("videoCamera").textContent=vid+" · "+(wire?"语义线框":"实体白膜");
 if(!d){$("videoStatus").textContent="正在读取所选白膜样式的视频清单…";return;}
 const names={source:"原日志白膜 MP4",branch:"新轨迹白膜 MP4",comparison:"新旧对比 MP4"};
 for(const v of d.videos.filter(v=>v.camera_id===vid))$("videoLinks").append(link(v,names[v.variant],base));
 $("videoArchive").append(link(d.archive,wire?"全部线框视频 ZIP":"全部实体白膜 ZIP",base));
 $("videoStatus").textContent=d.scene_duration_seconds+" 秒 · "+d.videos.length+" 个预生成视频 · "+d.output_fps+" fps 封装，保持源采样，不新增观测。视频采用固定标定机位和导出图层，不是当前自由环绕视角；虚拟相机仅提供新轨迹。";
}
async function render(){const mine=++version,i=index,id=cameraId,mode=$("mode").value,start=performance.now();ready=null;$("status").className="";$("status").textContent="读取当前机位…";$("stage").style.opacity=".45";$("stage").setAttribute("aria-busy","true");videoLinks();
 try{const f=scene.frames[i];let c;
 if(mode==="structure"){
   c={camera_id:id,owner_actor_id:scene.cameras[id].owner_actor_id,virtual:false,width:960,height:540,objects:[]};
 }else{
   const chunk=await getChunk(i,id);if(mine!==version)return;c=chunk.frames[i-chunk.start];if(!c||c.camera_id!==id||chunk.ticks[i-chunk.start]!==f.tick)throw Error("相机 / tick 绑定不一致");
   if(!calibratedFeatures){const m=await read(scene.map.path,scene.map.sha256,scene.map.gzip);if(mine!==version)return;calibratedFeatures=m.features;}
 }
 const needsRGB=mode==="rgb"||mode==="quad";const [rgb,photo]=needsRGB?await Promise.all([image(c.image),image(c.source)]):[null,null];if(mine!==version)return;
 ready={f,c,source:scene.source_frames[f.source_index],rgb,photo,mode};draw();lastElapsed=performance.now()-start;$("status").textContent="已对齐 tick "+f.tick+" · "+(mode==="structure"?$("viewpoint").selectedOptions[0].text:"当前机位 "+id)+" · "+(mode==="structure"?"结构化播放：无需相机数据或 RGB":needsRGB?"仅载入所选机位图片":"白膜模式未加载 RGB")+" · 本次准备 "+Math.round(lastElapsed)+" ms";$("stage").style.opacity="1";$("stage").setAttribute("aria-busy","false");
 // One bounded look-ahead chunk; no offscreen cameras or PNG preloading.
 if(mode!=="structure"&&i+scene.chunk_size<scene.frames.length)getChunk(i+scene.chunk_size,id).catch(()=>{});
 }catch(e){if(mine!==version)return;stop();ready=null;$("status").className="error";$("status").textContent=e.message;for(const id of ["sourceCanvas","branchCanvas","sourceWhitebox","branchWhitebox"])message($(id),"当前绑定不可用，未显示上一帧");$("stage").style.opacity="1";$("stage").setAttribute("aria-busy","false");}}
async function advance(){if(!playing)return;const myVersion=version;await render();if(!playing||version!==myVersion+1)return;if(index>=scene.frames.length-1){stop();return;}const dt=(scene.frames[index+1].time_seconds-scene.frames[index].time_seconds)*1000/Number($("speed").value);timer=setTimeout(()=>{if(playing){index++;advance();}},Math.max(0,dt-lastElapsed));}
function change(){stop();render();}
async function init(){const boot=await read("bootstrap.json");scene=await read(boot.scene.path,boot.scene.sha256,boot.scene.gzip);if(!scene.cameras[cameraId])cameraId=Object.keys(scene.cameras)[0];const tick=Number(q.get("tick")||0);index=Number.isFinite(tick)?scene.frames.findIndex(f=>f.tick>=tick):0;if(index<0)index=scene.frames.length-1;
 for(const [id,c]of Object.entries(scene.cameras)){const o=document.createElement("option");o.value=id;o.textContent=id+(c.virtual?" · 虚拟机位":" · ego");$("camera").append(o);$("downloadCamera").append(o.cloneNode(true));}$("camera").value=cameraId;$("downloadCamera").value=cameraId;$("time").max=scene.frames.length-1;
 if(["structure","whitebox","rgb","bev","quad"].includes(q.get("mode")))$("mode").value=q.get("mode");
 $("dataDownload").href=assetBase+"data.json";
 const ref=scene.display_map||scene.map,maps=await read(ref.path,ref.sha256,ref.gzip);mapFeatures=maps.features;
 inspector=new StructuredInspector(scene,mapFeatures,()=>{if($("mode").value==="structure")draw();});
 if(["front","chase","orbit","bird"].includes(q.get("view")))$("viewpoint").value=q.get("view");
 const focus=q.get("actor")||scene.cameras[cameraId].owner_actor_id;if(inspector.catalog.has(focus)){inspector.focus=focus;$("focusActor").value=focus;}
 for(const [name,info]of Object.entries(scene.map_layers)){const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.disabled=info.availability!=="available";input.checked=name!=="dubins_nodes"&&!input.disabled;if(input.checked)layers.add(name);input.onchange=()=>{input.checked?layers.add(name):layers.delete(name);draw();};label.append(input,document.createTextNode(name+" ("+(input.disabled?"未提供":info.feature_count)+")"));$("layers").append(label);}
 $("sourceMeta").textContent=scene.source_log+" · "+scene.source_frames.length+" 个源帧 / "+scene.frames.length+" 个分支采样时刻 · "+Object.keys(scene.cameras).length+" 路相机";
 for(const id of ["downloadCamera","downloadStyle"])$(id).onchange=videoLinks;
 $("camera").onchange=()=>{cameraId=$("camera").value;$("downloadCamera").value=cameraId;selected=null;change();};$("mode").onchange=change;
 for(const id of ["style","overlay"])$(id).onchange=()=>{$("downloadStyle").value=$("style").value;draw();videoLinks();};
 $("time").oninput=()=>{index=Number($("time").value);change();};
 for(const [id,delta]of [["prev",-1],["next",1]])$(id).onclick=()=>{index=Math.max(0,Math.min(scene.frames.length-1,index+delta));change();};
 $("reset").onclick=()=>{index=0;change();};
 $("play").onclick=()=>{if(playing){stop();return;}if(index===scene.frames.length-1)index=0;playing=true;$("play").textContent="暂停";advance();};
 $("objectDetails").ontoggle=details;$("logDetails").ontoggle=details;
 document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
 document.addEventListener("keydown",e=>{if(e.code==="Space"&&!["INPUT","SELECT","BUTTON"].includes(document.activeElement.tagName)){e.preventDefault();$("play").click();}});
 await render();
}
read(assetBase+"videos.json").then(d=>{downloads=d;videoLinks();}).catch(e=>{$("videoStatus").textContent="实体白膜视频暂不可用："+e.message;});
read("wireframe-videos.json").then(d=>{wireDownloads=d;videoLinks();}).catch(e=>{$("videoStatus").textContent="线框视频暂不可用："+e.message;});
init().catch(e=>{$("status").className="error";$("status").textContent=e.message;});
