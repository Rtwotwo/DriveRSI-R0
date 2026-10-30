"""Inspect source camera images with their source clock, never as simulator observations."""

import json
import math
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    return value


def _vector(value, length: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Invalid {name}")
    for item in value:
        _number(item, name)


def _identifier(value, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {name}")


def _asset(value, root: Path) -> None:
    if (not isinstance(value, str) or not value or "\\" in value or "%" in value
            or "?" in value or "#" in value or ":" in value):
        raise ValueError("Images must use local relative paths")
    path = PurePosixPath(value)
    url = urlsplit(value)
    if (path.is_absolute() or ".." in path.parts or url.scheme or url.netloc
            or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}):
        raise ValueError("Images must use local relative paths")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Image is missing or escapes gallery directory")


def _geometry(value) -> None:
    if not isinstance(value, dict) or value.get("type") not in {
        "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon",
        "GeometryCollection",
    }:
        raise ValueError("Unsupported map geometry")
    if value["type"] == "GeometryCollection":
        if not isinstance(value.get("geometries"), list):
            raise ValueError("GeometryCollection needs geometries")
        for geometry in value["geometries"]:
            _geometry(geometry)
        return
    depth = {"Point": 0, "MultiPoint": 1, "LineString": 1,
             "MultiLineString": 2, "Polygon": 2, "MultiPolygon": 3}[value["type"]]

    def check(coords, level):
        if not isinstance(coords, list) or not coords:
            raise ValueError("Empty map coordinates")
        if level:
            for nested in coords:
                check(nested, level - 1)
        else:
            if len(coords) not in {2, 3}:
                raise ValueError("Invalid map coordinate")
            for item in coords:
                _number(item, "map coordinate")

    check(value.get("coordinates"), depth)


def _layer_label(state, feature_count: int) -> str:
    """Read context's availability without mistaking missing layers for empty ones."""
    if isinstance(state, dict):
        availability = state.get("availability", state.get("status", state.get("available")))
    else:
        availability = state
    if availability in (None, False, "unavailable", "not_provided", "missing"):
        return "未提供"
    if availability in ("loaded_empty", "empty"):
        return "已加载但本区域为空"
    if availability in (True, "available", "provided", "loaded"):
        return f"已提供 · {feature_count} 个要素"
    return "来源状态未识别"


def _validate(bundle: dict, root: Path) -> dict:
    # Round-trip detaches caller-owned data and rejects non-JSON/NaN values before embedding.
    try:
        data = json.loads(json.dumps(bundle, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Gallery data must be finite JSON") from exc
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("kind") != "source_camera_gallery_not_simulation"
            or not isinstance(data.get("scenes"), list) or not data["scenes"]):
        raise ValueError("Unsupported source gallery schema")
    scene_ids = set()
    for scene in data["scenes"]:
        if not isinstance(scene, dict):
            raise ValueError("Invalid scene")
        for key in ("id", "label", "log_name", "map_name"):
            _identifier(scene.get(key), key)
        if scene["id"] in scene_ids:
            raise ValueError("Duplicate scene identity")
        scene_ids.add(scene["id"])
        if _number(scene.get("source_rate_hz"), "source rate") <= 0:
            raise ValueError("Source rate must be positive")
        frames = scene.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Scene needs source frames")
        previous_timestamp = previous_index = -1
        origin = frames[0].get("timestamp_us") if isinstance(frames[0], dict) else None
        camera_ids = None
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Invalid source frame")
            timestamp, index = frame.get("timestamp_us"), frame.get("source_index")
            if (type(timestamp) is not int or type(index) is not int or timestamp < 0 or index < 0
                    or timestamp <= previous_timestamp or index <= previous_index):
                raise ValueError("Source timestamps and indices must strictly increase")
            _number(frame.get("time_seconds"), "source time")
            if abs(frame["time_seconds"] - (timestamp - origin) / 1e6) > 1e-6:
                raise ValueError("Source time must follow timestamp_us, starting at zero")
            previous_timestamp, previous_index = timestamp, index
            _vector(frame.get("ego_pose"), 3, "ego pose")
            objects = frame.get("objects")
            if not isinstance(objects, list):
                raise ValueError("Missing source object list")
            object_ids = set()
            for item in objects:
                if not isinstance(item, dict):
                    raise ValueError("Invalid source object")
                for key in ("id", "kind"):
                    _identifier(item.get(key), f"object {key}")
                if item["id"] in object_ids:
                    raise ValueError("Duplicate source object identity")
                object_ids.add(item["id"])
                _vector(item.get("pose"), 4, "object pose")
                _vector(item.get("dimensions"), 3, "object dimensions")
                if any(size <= 0 for size in item["dimensions"]):
                    raise ValueError("Object dimensions must be positive")
            cameras = frame.get("cameras")
            if not isinstance(cameras, list) or not cameras:
                raise ValueError("Frame needs camera slots, including missing images")
            ids = []
            for camera in cameras:
                if not isinstance(camera, dict):
                    raise ValueError("Invalid camera")
                _identifier(camera.get("id"), "camera id")
                ids.append(camera["id"])
                if type(camera.get("available")) is not bool:
                    raise ValueError("Camera availability must be explicit")
                if not isinstance(camera.get("note", ""), str):
                    raise ValueError("Invalid camera note")
                if camera["available"]:
                    _asset(camera.get("image"), root)
                    if any(type(camera.get(k)) is not int or camera[k] <= 0
                           for k in ("width", "height")):
                        raise ValueError("Available image needs native dimensions")
                elif camera.get("image") is not None:
                    raise ValueError("Unavailable images must be null")
                projections = camera.get("projection", [])
                if not isinstance(projections, list):
                    raise ValueError("Invalid projections")
                for box in projections:
                    if not isinstance(box, dict) or not isinstance(box.get("segments"), list):
                        raise ValueError("Invalid projection segments")
                    for key in ("id", "kind"):
                        _identifier(box.get(key), f"projection {key}")
                    if box["id"] not in object_ids:
                        raise ValueError("Projection does not match a source object")
                    for segment in box["segments"]:
                        _vector(segment, 4, "projection segment")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate camera identity")
            if camera_ids is not None and ids != camera_ids:
                raise ValueError("Camera slots must retain their identity and ordering")
            camera_ids = ids
        source_map = scene.get("map")
        if (not isinstance(source_map, dict) or not isinstance(source_map.get("features"), list)
                or not isinstance(source_map.get("layers"), dict)):
            raise ValueError("Map needs features and layer availability")
        _vector(source_map.get("bounds"), 4, "map bounds")
        xmin, ymin, xmax, ymax = source_map["bounds"]
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid map bounds")
        for feature in source_map["features"]:
            if not isinstance(feature, dict):
                raise ValueError("Invalid map feature")
            _identifier(feature.get("layer"), "map layer")
            _geometry(feature.get("geometry"))
        source_map["layer_labels"] = {
            name: _layer_label(state, sum(f["layer"] == name for f in source_map["features"]))
            for name, state in source_map["layers"].items()
        }
    if data.get("diagnostic_image") is not None:
        _asset(data["diagnostic_image"], root)
    if "timing_scope" in data and not isinstance(data["timing_scope"], str):
        raise ValueError("Invalid timing scope")
    return data


def export_source_gallery(bundle: dict, output: Path) -> dict:
    """Write an inspectable HTML file beside assets; refuse overwrite and external images."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.suffix.lower() != ".html":
        raise ValueError("Output must be an HTML file")
    data = _validate(bundle, output.parent)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(_HTML.replace("__SOURCE_DATA__", payload))
    return {"output": str(output), "scenes": len(data["scenes"]),
            "frames": sum(len(scene["frames"]) for scene in data["scenes"]),
            "kind": data["kind"]}


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAVSIM 源相机与场景检查</title><style>
:root{color-scheme:dark;font:15px/1.55 system-ui,sans-serif;background:#0c1620;color:#e4eef6}
*{box-sizing:border-box}body{max-width:1640px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 6px}
h2{font-size:19px;margin:0 0 10px}p{margin:8px 0}.muted{color:#9fb4c7}.banner{border-left:4px solid #e8b96a;padding:12px 16px;background:#253039;margin:16px 0}
button,select{font:inherit;padding:7px 10px;background:#23394a;border:1px solid #496178;border-radius:7px;color:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.panel{padding:16px;background:#142432;border:1px solid #30495c;border-radius:12px;margin-top:16px}
#scrub{flex:1;min-width:180px;accent-color:#59b5ed}.summary{display:grid;grid-template-columns:1fr 1fr;gap:16px}.summary>div{min-width:0}
#clock{font-variant-numeric:tabular-nums;white-space:pre-line}#sceneMeta{overflow-wrap:anywhere}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}canvas{width:100%;height:390px;display:block;background:#101c27;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}.camera{padding:12px;background:#142432;border:1px solid #30495c;border-radius:12px;min-width:0}
.camera h3{font-size:16px;margin:0 0 7px}.picture{position:relative;background:#07111a;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;overflow:hidden}
.picture a{position:absolute;inset:0}.picture img{display:block;width:100%;height:100%;object-fit:contain}.picture svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.status{padding:20px;color:#b3c3d0;text-align:center}
.camera p{font-size:12px;overflow-wrap:anywhere}.layer{display:inline-block;padding:2px 7px;margin:2px;background:#1b3447;border-radius:5px;font-size:12px}
#diagnostic{max-width:640px;width:100%;image-rendering:pixelated;border:1px solid #496178}.hidden{display:none!important}.status.bad{color:#ffc296}
@media(max-width:850px){body{padding:12px}.grid,.summary{grid-template-columns:1fr}canvas{height:330px}h1{font-size:22px}}
</style></head><body>
<h1>NAVSIM 源相机与场景检查</h1><p class="muted">真实日志相机 · 原始采样时钟 · 同帧对象与语义地图</p>
<div class="banner"><strong>这些是数据集原始图像，属于日志回放。</strong>车辆改变动作后，原图不会改变；它们不能直接充当反事实闭环观测，也不能认证当前几何渲染器的写实质量。</div>
<div class="panel"><div class="toolbar"><label>场景 <select id="scene"></select></label><button id="play">播放</button><button id="prev">上一帧</button><button id="next">下一帧</button><select id="speed" aria-label="播放倍率"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><input id="scrub" aria-label="源帧位置" type="range" min="0" value="0" step="1"></div>
<p id="clock"></p><p class="muted">时间戳来自源 frame / LiDAR；未提供逐相机曝光时刻，不认证硬件同步。播放为原生时间间隔预览，加载可延迟或跳过中间帧，并非实时相机流。</p><p id="timingScope" class="muted"></p><p id="sceneMeta" class="muted"></p><label><input id="boxes" type="checkbox"> 显示源 3D 框的 pinhole 投影（默认关闭）</label><p class="muted">投影仅诊断标定和对象位置；未校正相机畸变，不等同于像素级真值。点击图像可打开原文件。切换帧时隐藏尚未加载的新图，不沿用旧图。</p></div>
<section class="panel summary"><div><h2>同帧源对象与地图</h2><canvas id="bev" aria-label="语义地图和源对象俯视图"></canvas></div><div><h2>图层与来源边界</h2><div class="legend"><span><i class="dot" style="background:#6bb9ff"></i>车辆</span><span><i class="dot" style="background:#f3a0cd"></i>行人</span><span><i class="dot" style="background:#f4ca7c"></i>其他对象</span><span style="color:#5de0c0">▲ 自车</span><span>＋ 地图点（非行人）</span></div><p id="objectCounts"></p><div id="layers"></div><p class="muted">对象位置与类别来自当前源帧；地图只显示数据实际提供的图层。未提供和“已提供但无对象”分开标记。地图点统一画十字，避免与行人混淆。</p><p id="frameNote" class="muted"></p></div></section>
<main id="cameras" class="grid"></main>
<section class="panel"><h2>怎样评价当前 RGB？</h2><p>原始相机包含道路纹理、建筑、植被、照明和真实遮挡；当前 primitive 图像使用低分辨率色块、立方体与平面地图，只验证几何投影、遮挡、时序和“图像改变是否引起动作改变”。它不具备写实图像训练所需的视觉覆盖。</p><img id="diagnostic" class="hidden" alt="当前 160×96 几何渲染示意"><p class="muted">即便画面来自同一日志，不同相机、世界状态或位姿的图像也不能直接计算 PSNR / SSIM 来比较渲染质量。先验证相机坐标与动态状态对齐，再在相同视角与时刻评估重建；几何交互正确性、视觉保真度和闭环驾驶表现应分开评测。</p></section>
<script id="source-data" type="application/json">__SOURCE_DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('source-data').textContent);
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
let sceneIndex=0,frameIndex=0,playing=false,anchorWall=0,anchorSource=0,request=0,revision=0;
function scene(){return DATA.scenes[sceneIndex]}function frame(){return scene().frames[frameIndex]}
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function stop(){playing=false;cancelAnimationFrame(request);$('play').textContent='播放'}
function start(){if(frameIndex===scene().frames.length-1){frameIndex=0;render()}playing=true;anchorWall=performance.now();anchorSource=frame().time_seconds;$('play').textContent='暂停';request=requestAnimationFrame(animate)}
function animate(now){if(!playing)return;const frames=scene().frames;const target=anchorSource+(now-anchorWall)/1000*Number($('speed').value);let next=frameIndex;while(next+1<frames.length&&frames[next+1].time_seconds<=target)next++;if(next!==frameIndex){frameIndex=next;render()}if(frameIndex===frames.length-1){stop();return}request=requestAnimationFrame(animate)}
DATA.scenes.forEach((s,i)=>{const option=el('option',s.label);option.value=i;$('scene').append(option)});
$('scene').onchange=()=>{stop();sceneIndex=Number($('scene').value);frameIndex=0;render()};
$('play').onclick=()=>playing?stop():start();$('prev').onclick=()=>{stop();frameIndex=Math.max(0,frameIndex-1);render()};$('next').onclick=()=>{stop();frameIndex=Math.min(scene().frames.length-1,frameIndex+1);render()};
$('scrub').oninput=()=>{stop();frameIndex=Number($('scrub').value);render()};$('speed').onchange=()=>{if(playing){stop();start()}};
$('boxes').onchange=()=>document.querySelectorAll('.projection').forEach(svg=>svg.classList.toggle('hidden',!$('boxes').checked));
function color(kind){const k=kind.toLowerCase();return k.includes('pedestrian')||k==='pedestrian'?'#f3a0cd':k.includes('vehicle')||k==='car'?'#6bb9ff':'#f4ca7c'}
function cameraCard(camera,token){const card=el('article',undefined,'camera');card.append(el('h3',camera.id));const picture=el('div',undefined,'picture');if(camera.width&&camera.height)picture.style.aspectRatio=camera.width+'/'+camera.height;const status=el('div',camera.available?'正在加载本帧…':'本帧图像未提供','status');picture.append(status);card.append(picture);card.append(el('p',(camera.available?camera.width+' × '+camera.height+' · 原始源图':'缺失图像')+(camera.note?' · '+camera.note:''),'muted'));if(!camera.available)return card;
const image=new Image();image.alt=camera.id+' · 源帧 '+frame().source_index;image.onload=()=>{if(token!==revision)return;if(image.naturalWidth!==camera.width||image.naturalHeight!==camera.height){status.textContent='原图尺寸与元数据不一致，停止显示';status.classList.add('bad');return}const link=el('a');link.href=camera.image;link.target='_blank';link.rel='noopener';link.append(image);status.remove();picture.append(link);const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 '+camera.width+' '+camera.height);svg.setAttribute('preserveAspectRatio','xMidYMid meet');svg.classList.add('projection');svg.classList.toggle('hidden',!$('boxes').checked);for(const box of camera.projection||[]){const group=document.createElementNS(NS,'g');group.setAttribute('stroke',color(box.kind));group.setAttribute('stroke-width','2.5');group.setAttribute('fill','none');for(const p of box.segments){const line=document.createElementNS(NS,'line');['x1','y1','x2','y2'].forEach((k,i)=>line.setAttribute(k,p[i]));group.append(line)}svg.append(group)}picture.append(svg)};image.onerror=()=>{if(token!==revision)return;status.textContent='本帧图像加载失败（未显示旧帧）';status.classList.add('bad')};image.src=camera.image;return card}
function render(){const s=scene(),f=frame(),token=++revision;$('scrub').max=s.frames.length-1;$('scrub').value=frameIndex;$('prev').disabled=frameIndex===0;$('next').disabled=frameIndex===s.frames.length-1;$('play').disabled=s.frames.length<2;const delta=frameIndex?((f.timestamp_us-s.frames[frameIndex-1].timestamp_us)/1e6).toFixed(6)+' s':'首帧';$('clock').textContent='源帧 '+f.source_index+' · '+(frameIndex+1)+' / '+s.frames.length+' · t='+f.time_seconds.toFixed(6)+' s · 时间戳 '+f.timestamp_us+' µs\n与上一帧间隔 '+delta+' · 源采样估计 '+s.source_rate_hz.toFixed(3)+' Hz · 播放按实际时间戳';$('sceneMeta').textContent=s.label+' · 地图 '+s.map_name+' · 日志 '+s.log_name;$('cameras').replaceChildren(...f.cameras.map(c=>cameraCard(c,token)));$('layers').replaceChildren(...Object.entries(s.map.layer_labels).map(([name,label])=>el('span',name+'：'+label,'layer')));const counts={};for(const o of f.objects)counts[o.kind]=(counts[o.kind]||0)+1;$('objectCounts').textContent='当前帧 '+f.objects.length+' 个源对象 · '+Object.entries(counts).map(([k,v])=>k+' '+v).join(' / ');$('frameNote').textContent='自车源位姿：x='+f.ego_pose[0].toFixed(2)+' m，y='+f.ego_pose[1].toFixed(2)+' m，yaw='+f.ego_pose[2].toFixed(3)+' rad。源对象不代表可控交互主体。';drawBEV()}
function drawBEV(){const canvas=$('bev'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,[xmin,ymin,xmax,ymax]=scene().map.bounds,scale=Math.min((w-28)/(xmax-xmin),(h-28)/(ymax-ymin));function p(v){return [w/2+(v[0]-(xmin+xmax)/2)*scale,h/2-(v[1]-(ymin+ymax)/2)*scale]}function line(coords,close=false){ctx.beginPath();coords.forEach((v,i)=>{const q=p(v);if(i)ctx.lineTo(...q);else ctx.moveTo(...q)});if(close)ctx.closePath()}function point(v){const [x,y]=p(v);ctx.beginPath();ctx.moveTo(x-3,y);ctx.lineTo(x+3,y);ctx.moveTo(x,y-3);ctx.lineTo(x,y+3);ctx.stroke()}function geometry(g){const c=g.coordinates;switch(g.type){case'Point':point(c);break;case'MultiPoint':c.forEach(point);break;case'LineString':line(c);ctx.stroke();break;case'MultiLineString':c.forEach(a=>{line(a);ctx.stroke()});break;case'Polygon':ctx.beginPath();for(const ring of c){ring.forEach((v,i)=>{const q=p(v);if(i)ctx.lineTo(...q);else ctx.moveTo(...q)});ctx.closePath()}ctx.fill('evenodd');ctx.stroke();break;case'MultiPolygon':c.forEach(a=>geometry({type:'Polygon',coordinates:a}));break;case'GeometryCollection':g.geometries.forEach(geometry);break}}
for(const feature of scene().map.features){const layer=feature.layer.toLowerCase();ctx.lineWidth=1;ctx.strokeStyle=layer.includes('stop')?'#cb8496':layer.includes('cross')?'#baa483':layer.includes('lane')?'#527a98':'#456072';ctx.fillStyle=layer.includes('drivable')?'#243e4b':layer.includes('walk')?'#465b44':layer.includes('cross')?'#4c4447':'#1a303e';geometry(feature.geometry)}for(const item of frame().objects){const [x,y]=p(item.pose);ctx.save();ctx.translate(x,y);ctx.rotate(-item.pose[3]);ctx.fillStyle=color(item.kind);ctx.globalAlpha=.9;ctx.fillRect(-item.dimensions[0]*scale/2,-item.dimensions[1]*scale/2,item.dimensions[0]*scale,item.dimensions[1]*scale);ctx.strokeStyle='#0b1420';ctx.lineWidth=.7;ctx.strokeRect(-item.dimensions[0]*scale/2,-item.dimensions[1]*scale/2,item.dimensions[0]*scale,item.dimensions[1]*scale);ctx.restore()}const ego=frame().ego_pose,[ex,ey]=p(ego);ctx.save();ctx.translate(ex,ey);ctx.rotate(-ego[2]);ctx.fillStyle='#5de0c0';ctx.beginPath();ctx.moveTo(8,0);ctx.lineTo(-6,-5);ctx.lineTo(-6,5);ctx.closePath();ctx.fill();ctx.restore();ctx.fillStyle='#9fb4c7';ctx.font='11px system-ui';ctx.fillText('源地图 / 源框 · 米制坐标',10,h-8)}
if(DATA.diagnostic_image){$('diagnostic').src=DATA.diagnostic_image;$('diagnostic').classList.remove('hidden')}
$('timingScope').textContent=DATA.timing_scope?'清单时钟范围：'+DATA.timing_scope:'';
window.addEventListener('resize',drawBEV);render();
</script></body></html>'''
