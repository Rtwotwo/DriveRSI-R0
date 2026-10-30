"""Standalone inspection of actual recorded visual policy inputs (no re-render)."""

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np


def _png(rgb: np.ndarray) -> str:
    """Lossless RGB PNG using standard-library chunks and unfiltered scanlines."""
    height, width, _ = rgb.shape

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    rows = b"".join(b"\0" + row.tobytes() for row in rgb)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _load(path: Path) -> tuple[dict, dict]:
    required = {"metadata", "actor_ids", "physical_actor_ids", "states", "motion", "rgb",
                "history_valid", "history_ticks", "telemetry", "navigation", "observation_tick",
                "newly_decided", "policy_actions", "executed_actions", "overrides", "memories",
                "trainable", "valid", "terminated", "truncated", "collision", "offroad", "goal"}
    with np.load(path, allow_pickle=False) as bundle:
        if missing := required - set(bundle.files):
            raise ValueError(f"Visual record missing fields: {', '.join(sorted(missing))}")
        raw = bundle["metadata"]
        if raw.shape != () or raw.dtype.kind not in "US":
            raise ValueError("Visual metadata must be a JSON scalar")

        def reject(value):
            raise ValueError(f"Non-finite metadata value: {value}")

        metadata = json.loads(str(raw), parse_constant=reject)
        if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
                or metadata.get("kind") != "visual_engineering_rollout"
                or metadata.get("modality") != "primitive_rgb"
                or metadata.get("learning_status") != "diagnostic_policy_no_model_update"
                or metadata.get("action_space") != "jerk_discrete_12"):
            raise ValueError("Unsupported visual engineering schema")
        arrays = {name: bundle[name].copy() for name in required - {"metadata"}}
    for key in ("actor_ids", "physical_actor_ids"):
        ids = arrays[key]
        if (ids.ndim != 1 or ids.dtype.kind not in "US" or not len(ids)
                or len(set(ids.tolist())) != len(ids) or any(not str(x) for x in ids)):
            raise ValueError(f"Invalid {key}")
    if not set(arrays["actor_ids"]) <= set(arrays["physical_actor_ids"]):
        raise ValueError("Visual actors must map to physical state rows")
    count, physical_count = len(arrays["actor_ids"]), len(arrays["physical_actor_ids"])
    steps, dt = metadata.get("steps"), metadata.get("dt_seconds")
    if (not isinstance(steps, int) or steps < 1 or not isinstance(dt, (int, float))
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Invalid physical step count or clock")
    rgb = arrays["rgb"]
    if rgb.ndim != 7 or rgb.shape[:2] != (steps, count) or rgb.shape[-1] != 3:
        raise ValueError("Invalid rgb shape; expected [step,actor,history,camera,height,width,3]")
    _, _, history, cameras, height, width, _ = rgb.shape
    if rgb.dtype != np.uint8 or min(history, cameras, height, width) <= 0:
        raise ValueError("Invalid RGB dimensions or dtype")
    rig = metadata.get("configuration", {}).get("rig", {}).get("cameras", [])
    if (len(rig) != cameras or len({c.get("camera_id") for c in rig}) != cameras
            or any(not c.get("camera_id") or c.get("width") != width
                   or c.get("height") != height for c in rig)):
        raise ValueError("Camera rig does not match recorded RGB")
    shapes = {"states": (steps+1, physical_count, 13), "motion": (steps+1, physical_count, 3),
              "history_valid": (steps, count, history, cameras),
              "history_ticks": (steps, count, history), "telemetry": (steps, count, 4),
              "navigation": (steps, count, 2), "observation_tick": (steps,),
              "newly_decided": (steps,)}
    booleans = {"history_valid", "newly_decided", "overrides", "trainable", "valid",
                "terminated", "truncated", "collision", "offroad", "goal"}
    integers = {"history_ticks", "observation_tick", "policy_actions", "executed_actions"}
    for name in booleans | {"policy_actions", "executed_actions"}:
        shapes.setdefault(name, (steps, count))
    for name, shape in shapes.items():
        value = arrays[name]
        if (value.shape != shape or value.dtype.kind not in "biuf"
                or not np.isfinite(value).all()
                or name in booleans and value.dtype.kind != "b"
                or name in integers and value.dtype.kind not in "iu"):
            raise ValueError(f"Invalid {name}")
    memory = arrays["memories"]
    if (memory.ndim != 3 or memory.shape[:2] != (steps+1, count)
            or memory.dtype.kind not in "fiu" or not np.isfinite(memory).all()):
        raise ValueError("Invalid actor memory boundaries")
    if arrays["trainable"].any():
        raise ValueError("Diagnostic records cannot contain trainable transitions")
    for name in ("policy_actions", "executed_actions"):
        if np.any((arrays[name] < 0) | (arrays[name] >= 12)):
            raise ValueError("Action must be in jerk_discrete_12")
    if np.any((arrays["policy_actions"] != arrays["executed_actions"]) & ~arrays["overrides"]):
        raise ValueError("Changed executed actions require an override marker")
    ticks, decided = arrays["observation_tick"], arrays["newly_decided"]
    if (not decided[0] or ticks[0] != 0 or np.any(ticks < 0)
            or np.any(ticks > np.arange(steps)) or np.any(np.diff(ticks) < 0)
            or np.any(ticks[decided] != np.arange(steps)[decided])
            or np.any(ticks[1:][~decided[1:]] != ticks[:-1][~decided[1:]])):
        raise ValueError("Observation clock contradicts physical/held-action timing")
    history_ticks, history_valid = arrays["history_ticks"], arrays["history_valid"]
    present = history_valid.any(axis=-1)
    if (np.any(history_ticks < -1) or np.any((history_ticks >= 0) != present)
            or np.any(history_ticks > ticks[:, None, None])
            or np.any(np.diff(history_ticks, axis=-1) < 0)
            or np.any(history_ticks[:, :, -1] != ticks[:, None])):
        raise ValueError("History ticks and validity masks disagree")
    identities = metadata.get("frame_identity", [])
    if (len(identities) != steps
            or any(item.get("tick") != int(tick) for item, tick in zip(identities, ticks))):
        raise ValueError("Frame identities disagree with observation ticks")
    for index in np.flatnonzero(~decided):
        for name in ("rgb", "history_valid", "history_ticks", "telemetry", "navigation", "policy_actions"):
            if not np.array_equal(arrays[name][index], arrays[name][index-1]):
                raise ValueError(f"Held observation changed {name}")
    valid = arrays["valid"]
    if (np.any(arrays["terminated"] & arrays["truncated"])
            or any(np.any(arrays[name] & ~valid)
                   for name in ("terminated", "truncated", "collision", "offroad", "goal"))):
        raise ValueError("Events and termination contradict valid transitions")
    return metadata, arrays


def export_visual_replay(visual_path: Path, output: Path) -> dict:
    """Export validated pre-action images; output must not already exist."""
    if output.exists():
        raise FileExistsError(output)
    metadata, arrays = _load(visual_path)
    rgb = arrays.pop("rgb")
    images, cache, frame_images = [], {}, []
    for frame in rgb:
        actor_images = []
        for actor in frame:
            camera_images = []
            for image in actor[-1]:
                key = hashlib.sha256(image.tobytes()).digest()
                if key not in cache:
                    cache[key] = len(images)
                    images.append(_png(image))
                camera_images.append(cache[key])
            actor_images.append(camera_images)
        frame_images.append(actor_images)
    payload = {"metadata": metadata, "images": images, "frame_images": frame_images,
               **{name: value.tolist() for name, value in arrays.items()}}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = _HTML.replace("__PAYLOAD__", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(html)
    return {"frames": metadata["steps"], "actors": len(arrays["actor_ids"]),
            "cameras": rgb.shape[3], "unique_images": len(images),
            "modality": "primitive_rgb", "output": str(output)}


_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · 视觉交互检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#101923;color:#e9eef5;font:15px system-ui,sans-serif}
header,main{max-width:1480px;margin:auto;padding:20px}h1{font-size:25px;margin:0 0 10px}
p{line-height:1.6;margin:8px 0}.muted{color:#aebdcd}.badge{color:#f9ce7b}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 0}
button,a{color:#e9eef5;background:#284158;border:1px solid #45627e;border-radius:6px;padding:8px 12px}
input{flex:1;min-width:150px}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
article{background:#192836;border:1px solid #334b5e;border-radius:10px;padding:16px;min-width:0}
h2{font-size:18px;overflow-wrap:anywhere}.cameras{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;flex:1;min-width:280px}img{display:block;width:100%;min-height:192px;max-height:384px;object-fit:contain;background:#050b11;image-rendering:pixelated}
figcaption{padding:6px 0;font-size:13px;color:#b8cbdc}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;font:13px system-ui}
canvas{width:100%;height:95px;background:#101923;border-radius:5px}.event{color:#ffc18a}#clock{font-weight:600}
</style><header><h1>DriveRSI · 多主体视觉交互检查</h1>
<p class="badge">primitive_rgb · 虚拟相机 · 几何观测 · 诊断策略，未进行模型学习</p>
<p class="muted">显示实际送入策略的最新历史图像，不重新渲染。持有动作期间保留原观测；本页展示决策前输入，末状态观测另存于 visual.npz，以其采样时刻为准。</p>
<p id="source"></p><div class="controls"><button id="play">播放</button><button id="prev">上一步</button><button id="next">下一步</button>
<input id="scrub" type="range" min="0" value="0"><a href="bev.html">打开配套 BEV 检查器</a></div>
<p id="clock"></p><p id="identity" class="muted"></p></header><main><div id="cards"></div></main>
<script>"use strict";
const D=__PAYLOAD__;
const M=D.metadata,S=M.steps,dt=M.dt_seconds,rig=M.configuration.rig.cameras;
let index=0,playing=false,last=0,acc=0;
const el=id=>document.getElementById(id),fmt=x=>Number(x).toFixed(3);
const row=D.actor_ids.map(a=>D.physical_actor_ids.indexOf(a));
const speed=(t,a)=>Math.hypot(D.states[t][row[a]][4],D.states[t][row[a]][5]);
const cards=D.actor_ids.map((actor,a)=>{
 const card=document.createElement('article'),title=document.createElement('h2');title.textContent=actor;card.append(title);
 const cameras=document.createElement('div');cameras.className='cameras';card.append(cameras);
 const cameraEls=rig.map(c=>{const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.alt='实际策略输入：'+c.camera_id;figure.append(img,caption);cameras.append(figure);return{img,caption};});
 const heading=document.createElement('p');heading.className='badge';heading.textContent='相机朝向资格：'+(M.configuration.heading_qualification?.[actor]||'未提供');card.append(heading);
 const status=document.createElement('pre'),events=document.createElement('p'),chart=document.createElement('canvas');events.className='event';chart.width=640;chart.height=120;chart.setAttribute('aria-label','实际世界速度曲线，单位米每秒');card.append(status,events,chart);el('cards').append(card);return{cameraEls,status,events,chart};
});
const timing=M.scene_metadata?.source_timing?.interval_seconds;
el('source').textContent=`物理记录 ${fmt(1/dt)} Hz · 总推进 ${fmt(S*dt)} s · ${D.actor_ids.length} 个视觉主体 · ${rig.length} 路相机/主体。`+(timing?.median>0?`源日志约 ${fmt(1/timing.median)} Hz；源帧率与相机采样不同。`:'源日志采样率未提供。');
el('scrub').max=S-1;
function drawChart(canvas,a){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);const values=Array.from({length:S+1},(_,t)=>speed(t,a)),top=Math.max(1,...values);ctx.strokeStyle='#73bfff';ctx.lineWidth=2;ctx.beginPath();values.forEach((v,t)=>{const x=10+t/S*(w-20),y=h-20-v/top*(h-35);t?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.strokeStyle='#f7c976';ctx.beginPath();ctx.moveTo(10+index/S*(w-20),5);ctx.lineTo(10+index/S*(w-20),h-15);ctx.stroke();ctx.fillStyle='#b8cbdc';ctx.font='14px sans-serif';ctx.fillText(`实际速度 0–${top.toFixed(1)} m/s · 0–${(S*dt).toFixed(1)} s`,12,15);}
function render(){el('scrub').value=index;const source=D.observation_tick[index],f=M.frame_identity[index];
 el('clock').textContent=`物理输入 tick ${index} (${fmt(index*dt)} s) → 后继 tick ${index+1} (${fmt((index+1)*dt)} s) ｜ 图像观测 tick ${source} (${fmt(source*dt)} s) ｜ ${D.newly_decided[index]?'新决策':'持有先前动作'}`;
 el('identity').textContent=`episode ${f.episode} · branch ${f.branch} · generation ${f.generation} · ${index===S-1?'最后一条已记录转移；末状态观测见 visual.npz':'图片对应动作执行前的策略输入'}`;
 cards.forEach((card,a)=>{const ticks=D.history_ticks[index][a],mask=D.history_valid[index][a];card.cameraEls.forEach((c,k)=>{const valid=mask[mask.length-1][k];c.img.src=D.images[D.frame_images[index][a][k]];c.img.style.opacity=valid?'1':'.25';c.caption.textContent=`${rig[k].camera_id} · ${rig[k].width}×${rig[k].height} · ${valid?'有效输入':'无效输入（不可作训练观测）'} · tick ${ticks[ticks.length-1]}`;});
 card.status.textContent=`世界速度：${fmt(speed(index,a))} → ${fmt(speed(index+1,a))} m/s\n观测自车速度：${fmt(D.telemetry[index][a][0])} m/s\n策略动作：${D.policy_actions[index][a]} ｜ 执行动作：${D.executed_actions[index][a]} ｜ ${D.overrides[index][a]?'显式干预':'未覆盖'}\n动作空间：jerk_discrete_12（索引）\n历史 ticks：${JSON.stringify(ticks)}\n历史有效掩码：${JSON.stringify(mask)}\n逐车记忆：${JSON.stringify(D.memories[index][a])} → ${JSON.stringify(D.memories[index+1][a])}\n有效转移：${D.valid[index][a]} ｜ trainable=false`;
 const names={collision:'碰撞',offroad:'越界',goal:'到达目标',terminated:'终止',truncated:'时间截断'};const events=Object.entries(names).filter(([key])=>D[key][index][a]).map(([,label])=>label);card.events.textContent=events.length?'本步事件：'+events.join('、'):'本步未记录上述事件（不等于驾驶资格通过）';drawChart(card.chart,a);});}
function stop(){playing=false;el('play').textContent='播放';}
el('play').onclick=()=>{playing=!playing;if(playing&&index===S-1)index=0;el('play').textContent=playing?'暂停':'播放';last=performance.now();acc=0;render();};
el('prev').onclick=()=>{stop();index=Math.max(0,index-1);render();};el('next').onclick=()=>{stop();index=Math.min(S-1,index+1);render();};el('scrub').oninput=e=>{stop();index=Number(e.target.value);render();};
function animate(now){if(playing){acc+=(now-last)/1000;if(acc>=dt){const n=Math.floor(acc/dt);acc-=n*dt;index=Math.min(S-1,index+n);if(index===S-1)stop();render();}}last=now;requestAnimationFrame(animate);}render();requestAnimationFrame(animate);
</script></html>'''
