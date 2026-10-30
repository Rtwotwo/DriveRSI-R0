"""Inspect source context, counterfactual history and future supervision separately."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def write_gallery(case_dir, sample_path):
    root, sample_path = Path(case_dir).resolve(), Path(sample_path).resolve()
    sample = json.loads(sample_path.read_text())
    history = json.loads((root / "source-history.json").read_text())
    report = json.loads((root / "report.json").read_text())

    def relative(path):
        return Path(os.path.relpath(path, root)).as_posix()

    source = []
    # Only the eight selected source files are copied for HTTP access. The
    # originals and all GS files remain untouched and content-bound.
    for i, frame in enumerate(history["frames"]):
        cameras = []
        for c in frame["cameras"]:
            original = Path(c["path"])
            if hashlib.sha256(original.read_bytes()).hexdigest() != c["sha256"]:
                raise ValueError("Source image changed")
            destination = root / "source-context" / f"{i}-{c['camera_id']}{original.suffix}"
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != c["sha256"]:
                    raise ValueError("Existing gallery image differs")
            else:
                shutil.copyfile(original, destination)
            cameras.append({"camera_id": c["camera_id"], "image": relative(destination)})
        source.append({"time": frame["relative_seconds"], "cameras": cameras})

    def frames(rows):
        return [
            {
                "time": f["relative_seconds"],
                "tick": f["tick"],
                "cameras": [
                    {"camera_id": c["camera_id"], "image": relative(c["image"]["path"])}
                    for c in f["cameras"]
                ],
            }
            for f in rows
        ]

    target = json.loads(Path(sample["target"]["trajectory"]["path"]).read_text())
    payload = {
        "source": source,
        "package_available": (sample_path.parent.parent / "manifest.json").is_file(),
        "package_root": relative(sample_path.parent.parent),
        "history": frames(sample["input"]["frames"]),
        "future": frames(sample["target"]["future_frames"]),
        "target": target,
        "source_span": history["span_seconds"],
        "sample": relative(sample_path),
        "report": {
            k: report[k]
            for k in ("baseline", "challenge", "responding_actors", "charged_joint_steps")
        },
        "rejections": sample["rejection_reasons"],
        "history_span": sample["history_span_seconds"],
        "history_hz": 3 / sample["history_span_seconds"],
        "session_viewer": relative(Path(sample["source"]["manifest"]).parent / "viewer.html"),
    }
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", chr(92) + "u003c")
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>四帧历史 → 反应式轨迹与多视角</title>
<style>body{font:15px system-ui;background:#111b26;color:#edf3fa;margin:28px}a{color:#80c6ff}section{background:#1b2b3b;padding:20px;border-radius:12px;margin:20px 0}select,input,button{font:inherit}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}img{width:100%;background:#000}figure{margin:0}figcaption{margin:8px 0}pre{white-space:pre-wrap}.note{color:#ffcd7e}canvas{width:100%;max-width:700px;background:#13202d}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}}</style>
<h1>四帧历史 → 反应式轨迹与多视角</h1><p>真实 NAVSIM 来源 · 共同世界驱动图像 · 规则候选，不是合格教师数据</p>
<section><label>自车相机 <select id="camera"></select></label><p id="clock"></p>
<h2>原始日志：初始化之前的四帧</h2><p>仅作来源上下文；不是偏离日志后可继续沿用的观测。</p><div class="grid" id="source"></div>
<h2>扰动后状态：真正对应新轨迹的四帧历史</h2><p>这四帧均来自同一模拟分支；当前锚点为 0 s，输入没有未来画面。</p><div class="grid" id="history"></div></section>
<section><h2>未来监督：执行轨迹与对应画面</h2><p class="note">用于标签或世界模型预测目标，不能进入当前 Actor 的输入。</p>
<label>未来帧 <input id="future-tick" type="range" min="0" value="0" step="1"></label><span id="future-label"></span>
<div class="grid" id="future"></div><canvas id="trajectory" width="700" height="220"></canvas></section>
<section><h2>检查与边界</h2><p class="note">trainable=false；尚无独立 PDM 评分，GS 质量准入保持关闭。</p><pre id="checks"></pre>
<a href="viewer.html">两分支 BEV 对照</a> · <a id="session-link">完整 GS 回放</a> · <a id="sample-link">样本清单</a></section>
<section id="package" hidden><h2>NAVSIM / SimScale 兼容数据包</h2><p>OpenScene 日志 · 4 帧历史 / 8 帧未来 · 8 路相机 · 后轴坐标轨迹。历史和未来图片分目录，原始数据保持独立。</p><p class="note">格式验收通过不等于训练准入；导航、信号、资产及画质证据仍有缺口。</p><a id="package-manifest">数据包字段与哈希</a> · <a id="package-loader">实际 NAVSIM 加载器验收</a></section>
<script>const D=__PAYLOAD__;
document.getElementById("package").hidden=!D.package_available;
document.getElementById("package-manifest").href=D.package_root+"/manifest.json";
document.getElementById("package-loader").href=D.package_root+"/loader-checks.json";
const camera=document.getElementById('camera'), slider=document.getElementById('future-tick');
D.history[0].cameras.forEach(c=>{const o=document.createElement('option');o.value=c.camera_id;o.textContent=c.camera_id;camera.append(o)});
slider.max=Math.max(0,D.future.length-1);slider.disabled=!D.future.length;
function render(id,rows){const box=document.getElementById(id);box.replaceChildren();rows.forEach(f=>{const c=f.cameras.find(c=>c.camera_id===camera.value);const fig=document.createElement('figure'), img=document.createElement('img'), cap=document.createElement('figcaption');if(c){img.src=c.image;img.alt=c.camera_id+' '+f.time+' s'}else{img.alt='本次来源对照未导出该相机'}cap.textContent=(f.time>0?'+':'')+f.time.toFixed(6)+' s'+(f.tick===undefined?'':' · tick '+f.tick);fig.append(img,cap);box.append(fig)})}
function draw(){render('source',D.source);render('history',D.history);render('future',D.future.length?[D.future[Number(slider.value)]]:[]);document.getElementById('future-label').textContent=D.future.length?' +'+D.future[Number(slider.value)].time.toFixed(1)+' s':'未导出未来图像';const c=document.getElementById('trajectory'),g=c.getContext('2d'),p=[[0,0],...D.target.poses];g.clearRect(0,0,c.width,c.height);const xmax=Math.max(1,...p.map(x=>x[0])),ymax=Math.max(2,...p.map(x=>Math.abs(x[1])));g.strokeStyle='#80c6ff';g.beginPath();p.forEach((v,i)=>{const x=30+v[0]/xmax*630,y=110-v[1]/ymax*80;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();g.fillStyle='#fff';g.fillText('实际执行轨迹 · 锚点车身中心坐标 · x 前向 / y 左向（横纵独立缩放）',20,205)}
camera.addEventListener('change',draw);slider.addEventListener('input',draw);window.addEventListener('pageshow',draw);
document.getElementById('clock').textContent='原始四帧跨度 '+D.source_span.toFixed(6)+' s；合成四帧按物理时钟采集，'+D.history_hz.toFixed(2)+' Hz、跨度 '+D.history_span.toFixed(3)+' s。';
document.getElementById('checks').textContent=JSON.stringify({运行:D.report,未准入原因:D.rejections},null,2);document.getElementById('sample-link').href=D.sample;document.getElementById('session-link').href=D.session_viewer;draw();</script></html>"""
    (root / "index.html").write_text(html.replace("__PAYLOAD__", data))
    return root / "index.html"
