"""Publish real native-video output and physical overlays, never training admission."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from driversi.rendering.video.validation import source_comparison, validate_output


def project_boxes(frame, cam):
    camera = frame["cameras"][cam]
    pose = np.asarray(camera["camera_to_world"])
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    output = []
    for actor in frame["actors"]:
        xyz = (np.asarray(actor["corners_world"]) - pose[:3, 3]) @ pose[:3, :3]
        uv = cv2.projectPoints(
            xyz,
            np.zeros(3),
            np.zeros(3),
            np.asarray(camera["intrinsic"]),
            np.asarray(camera["distortion"]),
        )[0].reshape(-1, 2)
        uv = uv * [403 / 1920, 226 / 1080] - [1, 2]
        lines = [
            [uv[a].tolist(), uv[b].tolist()]
            for a, b in edges
            if xyz[a, 2] > 0.3
            and xyz[b, 2] > 0.3
            and np.isfinite(uv[[a, b]]).all()
            and np.abs(uv[[a, b]]).max() < 5000
        ]
        output.append(dict(actor=actor["actor_id"], lines=lines))
    return output


def publish(novel, source, output, street=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    n, ng, _ = validate_output(novel)
    s, sg, _ = validate_output(source)
    quality = source_comparison(novel, source)
    geometry = {
        "novel": {x["tick"]: x for x in ng["frames"]},
        "source": {x["tick"]: x for x in sg["frames"]},
    }
    rows = []
    guides = json.loads(Path(street).read_text()) if street else None
    if guides and guides["branch_identity"] != n["branch_identity"]:
        raise ValueError("Street conditions from another branch")
    for nf, sf in zip(n["frames"], s["frames"]):
        tick = nf["tick"]
        row = dict(
            tick=tick,
            timestamp_us=nf["timestamp_us"],
            role="新历史" if tick <= 20 else "纠正未来",
            cameras={},
        )
        for cam in n["camera_order"]:
            paths = {}
            for mode, frame in [("novel", nf), ("source", sf)]:
                name = f"images/{mode}-{tick:04d}-{cam}.png"
                Image.open(frame["cameras"][cam]["path"]).save(output / name)
                paths[mode] = name
            raw = Image.open(geometry["source"][tick]["cameras"][cam]["source_image"]).convert(
                "RGB"
            )
            if raw.size != (1920, 1080):
                raise ValueError("Gallery crop only verified for measured source rig")
            name = f"images/raw-{tick:04d}-{cam}.png"
            raw.resize((403, 226), Image.Resampling.BILINEAR).crop((1, 2, 401, 226)).save(
                output / name
            )
            paths["raw"] = name
            if guides:
                name = f"images/guide-{tick:04d}-{cam}.png"
                Image.open(guides["frames"][tick]["cameras"][cam]["guide"]).save(output / name)
                paths["guide"] = name
            row["cameras"][cam] = dict(
                paths=paths,
                novel_boxes=project_boxes(geometry["novel"][tick], cam),
                source_boxes=project_boxes(geometry["source"][tick], cam),
            )
        rows.append(row)
    (output / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    data = dict(
        camera_order=n["camera_order"],
        rows=rows,
        quality=quality,
        street_generated=False,
        elapsed_seconds={"novel": n["elapsed_seconds"], "source": s["elapsed_seconds"]},
    )
    (output / "manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__DATA__", payload)
    (output / "index.html").write_text(html, encoding="utf-8")
    return quality


HTML = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>同分支视频生成验收 · DriveRSI</title>
<style>body{font:16px system-ui;background:#eff4f6;color:#163342;margin:24px auto;max-width:1380px;padding:0 20px}h1{font-size:26px}p{line-height:1.7}button,select,input{font:inherit;margin:5px;padding:8px}button{cursor:pointer}#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.tile{background:#102532;color:white;border-radius:8px;overflow:hidden}.tile label{display:block;padding:8px}.tile canvas{width:100%;display:block}#status{background:#fff1d6;padding:16px;border-radius:10px}section{background:white;padding:15px;border-radius:10px;margin:14px 0}#seek{width:65%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}#stamp{font-weight:bold}a{color:#007b88} @media(max-width:800px){#grid{grid-template-columns:1fr}}</style>
<h1>同一物理分支：WorldDreamer 与 StreetCrafter 接入检查</h1>
<p>Singapore navtrain · 4 帧新历史＋10 帧实际未来 · 6 相机 · 2 Hz。历史到达和教师后续来自同一执行分支。黄色框是物理对象的投影，不能证明生成图像遵守了这些位置。</p>
<div id="status">视觉未准入，未启动合成数据配对训练。WorldDreamer 已实际生成；StreetCrafter 当前展示点云条件，缺少可加载的官方权重，没有生成视频。</div>
<section><table><tr><th>验收项</th><th>实际结论</th></tr><tr><td>轨迹</td><td>95.18 反事实 PDMS（原生 4 秒评分）；10 个未来点为实际 5 秒执行，不能称为 5 秒官方基准分数。</td></tr><tr><td>源位姿图像复现</td><td id="metric"></td></tr><tr><td>车辆位置／多视角／时序</td><td>条件与执行状态已核对；生成像素尚未通过独立验收，保留框与视频逐帧核查。</td></tr><tr><td>训练反馈</td><td>视觉阶段拒绝，模型、loss 和评分未更改；未产生学生收益或晋级结论。</td></tr></table></section>
<section><select id="mode"><option value="novel">新轨迹：WorldDreamer 实际生成</option><option value="source">原日志位姿：WorldDreamer 对照</option><option value="raw">真实源图（仅对照，不是新视角真值）</option><option value="guide">StreetCrafter 点云条件（不是模型生成）</option></select><label><input id="boxes" type="checkbox" checked>显示物理框</label><button id="play">播放</button><button id="reset">回到开始</button><br><input id="seek" type="range" min="0" max="13" step="1" value="0"><span id="stamp"></span></section>
<div id="grid"></div><p id="note"></p><section><b>边界与失败保留</b><p>视频模型可能改变车辆外观、数量或道路标线。StreetCrafter 的离线点云包含后续来源观测，原生双向时序条件也可能影响历史像素；尚未通过历史因果一致性检查。原始图像不等于反事实真值；相邻帧相似也不等于正确运动。WorldDreamer 原生输出为 400×224，页面放大不增加信息，源图 PSNR 与高清 GS 指标也不等价。</p><p>官方加载与坐标修正前的失败、Harness 图像依赖错误和未完成的 StreetCrafter 权重下载保留在 exp/conditioned-video-loop-v1。完整结果：<a href="quality.json">图像对照记录</a> · <a href="manifest.json">逐帧清单</a></p></section>
<script>const data=__DATA__;let index=0,timer=null,revision=0;const mode=document.querySelector('#mode'),seek=document.querySelector('#seek'),boxes=document.querySelector('#boxes');const canvases={};for(const cam of data.camera_order){const tile=document.createElement('div');tile.className='tile';const label=document.createElement('label');label.textContent=cam;const canvas=document.createElement('canvas');canvas.width=400;canvas.height=224;tile.append(label,canvas);document.querySelector('#grid').append(tile);canvases[cam]=canvas;}document.querySelector('#metric').textContent=`均值 ${data.quality.source_psnr_mean_db?.toFixed(2)} dB；门槛保持 27 dB。仅在原生 400×224 源位姿对照上计算。`;
function draw(){const version=++revision,row=data.rows[index];seek.value=index;document.querySelector('#stamp').textContent=`${row.role} · t=${(row.tick/10).toFixed(1)} s · ${index+1}/14`;document.querySelector('#note').textContent=mode.value==='guide'?'点云来自对应重建资产；车辆按实际物理状态移动。此视图仅是生成条件，不是 StreetCrafter 输出；离线点云包含整段来源观测，不是仅历史资产。':'黄色框按实际相机畸变投影；高程与部分原生生成条件近似仍需独立检查。';for(const cam of data.camera_order){const item=row.cameras[cam],im=new Image();im.onload=()=>{if(version!==revision)return;const ctx=canvases[cam].getContext('2d');ctx.clearRect(0,0,400,224);ctx.drawImage(im,0,0,400,224);if(boxes.checked&&mode.value!=='guide'){const list=mode.value==='novel'?item.novel_boxes:item.source_boxes;ctx.strokeStyle='#ffe200';ctx.lineWidth=1.5;for(const actor of list){for(const edge of actor.lines){ctx.beginPath();ctx.moveTo(...edge[0]);ctx.lineTo(...edge[1]);ctx.stroke();}}}};im.src=item.paths[mode.value];}}
function stop(){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放';}document.querySelector('#play').onclick=()=>{if(timer){stop();return;}document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{if(index===13){stop();return;}index++;draw();},500);};seek.oninput=()=>{index=Number(seek.value);draw();};mode.onchange=draw;boxes.onchange=draw;document.querySelector('#reset').onclick=()=>{stop();index=0;draw();};draw();</script></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("novel", "source", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--street")
    args = parser.parse_args()
    print(
        json.dumps(publish(args.novel, args.source, args.output, args.street), ensure_ascii=False)
    )
