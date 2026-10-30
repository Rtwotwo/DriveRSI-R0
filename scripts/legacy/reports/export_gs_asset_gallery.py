"""Publish same-branch GS transport comparisons and rejected source coverage."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def build(experiment, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = {"before": experiment / "before-render", "after": experiment / "after-v5-render"}
    manifests = {k: read(p / "session/manifest.json") for k, p in paths.items()}
    for k, m in manifests.items():
        if m["status"] != "completed":
            raise ValueError("Only completed captures can be compared")
        dest = output / k
        target = paths[k] / "session"
        if dest.is_symlink():
            if dest.resolve() != target.resolve():
                raise ValueError("Existing gallery link points to another run")
        elif dest.exists():
            raise ValueError("Gallery session path already exists")
        else:
            dest.symlink_to(target.resolve(), target_is_directory=True)
    if len(manifests["before"]["frames"]) != len(manifests["after"]["frames"]):
        raise ValueError("Capture timeline differs")
    frames = []
    for b, a in zip(manifests["before"]["frames"], manifests["after"]["frames"]):
        if (b["tick"], b["simulation_timestamp_us"], b["actors"]) != (
            a["tick"],
            a["simulation_timestamp_us"],
            a["actors"],
        ):
            raise ValueError("Physical states differ; comparison is not same pose")
        cameras = []
        for bc, ac in zip(b["cameras"], a["cameras"]):
            if bc["camera_id"] != ac["camera_id"]:
                raise ValueError("Camera mapping differs")
            bi = np.asarray(Image.open(paths["before"] / "session" / bc["image"]), dtype=np.int16)
            ai = np.asarray(Image.open(paths["after"] / "session" / ac["image"]), dtype=np.int16)
            cameras.append(
                dict(
                    name=ac["camera_id"],
                    before="before/" + bc["image"],
                    after="after/" + ac["image"],
                    source="after/" + ac["source"],
                    pixel_mae=float(np.abs(bi - ai).mean()),
                    source_asset_support=ac.get("support"),
                    dynamic_binding=ac.get("dynamic_binding"),
                )
            )
        frames.append(
            dict(
                tick=a["tick"],
                time=a["time_seconds"],
                cameras=cameras,
                coverage=a["coverage"],
                role="anchor"
                if a["tick"] == 20
                else "history"
                if 5 <= a["tick"] < 20
                else "future"
                if a["tick"] > 20
                else "initial",
            )
        )
    selection = read(experiment / "window-selection-v2.json")
    package = read(experiment / "diagnostic-dataset/manifest.json")
    summary = dict(
        kind="gs_asset_transport_comparison",
        same_physics=True,
        trainable=False,
        trajectory_admitted=False,
        paired_training_started=False,
        scope="diagnostic_baseline_not_qualified_teacher_or_quality_claim",
        frames=frames,
        image_counts=package["image_counts"],
        rejection_reasons=package["rejection_reasons"],
        coverage=read(paths["after"] / "asset-preflight.json"),
        source_window_counts={
            s: sum(x["status"] == s for x in selection["rows"])
            for s in sorted({x["status"] for x in selection["rows"]})
        },
        render_receipt=read(paths["after"] / "report.json"),
    )
    (experiment / "comparison.json").write_text(json.dumps(summary, indent=2))
    (output / "comparison.json").write_text(json.dumps(summary, indent=2))
    for name in [
        "window-selection-v2.json",
        "worldengine-source-coverage.json",
        "reference-audit.json",
    ]:
        (output / name).write_text((experiment / name).read_text())
    payload = json.dumps(summary).replace("<", "\u003c")
    template = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · 动态资产与物理配准</title><style>
body{font:16px/1.65 system-ui;margin:0;background:#edf2f5;color:#172e3b}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:21px}section{background:white;padding:22px;border-radius:16px;margin:18px 0}a{color:#08747b}button,select{font:inherit;padding:7px 13px;border:1px solid #abc;border-radius:8px;background:white}input{width:100%}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.images img{width:100%;background:#071b27}.badge{display:inline-block;background:#fff0d4;color:#734600;padding:4px 12px;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dce4e9;padding:10px}p{max-width:1150px}.mono{font:13px monospace;word-break:break-all}#state{font-weight:bold}.warn{border-left:5px solid #d99428}footer{font-size:13px;color:#536b79}@media(max-width:850px){.images{grid-template-columns:1fr}}
</style><main><h1>动态对象资产与物理配准 · 实际执行对照</h1>
<p>navtrain · Boston · 同一物理分支 · 4 帧新历史 + 8 帧未来。修复相机、物体位移及外参的坐标基准；缺失资产和失败评分完整保留。</p>
<span class="badge">诊断数据：尚未准入训练</span>
<section class="warn"><h2>这次解决了什么，还缺什么</h2><p>原始地图位移现在先变换到源帧重建坐标，相机和动态对象使用同一旋转规则；相机到车体的外参同步更新，不再重复添加参考轨迹高度。它修复坐标计算，不等于证明地面几何、遮挡和外观正确。</p>
<p>旧两份高分分支没有对应动态节点。本次选用有 __BOUND__ 个对应节点的训练遍历；末帧 __DRIVEN__ 个节点由世界状态驱动，但原始窗口仍有 __MISSING__ 个身份缺少独立可控资产。未删对象、未替换身份，未放宽准入。</p>
<table><tr><th>证据</th><th>实际结果</th></tr><tr><td>新日志教师搜索</td><td>9 个候选，最高未来 PDMS 91.76；存在背景接触，0 个合格</td></tr><tr><td>本页渲染分支</td><td>有界控制扰动延续对照，非学习型 Self-Play；完整 70 步物理重放、60 步渲染状态核对</td></tr><tr><td>图像与样本</td><td>四相机 52 张连续图像，窗口为 16 张历史 + 32 张未来 + 8 个轨迹点</td></tr><tr><td>训练与收益</td><td>轨迹及视觉准入未通过；未启动配对训练，不报告收益</td></tr></table>
<p><strong>95 分门槛检查的是新锚点之后 4 秒的未来轨迹。</strong>历史检查连续性和接触；不会给历史图片套用“95 分”。此前两个 95 分分支和本次有资产分支不是同一个样本。</p></section>
<section><h2>同状态、同时间、同相机对照</h2><p>左：旧地图轴直接应用；中：源帧坐标变换修复；右：原始实拍，仅作来源参考，偏离原轨迹后不是像素真值。</p>
<button id="play">播放</button> <button id="reset">回到开始</button> <select id="camera" aria-label="相机"><option>CAM_F0</option><option>CAM_L0</option><option>CAM_R0</option><option>CAM_B0</option></select> <span id="state"></span>
<input id="time" aria-label="时间轴" type="range" min="0" max="12" value="0" step="1">
<div class="images"><div>修复前<img id="before" alt="同位姿修复前"></div><div>修复后<img id="after" alt="同位姿修复后"></div><div>原始照片参考<img id="source" alt="来源实拍参考"></div></div><p id="detail"></p>
<p><a href="before/viewer.html">原分支完整回放</a> · <a href="after/viewer.html">修复后完整回放</a> · <a href="comparison.json">逐帧证据</a></p></section>
<section><h2>资产筛选与替代方法</h2><p>当前 MTGS 检查点对应的 16 个可评估训练窗口没有完整独立对象覆盖；另外两份受保护日志未用于训练筛选。新取得的 WorldEngine 官方资产含 31 个对象节点，但对应原始日志的覆盖检查也未通过。这不等于所有缺失对象在图像中都为空：部分可能被烘焙在静态背景里，仍不能证明其随交互移动。</p>
<p><a href="window-selection-v2.json">逐窗口缺失原因</a> · <a href="worldengine-source-coverage.json">新资产覆盖检查</a> · <a href="reference-audit.json">参考源码与固定版本</a></p>
<table><tr><th>参考</th><th>本次结论</th></tr><tr><td>SimScale / WorldEngine</td><td>复用真实日志重建并按世界状态渲染；必须有匹配的动态资产和配准。已下载一份官方资产核查，尚未接入合格渲染。</td></tr><tr><td>VATIX</td><td>支持轨迹条件视频；输出编码帧率不等于任意物理采样时刻，多车状态和同步多相机尚未验证。</td></tr><tr><td>MagicDrive-V2</td><td>提供多视角控制生成；NAVSIM 条件、标定和交互一致性仍需适配，不可直接冒充物理分支。</td></tr></table></section>
<footer>评分、原生模型及 loss 未修改。执行完成与数据合格分别记录。全部产物保存在 32341，训练输出约定 exp/。</footer></main>
<script>const D=__DATA__, slider=document.getElementById('time'), camera=document.getElementById('camera');let timer=null;
function draw(){let f=D.frames[+slider.value],c=f.cameras.find(x=>x.name===camera.value);for(let k of ['before','after','source'])document.getElementById(k).src=c[k];let roles={initial:'初始诊断帧',history:'新历史输入',anchor:'第4帧历史 / 决策锚点',future:'未来监督（不进入当前输入）'};document.getElementById('state').textContent=` ${f.time.toFixed(1)} 秒 · tick ${f.tick} · ${roles[f.role]}`;document.getElementById('detail').textContent=`修复前后像素平均绝对差 ${c.pixel_mae.toFixed(3)} /255，仅表示变化量，不表示质量改善。轨迹、对象与物理时刻逐帧一致。`;}
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放';}
document.getElementById('play').onclick=()=>{if(timer)return stop();if(+slider.value===D.frames.length-1)slider.value=0;document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(+slider.value===D.frames.length-1)return stop();slider.value=+slider.value+1;draw();},500);draw();};document.getElementById('reset').onclick=()=>{stop();slider.value=0;draw();};slider.oninput=draw;camera.onchange=draw;slider.max=D.frames.length-1;draw();</script></html>"""
    coverage = manifests["after"]["frames"][-1]["coverage"]
    template = template.replace("__BOUND__", str(coverage["bound_rigid_count"]))
    template = template.replace("__DRIVEN__", str(len(coverage["world_driven_rigid_tokens"])))
    template = template.replace("__MISSING__", str(len(summary["coverage"]["missing_tokens"])))
    (output / "index.html").write_text(template.replace("__DATA__", payload))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    s = build(a.experiment, a.output)
    print(json.dumps(dict(frames=len(s["frames"]), trainable=s["trainable"])))
