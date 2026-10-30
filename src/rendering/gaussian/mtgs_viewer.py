"""Standalone inspection of real MTGS renders; never a driving success certificate."""

import base64
import json
from pathlib import Path


def export_mtgs_viewer(manifest: dict, output: Path) -> Path:
    """Embed a completed render probe beside its local image assets."""
    data = json.loads(json.dumps(manifest, allow_nan=False))
    if (data.get("schema_version") != 1 or data.get("kind") != "mtgs_renderer_probe"
            or data.get("status") != "completed"
            or data.get("trainable") is not False or not data.get("frames")):
        raise ValueError("Expected a non-training MTGS renderer probe")
    root = output.parent.resolve()
    for frame in data["frames"]:
        for name in ("baseline", "intervention", "restored", "source", "source_raw",
                     "camera_valid_mask"):
            asset = frame.get(name)
            if asset is None and name in {"source", "source_raw", "camera_valid_mask"}:
                continue
            if not isinstance(asset, str) or not asset:
                raise ValueError(f"Missing probe image: {name}")
            path = Path(asset)
            if (path.is_absolute() or ".." in path.parts or ":" in asset
                    or not (root / path).resolve().is_relative_to(root)
                    or not (root / path).is_file()):
                raise ValueError(f"Probe image is missing or outside output directory: {name}")
        if frame.get("camera_valid_mask"):
            frame["viewer_mask_data_url"] = "data:image/png;base64," + base64.b64encode(
                (root / frame["camera_valid_mask"]).read_bytes()).decode("ascii")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(_HTML.replace("__DATA__", payload))
    return output


_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · MTGS 写实渲染检查</title><style>
:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0b1420;color:#edf3f8}
body{max-width:1500px;margin:auto;padding:28px}h1{margin:8px 0 12px;font-size:28px}h2{font-size:17px}
p{line-height:1.7}.muted{color:#a6b8c9}.badge{border:1px solid #695a32;background:#2b281c;padding:10px 15px;border-radius:8px}
.toolbar{display:flex;gap:12px;align-items:center;margin:20px 0;flex-wrap:wrap}
button,select{background:#20364a;color:#fff;border:1px solid #486277;border-radius:6px;padding:9px;cursor:pointer}
button:disabled{opacity:.4;cursor:default}input[type=range]{flex:1;min-width:200px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
article{border:1px solid #304b62;background:#142434;border-radius:10px;padding:16px;min-width:0}
.picture{background:#070b11;min-height:100px;display:grid;place-items:center}
.picture a{position:relative;display:block;width:100%}.support-overlay{display:none;position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.show-support-mask .support-overlay{display:block}
img{width:100%;height:auto;display:block}a{color:#8dc7ff}.bad{color:#ffb1a6}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;max-height:360px;overflow:auto}
details{margin:16px 0;padding:16px;border:1px solid #304b62;border-radius:8px}
#metadata{overflow-wrap:anywhere}@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:16px}}
</style><body><p class="muted">DriveRSI / 单区域资产接入</p>
<h1>MTGS 写实渲染与对象干预</h1>
<p class="badge">渲染器检查 · trainable=false。此页检查图像和实例变换；没有进行车辆动力学推进或 Self-Play 训练。</p>
<p class="muted">每组保持相机与场景时刻一致，只修改指定车辆位姿，再恢复。源图属于采集日志；干预图没有对应实拍真值。未覆盖代理与资产限制见下方记录。</p>
<div class="toolbar"><button id="prev">上一组</button><input id="scrub" type="range" min="0" step="1" value="0" aria-label="检查帧"><button id="next">下一组</button><span id="clock"></span></div>
<p><label><input id="support" type="checkbox"> 黄色标出无源相机映射支持的区域</label> <span id="mask-status" class="muted"></span></p>
<p id="metadata"></p><p id="coverage" class="muted"></p><p><a id="raw" target="_blank" rel="noopener">查看加载器处理前的源图副本</a></p><div class="grid" id="images"></div>
<details open><summary>干预与恢复检查</summary><pre id="checks"></pre></details>
<details open><summary>对象覆盖和能力边界</summary><pre id="limits"></pre></details>
<details><summary>资产、相机与来源记录</summary><pre id="provenance"></pre></details>
<p class="muted">重建质量、对象完整性、交互正确性与采样性能需要分别验收。本页不以画面变化替代驾驶能力评价。</p>
<script>const DATA=__DATA__;let index=0,revision=0;const $=id=>document.getElementById(id);
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e}
function card(label,path,token,mask){const c=document.createElement('article');c.append(text('h2',label));const box=document.createElement('div');box.className='picture';c.append(box);const state=text('p',path?'加载本组图像…':'未提供源图','muted');box.append(state);if(!path)return c;const im=new Image();im.alt=label;im.onload=()=>{if(token!==revision)return;const link=document.createElement('a');link.href=path;link.target='_blank';link.rel='noopener';link.append(im);if(mask)addSupportOverlay(link,mask,token);box.replaceChildren(link);c.append(text('p',im.naturalWidth+' × '+im.naturalHeight,'muted'))};im.onerror=()=>{if(token!==revision)return;state.textContent='图像加载失败';state.className='bad'};im.src=path;return c}
function render(){const f=DATA.frames[index],token=++revision;renderMetadata(f);$('scrub').max=DATA.frames.length-1;$('scrub').value=index;$('prev').disabled=index===0;$('next').disabled=index===DATA.frames.length-1;$('clock').textContent=(index+1)+' / '+DATA.frames.length;$('images').replaceChildren(card('采集图像（匹配处理后的相机）',f.source,token,f.viewer_mask_data_url),card('原位姿重建',f.baseline,token,f.viewer_mask_data_url),card('车辆位姿干预',f.intervention,token,f.viewer_mask_data_url),card('恢复原位姿',f.restored,token,f.viewer_mask_data_url));$('checks').textContent=JSON.stringify({object:f.object,diagnostics:f.diagnostics},null,2);$('limits').textContent=JSON.stringify({capabilities:DATA.capabilities,limits:DATA.limits,source_inventory:f.source_inventory??DATA.source_inventory},null,2);$('provenance').textContent=JSON.stringify({backend:DATA.backend,assets:DATA.assets,camera:f.camera,frame_token:f.frame_token,cam_token:f.cam_token,travel_id:f.travel_id,source_raw:f.source_raw},null,2)}
function addSupportOverlay(link,url,token){
 const mask=new Image();mask.onload=()=>{if(token!==revision)return;
  const canvas=document.createElement('canvas');canvas.width=mask.naturalWidth;canvas.height=mask.naturalHeight;canvas.className='support-overlay';
  const ctx=canvas.getContext('2d');ctx.drawImage(mask,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
  for(let i=0;i<pixels.data.length;i+=4){const unsupported=pixels.data[i]<128;pixels.data[i]=255;pixels.data[i+1]=190;pixels.data[i+2]=35;pixels.data[i+3]=unsupported?150:0;}
  ctx.putImageData(pixels,0,0);link.append(canvas);
 };mask.src=url;
}
function renderMetadata(f){
 $('support').disabled=!f.viewer_mask_data_url;$('mask-status').textContent=f.viewer_mask_data_url?'仅表示镜头映射支持；不代表新视角或代理覆盖已验证':'未提供独立相机有效区域';
 const objects=f.source_inventory??[],bound=objects.filter(o=>o.reconstructed_rigid_node).length;
 const classes={};for(const o of objects)classes[o.class]=(classes[o.class]??0)+1;
 $('metadata').textContent='相机 '+f.camera_name+' · 源帧 '+f.frame_index+' · 相机时间 '+(f.camera_timestamp_us??'未提供')+' µs · LiDAR 时间 '+(f.source_timestamp_us??'未提供')+' µs';
 $('coverage').textContent='本源帧标注 '+objects.length+' 个对象（车辆 '+(classes.vehicle??0)+'、行人 '+(classes.pedestrian??0)+'、自行车 '+(classes.bicycle??0)+'）；'+bound+' 个有关联的刚体重建节点，其余 '+(objects.length-bound)+' 个外观覆盖未验证。所有对象尚未绑定驾驶物理世界。';
 if(!Array.isArray(f.source_inventory))$('coverage').textContent='本帧源对象清单未提供，不能判断覆盖情况。';
 $('raw').hidden=!f.source_raw;if(f.source_raw)$('raw').href=f.source_raw;
}
$('support').onchange=()=>document.body.classList.toggle('show-support-mask',$('support').checked);
$('prev').onclick=()=>{index=Math.max(0,index-1);render()};$('next').onclick=()=>{index=Math.min(DATA.frames.length-1,index+1);render()};$('scrub').oninput=()=>{index=Number($('scrub').value);render()};render();
</script></body></html>'''


def export_optics_viewer(sequence_dir: Path, output: Path) -> Path:
    """Show fixed-source optics comparisons with explicit FOV and exposure domains."""
    sequence_dir, output = Path(sequence_dir).resolve(), Path(output).resolve()
    data = json.loads((sequence_dir / "sequence.json").read_text())
    report = json.loads((sequence_dir / "report.json").read_text())
    if (data.get("trainable") is not False
            or data.get("reference_domain") != "official_checkpoint_exposure_aligned"
            or data.get("status") != "completed"
            or not data.get("frames")
            or data.get("expected_frames") != len(data["frames"])):
        raise ValueError("Optics viewer requires explicit non-training checkpoint reference domain")
    previous = None
    for frame in data["frames"]:
        stamp = frame["timestamp_us"]
        if type(stamp) is not int or (previous is not None and stamp <= previous):
            raise ValueError("Optics frames must be ordered by distinct source timestamps")
        previous = stamp
        assets = {}
        for key in ("raw", "lens", "native", "reference", "reference-checkpoint", "valid"):
            asset = (sequence_dir / frame["path"] / (key + ".png")).resolve()
            if not asset.is_relative_to(output.parent) or not asset.is_file():
                raise ValueError("Missing or out-of-directory optics image")
            assets[key] = asset.relative_to(output.parent).as_posix()
        frame["assets"] = assets
    data["diagnosis"] = {"top_blue_pixels": report["top_blue_pixels"],
                         "node_ablation": report.get("node_ablation", {}),
                         "summary": _optics_ablation_summary(report)}
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output.write_text(_OPTICS_HTML.replace("__DATA__", payload), encoding="utf-8")
    return output


def _optics_ablation_summary(report: dict) -> str:
    """Attribute the local color signature only when all diagnostic layers exist."""
    layers = report.get("node_ablation", {})
    counts = [layers.get(name, {}).get("top_blue_raw")
              for name in ("background", "skybox", "rigid")]
    if not all(type(value) is int and value >= 0 for value in counts):
        return "未完成三类节点的分层诊断，不能确定顶部蓝色碎片来源。"
    if counts[1] > 0 and counts[0] == counts[2] == 0:
        return "本次分层诊断仅在天空节点复现顶部蓝色签名；背景与刚体车辆层未复现。"
    return "分层诊断未把顶部蓝色签名单独定位到天空节点，请结合分层图像检查。"


_OPTICS_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · 同位姿镜头与画质对照</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101a27;color:#e4edf7}body{max-width:1500px;margin:auto;padding:24px}a{color:#83c9ff}button,input{font:inherit}button{padding:8px 16px;margin-right:8px}h1{font-size:27px}.note{background:#263749;border-left:4px solid #ffc66c;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;padding:12px;background:#1b293a}img{width:100%;display:block;background:black}figcaption{margin:8px 0}input{width:100%}pre{white-space:pre-wrap;word-break:break-word}summary{cursor:pointer;padding:12px}p{line-height:1.6}#status{font-variant-numeric:tabular-nums}#controls{position:sticky;top:0;background:#101a27;padding:12px 0;z-index:2}@media(max-width:800px){.grid{grid-template-columns:1fr}}.error{color:#ff9c9c}
</style><h1>同位姿镜头修复与连续帧检查</h1>
<p class="note">相同日志时刻、相机位姿与重建权重。左侧旧诊断图是扩展针孔画布，右侧为真实镜头映射；不是两张同视场图片。颜色误差仅在镜头映射后、与官方曝光参考对齐计算。数据仍为 trainable=false。</p>
<p id="diagnosis-summary"></p><p>镜头映射只修正视域，不保证重建缺陷消失。高分辨率试验保留作对照，未替换默认采样；本页不提供车辆纹理恢复证明。未重训资产、删除对象或补画像素。</p>
<div id="controls"><button id="play">播放</button><button id="pause">暂停</button><button id="previous">上一帧</button><button id="next">下一帧</button><button id="reset">重置</button><span id="status"></span><input id="time" type="range" min="0" value="0" aria-label="源帧时间轴"></div>
<div class="grid"><figure><figcaption>修复前：原始针孔画布（包含源镜头视场外区域）</figcaption><img id="raw" alt="原始针孔画布"></figure><figure><figcaption>修复后：按真实标定映射的 RGB</figcaption><img id="lens" alt="标定镜头渲染"></figure><figure><figcaption>真实照片：原曝光（保留原始内容）</figcaption><img id="reference" alt="原始真实照片"></figure><figure><figcaption>官方训练参考：同照片经 checkpoint 的曝光预处理</figcaption><img id="reference-checkpoint" alt="官方曝光参考"></figure></div>
<p id="metrics"></p><p id="error" class="error" role="alert"></p>
<details><summary>未采用的高分辨率试验与有效掩码</summary><div class="grid"><figure><figcaption>原生分辨率渲染再面积降采样：仅作对照</figcaption><img id="native" alt="高分辨率候选"></figure><figure><figcaption>共同采样有效区：不等于重建质量通过</figcaption><img id="valid" alt="共同有效掩码"></figure></div></details>
<details><summary>来源、分层诊断和连续帧指标定义</summary><p>本页沿原始来源相机序列取样，不是反事实多车 rollout。时间变化误差为未做光流补偿的参考残差变化，只作诊断，不是严格的闪烁评分。车辆 ROI 是固定右侧图像区域，包含车辆与背景，不能等同于逐实例评分。顶部蓝色计数是该缺陷的颜色签名，不是通用伪影检测器。</p><pre id="evidence"></pre></details>
<script>
'use strict';const DATA=__DATA__,F=DATA.frames,$=id=>document.getElementById(id);let at=F.length-1,timer=null,playing=false,closed=false;
$('diagnosis-summary').textContent=DATA.diagnosis.summary;
const show=()=>{const f=F[at];$('time').value=at;$('time').max=F.length-1;$('status').textContent=(at+1)+' / '+F.length+' · '+((f.timestamp_us-F[0].timestamp_us)/1e6).toFixed(3)+' s · 源帧 '+f.frame;for(const [key,path] of Object.entries(f.assets)){$(key).src=path;$(key).onerror=()=>{$('error').textContent='图像加载失败：'+key;};}$('metrics').textContent='同曝光域整图 PSNR：'+f.low.psnr_db.toFixed(2)+' dB；右侧车辆区域：'+f.vehicle_roi_low.psnr_db.toFixed(2)+' dB。曝光归一化系数：'+f.photometric.v_adjust_factor.toFixed(3)+'。顶部蓝色异常像素：'+f.lens_blue_top+'（不自动删改）。';$('previous').disabled=at===0;$('next').disabled=at===F.length-1;$('evidence').textContent=JSON.stringify({scope:DATA.scope,reference_domain:DATA.reference_domain,diagnosis:DATA.diagnosis,frame:f},null,2);};
function stop(){clearTimeout(timer);playing=false;$('play').textContent='播放';}
function queue(){if(!playing||closed)return;if(at===F.length-1){stop();return;}timer=setTimeout(()=>{at++;show();queue();},Math.max(1,(F[at+1].timestamp_us-F[at].timestamp_us)/1000));}
$('play').onclick=()=>{if(playing)return;if(at===F.length-1)at=0;playing=true;$('play').textContent='播放中';show();queue();};$('pause').onclick=stop;$('previous').onclick=()=>{stop();at=Math.max(0,at-1);show();};$('next').onclick=()=>{stop();at=Math.min(F.length-1,at+1);show();};$('reset').onclick=()=>{stop();at=0;show();};$('time').oninput=()=>{const requested=Number($('time').value);stop();at=requested;show();};window.addEventListener('pagehide',()=>{closed=true;stop();});show();
</script></html>'''
