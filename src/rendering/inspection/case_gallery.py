"""Server-hosted candidate inspection without copied media or training claims."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_candidate_viewer(manifest_path: Path, web_root: Path) -> Path:
    manifest_path, web_root = Path(manifest_path).resolve(), Path(web_root).resolve()
    root = manifest_path.parent
    document = json.loads(manifest_path.read_text())
    if (document.get("kind") != "mtgs_long_tail_candidates" or document.get("status") != "completed"
            or document.get("trainable") is not False or not isinstance(document.get("candidates"), list)):
        raise ValueError("A completed non-training candidate catalog is required")
    artifact_root = Path(document["source"]["artifact_root"]).resolve()
    if not root.is_relative_to(web_root) or not artifact_root.is_relative_to(web_root):
        raise ValueError("Candidate pages and source media must stay inside web_root")
    output = root/"viewer.html"
    if output.exists():
        raise FileExistsError("Candidate viewer already exists")

    def url(path):
        path = path.resolve()
        if not path.is_relative_to(web_root) or not path.is_file():
            raise ValueError("Candidate reference escapes web_root or is missing")
        return quote(os.path.relpath(path, root).replace(os.sep, "/"), safe="/")

    image_urls, candidates = {}, []
    for candidate in document["candidates"]:
        clip_path = (root/candidate["clip"]["path"]).resolve()
        if not clip_path.is_relative_to(root):
            raise ValueError("Candidate clip escapes catalog")
        raw = clip_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != candidate["clip"]["sha256"]:
            raise ValueError("Candidate clip changed after mining")
        clip = json.loads(raw)
        for frame in clip["sensors"]:
            for camera in frame["cameras"]:
                value = camera["artifacts"]["image"]["path"]
                path = (artifact_root/value).resolve()
                if not path.is_relative_to(artifact_root):
                    raise ValueError("Candidate image escapes its source artifact root")
                if value not in image_urls:
                    image_urls[value] = url(path)
        candidates.append({**candidate, "clip_url": url(clip_path)})
    session_viewer = Path(document["source"]["session_manifest"]).parent/"viewer.html"
    payload = {"candidates": candidates, "images": image_urls,
               "session_url": url(session_viewer) if session_viewer.exists() else None,
               "summary": document["summary"], "eligibility": document["eligibility"],
               "configuration": document["configuration"],
               "behavior_policies": document.get("behavior_policies")}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output.write_text(_HTML.replace("__DATA__", encoded), encoding="utf-8")
    return output


_HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · 交互与渲染候选片段</title>
<style>:root{font-family:system-ui,"Microsoft Yahei",sans-serif;color-scheme:dark;background:#09131d;color:#e2edf5}body{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px}a{color:#81cdff}button,select{font:inherit;padding:9px;background:#18364a;color:inherit;border:1px solid #42627a;border-radius:6px}button{cursor:pointer}button:disabled{opacity:.4}select{max-width:95%}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}.notice{background:#2e271a;border:1px solid #847041;padding:14px;border-radius:8px}.muted{color:#a6bdcf}.error{color:#ffb8aa}#images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}article,details{background:#132636;padding:16px;border:1px solid #2a485e;border-radius:8px}article h3{font-size:16px;margin-top:0}img{display:block;width:100%;background:#080c12}pre{font:12px/1.6 monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto}details{margin-top:16px}#scrub{flex:1;min-width:250px}#info{line-height:1.8}summary{cursor:pointer}.links{display:flex;gap:18px}@media(max-width:750px){#images{grid-template-columns:1fr}}</style>
<p class="muted">DriveRSI · 固定策略群体 / 候选采样</p><h1>从交互记录定位候选片段</h1>
<p class="notice">候选不等于“长尾已证实”或正确驾驶监督。驾驶事件、行为变化、干预记录和渲染缺口分开显示。策略输入记录不完整，所有片段保持 trainable=false。</p>
<div class="bar"><label>候选类型 <select id="channel"></select></label><label>片段 <select id="cases"></select></label></div>
<p id="info"></p><p id="error" class="error" role="alert"></p><div class="links"><a id="full" target="_blank" rel="noopener">在完整回合中检查</a><a id="json" target="_blank" rel="noopener">查看片段 JSON</a></div>
<div class="bar"><button id="play">播放片段</button><button id="pause">暂停</button><button id="previous">上一帧</button><button id="next">下一帧</button><input id="scrub" type="range" min="0" step="1" value="0" aria-label="片段帧"><span id="time"></span></div>
<section id="images"></section><details open><summary>触发证据和区间联合动作</summary><pre id="evidence"></pre><pre id="actions"></pre></details>
<details><summary>逐车策略身份</summary><pre id="population"></pre></details><details><summary>候选配置、计数与训练资格</summary><pre id="summary"></pre></details>
<script>
const DATA=__DATA__,$=id=>document.getElementById(id),names={interaction_intervention:'交互干预',driving_behavior:'驾驶行为',driving_event:'驾驶事件',render_quality:'渲染支持缺口'},reasons={severe_deceleration:'强减速',policy_execution_mismatch:'执行动作与策略不同',collision:'碰撞',offroad:'越界',ego_render_asset_missing:'采集车外观缺失',render_height_unsupported:'高程查询不支持',novel_view_quality_unverified:'新视角质量未认证'};
let selected=null,clip=null,index=0,timer=null,playing=false,loadEpoch=0,deadline=0;
function text(tag,value,className){const x=document.createElement(tag);x.textContent=value;if(className)x.className=className;return x;}
function stop(){playing=false;clearTimeout(timer);}
function render(){
 if(!clip)return;const f=clip.sensors[index];$('scrub').value=index;$('time').textContent='tick '+f.tick+' · '+f.time_seconds.toFixed(2)+' s · '+(index+1)+'/'+clip.sensors.length;$('previous').disabled=index===0;$('next').disabled=index===clip.sensors.length-1;$('play').disabled=index===clip.sensors.length-1;
 const cards=f.cameras.map(c=>{const a=document.createElement('article');a.append(text('h3',c.camera_id),text('p','相机主体 '+c.owner_actor_id+(c.virtual?' · 虚拟机位，质量未认证':''),'muted'));const im=document.createElement('img');im.alt=c.camera_id+' 候选片段渲染';im.src=DATA.images[c.artifacts.image.path];im.onerror=()=>{$('error').textContent='源图片加载失败；该观测不可当作有效输入';};a.append(im);const h=c.capture_metadata?.height_binding;if(h)a.append(text('p','高度修正 '+(h.correction_m*100).toFixed(2)+' cm · '+(h.supported?'参考轨迹代理支持（非画质认证）':'未支持：'+h.reason),h.supported?'muted':'error'));return a;});$('images').replaceChildren(...cards);
 const previous=index?clip.sensors[index-1].tick:f.tick;const rows=clip.physics.filter(p=>p.observation_tick>=previous&&p.successor_tick<=f.tick);$('actions').textContent=JSON.stringify({from_tick:previous,to_tick:f.tick,joint_actions:rows.map(p=>({observation_tick:p.observation_tick,successor_tick:p.successor_tick,policy:p.behavior.policy_actions,executed:p.behavior.executed_actions,origins:p.behavior.action_origins??null}))},null,2);
 $('full').hidden=!DATA.session_url;if(DATA.session_url)$('full').href=DATA.session_url+'?generation='+selected.generation+'&tick='+f.tick;
}
function advance(){if(!playing||!clip)return;if(index>=clip.sensors.length-1){stop();return;}deadline+=(clip.sensors[index+1].time_seconds-clip.sensors[index].time_seconds)*1000;timer=setTimeout(()=>{index++;render();advance();},Math.max(0,deadline-performance.now()));}
async function load(){
 stop();const epoch=++loadEpoch;clip=null;selected=DATA.candidates.find(c=>c.candidate_id===$('cases').value);$('images').replaceChildren();$('time').textContent='';$('scrub').disabled=true;$('full').hidden=true;$('error').textContent='';$('actions').textContent='';$('evidence').textContent='';for(const id of ['play','previous','next'])$(id).disabled=true;
 if(!selected){$('info').textContent='这一类型没有候选；这不等于已证明安全或画质合格。';$('full').hidden=true;$('json').hidden=true;return;}
 $('info').textContent=(names[selected.channel]??selected.channel)+' · '+(reasons[selected.reason]??selected.reason)+' · 主体 '+(selected.actor_id??'场景')+' · 回合 '+selected.generation+' · 证据 tick '+selected.evidence_start_tick+'–'+selected.evidence_end_tick+' · 片段 '+selected.start_tick+'–'+selected.end_tick+(selected.context_clipped_before||selected.context_clipped_after?' · 上下文受回合边界限制':'');$('json').hidden=false;$('json').href=selected.clip_url;$('full').hidden=true;
 try{const response=await fetch(selected.clip_url,{cache:'no-store'});if(!response.ok)throw new Error('片段读取失败 HTTP '+response.status);const next=await response.json();if(epoch!==loadEpoch)return;clip=next;index=0;$('scrub').disabled=false;$('scrub').max=clip.sensors.length-1;$('evidence').textContent=JSON.stringify(clip.evidence,null,2);render();}catch(e){if(epoch===loadEpoch)$('error').textContent=e.message;}
}
function filter(){const cases=DATA.candidates.filter(c=>c.channel===$('channel').value);$('cases').replaceChildren(...cases.map(c=>{const o=text('option',c.candidate_id+' · '+(reasons[c.reason]??c.reason)+' · 回合'+c.generation);o.value=c.candidate_id;return o;}));load();}
$('channel').replaceChildren(...Object.entries(names).map(([id,label])=>{const o=text('option',label+' ('+DATA.candidates.filter(c=>c.channel===id).length+')');o.value=id;return o;}));
$('channel').onchange=filter;$('cases').onchange=load;$('pause').onclick=stop;$('play').onclick=()=>{if(clip&&!playing){playing=true;deadline=performance.now();advance();}};
$('previous').onclick=()=>{stop();index=Math.max(0,index-1);render();};$('next').onclick=()=>{stop();index=Math.min(clip.sensors.length-1,index+1);render();};$('scrub').oninput=()=>{stop();if(clip){index=Number($('scrub').value);render();}};
$('population').textContent=JSON.stringify(DATA.behavior_policies??'旧记录未提供策略群体身份',null,2);$('summary').textContent=JSON.stringify({configuration:DATA.configuration,summary:DATA.summary,eligibility:DATA.eligibility},null,2);window.addEventListener('pagehide',()=>{loadEpoch++;stop();});filter();
</script></html>"""
