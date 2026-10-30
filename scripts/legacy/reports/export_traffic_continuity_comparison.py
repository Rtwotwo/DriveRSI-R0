"""Publish a reproducible all-actor comparison from recorded continuity runs."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from driversi.contracts.road import Route
from driversi.contracts.scene import Scene


def load(directory):
    scene = Scene.load(directory / "scene.npz")
    with np.load(directory / "rollout.npz", allow_pickle=False) as z:
        states, present = z["states"], z["presence"]
        life = json.loads(str(z["lifecycle"]))
        events = {k:int(z[k].sum()) for k in ("collision", "offroad")}
    route = Route(**scene.metadata["driving_tasks"]["actors"]["ego"]["route"])
    ego = scene.actor_ids.index("ego")
    forward = np.sum(states[:,:,4:6] * np.stack((np.cos(states[:,:,3]), np.sin(states[:,:,3])), -1), -1)
    metrics = {
        "duration_s": round((len(states)-1)*.1, 3),
        "identities": len(scene.actor_ids),
        "contact_state_count": sum(bool(x["contacts"]) for x in life),
        "owner_contact_state_count": sum(any("ego" in p for p in x["contacts"]) for x in life),
        "ego_route_progress_m": float(route.project(states[-1,ego,:2])[0]-route.project(states[0,ego,:2])[0]),
        "negative_longitudinal_actor_states": int(np.count_nonzero((forward < -.001) & present)),
        "ego_final_speed_mps": float(np.linalg.norm(states[-1,ego,4:6])),
        "policy_events": events,
        "rollout_sha256": hashlib.sha256((directory / "rollout.npz").read_bytes()).hexdigest(),
    }
    if "continuity" in life[-1]:
        c = life[-1]["continuity"]
        metrics.update(final_status={k:c["status"].count(k) for k in ("active","departed","pending","unseen")},
                       pending=c["pending"], departures=c["departures"], admissions=c["admissions"])
    rows = []
    for i, actor in enumerate(scene.actor_ids):
        contacts = [t for t,x in enumerate(life) if any(actor in p for p in x["contacts"])]
        rows.append({"actor":actor,"number":i+1,"kind":int(scene.kinds[i]),
                     "contact_states":len(contacts), "first_contact_tick":contacts[0] if contacts else None,
                     "min_longitudinal_speed_mps":float(forward[:,i][present[:,i]].min()) if present[:,i].any() else None,
                     "present_states":int(present[:,i].sum()),
                     "final_status":life[-1].get("continuity",{}).get("status",["source_or_sim_presence"]*len(scene.actor_ids))[i]})
    return scene, metrics, rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    scene, before, oldrows=load(args.before)
    afterscene, after, rows=load(args.after)
    if scene.actor_ids != afterscene.actor_ids or scene.metadata["source_log"] != afterscene.metadata["source_log"]:
        raise ValueError("Comparison must preserve source log and all identities")
    np.testing.assert_array_equal(scene.poses[0],afterscene.poses[0])
    if (len(scene.actor_ids) != 35 or after["duration_s"] != 60
            or {int(k):int((scene.kinds==k).sum()) for k in np.unique(scene.kinds)} != {1:17,2:1,3:2,10:15}
            or after["final_status"] != {"active":24,"departed":6,"pending":5,"unseen":0}):
        raise ValueError("This release page describes the fixed 60 s nuPlan regression; use a new report for different inputs")
    args.output.mkdir(parents=True,exist_ok=True)
    evidence={"schema":1,"before":before,"after":after,"actors_before":oldrows,"actors_after":rows,
              "source_log":scene.metadata["source_log"],"trainable":False,
              "qualification":"Changed admission times and generated navigation; no official PDM or full-source fidelity claim"}
    (args.output/"comparison.json").write_text(json.dumps(evidence,indent=2,allow_nan=False),encoding="utf-8")
    b=os.path.relpath(args.before/"viewer.html",args.output).replace(os.sep,"/")
    a=os.path.relpath(args.after/"viewer.html",args.output).replace(os.sep,"/")
    source=os.path.relpath(args.after/"source.html",args.output).replace(os.sep,"/")
    def link(base, actor, tick, label):
        return f'<a href="{html.escape(base)}?actor={quote(actor)}&amp;tick={tick}">{html.escape(label)}</a>'
    labels={1:"车辆",2:"行人",3:"骑行者",10:"其他物体"}
    status={"active":"在场","departed":"已越过出口","pending":"等待接管","unseen":"尚未出现"}
    options=''.join(f'<option value="{html.escape(x["actor"])}">{x["number"]} · {labels[x["kind"]]} · {html.escape(x["actor"])}</option>' for x in rows)
    stats=''.join(f'<tr><td>{label}</td><td>{before[key]:.1f}</td><td>{after[key]:.1f}</td></tr>' for label,key in [
        ("自车沿路线前进 / m","ego_route_progress_m"),("任意物体接触的状态数","contact_state_count"),
        ("含自车接触的状态数","owner_contact_state_count"),("负纵向速度的对象状态数","negative_longitudinal_actor_states"),
        ("自车末帧速度 / m/s","ego_final_speed_mps")])
    actorrows=''.join(f'<tr><td>{x["number"]}</td><td>{labels[x["kind"]]}</td><td>{link(a,x["actor"],600,x["actor"])}</td><td>{status[x["final_status"]]}</td><td>{oldrows[i]["contact_states"]} → {x["contact_states"]}</td></tr>' for i,x in enumerate(rows))
    pending=''.join(f'<li>{link(a,p["actor"],600,p["actor"])}：原始入场 {p["source_tick"]*.1:.1f}s；等待 {p["delay_steps"]*.1:.1f}s；占用者 '+', '.join(link(a,v,600,v) for v in p['blocking_actors'])+'</li>' for p in after['pending'])
    departures=''.join(f'<li>{link(a,actor,max(0,event["tick"]-10),actor)}：{event["tick"]*.1:.1f}s，全体轮廓越过指定出口且向外运动。</li>' for actor,event in after['departures'].items())
    document="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DriveRSI · 连续交通堵塞修复</title>
<style>body{margin:0;background:#edf2f4;color:#172c3c;font:16px/1.65 system-ui,sans-serif}main{max-width:1360px;margin:auto;padding:26px}header{background:#102b3a;color:white;border-radius:16px;padding:24px}h1{font-size:27px;margin:3px 0}h2{font-size:21px}a{color:#087a85}header a{color:#99f1e5}.tag{color:#9cd9d4}.panel{background:white;border-radius:14px;padding:20px;margin-top:18px}.warning{background:#fff3dc;border-left:5px solid #c38420;padding:14px}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #d9e2e7;padding:9px;text-align:left}select,input,button{padding:9px;font:inherit;border:1px solid #b6c8ce;border-radius:6px}button{background:#007f85;color:white;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}iframe{width:100%;height:750px;border:1px solid #cad8de;border-radius:10px}.scroll{overflow:auto}.small{font-size:14px;color:#516574}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:12px}iframe{height:680px}}</style>
<main><header><div class="tag">DRIVERSI / REAL NUPLAN / 60 s</div><h1>连续交通：堵塞修复前后对照</h1><p>保留同一日志、全部35个身份和相同自车意图。17辆车、1名行人、2名骑行者、15个其他物体。这里展示规则控制仿真；源日志另列。</p><a href="__AFTER__?actor=ego&amp;tick=286">打开修复后完整播放</a> · <a href="__BEFORE__?actor=ego&amp;tick=286">修复前播放</a> · <a href="__SOURCE__">源日志参考</a></header>
<section class="panel"><h2>结果与边界</h2><table><thead><tr><th>60秒执行指标</th><th>修复前</th><th>修复后</th></tr></thead><tbody>__STATS__</tbody></table><p class="small">接触状态数以10 Hz记录，持续接触会重复计数；不是独立事故数，也不是官方PDM。负纵向速度按当前朝向投影，统计阈值为−0.001 m/s。</p><p class="warning">末帧24个对象在场、6个通过指定出口离场、5个身份因重叠等待接管。等待身份没有删除，但尚未进入物理世界；这改变了原始入场时刻。内部地图死端仍保留停车，不能宣称全场景无堵塞或完整日志复现。当前 trainable=false。</p></section>
<section class="panel"><h2>同对象、同时刻对照</h2><label>对象 <select id="actor">__OPTIONS__</select></label> <label>时间 / 秒 <input id="seconds" type="number" min="0" max="60" step="0.1" value="28.6"></label> <button id="compare">跳到两侧相同时刻</button><p class="small">两侧各自播放；切换对象或时间后点击按钮重新对齐。优先看ego、27号车的路线延续、16/25号车的错时接管，以及等待身份。原始视频/图像未参与本次BEV验收。</p><div class="grid"><div><h3>修复前 · 保留原执行记录</h3><iframe id="before" title="修复前回放" src="__BEFORE__?actor=ego&amp;tick=286"></iframe></div><div><h3>修复后 · 连续接管与路线</h3><iframe id="after" title="修复后回放" src="__AFTER__?actor=ego&amp;tick=286"></iframe></div></div></section>
<section class="panel"><h2>等待接管：原始重叠未消失</h2><ul>__PENDING__</ul><h2>真实越界离场证据</h2><ul>__DEPARTURES__</ul></section>
<section class="panel"><h2>全部对象审计</h2><div class="scroll"><table><thead><tr><th>编号</th><th>类型</th><th>身份 / 末帧跳转</th><th>末帧状态</th><th>接触状态数：前 → 后</th></tr></thead><tbody>__ROWS__</tbody></table></div><p><a href="comparison.json">完整指标、身份与文件SHA256</a></p></section>
<script>const before=__BJSON__,after=__AJSON__;document.getElementById('compare').addEventListener('click',()=>{const actor=document.getElementById('actor').value;const t=Number(document.getElementById('seconds').value);if(!Number.isFinite(t))return;const tick=Math.round(Math.max(0,Math.min(60,t))*10);for(const [id,path]of [['before',before],['after',after]])document.getElementById(id).src=path+'?actor='+encodeURIComponent(actor)+'&tick='+tick;});</script></main></html>"""
    for key,value in {"__AFTER__":html.escape(a),"__BEFORE__":html.escape(b),"__SOURCE__":html.escape(source),"__STATS__":stats,"__OPTIONS__":options,"__PENDING__":pending,"__DEPARTURES__":departures,"__ROWS__":actorrows,"__BJSON__":json.dumps(b),"__AJSON__":json.dumps(a)}.items():
        document=document.replace(key,value)
    (args.output/"index.html").write_text(document,encoding="utf-8")
    print(json.dumps({"before":before,"after":after},ensure_ascii=False))


if __name__ == "__main__":
    main()
