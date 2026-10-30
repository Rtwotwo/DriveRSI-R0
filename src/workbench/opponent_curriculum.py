"""Inspectable safety curriculum and multi-city BEV evidence; no renderer claims."""
import argparse
import json
from pathlib import Path


def export_dashboard(run, matchups, output):
    run, output = Path(run), Path(output)
    report = json.loads((run / "report.json").read_text())
    config = json.loads((run / "config.json").read_text())
    matches = json.loads(Path(matchups).read_text())
    history = report["adaptive_opponents"]["state"]["history"]
    cases = []
    for row in matches["matchups"]:
        for ep in row["episodes"]:
            viewer = Path(ep["replay"]["viewer"]).resolve()
            # Relative in-gallery URLs only. Do not expose arbitrary server files.
            link = viewer.relative_to(output.parent.resolve()).as_posix()
            if not viewer.is_file():
                raise FileNotFoundError(viewer)
            cases.append({"city": ep["map_name"], "scene": ep["scenario_id"],
                          "source_log": ep["source_log"], "opponent": row["opponent_id"],
                          "seconds": ep["seconds"], "roles": ep["roles"],
                          "learner": ep["learner"], "opponent_metrics": ep["opponent"], "url": link})
    splits = {key: {"scenes": len(config["suite"][key]),
                    "logs": len({r.get("source_group", r["path"]) for r in config["suite"][key]})}
              for key in ("train", "curriculum", "evaluation")}
    data = {"history": history, "cases": cases, "splits": splits,
            "updates": report["updates"], "transitions": report["valid_transitions"],
            "seconds": report["seconds_including_eval_recording"],
            "limits": config["suite"]["limits"],
            "refreshes": report["adaptive_opponents"]["refreshes_this_run"]}
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", chr(92) + "u003c")
    text = _TEMPLATE.replace("__PAYLOAD__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return {"scenes": len({x["scene"] for x in cases}), "replays": len(cases),
            "cities": sorted({x["city"] for x in cases}), "revisions": len(history)}


_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DriveRSI · 对手课程与多城市场景</title>
<style>body{background:#0c1824;color:#e9f2f9;font:16px system-ui;margin:0}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:30px}h2{font-size:21px}.muted{color:#acc1d2;line-height:1.7}.notice{background:#2d2b1e;border-left:4px solid #efbc61;padding:16px}.stats,.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px;margin:18px 0}.card{background:#142b3b;border:1px solid #355164;border-radius:12px;padding:18px}.card p{line-height:1.5}a{color:#85d5ff}select,input,button{background:#203e50;color:white;border:1px solid #708d9f;padding:8px;border-radius:6px}table{width:100%;border-collapse:collapse;margin:18px 0}td,th{text-align:left;padding:12px;border-bottom:1px solid #385265}progress{width:100%}label{margin-right:16px}code{word-break:break-all}details{margin:18px 0}</style>
<main><h1>DriveRSI · 对手课程与多城市场景</h1>
<p class="muted">真实 NAVSIM 状态与动作回放 · 冻结历史对手 · 周期性安全 PFSP</p>
<p class="notice">本页展示 BEV 交互证据。没有为这些场景验证匹配 GS 资产，不能视为写实合成数据。碰撞、越界和低进度均保留；小规模验证不能证明驾驶能力提升。</p>
<div id="stats" class="stats"></div>
<h2>对手采样版本</h2><label>修订 <input id="revision" type="range" min="0" value="0"></label><span id="revisionText"></span>
<p class="muted">每轮同场景、同种子评测；安全失败 = 碰撞或越界。按平滑失败率 f 的 [f(1−f)] 权重与均匀探索混合。当前回合不换对手。评分使用同一种对手的整组车辆，是混合对手训练的代理指标。</p>
<table><thead><tr><th>冻结对手</th><th>失败／车辆回合</th><th>下次分配概率</th></tr></thead><tbody id="weights"></tbody></table>
<p id="cost" class="muted"></p><h2>独立评测场景回放</h2>
<label>城市 <select id="city"><option value="">全部</option></select></label>
<label>对手 <select id="opponent"><option value="">全部</option></select></label>
<p id="count" class="muted"></p><div id="cases" class="cards"></div>
<details><summary>数据划分和适用边界</summary><pre id="splits"></pre><ul id="limits"></ul></details>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
function el(tag,text,parent){const x=document.createElement(tag);x.textContent=text;if(parent)parent.append(x);return x;}
for(const text of [`${D.updates} 次 PPO 批次更新`,`${D.transitions} 条物理有效样本`,`${D.history.length} 次对手评测刷新`,`训练链含评测 ${D.seconds.toFixed(2)} 秒`])el('div',text,$('stats')).className='card';
$('revision').max=D.history.length-1;
function weights(){const r=D.history[+$('revision').value];$('revisionText').textContent=`r${r.revision} · PPO 批次更新 ${r.optimizer_update} · ${r.learner_version}`;$('weights').replaceChildren();for(const e of r.sampling.evidence){const tr=el('tr','',$('weights'));el('td',e.id,tr);el('td',`${e.unsafe} / ${e.actor_episodes}`,tr);el('td',`${(100*e.probability).toFixed(2)}%`,tr);}const c=D.refreshes.find(x=>x.revision===r.revision);$('cost').textContent=c?`刷新成本：${c.physical_world_steps}/${c.reserved_world_steps} 个实际/预留世界步 · 含记录 ${c.seconds_including_recording.toFixed(3)} 秒`:'本次运行未重做该修订；证据由检查点恢复。';}
for(const [id,key] of [['city','city'],['opponent','opponent']])for(const v of [...new Set(D.cases.map(x=>x[key]))].sort()){const o=el('option',v,$(id));o.value=v;}
function cases(){const rows=D.cases.filter(x=>(!$('city').value||x.city===$('city').value)&&(!$('opponent').value||x.opponent===$('opponent').value));$('count').textContent=`${new Set(rows.map(x=>x.scene)).size} 个场景 · ${rows.length} 组回放`;$('cases').replaceChildren();for(const c of rows){const card=el('article','',$('cases'));card.className='card';el('h3',c.city,card);el('code',c.scene,card);el('p',`${c.opponent} · 实际 ${c.seconds.toFixed(1)} 秒 · ${c.roles.length} 辆策略车辆`,card);el('p',`学习车辆：碰撞 ${c.learner.collision}，越界 ${c.learner.offroad}，终点进度 ${c.learner.endpoint_progress_m.toFixed(1)} m（欧氏诊断）`,card);el('p',`对手：碰撞 ${c.opponent_metrics.collision}，越界 ${c.opponent_metrics.offroad}`,card);const a=el('a','打开真实交互 BEV →',card);a.href=c.url;el('p','GS 资产：未验证 · 编译对象范围：不保证覆盖原日志全部对象',card).className='muted';}}
$('revision').addEventListener('input',weights);$('city').addEventListener('change',cases);$('opponent').addEventListener('change',cases);$('splits').textContent=JSON.stringify(D.splits,null,2);for(const t of D.limits)el('li',t,$('limits'));weights();cases();
window.addEventListener('pageshow',()=>setTimeout(()=>{weights();cases();},0));
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--matchups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_dashboard(args.run, args.matchups, args.output)))


if __name__ == "__main__":
    main()
