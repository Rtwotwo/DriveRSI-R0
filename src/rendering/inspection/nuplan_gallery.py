"""Bounded nuPlan inspection gallery: source reference versus executed diagnostic."""

import argparse
import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.rendering.bev.replay import export_replay, export_source_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout

CITIES = {
    "us-ma-boston": "Boston",
    "sg-one-north": "Singapore",
    "us-nv-las-vegas-strip": "Las Vegas",
    "us-pa-pittsburgh-hazelwood": "Pittsburgh",
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def thumbnail(scene, output):
    """Ego-centred source preview; cropping is display only, never data admission."""
    tick = int(np.argmax(scene.presence.sum(axis=1))) if scene.presence is not None else 0
    ego = scene.actor_ids.index("ego") if "ego" in scene.actor_ids else 0
    cx, cy = scene.poses[tick, ego, :2]

    def point(x, y):
        return (200 + (x - cx) * 3, 140 - (y - cy) * 3)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 280"><rect width="400" height="280" fill="#102031"/>'
    ]
    for i, kind in enumerate(scene.road_types):
        pts = scene.road_points[scene.road_offsets[i] : scene.road_offsets[i + 1], :2]
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(*p) for p in pts))
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{("#547383" if kind == 6 else "#2d4659")}" stroke-width="1"/>'
        )
    for i in np.flatnonzero(scene.presence_at(tick)):
        x, y = point(*scene.poses[tick, i, :2])
        length, width = scene.dimensions[i, :2] * 3
        color = "#65d5cb" if i == ego else "#edb86e" if scene.kinds[i] == 1 else "#d7a6e6"
        parts.append(
            f'<rect x="{-length / 2:.2f}" y="{-width / 2:.2f}" width="{length:.2f}" height="{width:.2f}" fill="{color}" transform="translate({x:.2f} {y:.2f}) rotate({-np.degrees(scene.poses[tick, i, 3]):.2f})"/>'
        )
    parts.append(
        f'<rect x="8" y="8" width="290" height="25" rx="4" fill="#102031dd"/><text x="16" y="25" fill="#d8e6ef" font-size="12">SOURCE · t={scene.times[tick]:.1f}s · ego vicinity</text></svg>'
    )
    output.write_text("".join(parts))


def diagnostic(scene, folder):
    with PufferDriveWorld(scene) as world:
        initial = world.snapshot()
        record = RolloutRecord.start(world)
        while not world.episode_done:
            actions = {actor: 6 for actor in world.actor_ids}
            step = world.step(actions)
            record.append(world, actions, step)
        write_rollout(
            scene,
            folder / "diagnostic.npz",
            record=record,
            snapshot=initial,
            controlled_ids=world.actor_ids,
            policy="fixed-action6-interface-diagnostic",
            controller_configuration={"controller": "fixed-action6-interface-diagnostic"},
            protocol={
                "purpose": "data_interface_inspection_not_driving_policy",
                "trainable": False,
            },
        )
        ego = scene.actor_ids.index("ego")
        onsets, previous, ego_onsets = 0, set(), 0
        for row in record.lifecycle:
            pairs = {tuple(sorted(pair)) for pair in row.get("contacts", [])}
            new = pairs - previous
            onsets += len(new)
            ego_onsets += sum("ego" in pair for pair in new)
            previous = pairs
        events = {
            name: int(sum(t.events[name].sum() for t in record.transitions))
            for name in ("collision", "offroad", "goal")
        }
        return {
            "steps": len(record.transitions),
            "seconds": len(record.transitions) * 0.1,
            "controlled_events": events,
            "contact_pair_onsets": onsets,
            "ego_contact_pair_onsets": ego_onsets,
            "ego_final_flags": record.states[-1][ego, 9:13].astype(int).tolist(),
            "initial_stopped": int(record.states[0][scene.controlled, 9].sum()),
            "backend_build": initial.build_id,
            "trainable": False,
        }


def render_index(output, report):
    def esc(x):
        return html.escape(str(x), quote=True)

    cards = []
    for row in sorted(report["cases"], key=lambda r: (-r.get("actors", 0), r["id"])):
        d = row.get("diagnostic", {})
        links = []
        for key, label in [
            ("full_log_viewer", "整段原始 DB"),
            ("source_viewer", "9 秒源参考"),
            ("diagnostic_viewer", "实际动作诊断"),
            ("evidence", "逐场景记录"),
            ("scene", "场景 NPZ"),
        ]:
            if row.get(key):
                links.append(f'<a href="{esc(row[key])}">{label} ↗</a>')
        facts = f"峰值同时在场 {row.get('peak_present', '—')} · 策略槽位 {row.get('controlled', '—')} · {esc(row.get('role', '未知'))}"
        status = (
            f"实际推进 {d['seconds']:.1f} / {row['source_seconds']:.1f} s · 物体接触开始 {d['contact_pair_onsets']} 次（含初态）<br>受控碰撞事件 {d['controlled_events']['collision']} · 越界 {d['controlled_events']['offroad']} · Ego 接触开始 {d['ego_contact_pair_onsets']} 次"
            if d
            else esc(row.get("error", "待生成"))
        )
        thumb = (
            f'<img loading="lazy" src="{esc(row["thumbnail"])}" alt="{esc(row["id"])} 源轨迹自车附近俯视预览">'
            if row.get("thumbnail")
            else ""
        )
        cards.append(
            f'<article data-city="{esc(row["city"])}" data-count="{row.get("actors", 0)}" data-search="{esc(row["id"] + " " + row.get("token", "") + " " + row.get("source_log", ""))}"><div class="preview">{thumb}</div><div class="body"><div class="eyebrow">{esc(CITIES.get(row["city"], row["city"]))} · {esc(row["status"])}</div><h2>{esc(row["id"])} <b>{row.get("actors", "—")} 对象</b></h2><p>{facts}</p><p class="token">{esc(row.get("token", ""))}</p><p class="status">{status}</p><p class="links">{"".join(links)}</p><details><summary>来源与类别</summary><p class="token">{esc(row.get("source_log", ""))}</p><p>{esc(json.dumps(row.get("kinds", {}), ensure_ascii=False))}</p><p>数据校验与驾驶质量分别验收；trainable=false。</p></details></div></article>'
        )
    cities = "".join(
        f'<option value="{esc(k)}">{esc(CITIES.get(k, k))}</option>'
        for k in sorted({r["city"] for r in report["cases"]})
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>nuPlan · 密集场景检查</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#1d3246;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1480px;margin:auto;padding:32px 24px}header{display:flex;gap:24px;align-items:start;justify-content:space-between}h1{font-size:32px;margin:0}h2{font-size:18px;margin:6px 0}h2 b{float:right;color:#0c7e80;font-size:16px}p{margin:8px 0}.muted{color:#667c8c}.badge{padding:8px 14px;border:1px solid #accacb;border-radius:20px;background:#f6fffd;white-space:nowrap}.notice{border-left:4px solid #c48b36;background:#fff9ee;padding:15px 20px;margin:22px 0}.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:center;background:white;border:1px solid #d6e1e8;padding:15px;border-radius:10px;margin:20px 0}select,input{font:inherit;border:1px solid #b9cbd5;padding:9px;border-radius:6px;background:white}input[type=search]{min-width:260px;flex:1}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}article{background:white;border:1px solid #d6e1e8;border-radius:13px;overflow:hidden}article[hidden]{display:none}.preview{height:210px;background:#102031}.preview img{height:100%;width:100%;object-fit:cover}.body{padding:17px}.eyebrow{font-size:12px;color:#648090;letter-spacing:.03em}.token{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere;color:#65798b}.status{background:#f4f7f9;padding:10px;font-size:13px;min-height:63px}.links{display:flex;flex-wrap:wrap;gap:8px}a{color:#087c81;text-decoration:none}a:hover{text-decoration:underline}.links a{border:1px solid #c7dddd;padding:5px 9px;border-radius:6px;font-size:13px}.links a:first-child{background:#087c81;color:#fff}details{font-size:12px;color:#647885}footer{margin-top:24px;font-size:13px}#empty{padding:30px;text-align:center;background:white}@media(max-width:500px){main{padding:18px 12px}header{display:block}#grid{grid-template-columns:1fr}h1{font-size:26px}.badge{display:inline-block;margin-top:8px}}</style>
<main><header><div><div class="muted">DriveRSI / nuPlan 数据接口</div><h1>密集场景检查</h1><p class="muted">四城市 · CASECOUNT 个已编译窗口 · 按对象数量排序 · 源轨迹与实际执行分开分析</p></div><span class="badge">世界容量 1,024 · 学习槽位上限 32</span></header>
<div class="notice"><b>先看三个超过旧容量的窗口：576、404、325 个对象。</b><br>对象数是整个窗口内出现过的独立 ID，总数不等于同一时刻在场数。源日志入口完整显示编译后的 9 秒插值轨迹；动作诊断使用固定动作 #6（纵向 jerk 0、横向 jerk −4），仅复现接口检查，不能代表驾驶模型或自博弈水平。提前结束不补帧。两者都不是 GS 相机图像或官方 PDM 评分。</div>
FULLLOGS<div class="filters"><label>城市 <select id="city"><option value="">全部城市</option>OPTIONS</select></label><label><input id="dense" type="checkbox">仅超过 256 对象</label><input id="search" type="search" placeholder="搜索场景 ID、token、原始日志" aria-label="搜索场景"><span id="count"></span></div>
<div id="grid">CARDS</div><p id="empty" hidden>当前筛选没有场景。</p><footer><a href="report.json">完整 JSON 证据</a> · <a href="scenes.csv">逐场景 CSV 汇总</a><p>源标注出现／中断不等于物理入场／离场；诊断接触不等于事故责任。保留集仍保持 heldout，浏览与数据接入不授予训练资格。STATUS</p></footer></main>
<script>const $=id=>document.getElementById(id),cards=[...document.querySelectorAll('article')];const q=new URLSearchParams(location.search);$('city').value=q.get('city')||'';$('dense').checked=q.get('dense')==='1';$('search').value=q.get('q')||'';function filter(update=true){let n=0;for(const c of cards){c.hidden=!!($('city').value&&c.dataset.city!==$('city').value)||($('dense').checked&&Number(c.dataset.count)<=256)||!c.dataset.search.toLowerCase().includes($('search').value.trim().toLowerCase());if(!c.hidden)n++;}$('count').textContent=n+' / '+cards.length+' 个场景';$('empty').hidden=!!n;if(update){const url=new URL(location.href);for(const[k,v]of [['city',$('city').value],['dense',$('dense').checked?'1':''],['q',$('search').value]])v?url.searchParams.set(k,v):url.searchParams.delete(k);history.replaceState(null,'',url);}}for(const id of ['city','dense','search'])$(id).addEventListener('input',()=>filter());window.addEventListener('pageshow',()=>filter(false));filter(false);</script></html>"""
    (output / "index.html").write_text(
        page.replace("OPTIONS", cities)
        .replace("CASECOUNT", str(len(report["cases"])))
        .replace(
            "FULLLOGS",
            '<section class="filters"><b>完整原始 DB（全部源帧）</b>'
            + "".join(
                f'<a href="{esc(log["viewer"])}">{esc(CITIES.get(log["map_name"], log["map_name"]))} · {log["duration_seconds"]:.1f}s · {log["frame_count"]}帧</a>'
                for log in report.get("full_logs", [])
            )
            + "</section>"
            if report.get("full_logs")
            else "",
        )
        .replace("CARDS", "".join(cards))
        .replace("STATUS", esc(report["status"]))
    )


def build(manifest_path, output, *, limit=24):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if not 1 <= limit <= 64:
        raise ValueError("Gallery budget is 1..64 scenes")
    source = json.loads(manifest_path.read_text())
    if source.get("source_format") != "nuplan_db" or source.get("state") != "complete":
        raise ValueError("Expected complete nuPlan compilation manifest")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "kind": "nuplan-inspection-gallery",
        "status": "running",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "trainable": False,
        "training_started": False,
        "cases": [],
    }
    for row in [r for r in source["records"] if r["status"] == "compiled"][:limit]:
        path = (manifest_path.parent / row["scene"]).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("Scene reference escapes manifest directory")
        key = Path(row["scene"]).stem
        folder = output / key
        if not folder.is_relative_to(output) or folder.exists():
            raise ValueError("Duplicate or invalid scene ID")
        folder.mkdir()
        case = {
            "id": key,
            "city": row["map_name"],
            "token": row["token"],
            "source_log": row["source_log"],
            "status": "building",
            "role": row["use"],
        }
        report["cases"].append(case)
        try:
            scene = Scene.load(path)
            case.update(
                actors=len(scene.actor_ids),
                controlled=int(scene.controlled.sum()),
                source_seconds=float(scene.times[-1]),
                peak_present=int(scene.presence.sum(axis=1).max()),
                kinds=dict(Counter(scene.metadata["source_actor_kinds"].values())),
                scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            shutil.copyfile(path, folder / "scene.npz")
            case["scene"] = key + "/scene.npz"
            thumbnail(scene, folder / "preview.svg")
            case["thumbnail"] = key + "/preview.svg"
            export_source_replay(
                scene,
                folder / "source.html",
                inspection={
                    "gallery": True,
                    "note": "源日志编译轨迹：10 Hz 插值、完整窗口；不是仿真执行，未计算驾驶评分。标注中断不代表物理消失。",
                },
            )
            case["source_viewer"] = key + "/source.html"
            case["diagnostic"] = diagnostic(scene, folder)
            export_replay(
                scene,
                folder / "diagnostic.npz",
                folder / "diagnostic.html",
                inspection={
                    "gallery": True,
                    "note": "固定动作 #6 的实际物理执行（纵向 jerk 0 / 横向 jerk −4）；仅验证数据接口，不代表驾驶模型或博弈策略。接触、越界及提前结束全部保留，无官方 PDM。",
                },
            )
            case["diagnostic_viewer"] = key + "/diagnostic.html"
            case["diagnostic_sha256"] = hashlib.sha256(
                (folder / "diagnostic.npz").read_bytes()
            ).hexdigest()
            case["status"] = "complete"
        except Exception as error:
            case.update(status="failed", error=f"{type(error).__name__}: {error}")
        case["evidence"] = key + "/evidence.json"
        save(folder / "evidence.json", case)
        save(output / "report.json", report)
        render_index(output, report)
    report["status"] = (
        "complete"
        if report["cases"] and all(r["status"] == "complete" for r in report["cases"])
        else "incomplete"
    )
    fields = [
        "id",
        "city",
        "token",
        "source_log",
        "role",
        "actors",
        "peak_present",
        "controlled",
        "status",
        "source_seconds",
    ]
    with (output / "scenes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])
    save(output / "report.json", report)
    render_index(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    report = build(args.manifest, args.output, limit=args.limit)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["cases"]),
                "failed": [r for r in report["cases"] if r["status"] != "complete"],
            },
            ensure_ascii=False,
        )
    )
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
