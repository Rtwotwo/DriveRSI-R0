"""Ego-first complex source inspection and rule-interaction replay.

Selection uses source geometry proxies, never evaluation scores or held-out logs.
Source photographs remain separate from counterfactual simulation observations.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.scene import Scene
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.data.road_network.lane_graph import compile_lane_graph
from driversi.data.participants.lifecycle import attach_logged_population
from driversi.data.sources.navsim import load_log, yaw_from_quaternion
from driversi.data.scenes.source_gallery import _scene, select_source_window
from driversi.evaluation.benchmarks.interaction import _exact
from driversi.evaluation.search.traffic import advance, metrics, population
from driversi.rendering.bev.replay import export_replay
from driversi.rendering.visual.source_gallery import export_source_gallery
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import write_rollout


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def complexity(window):
    """Observed proximity/density/ego-turn proxies, not risk or legal right of way."""
    counts, near, peds, crossing = [], [], [], []
    for f in window:
        names = np.asarray(f["anns"]["gt_names"])
        boxes = np.asarray(f["anns"]["gt_boxes"], dtype=float).reshape(-1, 7)
        radius = np.linalg.norm(boxes[:, :2], axis=1)
        vehicles = names == "vehicle"
        counts.append(int(np.sum(vehicles & (radius <= 40))))
        near.append(int(np.sum(vehicles & (radius <= 15))))
        peds.append(int(np.sum(np.isin(names, ["pedestrian", "bicycle"]) & (radius <= 30))))
        crossing.append(
            int(np.sum(vehicles & (radius <= 30) & (np.abs(np.sin(boxes[:, 6])) > 0.6)))
        )
    headings = np.unwrap([yaw_from_quaternion(f["ego2global_rotation"]) for f in window])
    return {
        "peak_vehicles_40m": max(counts),
        "peak_vehicles_15m": max(near),
        "peak_vulnerable_30m": max(peds),
        "peak_transverse_vehicles_30m": max(crossing),
        "ego_heading_span_degrees": float(np.degrees(np.ptp(headings))),
        "scope": "source-window proxies; parked vehicles included; not validated conflict labels",
    }


def select_diverse(rows, per_city=2):
    if type(per_city) is not int or not 1 <= per_city <= 4:
        raise ValueError("Require 1..4 scenes per city")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Scene IDs must be unique")
    chosen, used = [], set()
    for city in sorted({row["city"] for row in rows}):
        for slot in range(per_city):
            eligible = [r for r in rows if r["city"] == city and r["source_group"] not in used]
            if not eligible:
                break

            def key(row):
                c = row["complexity"]
                priority = (
                    (c["peak_vehicles_40m"], c["peak_vehicles_15m"], c["peak_vulnerable_30m"])
                    if slot % 2 == 0
                    else (
                        c["ego_heading_span_degrees"],
                        c["peak_transverse_vehicles_30m"],
                        c["peak_vehicles_15m"],
                    )
                )
                return (*(-float(x) for x in priority), row["id"])

            row = dict(min(eligible, key=key))
            row["selection_reason"] = (
                "dense-nearby-traffic" if slot % 2 == 0 else "ego-turn-and-transverse-traffic"
            )
            chosen.append(row)
            used.add(row["source_group"])
    return chosen


def index(output, report):
    cards = []
    for row in report["cases"]:
        q = html.escape
        key = q(row["id"], quote=True)
        c = row["complexity"]
        links = []
        for label, link in row.get("links", {}).items():
            links.append(f'<a href="{q(link, quote=True)}">{q(label)}</a>')
        thumbnail = row.get("thumbnail")
        picture = (
            f'<img loading="lazy" src="{q(thumbnail, quote=True)}" alt="原日志自车前相机；不是交互渲染">'
            if thumbnail
            else ""
        )
        m = row.get("metrics", {})
        progress = "—" if "route_progress_m" not in m else f"{m['route_progress_m']:.1f}m"
        if m.get("route_progress_supported") is False:
            progress += "（存在路线支持不足，仅累计受支持段）"
        cards.append(f"""<article data-city="{q(row["city"], quote=True)}"><h2>{q(row["city"])} · {key}</h2>{picture}
<p>真实相机日志预览（不是新轨迹图像）</p><p>40m 内车辆峰值 {c["peak_vehicles_40m"]}；15m 内 {c["peak_vehicles_15m"]}；30m 内行人/骑行者 {c["peak_vulnerable_30m"]}；自车航向跨度 {c["ego_heading_span_degrees"]:.1f}°</p>
<p>状态：{q(row["status"])}；策略槽位 {row.get("controlled", "—")}；世界对象 {row.get("objects", "—")}</p>
<p>自车接触帧 {m.get("owner_contact_ticks", "—")}；其他对象接触帧 {m.get("other_contact_ticks", "—")}；沿路线进度 {progress}</p>
<p>{" · ".join(links)}</p><details><summary>来源、选择依据与失败记录</summary><pre>{q(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></article>""")
    gs = report.get("gs", [])
    gsrows = "".join(
        f"<li>{html.escape(g['label'])}：{html.escape(g['status'])} "
        + (
            f'<a href="{html.escape(g["viewer"], quote=True)}">自车多视角 GS 交互</a>'
            if g.get("viewer")
            else ""
        )
        + f" <span>{html.escape(g.get('reason', ''))}</span></li>"
        for g in gs
    )
    cities = sorted({row["city"] for row in report["cases"]})
    options = "".join(f"<option>{html.escape(c)}</option>" for c in cities)
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>复杂场景 · 自车交互检查</title><style>body{max-width:1480px;margin:28px auto;padding:22px;background:#0c1521;color:#e1e9f2;font:16px/1.6 system-ui}a{color:#88d7ff}h1{font-size:30px}h2{font-size:19px}article{background:#182638;border:1px solid #35485d;padding:18px;border-radius:12px}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#070c12}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto}.note{padding:15px;background:#293544;border-left:4px solid #e1af60}select{font:inherit;padding:8px;margin-bottom:20px}article[hidden]{display:none}@media(max-width:500px){#grid{grid-template-columns:1fr}}</style>
<h1>复杂场景 · 以 ego 为主的交互检查</h1><p class="note">真实照片用于查看场景原貌；BEV 是全车辆接管后的实际仿真。二者不是同一反事实分支，不同步冒充。GS 单列，仅展示具有精确资产绑定的日志。密度、横向朝向、转弯是筛选线索，不代表已验证的高风险等级。</p>
<h2>自车多视角 GS 实际交互</h2><p>现有可编辑 GS 仍只覆盖一个 Boston 区域；不同采集日志不是不同城市。其他车辆机位为辅助诊断，覆盖不足不通过补色或裁剪伪装修复。</p><ul>GSROWS</ul>
<h2>四城市复杂场景 · 原相机与反应式 BEV</h2><p>训练来源日志内筛选；每城市优先一组高密度、一组明显转弯，来自不同源日志组。失败与提前结束均保留。背景路线含日志未来信息，只用于离线特权交通意图。规则交通而非学习型对抗；没有官方 PDM，trainable=false。</p><label>城市 <select id="city"><option value="">全部城市</option>OPTIONS</select></label>
<div id="grid">CARDS</div><p><a href="report.json">执行、恢复与筛选证据</a> · <a href="selection.json">全部候选筛选记录</a></p>
<script>const city=document.querySelector('#city');city.onchange=()=>document.querySelectorAll('article').forEach(a=>a.hidden=!!city.value&&a.dataset.city!==city.value);</script></html>"""
    (Path(output) / "index.html").write_text(
        page.replace("GSROWS", gsrows or "<li>资产审计中；尚未发布新渲染。</li>")
        .replace("OPTIONS", options)
        .replace("CARDS", "".join(cards))
    )


def build(suite, dataset, output, *, per_city=2, trusted=False):
    if not trusted:
        raise ValueError("Trusted source pickle authorization required")
    dataset, output = Path(dataset), Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("Do not overwrite an existing report")
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(suite).read_text())
    heldout = {r["source_group"] for r in source.get("evaluation", [])}
    if heldout & {r["source_group"] for r in source["train"]}:
        raise ValueError("Training and evaluation source groups overlap")
    candidates, rejected = [], []
    cached_log, frames = None, None
    for entry in sorted(source["train"], key=lambda r: r["path"]):
        scene = Scene.load(entry["path"])
        m = scene.metadata
        path = dataset / "navsim_logs/mini" / (m["source_log"] + ".pkl")
        if path != cached_log:
            frames, cached_log = load_log(path, trusted=True), path
        try:
            window = select_source_window(frames, m["source_start_frame"], m["source_frames"])
            candidates.append(
                {
                    **entry,
                    "id": Path(entry["path"]).stem,
                    "source_log": m["source_log"],
                    "start_frame": m["source_start_frame"],
                    "frames": m["source_frames"],
                    "complexity": complexity(window),
                }
            )
        except ValueError as exc:
            rejected.append({"id": Path(entry["path"]).stem, "reason": str(exc)})
    selected = select_diverse(candidates, per_city)
    requested_count = per_city * len({r["city"] for r in source["train"]})
    save_json(
        output / "selection.json",
        {
            "candidates": candidates,
            "selected": selected,
            "rejected": rejected,
            "split": "train_only",
            "per_city": per_city,
        },
    )
    report = {
        "kind": "complex-ego-inspection-v1",
        "requested_cases": requested_count,
        "selected_cases": len(selected),
        "status": "running",
        "trainable": False,
        "cases": [],
        "gs": [],
    }

    def save():
        save_json(output / "report.json", report)
        index(output, report)

    save()
    for row in selected:
        row = dict(row, status="building", links={})
        report["cases"].append(row)
        folder = output / row["id"]
        folder.mkdir()
        try:
            photos = _scene(
                dict(
                    id=row["id"],
                    log=row["source_log"] + ".pkl",
                    start_frame=row["start_frame"],
                    frames=row["frames"],
                ),
                dataset,
                folder,
                trusted=True,
            )
            bundle = {
                "schema_version": 1,
                "kind": "source_camera_gallery_not_simulation",
                "scenes": [photos],
            }
            save_json(folder / "source-gallery.json", bundle)
            export_source_gallery(bundle, folder / "source.html")
            row["links"]["八路真实相机日志"] = row["id"] + "/source.html"
            first = next(
                (
                    c
                    for c in photos["frames"][0]["cameras"]
                    if c["id"] == "CAM_F0" and c["available"]
                ),
                None,
            )
            if first:
                row["thumbnail"] = row["id"] + "/" + first["image"]
            scene = Scene.load(row["path"])
            frames = load_log(
                dataset / "navsim_logs/mini" / (row["source_log"] + ".pkl"), trusted=True
            )
            scene = attach_logged_population(scene, frames)
            graph = compile_lane_graph(scene, dataset / "maps")
            routes, routing = resolve_routes(scene, graph)
            scene = configure_vehicle_fleet(bind_driving_tasks(scene, routes))
            scene.save(folder / "scene.npz")
            graph.save(folder / "road_graph.json")
            save_json(folder / "routing.json", routing)
            row.update(
                controlled=int(scene.controlled.sum()),
                objects=len(scene.actor_ids),
                fleet=scene.metadata["vehicle_fleet"]["counts"],
            )
            controller = population(routes)
            with PufferDriveWorld(scene) as world:
                initial = world.snapshot()
                record = advance(world, controller, world.max_steps)
                row["metrics"] = metrics(scene, record, routes, "ego")
                row["executed_steps"] = len(record.transitions)
                world.restore(initial)
                prefix = advance(world, controller, min(10, world.max_steps))
                saved = world.snapshot()
                left = advance(world, controller, world.max_steps - world.observe().step)
                world.restore(saved)
                right = advance(world, controller, world.max_steps - world.observe().step)
                row["restore_exact"] = (
                    _exact(left, right)
                    and np.array_equal(left.presence, right.presence)
                    and left.lifecycle == right.lifecycle
                ) and all(
                    a.traffic_controls == b.traffic_controls
                    for a, b in zip(left.transitions, right.transitions)
                )
                row["replay_exact_scope"] = "full_initial_replay_state_arrays_only"
                row["restore_exact_scope"] = (
                    "suffix_state_motion_actions_events_traffic_controls_presence_lifecycle"
                )
                row["replay_exact"] = np.array_equal(
                    np.asarray(record.states), np.asarray(prefix.followed_by(left).states)
                )
                write_rollout(
                    scene,
                    folder / "interaction.npz",
                    record=record,
                    snapshot=initial,
                    controlled_ids=tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c),
                    policy="route-conflict-v1",
                    controller_configuration=controller.describe(),
                    protocol={"kind": "complex-ego-inspection-v1", "trainable": False},
                )
            export_replay(
                scene, folder / "interaction.npz", folder / "interaction.html", road_graph=graph
            )
            row["links"]["全车辆交互 BEV"] = row["id"] + "/interaction.html"
            row["status"] = (
                "completed"
                if row["restore_exact"] and row["replay_exact"]
                else "consistency_failed"
            )
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        save()
    report["status"] = (
        "completed"
        if report["cases"]
        and len(report["cases"]) == requested_count
        and all(c["status"] == "completed" for c in report["cases"])
        else "partial_failure"
    )
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-city", type=int, default=2)
    p.add_argument("--trusted-pickle", action="store_true")
    a = p.parse_args()
    result = build(a.suite, a.dataset, a.output, per_city=a.per_city, trusted=a.trusted_pickle)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": [
                    {k: c.get(k) for k in ("id", "status", "reason")} for c in result["cases"]
                ],
            }
        )
    )

    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
