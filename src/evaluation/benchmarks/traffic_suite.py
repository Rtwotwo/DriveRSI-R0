"""Deterministic four-city engineering audit, with visible failures and source provenance."""

import argparse
import html
import json
from pathlib import Path

from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.scene import Scene
from driversi.data.participants.background import configure_observed_background
from driversi.data.road_network.lane_graph import compile_lane_graph
from driversi.data.participants.lifecycle import attach_logged_population
from driversi.data.sources.navsim import load_log
from driversi.evaluation.search.traffic import (
    AnchorSearchConfig,
    advance,
    compare,
    crossing_scene,
    population,
    search,
)
from driversi.sim.backends.pufferdrive import PufferDriveWorld


def write_index(output, report):
    rows = []
    for case in report["cases"]:
        key = html.escape(case["id"], quote=True)
        old, new = case["modes"]["following"], case["modes"]["conflict"]
        rows.append(
            f"<tr><td>{html.escape(case['map_name'])}<br>{key}</td><td>{case['controlled']} / {case['world_objects']}</td>"
            f"<td>{old['collision_events']} → {new['collision_events']}</td><td>{old['offroad_events']} → {new['offroad_events']}</td>"
            f"<td>{old['route_progress_m']:.1f} → {new['route_progress_m']:.1f} m</td>"
            f"<td><a href='{key}/following.html'>旧跟驰</a> · <a href='{key}/conflict.html'>冲突控制</a></td></tr>"
        )
    candidates = []
    for group in report.get("searches", []):
        for row in group["attempts"]:
            name = html.escape(row["id"], quote=True)
            folder = html.escape(group["path"], quote=True)
            m = row.get("metrics", {})
            file = (
                name + ".html"
                if row["status"] == "completed"
                else row.get("partial_rollout", "").replace(".npz", ".html")
            )
            link = (
                f"<a href='{folder}/{html.escape(file, quote=True)}'>{name}</a>" if file else name
            )
            candidates.append(
                f"<tr><td>{folder}</td><td>{link}</td>"
                f"<td>{row['anchor_speed_mps']} / {row['delta_speed_mps']:+.1f} m/s</td>"
                f"<td>{row.get('accepted', False)}<br>{html.escape(', '.join(row.get('rejection_reasons', [])))}</td><td>{m.get('collision_events', '—')} / {m.get('offroad_events', '—')}</td>"
                f"<td>{m.get('route_progress_m', 0):.2f} m</td><td>{row.get('max_anchor_deviation_m')}</td></tr>"
            )
    text = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>多车冲突控制与响应式候选搜索</title><style>body{max-width:1300px;margin:30px auto;background:#101b27;color:#e3edf6;font:16px/1.6 system-ui;padding:20px}a{color:#8cddff}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:12px;border:1px solid #435364;text-align:left}p{max-width:1000px}.note{padding:16px;border-left:4px solid #edbd70;background:#263341}summary{cursor:pointer}pre{white-space:pre-wrap;max-height:400px;overflow:auto}</style>
<h1>多车冲突控制与响应式候选搜索</h1>
<p>CARLA 启发：同一世界时刻观察 → 路线冲突与连续包围盒检查 → 生成动作 → 联合推进。INTERACT 启发：不同速度意图 → 有界局部细化；每个候选均从同一快照执行，他车重新响应。</p>
<p class="note">工程基线：优先级是显式协作规则，不代表路权。不是学习型对抗、SMART 预测器或 CEM 复现。没有官方 PDM 评分或训练准入；所有成功和失败均保留。本页为 BEV 交互记录，未提升 GS 画质。</p>
<h2>完整场景对照</h2><p>真实场景仅从已有训练清单按路径顺序选择，路线不支持的案例保留拒绝原因。补齐源日志参与者，并显式记录背景响应与回放角色。</p>
<table><thead><tr><th>场景</th><th>受控 / 全部对象</th><th>碰撞事件：旧→新</th><th>越界事件：旧→新</th><th>主车路线进度</th><th>实际回放</th></tr></thead><tbody>ROWS</tbody></table>
<h2>同状态候选</h2><p>前 1 秒是真实共享历史，随后执行 4 秒候选。预算只计算新后缀。合格仅表示本窗口的几何检查通过；不等于高分驾驶。规划结果和实际执行分开于来源日志。</p>
<table><thead><tr><th>场景</th><th>候选</th><th>锚速度 / 细化</th><th>工程通过</th><th>碰撞 / 越界</th><th>进度</th><th>相对锚轨迹最大偏离(m)</th></tr></thead><tbody>CANDIDATES</tbody></table>
<details><summary>拒绝记录与验收边界</summary><pre>REJECTIONS</pre></details><p><a href="release-checks.json">完整机器可读记录</a></p></html>"""
    text = (
        text.replace("ROWS", "".join(rows))
        .replace("CANDIDATES", "".join(candidates))
        .replace(
            "REJECTIONS", html.escape(json.dumps(report["rejected"], ensure_ascii=False, indent=2))
        )
    )
    (output / "index.html").write_text(text, encoding="utf-8")


def run_suite(suite, dataset_root, maps_root, output, *, trusted=False, candidates_per_city=6):
    if not trusted:
        raise ValueError("Source NAVSIM pickle requires explicit trusted input")
    if not 1 <= candidates_per_city <= 12:
        raise ValueError("Candidate selection is bounded to 1..12 per city")
    rows = json.loads(Path(suite).read_text())["train"]
    for city in sorted({r["city"] for r in rows}):
        if len(list((Path(maps_root) / city).glob("*/map.gpkg"))) != 1:
            raise ValueError(f"Require exactly one map version for {city} under {maps_root}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "version": "traffic-interaction-suite-v1",
        "selection": "first_route_supported_training_scene_per_city_sorted_path",
        "cases": [],
        "rejected": [],
        "searches": [],
        "status": "running",
        "trainable": False,
    }

    def save():
        (output / "release-checks.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        write_index(output, report)

    save()
    scene, rs = crossing_scene()
    c, _ = compare(scene, rs, output / "canonical-crossing")
    c["id"] = "canonical-crossing"
    report["cases"].append(c)
    save()
    rows = json.loads(Path(suite).read_text())["train"]
    for city in sorted({r["city"] for r in rows}):
        candidates = sorted((r for r in rows if r["city"] == city), key=lambda r: r["path"])[
            :candidates_per_city
        ]
        for entry in candidates:
            source_path = Path(entry["path"])
            key = source_path.stem
            try:
                scene = Scene.load(source_path)
                source = (
                    Path(dataset_root)
                    / "navsim_logs"
                    / "mini"
                    / (scene.metadata["source_log"] + ".pkl")
                )
                frames = load_log(source, trusted=True)
                scene = attach_logged_population(scene, frames)
                try:
                    scene, replay_only = configure_observed_background(scene)
                except ValueError as exc:
                    if str(exc) != "No supported background navigation or measured-velocity task":
                        raise
                    replay_only = {"reason": str(exc)}
                graph = compile_lane_graph(scene, Path(maps_root))
                rs, routing = resolve_routes(scene, graph)
                scene = bind_driving_tasks(scene, rs)
            except (ValueError, FileNotFoundError) as exc:
                report["rejected"].append(
                    {"id": key, "city": city, "phase": "source_route_support", "reason": str(exc)}
                )
                save()
                continue
            c, _ = compare(scene, rs, output / key)
            graph.save(output / key / "road_graph.json")
            (output / key / "routing.json").write_text(
                json.dumps(routing, indent=2, allow_nan=False)
            )
            c.update(
                id=key,
                source_path=str(source_path),
                source_log_file=str(source),
                replay_only=replay_only,
                background_reactive=list(
                    scene.metadata.get("reactive_background", {}).get("assignments", {})
                ),
            )
            report["cases"].append(c)
            save()
            actor = "ego" if "ego" in rs else next(iter(rs))
            try:
                with PufferDriveWorld(scene) as world:
                    prefix = advance(world, population(rs), 10)
                    result, _ = search(
                        world,
                        rs,
                        actor,
                        output / (key + "-search"),
                        AnchorSearchConfig(max_wall_seconds=180.0),
                        prefix=prefix,
                    )
                result["path"] = key + "-search"
                report["searches"].append(result)
            except ValueError as exc:
                report["rejected"].append(
                    {"id": key, "city": city, "phase": "search_support", "reason": str(exc)}
                )
            save()
            break
    report["covered_cities"] = sorted(
        {c["map_name"] for c in report["cases"] if c["id"] != "canonical-crossing"}
    )
    report["status"] = (
        "completed"
        if len(report["covered_cities"]) == len({r["city"] for r in rows})
        else "coverage_incomplete"
    )
    report["restore_exact"] = all(c["restore_exact"] for c in report["cases"])
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-pickle", action="store_true")
    args = parser.parse_args()
    report = run_suite(
        args.suite, args.dataset_root, args.maps_root, args.output, trusted=args.trusted_pickle
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "cities": report["covered_cities"],
                "cases": len(report["cases"]),
                "rejected": report["rejected"],
            }
        )
    )


if __name__ == "__main__":
    main()
