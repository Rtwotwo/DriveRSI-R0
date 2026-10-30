"""Full vehicle ownership audit over a frozen real-source traffic release."""

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np

from driversi.contracts.driving_task import tasks_from_metadata
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import VEHICLE, Scene
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.evaluation.benchmarks.interaction import _exact
from driversi.evaluation.search.traffic import advance, metrics, population
from driversi.rendering.bev.replay import export_replay
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import write_rollout


def run(source_release, output, *, baseline_fleet_release=None):
    source_release, output = Path(source_release), Path(output)
    source = json.loads((source_release / "release-checks.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "version": "vehicle-fleet-audit-v2",
        "baseline_fleet_release": str(baseline_fleet_release) if baseline_fleet_release else None,
        "cases": [],
        "status": "running",
        "trainable": False,
        "scope": "fixed four-city mini source scenes; all traffic ownership, not all learned agents",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        rows = []
        for case in report["cases"]:
            key = html.escape(case["id"])
            counts = case["fleet"]["counts"]
            roles = " / ".join(f"{html.escape(k)}: {v}" for k, v in counts.items())
            rows.append(
                f"<tr><td>{html.escape(case['city'])}<br>{key}</td>"
                f"<td>{case['before_feedback_vehicles']} → {case['assigned_vehicles']} / {case['window_vehicles']}</td>"
                f"<td>{case['policy_action_vehicles']}</td><td>{roles}</td>"
                f"<td>{case['executed_traffic_controls']}</td><td>{case['restore_exact']}</td>"
                f"<td>{case['modes']['before']['owner_contact_ticks']} → {case['modes']['fleet']['owner_contact_ticks']}</td>"
                f"<td>{case['modes']['before']['other_contact_ticks']} → {case['modes']['fleet']['other_contact_ticks']}</td>"
                f"<td>{html.escape(str(case['modes']['fleet']['contact_evidence']['onset_counts']))}</td>"
                f"<td><a href='{key}/before.html'>之前</a> · <a href='{key}/fleet.html'>全车辆控制</a> · <a href='{key}/coverage.json'>逐车依据</a></td></tr>"
            )
        page = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>全车辆交通控制验收</title>
<style>body{background:#101b27;color:#e6edf5;font:16px/1.6 system-ui;max-width:1450px;margin:25px auto;padding:20px}a{color:#87d7ff}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #4c6071}.note{padding:16px;background:#283643;border-left:4px solid #efb761}code{color:#d2dca4}</style>
<h1>车辆控制修复与碰撞归属对照</h1><p class="note">自车与其他对象的接触独立统计；这些是工程几何证据，不是官方PDM，也不判定事故责任。初态或入场重叠仍保留，不通过删除车辆、瞬移或降低验收门槛消除失败。新版按当前速度预测冲突，路线耗尽后保持朝向制动，不转回身后的终点。</p><p>所有窗口内车辆均已分配反馈、停车保持或不确定运动制动控制；仅在入场后的有效物理步执行，末帧新入场只有初态。按首次标注入场，接管后缺标不再导致消失。行人和其他物体继续保留；窗口外车辆不提前生成。</p>
<p class="note">全车辆接管不等于全部学习型智能体。策略动作接口仍保留原有学习槽位；交通车辆使用有界运动学执行器，停车车不自动启动，朝向/路线不足时保留制动诊断。路线是离线特权任务，尚未验证交通路权、PDM 或 GS 训练质量。</p>
<p>控制记录保存动作前 tick、实际目标状态与下一步世界状态，回放逐项校验。每辆车可在选择菜单查看模式。完整保留失败证据，不据此宣称高分或成功博弈。碰撞／越界事件指标仍只覆盖策略槽位；另报涉及全部车辆的几何接触、路线偏差与实际执行覆盖，不代表完整道路合法性。</p>
<table><tr><th>场景</th><th>反馈车辆：之前→现在 / 窗口内</th><th>策略动作车辆</th><th>角色计数</th><th>实际交通控制条数</th><th>恢复一致</th><th>自车接触帧：前→后</th><th>其他接触帧：前→后</th><th>首次接触分类：新版本</th><th>查看</th></tr>ROWS</table><p><a href="report.json">完整指标、执行记录统计与局限</a></p>"""
        (output / "index.html").write_text(page.replace("ROWS", "".join(rows)), encoding="utf-8")

    save()
    for entry in source["cases"]:
        if entry["id"] == "canonical-crossing":
            continue
        key = entry["id"]
        folder = output / key
        folder.mkdir()
        original = Scene.load(source_release / key / "scene.npz")
        if baseline_fleet_release is not None:
            original = Scene.load(Path(baseline_fleet_release) / key / "fleet-scene.npz")
        fleet = configure_vehicle_fleet(original)
        ids = tuple(a for a, active in zip(original.actor_ids, original.controlled) if active)
        routes = {a: value[0] for a, value in tasks_from_metadata(original.metadata, ids).items()}
        controller = population(routes)
        actor = "ego" if "ego" in ids else ids[0]
        graph = RoadGraph.load(source_release / key / "road_graph.json")
        active = {
            a
            for i, a in enumerate(original.actor_ids)
            if original.kinds[i] == VEHICLE
            and any(original.presence_at(t)[i] for t in range(len(original.times)))
        }
        before_tasks = original.metadata.get("reactive_background", {}).get("assignments", {})
        counts = fleet.metadata["vehicle_fleet"]["counts"]
        case = {
            "id": key,
            "city": original.metadata["map_name"],
            "window_vehicles": len(active),
            "before_feedback_vehicles": len(active & (set(ids) | set(before_tasks))),
            "assigned_vehicles": sum(v for k, v in counts.items() if k != "outside_window"),
            "policy_action_vehicles": len(ids),
            "fleet": fleet.metadata["vehicle_fleet"],
            "metrics_scope": {
                "modes": "collision/offroad/modified commands: policy slots; route progress/error: selected owner; background contact ticks: full world",
                "fleet_contact_ticks": "ticks with any contact involving a vehicle, including initial overlaps",
                "fleet_contact_pairs": "unique world-object pairs involving any vehicle; no attribution of responsibility",
                "traffic_route_support": "maximum distance to assigned traffic route, not road/lane legality",
                "executed_traffic_controls": "all traffic tasks, including pre-existing pedestrian controllers",
                "vehicles_with_executed_control": "vehicles with at least one valid policy action or actual traffic proposal",
            },
            "modes": {},
        }
        for name, scene in (("before", original), ("fleet", fleet)):
            scene.save(folder / (name + "-scene.npz"))
            started = time.monotonic()
            with PufferDriveWorld(scene) as world:
                snapshot = world.snapshot()
                record = advance(world, controller, world.max_steps)
                case["modes"][name] = metrics(scene, record, routes, actor)
                case["modes"][name]["wall_seconds"] = time.monotonic() - started
                if name == "fleet":
                    traffic_tasks = scene.metadata.get("reactive_background", {}).get(
                        "assignments", {}
                    )
                    vehicle_ids = set(scene.metadata["vehicle_fleet"]["roles"])
                    case["fleet_contact_ticks"] = sum(
                        any(set(pair) & vehicle_ids for pair in frame.get("contacts", []))
                        for frame in record.lifecycle
                    )
                    case["fleet_contact_pairs"] = sorted(
                        {
                            tuple(sorted(pair))
                            for frame in record.lifecycle
                            for pair in frame.get("contacts", [])
                            if set(pair) & vehicle_ids
                        }
                    )
                    case["vehicles_with_executed_control"] = sorted(
                        {
                            d["actor"]
                            for tr in record.transitions
                            for d in tr.traffic_controls
                            if d["actor"] in vehicle_ids
                        }
                        | {
                            a
                            for i, a in enumerate(ids)
                            if any(tr.valid[i] for tr in record.transitions)
                        }
                    )
                    case["assigned_without_executed_step"] = sorted(
                        active - set(case["vehicles_with_executed_control"])
                    )
                    case["traffic_route_support"] = {}
                    for a, task in traffic_tasks.items():
                        if a not in vehicle_ids or task["desired_speed"] <= 0:
                            continue
                        from driversi.contracts.road import Route

                        route = Route(task["points"])
                        index = scene.actor_ids.index(a)
                        distances = [
                            route.project(state[index, :2])[1]
                            for state, present in zip(record.states, record.presence)
                            if present[index]
                        ]
                        case["traffic_route_support"][a] = {
                            "max_route_error_m": float(max(distances)) if distances else None,
                            "has_executed_control": a in case["vehicles_with_executed_control"],
                        }
                    case["executed_traffic_controls"] = sum(
                        len(t.traffic_controls) for t in record.transitions
                    )
                    case["executed_by_actor"] = {
                        a: sum(
                            any(c["actor"] == a for c in t.traffic_controls)
                            for t in record.transitions
                        )
                        for a in scene.metadata.get("reactive_background", {}).get(
                            "assignments", {}
                        )
                    }
                    # Re-run a nonzero snapshot suffix, including exact control records.
                    world.restore(snapshot)
                    prefix = advance(world, controller, min(10, world.max_steps))
                    saved = world.snapshot()
                    left = advance(world, controller, world.max_steps - world.observe().step)
                    world.restore(saved)
                    right = advance(world, controller, world.max_steps - world.observe().step)
                    case["restore_exact"] = _exact(left, right) and all(
                        a.traffic_controls == b.traffic_controls
                        for a, b in zip(left.transitions, right.transitions)
                    )
                    case["prefix_replay_exact"] = np.array_equal(
                        np.asarray(record.states), np.asarray(prefix.followed_by(left).states)
                    )
                write_rollout(
                    scene,
                    folder / (name + ".npz"),
                    record=record,
                    snapshot=snapshot,
                    controlled_ids=ids,
                    policy="route-conflict-v1",
                    controller_configuration=controller.describe(),
                    protocol={"kind": "vehicle-fleet-audit-v1", "mode": name, "trainable": False},
                )
            export_replay(
                scene, folder / (name + ".npz"), folder / (name + ".html"), road_graph=graph
            )
        (folder / "coverage.json").write_text(json.dumps(case, indent=2, allow_nan=False))
        report["cases"].append(case)
        save()
    report["status"] = "completed"
    report["all_window_vehicles_assigned"] = all(
        c["assigned_vehicles"] == c["window_vehicles"] for c in report["cases"]
    )
    report["restore_exact"] = all(
        c["restore_exact"] and c["prefix_replay_exact"] for c in report["cases"]
    )
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-fleet-release", type=Path)
    args = parser.parse_args()
    result = run(
        args.source_release, args.output, baseline_fleet_release=args.baseline_fleet_release
    )
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
