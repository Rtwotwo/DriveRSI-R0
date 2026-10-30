"""Native DriveRL parity and real long-log execution; no training or admission."""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from driversi.agents.backends.sources import require_source
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.evaluation.diagnosis.anchor_diagnosis import record_evidence
from driversi.rendering.bev.replay import export_replay, export_source_replay
from driversi.rendering.inspection.driverl import export_driverl_gallery
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi_agents.driverl.closed_loop import DriveRLSession, ego_vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model-config", type=Path, default=Path("src/config/agents/driverl/closed_loop.yaml")
    )
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--device")
    p.add_argument("--search-candidates", type=int, default=0)
    p.add_argument("--search-start", type=int)
    p.add_argument("--search-stop", type=int)
    p.add_argument("--repair-backend", choices=["pdm_route_v1"])
    p.add_argument("--repair-start", type=int)
    p.add_argument("--repair-stop", type=int)
    a = p.parse_args()
    if not 1 <= a.steps <= 6000:
        raise ValueError("bounded steps required")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ["scene.npz", "source.npz", "road-graph.json"]:
        if (a.source / name).exists():
            shutil.copy2(a.source / name, a.output / name)
    scene = Scene.load(a.output / "scene.npz")
    cfg = yaml.safe_load(a.model_config.read_text())
    if (
        cfg["schema_version"] != 1
        or cfg["runtime"] != "driverl_continuous_v1"
        or cfg["policy_scope"] != "ego"
        or cfg["train"] is not False
    ):
        raise ValueError("Unsupported native control deployment")
    s = DriveRLSession(
        scene,
        repository=root / cfg["repository"],
        checkpoint=root / cfg["checkpoint"],
        maps_root=cfg["maps_root"],
        device=a.device or cfg["device"],
        parity=True,
        search_candidates=a.search_candidates, search_start=a.search_start, search_stop=a.search_stop,
        repair_backend=a.repair_backend, repair_start=a.repair_start, repair_stop=a.repair_stop,
    )
    if (
        max(s.initial_conversion["max_pose_error"], s.initial_conversion["max_velocity_error"])
        > 5e-5
    ):
        s.close()
        raise AssertionError("Official initial state exceeds declared float32 tolerance")
    scene = s.scene
    scene.save(a.output / "scene.npz")
    t0 = time.monotonic()
    record = RolloutRecord.start(s.world)
    restore_tick = a.repair_start + 2 if a.repair_backend else (a.search_start + 2 if a.search_candidates else 90)
    snap = None
    truth = []
    reports = []
    native_history = []
    try:
        for tick in range(min(a.steps, len(scene.times) - 1)):
            if s.world.episode_done:
                break
            if tick == restore_tick:
                snap = s.snapshot()
            native_input = s.planner_input()
            native_history.append(
                dict(
                    ego=ego_vector(s.ego),
                    objects=s._objects(),
                    lights=[
                        dict(
                            status=x.status.name,
                            lane_connector_id=x.lane_connector_id,
                            timestamp=x.timestamp,
                        )
                        for x in native_input.traffic_light_data
                    ],
                )
            )
            tr = s.step()
            record.append(
                s.world, s.last_decision["executed_actions"], tr, s.last_decision["diagnostics"]
            )
            reports.append(s.parity[-1])
            if restore_tick <= tick < restore_tick + 20:
                truth.append(
                    (
                        s.world.state(),
                        s.world.motion(),
                        s.world.presence(),
                        ego_vector(s.ego),
                        copy.deepcopy(s.last_decision),
                    )
                )
            if (tick + 1) % 30 == 0:
                print(
                    json.dumps(
                        dict(
                            tick=tick + 1,
                            alive=bool(tr.observation.alive[0]),
                            contacts=s.world.contacts(),
                            wall=round(time.monotonic() - t0, 2),
                        )
                    ),
                    flush=True,
                )
        (a.output / "native-history.json").write_text(json.dumps(native_history, allow_nan=False))
        final = s.snapshot()
        restore = False
        if snap is not None:
            s.restore(snap)
            stored = snap["runtime"]["controller_state"]
            if stored is not None:
                np.testing.assert_array_equal(ego_vector(s.controller.get_state()), stored)
            for expected in truth:
                s.step()
                for x, y in zip(
                    (s.world.state(), s.world.motion(), s.world.presence(), ego_vector(s.ego)),
                    expected[:4],
                ):
                    np.testing.assert_array_equal(x, y)
                assert s.last_decision == expected[4]
            restore = True
            s.restore(final)
        evidence = record_evidence(scene, record, "ego")
        report = dict(
            initial_conversion=s.initial_conversion,
            source_input_parity=s.source_parity,
            steps=len(record.actions),
            seconds=len(record.actions) * 0.1,
            inference_steps=sum(v["inference"] for v in reports),
            correction_search=s.search,
            correction_backend=s.repair,
            repair_steps=sum(bool(v.get("repair_active")) for v in reports),
            native_executed_steps=sum(bool(v["inference"]) and not v.get("repair_active", False) for v in reports),
            search_steps=sum(v["native_search_active"] for v in reports),
            parity_steps=sum("native_output_max_abs_error" in v for v in reports),
            native_output_max_abs_error=max(
                (v.get("native_output_max_abs_error", 0.0) for v in reports), default=0.0
            ),
            native_feature_values_compared=sum(
                v.get("native_feature_values_compared", 0) for v in reports
            ),
            maximum_world_position_rounding_m=max(
                (v.get("world_position_rounding_m", 0.0) for v in reports), default=0.0
            ),
            restore_exact=restore,
            restore_steps=len(truth),
            restore_tick=restore_tick,
            wall_seconds=time.monotonic() - t0,
            contacts=evidence["contacts"],
            road_boundary_policy=scene.metadata.get("road_boundary_policy", "perimeter_stop_legacy"),
            road_boundary_contact_steps=(sum(bool(tr.events["road_boundary_contact"][0]) for tr in record.transitions)
                if record.transitions and all("road_boundary_contact" in tr.events for tr in record.transitions) else None),
            configuration=s.configuration,
            terminal_events={
                key: [
                    i + 1
                    for i, tr in enumerate(record.transitions)
                    if tr.events.get(key, np.zeros(1, bool))[0]
                ]
                for key in ("collision", "offroad", "goal", "route_support_end")
            },
            upstream_commit=require_source(root / "agents/driverl/native")["actual"],
            trainable=False,
            official_pdm=None,
            benchmark_parity_established=False,
            limitations=[
                "same-input native parity does not equate reactive traffic or a full benchmark",
                "source routing is privileged navigation; current light phase read causally; traffic controllers lack full right-of-way",
                "world float32 storage; explicit correction, if enabled, is a separate policy and native output is shadow only",
            ],
        )
        for name, v in [
            ("report.json", report),
            ("decisions.json", reports),
            ("execution-evidence.json", evidence),
        ]:
            (a.output / name).write_text(json.dumps(v, indent=2, allow_nan=False))
        protocol = dict(
            kind="driverl_harness_repair" if s.repair else "driverl_native",
            title="Harness 地图约束纠正（非 DriveRL 原生基准）" if s.repair else "DriveRL 官方权重：连续闭环实测",
            gallery=True,
            note=("显式切换 Harness-PDM 纠正；纠正期间 DriveRL 仅作影子对照，不是原生成绩。其他对象继续响应，碰撞与道路检查保留；不予训练准入。" if s.repair else "自车使用官方 DriveRL 原生观测与连续控制；其他对象响应式交通。碰撞后按平台终止语义停车，无规则接替。非官方PDM、非训练准入。"),
            paired_view={"path": "source.html", "label": "查看源日志（非反事实真值）"},
            trainable=False,
        )
        write_rollout(
            scene,
            a.output / "rollout.npz",
            record=record,
            snapshot=s.initial,
            controlled_ids=s.world.actor_ids,
            policy="DriveRL-with-explicit-Harness-repair" if s.repair else "DriveRL-official-native",
            controller_configuration=s.configuration,
            protocol=protocol,
        )
        graph = (
            RoadGraph.load(a.output / "road-graph.json")
            if (a.output / "road-graph.json").exists()
            else None
        )
        export_replay(
            scene,
            a.output / "rollout.npz",
            a.output / "viewer.html",
            inspection=protocol,
            road_graph=graph,
            compress_payload=True,
        )
        export_source_replay(
            Scene.load(a.output / "source.npz"),
            a.output / "source.html",
            inspection={"title": "nuPlan 源日志；非控制器仿真"},
        )
        from driversi.evaluation.metrics.nuplan_metrics import score_record

        try:
            official = score_record(
                scene,
                native_history,
                a.output / "official-metrics",
                root / "agents/driverl/native/DriveRL",
                cfg["maps_root"],
                planner_name="DriveRL+Harness-PDM-route-repair" if s.repair else "DriveRL",
                execution_policy={"native": "DriveRL", "correction": s.repair},
            )
        except Exception as error:
            official = dict(status="failed", error=repr(error), score=None, trainable=False)
        (a.output / "official-score.json").write_text(
            json.dumps(official, indent=2, allow_nan=False)
        )
        export_driverl_gallery(a.output)
        print(json.dumps({k: v for k, v in report.items() if k != "configuration"}), flush=True)
    finally:
        s.close()


if __name__ == "__main__":
    main()
