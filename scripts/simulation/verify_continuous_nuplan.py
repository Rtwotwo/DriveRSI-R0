"""Bounded continuous nuPlan execution and same-anchor diagnostics; no model training."""

import argparse
import copy
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.traffic import TrafficController
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import RoadGraph, Route
from driversi.contracts.scene import PEDESTRIAN, Scene
from driversi.data.participants.background import configure_background, propose_logged_tasks
from driversi.data.participants.fleet import configure_vehicle_fleet
from driversi.data.road_network.maps import attach_nuplan_map
from driversi.data.sources.nuplan_db import compile_window, read_window
from driversi.evaluation.diagnosis.anchor_diagnosis import diagnose_anchor, record_evidence
from driversi.evaluation.metrics.source_support import source_entry_overlaps
from driversi.rendering.bev.replay import export_replay, export_source_replay
from driversi.sim.execution.continuous import ContinuousSession, ControllerPolicy
from driversi.sim.persistence.recording import RolloutRecord, write_rollout


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare(manifest, output, maps_root, seconds):
    output.mkdir(parents=True, exist_ok=False)
    src = json.loads(manifest.read_text())
    conn = sqlite3.connect(f"file:{src['db_path']}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT token,timestamp FROM lidar_pc ORDER BY timestamp").fetchall()
    finally:
        conn.close()
    anchor = next(row for row in rows if row[1] - rows[0][1] >= 1_500_000)
    requested = min(seconds, (rows[-1][1] - anchor[1]) / 1e6)
    # Explicit complete 10 Hz intervals, never relabel the full source log duration.
    duration = np.floor(requested * 10) / 10
    window = read_window(src["db_path"], token=anchor[0].hex(), duration=float(duration))
    source = compile_window(window, duration=float(duration), max_actors=1, split="unassigned")
    attach_nuplan_map(source, maps_root, margin=80)
    source.save(output / "source.npz")
    tasks, rejected = propose_logged_tasks(source)
    peds = {a: t for a, t in tasks.items() if source.kinds[source.actor_ids.index(a)] == PEDESTRIAN}
    scene = configure_vehicle_fleet(configure_background(source, peds) if peds else source)
    i = scene.actor_ids.index("ego")
    points = source.poses[:, i, :2]
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]]
    route = Route(points, end_reason="source_support_end")
    scene = bind_driving_tasks(scene, {"ego": route})
    scene.metadata["continuous_validation"] = {
        "navigation": "privileged_logged_path_not_online_route_prediction",
        "source_log_duration_seconds": src["duration_seconds"],
        "requested_seconds": seconds,
        "actual_seconds": float(scene.times[-1]),
        "raw_anchor_token": anchor[0].hex(),
        "source_support_is_not_physical_exit": True,
        "nonreactive_reasons": {
            a: reason
            for a, reason in rejected.items()
            if a not in scene.metadata["reactive_background"]["assignments"]
        },
    }
    scene.save(output / "scene.npz")
    save(
        output / "preparation.json",
        {
            "source_log": src["source_log"],
            "city": src["map_name"],
            "duration": float(scene.times[-1]),
            "actors": len(scene.actor_ids),
            "policy_ids": ["ego"],
            "feedback_actors": len(scene.metadata["reactive_background"]["assignments"]),
            "feedback_pedestrians": len(peds),
            "map_lines": len(scene.road_types),
            "map_bounds_global": scene.metadata["map_bounds_global"],
            "trainable": False,
        },
    )
    print((output / "preparation.json").read_text(), flush=True)


def prefix(record, tick):
    return RolloutRecord(
        record.states[: tick + 1],
        record.motion[: tick + 1],
        record.transitions[:tick],
        record.actions[:tick],
        record.diagnostics[:tick],
        record.presence[: tick + 1],
        copy.deepcopy(record.lifecycle[: tick + 1]),
    )


def execute(output):
    start = time.monotonic()
    scene = Scene.load(output / "scene.npz")
    graph = RoadGraph.load(output / "road-graph.json") if (output / "road-graph.json").exists() else None
    routes = {
        a: Route(**task["route"]) for a, task in scene.metadata["driving_tasks"]["actors"].items()
    }
    policy = ControllerPolicy(TrafficController(routes, cooperative_ids=routes, forward_only=bool(scene.metadata.get("traffic_continuity"))))
    session = ContinuousSession(scene, policy, history_steps=20, window_steps=90, seed=42)
    try:
        record = RolloutRecord.start(session.world)
        checkpoints = {0: session.snapshot()}
        boundaries = []
        first_contact = None
        recent = checkpoints[0]
        fallback = None
        while not session.world.episode_done:
            tick = session.world.observe().step
            if tick % 10 == 0:
                recent = session.snapshot()
            if tick == 90:
                fallback = session.snapshot()
            transition = session.step()
            record.append(
                session.world,
                session.last_decision["executed_actions"],
                transition,
                session.last_decision["diagnostics"],
            )
            now = transition.observation.step
            if first_contact is None and any(
                "ego" in pair for pair in record.lifecycle[-1]["contacts"]
            ):
                first_contact = recent
            if now % 90 == 0 or session.world.episode_done:
                checkpoints[now] = session.snapshot()
                boundaries.append(
                    {
                        "tick": now,
                        "history_ticks": [o.step for o in session.history],
                        "policy_memory": copy.deepcopy(session.memories),
                        "present": int(session.world.presence().sum()),
                        "ego_alive": bool(transition.observation.alive[0]),
                    }
                )
                print(
                    "boundary",
                    now,
                    "present",
                    boundaries[-1]["present"],
                    "elapsed",
                    time.monotonic() - start,
                    flush=True,
                )
        final = session.snapshot()
        # Restore across a scheduling boundary and replay the identical next chunk.
        replay_start = 90 if 90 in checkpoints else 0
        replay_end = min(replay_start + 90, len(record.actions))
        session.restore(checkpoints[replay_start])
        for k in range(replay_start, replay_end):
            tr = session.step()
            for actual, expected in [
                (session.world.state(), record.states[k + 1]),
                (session.world.motion(), record.motion[k + 1]),
                (session.world.presence(), record.presence[k + 1]),
                (tr.observation.features, record.transitions[k].observation.features),
                (tr.rewards, record.transitions[k].rewards),
                (tr.valid, record.transitions[k].valid),
            ]:
                np.testing.assert_array_equal(actual, expected)
            assert session.last_decision["executed_actions"] == dict(
                zip(session.world.actor_ids, record.actions[k])
            )
        if replay_end in checkpoints:
            assert session.snapshot() == checkpoints[replay_end]
        session.restore(final)
        anchor = first_contact or fallback or checkpoints[0]
        remaining = len(scene.times) - 1 - anchor["world"]["tick"]
        diagnosis, branches = diagnose_anchor(
            session, anchor, owner="ego", steps=min(40, remaining), brake_steps=min(10, remaining)
        )
        save(output / "anchor.json", anchor)
        save(output / "diagnosis.json", diagnosis)
        full_evidence = record_evidence(scene, record, "ego")
        full_evidence["source_integrity"] = source_entry_overlaps(scene)
        save(output / "execution-evidence.json", full_evidence)
        inspection = {
            "kind": "continuous_nuplan",
            "paired_view": {"path": "source.html", "label": "查看同对象、同时刻的源轨迹"},
            "title": "连续物理世界：规则控制诊断",
            "gallery": True,
            "note": "同一物理世界持续推进；9秒只是记录边界。规则控制与特权日志路线，非学习模型/PDM验收。",
            "boundaries": boundaries,
            "diagnosis": diagnosis,
            "trainable": False,
        }
        write_rollout(
            scene,
            output / "rollout.npz",
            record=record,
            snapshot=session.initial,
            controlled_ids=session.world.actor_ids,
            policy="continuous-rule-diagnostic",
            controller_configuration=session.configuration,
            protocol=inspection,
        )
        export_replay(
            scene,
            output / "rollout.npz",
            output / "viewer.html",
            inspection=inspection,
            road_graph=graph,
            compress_payload=True,
        )
        export_source_replay(
            Scene.load(output / "source.npz"),
            output / "source.html",
            inspection={"kind": "source_reference", "title": "完整源轨迹；非反事实真值",
                        "paired_view": {"path": "viewer.html", "label": "查看同对象、同时刻的控制器仿真"}},
        )
        pre = prefix(record, anchor["world"]["tick"])
        for name in ("baseline", "brake"):
            joined = pre.followed_by(branches[name])
            write_rollout(
                scene,
                output / f"{name}.npz",
                record=joined,
                snapshot=session.initial,
                controlled_ids=session.world.actor_ids,
                policy="same-anchor-" + name,
                controller_configuration=session.configuration,
                protocol=diagnosis,
            )
        export_replay(
            scene,
            output / "baseline.npz",
            output / "anchor.html",
            compare_path=output / "brake.npz",
            inspection=inspection,
            road_graph=graph,
            compress_payload=True,
        )
        summary = json.loads((output / "preparation.json").read_text())
        summary.update(
            steps=len(record.actions),
            seconds=len(record.actions) * 0.1,
            boundaries=boundaries,
            ego_valid_steps=sum(int(t.valid.sum()) for t in record.transitions),
            restore={"start_tick": replay_start, "end_tick": replay_end, "exact": True},
            diagnosis={
                k: diagnosis[k]
                for k in [
                    "anchor_tick",
                    "repeat_exact",
                    "action_changed",
                    "evidence_flags",
                    "blame",
                ]
            },
            map_crop_warning_ticks=full_evidence["near_or_outside_map_crop_ticks"],
            owner_contact_ticks=full_evidence["contacts"]["owner_contact_ticks"],
            wall_seconds=time.monotonic() - start,
            status="executed",
            trainable=False,
        )
        save(output / "report.json", summary)
        save(output / "checkpoint-final.json", final)
        print(json.dumps({k: v for k, v in summary.items() if k != "boundaries"}), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "execute"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--maps-root", type=Path)
    p.add_argument("--seconds", type=float, default=60)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.manifest, a.output, a.maps_root, a.seconds)
    else:
        execute(a.output)
