"""Paired image-causality checks for a declared primitive visual diagnostic.

This verifies wiring and deterministic replay, not learned driving or Self-Play
improvement. The parent snapshot is shared; branch interventions affect executed
actions after all actors have decided from their own current images.
"""

import html
import json
import time
import tomllib
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.following import quantize_jerk
from driversi.agents.interfaces.visual import PixelBrakePolicy
from driversi.contracts.scene import Scene
from driversi.contracts.visual import CameraCalibration, CameraRig
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap
from driversi.evaluation.benchmarks.scenarios import following_scene
from driversi.sim.sensors.visual import VisualSession, VisualSnapshot
from driversi.sim.sensors.visual_recording import VisualRecord
from driversi.sim.sensors.visual_rollout import publish_visual_run, rollout_visual


def _brake_action(session: VisualSession, actor: str) -> int:
    index = session.world_frame().actor_ids.index(actor)
    state = session.world.state()[index]
    motion = session.world.motion()[index]
    speed = float(state[4]*np.cos(state[3]) + state[5]*np.sin(state[3]))
    return quantize_jerk(-3. if speed > 1. else 0., 0., motion[:2], speed)


def _first_difference(left: np.ndarray, right: np.ndarray, start: int) -> int | None:
    difference = np.any(left != right, axis=tuple(range(1, left.ndim))) if left.ndim > 1 else left != right
    found = np.flatnonzero(difference & (np.arange(len(difference)) >= start))
    return int(found[0]) if len(found) else None


def _precontact_states(scene: Scene, *records: VisualRecord) -> int:
    """Common follower/lead states before contact or either actor's stopping."""
    limit = min(len(record.physical.states) for record in records)
    for tick in range(limit):
        for record in records:
            state = record.physical.states[tick]
            boxes = [rectangle_corners(*state[index, :2], state[index, 3],
                                       *scene.dimensions[index, :2]) for index in (0, 1)]
            if np.any(state[:2, 9] != 0) or rectangles_overlap(*boxes):
                return tick
    return limit


def evaluate_visual(config_path: Path, output_dir: Path, *, real_scene: Path | None = None,
                    allow_unverified_heading: bool = False) -> dict:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    required = {"schema_version", "duration_seconds", "speed_mps", "gap_m", "intervention_seconds",
                "response_seconds", "width", "height", "sensor_stride", "history", "seed"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("Visual evaluation requires the declared schema-v1 configuration")
    start_seconds, response = config["intervention_seconds"], config["response_seconds"]
    if (not np.isfinite([start_seconds, response]).all() or start_seconds <= 0 or response <= 0
            or not np.isclose(start_seconds*10, round(start_seconds*10))
            or start_seconds + response > config["duration_seconds"]):
        raise ValueError("Intervention and response must fit the actual scene budget")
    if output_dir.exists():
        raise ValueError("Visual evaluation output directory must be new")
    scene = following_scene(config["speed_mps"], config["gap_m"], duration=config["duration_seconds"])
    rig = CameraRig((CameraCalibration.front(config["width"], config["height"]),))
    policy = PixelBrakePolicy(desired_speed=config["speed_mps"])
    intervention_tick = round(start_seconds * 10)
    output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "kind": "primitive_visual_interaction_protocol",
              "status": "running", "passed": False, "configuration": config,
              "scope": "one synthetic following template, diagnostic RGB policy; no model training"}
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    records, summaries = {}, {}
    with VisualSession(scene, policy, rig, sensor_stride=config["sensor_stride"],
                       history=config["history"], seed=config["seed"]) as parent:
        initial = parent.snapshot()
        prefix = VisualRecord.start(parent)
        prefix_start = time.perf_counter()
        while parent.tick < intervention_tick:
            prefix.append(parent, parent.step())
        prefix_seconds = time.perf_counter() - prefix_start
        parent_snapshot = parent.snapshot()
        parent_snapshot.save(output_dir / "fork-snapshot.npz")
        frozen_parent = parent.world.state()
        for name in ("baseline", "lead_brake", "far_brake", "action_replay", "repeat"):
            with ExitStack() as stack:
                session = stack.enter_context(parent.fork(name))
                if name == "repeat":
                    # Exercise disk restoration into an existing independent world.
                    session.restore(VisualSnapshot.load(output_dir / "fork-snapshot.npz"))
                    session = stack.enter_context(session.fork(name))
                record = prefix.copy_prefix()
                sampling_start = time.perf_counter()
                while not session.episode_done:
                    overrides = {}
                    if name in {"lead_brake", "action_replay", "repeat"}:
                        overrides["lead"] = _brake_action(session, "lead")
                    if name == "far_brake":
                        overrides["far"] = _brake_action(session, "far")
                    if name == "action_replay":
                        if session.tick >= len(records["baseline"].steps):
                            raise ValueError("Baseline action support exhausted; no padding allowed")
                        overrides["follower"] = int(records["baseline"].steps[session.tick].executed_actions[0])
                    record.append(session, session.step(overrides))
                elapsed = prefix_seconds + time.perf_counter() - sampling_start
                records[name] = record
                protocol = {"name": name, "intervention_seconds": start_seconds,
                            "brake_target_mps2": -3., "release_below_mps": 1.,
                            "elapsed_semantics": "measured shared prefix plus measured branch sampling",
                            "baseline_action_replay": name == "action_replay"}
                summaries[name] = publish_visual_run(scene, session, record, output_dir / name,
                                                     initial, elapsed, protocol)
        parent_unchanged = np.array_equal(parent.world.state(), frozen_parent)
        parent_memory_unchanged = np.array_equal(parent.memories, parent_snapshot.arrays["memories"])
    baseline, brake, far, replay, repeat = (records[name] for name in
                                          ("baseline", "lead_brake", "far_brake", "action_replay", "repeat"))
    positive_limit = _precontact_states(scene, baseline, brake)
    replay_limit = _precontact_states(scene, baseline, replay)
    n = min(len(baseline.steps), len(brake.steps), positive_limit)
    def rgb(record):
        return np.stack([s.observation.rgb[0, -1] for s in record.steps[:n]])

    def actions(record):
        return np.asarray([s.policy_actions[0] for s in record.steps[:n]])
    first_image = _first_difference(rgb(baseline), rgb(brake), intervention_tick)
    first_action = _first_difference(actions(baseline), actions(brake), intervention_tick)
    checks = {
        "multiple_visual_actors": len(scene.actor_ids) >= 2,
        "no_same_tick_image_anticipation": np.array_equal(rgb(baseline)[intervention_tick], rgb(brake)[intervention_tick]),
        "no_same_tick_action_anticipation": actions(baseline)[intervention_tick] == actions(brake)[intervention_tick],
        "intervention_changes_follower_image": first_image is not None and first_image > intervention_tick,
        "follower_policy_responds_after_image": first_action is not None and first_image is not None and first_action >= first_image,
        "response_within_budget": first_action is not None and (first_action-intervention_tick)*.1 <= response,
        "unrelated_intervention_preserves_follower_images": np.array_equal(rgb(baseline), rgb(far)),
        "unrelated_intervention_preserves_follower_actions": np.array_equal(actions(baseline), actions(far)),
        "action_replay_preserves_precontact_follower_motion": replay_limit > intervention_tick and np.array_equal(
            np.stack(baseline.physical.states)[:replay_limit, 0, :7],
            np.stack(replay.physical.states)[:replay_limit, 0, :7]),
        "disk_restore_repeats_world": np.array_equal(np.stack(brake.physical.states), np.stack(repeat.physical.states)),
        "disk_restore_repeats_images": np.array_equal(rgb(brake), rgb(repeat)),
        "disk_restore_repeats_actions": np.array_equal(actions(brake), actions(repeat)),
        "parent_world_unchanged": parent_unchanged,
        "parent_memories_unchanged": parent_memory_unchanged,
    }
    # Behavior is reported separately; event-free execution is not the pass gate.
    report.update(status="complete", checks={key: bool(value) for key, value in checks.items()},
                  passed=bool(all(checks.values())), branches=summaries,
                  positive_precontact_states=positive_limit, replay_precontact_states=replay_limit,
                  replay_last_compared_seconds=(replay_limit - 1)*.1 if replay_limit else None,
                  first_changed_image_seconds=None if first_image is None else first_image*.1,
                  first_changed_action_seconds=None if first_action is None else first_action*.1)
    real_link = ""
    if real_scene is not None:
        report["real_scene_run"] = rollout_visual(
            Scene.load(real_scene), output_dir / "navsim", width=config["width"], height=config["height"],
            sensor_stride=config["sensor_stride"], history=config["history"], seed=config["seed"],
            desired_speed=config["speed_mps"], allow_unverified_heading=allow_unverified_heading)
        real_link = ('<p><a href="navsim/viewer.html">NAVSIM 来源的几何视觉闭环样例</a> · '
                     '<a href="navsim/bev.html">对应物理世界</a>。该样例不计入合成跟驰协议通过项。</p>')
    partial = output_dir / "report.json.partial"
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    partial.replace(report_path)
    rows = "".join(f'<tr><td><a href="{name}/viewer.html">{html.escape(name)}</a></td>'
                   f'<td>{summary["actual_seconds"]:.1f} s</td><td>{summary["events"]["collision"]}</td>'
                   f'<td><a href="{name}/bev.html">物理世界 BEV</a></td></tr>'
                   for name, summary in summaries.items())
    checks_html = "".join(f'<li>{html.escape(key)}: {"通过" if value else "未通过"}</li>'
                          for key, value in report["checks"].items())
    (output_dir / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DriveRSI 视觉交互验证</title>'
        '<style>body{font:16px system-ui;background:#101c2c;color:#e5edf8;max-width:1100px;margin:40px auto}'
        'a{color:#7cd3ff}td,th{padding:12px;text-align:left}table{background:#192a40;width:100%}li{margin:6px}</style>'
        '<h1>DriveRSI · 逐车视觉交互</h1><p>当前物理世界 → 几何相机 → 像素诊断策略 → 联合动作。</p>'
        '<p>一个合成跟驰模板，三辆视觉主体；共享策略，独立图像历史和记忆。不是写实图像或已训练模型。</p>'
        '<p>在同一快照干预前车／远车；开放环动作作为负对照。查看各车实际输入和策略／执行动作。</p>'
        '<table><tr><th>分支</th><th>实际时域</th><th>发生碰撞的受控车</th><th>世界检查</th></tr>'
        + rows + '</table><h2>接口与因果检查</h2><ul>' + checks_html + '</ul>'
        + real_link + '<p><a href="report.json">完整报告与参数</a></p></html>', encoding="utf-8")
    return report
