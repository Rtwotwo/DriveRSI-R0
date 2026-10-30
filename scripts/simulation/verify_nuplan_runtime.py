"""Real nuPlan causal-history and 60-second continuity regression (no training)."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
from verify_continuous_nuplan import execute

from driversi.agents.traffic_control.traffic import TrafficController
from driversi.contracts.road import Route
from driversi.contracts.scene import Scene
from driversi.data.sources.nuplan_db import compile_window, read_window
from driversi.sim.execution.continuous import ContinuousSession, ControllerPolicy
from driversi.sim.execution.structured_history import StructuredHistory


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


class InitialCompiledWorld:
    """Initial-state-only reader; no simulation or collision claim in corpus audit."""

    def __init__(self, scene):
        self.scene = scene

    def state(self):
        s = self.scene
        out = np.zeros((len(s.actor_ids), 13))
        out[:, :4], out[:, 4:7] = s.poses[0], s.velocities[0]
        return out

    def presence(self):
        return self.scene.presence[0]

    def observe(self):
        class Clock:
            step = 0

        return Clock()


def write_index(output):
    from html import escape

    output = Path(output)
    corpus = json.loads((output / "corpus.json").read_text())
    report = json.loads((output / "report.json").read_text())
    history = json.loads((output / "model-history.json").read_text())
    contacts = json.loads((output / "execution-evidence.json").read_text())["contacts"]
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(v))}</td>"
            for v in [
                entry["city"],
                entry["source_log"],
                entry["world_registered"],
                entry["observed_history_identities"],
                "需绑定明确任务" if entry["requires_explicit_task"] else "已有移动控制槽位",
            ]
        )
        + "</tr>"
        for entry in corpus
    )
    prefix = output.name + "/"
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DriveRSI · nuPlan历史输入验收</title>
    <style>body{{font:16px/1.6 system-ui;margin:32px auto;padding:0 24px;max-width:1250px;color:#173342;background:#f4f8fa}}h1{{font-size:30px}}section{{background:white;padding:22px;border-radius:12px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d7e2e8;text-align:left}}a{{color:#00696c}}.note{{color:#7b4b12}}.links a{{margin-right:20px}}</style>
    <h1>nuPlan 接入 · 连续历史与运动语义</h1><p>参考 DriveRL 的时间、身份与坐标契约。结构化真值接口；未运行学习模型或训练。</p>
    <section><h2>同一长日志：{report["seconds"]:.1f} 秒</h2><p>自车接触状态：{contacts["owner_contact_ticks"]} · 其他对象接触状态：{contacts["other_contact_ticks"]} · 恢复后 {history["restored_model_inputs_exact"]} 组模型输入精确一致。</p>
    <div class="links"><a href="{prefix}viewer.html?actor=ego&tick=286">播放实际仿真</a><a href="{prefix}source.html?actor=ego&tick=286">对照原始日志</a><a href="../traffic-continuity-v1/final/viewer.html?actor=ego&tick=286">上一版仿真</a></div>
    <p class="note">接触状态数不是官方 PDM。仍有等待接管身份与路线支持边界，trainable=false。</p></section>
    <section><h2>四城市 {len(corpus)} 份真实日志</h2><p>窗口覆盖对象数包括后续入场对象；历史输入只含已观测对象，不能用两列差值判断“删车”。这些窗口完成数据接口检查，并未全部完成安全驾驶验收。</p>
    <table><thead><tr><th>城市</th><th>日志</th><th>窗口对象</th><th>历史对象</th><th>任务边界</th></tr></thead><tbody>{rows}</tbody></table></section>
    <section><h2>输入边界</h2><p>锚点前采用实测关联帧，锚点后采用实际物理世界；缺帧、速度缺测和跨来源导数分别标记。坐标统一为对象中心，接入原生模型时仍需验证专用转换。</p>
    <p>保留自车位姿与 lidar 时间偏差，超过 50 ms 明确拒绝；不宣称传感器严格同时采集。旧 continuous-session-v1 检查点需重新初始化，原场景与回放仍可读取。</p>
    <div class="links"><a href="{prefix}model-history.json">模型输入与恢复证据</a><a href="{prefix}corpus.json">四城市完整数据检查</a><a href="{prefix}execution-evidence.json">实际执行证据</a></div></section></html>"""
    (output.parent / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = []
    for path in sorted((args.corpus / "logs").glob("log-*/manifest.json")):
        descriptor = json.loads(path.read_text())
        from driversi.data.sources.nuplan_db import open_db

        with open_db(descriptor["db_path"]) as connection:
            rows = connection.execute(
                "SELECT token,timestamp FROM lidar_pc ORDER BY timestamp"
            ).fetchall()
        anchor = next(row for row in rows if row[1] - rows[0][1] >= 1500000)
        window = read_window(descriptor["db_path"], token=anchor[0].hex(), duration=9)
        scene = compile_window(window, duration=9, max_actors=1)
        history = StructuredHistory(scene, InitialCompiledWorld(scene), 20)
        sample = history.sample()
        assert sample["frame_valid"].all()
        assert (sample["sampled_timestamp_us"] <= sample["requested_timestamp_us"]).all()
        assert sample["origins"][-1] == "physical_world"
        # Tampering future labels must leave all model input byte-for-byte equal.
        mutated = copy.deepcopy(scene)
        mutated.poses[1:, :, :2] += 1234
        mutated.velocities[1:] = 333
        mutated.metadata["nuplan_source"]["future_source_tokens"] = ["future-poison"]
        alternate = StructuredHistory(mutated, InitialCompiledWorld(mutated), 20).sample()
        assert serial(sample) == serial(alternate)
        result = {
            "source_log": descriptor["source_log"],
            "city": descriptor["map_name"],
            "source_frames": len(window["history"]),
            "world_registered": len(scene.actor_ids),
            "observed_history_identities": len(sample["actor_ids"]),
            "history_future_invariance": True,
            "requires_explicit_task": scene.metadata.get("unbound_controlled_tasks", []),
            "timestamp_alignment": window["timestamp_alignment"],
            "sample": serial(sample),
        }
        corpus.append(result)
        print(
            "CORPUS",
            result["city"],
            result["world_registered"],
            result["observed_history_identities"],
            flush=True,
        )
    (args.output / "corpus.json").write_text(json.dumps(corpus, indent=2, allow_nan=False))
    scene = Scene.load(args.baseline / "scene.npz")
    source = scene.metadata["nuplan_source"]
    window = read_window(
        source["db_path"], token=source["anchor_token"], duration=float(scene.times[-1])
    )
    fresh = compile_window(window, duration=float(scene.times[-1]), max_actors=1)
    assert fresh.actor_ids == scene.actor_ids
    np.testing.assert_array_equal(scene.poses[0], fresh.poses[0])
    old_velocity = scene.velocities[0, 0, :2].copy()
    scene.velocities[0] = fresh.velocities[0]
    scene.metadata["nuplan_source"] = fresh.metadata["nuplan_source"]
    scene.metadata["velocity_model"] = fresh.metadata["velocity_model"]
    scene.save(args.output / "scene.npz")
    original_source = Scene.load(args.baseline / "source.npz")
    assert original_source.actor_ids == fresh.actor_ids
    original_source.velocities[0] = fresh.velocities[0]
    original_source.metadata["nuplan_source"] = fresh.metadata["nuplan_source"]
    original_source.metadata["velocity_model"] = fresh.metadata["velocity_model"]
    original_source.save(args.output / "source.npz")
    shutil.copyfile(args.baseline / "road-graph.json", args.output / "road-graph.json")
    prep = json.loads((args.baseline / "preparation.json").read_text())
    prep.update(
        comparison="same log and traffic configuration; reported local velocity initialization and causal model history",
        baseline=str(args.baseline),
        trainable=False,
    )
    (args.output / "preparation.json").write_text(json.dumps(prep, indent=2))
    execute(args.output)
    routes = {
        a: Route(**task["route"]) for a, task in scene.metadata["driving_tasks"]["actors"].items()
    }
    controller = ControllerPolicy(
        TrafficController(routes, cooperative_ids=routes, forward_only=True)
    )
    session = ContinuousSession(scene, controller, history_steps=20, window_steps=90, seed=42)
    samples = []
    try:
        samples.append(serial(session.model_input()))
        session.advance(90)
        checkpoint = json.loads(json.dumps(session.snapshot()))
        inputs = []
        for _ in range(20):
            inputs.append(serial(session.model_input()))
            session.step()
        session.restore(checkpoint)
        for expected in inputs:
            assert serial(session.model_input()) == expected
            session.step()
        samples.append(serial(session.model_input()))
        while not session.world.episode_done:
            session.step()
            if session.world.observe().step % 90 == 0 or session.world.episode_done:
                samples.append(serial(session.model_input()))
        result = {
            "old_initial_center_velocity": old_velocity.tolist(),
            "new_initial_center_velocity": scene.velocities[0, 0, :2].tolist(),
            "restored_model_inputs_exact": len(inputs),
            "sample_states": samples,
            "corpus_logs": len(corpus),
            "corpus_cities": sorted(set(v["city"] for v in corpus)),
            "trainable": False,
            "model_inference_or_training": False,
            "official_PDM": None,
            "limitations": [
                "model-neutral center-based history is not a DriveRL checkpoint tensor",
                "existing privileged logged routes remain simulator intents",
                "pending overlaps and map dead-ends retain prior handling",
                "public DriveRL supplies inference/evaluation, not the original training pipeline",
            ],
        }
        (args.output / "model-history.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
    finally:
        session.close()
    write_index(args.output)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
