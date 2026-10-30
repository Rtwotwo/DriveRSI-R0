"""Reproducible NAVSIM same-state branch checks over the existing MTGS assets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from driversi.data.assets.mtgs_episode import export_mtgs_episode
from driversi.data.datasets.visual_samples import export_visual_samples
from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer
from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime
from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession
from driversi.sim.persistence.session_checkpoint import SessionCheckpoint


def run(config: dict, output: Path, *, fork_tick=11):
    from PIL import Image

    output = Path(output)
    if output.exists():
        raise FileExistsError("Pilot output must be new")
    output.mkdir(parents=True)
    runtime = MTGSRuntime(**{k: Path(config[k]) for k in ("config_path", "checkpoint_path", "road_block_config")},
                          scale=config.get("scale", .5), trusted_assets=True)
    results, viewers = {}, []
    keys = ("scene", "road_graph", "binding", "world_python", "native_path", "population_config")
    settings = {k: Path(config[k]) for k in keys}
    settings.update(cameras=tuple(config["cameras"]), sensor_stride=config["sensor_stride"],
                    desired_speed=config["desired_speed"],
                    virtual_camera_actors=tuple(config["virtual_camera_actors"]), height_mode=config["height_mode"])
    for mode in ("structured", "visual"):
        started = time.perf_counter()
        parent = MTGSWorldSession(runtime=runtime, output=output/mode/"parent", **settings,
                                 visual_policy=mode == "visual", visual_actors=tuple(config["visual_actors"]) if mode == "visual" else ())
        children = []
        try:
            while parent.current["tick"] < fork_tick:
                parent.advance()
            checkpoint = parent.snapshot()
            checkpoint.save(output/mode/"checkpoint.npz")
            disk = SessionCheckpoint.load(output/mode/"checkpoint.npz")
            saved_parent = json.dumps(parent.current, sort_keys=True)
            replay = parent.fork(output/mode/"replay", disk)
            children.append(replay)
            intervention = parent.fork(output/mode/"brake", disk)
            children.append(intervention)
            # Branch creation cannot mutate the parent world or actor state.
            assert json.dumps(parent.current, sort_keys=True) == saved_parent
            observed_parent = parent.worker.request("observe")
            assert all(observed_parent[k] == parent.current[k] for k in ("tick", "state", "motion", "alive", "done"))
            for session in (parent, replay, intervention):
                while not session.current["done"]:
                    tick = session.current["tick"]
                    override = ({config["brake_actor"]: 1} if session is intervention and fork_tick <= tick < fork_tick+10 else None)
                    session.advance(override)
                session._publish()
                write_session_viewer(session.output/"manifest.json")
                exporter = export_visual_samples if mode == "visual" else export_mtgs_episode
                exporter(session.output/"manifest.json", output/mode/(session.output.name+"-sequence"))
                viewers.append((mode+" / "+session.output.name, session.output.relative_to(output).as_posix()+"/viewer.html"))
            fields = ("tick", "state", "motion", "alive", "done", "policy_actions", "executed_actions",
                      "diagnostics", "valid", "terminated", "truncated", "events", "behavior_policies")
            suffix_a = parent.manifest["steps"][fork_tick:]
            suffix_b = replay.manifest["steps"][fork_tick:]
            physics_exact = len(suffix_a) == len(suffix_b) and all(a[k] == b[k] for a, b in zip(suffix_a, suffix_b) for k in fields)
            exact_images, total_images, max_rgb_difference = 0, 0, 0
            for a, b in zip(parent.manifest["frames"], replay.manifest["frames"], strict=True):
                for av, bv in zip(a["cameras"], b["cameras"], strict=True):
                    with Image.open(parent.output/av["image"]) as im:
                        left = np.asarray(im).astype(np.int16)
                    with Image.open(replay.output/bv["image"]) as im:
                        right = np.asarray(im).astype(np.int16)
                    exact_images += int(np.array_equal(left, right))
                    total_images += 1
                    max_rgb_difference = max(max_rgb_difference, int(np.abs(left-right).max()))
            decisions_a, decisions_b = parent.manifest.get("decisions", []), replay.manifest.get("decisions", [])
            visual_exact = len(decisions_a) == len(decisions_b) and all(
                all(a[k] == b[k] for k in ("actions", "memory_before", "memory_after", "rng_before", "rng_after", "actors"))
                for a, b in zip(decisions_a, decisions_b))
            changed = [b["tick"] for a, b in zip(parent.manifest["steps"], intervention.manifest["steps"], strict=True)
                       if a["state"] != b["state"]]
            results[mode] = {"fork_tick": fork_tick, "physics_suffix_exact": physics_exact,
                "visual_decisions_memory_rng_exact": visual_exact if mode == "visual" else None,
                "images_exact": exact_images, "images_compared": total_images,
                "max_rgb_channel_difference": max_rgb_difference,
                "intervention_first_world_difference_tick": min(changed) if changed else None,
                "parent_unchanged_by_fork": True, "inherited_prefix_steps": fork_tick,
                "new_suffix_steps_per_branch": len(suffix_a), "wall_seconds": time.perf_counter()-started,
                "trainable": False, "visual_policy": "pixel_dependency_diagnostic" if mode == "visual" else None}
            results[mode]["brake_actor"] = config["brake_actor"]
            results[mode]["override_action_difference_steps"] = sum(
                row["executed_actions"] != row["policy_actions"] for row in intervention.manifest["steps"])
            (output/"verification.json").write_text(json.dumps(results, indent=2))
            if not physics_exact or (mode == "visual" and not visual_exact) or not changed:
                raise AssertionError("Branch replay/intervention check failed")
        finally:
            for session in children:
                session.close()
            parent.close()
    write_pilot_index(output, results, viewers)
    return results


def write_pilot_index(output: Path, results: dict, viewers: list):
    """Regenerate only the UTF-8 gallery; never rerun or rewrite physics."""
    links = "".join(f'<li><a href="{url}">{label}</a></li>' for label, url in viewers)
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>DriveRSI branch checks</title>'
        '<style>body{background:#101a25;color:#e4edf4;font:18px system-ui;max-width:1000px;margin:40px auto}a{color:#72c6ff}li{margin:16px}</style>'
        '<h1>NAVSIM 写实交互分叉</h1><p>1.1秒同一检查点：原分支 / 磁盘恢复分支 / 前车制动分支。</p>'
        '<p>0–1.1秒为复制的共同前缀，不算新样本。默认结构化；视觉分支为像素诊断，非学习型 Self-Play。'
        '渲染伪影仍保留，trainable=false，串行渲染。</p><ul>'+links+'</ul><pre>'+json.dumps(results,ensure_ascii=False,indent=2)+'</pre>', encoding='utf-8')
    return output/"index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fork-tick", type=int, default=11)
    args = parser.parse_args()
    if args.fork_tick < 1:
        parser.error("fork-tick must be positive")
    print(json.dumps(run(json.loads(args.config.read_text()), args.output, fork_tick=args.fork_tick)))


if __name__ == "__main__":
    main()
