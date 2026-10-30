"""Immutable physical conditions shared by video generators, never image admission."""

import json
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
from driversi.rsi.plugins.synthesis import check_files

CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_R1", "CAM_B0", "CAM_L1")
HISTORY = (5, 10, 15, 20)
FUTURE = tuple(range(25, 71, 5))


def clip_ranges(ticks, length=25, overlap=5):
    ticks = tuple(ticks)
    if (
        length <= overlap
        or overlap < 1
        or len(ticks) < length
        or any(b - a != 1 for a, b in zip(ticks, ticks[1:]))
    ):
        raise ValueError("Native clips require contiguous actual physics ticks; no padding")
    starts = list(range(0, len(ticks) - length + 1, length - overlap))
    if starts[-1] != len(ticks) - length:
        starts.append(len(ticks) - length)
    return [list(ticks[i : i + length]) for i in starts]


def load_physics(branch_path, window_path):
    branch, window = read_mapping(branch_path), read_mapping(window_path)
    if (
        branch.get("kind") != "selected_native_branch"
        or branch.get("teacher_qualified") is not True
        or branch.get("anchor_tick") != 20
        or branch.get("future_frames") != 10
    ):
        raise ValueError("Qualified changed-history, ten-future branch required")
    validate_trajectory_evidence(branch["trajectory_evidence"])
    check_files({**branch["files"], **branch["upstream_source_files"]})
    if digest(window_path) != branch["trajectory_window_sha256"]:
        raise ValueError("Trajectory window changed")
    execution = Path(branch["execution"])
    scene_path, rollout_path = execution / "scene.npz", execution / (branch["branch"] + ".npz")
    for path in (scene_path, rollout_path):
        if branch["files"].get(str(path)) != digest(path):
            raise ValueError("Unsealed physical payload")
    scene = Scene.load(scene_path)
    with np.load(rollout_path, allow_pickle=False) as data:
        states, presence = data["states"], data["presence"]
        actor_ids = tuple(data["actor_ids"].tolist())
    if scene.metadata["split"] != "train" or scene.metadata["source_log"] != branch["source_log"]:
        raise ValueError("Training source differs")
    if (
        actor_ids != scene.actor_ids
        or len(states) <= 70
        or len(presence) != len(states)
        or not np.isfinite(states[presence]).all()
    ):
        raise ValueError("Incomplete or invalid executed population")
    raw_path = scene.metadata["navsim_source"]["log_path"]
    if branch["upstream_source_files"].get(raw_path) != digest(raw_path):
        raise ValueError("Raw source not sealed")
    return branch, window, scene, states, presence


def write_bundle(branch_path, window_path, output):
    branch, window, scene, states, presence = load_physics(branch_path, window_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ticks = np.arange(5, 71, dtype=np.int64)
    physical = output / "physics.npz"
    np.savez_compressed(
        physical,
        ticks=ticks,
        timestamps_us=scene.metadata["source_start_timestamp_us"] + ticks * 100000,
        states=states[ticks],
        presence=presence[ticks],
        actor_ids=np.asarray(scene.actor_ids),
        dimensions=scene.dimensions,
    )
    files = {str(Path(p).resolve()): digest(p) for p in (branch_path, window_path, physical)}
    files.update(branch["files"])
    files.update(branch["upstream_source_files"])
    report = dict(
        schema_version=1,
        kind="conditioned_video_physics",
        source_log=branch["source_log"],
        branch_report=str(Path(branch_path).resolve()),
        trajectory_window=str(Path(window_path).resolve()),
        physics=str(physical.resolve()),
        branch_identity=digest(branch_path),
        files=files,
        history_ticks=list(HISTORY),
        future_ticks=list(FUTURE),
        camera_order=list(CAMERAS),
        source_start_timestamp_us=scene.metadata["source_start_timestamp_us"],
        native_streetcrafter_clips=clip_ranges(ticks.tolist()),
        trajectory_score=branch["trajectory_evidence"]["score"],
        trainable=False,
        visual_admitted=False,
        limitations=[
            "Camera geometry and point-cloud/map conditions require provider preparation",
            "Correct conditions do not prove correct generated pixels",
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def validate_bundle(path):
    value = read_mapping(path)
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "conditioned_video_physics"
        or value.get("camera_order") != list(CAMERAS)
        or value.get("history_ticks") != list(HISTORY)
        or value.get("future_ticks") != list(FUTURE)
    ):
        raise ValueError("Unexpected video condition contract")
    check_files(value["files"])
    if value["branch_identity"] != digest(value["branch_report"]):
        raise ValueError("Condition bundle branch identity differs")
    branch, window, scene, states, presence = load_physics(
        value["branch_report"], value["trajectory_window"]
    )
    with np.load(value["physics"], allow_pickle=False) as data:
        ticks = np.arange(5, 71)
        if (
            not np.array_equal(data["ticks"], ticks)
            or not np.array_equal(data["states"], states[ticks])
            or not np.array_equal(data["presence"], presence[ticks])
            or tuple(data["actor_ids"].tolist()) != scene.actor_ids
            or not np.array_equal(data["dimensions"], scene.dimensions)
        ):
            raise ValueError("Generated conditions changed actual physics")
        if not np.array_equal(
            data["timestamps_us"], scene.metadata["source_start_timestamp_us"] + ticks * 100000
        ):
            raise ValueError("Condition timestamps changed")
    if value["native_streetcrafter_clips"] != clip_ranges(list(range(5, 71))):
        raise ValueError("Native video clips changed")
    return value
