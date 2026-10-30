"""Validate a locally generated DriveRSI package with an installed NAVSIM loader.

Run in the model environment, without importing the DriveRSI native runtime.
Only the exporter's hashed log files are passed to the pickle-based loader.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(root):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    root = Path(root).resolve()
    report = json.loads((root / "manifest.json").read_text())
    if (
        report.get("version") != "navsim-simscale-synthetic-v1"
        or report.get("status") != "completed"
    ):
        raise ValueError("Not a completed DriveRSI-generated package")
    for name, expected in report["sha256"].items():
        p = (root / name).resolve()
        if root not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("Package path/hash mismatch")
    log = root / report["log_file"]
    if log.relative_to(root).as_posix() not in report["sha256"]:
        raise ValueError("Unverified pickle log")
    if set(log.parent.iterdir()) != {log}:
        raise ValueError("Loader directory contains unverified additional logs")
    sensors = SensorConfig(**report["sensor_config"])
    loader = SceneLoader(
        log.parent, root / "sensor_blobs_hist", SceneFilter(**report["scene_filter"]), sensor_config=sensors
    )
    assert loader.tokens == [report["token"]]
    scene = loader.get_scene_from_token(report["token"])
    agent = scene.get_agent_input()
    direct = loader.get_agent_input_from_token(report["token"])
    assert len(direct.cameras) == 4
    for x, y in zip(agent.ego_statuses, direct.ego_statuses):
        np.testing.assert_allclose(x.ego_pose, y.ego_pose, rtol=0, atol=1e-5)
    assert len(scene.frames) == 12 and len(agent.cameras) == 4
    assert [f.timestamp for f in scene.frames] == list(
        range(scene.frames[0].timestamp, scene.frames[0].timestamp + 6000000, 500000)
    )
    loaded = 0
    for i, cameras in enumerate(agent.cameras):
        for name in report["cameras"]:
            image = getattr(cameras, name.lower()).image
            assert image is not None and image.ndim == 3 and image.shape[2] == 3
            loaded += 1
    assert all(
        getattr(f.cameras, c.lower()).image is None
        for f in scene.frames[4:]
        for c in report["cameras"]
    )
    predicted = scene.get_future_trajectory().poses
    physics = [
        json.loads(line)
        for line in (root / "audit/episode/generation-000-physics.jsonl").read_text().splitlines()
    ]
    episode = json.loads((root / "audit/episode/manifest.json").read_text())
    sample = json.loads((root / "audit/sample.json").read_text())
    slot = episode["actor_ids"].index("ego")
    states = np.array(
        [physics[0]["privileged"]["world_before"]["state"]]
        + [p["privileged"]["world_after"]["state"] for p in physics]
    )
    anchor = sample["anchor_tick"]
    # Labels remain present when optional future RGB is omitted.
    label_times = np.arange(0.5, 4.01, 0.5)
    target_ticks = anchor + np.rint(label_times / sample["dt_seconds"]).astype(int)
    poses = states[[anchor] + target_ticks.tolist(), slot][
        :, [0, 1, 3]
    ].copy()
    # Independent reference, vehicle centre -> nuPlan rear axle.
    poses[:, 0] -= 1.461 * np.cos(poses[:, 2])
    poses[:, 1] -= 1.461 * np.sin(poses[:, 2])
    c, s = np.cos(poses[0, 2]), np.sin(poses[0, 2])
    local = (poses[1:, :2] - poses[0, :2]) @ np.array([[c, -s], [s, c]])
    yaw = poses[1:, 2] - poses[0, 2]
    expected = np.column_stack((local, np.arctan2(np.sin(yaw), np.cos(yaw))))
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-6)
    assert len(scene.frames[0].annotations.boxes) > 0
    result = {
        "loader": "navsim.common.dataloader.SceneLoader",
        "direct_agent_input_verified": True,
        "scene_count": len(loader.tokens),
        "frames": len(scene.frames),
        "history_images_loaded": loaded,
        "future_images_in_agent_input": 0,
        "future_trajectory_shape": list(predicted.shape),
        "rear_axle_label_max_error_m_rad": float(np.abs(predicted - expected).max()),
        "timestamp_interval_seconds": 0.5,
        "package_hashes_verified": len(report["sha256"]),
        "trainable": report["trainable"],
        "scope": "format_and_execution_alignment_not_quality",
    }
    (root / "loader-checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    print(json.dumps(validate(parser.parse_args().package)))
