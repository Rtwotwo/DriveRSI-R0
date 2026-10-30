"""Run the actual image->policy->joint-world loop and publish its inspectable record."""

import json
import time
from pathlib import Path

from driversi.agents.interfaces.visual import PixelBrakePolicy
from driversi.contracts.scene import Scene
from driversi.contracts.visual import CameraCalibration, CameraRig
from driversi.rendering.bev.replay import export_replay
from driversi.rendering.visual.visual_replay import export_visual_replay
from driversi.sim.sensors.visual import VisualSession
from driversi.sim.sensors.visual_recording import VisualRecord, write_visual_record


def publish_visual_run(scene, session, record, directory, initial_snapshot, elapsed, protocol=None):
    report = write_visual_record(scene, session, record, directory,
                                 initial_snapshot=initial_snapshot, elapsed_seconds=elapsed,
                                 protocol=protocol)
    export_replay(scene, directory / "rollout.npz", directory / "bev.html")
    export_visual_replay(directory / "visual.npz", directory / "viewer.html")
    temporary = directory / "manifest.json.partial"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return report


def rollout_visual(scene: Scene, output_dir: Path, *, width: int = 160, height: int = 96,
                   sensor_stride: int = 1, history: int = 2, seed: int = 0,
                   desired_speed: float = 4., allow_unverified_heading: bool = False) -> dict:
    """No trainer or log-image substitution; every policy camera is rendered online."""
    if output_dir.exists():
        raise ValueError("Visual output directory must be new")
    rig = CameraRig((CameraCalibration.front(width, height),))
    policy = PixelBrakePolicy(desired_speed=desired_speed)
    with VisualSession(scene, policy, rig, sensor_stride=sensor_stride, history=history, seed=seed,
                       allow_unverified_heading=allow_unverified_heading) as session:
        if session.episode_done:
            raise ValueError("Scene has no initially active controlled actors")
        output_dir.mkdir(parents=True)
        start = time.perf_counter()
        snapshot = session.snapshot()
        record = VisualRecord.start(session)
        while not session.episode_done:
            record.append(session, session.step())
        elapsed = time.perf_counter() - start
        return publish_visual_run(scene, session, record, output_dir, snapshot, elapsed)
