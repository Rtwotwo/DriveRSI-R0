"""Render-independent four-history/eight-or-ten-future executed physics window."""

import json
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.contracts.scene import Scene
from driversi.data.datasets.history_windows import four_frame_ticks
from driversi.data.datasets.navsim_synthetic import rear_axle_states


def trajectory_window(states, presence, actor_ids, *, anchor_tick, dt, future_frames=8):
    states = np.asarray(states, dtype=float)
    presence = np.asarray(presence)
    if (
        states.ndim != 3
        or states.shape[2] != 13
        or not np.isfinite(states).all()
        or presence.shape != states.shape[:2]
        or presence.dtype.kind != "b"
        or len(actor_ids) != states.shape[1]
        or len(set(actor_ids)) != len(actor_ids)
        or "ego" not in actor_ids
        or not np.isclose(dt, 0.1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Executed world identities, states, presence and 10Hz clock required")
    if type(future_frames) is not int or future_frames not in (8, 10):
        raise ValueError("Future frame count must be explicitly eight or ten")
    history = four_frame_ticks(anchor_tick, 5)
    future = tuple(range(anchor_tick + 5, anchor_tick + 5 * future_frames + 1, 5))
    if future[-1] >= len(states):
        raise ValueError("Complete requested four-history/future trajectory required")
    ego_index = list(actor_ids).index("ego")
    if not presence[history[0] : future[-1] + 1, ego_index].all():
        raise ValueError("Ego is absent within the requested window")
    rear = rear_axle_states(states, ego_index)
    anchor = rear[anchor_tick]
    c, s = np.cos(anchor[3]), np.sin(anchor[3])
    xy = (rear[list(future), :2] - anchor[:2]) @ np.array([[c, -s], [s, c]])
    yaw = rear[list(future), 3] - anchor[3]
    poses = np.column_stack((xy, np.arctan2(np.sin(yaw), np.cos(yaw))))

    def frames(ticks):
        return [
            dict(
                tick=t,
                seconds=t * dt,
                relative_seconds=(t - anchor_tick) * dt,
                ego_rear_axle_pose=rear[t].tolist(),
                objects=[
                    dict(actor_id=a, center_state=states[t, i].tolist())
                    for i, a in enumerate(actor_ids)
                    if presence[t, i]
                ],
            )
            for t in ticks
        ]

    return dict(
        kind=f"executed_4history_{future_frames}future_v1",
        future_frame_count=future_frames,
        anchor_tick=anchor_tick,
        dt_seconds=dt,
        history=frames(history),
        future=frames(future),
        future_trajectory=dict(
            poses=poses.tolist(),
            times=[(t - anchor_tick) * dt for t in future],
            coordinate_frame="ego_rear_axle_at_anchor",
        ),
        world_frame="map_axes_translated_from_initial_ego",
        input_scope="history_only",
        future_scope="supervision_only",
        trainable=False,
        qualification="requires_independent_trajectory_and_visual_admission",
    )


def export_executed_window(scene_path, rollout_path, output, *, anchor_tick, future_frames=8):
    from driversi.rendering.bev.replay import _load_run, _scene_key

    scene_path, rollout_path, output = map(Path, (scene_path, rollout_path, output))
    if output.exists():
        raise FileExistsError(output)
    scene = Scene.load(scene_path)
    if scene.metadata.get("pose_reference") != "vehicle_center":
        raise ValueError("Explicit vehicle-center pose convention required")
    if scene.metadata.get("coordinate_frame") != "map_axes_translated_from_initial_ego":
        raise ValueError("Explicit map-axis coordinate convention required")
    run = _load_run(scene, rollout_path, _scene_key(scene))
    result = trajectory_window(
        run["states"],
        run["presence"],
        scene.actor_ids,
        anchor_tick=anchor_tick,
        dt=run["metadata"]["dt_seconds"],
        future_frames=future_frames,
    )
    result.update(
        source_log=scene.metadata["source_log"],
        source_token=scene.metadata["scenario_id"],
        source_split=scene.metadata["split"],
        source_files={str(p.resolve()): digest(p) for p in (scene_path, rollout_path)},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
