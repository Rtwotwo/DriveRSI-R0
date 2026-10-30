"""Validate captured independent plans before calling the original EC function.

Executed-path slices and DriveRLActionTrajectory dummy paths are not plans.
No current DriveRL producer emits this contract; absence is a hard rejection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

from driversi.agents.backends.native import digest
from driversi.evaluation.epdms.state_codec import unpack_ego


def extended_comfort(data, actual, sampling):
    if (sampling.num_poses != 40 or sampling.interval_length != 0.1
            or actual.shape != (41, StateIndex.size())):
        raise ValueError("Full native four-second state sampling required")
    pair = data.get("adjacent_native_plans")
    if not isinstance(pair, dict):
        raise ValueError("Complete EPDMS requires captured adjacent native plans; EC cannot be omitted")
    if (pair.get("kind") != "independently_generated_native_plans"
            or pair.get("uses_dummy_trajectory") is not False
            or pair.get("copied_from_executed_rollout") is not False
            or pair.get("sampling_interval_s") != 0.1):
        raise ValueError("Dummy trajectories or duplicated rollout slices cannot establish EC")
    capture = pair.get("capture_file", {})
    if set(capture) != {"path", "sha256"} or digest(capture["path"]) != capture["sha256"]:
        raise ValueError("Independent native planning capture is missing or changed")
    if json.loads(Path(capture["path"]).read_text()) != {k: v for k, v in pair.items() if k != "capture_file"}:
        raise ValueError("Planning arrays differ from the sealed independent capture")
    previous, current = (np.asarray(pair[k]["simulated_states"], dtype=np.float64)
                         for k in ("previous", "current"))
    if previous.shape != actual.shape or current.shape != actual.shape:
        raise ValueError("Two full native 4-second plans with equal state dimensions required")
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("Non-finite planning pair")
    anchor = data["context"]["anchor_timestamp_us"]
    now = pair["current"]["anchor_timestamp_us"]
    before = pair["previous"]["anchor_timestamp_us"]
    if now != anchor or now - before != 500000:
        raise ValueError("Adjacent native planning anchors must be 0.5 seconds apart")
    history = data.get("prefix_history", [])
    for timestamp, initial in ((before, previous[0]), (now, current[0])):
        frames = [frame for frame in history
                  if frame.get("origin") == "world" and frame.get("ego", [None])[-1] == timestamp]
        if len(frames) != 1:
            raise ValueError("Planning anchor requires exactly one actual generated history state")
        state = ego_states_to_state_array([unpack_ego(frames[0]["ego"])])[0]
        if not np.allclose(initial, state, rtol=0, atol=1e-7):
            raise ValueError("Planning anchor differs from actual generated history")
    if not np.allclose(current, actual, rtol=0, atol=1e-7):
        raise ValueError("Current scored trajectory differs from the captured current plan")
    frame = pd.DataFrame([
        dict(token="previous", start_time=before / 1e6, ego_simulated_states=previous),
        dict(token="current", start_time=now / 1e6, ego_simulated_states=current),
    ]).set_index("token")
    update = SceneAggregator("current", "previous", frame, sampling).aggregate_scores(one_stage_only=True)
    result = float(update.iloc[0]["two_frame_extended_comfort"])
    if result not in (0.0, 1.0):
        raise ValueError("Invalid native extended comfort result")
    return result
