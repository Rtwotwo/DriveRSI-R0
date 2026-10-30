"""NAVSIM structured frames -> unchanged DriveRL native scenario inputs.

No source-frame interpolation is performed here. Dense teacher history must come
from physical execution before model actions are eligible.
"""

import os
import sys
from pathlib import Path

import numpy as np


def scenario_from_scene(scene, maps_root):
    from driversi.agents.backends.native import project_root
    from driversi.data.sources.navsim import load_log

    source = scene.metadata["navsim_source"]
    os.environ["NUPLAN_MAPS_ROOT"] = str(maps_root)
    native = project_root() / "src/vendor/driving/navsim-v1.1"
    if str(native) not in sys.path:
        sys.path.insert(0, str(native))
    from navsim.common.dataclasses import Scene as NativeScene
    from navsim.common.dataclasses import SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    frames = load_log(Path(source["log_path"]), trusted=True)
    start = scene.metadata["source_start_frame"]
    if start < 3 or frames[start]["token"] != scene.metadata["scenario_id"]:
        raise ValueError("NAVSIM source anchor/history mismatch")
    stop = scene.metadata["horizon_plan"]["source_end_frame"]
    import yaml

    allowed = yaml.safe_load(
        (
            native
            / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
        ).read_text()
    )["log_names"]
    if (
        scene.metadata["split"] != "train"
        or scene.metadata["source_log"] not in allowed
        or Path(source["log_path"]).stem != scene.metadata["source_log"]
    ):
        raise ValueError("Teacher synthesis requires an official navtrain source log")
    if not frames[start].get("roadblock_ids"):
        raise ValueError("Native navigation unavailable at NAVSIM source anchor")
    raw = frames[start - 3 : stop + 1]
    data = NativeScene.from_scene_dict_list(
        raw, Path(source["sensor_root"]), 4, len(raw) - 4, SensorConfig.build_no_sensors()
    )
    native_scenario = NavSimScenario(data, str(maps_root), "nuplan-maps-v1.0")
    return NavsimContext(native_scenario, history_mode=scene.metadata.get("driverl_history_mode"))


class NavsimContext:
    """Supply real past frames missing in upstream NavSimScenario's past stubs."""

    driversi_history_interval = 0.5
    driversi_history_scope = "navsim_actual_2hz_frames_no_interpolation"

    def __init__(self, native, history_mode=None):
        self.native = native
        self._bootstrap = None
        if history_mode is not None:
            if history_mode != "causal_resampled_10hz_v1":
                raise ValueError("Unknown explicit NAVSIM initialization history mode")
            from driversi_agents.driverl.history_bootstrap import bootstrap

            self._bootstrap = bootstrap(native)
            self.driversi_history_interval = 0.1
            self.driversi_history_scope = history_mode
            self.driversi_history_evidence = self._bootstrap[2]

    def __getattr__(self, key):
        return getattr(self.native, key)

    def get_ego_past_trajectory(self, iteration, time_horizon, num_samples=None):
        if iteration != 0:
            raise ValueError("Source context only initializes the physical world")
        if self._bootstrap is not None:
            yield from self._bootstrap[0]
            return
        for i in (-3, -2, -1):
            yield self.native.get_ego_state_at_iteration(i)

    def get_past_tracked_objects(self, iteration, time_horizon, num_samples=None):
        if iteration != 0:
            raise ValueError("Source context only initializes the physical world")
        if self._bootstrap is not None:
            yield from self._bootstrap[1]
            return
        for i in (-3, -2, -1):
            yield self.native.get_tracked_objects_at_iteration(i)


def require_dense_history(history, anchor_us, *, steps=5, interval_us=200000, tolerance_us=1000):
    """No repeated source frame can masquerade as native dense history."""
    times = np.array([int(v["ego"][-1]) for v in history], dtype=np.int64)
    targets = anchor_us - np.arange(steps - 1, -1, -1) * interval_us
    if not len(times):
        raise ValueError("Dense physical teacher history unavailable")
    indices = np.argmin(abs(times[:, None] - targets[None, :]), axis=0)
    if len(set(indices.tolist())) != steps or np.any(abs(times[indices] - targets) > tolerance_us):
        raise ValueError("Dense physical teacher history unavailable; execute arrival prefix first")
    return {
        "sampled_timestamp_us": times[indices].tolist(),
        "max_error_us": int(max(abs(times[indices] - targets))),
    }


def verify_current_input(session):
    import copy

    import torch
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )

    from driversi_agents.driverl.closed_loop import (
        DriveRLOneStageController,
        ego_vector,
        make_planner,
        native_tensor_equal,
    )

    support = require_dense_history(session.history, session.ego.time_us)
    from driversi_agents.driverl import random_state

    ambient_rng = random_state.capture()
    try:
        reference = make_planner(session.config, session.initialization, session.scenario)
        controller = DriveRLOneStageController(session.scenario, **session.configuration["controller"])
        memory = session._runtime_memory(session.planner, session.controller)
        session._restore_runtime(reference, controller, memory)
        session._set_search_tick(session.world.observe().step)
        if session.planner._tts_selector is None:
            reference._tts_selector = None
    finally:
        random_state.restore(ambient_rng)
    inp = session.planner_input()
    try:
        a = reference._feature_builder.build(copy.deepcopy(inp), session.initialization)
        b = session.planner._feature_builder.build(inp, session.initialization)
        fields = []
        for name, value in vars(a).items():
            if isinstance(value, torch.Tensor):
                if not native_tensor_equal(value, getattr(b, name)):
                    raise AssertionError("Native input mismatch: " + name)
                fields.append(name)
        forward_rng = random_state.capture()
        ta = reference.compute_planner_trajectory(copy.deepcopy(inp))
        random_state.restore(forward_rng)
        tb = session.planner.compute_planner_trajectory(inp)
        np.testing.assert_array_equal(ta.raw_action.numpy(), tb.raw_action.numpy())
        nxt = SimulationIteration(
            TimePoint(session.ego.time_us + 100000), session.world.observe().step + 1
        )
        controller.update_state(inp.iteration, nxt, session.ego, ta)
        session.controller.update_state(inp.iteration, nxt, session.ego, tb)
        np.testing.assert_array_equal(
            ego_vector(controller.get_state()), ego_vector(session.controller.get_state())
        )
        return dict(
            history_support=support,
            tensor_fields=fields,
            feature_max_abs_error=0.0,
            action_max_abs_error=0.0,
            state_max_abs_error=0.0,
            scope="same_input_native_runtime_only_not_benchmark_performance",
        )
    finally:
        session._restore_runtime(session.planner, session.controller, memory)
        random_state.restore(ambient_rng)
