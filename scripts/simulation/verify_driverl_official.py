"""Unmodified nuPlan reactive simulation with a serialized-input shadow planner."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from driverl.nuplan.controller import DriveRLOneStageController
from hydra.utils import instantiate
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from omegaconf import OmegaConf

from driversi.contracts.scene import Scene
from driversi.evaluation.metrics.nuplan_metrics import score_history
from driversi_agents.driverl.closed_loop import (
    ego_vector,
    make_planner,
    native_scenario,
    pack_objects,
    unpack_ego,
    unpack_objects,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[2]
    repo = root / "agents/driverl/native/DriveRL"
    scene = Scene.load(a.source / "scene.npz")
    scenario = native_scenario(
        scene, "/data/xinyao/navsim_workspace/dataset/maps", duration=float(scene.times[-1]) + 0.05
    )
    if scenario.get_number_of_iterations() != round(float(scene.times[-1]) * 10) + 1:
        raise ValueError("Official source/evaluation window mismatch")
    cfg_root = repo / "nuplan-devkit/nuplan/planning/script/config/simulation"
    controller_config = {
        k: v
        for k, v in yaml.safe_load(
            (cfg_root / "ego_controller/driverl_one_stage_controller.yaml").read_text()
        ).items()
        if not k.startswith("_")
    }
    controller_config["device"] = a.device
    controller = DriveRLOneStageController(scenario, **controller_config)
    reference = DriveRLOneStageController(scenario, **controller_config)
    observations = instantiate(
        OmegaConf.load(cfg_root / "observation/idm_agents_observation.yaml"), scenario=scenario
    )
    simulation = Simulation(
        SimulationSetup(StepSimulationTimeController(scenario), observations, controller, scenario),
        simulation_history_buffer_duration=1.0,
    )
    initialization = simulation.initialize()
    config = dict(
        config_path=str(repo / "release/configs/driverl_teacher.yaml"),
        checkpoint_path=str(root / "weights/driverl/checkpoint_2400.pt"),
        device=a.device,
        strict_checkpoint=True,
        sampling_method="argmax",
        trajectory_mode="action",
        compile_agent=False,
        tts_enabled=False,
    )
    native = make_planner(config, initialization)
    adapted = make_planner(config, initialization)
    checks = []
    rows = []
    while simulation.is_simulation_running():
        inp = simulation.get_planner_input()
        egos = inp.history.ego_states
        objects = inp.history.observations
        clone = PlannerInput(
            inp.iteration,
            SimulationHistoryBuffer.initialize_from_list(
                len(egos),
                [unpack_ego(ego_vector(e)) for e in egos],
                [unpack_objects(pack_objects(o), e.time_us) for o, e in zip(objects, egos)],
                inp.history.sample_interval,
            ),
            copy.deepcopy(inp.traffic_light_data),
        )
        a_traj = native.compute_planner_trajectory(inp)
        b_traj = adapted.compute_planner_trajectory(clone)
        if (
            not np.isfinite(a_traj.raw_action.numpy()).all()
            or not np.isfinite(b_traj.raw_action.numpy()).all()
        ):
            raise AssertionError("Non-finite official action")
        np.testing.assert_array_equal(a_traj.raw_action.numpy(), b_traj.raw_action.numpy())
        np.testing.assert_array_equal(
            [a_traj.jerk_long, a_traj.lat_command], [b_traj.jerk_long, b_traj.lat_command]
        )
        index = inp.iteration.index
        if index + 1 < scenario.get_number_of_iterations() - 1:
            reference.update_state(
                clone.iteration,
                SimulationIteration(scenario.get_time_point(index + 1), index + 1),
                clone.history.ego_states[-1],
                b_traj,
            )
        rows.append(
            dict(
                ego=ego_vector(egos[-1]),
                objects=pack_objects(objects[-1]),
                lights=[
                    dict(
                        status=x.status.name,
                        lane_connector_id=x.lane_connector_id,
                        timestamp=x.timestamp,
                    )
                    for x in inp.traffic_light_data
                ],
            )
        )
        simulation.propagate(a_traj)
        if index + 1 < scenario.get_number_of_iterations() - 1:
            np.testing.assert_array_equal(
                ego_vector(controller.get_state()), ego_vector(reference.get_state())
            )
        checks.append(dict(tick=index, action_error=0.0, state_error=0.0))
        if index % 30 == 0:
            print(index, flush=True)
            (a.output / "native-history-partial.json").write_text(json.dumps(rows, allow_nan=False))
    (a.output / "native-history.json").write_text(json.dumps(rows, allow_nan=False))
    (a.output / "parity.json").write_text(
        json.dumps(
            dict(
                status="passed",
                steps=len(checks),
                checks=checks,
                history_buffer_seconds=1.0,
                world="official_nuplan_IDM",
                official_benchmark=False,
            ),
            indent=2,
        )
    )
    score = score_history(
        simulation.history,
        scenario,
        a.output / "official-metrics",
        repo,
        world_protocol="official_nuplan_IDM_on_custom_60_second_logs",
    )
    (a.output / "official-score.json").write_text(json.dumps(score, indent=2))
    print(json.dumps(score), flush=True)


if __name__ == "__main__":
    main()
