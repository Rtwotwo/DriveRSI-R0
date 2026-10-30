"""Native PDM metrics on an actually executed counterfactual world.

This is a branch evaluation protocol, not a logged NAVSIM benchmark cache.
The fixed PDM reference is generated at the reached anchor before branching.
"""

from dataclasses import asdict

import numpy as np


def prepare_reference(planner_input, initialization):
    from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
    from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
        BatchIDMPolicy,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_states_to_state_array,
    )
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    # Exact reference recipe from native MetricCacheProcessor.__init__.
    planner = PDMClosedPlanner(
        trajectory_sampling=TrajectorySampling(num_poses=50, interval_length=0.1),
        proposal_sampling=sampling,
        idm_policies=BatchIDMPolicy(
            speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
            fallback_target_velocity=15.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.5,
            decel_max=3.0,
        ),
        lateral_offsets=[-1.0, 1.0],
        map_radius=100,
    )
    planner.initialize(initialization)
    trajectory = planner.compute_planner_trajectory(planner_input)
    ego = planner_input.history.current_state[0]
    times = [TimePoint(ego.time_us + i * 100000) for i in range(41)]
    states = ego_states_to_state_array(trajectory.get_state_at_times(times))
    reference = PDMSimulator(sampling).simulate_proposals(states[None], ego)[0]
    return dict(
        reference=reference,
        trajectory=trajectory,
        centerline=planner._centerline,
        route_lane_ids=list(planner._route_lane_dict),
        route_lane_polygons={key: lane.polygon for key, lane in planner._route_lane_dict.items()},
        drivable_area_map=planner._drivable_area_map,
        anchor_timestamp_us=ego.time_us,
        initial_ego=ego,
    )


def score_execution(context, egos, observations):
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
        PDMObservation,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_states_to_state_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    # Four-second score and the additional one second used by native TTC.
    if len(egos) != 51 or len(observations) != 51:
        raise ValueError("Counterfactual PDM requires 5 seconds of actual world states")
    times = np.array([e.time_us for e in egos], dtype=np.int64)
    expected = context["anchor_timestamp_us"] + np.arange(51) * 100000
    if not np.array_equal(times, expected):
        raise ValueError("Counterfactual clocks differ or have missing states")
    states = ego_states_to_state_array(egos[:41])
    initial = ego_states_to_state_array([context["initial_ego"]])[0]
    if not np.allclose(states[0], initial, rtol=0, atol=1e-7):
        raise ValueError("Reference and executed branch start at different states")
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    observation = PDMObservation(
        TrajectorySampling(num_poses=50, interval_length=0.1),
        sampling,
        100,
        observation_sample_res=1,
    )
    observation.update_detections_tracks(observations)
    scorer = PDMScorer(sampling)
    scores = scorer.score_proposals(
        np.stack([context["reference"], states]),
        observation,
        context["centerline"],
        context["route_lane_ids"],
        context["drivable_area_map"],
    )
    out = {"score": float(scores[1]), "reference_score_in_branch_world": float(scores[0])}
    for key, idx in [
        ("no_at_fault_collisions", MultiMetricIndex.NO_COLLISION),
        ("drivable_area_compliance", MultiMetricIndex.DRIVABLE_AREA),
    ]:
        out[key] = float(scorer._multi_metrics[idx, 1])
    for key, idx in [
        ("ego_progress", WeightedMetricIndex.PROGRESS),
        ("time_to_collision_within_bound", WeightedMetricIndex.TTC),
        ("comfort", WeightedMetricIndex.COMFORTABLE),
        ("driving_direction_compliance", WeightedMetricIndex.DRIVING_DIRECTION),
    ]:
        out[key] = float(scorer._weighted_metrics[idx, 1])
    out.update(
        protocol="navsim-v1-native-math-executed-counterfactual-v1",
        score_configuration=asdict(scorer._config),
        native_score_threshold=0.95,
        score_pass=bool(scores[1] >= 0.95),
        scored_seconds=4.0,
        actual_world_seconds=5.0,
        logged_benchmark=False,
        trainable=False,
    )
    return out
