"""Native v2 metric math on a sealed, actually executed synthetic branch.

The complete native aggregation is reused with all five weighted metrics. This is a branch
admission score, not a score obtained by running the official navtest dataset.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array

from driversi.agents.backends.native import digest, project_root
from driversi.evaluation.epdms import two_stage
from driversi.evaluation.epdms.pickle_compat import load
from driversi.evaluation.epdms.state_codec import unpack_ego
from driversi.rsi.correction.epdms_admission import PROTOCOL, validate_score, scorer_sources
from driversi.runtime.budget import write_json


def evaluate(context_path, expected_sha):
    started = time.monotonic()
    context_path = Path(context_path).resolve(strict=True)
    if digest(context_path) != expected_sha:
        raise ValueError("Executed context changed before EPDMS")
    with context_path.open("rb") as stream:
        data = load(stream)
    if data.get("kind") != "executed_counterfactual_context" or data.get("schema_version") != 1 or data.get("epdms_context_version") != 1:
        raise ValueError("EPDMS requires a v2 executed context with actual lights and history")
    files = data["branch_files"]
    if len(files) != 2 or {Path(p).name for p in files} != {"scene.npz", "teacher.npz"}:
        raise ValueError("Exact teacher branch binding required")
    if any(digest(p) != sha for p, sha in files.items()):
        raise ValueError("Physical branch changed")
    context, egos, tracks, lights = (data[k] for k in ("context", "egos", "observations", "traffic_lights"))
    anchor = context["anchor_timestamp_us"]
    expected = anchor + np.arange(51, dtype=np.int64) * 100000
    if len(egos) != 51 or len(tracks) != 51 or len(lights) != 51:
        raise ValueError("Five seconds of actual branch observations required")
    if not np.array_equal([e.time_us for e in egos], expected):
        raise ValueError("Executed EPDMS clocks differ")
    for stamp, frame in zip(expected, lights):
        if any(signal.timestamp != stamp for signal in frame):
            raise ValueError("Traffic-light timestamps differ from executed branch")
    history = data["prefix_history"]
    if len(history) != 21 or any(v.get("origin") != "world" for v in history):
        raise ValueError("Two seconds of actual generated history required")
    if not np.array_equal([v["ego"][-1] for v in history], anchor + np.arange(-20, 1) * 100000):
        raise ValueError("Generated history is incomplete")
    past = [unpack_ego(v["ego"]) for v in history]
    actual = ego_states_to_state_array(egos[:41])
    if not np.allclose(actual[0], ego_states_to_state_array(past[-1:])[0], rtol=0, atol=1e-7):
        raise ValueError("Generated history and future have different anchors")
    if not np.allclose(actual[0], ego_states_to_state_array([context["initial_ego"]])[0], rtol=0, atol=1e-7):
        raise ValueError("Reference belongs to a different anchor")
    route = {key: SimpleNamespace(polygon=value) for key, value in context["route_lane_polygons"].items()}
    if not route or set(route) != set(context["route_lane_ids"]):
        raise ValueError("Complete reached-anchor route polygons required")
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    observation = PDMObservation(TrajectorySampling(num_poses=50, interval_length=0.1),
                                 sampling, 100, observation_sample_res=1)
    observation.update_detections_tracks(tracks, lights, route, compute_traffic_light_data=True)
    config = PDMScorerConfig(human_penalty_filter=False)
    scorer = PDMScorer(sampling, config)
    rows = scorer.score_proposals(
        np.stack([context["reference"], actual]), observation, context["centerline"],
        context["route_lane_ids"], context["drivable_area_map"], None,
        human_past_trajectory=InterpolatedTrajectory(past))
    raw = rows[1].copy()
    # No isolated-anchor fallback for data admission. The benchmark replay
    # tool retains the upstream policy; generated training data requires EC.
    from driversi.evaluation.epdms.planning_pair import extended_comfort
    raw["two_frame_extended_comfort"] = extended_comfort(data, actual, sampling)
    final = two_stage.compute_final_scores(raw).iloc[0]
    keys = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
            "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
            "lane_keeping", "history_comfort", "two_frame_extended_comfort")
    result = dict(schema_version=1, kind="executed_counterfactual_epdms", protocol=PROTOCOL,
        score=float(final["score"]), score_unit="0_to_1", strict_threshold=0.95,
        passed=bool(final["score"] > 0.95), components={k: float(final[k]) for k in keys},
        two_frame_extended_comfort=float(final["two_frame_extended_comfort"]),
        two_frame_comfort_policy="required_native_adjacent_plans",
        score_configuration=asdict(config), branch_files=files,
        context=dict(path=str(context_path), sha256=expected_sha),
        source_files=scorer_sources(),
        planning_capture=data["adjacent_native_plans"]["capture_file"],
        reference_generator="fixed_reached_anchor_PDM_recipe",
        observation_origin="actual_executed_counterfactual_world",
        history_origin="actual_generated_world", scored_seconds=4.0, actual_world_seconds=5.0,
        original_human_filter_applied=False, logged_benchmark=False,
        native_metric_math_unmodified=True, wall_seconds=time.monotonic() - started)
    validate_score(result, require_pass=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite a metric receipt")
    write_json(args.output, evaluate(args.context, args.context_sha256))


if __name__ == "__main__":
    main()
