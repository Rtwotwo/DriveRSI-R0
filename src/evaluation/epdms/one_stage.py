from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import SceneFrameType
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex


def infer_start_adjacent_mapping(score_df: pd.DataFrame, time_gap_threshold: float = 0.55) -> Dict[str, str]:
    """
    Infers an adjacent mapping from the score_df DataFrame by start time.
    Each current-token is mapped to its previous-token if they are adjacent.
    Used to create the two-frame extended comfort score (reversed direction).

    :param score_df: DataFrame containing at least 'token', 'log_name', 'start_time'.
    :param time_gap_threshold: Maximum allowed gap (in seconds) between two frames to
                               consider them "adjacent".
    :return: Dictionary mapping each current-token to one previous-token.
    """
    adjacent_mapping: Dict[str, str] = {}

    for log_name, group_df in score_df[score_df["frame_type"] == SceneFrameType.ORIGINAL].groupby("log_name"):
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)

        for i in range(1, len(group_df)):
            prev_row = group_df.iloc[i - 1]
            current_row = group_df.iloc[i]

            prev_token = prev_row["token"]
            current_token = current_row["token"]
            time_diff = current_row["start_time"] - prev_row["start_time"]

            if abs(time_diff) <= time_gap_threshold:
                adjacent_mapping[current_token] = prev_token

    return adjacent_mapping

def create_scene_aggregators(
    all_mappings: Dict[str, str],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []

    for now_frame, previous_frame in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores(one_stage_only=True)

        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df

def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores for each row in pdm_score_df after updating
    the weighted metrics with two-frame extended comfort.

    If 'two_frame_extended_comfort' is NaN for a row, the corresponding
    metric and its weight are set to zero, effectively ignoring it
    during normalization.

    :param pdm_score_df: DataFrame containing PDM scores and metrics.
    :return: A new DataFrame with the computed final scores.
    """
    df = pdm_score_df.copy()

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()  # shape: (N, )
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    mask = np.isnan(two_frame_scores)
    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT

    weighted_metrics[mask, two_frame_idx] = 0.0
    weighted_metrics_array[mask, two_frame_idx] = 0.0

    non_mask = ~mask
    weighted_metrics[non_mask, two_frame_idx] = two_frame_scores[non_mask]

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    total_weight[total_weight == 0.0] = np.nan
    weighted_metric_scores = weighted_sum / total_weight

    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"],
        inplace=True,
    )

    return df
