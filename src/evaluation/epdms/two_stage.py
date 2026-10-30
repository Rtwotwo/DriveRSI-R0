from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import SceneFrameType
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    all_seen_tokens = set()

    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()

        all_seen_tokens.update(updated_rows["token"])
        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df

def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores after injecting the two-frame extended comfort score
    into the weighted metrics.

    :param pdm_score_df: Original PDM score DataFrame with all tokens
    :return: DataFrame with final score computed
    """
    df = pdm_score_df.reset_index()

    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)

    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores

    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)

    return df

def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the weighted average for all score columns in the dataframe.
    Automatically normalizes weights if they do not sum to 1.
    """
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]

    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()

    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)

def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame, all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]]
) -> pd.Series:
    """
    Args:
        pdm_score_df: DataFrame containing scores for all tokens
        all_mappings: Two-stage mapping dictionary
    Returns:
        Average score calculated using group multiplication
    """
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():

        first_tokens = [pair[0] for pair in second_stage_pairs if len(pair) > 0]
        second_tokens = [pair[1] for pair in second_stage_pairs if len(pair) > 1]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[pdm_score_df["token"].isin(first_tokens)]

        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[pdm_score_df["token"].isin(second_tokens)]

        group1_stage1_scores = calculate_weighted_average_score(group1_stage1_df)
        group1_stage2_scores = calculate_weighted_average_score(group1_stage2_df)

        group2_stage1_scores = calculate_weighted_average_score(group2_stage1_df)
        group2_stage2_scores = calculate_weighted_average_score(group2_stage2_df)

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)

        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores

        group2_scores = group2_stage1_scores * group2_stage2_scores

        avg_scores = (group1_scores + group2_scores) / 2
        all_group_scores.append(avg_scores)

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )
