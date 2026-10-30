"""Optional native-CUDA experiments for the isolated GPU PDM backend."""

from .triton_backend import (
    run_v1_triton_pipeline,
    score_v1_triton,
    triton_aggregate_v1_scores,
    triton_pairwise_box_intersections,
    triton_points_in_polygon_edges,
    triton_project_points_to_polyline,
    triton_proposals_to_state_array,
    triton_state_array_to_coords,
)

__all__ = [
    "run_v1_triton_pipeline",
    "score_v1_triton",
    "triton_aggregate_v1_scores",
    "triton_pairwise_box_intersections",
    "triton_points_in_polygon_edges",
    "triton_project_points_to_polyline",
    "triton_proposals_to_state_array",
    "triton_state_array_to_coords",
]
