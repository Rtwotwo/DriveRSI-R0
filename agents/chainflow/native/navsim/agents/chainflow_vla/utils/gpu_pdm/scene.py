"""Pack NAVSIM metric-cache scene objects into padded scorer tensors.

This is intentionally the only GPU-PDM module that touches Python/NAVSIM
objects. It runs once per scene or offline; proposal simulation and scoring
remain tensor-only.
"""

from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from .scorer import PreparedPDMScene


DRIVABLE_LAYERS = frozenset(
    {
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA,
        SemanticMapLayer.CARPARK_AREA,
    }
)
LANE_LAYERS = frozenset({SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR})


@dataclass(frozen=True)
class _SceneData:
    origin: Tensor
    agent_corners: Tensor
    agent_valid: Tensor
    agent_is_dynamic: Tensor
    agent_is_stopped: Tensor
    agent_is_red_light: Tensor
    agent_initially_collided: Tensor
    map_edge_start: Tensor
    map_edge_end: Tensor
    map_edge_polygon: Tensor
    map_polygon_min: Tensor
    map_polygon_max: Tensor
    lane_polygon_mask: Tensor
    drivable_polygon_mask: Tensor
    route_lane_polygon_mask: Tensor
    intersection_polygon_mask: Tensor
    centerline: Tensor


def load_metric_cache(path: str | Path) -> Any:
    """Load one LZMA-compressed NAVSIM metric-cache pickle."""

    with lzma.open(Path(path), "rb") as file:
        return pickle.load(file)


def _polygon_ring_tensor(ring: Any, dtype: torch.dtype, origin: Tensor) -> Tensor:
    coordinates = list(ring.coords)
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]
    vertices = (torch.tensor(coordinates, dtype=torch.float64) - origin).to(dtype)
    if len(vertices) > 1:
        vertices = vertices[(vertices != torch.roll(vertices, 1, 0)).any(dim=-1)]
    changed = True
    while changed and len(vertices) > 3:
        previous = torch.roll(vertices, 1, 0)
        following = torch.roll(vertices, -1, 0)
        incoming = vertices - previous
        outgoing = following - vertices
        cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        same_direction = (incoming * outgoing).sum(dim=-1) >= 0
        keep = ~((cross == 0) & same_direction)
        changed = bool((~keep).any())
        vertices = vertices[keep]
    if len(vertices) < 3:
        raise ValueError("polygon rings must contain at least three distinct vertices")
    return vertices


def _box_corners(geometry: Any, dtype: torch.dtype, origin: Tensor) -> Tensor:
    corners = _polygon_ring_tensor(geometry.exterior, dtype, origin)
    if corners.shape != (4, 2):
        raise ValueError(
            f"PDM tracked-object geometry must be an oriented box with four corners, got {tuple(corners.shape)}"
        )
    return corners


def _append_ring_edges(
    ring: Any,
    polygon_idx: int,
    dtype: torch.dtype,
    origin: Tensor,
    edge_starts: list[Tensor],
    edge_ends: list[Tensor],
    edge_polygons: list[Tensor],
) -> None:
    vertices = _polygon_ring_tensor(ring, dtype, origin)
    next_vertices = torch.roll(vertices, shifts=-1, dims=0)
    valid = (vertices != next_vertices).any(dim=-1)
    edge_starts.append(vertices[valid])
    edge_ends.append(next_vertices[valid])
    edge_polygons.append(
        torch.full((int(valid.sum()),), polygon_idx, dtype=torch.long)
    )


def _extract_scene(
    metric_cache: Any,
    observation_steps: int,
    dtype: torch.dtype,
    local_frame: bool,
) -> _SceneData:
    origin = torch.tensor(
        (metric_cache.ego_state.rear_axle.x, metric_cache.ego_state.rear_axle.y),
        dtype=torch.float64,
    )
    if not local_frame:
        origin.zero_()
    observation = metric_cache.observation
    occupancy_maps = [observation[time_idx] for time_idx in range(observation_steps)]
    red_light_marker = observation.red_light_token

    tokens: list[str] = []
    token_set: set[str] = set()
    for occupancy_map in occupancy_maps:
        for token in occupancy_map.tokens:
            if red_light_marker in token or token in token_set:
                continue
            token_set.add(token)
            tokens.append(token)
    token_to_index = {token: index for index, token in enumerate(tokens)}
    num_agents = len(tokens)

    agent_corners = torch.zeros((observation_steps, num_agents, 4, 2), dtype=dtype)
    agent_valid = torch.zeros((observation_steps, num_agents), dtype=torch.bool)
    for time_idx, occupancy_map in enumerate(occupancy_maps):
        for token in occupancy_map.tokens:
            agent_idx = token_to_index.get(token)
            if agent_idx is None:
                continue
            agent_corners[time_idx, agent_idx] = _box_corners(
                occupancy_map[token], dtype, origin
            )
            agent_valid[time_idx, agent_idx] = True

    agent_is_dynamic = torch.zeros(num_agents, dtype=torch.bool)
    agent_is_stopped = torch.ones(num_agents, dtype=torch.bool)
    for agent_idx, token in enumerate(tokens):
        tracked_object = observation.unique_objects[token]
        agent_is_dynamic[agent_idx] = tracked_object.tracked_object_type in AGENT_TYPES
        if isinstance(tracked_object, Agent):
            agent_is_stopped[agent_idx] = tracked_object.velocity.magnitude() <= 5e-2
    collided_tokens = set(observation.collided_track_ids)
    agent_initially_collided = torch.tensor(
        [token in collided_tokens for token in tokens], dtype=torch.bool
    )

    drivable_map = metric_cache.drivable_area_map
    route_lane_ids = set(metric_cache.route_lane_ids)
    edge_starts: list[Tensor] = []
    edge_ends: list[Tensor] = []
    edge_polygons: list[Tensor] = []
    polygon_mins: list[tuple[float, float]] = []
    polygon_maxs: list[tuple[float, float]] = []
    lane_mask: list[bool] = []
    drivable_mask: list[bool] = []
    route_lane_mask: list[bool] = []
    intersection_mask: list[bool] = []
    for polygon_idx, (token, map_type) in enumerate(
        zip(drivable_map.tokens, drivable_map.map_types)
    ):
        polygon = drivable_map._geometries[polygon_idx]
        if polygon.geom_type != "Polygon":
            raise ValueError(f"unsupported drivable geometry type: {polygon.geom_type}")
        min_x, min_y, max_x, max_y = polygon.bounds
        polygon_mins.append((min_x - float(origin[0]), min_y - float(origin[1])))
        polygon_maxs.append((max_x - float(origin[0]), max_y - float(origin[1])))
        _append_ring_edges(
            polygon.exterior,
            polygon_idx,
            dtype,
            origin,
            edge_starts,
            edge_ends,
            edge_polygons,
        )
        for interior in polygon.interiors:
            _append_ring_edges(
                interior,
                polygon_idx,
                dtype,
                origin,
                edge_starts,
                edge_ends,
                edge_polygons,
            )
        is_lane = map_type in LANE_LAYERS
        lane_mask.append(is_lane)
        drivable_mask.append(map_type in DRIVABLE_LAYERS)
        route_lane_mask.append(is_lane and token in route_lane_ids)
        intersection_mask.append(map_type == SemanticMapLayer.INTERSECTION)

    if hasattr(metric_cache.centerline, "discrete_path"):
        centerline_points = [
            (state.x, state.y) for state in metric_cache.centerline.discrete_path
        ]
    else:
        centerline_points = list(metric_cache.centerline.coords)
    centerline = (torch.tensor(centerline_points, dtype=torch.float64) - origin).to(dtype)
    if centerline.shape[0] < 2:
        raise ValueError("PDM centerline must contain at least two points")

    map_edge_start = torch.cat(edge_starts, dim=0) if edge_starts else torch.empty((0, 2), dtype=dtype)
    map_edge_end = torch.cat(edge_ends, dim=0) if edge_ends else torch.empty((0, 2), dtype=dtype)
    map_edge_polygon = (
        torch.cat(edge_polygons, dim=0) if edge_polygons else torch.empty(0, dtype=torch.long)
    )

    return _SceneData(
        origin=origin,
        agent_corners=agent_corners,
        agent_valid=agent_valid,
        agent_is_dynamic=agent_is_dynamic,
        agent_is_stopped=agent_is_stopped,
        agent_is_red_light=torch.zeros(num_agents, dtype=torch.bool),
        agent_initially_collided=agent_initially_collided,
        map_edge_start=map_edge_start,
        map_edge_end=map_edge_end,
        map_edge_polygon=map_edge_polygon,
        map_polygon_min=torch.tensor(polygon_mins, dtype=dtype).reshape(-1, 2),
        map_polygon_max=torch.tensor(polygon_maxs, dtype=dtype).reshape(-1, 2),
        lane_polygon_mask=torch.tensor(lane_mask, dtype=torch.bool),
        drivable_polygon_mask=torch.tensor(drivable_mask, dtype=torch.bool),
        route_lane_polygon_mask=torch.tensor(route_lane_mask, dtype=torch.bool),
        intersection_polygon_mask=torch.tensor(intersection_mask, dtype=torch.bool),
        centerline=centerline,
    )


def _move(tensor: Tensor, device: torch.device, pin_memory: bool) -> Tensor:
    if pin_memory and device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=pin_memory)


def prepare_pdm_scenes(
    metric_caches: Sequence[Any],
    num_steps: int = 41,
    ttc_lookahead_steps: int = 9,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = False,
    local_frame: bool = True,
) -> PreparedPDMScene:
    """Pack a batch of metric-cache objects into one padded tensor scene."""

    if not metric_caches:
        raise ValueError("metric_caches must not be empty")
    if num_steps < 2 or ttc_lookahead_steps < 0:
        raise ValueError("invalid PDM or TTC horizon")
    device = torch.device(device)
    scenes = [
        _extract_scene(cache, num_steps + ttc_lookahead_steps, dtype, local_frame)
        for cache in metric_caches
    ]
    batch_size = len(scenes)
    observation_steps = num_steps + ttc_lookahead_steps
    max_agents = max(scene.agent_corners.shape[1] for scene in scenes)
    max_polygons = max(len(scene.lane_polygon_mask) for scene in scenes)
    max_edges = max(len(scene.map_edge_start) for scene in scenes)
    max_centerline = max(len(scene.centerline) for scene in scenes)

    origin = torch.stack([scene.origin for scene in scenes])
    agent_corners = torch.zeros(
        (batch_size, observation_steps, max_agents, 4, 2), dtype=dtype
    )
    agent_valid = torch.zeros((batch_size, observation_steps, max_agents), dtype=torch.bool)
    agent_is_dynamic = torch.zeros((batch_size, max_agents), dtype=torch.bool)
    agent_is_stopped = torch.ones((batch_size, max_agents), dtype=torch.bool)
    agent_is_red_light = torch.zeros((batch_size, max_agents), dtype=torch.bool)
    agent_initially_collided = torch.ones((batch_size, max_agents), dtype=torch.bool)
    map_edge_start = torch.zeros((batch_size, max_edges, 2), dtype=dtype)
    map_edge_end = torch.zeros((batch_size, max_edges, 2), dtype=dtype)
    map_edge_polygon = torch.zeros((batch_size, max_edges), dtype=torch.long)
    map_edge_valid = torch.zeros((batch_size, max_edges), dtype=torch.bool)
    map_polygon_min = torch.full((batch_size, max_polygons, 2), torch.inf, dtype=dtype)
    map_polygon_max = torch.full((batch_size, max_polygons, 2), -torch.inf, dtype=dtype)
    lane_polygon_mask = torch.zeros((batch_size, max_polygons), dtype=torch.bool)
    drivable_polygon_mask = torch.zeros((batch_size, max_polygons), dtype=torch.bool)
    route_lane_polygon_mask = torch.zeros((batch_size, max_polygons), dtype=torch.bool)
    intersection_polygon_mask = torch.zeros((batch_size, max_polygons), dtype=torch.bool)
    centerline = torch.zeros((batch_size, max_centerline, 2), dtype=dtype)
    centerline_point_mask = torch.zeros((batch_size, max_centerline), dtype=torch.bool)

    for batch_idx, scene in enumerate(scenes):
        num_agents = scene.agent_corners.shape[1]
        agent_corners[batch_idx, :, :num_agents] = scene.agent_corners
        agent_valid[batch_idx, :, :num_agents] = scene.agent_valid
        agent_is_dynamic[batch_idx, :num_agents] = scene.agent_is_dynamic
        agent_is_stopped[batch_idx, :num_agents] = scene.agent_is_stopped
        agent_is_red_light[batch_idx, :num_agents] = scene.agent_is_red_light
        agent_initially_collided[batch_idx, :num_agents] = scene.agent_initially_collided

        num_edges = len(scene.map_edge_start)
        map_edge_start[batch_idx, :num_edges] = scene.map_edge_start
        map_edge_end[batch_idx, :num_edges] = scene.map_edge_end
        map_edge_polygon[batch_idx, :num_edges] = scene.map_edge_polygon
        map_edge_valid[batch_idx, :num_edges] = True

        num_polygons = len(scene.lane_polygon_mask)
        map_polygon_min[batch_idx, :num_polygons] = scene.map_polygon_min
        map_polygon_max[batch_idx, :num_polygons] = scene.map_polygon_max
        lane_polygon_mask[batch_idx, :num_polygons] = scene.lane_polygon_mask
        drivable_polygon_mask[batch_idx, :num_polygons] = scene.drivable_polygon_mask
        route_lane_polygon_mask[batch_idx, :num_polygons] = scene.route_lane_polygon_mask
        intersection_polygon_mask[batch_idx, :num_polygons] = scene.intersection_polygon_mask
        centerline[batch_idx, : len(scene.centerline)] = scene.centerline
        centerline_point_mask[batch_idx, : len(scene.centerline)] = True

    values = {
        "origin": origin,
        "agent_corners": agent_corners,
        "agent_valid": agent_valid,
        "agent_is_dynamic": agent_is_dynamic,
        "agent_is_stopped": agent_is_stopped,
        "agent_is_red_light": agent_is_red_light,
        "agent_initially_collided": agent_initially_collided,
        "map_edge_start": map_edge_start,
        "map_edge_end": map_edge_end,
        "map_edge_polygon": map_edge_polygon,
        "map_edge_valid": map_edge_valid,
        "map_polygon_min": map_polygon_min,
        "map_polygon_max": map_polygon_max,
        "lane_polygon_mask": lane_polygon_mask,
        "drivable_polygon_mask": drivable_polygon_mask,
        "route_lane_polygon_mask": route_lane_polygon_mask,
        "intersection_polygon_mask": intersection_polygon_mask,
        "centerline": centerline,
        "centerline_point_mask": centerline_point_mask,
    }
    return PreparedPDMScene(
        **{name: _move(value, device, pin_memory) for name, value in values.items()}
    )


def prepare_pdm_scene(
    metric_cache: Any,
    **kwargs: Any,
) -> PreparedPDMScene:
    """Pack one metric-cache object with a leading batch dimension."""

    return prepare_pdm_scenes([metric_cache], **kwargs)
