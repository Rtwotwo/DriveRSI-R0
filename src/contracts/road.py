"""Versioned lane topology and bounded local routes, independent of log futures."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LANE_MATCH_DISTANCE_LIMIT_M = 4.0
LANE_MATCH_HEADING_LIMIT_RAD = np.pi / 3
LANE_MATCH_HEADING_WEIGHT_M_PER_RAD = 2.0
LANE_MATCH_SCORE_TOLERANCE_M = .25


@dataclass(frozen=True)
class LaneMatchCandidate:
    node_id: str
    distance_m: float
    heading_error_rad: float
    score_m: float
    arc_length_m: float
    length_m: float
    outgoing: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "distance_m": self.distance_m,
                "heading_error_rad": self.heading_error_rad, "score_m": self.score_m,
                "arc_length_m": self.arc_length_m,
                "length_m": self.length_m, "outgoing": list(self.outgoing)}


@dataclass(frozen=True)
class LaneMatch:
    """Current-pose evidence; a match never infers an unobserved turn intention."""

    status: str
    reason: str
    selected_node_id: str | None
    candidates: tuple[LaneMatchCandidate, ...] = ()
    ambiguous_node_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"schema_version": 1, "status": self.status, "reason": self.reason,
                "selected_node_id": self.selected_node_id,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "ambiguous_node_ids": list(self.ambiguous_node_ids),
                "matching_policy": {"version": "pose-distance-heading-v1",
                                    "score": "distance_m + heading_weight_m_per_rad * heading_error_rad",
                                    "distance_limit_m": LANE_MATCH_DISTANCE_LIMIT_M,
                                    "heading_limit_rad": LANE_MATCH_HEADING_LIMIT_RAD,
                                    "heading_weight_m_per_rad": LANE_MATCH_HEADING_WEIGHT_M_PER_RAD,
                                    "ambiguity_score_tolerance_m": LANE_MATCH_SCORE_TOLERANCE_M,
                                    "seam_projection_limit_m": 1.0, "seam_geometry_gap_limit_m": .25}}


class LaneMatchError(ValueError):
    def __init__(self, diagnostic: LaneMatch):
        self.diagnostic = diagnostic
        message = {"ambiguous_initial_lane": "Ambiguous initial lane match",
                   "no_compatible_lane": "No direction-compatible lane within 4 m",
                   "outside_loaded_region": "Pose outside loaded map region",
                   "explicit_node_not_candidate": "Explicit initial node is not a direction-compatible lane within 4 m"}
        super().__init__(message.get(diagnostic.reason, diagnostic.reason))


@dataclass(frozen=True)
class Route:
    points: np.ndarray
    node_ids: tuple[str, ...] = ()
    end_reason: str = "explicit_endpoint"
    speed_limit_mps: float | None = None

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float).copy()
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Route requires at least two finite xy points")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-6):
            raise ValueError("Route contains duplicate consecutive points")
        if self.speed_limit_mps is not None and (not np.isfinite(self.speed_limit_mps) or self.speed_limit_mps <= 0):
            raise ValueError("Known speed limit must be positive and finite")
        points.setflags(write=False)
        object.__setattr__(self, "points", points)

    @property
    def distances(self) -> np.ndarray:
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]

    def project(self, xy) -> tuple[float, float, float]:
        """Closest arc length, unsigned distance and tangent heading."""
        edges = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        fractions = np.clip(np.sum((np.asarray(xy) - self.points[:-1]) * edges, axis=1) / lengths**2, 0, 1)
        distances = np.linalg.norm(self.points[:-1] + fractions[:, None] * edges - xy, axis=1)
        index = int(np.argmin(distances))
        return (float(self.distances[index] + fractions[index] * lengths[index]),
                float(distances[index]), float(np.arctan2(*edges[index, ::-1])))

    def project_many(self, xy) -> np.ndarray:
        """Same projection/tie rule as project(), batched over query positions."""
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Expected Nx2 positions")
        if not len(xy):
            return np.empty((0, 3), dtype=float)
        edges = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        fractions = np.clip(np.sum((xy[:, None, :] - self.points[None, :-1, :]) * edges[None, :, :], axis=2) / lengths[None, :]**2, 0, 1)
        distances = np.linalg.norm(self.points[None, :-1, :] + fractions[:, :, None] * edges[None, :, :] - xy[:, None, :], axis=2)
        index = np.argmin(distances, axis=1)
        rows = np.arange(len(xy))
        return np.column_stack((self.distances[index] + fractions[rows, index] * lengths[index],
            distances[rows, index], np.arctan2(edges[index, 1], edges[index, 0])))

    def at(self, distance: float) -> np.ndarray:
        return np.asarray([np.interp(distance, self.distances, self.points[:, i]) for i in (0, 1)])

    def to_dict(self) -> dict:
        return {"points": self.points.tolist(), "node_ids": self.node_ids,
                "end_reason": self.end_reason, "speed_limit_mps": self.speed_limit_mps}


@dataclass
class RoadGraph:
    metadata: dict
    nodes: dict[str, dict]

    def validate(self) -> None:
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported road graph schema")
        if not self.nodes:
            raise ValueError("Empty road graph")
        for key in ("map_name", "map_version", "origin_xyz", "bounds_local"):
            if key not in self.metadata:
                raise ValueError(f"Missing road graph metadata: {key}")
        for key, node in self.nodes.items():
            if node["kind"] not in {"lane", "connector"} or not key.startswith(node["kind"] + ":"):
                raise ValueError("Invalid lane node identity")
            Route(node["points"], speed_limit_mps=node["speed_limit_mps"])
            if len(node["outgoing"]) != len(set(node["outgoing"])):
                raise ValueError("Duplicate lane graph edges")

    def validate_scene(self, scene) -> None:
        self.validate()
        if (self.metadata["map_name"] != scene.metadata["map_name"]
                or self.metadata["map_version"] != scene.metadata.get("map_version")
                or not np.array_equal(self.metadata["origin_xyz"], scene.metadata["origin_xyz"])):
            raise ValueError("Road graph map version or coordinate origin does not match scene")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": self.metadata, "nodes": self.nodes}, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        graph = cls(**json.loads(path.read_text(encoding="utf-8")))
        graph.validate()
        return graph

    def match_pose(self, x: float, y: float, yaw: float, *, initial_node_id: str | None = None) -> LaneMatch:
        if not np.isfinite([x, y, yaw]).all():
            raise ValueError("Invalid route matching request")
        xmin, ymin, xmax, ymax = self.metadata["bounds_local"]
        if not xmin <= x <= xmax or not ymin <= y <= ymax:
            return LaneMatch("unmatched", "outside_loaded_region", None)
        matches = []
        for node_id, node in self.nodes.items():
            baseline = Route(node["points"])
            arc_length, distance, heading = baseline.project([x, y])
            angle = abs(np.arctan2(np.sin(yaw-heading), np.cos(yaw-heading)))
            if distance <= LANE_MATCH_DISTANCE_LIMIT_M and angle <= LANE_MATCH_HEADING_LIMIT_RAD:
                score = distance + LANE_MATCH_HEADING_WEIGHT_M_PER_RAD * angle
                matches.append(LaneMatchCandidate(node_id, distance, float(angle), float(score), arc_length,
                                                  float(baseline.distances[-1]), tuple(node["outgoing"])))
        # Node ID orders diagnostics only; it must never break an ambiguous match.
        matches.sort(key=lambda match: (match.score_m, match.distance_m, match.heading_error_rad, match.node_id))
        candidates = tuple(matches)
        if initial_node_id is not None:
            if any(match.node_id == initial_node_id for match in matches):
                return LaneMatch("matched", "explicit_initial_node", initial_node_id, candidates)
            return LaneMatch("unmatched", "explicit_node_not_candidate", None, candidates)
        if not matches:
            return LaneMatch("unmatched", "no_compatible_lane", None)
        nearest = matches[0]
        near = [match for match in matches
                if match.score_m-nearest.score_m < LANE_MATCH_SCORE_TOLERANCE_M]
        if len(near) == 1:
            return LaneMatch("matched", "unique_best_score", nearest.node_id, candidates)
        selected = self._match_topology_seam(near)
        if selected is not None:
            return LaneMatch("matched", "continuous_topology_seam", selected, candidates)
        return LaneMatch("ambiguous", "ambiguous_initial_lane", None, candidates,
                         tuple(match.node_id for match in near))

    def _match_topology_seam(self, candidates: list[LaneMatchCandidate]) -> str | None:
        """Resolve only one connected, non-branching chain at a baseline seam."""
        by_id = {candidate.node_id: candidate for candidate in candidates}
        successors = {}
        for candidate in candidates:
            if len(candidate.outgoing) != 1 or candidate.outgoing[0] not in by_id:
                continue
            following = by_id[candidate.outgoing[0]]
            gap = np.linalg.norm(np.asarray(self.nodes[candidate.node_id]["points"][-1])
                                 - self.nodes[following.node_id]["points"][0])
            if candidate.length_m-candidate.arc_length_m <= 1 and following.arc_length_m <= 1 and gap <= .25:
                successors[candidate.node_id] = following.node_id
        roots = set(by_id)-set(successors.values())
        if len(roots) != 1:
            return None
        visited = []
        current = next(iter(roots))
        while current not in visited:
            visited.append(current)
            if current not in successors:
                return current if len(visited) == len(candidates) else None
            current = successors[current]
        return None

    def route_from_pose(self, x: float, y: float, yaw: float, *, max_nodes: int = 32,
                        initial_node_id: str | None = None, successors: dict[str, str] | None = None) -> Route:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= 32:
            raise ValueError("Invalid route node budget")
        choices = {} if successors is None else successors
        if not isinstance(choices, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in choices.items()):
            raise ValueError("Explicit successor instructions require a node-to-node mapping")
        for upstream, downstream in choices.items():
            if upstream not in self.nodes or downstream not in self.nodes:
                raise ValueError(f"Explicit successor instruction references a missing node: {upstream} -> {downstream}")
            if downstream not in self.nodes[upstream]["outgoing"]:
                raise ValueError(f"Explicit successor instruction is not adjacent: {upstream} -> {downstream}")
        match = self.match_pose(x, y, yaw, initial_node_id=initial_node_id)
        if match.selected_node_id is None:
            raise LaneMatchError(match)
        node_id = match.selected_node_id
        visited, points, limits = [], [], []
        end_reason = "node_budget"
        for _ in range(max_nodes):
            if node_id in visited:
                if choices:
                    raise ValueError("Route with explicit successor instructions enters a cycle")
                end_reason = "cycle"
                break
            node = self.nodes[node_id]
            part = np.asarray(node["points"])
            if points and np.linalg.norm(part[0] - points[-1]) > 2:
                end_reason = "geometry_gap"
                break
            for point in part:
                if not points or np.linalg.norm(point-points[-1]) > 1e-6:
                    points.append(point)
            visited.append(node_id)
            limits.append(node["speed_limit_mps"])
            outgoing = node["outgoing"]
            if node_id in choices:
                node_id = choices[node_id]
            elif len(outgoing) != 1:
                end_reason = "branch" if outgoing else "dead_end"
                break
            else:
                node_id = outgoing[0]
            if node_id not in self.nodes:
                end_reason = "unloaded_exit"
                break
            if choices and node_id in visited:
                raise ValueError("Route with explicit successor instructions enters a cycle")
        traversed = dict(zip(visited[:-1], visited[1:]))
        unused = sorted(upstream for upstream, downstream in choices.items() if traversed.get(upstream) != downstream)
        if unused:
            raise ValueError("Unused explicit successor instructions: " + ", ".join(unused))
        known_limits = [value for value in limits if value is not None]
        return Route(np.asarray(points), tuple(visited), end_reason,
                     min(known_limits) if known_limits else None)
