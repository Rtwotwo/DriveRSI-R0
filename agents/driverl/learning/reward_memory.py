"""Ego reward state independent of transient neighbor and map-array ordering."""

import copy

import numpy as np
import torch

SCALARS = (
    "_nuplan_ttc_triggered",
    "_comfort_triggered",
    "_cross_lane_deadband_counter",
    "_direction_progress_buffer",
    "_direction_progress_last",
)


def world_baselines(data, ego):
    points = (
        data.lanes_centers_lines if data.lanes_centers_lines.numel() else data.lanes_centers_points
    )
    xy = points[0, ..., :2].detach().cpu().numpy().astype(float)
    h = ego.rear_axle.heading
    rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    return xy @ rot.T + np.array([ego.rear_axle.x, ego.rear_axle.y])


class RewardMemory:
    def __init__(self):
        self.scalars = {}
        self.collisions = set()
        self.goal = False
        self.baseline = None

    def apply(self, data, keys, baselines):
        shape = data.npc_mask_all.shape[:2]
        device = data.npc_mask_all.device
        for name, value in self.scalars.items():
            target = torch.zeros((*shape, *value.shape[2:]), device=device, dtype=value.dtype)
            target[:, :1] = value.to(device)
            setattr(data, name, target)
        if self.baseline is not None:
            # Geometric identity only, not a scoring tolerance. Reordered map rows
            # retain progress; a changed polyline cannot inherit another lane's state.
            idx = -1
            if baselines.shape[1:] == self.baseline.shape:
                errors = np.max(
                    np.abs(baselines - self.baseline), axis=tuple(range(1, baselines.ndim))
                )
                matches = np.flatnonzero(errors < 1e-3)
                if len(matches) == 1:
                    idx = int(matches[0])
            target = torch.full(shape, -1, device=device, dtype=torch.long)
            target[0, 0] = idx
            data._direction_baseline_last = target
        matrix = torch.zeros((*shape, shape[1]), device=device, dtype=torch.bool)
        for i, key in enumerate(keys):
            if key in self.collisions:
                matrix[0, 0, i] = True
        data._nuplan_ttc_collided_matrix = matrix
        data.agent_goal_reached_all[:, 0, :] = self.goal

    def capture(self, data, keys, baselines, goal):
        self.scalars = {
            name: getattr(data, name)[:, :1].detach().cpu().clone()
            for name in SCALARS
            if getattr(data, name, None) is not None
        }
        matrix = getattr(data, "_nuplan_ttc_collided_matrix", None)
        if matrix is not None:
            self.collisions.update(key for i, key in enumerate(keys) if bool(matrix[0, 0, i]))
        index = getattr(data, "_direction_baseline_last", None)
        if index is not None:
            self.baseline = baselines[int(index[0, 0])].copy()
        self.goal = self.goal or bool(goal)

    def snapshot(self):
        return copy.deepcopy(vars(self))

    def restore(self, state):
        self.__dict__.update(copy.deepcopy(state))
