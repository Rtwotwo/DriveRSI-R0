"""Small independent geometric oracles; not the native backend's collision flags."""

import numpy as np


def rectangle_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    if not np.isfinite([x, y, yaw, length, width]).all() or min(length, width) <= 0:
        raise ValueError("Rectangle requires finite pose and positive dimensions")
    local = np.asarray([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.asarray([[c, s], [-s, c]]) + [x, y]


def rectangles_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating-axis test, including exact boundary contact, for four corners."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != (4, 2) or second.shape != (4, 2):
        raise ValueError("Expected two rectangles with four xy corners")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Non-finite corners")
    for polygon in (first, second):
        for edge in np.diff(np.vstack((polygon, polygon[0])), axis=0):
            axis = np.asarray([-edge[1], edge[0]])
            a, b = first @ axis, second @ axis
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True
