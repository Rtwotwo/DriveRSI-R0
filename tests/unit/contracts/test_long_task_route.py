import numpy as np
import pytest
from driversi.contracts.driving_task import tasks_from_metadata, TASK_VERSION
from driversi.contracts.road import Route


def metadata(n):
    route = Route(np.c_[np.linspace(0, 1031., n), np.zeros(n)])
    return dict(driving_tasks=dict(version=TASK_VERSION, actors={"ego":dict(route=route.to_dict(), endpoint="support_boundary", corridor_m=2., rule_status="unknown")}),task_goals_xy={"ego":route.points[-1].tolist()}), route


def test_full_loaded_4143_point_route_survives_without_geometry_loss():
    meta, route = metadata(4143)
    recovered, _ = tasks_from_metadata(meta, ("ego",))["ego"]
    np.testing.assert_array_equal(recovered.points, route.points)
    np.testing.assert_array_equal(recovered.distances, route.distances)


def test_larger_payload_remains_bounded():
    meta, _ = metadata(8193)
    with pytest.raises(ValueError, match="8192"):
        tasks_from_metadata(meta,("ego",))
