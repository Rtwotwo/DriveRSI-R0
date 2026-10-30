import sqlite3
from types import SimpleNamespace as NS

import numpy as np
import pytest

from driversi.data.sources.nuplan import compile_scenario, db_anchor, scenario_frames


class Scenario:
    token = "0123456789abcdef"
    log_name = "original-driving-log"
    map_api = NS(map_name="test-city")

    def get_number_of_iterations(self):
        return 21

    def get_time_point(self, i):
        return NS(time_us=1000000 + i * 100000)

    def get_ego_state_at_iteration(self, i):
        centre = NS(x=100 + i * 0.4, y=200.0, heading=np.pi / 2)
        return NS(
            center=centre, car_footprint=NS(oriented_box=NS(length=5.0, width=2.0, height=1.8))
        )

    def get_tracked_objects_at_iteration(self, i):
        items = []
        for token, kind, x, y in [
            ("car", "VEHICLE", 110 + i * 0.3, 201.0),
            ("ped", "PEDESTRIAN", 120.0, 202 + i * 0.1),
            ("cone", "TRAFFIC_CONE", 130.0, 200.0),
        ]:
            items.append(
                NS(
                    track_token=token,
                    tracked_object_type=NS(name=kind),
                    center=NS(x=x, y=y, heading=0.0),
                    box=NS(
                        length=4.0 if token == "car" else 0.5,
                        width=2.0 if token == "car" else 0.5,
                        height=1.5,
                    ),
                )
            )
        return NS(tracked_objects=items)

    def get_traffic_light_status_at_iteration(self, i):
        return [
            NS(
                lane_connector_id=10,
                status=NS(name="UNKNOWN" if i < 10 else "YELLOW"),
                timestamp=1000000 + i * 100000,
            )
        ]

    def get_mission_goal(self):
        return NS(x=140.0, y=200.0, heading=0.0)

    def get_route_roadblock_ids(self):
        return ["101", "102"]


def test_sdk_coordinate_roundtrip_and_all_participant_types():
    source = Scenario()
    scene = compile_scenario(source, duration=2, split="mini")
    origin = np.asarray(scene.metadata["origin_xyz"])
    np.testing.assert_allclose(scene.poses[0, 0, :2] + origin[:2], [100, 200])
    car = scene.actor_ids.index("car")
    np.testing.assert_allclose(scene.poses[-1, car, :2] + origin[:2], [116, 201])
    np.testing.assert_allclose(scene.dimensions[0], [5, 2, 1.8])
    assert set(scene.actor_ids) == {"ego", "car", "ped", "cone"}
    assert scene.kinds[scene.actor_ids.index("cone")] == 10
    assert len(scene.metadata["scenario_id"]) == 16
    records = scene.metadata["nuplan_source"]["signals"]
    assert {r["states"][0]["status"] for r in records} == {"unknown", "yellow"}
    assert all(not r["available"] for r in scene.metadata["traffic_lights"])
    assert scene.metadata["source_log"] == source.log_name


def test_missing_track_identity_is_rejected():
    class Broken(Scenario):
        def get_tracked_objects_at_iteration(self, i):
            result = super().get_tracked_objects_at_iteration(i)
            result.tracked_objects[0].track_token = None
            return result

    with pytest.raises(ValueError, match="stable"):
        scenario_frames(Broken())


def test_source_exhaustion_and_protected_source_labels_are_preserved():
    scene = compile_scenario(Scenario(), duration=22, split="heldout")
    assert scene.times[-1] == 2 and scene.metadata["split"] == "heldout"
    assert scene.metadata["horizon_plan"]["limiting_factors"] == ["source_exhaustion"]


def test_db_anchor_is_read_only_and_checks_token(tmp_path):
    db = tmp_path / "sample.db"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE log(token BLOB,map_version TEXT); CREATE TABLE lidar(token BLOB, log_token BLOB); CREATE TABLE lidar_pc(token BLOB,lidar_token BLOB,timestamp INTEGER);"
        )
        c.execute("INSERT INTO log VALUES (?,?)", (b"log", "city"))
        c.execute("INSERT INTO lidar VALUES (?,?)", (b"lidar", b"log"))
        c.execute("INSERT INTO lidar_pc VALUES (?,?,?)", (b"a", b"lidar", 123))
    original = db.read_bytes()
    assert db_anchor(db, "61") == (123, "city")
    assert db.read_bytes() == original
    with pytest.raises(ValueError, match="not found"):
        db_anchor(db, "62")
    with pytest.raises(FileNotFoundError):
        db_anchor(tmp_path / "absent.db", "61")
    assert not (tmp_path / "absent.db").exists()
