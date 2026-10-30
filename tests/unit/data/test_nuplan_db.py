import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from driversi.cli import main
from driversi.contracts.scene import Scene
from driversi.data.scenes.catalog import list_scenes
from driversi.data.scenes.library import compile_catalog
from driversi.data.sources.nuplan_catalog import (
    build_catalog,
    inventory,
    role_for,
    source_group,
    split_rules,
)
from driversi.data.sources.nuplan_db import compile_window, open_db, read_window


def token(i):
    return int(i).to_bytes(8, "big")


def make_db(
    root, partition="boston", name="2021.08.18_veh-28_00000_00006", yaw=0.0, frame_count=120
):
    path = root / "nuplan-v1.1/splits" / partition / (name + ".db")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as c:
        c.executescript("""
        CREATE TABLE log(token BLOB,logfile TEXT,location TEXT,map_version TEXT);
        CREATE TABLE lidar(token BLOB,log_token BLOB);
        CREATE TABLE lidar_pc(token BLOB,timestamp INTEGER,ego_pose_token BLOB,lidar_token BLOB,scene_token BLOB);
        CREATE TABLE ego_pose(token BLOB,timestamp INTEGER,x REAL,y REAL,z REAL,qw REAL,qx REAL,qy REAL,qz REAL,vx REAL,vy REAL,vz REAL,acceleration_x REAL,acceleration_y REAL,acceleration_z REAL);
        CREATE TABLE scene(token BLOB,goal_ego_pose_token BLOB,roadblock_ids TEXT);
        CREATE TABLE track(token BLOB,category_token BLOB);
        CREATE TABLE category(token BLOB,name TEXT);
        CREATE TABLE lidar_box(token BLOB,lidar_pc_token BLOB,track_token BLOB,x REAL,y REAL,z REAL,width REAL,length REAL,height REAL,vx REAL,vy REAL,vz REAL,yaw REAL);
        CREATE TABLE traffic_light_status(token BLOB,lidar_pc_token BLOB,lane_connector_id INTEGER,status TEXT);
        CREATE TABLE scenario_tag(lidar_pc_token BLOB,type TEXT);
        """)
        c.execute("INSERT INTO log VALUES(?,?,?,?)", (token(1), name, "test-map", "test-map"))
        c.execute("INSERT INTO lidar VALUES(?,?)", (token(2), token(1)))
        c.execute("INSERT INTO scene VALUES(?,?,?)", (token(3), token(1000 + 119), "10 20 30"))
        for i, kind in enumerate(
            [
                "vehicle",
                "pedestrian",
                "traffic_cone",
                "bicycle",
                "barrier",
                "generic_object",
                "czone_sign",
            ]
        ):
            c.execute("INSERT INTO category VALUES(?,?)", (token(10 + i), kind))
            c.execute("INSERT INTO track VALUES(?,?)", (token(20 + i), token(10 + i)))
        for i in range(frame_count):
            stamp = 1_000_000 + 50_000 * i
            x = 100 + 0.2 * i
            c.execute(
                "INSERT INTO lidar_pc VALUES(?,?,?,?,?)",
                (token(100 + i), stamp, token(1000 + i), token(2), token(3)),
            )
            c.execute(
                "INSERT INTO ego_pose VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    token(1000 + i),
                    stamp + 123,
                    x,
                    200.0,
                    5.0,
                    np.cos(yaw / 2),
                    0.0,
                    0.0,
                    np.sin(yaw / 2),
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            )
            for j in range(7):
                if j == 1 and not 40 <= i < 80:
                    continue
                c.execute(
                    "INSERT INTO lidar_box VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        token(2000 + 10 * i + j),
                        token(100 + i),
                        token(20 + j),
                        x + 10 + 5 * j,
                        205.0,
                        5.5,
                        2.0,
                        4.0,
                        1.5,
                        4.0,
                        0.0,
                        0.0,
                        yaw,
                    ),
                )
            c.execute(
                "INSERT INTO traffic_light_status VALUES(?,?,?,?)",
                (token(4000 + i), token(100 + i), 7, ["green", "yellow", "red", "unknown"][i % 4]),
            )
        c.execute("INSERT INTO scenario_tag VALUES(?,?)", (token(140), "pedestrian_crossing"))
    return path


def test_window_preserves_raw_identity_coordinate_and_measured_history(tmp_path):
    path = make_db(tmp_path, yaw=np.pi / 2)
    w = read_window(path, token=token(140).hex(), duration=2.0, history_seconds=1.5)
    assert w["frames"][0]["token"] == token(140).hex()
    assert w["frames"][0]["timestamp"] == 3_000_000
    assert w["history"][-1]["ego_pose_timestamp_us"] == 3_000_123
    assert len(w["history"]) == 31
    assert max(v["timestamp_us"] for v in w["history"]) == 3_000_000
    assert min(v["timestamp_us"] for v in w["future_source"]) > 3_000_000
    assert w["route_roadblock_ids"] == ["10", "20", "30"]
    assert w["scenario_types"] == ["pedestrian_crossing"]
    np.testing.assert_allclose(w["frames"][0]["anns"]["gt_boxes"][0, :2], [5, -10], atol=1e-8)
    np.testing.assert_allclose(w["frames"][0]["anns"]["gt_velocity_3d"][0], [0, -4, 0], atol=1e-8)
    assert w["history"][-1]["ego_rear_axle_global"] == [108.0, 200.0, 5.0]


def test_compile_keeps_all_kinds_and_lifecycles_with_no_false_green(tmp_path):
    w = read_window(make_db(tmp_path), token=token(130).hex(), duration=3.0, history_seconds=1.0)
    scene = compile_window(w, duration=3.0, duration_mode="exact")
    assert len(scene.actor_ids) == 8
    ped = scene.actor_ids.index(token(21).hex())
    assert not scene.presence[0, ped] and scene.presence[10, ped] and not scene.presence[-1, ped]
    assert set(scene.metadata["source_actor_kinds"].values()) == {
        "vehicle",
        "pedestrian",
        "traffic_cone",
        "bicycle",
        "barrier",
        "generic_object",
        "czone_sign",
    }
    assert set(
        s["status"] for f in scene.metadata["nuplan_source"]["signals"] for s in f["states"]
    ) == {"green", "yellow", "red", "unknown"}
    assert scene.metadata["trainable"] is False
    assert (
        scene.metadata["nuplan_source"]["future_source_timestamps_us"][0] > w["anchor_timestamp_us"]
    )
    out = tmp_path / "scene.npz"
    scene.save(out)
    restored = Scene.load(out)
    np.testing.assert_array_equal(scene.presence, restored.presence)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("DELETE FROM ego_pose WHERE token=X'0000000000000410'", "Missing ego pose"),
        ("DELETE FROM lidar_pc WHERE timestamp BETWEEN 3050000 AND 3250000", "gap"),
        ("UPDATE category SET name='alien' WHERE name='barrier'", "category"),
        ("UPDATE lidar_pc SET timestamp=3000000 WHERE timestamp=3050000", "non-monotonic"),
        ("UPDATE lidar_box SET width=-1 WHERE track_token=X'0000000000000014'", "dimensions"),
    ],
)
def test_invalid_source_fails_closed(tmp_path, mutation, match):
    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute(mutation)
    with pytest.raises(ValueError, match=match):
        read_window(path, token=token(140).hex(), duration=2.0)


def test_readonly_no_missing_db_creation_and_no_padding(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        read_window(missing, token=token(140).hex())
    assert not missing.exists()
    path = make_db(tmp_path)
    with open_db(path) as c:
        with pytest.raises(sqlite3.OperationalError):
            c.execute("DELETE FROM log")
    with pytest.raises(ValueError, match="history"):
        read_window(path, token=token(101).hex())
    with pytest.raises(ValueError, match="future"):
        read_window(path, token=token(219).hex())


def test_split_groups_preserve_holdouts_across_shards(tmp_path):
    a = make_db(tmp_path)
    b = make_db(tmp_path, "test", name="2021.08.18_veh-28_10000_10006")
    files = {"boston": [a], "test": [b]}
    rules = split_rules(files)
    assert role_for(a.stem, "boston", rules) == "heldout"
    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps(["other-log"]))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "source_roles": {a.stem: "train"}}))
    with pytest.raises(ValueError, match="Protected"):
        split_rules(files, policy, [protected])
    with pytest.raises(ValueError, match="registry"):
        split_rules({"boston": [a]}, policy)
    assert source_group(a.name) == source_group(a.stem + ".pkl")


def test_shared_catalogue_compiles_without_pickle_trust_and_rejects_source_change(
    tmp_path, monkeypatch
):
    path = make_db(tmp_path)
    catalog = tmp_path / "catalog.sqlite"
    result = build_catalog(
        tmp_path, catalog, partitions=["boston"], duration=2.0, windows_per_log=2
    )
    assert result["scene_statuses"] == {"geometry_pending": 2}
    rows = list_scenes(catalog)
    assert rows[0]["token"] != rows[1]["token"]
    monkeypatch.setattr(
        "driversi.data.scenes.library.attach_nuplan_map",
        lambda scene, root: scene.metadata.update(map_version="fixture"),
    )
    output = tmp_path / "compiled"
    report = compile_catalog(catalog, output)
    assert report["compiled"] == 2
    assert len((output / "unassigned.txt").read_text().splitlines()) == 2
    assert not (output / "train.txt").read_text()
    with sqlite3.connect(path) as c:
        c.execute("UPDATE log SET map_version='changed'")
    report = compile_catalog(catalog, tmp_path / "changed")
    assert report["failed"] == 2


def test_explicit_source_policy_and_external_protection(tmp_path, monkeypatch):
    a = make_db(tmp_path)
    b = make_db(tmp_path, name="2021.08.19_veh-28_00000_00006")
    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps(["unrelated"]))
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"schema_version": 1, "source_roles": {a.stem: "train", b.stem: "development"}})
    )
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(
        tmp_path,
        catalog,
        partitions=["boston"],
        max_logs=2,
        duration=2.0,
        split_policy=policy,
        protected_logs=[protected],
    )
    monkeypatch.setattr(
        "driversi.data.scenes.library.attach_nuplan_map",
        lambda scene, root: scene.metadata.update(map_version="fixture"),
    )
    compile_catalog(catalog, tmp_path / "compiled")
    for role in ("train", "development"):
        paths = (tmp_path / "compiled" / (role + ".txt")).read_text().splitlines()
        assert len(paths) == 2
        assert Scene.load(Path(paths[0])).metadata["split"] == role


def test_cli_inventory_and_bad_database_are_auditable(tmp_path, capsys):
    path = make_db(tmp_path)
    (path.parent / "broken.db").write_bytes(b"not SQLite")
    assert inventory(tmp_path)["partitions"]["boston"]["files"] == 2
    assert (
        main(
            [
                "index-nuplan",
                "--dataset-root",
                str(tmp_path),
                "--partitions",
                "boston",
                "--max-logs",
                "2",
                "--duration",
                "2",
                "--output",
                str(tmp_path / "c.sqlite"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["log_statuses"] == {"indexed": 1, "rejected": 1}
    assert "database" in report["log_errors"][0]["reason"]


def test_history_duplicate_track_is_rejected(tmp_path):
    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute(
            "INSERT INTO lidar_box SELECT * FROM lidar_box WHERE lidar_pc_token=? AND track_token=?",
            (token(139), token(20)),
        )
    with pytest.raises(ValueError, match="duplicate track"):
        read_window(path, token=token(140).hex(), duration=2.0)


def test_vegas_location_alias_and_causal_initial_velocity(tmp_path):
    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE log SET location='las_vegas'")
    a = compile_window(read_window(path, token=token(140).hex(), duration=2.0), duration=2.0)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE ego_pose SET x=x+0.1 WHERE timestamp>3000123")
        c.execute("UPDATE lidar_box SET x=x+0.1 WHERE lidar_pc_token>?", (token(140),))
    b = compile_window(read_window(path, token=token(140).hex(), duration=2.0), duration=2.0)
    np.testing.assert_array_equal(a.velocities[0], b.velocities[0])
    assert a.metadata["nuplan_source"]["source_location"] == "las_vegas"
    assert a.metadata["velocity_model"].startswith("causal_initial")


def test_long_db_window_preserves_identity_and_real_source_clock(tmp_path):
    path = make_db(tmp_path, frame_count=660)
    window = read_window(path, token=token(140).hex(), duration=30)
    scene = compile_window(window, duration=30, max_actors=1)
    assert len(scene.times) == 301 and scene.times[-1] == 30
    assert scene.actor_ids[0] == "ego"
    assert len(scene.actor_ids) == 8
    assert window["history"][-1]["timestamp_us"] == 3_000_000
    assert window["future_source"][-1]["timestamp_us"] >= 33_000_000
    assert scene.metadata["nuplan_source"]["history"] == window["history"]


def test_raw_motion_evidence_distinguishes_jitter_departure_and_fragment(tmp_path):
    path = make_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute(
            "UPDATE lidar_box SET vx=0,vy=0,vz=0,x=120+0.03*(rowid%10),y=205 WHERE track_token=?",
            (token(20),),
        )
    w = read_window(path, token=token(140).hex(), duration=2)
    scene = compile_window(w, duration=2, max_actors=1)
    e = scene.metadata["source_motion_evidence"]["actors"][token(20).hex()]
    assert e["samples"] >= 40 and e["finite_velocity_samples"] == e["samples"]
    assert e["max_reported_speed_mps"] == 0 and e["max_position_excursion_m"] < 0.3
    from driversi.data.participants.fleet import configure_vehicle_fleet

    fleet = configure_vehicle_fleet(scene)
    assert fleet.metadata["vehicle_fleet"]["roles"][token(20).hex()]["mode"] == "stationary_hold"
    with sqlite3.connect(path) as c:
        c.execute(
            "UPDATE lidar_box SET vx=3 WHERE track_token=? AND lidar_pc_token>?",
            (token(20), token(160)),
        )
    moving = compile_window(
        read_window(path, token=token(140).hex(), duration=2), duration=2, max_actors=1
    )
    assert (
        moving.metadata["source_motion_evidence"]["actors"][token(20).hex()][
            "max_reported_speed_mps"
        ]
        == 3
    )
    with sqlite3.connect(path) as c:
        c.execute(
            "DELETE FROM lidar_box WHERE track_token=? AND lidar_pc_token=?",
            (token(20), token(150)),
        )
    fragmented = compile_window(
        read_window(path, token=token(140).hex(), duration=2), duration=2, max_actors=1
    )
    e = fragmented.metadata["source_motion_evidence"]["actors"][token(20).hex()]
    assert e["samples"] == 10 and e["max_reported_speed_mps"] == 0
    assert e["last_timestamp_us"] == 3_450_000
