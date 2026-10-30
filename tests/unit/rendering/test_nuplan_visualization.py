import json
import sqlite3

import numpy as np
import pytest
from tests.unit.data.test_nuplan_db import make_db, token

from driversi.data.sources.nuplan_db import compile_window, read_window
from driversi.rendering.inspection.nuplan_log import export_log
from driversi.rendering.bev.replay import export_source_replay


def test_full_db_preserves_frames_boxes_and_annotation_intervals(tmp_path):
    db = make_db(tmp_path)
    before = db.read_bytes()
    output = tmp_path / "viewer"
    report = export_log(db, output, chunk_size=17)
    assert report["frame_count"] == 120 and report["duration_seconds"] == 5.95
    assert len(report["chunks"]) == 8
    frames = [
        frame
        for row in report["chunks"]
        for frame in json.loads((output / row["path"]).read_text())["frames"]
    ]
    assert [f["index"] for f in frames] == list(range(120))
    assert frames[0]["token"] == token(100).hex()
    assert frames[-1]["token"] == token(219).hex()
    assert frames[0]["ego_pose_timestamp_us"] - frames[0]["timestamp_us"] == 123
    np.testing.assert_allclose(frames[0]["objects"][0][1:3], [1.461, 0])
    assert next(a for a in report["actors"] if a["id"] == token(21).hex())["kind"] == "pedestrian"
    assert next(a for a in report["actors"] if a["id"] == token(21).hex())["intervals"] == [
        [40, 79]
    ]
    assert report["counts"][39] == 7 and report["counts"][40] == 8
    assert frames[0]["signals"] == [["7", "green"]]
    assert report["collision_assessed"] is False and report["simulation"] is False
    assert db.read_bytes() == before
    with pytest.raises(FileExistsError):
        export_log(db, output)


def test_full_log_preserves_time_gaps_without_interpolation(tmp_path):
    db = make_db(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM lidar_pc WHERE timestamp BETWEEN 3050000 AND 3250000")
    report = export_log(db, tmp_path / "viewer")
    assert report["frame_count"] == 115
    assert report["gaps_before_frames"] == [41]
    assert next(a for a in report["actors"] if a["id"] == token(21).hex())["intervals"] == [
        [40, 40],
        [41, 74],
    ]


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("UPDATE lidar_pc SET timestamp=3000000 WHERE timestamp=3050000", "Non-monotonic"),
        ("DELETE FROM ego_pose WHERE token=X'0000000000000410'", "Missing ego"),
        ("UPDATE category SET name='unknown' WHERE name='barrier'", "category"),
        ("UPDATE lidar_box SET width=-1", "geometry"),
        ("INSERT INTO lidar_box SELECT * FROM lidar_box LIMIT 1", "Duplicate track"),
    ],
)
def test_full_log_invalid_source_never_publishes_manifest(tmp_path, mutation, message):
    db = make_db(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute(mutation)
    output = tmp_path / "viewer"
    with pytest.raises(ValueError, match=message):
        export_log(db, output, chunk_size=17)
    assert not (output / "manifest.json").exists()


def test_full_log_explicit_budget_rejects_not_truncates(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        export_log(make_db(tmp_path), tmp_path / "viewer", max_frames=100)
    assert not (tmp_path / "viewer/manifest.json").exists()


def test_source_viewer_is_not_controlled_simulation_and_keeps_input(tmp_path):
    window = read_window(
        make_db(tmp_path), token=token(130).hex(), duration=3.0, history_seconds=1.0
    )
    scene = compile_window(window, duration=3.0, duration_mode="exact")
    metadata = json.dumps(scene.metadata)
    controls = scene.controlled.copy()
    target = tmp_path / "source.html"
    result = export_source_replay(scene, target)
    assert result["simulation"] is False
    assert result["collision_assessed"] is False
    assert json.dumps(scene.metadata) == metadata
    np.testing.assert_array_equal(scene.controlled, controls)
    text = target.read_text()
    assert "source_reference" in text and "源轨迹插值" in text
    assert "原始 DB" not in text  # this is the compiled window, not the full DB
