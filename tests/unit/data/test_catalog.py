import json
import pickle
import sqlite3

import numpy as np
import pytest

from driversi.cli import main
from driversi.contracts.scene import Scene
from driversi.data.scenes.catalog import build_catalog, catalog_summary, inventory, list_scenes
from driversi.data.scenes.library import compile_catalog


def write_log(root, partition="mini", name="source", *, timestamps=None, speed=4):
    times = timestamps if timestamps is not None else [1_000_000 + 500_000 * i for i in range(8)]
    frames = [{
        "timestamp": t, "token": f"{name}-{i}", "log_name": name,
        "log_token": f"original-{name}", "map_location": "test-map",
        "ego2global_translation": [(t - times[0]) / 1e6 * speed, 0, 0],
        "ego2global_rotation": [1, 0, 0, 0],
        "anns": {"gt_boxes": np.empty((0, 7)), "gt_names": [], "track_tokens": []},
    } for i, t in enumerate(times)]
    path = root / "navsim_logs" / partition / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(frames))
    return path


def test_inventory_does_not_load_pickles_or_private_data(tmp_path):
    write_log(tmp_path)
    private = tmp_path / "navsim_logs" / "private_test_hard"
    private.mkdir()
    (tmp_path / "navsim_logs/mini/unreadable.pkl").write_bytes(b"not a pickle")
    actual = inventory(tmp_path)
    assert actual["partitions"]["mini"]["files"] == 2
    assert actual["ignored_log_directories"] == ["private_test_hard"]
    assert actual["partitions"]["trainval"]["present"] is False


def test_cli_indexes_windows_with_bounded_coverage_and_no_official_filter(tmp_path, capsys):
    write_log(tmp_path, name="a")
    write_log(tmp_path, name="b")
    catalog = tmp_path / "catalog.sqlite"
    assert main(["index-navsim", "--dataset-root", str(tmp_path), "--max-logs", "1",
                 "--duration", "1.5", "--trusted-pickle", "--output", str(catalog)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["partitions"]["mini"] == {
        "available_files": 2, "selected_files": 1, "use": "engineering"}
    assert result["official_scene_filter"] is None
    rows = list_scenes(catalog)
    assert len(rows) == 2
    assert rows[0]["start_us"] == 1_000_000
    assert rows[-1]["end_us"] == 4_500_000
    assert all(row["relative_path"] == "navsim_logs/mini/a.pkl" for row in rows)
    with pytest.raises(FileExistsError):
        build_catalog(tmp_path, catalog, partitions=["mini"], trusted=True)


def test_overlap_blocks_both_partitions_even_with_different_filenames(tmp_path):
    write_log(tmp_path, "mini", "shared")
    p = write_log(tmp_path, "test", "shared")
    p.rename(p.with_name("renamed.pkl"))
    catalog = tmp_path / "catalog.sqlite"
    result = build_catalog(tmp_path, catalog, partitions=["mini", "test"], duration=1.5, trusted=True)
    assert result["cross_partition_log_overlaps"] == 1
    assert list_scenes(catalog) == []
    assert len(list_scenes(catalog, status="conflict")) == 4


def test_nonoverlapping_intervals_of_same_source_do_not_conflict(tmp_path):
    write_log(tmp_path, "mini")
    write_log(tmp_path, "test", timestamps=[10_000_000 + 500_000 * i for i in range(8)])
    catalog = tmp_path / "catalog.sqlite"
    result = build_catalog(tmp_path, catalog, partitions=["mini", "test"], duration=1.5, trusted=True)
    assert result["cross_partition_log_overlaps"] == 0
    assert len(list_scenes(catalog)) == 4


def test_short_source_still_blocks_overlap_in_other_partition(tmp_path):
    write_log(tmp_path, "mini")
    write_log(tmp_path, "test", timestamps=[1_000_000, 1_500_000])
    catalog = tmp_path / "catalog.sqlite"
    result = build_catalog(tmp_path, catalog, partitions=["mini", "test"], duration=1.5, trusted=True)
    assert result["log_statuses"]["rejected"] == 1
    assert result["cross_partition_log_overlaps"] == 1
    assert list_scenes(catalog, status="conflict")[0]["start_us"] == 1_000_000
    assert len(list_scenes(catalog)) == 1  # Later, disjoint window remains usable.


def test_invalid_annotations_do_not_hide_known_source_overlap(tmp_path):
    write_log(tmp_path, "mini")
    path = write_log(tmp_path, "test")
    frames = pickle.loads(path.read_bytes())
    frames[0]["anns"]["gt_boxes"] = np.ones((1, 7))
    path.write_bytes(pickle.dumps(frames))
    catalog = tmp_path / "catalog.sqlite"
    result = build_catalog(tmp_path, catalog, partitions=["mini", "test"], duration=1.5, trusted=True)
    assert result["log_statuses"]["rejected"] == 1
    assert result["cross_partition_log_overlaps"] == 1
    assert result["overlap_unchecked_logs"] == []
    assert list_scenes(catalog) == []


def test_bad_log_and_stationary_window_remain_visible(tmp_path):
    p = write_log(tmp_path, name="stationary", speed=0)
    p.with_name("bad.pkl").write_bytes(b"invalid pickle")
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=1.5, trusted=True)
    summary = catalog_summary(catalog)
    assert summary["log_statuses"] == {"indexed": 1, "rejected": 1}
    assert len(summary["overlap_unchecked_logs"]) == 1
    assert summary["scene_statuses"] == {"rejected": 2}
    assert len(list_scenes(catalog, status="rejected", min_controlled=0)) == 2
    assert "No moving vehicle" in summary["rejections"][0]["reason"]


def test_untrusted_or_nonpublic_partition_rejected_before_output(tmp_path):
    catalog = tmp_path / "catalog.sqlite"
    with pytest.raises(ValueError, match="trusted"):
        build_catalog(tmp_path, catalog, partitions=["mini"])
    with pytest.raises(ValueError, match="Unsupported"):
        build_catalog(tmp_path, catalog, partitions=["trainval_fut"], trusted=True)
    assert not catalog.exists()


def test_library_preserves_source_purpose_and_records_map_failures(tmp_path, monkeypatch, straight_scene):
    write_log(tmp_path, "trainval", "a")
    write_log(tmp_path, "trainval", "b")
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["trainval"], duration=1.5,
                  windows_per_log=1, trusted=True)

    def map_for_test(scene, _root):
        if scene.metadata["source_log"] == "b":
            raise ValueError("No usable lane geometry")
        for key in ("road_points", "road_offsets", "road_types"):
            setattr(scene, key, getattr(straight_scene, key).copy())
        scene.metadata.update(map_complete=True, map_version="test-version")

    monkeypatch.setattr("driversi.data.scenes.library.attach_nuplan_map", map_for_test)
    output = tmp_path / "library"
    result = compile_catalog(catalog, output, trusted=True)
    assert (result["selected"], result["compiled"], result["failed"]) == (2, 1, 1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["state"] == "complete"
    assert manifest["records"][1]["reason"] == "No usable lane geometry"
    scene = Scene.load(output / manifest["records"][0]["scene"])
    assert scene.metadata["split"] == scene.metadata["intended_use"] == "unassigned"
    assert scene.metadata["source_log_token"] == "original-a"
    assert scene.metadata["source_interval_us"] == [1_000_000, 2_500_000]
    assert scene.metadata["official_scene_filter"] is None
    with pytest.raises(FileExistsError):
        compile_catalog(catalog, output, trusted=True)


def test_library_refuses_changed_source_before_map_access(tmp_path, monkeypatch):
    path = write_log(tmp_path)
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=1.5, trusted=True)
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr("driversi.data.scenes.library.attach_nuplan_map",
                        lambda *_: pytest.fail("Changed source reached map compiler"))
    result = compile_catalog(catalog, tmp_path / "library", trusted=True)
    assert result["compiled"] == 0
    manifest = json.loads((tmp_path / "library/manifest.json").read_text())
    assert all("changed since indexing" in r["reason"] for r in manifest["records"])


def test_conflicting_catalogue_cannot_produce_scene_assets(tmp_path):
    write_log(tmp_path, "mini")
    write_log(tmp_path, "test")
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini", "test"], duration=1.5, trusted=True)
    output = tmp_path / "library"
    with pytest.raises(ValueError, match="No geometry_pending"):
        compile_catalog(catalog, output, trusted=True)
    assert not output.exists()


def test_optional_log_token_has_same_rules_at_index_and_compile(tmp_path, monkeypatch, straight_scene):
    path = write_log(tmp_path)
    frames = pickle.loads(path.read_bytes())
    for frame in frames[1:]:
        del frame["log_token"]
    path.write_bytes(pickle.dumps(frames))
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=1.5, trusted=True)

    def map_for_test(scene, _root):
        for key in ("road_points", "road_offsets", "road_types"):
            setattr(scene, key, getattr(straight_scene, key).copy())
        scene.metadata.update(map_complete=True, map_version="fixture")

    monkeypatch.setattr("driversi.data.scenes.library.attach_nuplan_map", map_for_test)
    result = compile_catalog(catalog, tmp_path / "library", trusted=True)
    assert result["compiled"] == 2
    assert result["failed"] == 0


def _fixture_map(monkeypatch, straight_scene):
    def attach(scene, _root):
        for key in ("road_points", "road_offsets", "road_types"):
            setattr(scene, key, getattr(straight_scene, key).copy())
        scene.metadata.update(map_complete=True, map_version="fixture")
    monkeypatch.setattr("driversi.data.scenes.library.attach_nuplan_map", attach)


def test_catalog_fit_source_freezes_starts_and_retains_short_window_rejections(tmp_path):
    write_log(tmp_path, speed=6)
    catalog = tmp_path / "adaptive.sqlite"
    summary = build_catalog(tmp_path, catalog, partitions=["mini"], duration=9,
                            duration_mode="fit_source", min_duration=1,
                            windows_per_log=3, trusted=True)
    rows = list_scenes(catalog, status="all", min_controlled=0)
    assert summary["schema_version"] == 2
    assert [row["start_frame"] for row in rows] == [0, 3, 6]
    assert [row["planned_duration_seconds"] for row in rows] == [3.5, 2, None]
    assert rows[-1]["status"] == "rejected"
    assert rows[-1]["horizon_plan"]["candidate_duration_seconds"] == .5
    assert rows[-1]["horizon_plan"]["rejection_reasons"] == ["candidate_below_min_duration"]
    assert rows[-1]["end_frame"] == rows[-1]["start_frame"]  # No fictional selected endpoint.


def test_catalog_fit_source_records_isolated_segment_starts_instead_of_retrying(tmp_path):
    write_log(tmp_path, timestamps=[0, 500_000, 1_500_000, 2_500_000, 3_000_000], speed=6)
    catalog = tmp_path / "gaps.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=3,
                  duration_mode="fit_source", windows_per_log=4, trusted=True)
    rows = list_scenes(catalog, status="all", min_controlled=0)
    assert [row["start_frame"] for row in rows] == [0, 1, 2, 3]
    assert [row["status"] for row in rows] == ["geometry_pending", "rejected", "rejected", "geometry_pending"]
    assert [row["horizon_plan"]["available_future_seconds"] for row in rows] == [.5, 0, 0, .5]


def test_library_consumes_row_durations_and_preserves_original_budget(tmp_path, monkeypatch, straight_scene):
    write_log(tmp_path, speed=6)
    catalog = tmp_path / "adaptive.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=22,
                  duration_mode="fit_source", windows_per_log=3, trusted=True)
    _fixture_map(monkeypatch, straight_scene)
    result = compile_catalog(catalog, tmp_path / "library", trusted=True)
    assert (result["compiled"], result["failed"]) == (3, 0)
    manifest = json.loads((tmp_path / "library/manifest.json").read_text())
    assert manifest["catalog_schema_version"] == 2
    assert [row["planned_duration_seconds"] for row in manifest["records"]] == [3.5, 2, .5]
    for row in manifest["records"]:
        scene = Scene.load(tmp_path / "library" / row["scene"])
        assert scene.times[-1] == row["planned_duration_seconds"]
        assert scene.metadata["horizon_plan"] == row["horizon_plan"]
        assert row["horizon_plan"]["requested_duration_seconds"] == 22
        assert row["horizon_plan"]["mode"] == "fit_source"


def test_library_reads_legacy_catalog_without_modifying_it(tmp_path, monkeypatch, straight_scene):
    write_log(tmp_path, speed=6)
    catalog = tmp_path / "legacy.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=1.5, trusted=True)
    # Make the exact historical schema, rather than labelling a v2 table as v1.
    with sqlite3.connect(catalog) as connection:
        connection.execute("CREATE TABLE legacy_scenes AS SELECT id,log_id,start_frame,end_frame,"
                           "token,start_us,end_us,status,reason,actors,controlled,exclusions FROM scenes")
        connection.execute("DROP TABLE scenes")
        connection.execute("ALTER TABLE legacy_scenes RENAME TO scenes")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        connection.execute("UPDATE metadata SET value=? WHERE key='parameters'", (
            json.dumps({"duration": 1.5, "max_actors": 32, "windows_per_log": 2,
                        "max_logs_per_partition": 3}),))
    before = catalog.read_bytes()
    _fixture_map(monkeypatch, straight_scene)
    rows = list_scenes(catalog)
    assert all(row["horizon_plan"]["legacy_catalog"] for row in rows)
    assert compile_catalog(catalog, tmp_path / "library", trusted=True)["compiled"] == 2
    scene = Scene.load(tmp_path / "library/scene-000001.npz")
    assert scene.times[-1] == 1.5
    assert scene.metadata["horizon_plan"]["provenance"] == "revalidated_legacy_catalog_v1_exact_duration"
    assert catalog.read_bytes() == before


@pytest.mark.parametrize("column", ["planned_duration_seconds", "horizon_plan"])
def test_library_refuses_inconsistent_indexed_plan_before_map(tmp_path, monkeypatch, column):
    write_log(tmp_path, speed=6)
    catalog = tmp_path / "adaptive.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=9,
                  duration_mode="fit_source", windows_per_log=1, trusted=True)
    with sqlite3.connect(catalog) as connection:
        if column == "planned_duration_seconds":
            connection.execute("UPDATE scenes SET planned_duration_seconds=1")
        else:
            plan = json.loads(connection.execute("SELECT horizon_plan FROM scenes").fetchone()[0])
            plan["available_future_seconds"] = 99
            connection.execute("UPDATE scenes SET horizon_plan=?", (json.dumps(plan),))
    monkeypatch.setattr("driversi.data.scenes.library.attach_nuplan_map",
                        lambda *_: pytest.fail("Inconsistent horizon reached map compiler"))
    result = compile_catalog(catalog, tmp_path / "library", trusted=True)
    assert (result["compiled"], result["failed"]) == (0, 1)
    manifest = json.loads((tmp_path / "library/manifest.json").read_text())
    assert "horizon" in manifest["records"][0]["reason"]


def test_spawn_compiler_preserves_order_and_data_rejections(tmp_path):
    write_log(tmp_path, name="a")
    write_log(tmp_path, name="b")
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], max_logs=2,
                  duration=1.5, windows_per_log=2, trusted=True)
    serial = compile_catalog(catalog, tmp_path / "serial", trusted=True)
    parallel = compile_catalog(catalog, tmp_path / "parallel", trusted=True, workers=2)
    assert serial["failed"] == parallel["failed"] == 4  # Explicit missing map assets.
    a = json.loads((tmp_path / "serial/manifest.json").read_text())
    b = json.loads((tmp_path / "parallel/manifest.json").read_text())
    assert a["records"] == b["records"]
    assert b["state"] == "complete" and b["workers"] == 2


def test_compiler_infrastructure_failure_is_not_a_success_manifest(tmp_path, monkeypatch):
    write_log(tmp_path)
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(tmp_path, catalog, partitions=["mini"], duration=1.5, trusted=True)

    def fail(*_args):
        raise RuntimeError("compile worker unavailable")

    monkeypatch.setattr("driversi.data.scenes.library._group_results", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        compile_catalog(catalog, tmp_path / "library", workers=2, trusted=True)
    manifest = json.loads((tmp_path / "library/manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert "unavailable" in manifest["infrastructure_error"]
    assert manifest["compiled"] == 0


def test_compile_worker_limit_is_checked_before_output_creation(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        compile_catalog(tmp_path / "missing", tmp_path / "library", workers=65, trusted=True)
    assert not (tmp_path / "library").exists()


@pytest.mark.parametrize("override", [False, True])
def test_map_root_selection_is_recorded_and_reaches_compilation(tmp_path, monkeypatch, override):
    root = tmp_path / "dataset"
    write_log(root, "mini", "map-source")
    catalog = tmp_path / "catalog.sqlite"
    build_catalog(root, catalog, partitions=["mini"], duration=3, windows_per_log=1, trusted=True)
    selected_root = tmp_path / "cached-maps" if override else root / "maps"
    observed = []

    def check_map(scene, maps_root):
        observed.append(maps_root)
        raise ValueError("map-root-test")

    monkeypatch.setattr("driversi.data.scenes.library.attach_nuplan_map", check_map)
    options = {"maps_root": selected_root} if override else {}
    compile_catalog(catalog, tmp_path / "scenes", trusted=True, **options)
    report = json.loads((tmp_path / "scenes" / "manifest.json").read_text())
    assert observed == [selected_root.resolve()]
    assert report["maps_root"] == str(selected_root.resolve())
