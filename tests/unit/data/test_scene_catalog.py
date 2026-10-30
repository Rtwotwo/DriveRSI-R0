import json
import pickle

import pytest

from driversi.data.catalog.observations import history_evidence, parse_response
from driversi.data.catalog.store import Catalog, identity


def scene(token="a", split="navtrain"):
    return {
        "record": {"token": token, "source_log": "log", "kind": "original_" + split},
        "descriptor": {"token": token, "source_log": "log", "source_group": "group",
                       "conditions": ["traffic_dense"]},
    }


def populated(tmp_path, split="navtrain"):
    c = Catalog(tmp_path / (split + ".sqlite"), split)
    s = scene(split=split)
    c.ingest([s["record"]], [s["descriptor"]], {"split": split, "index_sha256": "index", "complete": True})
    return c


def observation():
    return {"caption": "A vehicle is visible ahead.", "observations": [
        {"tag": "vehicle_ahead_visible", "evidence": ["h3_CAM_F0"], "description": "car ahead"}
    ], "unknown": ["whether the vehicle is stopped"]}


def test_split_isolation(tmp_path):
    c = populated(tmp_path, "navtest")
    with pytest.raises(ValueError, match="disagree"):
        c.query("training")
    assert len(c.query("evaluation")) == 1
    c.close()
    with pytest.raises(ValueError, match="split differs"):
        Catalog(tmp_path / "navtest.sqlite", "navtrain")


def test_atomic_import_rollback(tmp_path):
    c = populated(tmp_path)
    b, bad = scene("b"), scene("c", "navtest")
    with pytest.raises(ValueError, match="split mismatch"):
        c.ingest([b["record"], bad["record"]], [b["descriptor"], bad["descriptor"]], {})
    assert c.stats()["scenes"] == 1
    c.close()


def test_changed_scene_and_duplicate_descriptor_rejected(tmp_path):
    c = populated(tmp_path)
    s = scene()
    with pytest.raises(ValueError, match="Duplicate descriptor"):
        c.ingest([s["record"]], [s["descriptor"]] * 2, {})
    s["descriptor"]["conditions"].append("another_tag")
    with pytest.raises(ValueError, match="Existing scene changed"):
        c.ingest([s["record"]], [s["descriptor"]], {})
    c.close()


def test_profile_evidence_resume_and_query(tmp_path):
    c = populated(tmp_path)
    p = c.profile({"model": "fixed"})
    e = [{"id": "h3_CAM_F0", "sha256": "original"}]
    c.record_attempt("a", p, e, "raw", {}, observation=observation())
    assert c.completed("a", p, identity(e))
    assert not c.completed("a", c.profile({"model": "new"}), identity(e))
    assert not c.completed("a", p, identity([{**e[0], "sha256": "changed"}]))
    assert len(c.query("training", profile=p, visual_tags=["vehicle_ahead_visible"])) == 1
    assert len(c.query("training", profile=p, text="vehicle")) == 1
    assert c.query("training", profile=p, text='" OR 1=1 --') == []
    with pytest.raises(ValueError, match="profile"):
        c.query("training", text="vehicle")
    c.close()


def test_failed_output_is_not_searchable(tmp_path):
    c = populated(tmp_path)
    p = c.profile({"model": "fixed"})
    c.record_attempt("a", p, [], "vehicle ahead", {}, error="invalid JSON")
    assert not c.completed("a", p, identity([]))
    assert c.query("training", profile=p, text="vehicle") == []
    assert c.stats()["attempts"] == {"failed": 1}
    c.close()


@pytest.mark.parametrize("mutation", ["future", "score", "tag", "evidence", "duplicate"])
def test_untrusted_response_rejected(mutation):
    value = observation()
    if mutation == "future":
        value["observations"][0]["evidence"] = ["future4_CAM_F0"]
    elif mutation == "score":
        value["pdm"] = 100
    elif mutation == "tag":
        value["observations"][0]["tag"] = "driver_failed"
    elif mutation == "evidence":
        value["observations"][0]["evidence"] = []
    else:
        value["observations"] *= 2
    with pytest.raises(ValueError):
        parse_response(json.dumps(value), [{"id": "h3_CAM_F0"}])


def test_history_uses_real_frames_not_future(tmp_path):
    from PIL import Image

    frames = []
    for i in range(6):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (320, 160), (i * 30, 0, 0)).save(path)
        frames.append({"token": str(i), "log_name": "log", "timestamp": 1000000 + i * 500000,
                       "cams": {"CAM_F0": {"data_path": path.name}}})
    log = tmp_path / "log.pkl"
    log.write_bytes(pickle.dumps(frames))
    s = {"record": {"token": "3", "log_path": str(log), "source_log": "log",
                    "anchor_index": 3, "sensor_root": str(tmp_path)},
         "descriptor": {"history": {"continuous": True, "anchor_timestamp_us": 2500000}}}
    evidence, images = history_evidence(s, ["CAM_F0"], 256)
    assert [v["id"] for v in evidence] == [f"h{i}_CAM_F0" for i in range(4)]
    assert [v["relative_time_s"] for v in evidence] == [-1.5, -1.0, -0.5, 0]
    assert images[-1].size == (256, 128)
    frames[4]["timestamp"] = -1
    frames[4]["cams"] = {}  # Future corruption cannot change the prompt.
    log.write_bytes(pickle.dumps(frames))
    assert history_evidence(s, ["CAM_F0"], 256)[0] == evidence
    frames[2]["timestamp"] += 100000
    log.write_bytes(pickle.dumps(frames))
    with pytest.raises(ValueError, match="sampling"):
        history_evidence(s, ["CAM_F0"], 256)


def test_training_catalog_structured_filter(tmp_path):
    c = populated(tmp_path)
    assert len(c.query("training", ["traffic_dense"])) == 1
    assert c.query("training", ["unobserved"]) == []
    with pytest.raises(ValueError):
        c.query("training", limit=100001)
    c.close()


def score_fixture():
    metrics = {k: 1.0 for k in ("no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance", "ego_progress", "time_to_collision_within_bound", "comfort")}
    return {"token": "a", "log_name": "log", "valid": True, "score": 0.8, **metrics}


def test_scores_do_not_cross_splits_or_model_identity(tmp_path):
    c = populated(tmp_path)
    provenance = {"split": "navtrain", "purpose": "training_pool_mining", "index_sha256": "index"}
    row = score_fixture()
    run = c.import_train_scores([row], provenance)
    assert len(c.query("training", score_run=run, score_below=0.95)) == 1
    assert c.query("training", score_run=run, score_below=0.5) == []
    assert c.import_train_scores([row], provenance) == run
    with pytest.raises(ValueError, match="changed"):
        c.import_train_scores([{**row, "score": 0.9}], provenance)
    with pytest.raises(ValueError, match="training-owned"):
        c.import_train_scores([row], {**provenance, "split": "navtest"})
    with pytest.raises(ValueError, match="population"):
        c.import_train_scores([row], {**provenance, "index_sha256": "another"})
    with pytest.raises(ValueError, match="Normalized"):
        c.import_train_scores([{**row, "score": float("nan")}], provenance)
    c.close()


def test_harness_freezes_train_candidates_and_rejects_test(tmp_path, monkeypatch):
    from driversi.rsi.plugins.scene_catalog import execute, prepare
    from driversi_agents.chainflow import benchmark

    monkeypatch.setattr(benchmark, "official_filter", lambda _: (None, {"tokens": ["a"], "log_names": ["log"]}))
    c = populated(tmp_path)
    c.close()
    config = {"database": "navtrain.sqlite", "structured_tags": ["traffic_dense"],
              "visual_tags": [], "profile": None, "text": None, "limit": 1,
              "score_run": None, "score_below": None}
    prepared, steps = prepare(config, tmp_path, None)
    assert steps == 0
    result = execute(prepared, {}, tmp_path / "artifact", None)
    assert result["summary"]["candidates"][0]["token"] == "a"
    assert not result["summary"]["synthesis_authorized"]
    prepared["report"]["candidates"][0]["trainable"] = True
    with pytest.raises(ValueError, match="changed"):
        execute(prepared, {}, tmp_path / "another", None)
    populated(tmp_path, "navtest").close()
    with pytest.raises(ValueError, match="split differs"):
        prepare({**config, "database": "navtest.sqlite"}, tmp_path, None)


def test_readonly_missing_catalog_does_not_create_file(tmp_path):
    import sqlite3

    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        Catalog(path, "navtrain", readonly=True)
    assert not path.exists()


def test_latest_image_version_and_review_exclude_old_tags(tmp_path):
    c = populated(tmp_path)
    p = c.profile({"model": "fixed"})
    old = [{"id": "h3_CAM_F0", "sha256": "old"}]
    new = [{"id": "h3_CAM_F0", "sha256": "new"}]
    c.record_attempt("a", p, old, "raw", {}, observation=observation())
    c.record_attempt("a", p, new, "invalid", {}, error="failed")
    assert not c.completed("a", p, identity(old))
    assert c.query("training", profile=p, text="vehicle") == []
    attempt = c.record_attempt("a", p, new, "raw", {}, observation=observation())
    assert c.annotation("a", p)["evidence"] == new
    assert c.query("training", profile=p, text="vehicle")
    c.review(attempt, "test-reviewer", "rejected", "Object unsupported by image")
    assert c.annotation("a", p) is None
    assert c.query("training", profile=p, text="vehicle") == []
    c.close()
