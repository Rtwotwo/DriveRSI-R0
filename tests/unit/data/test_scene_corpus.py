import json

import numpy as np
import pytest

from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore


def test_pipeline_preserves_order_resume_and_failures(tmp_path, monkeypatch):
    from driversi.data.catalog import corpus
    c = Catalog(tmp_path / "catalog.sqlite", "navtrain")
    c.ingest([{"token": t, "source_log": t, "kind": "original_navtrain"} for t in ["a", "b"]],
             [{"token": t, "source_log": t, "source_group": t} for t in ["a", "b"]], {"complete": True})
    config = {"cameras": ["CAM_F0"], "dimension": 2}
    monkeypatch.setattr(corpus, "feature_profile", lambda _: config)
    evidence = [{"id": f"h{i}_CAM_F0"} for i in range(4)]
    monkeypatch.setattr(corpus, "load_evidence", lambda scene, cfg: (evidence, [scene["record"]["token"]], None))
    def backend(cfg):
        return lambda images: (np.array([[1., 0.]] * 4), {})
    r = corpus.dump_corpus(c, config, tmp_path / "first", backend_factory=backend, workers=2)
    assert r["complete"] and r["ready"] == 2
    def forbidden(cfg):
        raise AssertionError("No inference on exact resume")
    r = corpus.dump_corpus(c, config, tmp_path / "resume", backend_factory=forbidden)
    assert r["reused"] == 2 and r["failed"] == 0
    monkeypatch.setattr(corpus, "load_evidence", lambda *a: ([], [], "missing source"))
    r = corpus.dump_corpus(c, config, tmp_path / "invalid", backend_factory=forbidden)
    assert r["complete"] and r["failed"] == 2
    with pytest.raises(ValueError, match="No ready"):
        VectorStore(c).get("a", r["profile"])
    with pytest.raises(ValueError):
        corpus.dump_corpus(c, config, tmp_path / "bad", tokens=["test_token"])
    c.close()


def test_camera_grounding_cannot_choose_another_camera():
    from driversi.data.catalog.grounding import combine_camera_observations, parse_camera
    value = {"caption": "A vehicle at image-left.", "observations": [
        {"tag": "vehicle_to_side_visible", "description": "Vehicle at image-left."}], "unknown": ["motion"]}
    e = [{"id": "h3_CAM_L0", "camera": "CAM_L0", "history_index": 3}]
    p = parse_camera(json.dumps(value), e[0])
    assert p["observations"][0]["evidence"] == ["h3_CAM_L0"]
    merged = combine_camera_observations([(e[0], p)], e)
    assert merged["caption"].startswith("CAM_L0:")
    value["observations"][0]["evidence"] = "h3_CAM_F0"
    with pytest.raises(ValueError, match="must not assign"):
        parse_camera(json.dumps(value), e[0])


def test_fine_marking_claims_reject_not_rewrite():
    from driversi.data.catalog.grounding import parse_camera
    row = {"id": "h3_CAM_F0"}
    value = {"caption": "A dashed line on the road.", "observations": [], "unknown": []}
    with pytest.raises(ValueError, match="fine marking"):
        parse_camera(json.dumps(value), row)


def test_camera_bundle_reparses_raw_and_checks_coverage():
    from driversi.data.catalog.prompts import parse_annotation_response
    e = [{"id": "h3_CAM_F0", "camera": "CAM_F0", "history_index": 3}]
    raw = json.dumps({"caption": "A road.", "observations": [], "unknown": []})
    bundle = {"canonical": "not trusted", "camera_responses": [{"image_id": "h3_CAM_F0", "raw": raw}]}
    result = parse_annotation_response(json.dumps(bundle), e, {"prompt_version": "camera_grounded_v6"})
    assert result["caption"] == "CAM_F0: A road."
    bundle["camera_responses"] *= 2
    with pytest.raises(ValueError, match="coverage"):
        parse_annotation_response(json.dumps(bundle), e, {"prompt_version": "camera_grounded_v6"})


def test_expansion_plan_is_training_only_and_preserves_score_evidence(tmp_path, monkeypatch):
    from driversi.rsi.decisions.scene_expansion import expansion_plan
    from driversi_agents.chainflow import benchmark
    c = Catalog(tmp_path / "train.sqlite", "navtrain")
    tokens = ["a", "b", "c"]
    c.ingest([{"token": t, "source_log": t, "kind": "original_navtrain"} for t in tokens],
             [{"token": t, "source_log": t, "source_group": t, "history": {"continuous": t != "c"}} for t in tokens],
             {"complete": True, "index_sha256": "i"})
    v = VectorStore(c, writable=True)
    p = v.profile({"cameras": ["CAM_F0"], "dimension": 2})
    evidence = [{"id": f"h{i}_CAM_F0"} for i in range(4)]
    for t in tokens:
        v.put(t, p, evidence, [[1, 0]]*4, {})
    base = {k: 1 for k in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "time_to_collision_within_bound", "comfort")}
    rows = [{"token": t, "log_name": t, "valid": True, "score": .7, "ego_progress": .6, **base} for t in tokens]
    run = c.import_train_scores(rows, {"split": "navtrain", "purpose": "training_pool_mining", "index_sha256": "i"})
    monkeypatch.setattr(benchmark, "official_filter", lambda _: (None, {"tokens": tokens, "log_names": tokens}))
    result = expansion_plan(c, run, p, per_category=3, probes=1, neighbors=2)
    assert result["test_samples_consumed"] == 0
    assert len(result["candidates"]) == 3
    assert next(x for x in result["candidates"] if x["token"] == "c")["synthesis_requested"] is False
    assert result["retrieval_validation"][0]["unfiltered_vector_same_metric_hits"] == 2
    assert all(x["teacher_pdms_minimum"] == 95 and not x["trainable"] for x in result["candidates"])
    c.close()
    test = Catalog(tmp_path / "test.sqlite", "navtest")
    with pytest.raises(ValueError, match="training catalog"):
        expansion_plan(test, run, p)
    test.close()


def test_camera_json_completion_fence_and_v7_tag_wire():
    from driversi.data.catalog.grounding import parse_camera
    row = {"id": "h3_CAM_F0", "camera": "CAM_F0"}
    raw = '{"caption":"Road beside a hedge.","tags":["straight_road_visible"],"unknown":[]}'
    value = parse_camera('```json\n'+raw, row, 'camera_grounded_v7')
    assert value['observations'][0]['description'] == value['caption']
    assert parse_camera(raw+'}', row, 'camera_grounded_v7') == value
    with pytest.raises(ValueError, match='Unexpected'):
        parse_camera(raw+' commentary', row, 'camera_grounded_v7')
