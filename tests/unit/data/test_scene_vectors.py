import json

import numpy as np
import pytest

from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore


def populated(tmp_path, split="navtrain"):
    c = Catalog(tmp_path / (split + ".sqlite"), split)
    records, descriptors = [], []
    for token, group in [("a", "ga"), ("b", "gb"), ("c", "gc"), ("d", "ga"), ("e", "gb")]:
        records.append({"token": token, "source_log": group, "kind": "original_" + split})
        descriptors.append({"token": token, "source_log": group, "source_group": group,
                            "conditions": ["intersection"] if token in ("a", "b") else []})
    c.ingest(records, descriptors, {"complete": True, "index_sha256": "i"})
    v = VectorStore(c, writable=True)
    p = v.profile({"dimension": 3, "cameras": ["CAM_F0"]})
    evidence = [{"id": f"h{i}_CAM_F0", "sha256": "original"} for i in range(4)]
    for token, x in [("a", [1, 0, 0]), ("b", [1, .1, 0]), ("c", [0, 1, 0]),
                     ("d", [1, 0, 0]), ("e", [1, .2, 0])]:
        v.put(token, p, evidence, [x] * 4, {})
    return c, v, p, evidence


def test_exact_retrieval_filter_and_source_diversity(tmp_path):
    c, v, p, _ = populated(tmp_path)
    report = v.search("training", "a", p, limit=3)
    assert [r["token"] for r in report["results"]] == ["b", "c"]
    assert report["scanned_vectors"] == 3
    assert report["results"][0]["aligned_view_cosine"] > .99
    report = v.search("training", "a", p, limit=3, exclude_source=False, diverse_sources=False)
    assert report["results"][0]["token"] == "d"
    report = v.search("training", "a", p, filters={"tags": ["intersection"]})
    assert [r["token"] for r in report["results"]] == ["b"]
    c.close()


def test_split_query_and_unknown_profile_fail_closed(tmp_path):
    c, v, p, _ = populated(tmp_path, "navtest")
    with pytest.raises(ValueError, match="disagree"):
        v.search("training", "a", p)
    with pytest.raises(ValueError, match="profile"):
        v.search("evaluation", "a", "unmatched_model")
    with pytest.raises(ValueError, match="No ready"):
        v.search("evaluation", "train_token_not_in_test", p)
    assert v.search("evaluation", "a", p)["results"]
    c.close()


def test_invalid_and_changed_input_invalidates_resume(tmp_path):
    c, v, p, evidence = populated(tmp_path)
    assert v.completed("a", p, evidence)
    changed = [{**x, "sha256": "new"} for x in evidence]
    assert not v.completed("a", p, changed)
    for values in (np.zeros((4, 3)), np.full((4, 3), np.nan), np.ones((3, 3))):
        with pytest.raises(ValueError):
            v.put("a", p, evidence, values, {})
    with pytest.raises(ValueError, match="order"):
        v.put("a", p, evidence[::-1], np.ones((4, 3)), {})
    v.put("a", p, changed, None, {}, "decode failed")
    assert not v.completed("a", p, evidence)
    with pytest.raises(ValueError, match="No ready"):
        v.search("training", "a", p)
    assert all(r["token"] != "a" for r in v.search("training", "b", p)["results"])
    c.close()


def test_rerank_preserves_camera_time_layout(tmp_path):
    c, v, p, e = populated(tmp_path)
    a = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    v.put("a", p, e, a, {})
    v.put("b", p, e, a, {})
    v.put("c", p, e, a[::-1], {})
    report = v.search("training", "a", p, filters={}, limit=3)
    assert report["results"][0]["token"] == "b"
    bad = next(r for r in report["results"] if r["token"] == "c")
    assert bad["coarse_cosine"] == pytest.approx(1, abs=1e-6)
    assert bad["aligned_view_cosine"] == 0
    c.close()


def test_token_query_not_truncated_and_still_validates_purpose(tmp_path):
    c, _, _, _ = populated(tmp_path)
    sql, args = c.token_query("training", tags=["intersection"])
    assert [r[0] for r in c.db.execute(sql, args)] == ["a", "b"]
    with pytest.raises(ValueError):
        c.token_query("evaluation")
    c.close()


def test_versioned_retrieval_prompt_preserves_old_prompt():
    from driversi.data.catalog.observations import PROMPT, parse_response
    from driversi.data.catalog.prompts import annotation_prompt

    assert annotation_prompt({}) == PROMPT
    prompt = annotation_prompt({"prompt_version": "retrieval_v2"})
    assert "Do not force different labels" in prompt
    assert "BOTH times" in prompt
    with pytest.raises(ValueError):
        annotation_prompt({"prompt_version": "missing"})
    value = {"caption": "A truck obscures the junction ahead.", "observations": [
        {"tag": "junction_occluded_visible", "evidence": ["h3_CAM_F0"], "description": "truck ahead"}
    ], "unknown": ["right of way"]}
    assert parse_response(json.dumps(value), [{"id": "h3_CAM_F0"}]) == value


def test_harness_vector_snapshot_keeps_evidence_and_cannot_mutate(tmp_path, monkeypatch):
    from driversi.rsi.plugins.scene_catalog import execute, prepare
    from driversi_agents.chainflow import benchmark

    c, _, profile, _ = populated(tmp_path)
    c.close()
    monkeypatch.setattr(benchmark, "official_filter", lambda _: (None, {
        "tokens": ["a", "b", "c", "d", "e"], "log_names": ["ga", "gb", "gc"]}))
    cfg = {"database": "navtrain.sqlite", "structured_tags": [], "visual_tags": [],
           "profile": None, "text": None, "limit": 2, "score_run": None, "score_below": None,
           "vector": {"query_token": "a", "profile": profile, "shortlist": 16,
                      "exclude_source": True, "diverse_sources": True}}
    prepared, _ = prepare(cfg, tmp_path, None)
    assert len(prepared["report"]["vector_retrieval"]["query_evidence"]) == 4
    assert [r["token"] for r in prepared["report"]["candidates"]] == ["b", "c"]
    result = execute(prepared, {}, tmp_path / "out", None)
    assert result["summary"]["test_samples_consumed"] == 0
    prepared["report"]["vector_retrieval"]["results"][0]["aligned_view_cosine"] = 0
    with pytest.raises(ValueError, match="changed"):
        execute(prepared, {}, tmp_path / "bad", None)


def test_score_and_rejected_label_filters_apply_before_ranking(tmp_path):
    c, v, p, _ = populated(tmp_path)
    metrics = {key: 1 for key in ("no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance", "ego_progress", "time_to_collision_within_bound", "comfort")}
    rows = [{"token": t, "log_name": g, "valid": True, "score": score, **metrics}
            for t, g, score in [("b", "gb", .99), ("c", "gc", .6)]]
    run = c.import_train_scores(rows, {"split": "navtrain", "purpose": "training_pool_mining",
                                     "index_sha256": "i"})
    result = v.search("training", "a", p, filters={"score_run": run, "score_below": .95})
    assert [r["token"] for r in result["results"]] == ["c"]
    label_profile = c.profile({"prompt": "versioned"})
    for t in ("b", "c"):
        attempt = c.record_attempt(t, label_profile, [{"id": "h3_CAM_F0"}], "raw", {},
            observation={"caption": "vehicle ahead", "observations": [{"tag": "vehicle_ahead_visible",
                "evidence": ["h3_CAM_F0"], "description": "vehicle ahead"}], "unknown": []})
        if t == "b":
            c.review(attempt, "reviewer", "rejected", "not visible")
    result = v.search("training", "a", p, filters={"profile": label_profile,
                                                 "visual_tags": ["vehicle_ahead_visible"]})
    assert [r["token"] for r in result["results"]] == ["c"]
    c.close()


def test_dump_invalid_output_removes_stale_ready_snapshot(tmp_path, monkeypatch):
    from driversi.data.catalog import features

    c, v, p, evidence = populated(tmp_path)
    config = {"cameras": ["CAM_F0"], "max_side": 512}
    monkeypatch.setattr(features, "feature_profile", lambda _: v.profile_data(p))
    new = [{**e, "sha256": "changed"} for e in evidence]
    monkeypatch.setattr(features, "history_evidence", lambda *args: (new, []))
    monkeypatch.setattr(features, "VisionEncoder", lambda _: lambda _: (np.zeros((4, 3)), {}))
    report = features.dump_batch(c, config, ["a"], tmp_path / "dump", max_scenes=1)
    assert report["rows"][0]["status"] == "failed"
    assert "zero-norm" in report["rows"][0]["error"]
    with pytest.raises(ValueError, match="No ready"):
        v.search("training", "a", p)
    c.close()


def test_batch_plan_exact_coverage_and_split_purpose(tmp_path):
    from driversi.data.catalog.batches import plan_batches

    c, _, _, _ = populated(tmp_path)
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    out = tmp_path / "plan"
    p = plan_batches(c, "training", cfg, out, "embed", 2)
    assert p["kind"] == "planned_not_executed"
    assert p["scene_count"] == 5
    assert [x["tokens"] for x in p["jobs"]] == [2, 2, 1]
    values = [t for f in sorted(out.glob("tokens-*.json")) for t in json.loads(f.read_text())]
    assert values == ["a", "b", "c", "d", "e"]
    assert len(set(values)) == len(values)
    with pytest.raises(ValueError, match="disagree"):
        plan_batches(c, "evaluation", cfg, tmp_path / "test", "embed")
    with pytest.raises(FileExistsError):
        plan_batches(c, "training", cfg, out, "embed")
    c.close()


def test_single_reference_contract_is_lossless_and_does_not_repair():
    from driversi.data.catalog.prompts import parse_annotation_response

    cfg = {"prompt_version": "retrieval_v4"}
    e = [{"id": "h3_CAM_F0", "history_index": 3}, {"id": "h0_CAM_F0", "history_index": 0}]
    value = {"caption": "Bus ahead obscures the junction.", "unknown": ["motion"], "observations": [
        {"tag": "junction_occluded_visible", "evidence": "h3_CAM_F0", "description": "bus ahead"}
    ]}
    result = parse_annotation_response(json.dumps(value), e, cfg)
    assert result["observations"][0]["evidence"] == ["h3_CAM_F0"]
    for reference in ("h0_CAM_F0", "h3_CAM_MISSING", ["h3_CAM_F0"], ""):
        value["observations"][0]["evidence"] = reference
        with pytest.raises(ValueError):
            parse_annotation_response(json.dumps(value), e, cfg)
    del value["unknown"]
    value["observations"][0]["evidence"] = "h3_CAM_F0"
    with pytest.raises(ValueError):
        parse_annotation_response(json.dumps(value), e, cfg)


def test_equivalent_singleton_container_normalization_is_explicit():
    from driversi.data.catalog.prompts import parse_annotation_response

    e = [{"id": "h3_CAM_F0", "history_index": 3}]
    v = {"caption": "Barrier beside the road.", "unknown": "motion",
         "observations": [{"tag": "cone_or_barrier_visible", "evidence": ["h3_CAM_F0"],
                           "description": "barrier beside the road"}]}
    result = parse_annotation_response(json.dumps(v), e, {"prompt_version": "retrieval_v5"})
    assert result["unknown"] == ["motion"]
    assert result["observations"][0]["evidence"] == ["h3_CAM_F0"]
    with pytest.raises(ValueError):
        parse_annotation_response(json.dumps(v), e, {"prompt_version": "retrieval_v4"})
    for bad in (None, [], ["h3_CAM_F0", "h3_CAM_F0"], ["h3_missing"]):
        v["observations"][0]["evidence"] = bad
        with pytest.raises(ValueError):
            parse_annotation_response(json.dumps(v), e, {"prompt_version": "retrieval_v5"})
