
import pytest

from driversi.data.catalog.hybrid import HybridRetriever
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore


def fixture(tmp_path, split="navtrain"):
    c = Catalog(tmp_path / "scenes.sqlite", split)
    spec = [("q", "qlog", ["bend", "truck"], [1, 0, 0]),
            ("visual", "visual_log", ["straight"], [1, .01, 0]),
            ("semantic", "semantic_log", ["bend", "truck"], [0, 1, 0]),
            ("shared", "shared_log", ["bend"], [1, .02, 0]),
            ("duplicate", "qlog", ["bend", "truck"], [1, 0, 0])]
    c.ingest([{"token": t, "source_log": g, "kind": "original_" + split} for t,g,_,_ in spec],
             [{"token": t, "source_log": g, "source_group": g, "conditions": tags,
               "history": {"continuous": True}} for t,g,tags,_ in spec],
             {"complete": True, "index_sha256": "i"})
    v = VectorStore(c, writable=True)
    p = v.profile({"dimension": 3, "cameras": ["CAM_F0"]})
    evidence = [{"id": f"h{i}_CAM_F0"} for i in range(4)]
    for t,_,_,values in spec:
        v.put(t,p,evidence,[values]*4,{})
    return c, p


def annotate(c, profile, token, caption, verdict=None):
    attempt = c.record_attempt(token, profile, [{"id": "h3_CAM_F0"}], caption, {},
        observation={"caption": caption, "observations": [], "unknown": []})
    if verdict:
        c.review(attempt,"unit_test",verdict,"controlled fixture")
    return attempt


def test_independent_recall_union_source_exclusion_and_fusion(tmp_path):
    c,p = fixture(tmp_path)
    d = HybridRetriever(c,p).search("q",limit=3,pool=3)
    assert d["routes"]["visual"][0]["token"] == "visual"
    assert d["routes"]["semantic"][0]["token"] == "semantic"
    assert {x["token"] for x in d["results"]} == {"visual", "semantic", "shared"}
    assert all(x["token"] not in ("q", "duplicate") for x in d["results"])
    assert d["caption_channel_active"] is False
    assert d["scores_used_for_ranking"] is False
    assert d["semantic_sources"] == ["structured_conditions"]
    c.close()


def test_only_reviewed_latest_captions_rank_and_statistics_are_isolated(tmp_path):
    c,p = fixture(tmp_path)
    profile=c.profile({"test":"captions"})
    annotate(c,profile,"q","A truck beside a barrier.","spot_check_pass")
    annotate(c,profile,"semantic","A truck beside a barrier.","spot_check_pass")
    bad=annotate(c,profile,"visual","A truck beside a barrier.","rejected")
    annotate(c,profile,"shared","A truck beside a barrier.")
    first=HybridRetriever(c,p,profile).search("q",limit=3,pool=3)
    assert [x["token"] for x in first["routes"]["caption"]] == ["semantic"]
    assert first["reviewed_caption_population"] == 2
    # A newer failed attempt invalidates an earlier reviewed caption.
    c.record_attempt("semantic",profile,[{"id":"h3_CAM_F0"}],"invalid",{},error="bad")
    second=HybridRetriever(c,p,profile).search("q",limit=3,pool=3)
    assert second["routes"]["caption"] == []
    assert second["reviewed_caption_population"] == 1
    assert bad not in [x.get("caption_attempt") for x in first["results"]]
    c.close()


def test_no_semantic_evidence_falls_back_to_visual_and_split_rejected(tmp_path):
    c,p=fixture(tmp_path)
    with c.db:
        c.db.execute("DELETE FROM tags")
    d=HybridRetriever(c,p).search("q",limit=3,pool=3)
    assert d["routes"]["semantic"] == []
    assert [x["token"] for x in d["results"]] == [x["token"] for x in d["routes"]["visual"]]
    with pytest.raises(ValueError,match="caption profile"):
        HybridRetriever(c,p,"missing")
    c.close()
    c,p=fixture(tmp_path / "test", "navtest")
    with pytest.raises(ValueError,match="navtrain"):
        HybridRetriever(c,p)
    c.close()


def test_harness_needs_no_student_measurement_and_rejects_score_config(tmp_path, monkeypatch):
    from driversi.rsi.plugins.scene_catalog import execute, prepare_hybrid
    from driversi_agents.chainflow import benchmark
    c,p=fixture(tmp_path)
    assert c.db.execute("SELECT count(*) FROM scores").fetchone()[0] == 0
    c.close()
    monkeypatch.setattr(benchmark,"official_filter",lambda _: (None,{
        "tokens":["q","visual","semantic","shared","duplicate"],
        "log_names":["qlog","visual_log","semantic_log","shared_log"]}))
    config={"database":"scenes.sqlite","query_token":"q","vector_profile":p,
            "caption_profile":None,"limit":3,"pool":3,"target_category":"progress"}
    prepared,_=prepare_hybrid(config,tmp_path,None)
    d=prepared["report"]
    assert set(d["expansion_tokens"]) == {"visual", "semantic", "shared"}
    assert all(x["status"]=="condition_match_requires_diagnosis" for x in d["candidates"])
    assert all("student_metrics" not in x for x in d["candidates"])
    receipt=execute(prepared,{},tmp_path/"out",None)
    assert receipt["summary"]["test_samples_consumed"] == 0
    with pytest.raises(ValueError):
        prepare_hybrid({**config,"score_run":"forbidden"},tmp_path,None)
    prepared["report"]["expansion_tokens"].append("new")
    with pytest.raises(ValueError,match="changed"):
        execute(prepared,{},tmp_path/"bad",None)


def test_union_preserves_visual_topk_and_reports_extra_budget(tmp_path):
    c,p=fixture(tmp_path)
    d=HybridRetriever(c,p).search("q",limit=2,pool=3)
    visual={x["token"] for x in d["routes"]["visual"]}
    union={x["token"] for x in d["results"]}
    assert visual <= union
    assert "semantic" in union
    assert len(union)==3
    assert len(d["routes"]["fused"])==2
    assert len(d["routes"]["visual_budget_control"])==3
    assert d["maximum_candidates"]==4
    c.close()


def test_union_keeps_visual_representative_when_semantics_prefers_same_log(tmp_path):
    c,p=fixture(tmp_path)
    c.ingest([{"token":"alt","source_log":"visual_log","kind":"original_navtrain"}],
        [{"token":"alt","source_log":"visual_log","source_group":"visual_log",
          "conditions":["bend","truck"]}],{"complete":True,"index_sha256":"i"})
    VectorStore(c).put("alt",p,[{"id":f"h{i}_CAM_F0"} for i in range(4)],[[-1,0,0]]*4,{})
    d=HybridRetriever(c,p).search("q",limit=2,pool=3)
    assert d["routes"]["semantic"][0]["token"] == "alt"
    assert "visual" in {x["token"] for x in d["results"]}
    assert "alt" not in {x["token"] for x in d["results"]}
    assert len({x["source_group"] for x in d["results"]})==len(d["results"])
    c.close()


def test_condition_retrieval_all_channels_without_score_access(tmp_path):
    import sqlite3
    c,p=fixture(tmp_path)
    with c.db:
        c.db.execute("UPDATE tags SET tag='history_turning' WHERE tag='bend'")
        c.db.execute("UPDATE tags SET tag='nearby_vehicle' WHERE tag='truck'")
    profile=c.profile({"test":"condition-captions"})
    annotate(c,profile,"semantic","A barrier beside a curved road.","spot_check_pass")
    annotate(c,profile,"visual","A barrier beside a curved road.","rejected")
    def no_scores(action, table, column, database, source):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table in ("scores","score_runs") else sqlite3.SQLITE_OK
    c.db.set_authorizer(no_scores)
    report=HybridRetriever(c,p,profile).search_conditions(["history_turning"], "curved road barrier",limit=3,pool=3,exemplars=1)
    assert report["caption_channel_active"] is True
    assert report["routes"]["caption"][0]["token"] == "semantic"
    assert report["test_samples_consumed"]==0
    assert len({x["source_group"] for x in report["results"]})==len(report["results"])
    with pytest.raises(ValueError,match="conditions"):
        HybridRetriever(c,p).search_conditions(["test-token-invented-condition"])
    c.close()
