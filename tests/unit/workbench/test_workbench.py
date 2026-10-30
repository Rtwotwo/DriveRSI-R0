import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from driversi.workbench.catalog import SceneCatalog
from driversi.workbench.server import make_handler


def test_suite_rejects_protected_and_duplicate_sources(tmp_path):
    path = tmp_path / "suite.json"
    data = {"official_test": True, "train": [], "evaluation": []}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-official"):
        SceneCatalog(path)
    data["official_test"] = False
    row = {"path": str(tmp_path / "a.npz"), "source_group": "log1", "city": "x",
           "actors": 2, "seconds": 3}
    data["train"] = [row]
    data["evaluation"] = [{**row, "path": str(tmp_path / "b.npz")}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap"):
        SceneCatalog(path)


def test_suite_blocks_outside_paths(tmp_path):
    path = tmp_path / "suite.json"
    data = {"official_test": False, "train": [{"path": "/etc/a.npz"}], "evaluation": []}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="within"):
        SceneCatalog(path)


def test_http_only_catalog_routes(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"official_test": False, "train": [], "evaluation": []}))
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SceneCatalog(p)))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}"
    try:
        with urlopen(url + "/api/health") as response:
            assert json.load(response)["read_only"]
        for headers in ({"Host": "attacker.example"},
                        {"Origin": "http://attacker.example"},
                        {"Origin": "null"},
                        {"Host": "localhost:bad"},
                        {"Sec-Fetch-Site": "cross-site"}):
            with pytest.raises(HTTPError) as denied:
                urlopen(Request(url + "/api/catalog", headers=headers))
            assert denied.value.code == 403
        with urlopen(Request(url + "/api/health", headers={"Origin": url})) as response:
            assert json.load(response)["read_only"]
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/api/scene?id=../../etc/passwd")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/etc/passwd")
        assert error.value.code == 404
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_newly_reserved_gate_log_hidden_even_after_cache(straight_scene, tmp_path):
    from driversi.rsi.experiments.campaign_store import CampaignStore
    source = tmp_path / "gate-scene.npz"
    straight_scene.save(source)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"official_test": False, "train": [{"path": str(source),
        "source_group": "private-log", "city": "synthetic", "actors": 2, "seconds": 3}],
        "evaluation": []}))
    store = CampaignStore(tmp_path / "ledger.sqlite")
    catalog = SceneCatalog(suite)
    assert catalog.scene("gate-scene")["id"] == "gate-scene"
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(catalog, store.registry))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}"
    try:
        with store.registry._connect() as db:
            db.execute("INSERT INTO campaign_groups VALUES ('TEST_ONLY','private-log','promotion')")
        with urlopen(url + "/api/catalog") as response:
            assert json.load(response)["scenes"] == []
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/api/scene?id=gate-scene")
        assert error.value.code == 400
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_protected_groups_hide_later_standalone_development_records(tmp_path):
    from driversi.rsi.experiments.campaign_store import CampaignStore
    from driversi.rsi.contracts import BUDGET_FIELDS
    store = CampaignStore(tmp_path / "ledger.sqlite")
    with store.registry._connect() as db:
        db.execute("INSERT INTO campaign_groups VALUES ('TEST_ONLY','log-b','promotion')")
    arm = {"version": "base", "initial_model": "m", "information_set": "i", "tools": "t"}
    plan = {"schema_version": 1, "id": "standalone", "target": "model", "hypothesis": "TEST ONLY",
        "arms": {"baseline": arm, "candidate": {**arm, "version": "new"}},
        "protocol": {"id": "p", "evaluator_version": "driversi.teacher.v1", "role": "development",
                     "task_set": "tasks", "metric": "goal_per_actor", "direction": "maximize"},
        "data": {"development_groups": ["log-a"], "evaluation_groups": ["log-b"]},
        "budget_per_arm": dict.fromkeys(BUDGET_FIELDS, 1), "scene_ids": ["protected-example"]}
    store.registry.register(plan)
    assert store.registry.public_summary()["experiments"] == []
    plan["id"] = "public"
    plan["data"]["evaluation_groups"] = ["log-c"]
    plan["scene_ids"] = ["public-example"]
    store.registry.register(plan)
    public = store.registry.public_summary()["experiments"]
    assert len(public) == 1 and public[0]["plan"]["id"] == "public"


def test_quality_report_is_explicit_snapshot_without_raw_case_leakage(tmp_path):
    from driversi.workbench.server import quality_summary

    path = tmp_path / "quality.json"
    value = {"kind": "platform_quality_report", "schema_version": 1, "generated_at": "2026-10-15T00:00:00+00:00",
             "private_cases": ["hidden-case"], "components": [{"name": "rendering", "status": "limited",
             "evidence": "test", "limits": ["novel views unverified"],
             "metrics": {"camera_frames": 10}, "private_case": "hidden-case"}]}
    path.write_text(json.dumps(value))
    result = quality_summary(path)
    assert result["status"] == "evidence_snapshot" and result["industrial_ready"] is False
    assert "hidden-case" not in json.dumps(result)
    assert quality_summary(None)["status"] == "unavailable"
    value["components"][0]["status"] = "industrial_ready"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        quality_summary(path)


def test_quality_route_cannot_select_browser_file(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"official_test": False, "train": [], "evaluation": []}))
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SceneCatalog(p)))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}"
    try:
        with urlopen(url + "/api/quality") as response:
            assert json.load(response)["status"] == "unavailable"
        with pytest.raises(HTTPError) as denied:
            urlopen(url + "/api/quality?path=/etc/passwd")
        assert denied.value.code == 400
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize("field", ["generated_at", "metrics"])
def test_quality_rejects_identity_payload_in_aggregate_fields(tmp_path, field):
    from driversi.workbench.server import quality_summary

    value = {"kind": "platform_quality_report", "schema_version": 1,
             "generated_at": "2026-10-15T00:00:00+00:00",
             "components": [{"name": "render", "status": "limited", "evidence": "redacted summary",
                             "limits": [], "metrics": {"camera_frames": 1}}]}
    if field == "generated_at":
        value[field] = {"private_cases": ["protected-fixture"]}
    else:
        value["components"][0][field] = {"source_token": "protected-fixture"}
    p = tmp_path / "quality.json"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        quality_summary(p)


def test_harness_http_is_observation_only(tmp_path):
    from driversi.rsi.workflow.store import HarnessStore
    from tests.unit.rsi.test_harness import prepare, providers, request
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"official_test": False, "train": [], "evaluation": []}))
    store = HarnessStore(tmp_path / "db")
    store.register(prepare(request(), tmp_path, tmp_path / "db", providers()), tmp_path / "output")
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SceneCatalog(path), store.registry))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}/api/harness"
    try:
        with urlopen(url) as response:
            value = json.load(response)
            assert value["read_only"] and value["tasks"][0]["id"] == "task"
            assert "directory" not in value["tasks"][0]
        for request_value in (url + "?path=/etc/passwd", Request(url, method="POST")):
            with pytest.raises(HTTPError):
                urlopen(request_value)
        assert store.get("task")["status"] == "registered"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
