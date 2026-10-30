"""Guard the real production boundary, not numerical model internals."""

import copy
import json
import pickle
from pathlib import Path

import pytest
from PIL import Image

from driversi.agents.backends.native import digest
from driversi.data.catalog.derived_features import history
from driversi.rendering.diagnostics.gs_sequence_critic import CHECKS, checked_response, frame_passes
from driversi.rsi.experiments.reactive_assets import supported_window
from driversi.rsi.experiments.reactive_selection import ordered_failures


def test_full_results_required_and_zero_priority():
    rows = [dict(token=str(i), metrics={"PDMS": s}) for i, s in enumerate([0.94, 0, 1, 0.50])]
    report = dict(complete=True, completed=4, rows=rows)
    assert [x["token"] for x in ordered_failures(report, list("0123"))] == ["1", "3", "0"]
    for bad in (
        dict(report, complete=False),
        dict(report, rows=rows[:3]),
        dict(report, rows=[*rows[:3], rows[2]]),
    ):
        with pytest.raises(ValueError):
            ordered_failures(bad, list("0123"))
    bad = copy.deepcopy(report)
    bad["rows"][0]["metrics"]["PDMS"] = float("nan")
    with pytest.raises(ValueError):
        ordered_failures(bad, list("0123"))


def test_native_arrival_needs_real_preceding_history_and_full_future():
    raw = [dict(token=str(i), timestamp=i * 500000) for i in range(20)]
    canonical = {
        x["token"]: dict(x, cams={k: {} for k in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")})
        for x in raw
    }
    assert not supported_window(raw, 4, canonical)
    assert supported_window(raw, 7, canonical)
    assert not supported_window(raw, 10, canonical)
    canonical["8"]["skipped"] = True
    assert not supported_window(raw, 7, canonical)


def test_actor_uncertainty_vetoes_global_visual_pass():
    facts = dict(target_id="0010-CAM_F0", image_sha256="pixels", actor_ids=["vehicle"])
    value = dict(
        target_id=facts["target_id"],
        image_sha256="pixels",
        checks={k: dict(status="pass", evidence="observed") for k in CHECKS},
        actors=[
            dict(
                actor_id="vehicle",
                appearance_status="pass",
                alignment_status="pass",
                evidence="observed",
            )
        ],
        issues=[],
        void_polygons=[],
    )
    assert frame_passes(checked_response(value, facts))
    value["actors"][0]["alignment_status"] = "unknown"
    assert not frame_passes(checked_response(value, facts))
    value["actors"] = []
    with pytest.raises(ValueError):
        checked_response(value, facts)
    value["image_sha256"] = "other"
    with pytest.raises(ValueError):
        checked_response(value, facts)


def test_generated_history_uses_new_pixels_only_and_rejects_mutation(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    files = {}
    frames = []
    for i in range(14):
        p = root / f"{i}.png"
        Image.new("RGB", (12, 8), (i, 20, 40)).save(p)
        files[str(p)] = digest(p)
        frames.append(dict(timestamp=i * 500000, cams={"CAM_F0": {"data_path": p.name}}))
    log = tmp_path / "new.pkl"
    log.write_bytes(pickle.dumps(frames))
    files[str(log)] = digest(log)
    export = tmp_path / "export.json"
    export.write_text(json.dumps(dict(files=files, log_path=str(log), sensor_root=str(root))))
    _, evidence, images = history(export, ["CAM_F0"], 8)
    assert len(images) == 4 and [x["history_index"] for x in evidence] == [0, 1, 2, 3]
    assert images[3].getpixel((0, 0))[0] == 3
    Image.new("RGB", (12, 8), "red").save(root / "0.png")
    with pytest.raises(ValueError):
        history(export, ["CAM_F0"], 8)


def test_hard_visual_rejection_does_not_call_vlm_or_publish(tmp_path, monkeypatch):
    from driversi.rsi.experiments import reactive_postprocess as mod
    from driversi.rsi.experiments import readiness

    p = tmp_path / "render.json"
    p.write_text("{}")
    source = tmp_path / "source.json"
    source.write_text("{}")
    g = tmp_path / "generation.json"
    g.write_text(
        json.dumps(
            dict(
                status="rendered",
                files={str(p): digest(p), str(source): digest(source)},
                render_report=str(p),
                source_report=str(source),
            )
        )
    )
    c = tmp_path / "config.json"
    c.write_text("{}")
    monkeypatch.setattr(
        readiness, "analyze_render", lambda *a: dict(blockers=["finite_terrain_support_missing"])
    )
    monkeypatch.setattr(
        mod,
        "run_native",
        lambda *a, **k: pytest.fail("Rejected pixels must not consume VLM/train budget"),
    )
    result = mod.process(g, c, tmp_path / "result")
    assert result["status"] == "visual_preflight_rejected" and not result["trainable"]


def test_batch_retrieval_matches_original_and_excludes_non_navval(monkeypatch, tmp_path):
    import sqlite3
    import types

    import numpy as np

    from driversi.data.catalog.vectors import normalize
    from driversi.rsi.experiments import navval_batch_retrieval as new
    from driversi.rsi.experiments import navval_retrieval as old

    def catalog(split):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE scenes(token TEXT PRIMARY KEY,source_group TEXT)")
        db.execute(
            "CREATE TABLE vectors(token TEXT,profile TEXT,status TEXT,centroid BLOB,image_vectors BLOB,views INT,dimension INT,input_id TEXT)"
        )
        return types.SimpleNamespace(split=split, db=db)

    q, t = catalog("navtest"), catalog("navtrain")
    rng = np.random.default_rng(8)
    for cat, n in ((q, 3), (t, 30)):
        for i in range(n):
            vv = normalize(rng.normal(size=(12, 6)))
            v = normalize(vv.mean(0))
            cat.db.execute("INSERT INTO scenes VALUES (?,?)", (str(i), str(i % 7)))
            cat.db.execute(
                "INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?)",
                (
                    str(i),
                    "p",
                    "ready",
                    v.astype("<f4").tobytes(),
                    vv.astype("<f2").tobytes(),
                    12,
                    6,
                    "input-" + str(i),
                ),
            )

    class Store:
        def __init__(self, cat):
            self.db = cat.db

        def profile_data(self, p):
            return {"layout": "same"}

        def get(self, token, p):
            return self.db.execute("SELECT * FROM vectors WHERE token=?", (token,)).fetchone()

    allowed = {str(i): "trainlog-" + str(i % 7) for i in range(24)}
    for mod in (old, new):
        monkeypatch.setattr(mod, "VectorStore", Store)
    monkeypatch.setattr(old, "navval_members", lambda cat: (allowed, {}))
    monkeypatch.setattr(new, "training_members", lambda cat, population: (allowed, {}))
    a = old.retrieve(q, t, "p", ["0", "1", "2"], limit=4)
    b = new.retrieve(q, t, "p", ["0", "1", "2"], limit=4)
    assert a["rows"] == b["rows"]
    assert a["navval_members"] == b["target_members"]
    assert b["target_population"] == "navval"
    assert all(x["token"] in allowed for row in b["rows"] for x in row["candidates"])

    # Moving an unchanged matrix must preserve exact coarse/fine rankings, while
    # a changed catalog must fail closed instead of using stale matrix contents.
    import json

    from driversi.data.catalog import centroid_cache

    cached = list(
        t.db.execute(
            "SELECT token,centroid,source_group FROM vectors JOIN scenes USING(token) ORDER BY token"
        )
    )
    matrix = normalize(np.stack([np.frombuffer(x["centroid"], dtype="<f4") for x in cached]))
    manifest = dict(cache_id="c", profile="p", split="navtrain", collection_id="collection")
    t.path = tmp_path / "catalog.sqlite"
    (tmp_path / "location-binding.json").write_text(
        json.dumps(dict(catalog_signature="same", cache_id="c"))
    )
    monkeypatch.setattr(
        centroid_cache, "open_cache", lambda path, verify: (manifest, cached, matrix)
    )
    monkeypatch.setattr(centroid_cache, "signature", lambda path: "same")
    monkeypatch.setattr(new, "active_collection_identity", lambda *args: "collection")
    c = new.retrieve(q, t, "p", ["0", "1", "2"], limit=4, centroid_cache_path=tmp_path)
    assert c["rows"] == b["rows"]
    monkeypatch.setattr(centroid_cache, "signature", lambda path: "changed")
    with pytest.raises(ValueError, match="stale source"):
        new.retrieve(q, t, "p", ["0"], limit=4, centroid_cache_path=tmp_path)


@pytest.mark.parametrize(
    "vector_hash",
    [
        "d53ea2e357e596715b3ceb53643339fe8d48fe973714afa57992f69e59fe10d5",
        "d63732294e37d2bfa6cd39dde9267ad327b075b299bb65c8ec4a8d9c0a70686f",
    ],
)
def test_encoder_compatibility_rejects_any_non_sql_change(vector_hash):
    from driversi.data.catalog.derived_features import encoder_compatibility

    old = dict(
        source={
            "vectors.py": "2276d62b49992c747f1ca8c785bb40c09ca2013521f9e3ac61465fef1b06f684",
            "features.py": "same",
        },
        runtime="same",
    )
    new = dict(
        source={
            **old["source"],
            "vectors.py": vector_hash,
        },
        runtime="same",
    )
    assert encoder_compatibility(old, new)["compatible"]
    unknown = copy.deepcopy(new)
    unknown["source"]["vectors.py"] = "unreviewed"
    with pytest.raises(ValueError):
        encoder_compatibility(old, unknown)
    new["runtime"] = "different"
    with pytest.raises(ValueError):
        encoder_compatibility(old, new)


def test_generated_search_does_not_expose_quarantined_pixels(tmp_path):
    import sqlite3

    import numpy as np

    from driversi.data.catalog.derived_features import search, search_text

    p = tmp_path / "vectors.sqlite"
    db = sqlite3.connect(p)
    db.execute(
        "CREATE TABLE generated_vectors(id TEXT,views BLOB,receipt TEXT,admission TEXT,searchable INT,centroid BLOB,payload_sha TEXT)"
    )
    db.execute("INSERT INTO generated_vectors VALUES ('unadmitted',NULL,'{}',NULL,0,NULL,NULL)")
    db.execute("CREATE TABLE generated_text(id TEXT,document TEXT,scope TEXT)")
    db.execute(
        "INSERT INTO generated_text VALUES ('unadmitted','turning','executed_four_frame_history_only')"
    )
    db.commit()
    db.close()
    assert (
        search_text(
            p, dict(history={"information_set": "current_state_history"}, conditions=["turning"])
        )
        == []
    )
    assert search(p, np.ones((12, 2048), dtype=np.float32), "profile") == []


def test_historical_text_rejects_future_and_preserves_unknown():
    from driversi.rsi.experiments.full_training_mining import describe

    value = dict(
        history={"information_set": "current_state_history"},
        conditions=["turning"],
        features={"vehicle_count_30m": None, "pedestrian_count_15m": 0},
    )
    text = describe(value)
    assert "vehicle_count_30m_unknown" in text
    assert "pedestrian_count_15m_none" in text
    value["history"]["information_set"] = "future"
    with pytest.raises(ValueError):
        describe(value)


def test_full_training_mapping_preserves_missing_assets_and_native_scope(tmp_path, monkeypatch):
    from driversi.rsi.experiments import navval_retrieval
    from driversi.rsi.experiments import reactive_selection as mod

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    checkpoint = tmp_path / "weights"
    save(checkpoint / "model.safetensors", {})
    baseline = tmp_path / "parent"
    save(baseline / "baseline/report.json", {})
    cfg = save(
        tmp_path / "config.json",
        dict(output=str(baseline), train_catalog="train", checkpoint=str(checkpoint)),
    )
    mining = tmp_path / "mining"
    save(
        mining / "summary.json",
        dict(
            status="completed",
            training_members=103288,
            ready_vectors=2,
            mapped_queries=2,
            zero_count=1,
            nonzero_below_06=1,
        ),
    )
    save(
        mining / "inputs.json",
        dict(
            population="official_navtrain",
            config_sha256=digest(cfg),
            baseline_sha256=digest(baseline / "baseline/report.json"),
        ),
    )
    candidates = [
        dict(token="training-a", source_log="log-a"),
        dict(token="training-b", source_log="log-b"),
    ]
    save(
        mining / "training-production-queue.json", dict(rows=candidates, contains_test_tokens=False)
    )
    coverage = save(
        tmp_path / "coverage.json",
        dict(
            population="navtrain",
            training_members=2,
            files={},
            rows=[dict(candidates[0], source_group="group-a", bundle_root="asset")],
        ),
    )

    class Cat:
        def close(self):
            pass

    monkeypatch.setattr(mod, "Catalog", lambda *a, **k: Cat())
    monkeypatch.setattr(
        navval_retrieval,
        "training_members",
        lambda *a: ({x["token"]: x["source_log"] for x in candidates}, {}),
    )
    result = mod.select_full_training(cfg, mining, coverage, tmp_path / "selection")
    assert result["selected"] == 1 and result["pending_assets"] == 1
    plan = json.loads(Path(result["plan"]).read_text())
    assert plan["assignments"][0]["skill"] == "driverl_full_window"
    assert not plan["assignments"][0]["trainable"]
    assert result == mod.select_full_training(cfg, mining, coverage, tmp_path / "selection")
    skipped = mod.select_full_training(
        cfg, mining, coverage, tmp_path / "excluded", exclude_tokens=["training-a"]
    )
    assert skipped["selected"] == 0 and skipped["excluded_prior_attempts"] == 1
    perturbed = mod.select_full_training(
        cfg, mining, coverage, tmp_path / "perturbed", generation_skill="perturbed_history_tts"
    )
    newplan = json.loads(Path(perturbed["plan"]).read_text())
    assert newplan["assignments"][0]["skill"] == "perturbed_history_tts"
    assert perturbed["inputs"]["generation_skill"] == "perturbed_history_tts"
    with pytest.raises(ValueError):
        mod.select_full_training(cfg, mining, coverage, tmp_path / "perturbed")
    candidates[1]["token"] = "private-test-token"
    save(
        mining / "training-production-queue.json", dict(rows=candidates, contains_test_tokens=True)
    )
    with pytest.raises(ValueError):
        mod.select_full_training(cfg, mining, coverage, tmp_path / "invalid")


def test_vector_only_resume_preserves_private_query_boundary(tmp_path):
    from driversi.rsi.experiments.full_training_mining import vector_only_snapshot

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    full = tmp_path / "parent"
    report = save(full / "baseline/report.json", {})
    config = save(tmp_path / "config.json", dict(output=str(full)))
    source = tmp_path / "source"
    save(
        source / "inputs.json",
        dict(
            config_sha256=digest(config),
            baseline_sha256=digest(report),
            population="official_navtrain",
        ),
    )
    save(
        source / "private-failure-queue.json",
        dict(rows=[dict(token="private-query")], zero_count=1, nonzero_below_06=0),
    )
    save(
        source / "private-vector-retrieval.json",
        dict(
            target_population="navtrain",
            target_members=103288,
            ready_vectors=103044,
            rows=[
                dict(
                    query_token="private-query",
                    candidates=[dict(token="training", source_log="train-log", similarity=0.8)],
                )
            ],
        ),
    )
    output = tmp_path / "output"
    result = vector_only_snapshot(config, source, output)
    assert result["missing_training_vectors"] == 244
    assert "private-query" not in (output / "training-production-queue.json").read_text()
    assert result == vector_only_snapshot(config, source, output)


def test_changed_collection_cannot_overwrite_frozen_mapping(tmp_path):
    from driversi.rsi.experiments.full_training_mining import cache_collection_matches

    old = {
        "target_collection_id": "old",
        "query_collection_id": "q",
        "rows": [{"query_token": "private"}],
    }
    mapping = tmp_path / "private-vector-retrieval.json"
    mapping.write_text(json.dumps(old))
    before = mapping.read_bytes()
    with pytest.raises(ValueError, match="new output directory"):
        cache_collection_matches(old, "new", "q", existing_mapping=mapping.exists())
    assert mapping.read_bytes() == before
    assert not cache_collection_matches(old, "new", "q", existing_mapping=False)
    assert cache_collection_matches(old, "old", "q", existing_mapping=True)


def test_terrain_repair_is_recorded_without_admitting_old_pixels(tmp_path, monkeypatch):
    from driversi.rsi.experiments import reactive_postprocess as mod
    from driversi.rsi.experiments import readiness

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    payload = save(tmp_path / "payload.json", dict(renderer_backend="worldengine_portable"))
    receipt = save(tmp_path / "receipt.json", {})
    source = save(tmp_path / "source.json", {})
    generation = save(
        tmp_path / "generation.json",
        dict(
            status="rendered",
            files={str(p): digest(p) for p in (payload, receipt, source)},
            render_payload=str(payload),
            render_report=str(receipt),
            source_report=str(source),
        ),
    )
    config = save(tmp_path / "config.json", {})
    monkeypatch.setattr(
        readiness, "analyze_render", lambda *args: dict(blockers=["finite_terrain_support_missing"])
    )
    calls = []

    def dispatch(action, arguments, directory, **kwargs):
        calls.append(action)
        assert action == "repair_terrain"
        save(
            Path(arguments["output"]) / "report.json",
            dict(status="ready_for_rerender", failures_before=1, failures_after=0),
        )

    monkeypatch.setattr(mod, "run_native", dispatch)
    result = mod.process(generation, config, tmp_path / "output")
    assert calls == ["repair_terrain"]
    assert result["status"] == "visual_preflight_rejected" and not result["trainable"]
    assert result["terrain_repair"]["rerender_required"]
    assert not result["terrain_repair"]["admission_granted"]
