import json

import numpy as np
import pytest

from driversi.data.catalog.centroid_cache import digest, write_json
from driversi.rsi.experiments.scene_cluster_mapping import run, validate_priority


def make_cache(path, split, count):
    path.mkdir()
    rng = np.random.default_rng(count)
    vectors = rng.normal(size=(count, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(path / "centroids.npy", vectors)
    rows = [dict(token=f"{split}_{i:03}", source_log=f"log{i % 12}",
                 source_group=f"g{i % 12}", features={"ego_speed_mps": i % 17},
                 conditions=["ego_moving"], information_set="current_state_history",
                 history_continuous=i != 0, input_id=f"input_{split}_{i}") for i in range(count)]
    write_json(path / "members.json", rows)
    write_json(path / "manifest.json", dict(
        status="complete", split=split, profile="same_encoder", encoder={"dimension": 16},
        cache_id=f"{split}_version1", count=count, dimension=16,
        files={n: digest(path / n) for n in ["centroids.npy", "members.json"]}))
    return rows


def test_real_mapping_restart_and_private_queue(tmp_path, monkeypatch):
    import driversi.rsi.experiments.scene_cluster_mapping as module

    train = make_cache(tmp_path / "train", "navtrain", 48)
    test = make_cache(tmp_path / "test", "navtest", 12)
    monkeypatch.setattr(module, "_verify_membership", lambda m, r: {"split": m["split"]})
    baseline = tmp_path / "parent" / "baseline"
    baseline.mkdir(parents=True)
    write_json(baseline / "plan.json", {"records": test})
    write_json(baseline / "report.json", dict(
        complete=True, expected=12, completed=12, plan_sha256=digest(baseline / "plan.json"),
        rows=[dict(token=r["token"], metrics={"PDMS": 0.1 if i < 2 else 0.9})
              for i, r in enumerate(test)]))
    write_json(tmp_path / "parent.json", {"output": str(tmp_path / "parent")})
    write_json(tmp_path / "queue.json", {"rows": [{"token": r["token"]} for r in test[:2]]})
    config = dict(train_cache=str(tmp_path / "train"), test_cache=str(tmp_path / "test"),
                  baseline_config=str(tmp_path / "parent.json"),
                  priority_queue=str(tmp_path / "queue.json"),
                  clusters=4, probes=2, fit_samples=32, iterations=3,
                  benchmark_queries=4, expansion_per_query=20)
    write_json(tmp_path / "config.json", config)
    out = tmp_path / "run"
    first = run(tmp_path / "config.json", out)
    assert first["mapped_queries"] == 2 and first["semantic_recall"] is None
    saved = (out / "production/training-production-queue.json").read_bytes()
    assert run(tmp_path / "config.json", out)["reused"]
    assert (out / "production/training-production-queue.json").read_bytes() == saved
    production = json.loads(saved)
    assert production["contains_test_tokens"] is False
    assert {r["token"] for r in production["rows"]} <= {r["token"] for r in train}
    assert all(not r["trainable"] for r in production["rows"])
    model = json.loads((out / "model.json").read_text())
    for split in ["navtrain", "navtest"]:
        assert json.loads((out / split / "pool.json").read_text())["model_id"] == model["model_id"]
    (out / "model.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="output changed"):
        run(tmp_path / "config.json", out)


def test_priority_rejects_missing_query_and_score_scale():
    plan = {"records": [{"token": "a"}, {"token": "b"}]}
    report = dict(complete=True, expected=2, completed=2, plan_sha256="p",
                  rows=[{"token": "a", "metrics": {"PDMS": 0}},
                        {"token": "b", "metrics": {"PDMS": 0.4}}])
    assert validate_priority(report, plan, "p", ["a", "b"], {"a", "b"}) == (1, 1)
    with pytest.raises(ValueError, match="complete zero/low"):
        validate_priority(report, plan, "p", ["a"], {"a", "b"})
    report["rows"][1]["metrics"]["PDMS"] = 40
    with pytest.raises(ValueError, match="Normalized"):
        validate_priority(report, plan, "p", ["a", "b"], {"a", "b"})
