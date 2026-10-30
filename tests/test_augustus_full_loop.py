import json

import pytest

from driversi.agents.backends.native import digest
from driversi_agents.augustus import benchmark
from driversi_agents.augustus.full_evaluation import aggregate_rows


def sample(token="a", score=0.5):
    return dict(
        token=token,
        source_log="test",
        metrics={k: score for k in ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C")},
    )


def test_partial_test_never_has_formal_score():
    summary = aggregate_rows(
        [sample()], [dict(token="a", source_log="test"), dict(token="b", source_log="test")]
    )
    assert summary["metrics"] is None and not summary["formal_score_available"]
    assert summary["completed"] == 1 and summary["missing"] == 1


def test_full_score_duplicate_and_nonfinite():
    records = [dict(token="a", source_log="test"), dict(token="b", source_log="test")]
    result = aggregate_rows([sample("a", 0), sample("b", 1)], records)
    assert result["metrics"]["PDMS"] == 0.5 and result["zero_count"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_rows([sample(), sample()], records)
    with pytest.raises(ValueError, match="Invalid"):
        aggregate_rows([sample(score=float("nan"))], records)
    with pytest.raises(ValueError, match="Unexpected"):
        aggregate_rows([sample("train")], records)


def test_full_membership_requires_all12146(tmp_path, monkeypatch):
    p = tmp_path / "index.json"
    p.write_text("frozen")
    rows = [dict(token=str(i), source_log="test") for i in range(12146)]
    monkeypatch.setattr(benchmark, "read_index", lambda *a: dict(records=rows))
    monkeypatch.setattr(
        benchmark,
        "official_filter",
        lambda *a: (p, dict(tokens=["train", "val"], log_names=["training", "validation"])),
    )
    value = dict(
        benchmark=dict(
            protocol="official_navtest_v1", test_subset=str(p), test_subset_sha256=digest(p)
        ),
        splits=dict(
            train=dict(records=[dict(frame_id="train", log_name="training")]),
            validation=dict(records=[dict(frame_id="val", log_name="validation")]),
            development=dict(
                records=[dict(frame_id=x["token"], log_name=x["source_log"]) for x in rows]
            ),
        ),
    )
    benchmark.validate_membership(value)
    value["splits"]["development"]["records"].pop()
    with pytest.raises(ValueError, match="population"):
        benchmark.validate_membership(value)


def test_training_cannot_map_partial_evaluation(tmp_path):
    from driversi.rsi.experiments.navval_loop import retrieve_and_relabel

    root = tmp_path / "run"
    (root / "baseline").mkdir(parents=True)
    (root / "baseline/report.json").write_text(json.dumps(dict(complete=False, completed=32)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(output=str(root), evaluation_count=12146)))
    with pytest.raises(ValueError, match="completed full"):
        retrieve_and_relabel(config)


def test_relabel_memory_record_is_idempotent(tmp_path, monkeypatch):
    from driversi.rsi.experiments import augustus_pair

    observations = []

    class Memory:
        def __init__(self, path):
            pass

        def record(self, obs):
            observations.append(obs)
            return dict(ok=True)

        def close(self):
            pass

    monkeypatch.setattr(augustus_pair, "SkillMemory", Memory)
    report = dict(
        parent_assets={},
        evaluation_scope="full_official_navtest_adaptive",
        augmentation_kind="original_anchor_relabel",
        gain_vs_control_pdms_points=0.1,
        safety_regressions=[],
        training_evaluation_costs={},
    )
    for _ in range(2):
        augustus_pair.record(
            report, tmp_path / "report.json", tmp_path / "memory.sqlite", resume=True
        )
    assert observations[0] == observations[1]
    assert observations[0]["skill"] == "same_anchor_relabel"
    with pytest.raises(ValueError, match="Preserve"):
        augustus_pair.record(
            dict(report, gain_vs_control_pdms_points=1),
            tmp_path / "report.json",
            tmp_path / "memory.sqlite",
        )


def test_generated_vectors_reuse_images_without_test_data(tmp_path, monkeypatch):
    import hashlib

    import numpy as np

    from driversi.data.catalog.augmentations import register, search
    from driversi.data.catalog.store import Catalog
    from driversi.data.catalog.vectors import VectorStore
    from driversi.training.sampling.environment import source_log_group
    from driversi_agents.augustus import data

    image = tmp_path / "image.jpg"
    image.write_bytes(b"test-only-image")
    cat = Catalog(tmp_path / "train.sqlite", "navtrain")
    cat.ingest(
        [dict(kind="original_navtrain", token="train", source_log="train-log")],
        [
            dict(
                token="train",
                source_log="train-log",
                source_group=source_log_group("train-log"),
                conditions=[],
            )
        ],
        {},
    )
    vs = VectorStore(cat, writable=True)
    profile = vs.profile(dict(cameras=["CAM_F0"], dimension=2))
    evidence = [dict(id=f"h{i}_CAM_F0", path=str(image), sha256=digest(image)) for i in range(4)]
    vs.put("train", profile, evidence, np.array([[1, 0]] * 4), dict(calls=1))
    cat.close()
    original = tmp_path / "original.json"
    original.write_text("original")
    target = tmp_path / "target.json"
    target.write_text("target")
    left = dict(
        frame_id="train",
        video=b"identical-history",
        historical_x_y_angle="history",
        future_x_y_angle="old",
        solution_prediction="same cot",
        solution_planning="old",
        reasoning_trace="old",
    )
    right = dict(left, future_x_y_angle="new", solution_planning="", reasoning_trace="")
    monkeypatch.setattr(
        data,
        "manifest",
        lambda path, **k: dict(
            splits=dict(train=dict(cache="original" if str(path) == str(original) else "target"))
        ),
    )
    monkeypatch.setattr(
        data, "read_cache", lambda block: [left if block["cache"] == "original" else right]
    )
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            dict(
                kind="augustus_relabel_dataset",
                new_images=0,
                source_manifest_sha256=digest(original),
                manifest=str(target),
                manifest_sha256=digest(target),
                source_cache_files={},
                cache_files={},
                replacements=[
                    dict(token="train", video_sha256=hashlib.sha256(right["video"]).hexdigest())
                ],
            )
        )
    )
    db = tmp_path / "generated.sqlite"
    one = register(tmp_path / "train.sqlite", profile, original, proof, db)
    two = register(tmp_path / "train.sqlite", profile, original, proof, db)
    assert one == two and one["encoder_calls"] == 0
    results = search(db, profile, [1, 0])
    assert len(results) == 1 and results[0]["cosine"] == pytest.approx(1)
    assert results[0]["sample"]["source_token"] == "train"
    right["video"] = b"changed-view"
    with pytest.raises(ValueError, match="fresh feature"):
        register(tmp_path / "train.sqlite", profile, original, proof, db)
    right["video"] = left["video"]
    image.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Source image changed"):
        register(tmp_path / "train.sqlite", profile, original, proof, db)


@pytest.mark.parametrize("wrong_parent", [False, True])
def test_full_candidate_rejects_insufficient_budget_or_other_parent(
    tmp_path, monkeypatch, wrong_parent
):
    from driversi_agents.augustus import full_pair_evaluation as candidate

    parent = tmp_path / "parent"
    parent.mkdir()
    weight = parent / "model.safetensors"
    weight.write_bytes(b"unit-test-weight")
    source = tmp_path / "plan.json"
    source.write_text("frozen protocol")
    (tmp_path / "report.json").write_text(
        json.dumps(dict(complete=True, plan_sha256=digest(source)))
    )
    frozen = dict(
        config=dict(checkpoint=str(parent)),
        inputs={str(weight): digest(weight)},
        output=str(tmp_path),
        records=[{}],
    )
    monkeypatch.setattr(candidate, "check_plan", lambda p: frozen)
    plan = dict(
        data=dict(full_evaluation_plan=str(source)),
        model_config=dict(weights=dict(parent=str(tmp_path / "wrong" if wrong_parent else parent))),
        output=str(tmp_path / "candidate"),
        resolved=dict(request=dict(budget=dict(max_scoring_agent_steps=40))),
    )
    with pytest.raises(ValueError, match="parent differs" if wrong_parent else "scoring budget"):
        candidate.run(plan, dict(checkpoint="unused"))
