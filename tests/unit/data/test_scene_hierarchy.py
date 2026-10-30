import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
from driversi.data.catalog.scene_hierarchy import (
    ancestor,
    anchor_facts,
    assign_hierarchy,
    caption_text,
    export_members,
    fit_hierarchy,
    load_corpus,
    nearest_leaves,
)


def config():
    return {"seed": 42, "target_leaf_size": 16, "min_child_size": 4,
            "max_depth": 12, "split_fit_samples": 80, "cuts": {"coarse": 2}}


def test_hierarchy_replay_coverage_and_determinism():
    x = np.random.default_rng(4).normal(size=(160, 9)).astype(np.float32)
    a = fit_hierarchy(x, config())
    b = fit_hierarchy(x, config())
    assert a[0] == b[0]
    ids, route = assign_hierarchy(x, a[0], a[3])
    for key in a[1]:
        np.testing.assert_array_equal(a[1][key], route[key])
    leaf_indices = [a[1][n["id"]] for n in a[0] if not n["children"]]
    np.testing.assert_array_equal(np.sort(np.concatenate(leaf_indices)), np.arange(len(x)))
    assert all(len(i) >= 4 for i in leaf_indices)
    assert len(set(ids)) > 2
    assert all(a[0][ancestor(int(i), a[0], 2)]["depth"] <= 2 for i in ids)


def test_identical_points_not_arbitrarily_split():
    x = np.ones((80, 4), dtype=np.float32)
    nodes, _, _, _ = fit_hierarchy(x, config())
    assert len(nodes) == 1
    assert nodes[0]["stop_reason"] == "identical_representation"


def test_test_assignment_does_not_change_training_tree():
    x = np.random.default_rng(1).normal(size=(120, 5)).astype(np.float32)
    nodes, _, centers, split = fit_hierarchy(x, config())
    frozen = json.dumps(nodes)
    split_copy = {k: v.copy() for k, v in split.items()}
    result, groups = assign_hierarchy(np.full((27, 5), 1000, dtype=np.float32), nodes, split)
    assert len(result) == 27 and len(groups[0]) == 27
    assert json.dumps(nodes) == frozen
    for k in split:
        np.testing.assert_array_equal(split_copy[k], split[k])


def test_uncertain_and_future_prose_not_used():
    c = {"scene_description_zh": "未来撞车", "attributes": [
        {"description": "斑马线", "certainty": "visible"},
        {"description": "似乎变道", "certainty": "uncertain"}],
        "retrieval_tags": ["斑马线", "变道", "未来撞车"]}
    assert "斑马线" in caption_text(c)
    assert "变道" not in caption_text(c) and "撞车" not in caption_text(c)


def test_nearest_alternatives_match_brute_force():
    rng = np.random.default_rng(10)
    x, c = rng.normal(size=(19, 5)), rng.normal(size=(8, 5))
    ids, ds = nearest_leaves(x, c, list(range(10, 18)))
    exact = np.linalg.norm(x[:, None] - c[None], axis=2)
    ix = np.argsort(exact, axis=1)[:, :3]
    np.testing.assert_array_equal(ids, ix + 10)
    np.testing.assert_allclose(ds, np.take_along_axis(exact, ix, axis=1))


def test_members_keep_uncertainty_and_admission(tmp_path):
    corpus = {"members": [{"token": "train1", "source_log": "log1", "history_continuous": False}],
              "captions": [{"split": "navtrain", "model": "codex_visual_review", "review_status": "draft"}]}
    file = tmp_path / "navtrain" / "members.jsonl"
    flagged = export_members(file, corpus, np.array([0]), [{"id": 0, "parent": -1, "depth": 0}],
                             np.array([[1, 0]]), np.array([[0, 0]]), {0: .5}, config())
    row = json.loads(file.read_text())
    assert flagged == 1 and row["out_of_train_radius"] is True
    assert row["history_continuous"] is False
    assert row["training_admission_granted"] is False
    assert row["caption_model"] == "codex_visual_review"


def test_caption_corpus_does_not_require_vector_library(tmp_path):
    root = tmp_path / "lib/cluster_caption/navtrain"
    root.mkdir(parents=True)
    original = {"log_00001_00010_token1": {"scene_name": "log_00001_00010", "solution": "FORBIDDEN_FUTURE", "ego_dynamic_state": "ax: 0.23 m/s², ay: -0.08 m/s²"}}
    (root / "autodrive-p3.original.json").write_text(json.dumps(original))
    c = {"token": "token1", "split": "navtrain", "observation_scope": "current_anchor_four_cameras", "future_images_included": False, "anchor_speed_kmh": 24, "navigation_command": "GO STRAIGHT", "model": "fixture"}
    c["status"] = "machine_draft"
    (root / "shards").mkdir()
    row = {**original["log_00001_00010_token1"], "annotation_key": "log_00001_00010_token1", "cluster_caption": c}
    (root / "shards/batch-00000.jsonl").write_text(json.dumps(row) + "\n")
    result = load_corpus(tmp_path, "navtrain", 1)
    assert result["summary"]["visual_vectors_used"] is False
    assert result["members"][0]["source_group"] == "log"
    assert result["members"][0]["features"]["anchor_ax_mps2"] == .23
    assert "FORBIDDEN_FUTURE" not in str(result)


def test_navigation_aliases_and_no_inferred_actions():
    a = anchor_facts({"navigation_command": "GO FORWARD", "anchor_speed_kmh": 5}, {})
    b = anchor_facts({"navigation_command": "GO STRAIGHT", "anchor_speed_kmh": 5}, {})
    assert a["nav_command_slot"] == b["nav_command_slot"] == 0
    assert np.isnan(a["anchor_ax_mps2"])
    assert "maneuver" not in a
