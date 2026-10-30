import json

import pytest

from driversi.visualization.clustering import public_clusters


def setup(tmp_path, split="navtrain"):
    out = tmp_path / "exp/test"
    out.mkdir(parents=True)
    summary = {"status": "complete", "fit_split": "navtrain", "members": {"navtrain": 5, "navtest": 2}, "leaf_clusters": 1, "level_clusters": {}, "leaf_size": {"min": 5, "max": 5, "median": 5}, "out_of_train_radius": {"navtrain": 0, "navtest": 1}, "single_source_leaves": 0, "elapsed_seconds": 1}
    node = {"id": 0, "parent": -1, "depth": 0, "children": [], "train_count": 5, "test_count": 2, "source_groups": 2, "mean_distance": .5, "radius_p99": 1, "speed_mean_kmh": 10, "speed_range_kmh": [5, 15], "xy": [0, 0], "label": "路口", "tags": [{"text": "行人", "count": 3}], "secret_test_tokens": ["NEVER_EXPORT"], "representatives": [{"split": split, "token": "train-token", "description": "前方车辆", "speed_kmh": 10, "navigation_command": "TURN LEFT", "tags": ["车辆"], "private_receipt": "/data/private"}]}
    (out / "summary.json").write_text(json.dumps(summary))
    (out / "clusters.json").write_text(json.dumps({"nodes": [node], "cuts": {"coarse": 2}, "test_captions": ["PRIVATE"]}))
    return {"summary": "exp/test/summary.json", "clusters": "exp/test/clusters.json"}


def test_projection_allowlist_excludes_private_fields(tmp_path):
    public = public_clusters(setup(tmp_path), tmp_path)
    text = json.dumps(public)
    assert "NEVER_EXPORT" not in text and "/data/" not in text and "PRIVATE" not in text
    assert public["nodes"][0]["test_count"] == 2
    assert public["nodes"][0]["representatives"][0]["split"] == "navtrain"


def test_test_representative_fails_publication(tmp_path):
    with pytest.raises(ValueError, match="Test scene"):
        public_clusters(setup(tmp_path, "navtest"), tmp_path)
