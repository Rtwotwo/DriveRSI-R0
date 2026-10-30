from pathlib import Path

from navsim.common.dataloader import MetricCacheLoader, SceneLoader


def test_scene_loader_reuses_ordered_tokens_and_constant_time_membership():
    loader = SceneLoader.__new__(SceneLoader)
    loader.scene_frames_dicts = {
        "token-b": [{"log_name": "log-b"}],
        "token-a": [{"log_name": "log-a"}],
    }
    loader._tokens = list(loader.scene_frames_dicts)

    assert loader.tokens is loader.tokens
    assert loader.tokens == ["token-b", "token-a"]
    assert len(loader) == 2
    assert loader[1] == "token-a"


def test_metric_cache_loader_relocates_stale_prefix_without_per_row_stat(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "train_metric_cache"
    metadata = cache_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "index.csv").write_text(
        "file_name\n"
        "/old/machine/train_metric_cache/log-a/unknown/token-a/metric_cache.pkl\n"
        "/old/machine/train_metric_cache/log-b/unknown/token-b/metric_cache.pkl\n",
        encoding="utf-8",
    )

    def fail_exists(_path: Path) -> bool:
        raise AssertionError("MetricCacheLoader must not stat every indexed sample")

    monkeypatch.setattr(Path, "exists", fail_exists)
    loader = MetricCacheLoader(cache_root)

    assert loader.metric_cache_paths == {
        "token-a": str(cache_root / "log-a/unknown/token-a/metric_cache.pkl"),
        "token-b": str(cache_root / "log-b/unknown/token-b/metric_cache.pkl"),
    }


def test_metric_cache_loader_rejects_conflicting_duplicate_tokens(tmp_path):
    cache_root = tmp_path / "metric"
    metadata = cache_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "index.csv").write_text(
        "file_name\n"
        "/old/metric/log-a/unknown/shared/metric_cache.pkl\n"
        "/old/metric/log-b/unknown/shared/metric_cache.pkl\n",
        encoding="utf-8",
    )

    try:
        MetricCacheLoader(cache_root)
    except ValueError as exc:
        assert "Duplicate metric token shared" in str(exc)
    else:
        raise AssertionError("Conflicting duplicate metric tokens must fail")
