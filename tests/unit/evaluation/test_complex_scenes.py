import numpy as np
import pytest

from driversi.evaluation.benchmarks.complex_scenes import complexity, index, select_diverse


def row(name, group, city, density, heading):
    return {
        "id": name,
        "source_group": group,
        "city": city,
        "status": "completed",
        "complexity": {
            "peak_vehicles_40m": density,
            "peak_vehicles_15m": density,
            "peak_vulnerable_30m": 0,
            "peak_transverse_vehicles_30m": 1,
            "ego_heading_span_degrees": heading,
        },
    }


def test_different_logs_density_and_turn_not_path_order():
    rows = [
        row("a", "one", "Boston", 10, 1),
        row("b", "one", "Boston", 1, 90),
        row("c", "two", "Boston", 2, 60),
        row("d", "three", "Singapore", 3, 10),
    ]
    assert [x["id"] for x in select_diverse(rows)] == ["a", "c", "d"]
    assert select_diverse(rows) == select_diverse(list(reversed(rows)))


def test_selection_rejects_duplicates_and_invalid_counts():
    r = row("a", "one", "Boston", 10, 1)
    with pytest.raises(ValueError):
        select_diverse([r, r])
    with pytest.raises(ValueError):
        select_diverse([r], True)


def test_source_proxies_include_parked_and_no_false_conflict_label():
    frame = {
        "ego2global_rotation": [1, 0, 0, 0],
        "anns": {
            "gt_names": ["vehicle", "vehicle", "pedestrian"],
            "gt_boxes": np.array(
                [[10, 0, 0, 4, 2, 2, 0], [50, 0, 0, 4, 2, 2, 1.57], [3, 0, 0, 1, 1, 2, 0]]
            ),
        },
    }
    c = complexity([frame, frame])
    assert c["peak_vehicles_40m"] == 1 and c["peak_vulnerable_30m"] == 1
    assert c["peak_transverse_vehicles_30m"] == 0
    assert c["ego_heading_span_degrees"] == 0


def test_page_distinguishes_source_simulation_and_missing_gs(tmp_path):
    r = row("case", "one", "Boston", 10, 30)
    r["reason"] = "bad <script>alert(1)</script>"
    index(tmp_path, {"cases": [r], "gs": []})
    page = (tmp_path / "index.html").read_text()
    assert "<img " not in page  # no thumbnail without an actual image
    assert "尚未发布新渲染" in page and "trainable=false" in page
    assert "&lt;script&gt;" in page and "<script>alert(1)" not in page


def test_source_partition_cannot_escape(tmp_path):
    from driversi.data.scenes.source_gallery import _scene

    with pytest.raises(ValueError, match="partition"):
        _scene(
            {"id": "case", "split": "../test", "log": "any.pkl"}, tmp_path, tmp_path, trusted=True
        )


def test_missing_or_unsupported_progress_is_explicit(tmp_path):
    r = row("case", "one", "Boston", 2, 0)
    index(tmp_path, {"cases": [r]})
    assert "沿路线进度 —" in (tmp_path / "index.html").read_text()
    r["metrics"] = {"route_progress_m": 9.59, "route_progress_supported": False}
    index(tmp_path, {"cases": [r]})
    assert "9.6m（存在路线支持不足" in (tmp_path / "index.html").read_text()


def test_cli_incomplete_run_is_nonzero(monkeypatch, tmp_path):
    import sys

    import driversi.evaluation.benchmarks.complex_scenes as module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "complex-scenes",
            "--suite",
            str(tmp_path / "suite.json"),
            "--dataset",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setattr(
        module, "build", lambda *a, **kw: {"status": "partial_failure", "cases": []}
    )
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
