import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from driversi.data.datasets.tto_audit import (
    ROW_DTYPE,
    category,
    distinct_qualified,
    sha256,
    verified_rows,
    zero_offset_heading,
)
from driversi.workbench.server import make_handler
from driversi.workbench.tto_inspector import TTOInspector, contained, local_path_to_map, navsim_yaw


def published(tmp_path):
    p = tmp_path / "train" / "0123456789abcdef"
    p.mkdir(parents=True)
    rows = np.zeros(3, dtype=ROW_DTYPE)
    rows["labels"] = 1
    np.save(p / "candidates.npy", rows)
    (p / "state.json").write_text(json.dumps({"count": 3}))
    marker = {"format": "all_scored_candidates_v1", "passed": True, "token": p.name,
              "split": "train", "candidates": 3, "data_digest": sha256(p / "candidates.npy")}
    (p / "ALL_SCORES.json").write_text(json.dumps(marker))
    return p, rows, marker


def test_committed_table_rejects_mutation_or_partial_commit(tmp_path):
    p, rows, marker = published(tmp_path)
    assert len(verified_rows(p)[0]) == 3
    (p / "state.json").write_text(json.dumps({"count": 2}))
    with pytest.raises(ValueError, match="committed"):
        verified_rows(p)
    (p / "state.json").write_text(json.dumps({"count": 3}))
    rows["trajectories"][0, 1, 0] = 1
    np.save(p / "candidates.npy", rows)
    with pytest.raises(ValueError, match="publication"):
        verified_rows(p)
    marker["data_digest"] = sha256(p / "candidates.npy")
    (p / "ALL_SCORES.json").write_text(json.dumps(marker))
    assert len(verified_rows(p)[0]) == 3


def test_official_formula_and_exact_gt_exclusion(tmp_path):
    p, rows, marker = published(tmp_path)
    rows["trajectories"][1, 3, 0] = 2
    rows["trajectories"][2] = rows["trajectories"][1]
    assert distinct_qualified(rows, np.zeros((8, 3))) == 1
    rows["labels"][0, 0] = 0
    np.save(p / "candidates.npy", rows)
    marker["data_digest"] = sha256(p / "candidates.npy")
    (p / "ALL_SCORES.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="components"):
        verified_rows(p)


@pytest.mark.parametrize("score,count,expected", [(0,0,"all_zero"), (.79,0,"below_80"),
    (.8,0,"80_to_90"), (.9,0,"90_to_95"), (.95,1,"partial_high_score")])
def test_category_boundaries(score, count, expected):
    assert category(score, count) == expected


def test_navsim_yaw_with_pitch_and_roll_matches_official_known_pose():
    q = [0.95800931, 0.00918304, -0.00180126, -0.28658435]
    assert navsim_yaw(q) == pytest.approx(-0.58127349, abs=1e-8)
    assert np.allclose(local_path_to_map([[2, 0, 0]], np.pi/2), [[0, 2, np.pi/2]], atol=1e-12)
    with pytest.raises(ValueError):
        navsim_yaw([0, 0, 0, 0])


def test_zero_offset_does_not_guarantee_original_heading():
    gt = np.zeros((8, 3))
    gt[:, 0] = np.arange(1, 9)
    gt[:, 1] = np.arange(1, 9)*.1
    assert np.max(np.abs(zero_offset_heading(gt)-gt[:, 2])) > .09


def test_paths_and_explicit_pickle_trust(tmp_path):
    with pytest.raises(ValueError):
        contained(tmp_path, "../secret")
    with pytest.raises(ValueError):
        contained(tmp_path, "/etc/passwd")
    outside = tmp_path.parent / "outside"
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        contained(tmp_path, "link/image.jpg")
    with pytest.raises(ValueError, match="trusted"):
        TTOInspector(tmp_path / "missing", tmp_path)


def test_tto_http_queries_are_typed_and_read_only():
    class Fixture:
        def index(self, protected):
            return {"scenes": [], "read_only": True}
        def scene(self, token, protected):
            if token != "0123456789abcdef":
                raise ValueError("Unknown")
            return {"token": token}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(None, tto=Fixture()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urlopen(base+"/api/health"))["tto_available"]
        assert json.load(urlopen(base+"/api/tto/catalog"))["read_only"]
        for url in ["/api/tto/scene?token=../etc/passwd", "/api/tto/catalog?file=/etc/passwd",
                    "/api/tto/scene?token=a&token=b", "/api/tto/scene?token=a&row=1"]:
            with pytest.raises(HTTPError) as error:
                urlopen(base+url)
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_snapshot_csv_percent_score_is_not_a_bank_change(tmp_path):
    from driversi.data.datasets.tto_audit import COMPONENTS, audit_scene
    bank = tmp_path / "bank"
    p, rows, marker = published(bank / "scenes")
    rows["trajectories"][:, 3, 0] = 2
    np.save(p / "candidates.npy", rows)
    marker["data_digest"] = sha256(p / "candidates.npy")
    (p / "ALL_SCORES.json").write_text(json.dumps(marker))
    gt = np.zeros((8, 3)).tolist()
    identity = {"token": p.name, "split": "train", "log_name": "trusted-log", "expert_gt": gt,
                "component_names": list(COMPONENTS), "ego_pose": [0,0,0], "timestamp_us": 1000}
    (p / "contract.json").write_text(json.dumps({"identity": identity, "protocol": {"constructor": {}}}))
    (p / "expert.json").write_text(json.dumps({"trajectory": gt, "labels": [1]*7}))
    result = audit_scene(bank, {"token": p.name, "split": "train", "log_name": "trusted-log",
                                "qualified_unique": "1", "best_pdms": "100.0"})
    assert result["best_pdms"] == 1 and result["qualified_unique"] == 1
    assert result["snapshot_changed"] is False


def test_tto_reserved_sources_remain_hidden_after_cache(tmp_path):
    p = tmp_path / "audit.json"
    row = {"token": "0123456789abcdef", "split": "train", "log_name": "reserved", "category": "all_zero",
           "queries": 3, "qualified_unique": 0, "best_pdms": 0, "expert_labels": [1]*7,
           "gt_high_but_candidates_low": True, "component_zero_counts": {}, "best_labels": [0]*7}
    p.write_text(json.dumps({"kind": "tto_shortfall_audit", "schema_version": 1, "scenes": [row],
                            "bank_root": str(tmp_path), "components": [], "summary": {}, "generated_at": "now"}))
    inspector = TTOInspector(p, tmp_path, trusted=True)
    inspector.cache[row["token"]] = ({"secret": True}, None, None)
    assert inspector.index(["reserved"])["scenes"] == []
    with pytest.raises(ValueError, match="reserved"):
        inspector.scene(row["token"], ["reserved"])
