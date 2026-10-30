import json

import numpy as np
import pytest

from driversi.data.catalog.centroid_cache import build_cache, open_cache
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore


def source(tmp_path):
    path = tmp_path / "source.sqlite"
    c = Catalog(path, "navtrain")
    with c.db:
        c.db.execute("INSERT INTO scenes VALUES (?,?,?,?,?)", (
            "a", "log", "group", json.dumps({"kind": "original_navtrain"}),
            json.dumps({"history": {"information_set": "current_state_history",
                                   "continuous": True}, "features": {"ego_speed_mps": 2}})))
    v = VectorStore(c, writable=True)
    p = v.profile({"cameras": ["F"], "dimension": 2})
    v.put("a", p, [{"id": f"h{i}_F"} for i in range(4)], np.ones((4, 2)), {})
    c.close()
    return path, p


def test_cache_real_sqlite_restart_and_corruption(tmp_path):
    path, p = source(tmp_path)
    out = tmp_path / "cache"
    first = build_cache(path, "navtrain", p, out)
    assert first["count"] == 1 and not first["reused"]
    assert build_cache(path, "navtrain", p, out)["reused"]
    _, rows, matrix = open_cache(out, verify=True)
    assert rows[0]["token"] == "a"
    np.testing.assert_allclose(matrix, [[2**-0.5, 2**-0.5]])
    (out / "members.json").write_text("[]")
    with pytest.raises(ValueError, match="content changed"):
        open_cache(out, verify=True)


def test_wrong_split_and_changed_source_rejected(tmp_path):
    path, p = source(tmp_path)
    with pytest.raises(ValueError, match="split"):
        build_cache(path, "navtest", p, tmp_path / "bad")
    build_cache(path, "navtrain", p, tmp_path / "cache")
    c = Catalog(path, "navtrain")
    with c.db:
        c.db.execute("INSERT INTO metadata VALUES ('revision','2')")
    c.close()
    with pytest.raises(ValueError, match="Source changed"):
        build_cache(path, "navtrain", p, tmp_path / "cache")
