import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from driversi.rsi.experiments import navval_retrieval as mod


def catalog(split, rows):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        "CREATE TABLE scenes(token TEXT PRIMARY KEY,source_group TEXT,source_log TEXT); CREATE TABLE vector_profiles(id TEXT PRIMARY KEY,config TEXT); CREATE TABLE vectors(profile TEXT,token TEXT,status TEXT,centroid BLOB,image_vectors BLOB,views INTEGER,dimension INTEGER,input_id TEXT);"
    )
    db.execute(
        "INSERT INTO vector_profiles VALUES (?,?)",
        ("p", json.dumps({"dimension": 2, "cameras": ["F"]})),
    )
    for token, group, vec in rows:
        vec = np.asarray(vec, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        db.execute("INSERT INTO scenes VALUES (?,?,?)", (token, group, group))
        db.execute(
            "INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?)",
            (
                "p",
                token,
                "ready",
                vec.astype("<f4").tobytes(),
                vec.astype("<f2").tobytes(),
                1,
                2,
                token,
            ),
        )
    return SimpleNamespace(split=split, db=db)


def test_navval_only_despite_more_similar_nonmember(monkeypatch):
    q = catalog("navtest", [("q", "test", [1, 0])])
    t = catalog("navtrain", [("outside", "x", [1, 0]), ("inside", "v", [0.9, 0.1])])
    monkeypatch.setattr(mod, "navval_members", lambda _: ({"inside": "v"}, {"split": "hash"}))
    out = mod.retrieve(q, t, "p", ["q"], limit=1)
    assert out["rows"][0]["candidates"][0]["token"] == "inside"
    assert out["publishable"] is False
    assert out["claim"] == "visual_similarity_not_confirmed_failure"


def test_reverse_split_rejected():
    with pytest.raises(ValueError, match="Explicit test"):
        mod.retrieve(
            SimpleNamespace(split="navtrain"), SimpleNamespace(split="navtest"), "p", ["q"]
        )


def test_profile_mismatch_rejected(monkeypatch):
    q = catalog("navtest", [("q", "t", [1, 0])])
    t = catalog("navtrain", [("x", "v", [1, 0])])
    t.db.execute(
        "UPDATE vector_profiles SET config=?", (json.dumps({"dimension": 2, "cameras": ["other"]}),)
    )
    with pytest.raises(ValueError, match="profiles differ"):
        mod.retrieve(q, t, "p", ["q"])
