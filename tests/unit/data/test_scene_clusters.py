import numpy as np
import pytest

from driversi.data.catalog.scene_clusters import (
    assign, candidate_indices, fit, rank_pool, scene_facts, transform,
)


def sample():
    rng = np.random.default_rng(4)
    v = rng.normal(size=(80, 12)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rows = [dict(token=str(i).zfill(3), source_group=str(i % 10),
                 information_set="current_state_history", conditions=[],
                 features={"ego_speed_mps": float(i % 20), "nav_command_slot": i % 3})
            for i in range(len(v))]
    return v, rows


def test_train_only_fit_deterministic_shared_assignment():
    v, rows = sample()
    with pytest.raises(ValueError, match="navtrain"):
        fit(v, rows, split="navtest", clusters=4)
    m, x, lab, _ = fit(v, rows, split="navtrain", clusters=4, iterations=3)
    m2, _, lab2, _ = fit(v, rows, split="navtrain", clusters=4, iterations=3)
    np.testing.assert_array_equal(lab, lab2)
    np.testing.assert_array_equal(m["centers"], m2["centers"])
    frozen = {k: a.copy() for k, a in m.items()}
    test_context = transform(v[:7], rows[:7], m)
    np.testing.assert_allclose(test_context, x[:7])
    np.testing.assert_array_equal(assign(test_context, m["centers"])[0], lab[:7])
    for k in m:
        np.testing.assert_array_equal(m[k], frozen[k])


def test_all_clusters_match_exact_and_source_diversity():
    v, rows = sample()
    m, x, labels, _ = fit(v, rows, split="navtrain", clusters=4, iterations=3)
    indices, _ = candidate_indices(x[0], m["centers"], labels, 4)
    ranked = rank_pool(v, v[0], indices, rows, 20)
    reference = np.lexsort((np.arange(len(v)), -(v @ v[0])))[:20]
    assert [i for i, _ in ranked] == list(reference)
    diverse = rank_pool(v, v[0], indices, rows, 20, per_source=1)
    assert len(diverse) == len({rows[i]["source_group"] for i, _ in diverse})


def test_missing_is_distinct_and_future_rejected():
    _, rows = sample()
    a = dict(rows[0], features={})
    b = dict(rows[0], features={"ego_speed_mps": 0.0})
    assert not np.array_equal(scene_facts([a]), scene_facts([b]))
    with pytest.raises(ValueError, match="Future"):
        scene_facts([dict(a, information_set="future")])
    with pytest.raises(ValueError, match="Invalid"):
        scene_facts([dict(a, features={"ego_speed_mps": float("nan")})])


def test_fourth_native_command_is_not_unknown():
    _, rows = sample()
    a = dict(rows[0], features={"nav_command_slot": 3})
    b = dict(rows[0], features={"nav_command_slot": None})
    assert not np.array_equal(scene_facts([a]), scene_facts([b]))
