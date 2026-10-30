import copy

import numpy as np
import pytest

from driversi.rsi.correction.history_perturbation import (
    FAMILY,
    arrival_search_compatible,
    perturbation_grid,
    quintic_reference,
    validate_perturbation,
)
from driversi.rsi.experiments.reactive_cohort import expand_plan, validate_budget


def test_quintic_joins_position_velocity_acceleration_and_extrapolates():
    start = np.array([[100, 200], [4, 1], [0.4, -0.2]])
    end = np.array([[108, 204], [3, 2], [0, 0]])
    p, v, a = quintic_reference(start, end, [0, 2, 2.1, 4])
    np.testing.assert_allclose(np.array([p[0], v[0], a[0]]), start, atol=1e-12)
    np.testing.assert_allclose(np.array([p[1], v[1], a[1]]), end, atol=1e-12)
    np.testing.assert_allclose(p[2:], end[0] + np.array([0.1, 2])[:, None] * end[1], atol=1e-12)
    np.testing.assert_allclose(v[2:], [end[1], end[1]], atol=1e-12)
    np.testing.assert_allclose(a[2:], 0, atol=1e-12)
    eps = 1e-4
    pp, vv, aa = quintic_reference(start, end, [0.7 - eps, 0.7, 0.7 + eps])
    np.testing.assert_allclose((pp[2] - pp[0]) / (2 * eps), vv[1], atol=1e-7)
    np.testing.assert_allclose((vv[2] - vv[0]) / (2 * eps), aa[1], atol=1e-7)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 6, -6])
def test_invalid_proposal_rejected(bad):
    with pytest.raises(ValueError):
        validate_perturbation(dict(longitudinal_m=bad, lateral_m=0, speed_delta_mps=0))


def test_grid_is_bounded_unique_and_family_preserves_provenance():
    grid = perturbation_grid()
    assert len(grid) == 24 and len({tuple(x.values()) for x in grid}) == 24
    for x in grid:
        validate_perturbation(x)
    policy = dict(
        method=FAMILY,
        target_per_anchor=10,
        attempts_per_anchor=24,
        minimum_ade_m=0.5,
        seed_start=42,
    )
    config = dict(
        trajectory_family=policy,
        max_samples=10,
        max_candidates=1,
        max_rendered_branches=24,
        generation_skill="perturbed_history_tts",
        visual_quality_profile="experimental_numeric27_ade05_v1",
    )
    validate_budget(config)
    original = dict(assignments=[dict(id="a", token="train", skill="perturbed_history_tts")])
    before = copy.deepcopy(original)
    plan = expand_plan(original, policy, admission_profile="pdms_gt95_v1")
    assert original == before
    assert [x["history_perturbation"] for x in plan["assignments"]] == grid
    assert all(x["trajectory_family_version"] == FAMILY for x in plan["assignments"])
    assert all("distinct" not in x for x in plan["assignments"])
    # Execution reuse is authenticated by exact source/control/plan hashes in produce.
    validate_budget(dict(config, reuse_completed_generations={"same": "sealed-later"}))
    with pytest.raises(ValueError):
        validate_budget(dict(config, trajectory_family=dict(policy, attempts_per_anchor=25)))


def test_native_search_and_history_teacher_never_overlap():
    search = dict(start_tick=20, stop_tick=70)
    for tick in range(20):
        assert arrival_search_compatible(tick, search)
    for tick in [-1, 20, 69, 70]:
        assert not arrival_search_compatible(tick, search)
    assert not arrival_search_compatible(0, None)
    assert not arrival_search_compatible(0, dict(start_tick=0, stop_tick=70))


def test_numeric27_boundary_and_original_profiles_preserved():
    from driversi.rendering.diagnostics.numeric_admission import numeric_settings
    from driversi.rendering.diagnostics.worldengine_admission import (
        photometric_rejection,
        visual_policy,
    )

    profile = "experimental_numeric27_ade05_v1"
    assert numeric_settings(profile) == dict(psnr_db=27.0, strict_psnr=False, minimum_ade_m=0.5)
    assert photometric_rejection(27, profile) is None
    assert photometric_rejection(26.999, profile)
    assert numeric_settings("experimental_numeric24_ade03_v1")["minimum_ade_m"] == 0.3
    assert not visual_policy(profile)["experimental_low_fidelity"]
    assert not visual_policy(profile)["require_pixel_review"]
    assert not visual_policy(profile)["require_terrain_support"]


def test_historical_proposals_cannot_silently_be_used_as_native_actions():
    from pathlib import Path

    from driversi.rsi.plugins.navsim_teacher import prepare

    cfg = dict(
        scene="missing",
        road_graph="missing",
        python="missing",
        checkpoint="missing",
        maps_root="missing",
        arrival_action=[0, 0],
        search_candidates=0,
        device="cpu",
        max_wall_seconds=300,
        arrival_backend="source_history_perturbation_v1",
        history_perturbation=perturbation_grid()[0],
    )
    with pytest.raises(ValueError, match="future TTS"):
        prepare(cfg, Path("/"), None)
    with pytest.raises(ValueError, match="different backend"):
        prepare(dict(cfg, arrival_backend="driverl_native"), Path("/"), None)


def test_production_passes_campaign_generation_skill(monkeypatch, tmp_path):
    import json

    from driversi.rsi.experiments import reactive_autoloop as mod
    from driversi.rsi.experiments import reactive_selection

    class ReachedSelection(Exception):
        pass

    mining = tmp_path / "mining"
    mining.mkdir()
    (mining / "summary.json").write_text("{}")
    config = dict(
        output=str(tmp_path / "run"),
        max_wall_seconds=60,
        selection_mode="full_training_hybrid",
        mining_root=str(mining),
        baseline_config="parent-config-without-generation-skill",
        coverage="coverage",
        max_candidates=2,
        generation_skill="perturbed_history_tts",
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(mod, "wait_baseline", lambda *a: None)
    monkeypatch.setattr(mod, "status", lambda *a, **k: None)

    def selected(*args, **kwargs):
        assert kwargs["generation_skill"] == "perturbed_history_tts"
        raise ReachedSelection

    monkeypatch.setattr(reactive_selection, "select_full_training", selected)
    with pytest.raises(ReachedSelection):
        mod.production(path)


def test_per_source_ten_cannot_be_replaced_by_global_twenty():
    from driversi.rsi.experiments.reactive_cohort import family_training_gate

    rows = [
        dict(
            anchor_assignment_id=anchor,
            status="training_sample_ready",
            trainable=True,
            rendered=True,
            sample_report=f"{anchor}-{i}",
            dataset_reference={"package_id": f"{anchor}-{i}"},
        )
        for anchor, count in [("a", 9), ("b", 11)]
        for i in range(count)
    ]
    result = family_training_gate(dict(rows=rows, qualified=20), ["a", "b"], 10)
    assert not result["passed"] and result["shortages"] == {"a": 1, "b": 0}
    rows.append(dict(rows[0], sample_report="a-9", dataset_reference={"package_id": "a-9"}))
    assert family_training_gate(dict(rows=rows, qualified=21), ["a", "b"], 10)["passed"]
    rows[-1]["rendered"] = False
    with pytest.raises(ValueError):
        family_training_gate(dict(rows=rows, qualified=21), ["a", "b"], 10)


def test_explicit_grid_is_bounded_and_no_command_duplicates():
    from driversi.rsi.experiments.reactive_cohort import history_grid

    grid = perturbation_grid()
    grid.append(dict(longitudinal_m=4, lateral_m=0.5, speed_delta_mps=0))
    assert len(history_grid(dict(proposal_grid=grid))) == 25
    with pytest.raises(ValueError):
        history_grid(dict(proposal_grid=grid + [grid[0]]))
    grid[-1]["longitudinal_m"] = 6
    with pytest.raises(ValueError):
        history_grid(dict(proposal_grid=grid))


def test_duplicate_package_under_two_export_paths_cannot_fill_quota():
    from driversi.rsi.experiments.reactive_cohort import family_training_gate

    rows = [
        dict(
            anchor_assignment_id="a",
            status="training_sample_ready",
            trainable=True,
            rendered=True,
            sample_report=str(i),
            dataset_reference={"package_id": "same"},
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="unique outcomes"):
        family_training_gate(dict(rows=rows, qualified=10), ["a"], 10)


def test_explicit_proposal_seeds_preserve_exact_reuse_and_default():
    from driversi.rsi.experiments.reactive_cohort import proposal_seeds

    policy = dict(
        method=FAMILY,
        target_per_anchor=10,
        attempts_per_anchor=10,
        minimum_ade_m=0.5,
        seed_start=42,
    )
    assert proposal_seeds(policy) == list(range(42, 52))
    seeds = [44, 45, 53, 61, 67, 86, 102, 106, 107, 108]
    explicit = dict(policy, proposal_seeds=seeds)
    original = dict(assignments=[dict(id="a", token="train", skill="perturbed_history_tts")])
    rows = expand_plan(original, explicit)["assignments"]
    assert [r["sampling_seed"] for r in rows] == seeds
    result = proposal_seeds(explicit)
    result[0] = 123
    assert explicit["proposal_seeds"][0] == 44
    for bad in [
        seeds[:-1],
        [44] * 10,
        [True] + seeds[1:],
        [-1] + seeds[1:],
        [2**31] + seeds[1:],
        tuple(seeds),
    ]:
        with pytest.raises(ValueError):
            proposal_seeds(dict(policy, proposal_seeds=bad))
