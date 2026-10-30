import copy
import json

import numpy as np
import pytest

from driversi.rendering.diagnostics.numeric_admission import validate_diversity
from driversi.rendering.diagnostics.worldengine_admission import (
    photometric_rejection,
    visual_policy,
)
from driversi.rsi.decisions.trajectory_diversity import compare
from driversi_agents.augustus.branch_supervision import (
    geometry_label_contract,
    require_label_contract,
)


def test_numeric_quality_is_strict_and_separate_from_legacy():
    name = "experimental_numeric25_v1"
    assert photometric_rejection(25.0, name)
    assert photometric_rejection(25.0001, name) is None
    assert not visual_policy(name)["require_pixel_review"]
    assert visual_policy("experimental_egoview25_v1")["require_pixel_review"]
    with pytest.raises(ValueError):
        photometric_rejection(float("nan"), name)


def test_geometry_label_is_not_a_visual_approval(tmp_path):
    p = tmp_path / "language-draft.json"
    d = dict(
        objects=[dict(actor_id="measured")],
        unresolved=[dict(actor_id="gone", reason="future_identity_unavailable")],
        branch_report_sha256="branch",
    )
    p.write_text(json.dumps(d))
    c = geometry_label_contract(tmp_path, tmp_path / "labels")
    q = tmp_path / "labels/label-review.json"
    assert c["visibility_reviewed"] is False and "approved" not in c
    profile = dict(visual_quality_profile="experimental_numeric25_v1")
    assert require_label_contract(p, q, profile) == d
    with pytest.raises(ValueError):
        require_label_contract(p, q, dict(visual_quality_profile="experimental_egoview25_v1"))
    c["actor_ids"].append("invented")
    q.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        require_label_contract(p, q, profile)


def test_ade_is_recomputed_and_foreign_history_rejected(tmp_path, monkeypatch):
    import driversi.rsi.decisions.trajectory_diversity as mod

    desc = dict(
        candidate_id="family-v001",
        comparison_key={"token": "t"},
        xy=np.zeros((71, 2)).tolist(),
        files={},
    )
    monkeypatch.setattr(mod, "describe", lambda *args: copy.deepcopy(desc))
    gp = tmp_path / "generation.json"
    gp.write_text(json.dumps(dict(files={})))
    path = tmp_path / "teacher.npz"
    states = np.zeros((71, 1, 13))
    states[:, :, 0] = 1
    np.savez(path, states=states, actor_ids=np.array(["ego"]))
    from driversi.agents.backends.native import digest

    prior = dict(
        desc,
        candidate_id="family-v000",
        xy=states[:, 0, :2].tolist(),
        files={str(path): digest(path)},
    )
    evidence = dict(description=desc, retained=[prior], decision=compare(desc, [prior]))
    dp = tmp_path / "diversity.json"
    dp.write_text(json.dumps(evidence))
    validate_diversity(gp, dp)
    evidence["decision"]["comparisons"][0]["ade_7s_m"] = 99
    dp.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="ADE decision"):
        validate_diversity(gp, dp)
    evidence.pop("retained")
    dp.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="Prior trajectory"):
        validate_diversity(gp, dp)


def test_ego_camera_motion_cannot_be_swapped():
    from types import SimpleNamespace

    from driversi.rendering.diagnostics.numeric_admission import validate_ego_motion
    from driversi.rendering.gaussian.worldengine_replay import physical_delta

    scene = SimpleNamespace(actor_ids=("ego",), poses=np.zeros((1, 1, 4)))
    states = scene.poses.copy()
    states[0, 0, 0] = 2.0
    motion = physical_delta(states[0, 0], scene.poses[0, 0])
    frame = dict(tick=0, motions={"ego": motion})
    validate_ego_motion(frame, scene, states)
    frame["motions"]["ego"] = np.asarray(motion) + 1
    with pytest.raises(ValueError, match="ego execution"):
        validate_ego_motion(frame, scene, states)


def test_relaxed_profile_boundary_and_legacy_stays_strict():
    from driversi.rendering.diagnostics.numeric_admission import numeric_settings

    profile = "experimental_numeric24_ade03_v1"
    assert photometric_rejection(24.0, profile) is None
    assert photometric_rejection(23.999, profile)
    assert photometric_rejection(25.0, "experimental_numeric25_v1")
    assert numeric_settings(profile)["minimum_ade_m"] == 0.3
    assert not visual_policy(profile)["require_pixel_review"]
    assert not visual_policy(profile)["require_terrain_support"]


def test_relaxed_ade_replay_does_not_admit_under_legacy_profile(tmp_path, monkeypatch):
    import driversi.rsi.decisions.trajectory_diversity as mod
    from driversi.agents.backends.native import digest

    desc = dict(
        candidate_id="family-v001",
        comparison_key={"token": "t"},
        xy=np.zeros((71, 2)).tolist(),
        files={},
    )
    monkeypatch.setattr(mod, "describe", lambda *args: copy.deepcopy(desc))
    gp = tmp_path / "generation.json"
    gp.write_text(json.dumps(dict(files={})))
    path = tmp_path / "teacher.npz"
    states = np.zeros((71, 1, 13))
    states[:, :, 0] = 0.35
    np.savez(path, states=states, actor_ids=np.array(["ego"]))
    prior = dict(
        desc,
        candidate_id="family-v000",
        xy=states[:, 0, :2].tolist(),
        files={str(path): digest(path)},
    )
    evidence = dict(
        description=desc, retained=[prior], decision=compare(desc, [prior], minimum_ade_m=0.3)
    )
    dp = tmp_path / "diversity.json"
    dp.write_text(json.dumps(evidence))
    validate_diversity(gp, dp, profile="experimental_numeric24_ade03_v1")
    with pytest.raises(ValueError, match="declared profile"):
        validate_diversity(gp, dp)
    evidence["decision"]["comparisons"][0]["ade_future_5s_m"] = 0.4
    dp.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="ADE decision"):
        validate_diversity(gp, dp, profile="experimental_numeric24_ade03_v1")
