import pytest
from driversi.rendering.diagnostics.worldengine_admission import (
    visual_policy,
    photometric_rejection,
)
from driversi.rendering.diagnostics.gs_sequence_critic import CHECKS, frame_passes
from driversi.rendering.diagnostics.scene_quality import frame_allowed
from driversi.rsi.experiments.readiness import analyze_render

PROFILE = "experimental_egoview25_v1"


def test_scope_explicit_and_strict_defaults_unchanged():
    assert visual_policy()["require_terrain_support"]
    assert visual_policy()["require_complete_object_binding"]
    assert not visual_policy(PROFILE)["require_terrain_support"]
    assert not visual_policy(PROFILE)["require_complete_object_binding"]
    assert photometric_rejection(25.0, PROFILE)
    assert not photometric_rejection(25.00001, PROFILE)


def test_unknown_object_metadata_is_diagnostic_but_bad_actual_pixels_still_reject():
    value = dict(
        checks={k: dict(status="pass") for k in CHECKS},
        issues=[],
        actors=[dict(appearance_status="unknown", alignment_status="unknown")],
    )
    assert not frame_passes(value)
    assert frame_passes(value, profile=PROFILE)
    for key in CHECKS:
        value["checks"][key]["status"] = "fail"
        assert not frame_passes(value, profile=PROFILE)
        value["checks"][key]["status"] = "pass"
    value["issues"] = ["on_road_ghost_vehicle"]
    assert not frame_passes(value, profile=PROFILE)
    r = dict(verdict="experimental_pass", issues=[], **dict.fromkeys(CHECKS, True))
    assert frame_allowed(
        r, dict(total_fraction=0.0, largest_component_fraction=0.0), profile=PROFILE
    )
    assert not frame_allowed(
        r, dict(total_fraction=0.06, largest_component_fraction=0.03), profile=PROFILE
    )


def test_readiness_terrain_and_binding_are_diagnostic_only_in_new_profile():
    cams = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
    ticks = list(range(0, 71, 5))
    report = dict(
        source_log="log",
        asset_id="asset",
        terrain_fully_supported=False,
        history_ticks=[5, 10, 15, 20],
        future_ticks=ticks[5:],
        frames=[
            dict(
                tick=t,
                physical_missing_appearance=["actor"],
                cameras={c: dict(terrain_fully_supported=False) for c in cams},
            )
            for t in ticks
        ],
    )
    receipt = dict(render_report=report, native_future_points=10)
    source = dict(
        source_mode="canonical_source_pose_without_motion_v1",
        frames=[dict(tick=t, camera=c, measured_psnr_db=26.97) for t in ticks for c in cams],
    )
    old = analyze_render(receipt, source, "experimental_scene25_v1")
    new = analyze_render(receipt, source, PROFILE)
    assert len(old["blockers"]) == 2
    assert new["blockers"] == [] and len(new["diagnostic_warnings"]) == 2
    assert not new["trainable"]
    source["frames"].pop()
    with pytest.raises(ValueError, match="missing"):
        analyze_render(receipt, source, PROFILE)


def test_review_scope_cannot_be_upgraded_or_silently_reinterpreted():
    from driversi.rendering.diagnostics.worldengine_admission import validate_review_scope

    validate_review_scope({}, "source27_v1")
    validate_review_scope({"visual_quality_profile": PROFILE}, PROFILE)
    for declaration, target in [
        ({}, PROFILE),
        ({"visual_quality_profile": PROFILE}, "source27_v1"),
        ({"visual_quality_profile": "experimental_scene25_v1"}, PROFILE),
    ]:
        with pytest.raises(ValueError, match="scope"):
            validate_review_scope(declaration, target)


def test_ego_global_response_is_complete_without_per_object_identity_claims():
    from driversi.rendering.diagnostics.gs_sequence_critic import checked_response

    facts = dict(target_id="frame", image_sha256="hash", actor_ids=[])
    response = dict(
        target_id="frame",
        image_sha256="hash",
        actors=[],
        issues=[],
        void_polygons=[],
        checks={k: dict(status="pass", evidence="visible pixels") for k in CHECKS},
    )
    assert checked_response(response, facts) == response
    with pytest.raises(ValueError, match="actor review"):
        checked_response(response, {**facts, "actor_ids": ["known_actor"]})
    response["checks"].pop("occlusion_consistent")
    with pytest.raises(ValueError, match="pixel checks"):
        checked_response(response, facts)


def test_lossless_actor_nesting_repair_preserves_every_requested_check():
    from driversi.rendering.diagnostics.gs_sequence_critic import checked_response

    facts = dict(target_id="frame", image_sha256="hash", actor_ids=[])
    response = dict(
        target_id="frame",
        image_sha256="hash",
        issues=[],
        void_polygons=[],
        checks={k: dict(status="pass", evidence="pixels") for k in CHECKS},
    )
    response["checks"]["actors"] = []
    fixed = checked_response(response, facts)
    assert "actors" in response["checks"]  # Original response is preserved.
    assert fixed["actors"] == [] and set(fixed["checks"]) == set(CHECKS)
    response["checks"].pop("road_geometry_consistent")
    with pytest.raises(ValueError, match="pixel checks"):
        checked_response(response, facts)


def test_review_preflight_counts_existing_charged_attempts(tmp_path, monkeypatch):
    import sqlite3
    from driversi.data.catalog import bailian_caption
    from driversi.rendering.diagnostics.gs_sequence_critic import review_capacity

    monkeypatch.setattr(bailian_caption, "private_config", lambda _: dict(max_requests=62))
    path = tmp_path / "budget.sqlite"
    assert review_capacity("private", path)["sufficient"]
    with sqlite3.connect(path) as db:
        db.execute("create table reservations(signature text primary key, started real)")
        db.execute("insert into reservations values('charged-failed-request',1)")
    assert review_capacity("private", path)["sufficient"]
    with sqlite3.connect(path) as db:
        db.execute("insert into reservations values('charged-successful-request',2)")
    result = review_capacity("private", path)
    assert result["remaining"] == 60 and not result["sufficient"]


@pytest.mark.parametrize(
    "error,expected",
    [
        ("Declared VLM request budget exhausted", "review_budget_exhausted"),
        ("Truncated review is not admission", "native_language_review_incomplete"),
    ],
)
def test_language_technical_failures_remain_structured_and_unapproved(
    tmp_path, monkeypatch, error, expected
):
    import json
    from driversi.rendering.diagnostics import gs_sequence_critic as mod

    def fail(*a, **k):
        raise ValueError(error)

    monkeypatch.setattr(mod, "_review_labels", fail)
    out = tmp_path / "labels"
    result = mod.review_labels(
        "preview",
        "render",
        out,
        private_config="private",
        credential_file="private",
        request_budget="budget",
    )
    assert result["approved"] is False
    assert result["technical_review_failure"] == expected
    assert json.loads((out / "label-review.json").read_text()) == result
