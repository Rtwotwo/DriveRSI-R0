"""Cheap repair routing over sealed renderer receipts; never visual admission."""

import math
import statistics
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.rendering.diagnostics.worldengine_admission import (
    CAMERAS,
    photometric_rejection,
    visual_policy,
)


def analyze_render(receipt, source, profile):
    policy = visual_policy(profile)
    report = receipt["render_report"]
    future = receipt["native_future_points"]
    if type(future) is not int or future not in (8, 10):
        raise ValueError("Native future horizon required")
    ticks = list(range(0, 21 + 5 * future, 5))
    if (
        [f["tick"] for f in report["frames"]] != ticks
        or report["history_ticks"] != [5, 10, 15, 20]
        or report["future_ticks"] != ticks[5:]
        or any(set(f["cameras"]) != set(CAMERAS) for f in report["frames"])
    ):
        raise ValueError("Complete native history/future and camera coverage required")
    expected = {(t, c) for t in ticks for c in CAMERAS}
    rows = {(f["tick"], f["camera"]): f for f in source["frames"]}
    if len(rows) != len(source["frames"]) or set(rows) != expected:
        raise ValueError("Source comparison has missing or duplicate camera frames")
    if source.get("source_mode") != "canonical_source_pose_without_motion_v1":
        raise ValueError("Canonical source comparison required")
    scores = [row["measured_psnr_db"] for row in rows.values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in scores):
        raise ValueError("Finite source comparison evidence required")
    mean = statistics.mean(scores)
    blockers, actions, diagnostics = [], [], []
    terrain = all(
        c.get("terrain_fully_supported") is True
        for f in report["frames"]
        for c in f["cameras"].values()
    )
    if terrain != (report.get("terrain_fully_supported") is True):
        raise ValueError("Terrain summary differs from camera evidence")
    if not terrain:
        (blockers if policy["require_terrain_support"] else diagnostics).append(
            "finite_terrain_support_missing"
        )
        actions.append("repair_or_replace_height_supported_asset")
    quality = photometric_rejection(mean, profile)
    if quality:
        blockers.append(quality)
        actions.append("refine_or_replace_reconstruction_before_rendering_more_branches")
    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support

    prerequisites = []
    for frame in report["frames"]:
        static = {item["actor_id"]: item for item in frame.get("static_background_candidates", [])}
        for token in frame["physical_missing_appearance"]:
            corners = frame.get("unresolved_physical_boxes", {}).get(token)
            for camera, item in frame["cameras"].items():
                outside = (
                    corners is not None
                    and box_frustum_support(
                        corners, item["camera_to_world"], item["output_ray_bounds"]
                    )["outside_frustum"]
                )
                if outside:
                    continue
                candidate = static.get(token)
                if not candidate:
                    reason = "physical_object_appearance_unresolved"
                    detail = "no_static_candidate_for_potentially_visible_unbound_actor"
                elif (
                    candidate.get("physical_immobile") is not True
                    or candidate.get("physical_background_geometry", {}).get("gaussian_count", 0)
                    <= 0
                ):
                    reason = "static_physical_registration_unresolved"
                    detail = "static_candidate_moves_or_lacks_physical_box_gaussian_support"
                else:
                    continue
                prerequisites.append(
                    dict(
                        tick=frame["tick"],
                        camera=camera,
                        actor_id=token,
                        reason=reason,
                        evidence=detail,
                    )
                )
    if prerequisites:
        (blockers if policy["require_complete_object_binding"] else diagnostics).extend(
            sorted({x["reason"] for x in prerequisites})
        )
        actions.append("supply_registered_appearance_for_present_objects_before_visual_review")
    missing = sorted(
        {token for f in report["frames"] for token in f["physical_missing_appearance"]}
    )
    errors = report.get("static_background_binding_errors", {})
    active_errors = {k: v for k, v in errors.items() if k in missing}
    unused_errors = {k: v for k, v in errors.items() if k not in missing}
    if missing:
        actions.append("verify_unresolved_object_appearance_against_source_images")
    if active_errors:
        actions.append("audit_static_lifecycle_and_geometry_before_background_binding")
    return dict(
        schema_version=1,
        kind="same_branch_render_readiness",
        status="repair_required" if blockers else "independent_visual_review_required",
        source_log=report["source_log"],
        asset_id=report["asset_id"],
        policy=policy,
        blockers=blockers,
        diagnostic_warnings=diagnostics,
        repair_actions=actions,
        source_psnr_mean_db=mean,
        source_psnr_scope=policy["quality_claim"],
        terrain_fully_supported=terrain,
        unresolved_appearance_tokens=missing,
        static_binding_errors=errors,
        active_static_binding_errors=active_errors,
        unused_static_binding_attempt_errors=unused_errors,
        missing_appearance_prerequisites=prerequisites,
        frame_camera_pairs=len(expected),
        history_frames=4,
        future_points=future,
        appearance_interpretation="unresolved_binding_does_not_prove_absent_pixels",
        admission_granted=False,
        trainable=False,
        student_failure_established=False,
        learning_gain_credited=False,
        promoted=False,
        retry_rule="No automatic rerender or training. Repair inputs in a new version; preserve rejected evidence.",
    )


def inspect_stored_render(refs, registry, profile):
    from driversi.agents.backends.native import digest
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage

    _, branch, bref = _stored_stage(refs["branch"], "selected_native_branch", Path(registry))
    _, rendered, rref = _stored_stage(
        refs["rendered"], "worldengine_inspection_window", Path(registry)
    )
    _, comparison, sref = _stored_stage(
        refs["source"], "worldengine_source_comparison", Path(registry)
    )
    receipt = read_mapping(rendered["artifacts"]["report"])
    source = read_mapping(comparison["artifacts"]["report"])
    payload = read_mapping(rendered["artifacts"]["render_report"])
    if (
        receipt["branch_identity"] != bref["identity"]
        or source["render_identity"] != rref["identity"]
        or source["render_report_sha256"] != digest(rendered["artifacts"]["render_report"])
        or payload != receipt["render_report"]
    ):
        raise ValueError("Readiness evidence belongs to different branches or renders")
    result = analyze_render(receipt, source, profile)
    result["references"] = dict(branch=bref, rendered=rref, source=sref)
    result["evidence_paths"] = {
        key: str(value["artifacts"]["report"])
        for key, value in [("branch", branch), ("rendered", rendered), ("source", comparison)]
    }
    return result
