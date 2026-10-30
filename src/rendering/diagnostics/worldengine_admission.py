"""Independent, sealed admission for executed WorldEngine training windows.

Source-view PSNR is a reconstruction prefilter, never novel-view ground truth.
All novelty views, physical objects, and finite terrain remain separate checks.
A renderer's trainable/quality flags are never sufficient admission evidence.
"""

import json
import math
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
TICKS = tuple(range(0, 71, 5))
SOURCE_PSNR_MIN_DB = 27.0
NUMERIC_PROFILES = (
    "dynamic_history25_v1",
    "dynamic_anchor25_v1",
    "experimental_numeric25_adaptive_v1",
    "experimental_numeric25_spatial05_v1",
    "experimental_numeric25_v1",
    "experimental_numeric24_ade03_v1",
    "experimental_numeric27_ade05_v1",
    "experimental_numeric27_strict_ade05_v1",
)


def visual_policy(name="source27_v1"):
    """Explicit reconstruction prefilters; neither certifies held-out NVS PSNR.

    The 24 dB pilot is an operator-authorized low-fidelity experiment, not the
    SimScale main protocol. Geometry, object binding and frame checks are shared.
    """
    thresholds = {
        "dynamic_history25_v1": 25.0,
        "dynamic_anchor25_v1": 25.0,
        "experimental_numeric25_adaptive_v1": 25.0,
        "source27_v1": 27.0,
        "experimental_source24_v1": 24.0,
        "experimental_scene24_v1": 24.0,
        "experimental_scene25_v1": 25.0,
        "experimental_egoview25_v1": 25.0,
        "experimental_numeric25_v1": 25.0,
        "experimental_numeric24_ade03_v1": 24.0,
        "experimental_numeric27_ade05_v1": 27.0,
        "experimental_numeric27_strict_ade05_v1": 27.0,
        "experimental_numeric25_spatial05_v1": 25.0,
    }
    if not isinstance(name, str) or name not in thresholds:
        raise ValueError("Unknown visual quality profile")
    return {
        "name": name,
        "source_psnr_min_db": thresholds[name],
        "experimental_low_fidelity": thresholds[name] < 27.0,
        "quality_claim": "source_reconstruction_proxy_not_held_out_nvs_quality",
        "geometry_and_temporal_checks": (
            "same_branch_ego_objects_history_images_and_future_states"
            if name in ("dynamic_history25_v1", "dynamic_anchor25_v1")
            else "sealed_ego_motion_and_complete_window_only"
            if name in NUMERIC_PROFILES
            else "ego_camera_rgb"
            if name == "experimental_egoview25_v1"
            else "unchanged"
        ),
        "require_terrain_support": name not in ("experimental_egoview25_v1", *NUMERIC_PROFILES),
        "require_complete_object_binding": name in ("dynamic_history25_v1", "dynamic_anchor25_v1") or name
        not in ("experimental_egoview25_v1", *NUMERIC_PROFILES),
        "require_pixel_review": name not in NUMERIC_PROFILES,
        "limited_appearance_experiment": name
        in ("experimental_scene24_v1", "experimental_scene25_v1", "experimental_egoview25_v1"),
        **(
            {
                "max_total_void_fraction": 0.05,
                "max_connected_void_fraction": 0.02,
                "max_uncertain_static_projection_fraction": 0.005,
            }
            if name
            in ("experimental_scene24_v1", "experimental_scene25_v1", "experimental_egoview25_v1")
            else {}
        ),
    }


def validate_review_scope(review, profile):
    """Declared review scope cannot be relabelled; legacy strict receipts stay readable."""
    visual_policy(profile)
    declared = review.get("visual_quality_profile")
    if (declared is not None and declared != profile) or (
        profile == "experimental_egoview25_v1" and declared != profile
    ):
        raise ValueError("Visual review scope differs from requested admission profile")


def photometric_rejection(mean_psnr, profile):
    policy = visual_policy(profile)
    if not math.isfinite(mean_psnr):
        raise ValueError("Non-finite reconstruction PSNR")
    if mean_psnr < policy["source_psnr_min_db"] or (
        profile in ("dynamic_history25_v1", "dynamic_anchor25_v1", "experimental_numeric25_adaptive_v1", "experimental_egoview25_v1", "experimental_numeric25_v1", "experimental_numeric25_spatial05_v1")
        and mean_psnr == policy["source_psnr_min_db"]
    ):
        return "source_reconstruction_psnr_below_" + str(int(policy["source_psnr_min_db"])) + "db"
    return None


def _read(path):
    return json.loads(Path(path).read_text())


def _sealed(path, files):
    path = Path(path).resolve(strict=True)
    if str(path) not in files or digest(path) != files[str(path)]:
        raise ValueError("Unsealed or changed visual evidence: " + str(path))
    return path


def _rows_by_frame(rows, label, *, future_points=10, image_contract="history_future_rgb_v1"):
    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Native eight or ten future points required")
    if image_contract not in {"history_future_rgb_v1", "history_rgb_future_states_v1", "anchor_rgb_history_future_states_v1"}:
        raise ValueError("Unknown image contract")
    ticks = ((20,) if image_contract == "anchor_rgb_history_future_states_v1" else (5, 10, 15, 20) if image_contract == "history_rgb_future_states_v1" else range(0, 21 + 5 * future_points, 5))
    expected = {(tick, camera) for tick in ticks for camera in CAMERAS}
    found = {}
    for row in rows:
        key = (row.get("tick"), row.get("camera"))
        if key not in expected or key in found:
            raise ValueError("Unexpected or repeated " + label + " frame")
        found[key] = row
    if set(found) != expected:
        raise ValueError("Incomplete " + label + " frame coverage")
    return found


def source_view_psnr(render_file, original_file, exposure_factor):
    """Same fixed upstream HSV exposure conversion and area resize as MTGS."""
    import cv2
    from PIL import Image

    if (
        type(exposure_factor) not in (int, float)
        or not math.isfinite(exposure_factor)
        or exposure_factor <= 0
    ):
        raise ValueError("Finite positive upstream source exposure required")
    image = np.asarray(Image.open(render_file).convert("RGB"))
    original = np.asarray(Image.open(original_file).convert("RGB"))
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(float)
    hsv[..., 2] = np.clip(hsv[..., 2] * exposure_factor, 0, 255)
    reference = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    reference = cv2.resize(
        reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
    )
    mse = float(np.mean((image.astype(float) - reference.astype(float)) ** 2))
    return float(10 * np.log10(255.0**2 / max(mse, 1e-12)))


def _validate_source_frame(source, canonical, camera, files, *, native_canonical_pose=None):
    """Reject source-photo swaps and transported poses before comparing pixels.

    Producer receipts are sealed by the Harness. Here we also recompute their
    canonical pose from the bound reconstruction metadata instead of trusting a
    caller's source_motions or quality flag.
    """
    from driversi.rendering.gaussian.worldengine_portable import camera_pose

    rendered = _sealed(source["rendered_source_image"], files)
    real = _sealed(source["real_source_image"], files)
    expected_name = f"{source['tick']:04d}-{camera}-source-render.png"
    if rendered == real or rendered.name != expected_name or source.get("source_motions") != {}:
        raise ValueError("Real photos or novel poses cannot replace source reconstruction")
    if source.get("source_token") != canonical["token"] or source.get(
        "source_appearance_timestamp_us"
    ) != int(canonical["timestamp"]):
        raise ValueError("Source reconstruction capture or appearance time differs")
    expected_pose = (
        camera_pose(canonical, canonical["cams"][camera])
        if native_canonical_pose is None
        else np.asarray(native_canonical_pose)
    )
    pose = np.asarray(source.get("source_camera_to_world"), dtype=float)
    if pose.shape != (3, 4) or not np.allclose(pose, expected_pose, rtol=0, atol=1e-8):
        raise ValueError("Source comparison is not at the canonical reconstruction pose")
    return rendered, real


def _validate_physical_frame(frame, camera, scene, states, presence):
    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support
    from driversi.rendering.gaussian.worldengine_replay import physical_delta

    tick, actor_ids = frame["tick"], scene.actor_ids
    missing = set(frame["physical_missing_appearance"])
    actual_present = {token for i, token in enumerate(actor_ids) if presence[tick, i]}
    active = set(camera["active_object_tokens"])
    if actual_present - {"ego"} - active != missing:
        raise ValueError("Renderer omitted physical appearance deficits")
    expected_motions = actual_present & (active | {"ego"})
    if set(frame["motions"]) != expected_motions:
        raise ValueError("Rendered motion actors differ from executed physics")
    for token in expected_motions:
        index = actor_ids.index(token)
        expected_delta = physical_delta(states[tick, index], scene.poses[tick, index])
        if not np.allclose(frame["motions"][token], expected_delta, rtol=0, atol=1e-8):
            raise ValueError("Renderer did not use the executed object motion")
    unsupported = []
    for token in active - set(actor_ids):
        corners = frame.get("source_only_object_boxes", {}).get(token)
        if (
            corners is None
            or not box_frustum_support(
                corners, camera["camera_to_world"], camera["output_ray_bounds"]
            )["outside_frustum"]
        ):
            unsupported.append(token)
    return unsupported


def _validate_background_frame(frame, background_bindings, scene, states):
    """Recompute immutable corners and identity; reject per-frame report swaps."""
    from driversi.rendering.gaussian.worldengine_static_bindings import (
        static_background_box_at_tick,
    )

    tick = frame["tick"]
    candidates = {item["actor_id"]: item for item in frame.get("static_background_candidates", [])}
    for token in set(frame["physical_missing_appearance"]) & set(background_bindings):
        corners, anchor = static_background_box_at_tick(
            background_bindings[token],
            states[tick, scene.actor_ids.index(token)],
            tick,
            scene.metadata["source_start_timestamp_us"] + tick * 100000,
        )
        recorded = np.asarray(frame.get("unresolved_physical_boxes", {}).get(token))
        if (
            recorded.shape != corners.shape
            or not np.allclose(recorded, corners, atol=1e-8, rtol=0)
            or candidates.get(token, {}).get("immutable_source_binding") != anchor
        ):
            raise ValueError("Static background frame differs from immutable source registration")


def validate_admission(path):
    """Verify evidence content and derive admission; stale/foreign data raises.

    A complete, internally consistent rejection is returned with admitted=False.
    This allows Harness to retain failed syntheses without granting training.
    """
    from driversi.contracts.scene import Scene
    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support
    from driversi.rsi.plugins.native_branch import validate_trajectory_evidence
    from driversi.rsi.plugins.worldengine import validate_render

    value = _read(path)
    numeric = value.get("kind") == "worldengine_numeric_admission"
    if value.get("schema_version") != 1 or value.get("kind") not in (
        "worldengine_visual_admission",
        "worldengine_numeric_admission",
    ):
        raise ValueError("Unsupported WorldEngine admission schema")
    policy = visual_policy(value.get("visual_quality_profile", "source27_v1"))
    if numeric != (policy["name"] in NUMERIC_PROFILES):
        raise ValueError("Numeric admission requires its own explicit experimental profile")
    files = value["files"]
    for name in files:
        _sealed(name, files)
    paths = {
        key: _sealed(value[key], files)
        for key in (
            "branch_report",
            "render_report",
            "trajectory_window",
            "source_comparison",
            *(("generation", "diversity") if numeric else ("visual_review",)),
        )
    }
    branch, report = _read(paths["branch_report"]), _read(paths["render_report"])
    window = _read(paths["trajectory_window"])
    future_points = branch.get("future_frames")
    if (
        branch.get("source_log") != value["source_log"]
        or report.get("source_log") != value["source_log"]
        or branch.get("trajectory_window_sha256") != digest(paths["trajectory_window"])
        or type(future_points) is not int
        or future_points not in (8, 10)
        or len(window.get("future", [])) != future_points
        or branch.get("anchor_tick") != 20
    ):
        raise ValueError("Branch, source and declared native future window differ")
    for name, sha in {
        **branch["files"],
        **branch["upstream_source_files"],
        **report["inputs"],
    }.items():
        if files.get(str(Path(name).resolve())) != sha:
            raise ValueError("Admission omits bound branch or asset evidence")
    validate_trajectory_evidence(branch["trajectory_evidence"])
    if branch.get("teacher_qualified") is not True:
        raise ValueError("Teacher trajectory was not admitted")
    rollout_path = Path(branch["execution"]) / (branch["branch"] + ".npz")
    with np.load(rollout_path, allow_pickle=False) as data:
        states, presence = data["states"], data["presence"]
        actor_ids = tuple(data["actor_ids"].tolist())
    image_files = validate_render(
        report, paths["render_report"].parent, report["inputs"], future_points=future_points
    )
    if any(files.get(name) != sha for name, sha in image_files.items()):
        raise ValueError("Admission omits or changes rendered pixels")
    execution = Path(branch["execution"])
    scene = Scene.load(execution / "scene.npz")
    if actor_ids != scene.actor_ids or len(states) <= 70:
        raise ValueError("Visual physics population or complete future differs")
    if scene.metadata["source_log"] != value["source_log"]:
        raise ValueError("Physical scene source differs")
    if (
        report["inputs"].get(str(execution / "scene.npz"))
        != branch["files"][str(execution / "scene.npz")]
    ):
        raise ValueError("Renderer did not bind exact selected physical scene")
    if (
        report["inputs"].get(str(execution / (branch["branch"] + ".npz")))
        != branch["files"][str(execution / (branch["branch"] + ".npz"))]
    ):
        raise ValueError("Renderer did not bind exact selected physical rollout")
    # Parent is the reached anchor's source token, not the earlier scene start.
    anchor = next(row for row in report["frames"] if row["tick"] == 20)
    if anchor["source_token"] != value["parent_token"]:
        raise ValueError("Admission parent token differs from the source anchor")
    comparison = _read(paths["source_comparison"])
    review = _read(paths["visual_review"]) if not numeric else {}
    if comparison.get("render_report_sha256") != digest(paths["render_report"]) or (
        not numeric
        and (
            review.get("render_report_sha256") != digest(paths["render_report"])
            or review.get("branch_identity") != value["branch_identity"]
        )
    ):
        raise ValueError("Independent comparison/review refers to another branch")
    if (
        comparison.get("kind") != "worldengine_source_comparison"
        or comparison.get("source_mode") != "canonical_source_pose_without_motion_v1"
    ):
        raise ValueError("A source-render producer receipt is required")
    native = report.get("renderer_backend") == "mtgs_native"
    producer = "mtgs_training_window.py" if native else "worldengine_source.py"
    expected_code = Path(__file__).resolve().parents[1] / "gaussian" / producer
    if comparison.get("producer_code", {}).get(str(expected_code)) != digest(expected_code):
        raise ValueError("Source comparison did not use the registered source renderer")
    for name, sha in {
        **comparison["inputs"],
        **comparison["producer_code"],
        **comparison["files"],
    }.items():
        if files.get(str(Path(name).resolve())) != sha:
            raise ValueError("Admission omits source producer evidence")
    for name, sha in comparison["inputs"].items():
        if name.endswith(".ckpt") or Path(name).name == "video_scene_dict.pkl":
            if report["inputs"].get(name) != sha:
                raise ValueError("Source renderer used another reconstruction")
    import pickle

    metadata = [
        Path(name) for name in report["inputs"] if Path(name).name == "video_scene_dict.pkl"
    ]
    if len(metadata) != 1:
        raise ValueError("A unique bound reconstruction metadata file is required")
    with metadata[0].open("rb") as stream:
        videos = pickle.load(stream)
    video_id = report["asset_id"] + "-" + str(report["travel_id"] if native else 0)
    if video_id not in videos or (not native and set(videos) != {video_id}):
        raise ValueError("Source reconstruction metadata identity differs")
    if native and comparison.get("renderer_backend") != "mtgs_native":
        raise ValueError("Source and branch use different renderer backends")
    video = videos[video_id]
    if video["log_name"] != value["source_log"]:
        raise ValueError("Source reconstruction log differs")
    canonical = {item["token"]: item for item in video["frame_infos"] if not item.get("skipped")}
    background_bindings = report.get("static_background_bindings", {})
    if background_bindings and (not numeric or policy["name"] in ("dynamic_history25_v1", "dynamic_anchor25_v1")):
        for name in (
            "worldengine_replay.py",
            "worldengine_geometry.py",
            "worldengine_static_bindings.py",
        ):
            code = (Path(__file__).resolve().parents[1] / "gaussian" / name).resolve()
            if report.get("producer_code", {}).get(str(code)) != digest(_sealed(code, files)):
                raise ValueError("Static background renderer implementation changed")
        from driversi.rendering.gaussian.worldengine_static_bindings import (
            validate_static_background_binding,
        )

        raw_paths = [
            Path(name)
            for name in report["inputs"]
            if Path(name).name == value["source_log"] + ".pkl"
        ]
        if len(raw_paths) != 1:
            raise ValueError("Static background requires unique sealed original source log")
        with _sealed(raw_paths[0], files).open("rb") as stream:
            raw_frames = pickle.load(stream)
        for token, binding in background_bindings.items():
            if token != binding.get("actor_id"):
                raise ValueError("Static background identity key differs")
            validate_static_background_binding(
                binding, scene, states, presence, raw_frames, canonical.__getitem__
            )
    source_rows = _rows_by_frame(
        comparison["frames"], "source comparison", future_points=future_points,
        image_contract=report.get("image_contract", "history_future_rgb_v1"),
    )
    if numeric:
        from driversi.rendering.diagnostics.numeric_admission import validate_numeric_pixels

        return validate_numeric_pixels(
            value, paths, branch, report, source_rows, canonical, scene, files, window
        )
    review_rows = _rows_by_frame(review["frames"], "visual review", future_points=future_points)
    reasons, scores, diagnostics = [], [], []
    object_rejections = reasons if policy["require_complete_object_binding"] else diagnostics
    validate_review_scope(review, policy["name"])
    immutable_anchors = {}
    if (
        not review.get("reviewer")
        or review.get("method") != "paired_source_and_counterfactual_sequence_inspection_v1"
    ):
        reasons.append("independent_sequence_review_missing")
    if report.get("terrain_fully_supported") is not True:
        (reasons if policy["require_terrain_support"] else diagnostics).append(
            "finite_terrain_support_missing"
        )
    for frame in report["frames"]:
        tick = frame["tick"]
        missing = set(frame["physical_missing_appearance"])
        static = {item["actor_id"]: item for item in frame.get("static_background_candidates", [])}
        _validate_background_frame(frame, background_bindings, scene, states)
        for cam, item in frame["cameras"].items():
            if _validate_physical_frame(frame, item, scene, states, presence):
                object_rejections.append("rendered_actor_without_physical_counterpart")
            if native:
                from driversi.rendering.gaussian.mtgs_bindings import validate_binding

                if item.get("object_transport") != "decoded_native_cloud_rigid_v2":
                    raise ValueError("Native object anchoring contract missing")
                bindings = item.get("object_bindings", {})
                expected = set(frame["motions"]) - {"ego"}
                if set(bindings) != expected:
                    raise ValueError("Native object anchors omit executed actors")
                for token, binding in bindings.items():
                    anchor_source = canonical.get(binding.get("source_token"))
                    if anchor_source is None:
                        raise ValueError("Native object anchor absent from sealed source metadata")
                    validate_binding(
                        binding,
                        anchor_source,
                        token,
                        states[tick, actor_ids.index(token)],
                        tick,
                        int(canonical[frame["source_token"]]["timestamp"]),
                        scene,
                    )
                    if binding["policy"] == "immutable_static_observation_v1":
                        previous = immutable_anchors.setdefault(token, binding["source_token"])
                        if previous != binding["source_token"]:
                            raise ValueError("Static native object changed its immutable anchor")
            key = tick, cam
            source, checked = source_rows[key], review_rows[key]
            if (
                report.get("renderer_backend") == "worldengine_portable"
                and review.get("reviewer") == "independent_private_vlm_exact_pixel_critic_v1"
            ):
                from driversi.rendering.diagnostics.window_review import required_portable_actors

                required = required_portable_actors(
                    frame,
                    item,
                    scene,
                    states,
                    presence,
                    canonical,
                    allow_unknown_height=not policy["require_terrain_support"],
                )
                actor_checks = checked.get("actor_checks", [])
                ids = [a.get("actor_id") for a in actor_checks]
                if policy["require_complete_object_binding"] and (
                    len(ids) != len(set(ids)) or set(ids) != required
                ):
                    reasons.append("automatic_review_physical_actor_coverage_missing")
                if any(
                    a.get("appearance_status") != "pass" or a.get("alignment_status") != "pass"
                    for a in actor_checks
                ):
                    object_rejections.append("automatic_review_actor_geometry_unconfirmed")
            if source.get("source_token") != frame["source_token"]:
                raise ValueError("Source comparison uses another capture")
            if frame["source_token"] not in canonical:
                raise ValueError("Rendered source absent from bound reconstruction")
            rendered, real = _validate_source_frame(
                source,
                canonical[frame["source_token"]],
                cam,
                files,
                native_canonical_pose=item["canonical_source_camera_to_world"] if native else None,
            )
            expected_real = (
                Path(scene.metadata["navsim_source"]["sensor_root"])
                / item["source_image_relative_path"]
            )
            if (
                real != expected_real.resolve()
                or source.get("exposure_factor") != item["source_exposure_factor"]
            ):
                raise ValueError("Comparison substituted a source image or exposure")
            scores.append(source_view_psnr(rendered, real, source["exposure_factor"]))
            if (
                checked.get("image_sha256")
                != image_files[str(paths["render_report"].parent / item["file"])]
            ):
                raise ValueError("Visual review did not inspect exact output pixels")
            if policy["limited_appearance_experiment"]:
                from PIL import Image

                from .scene_quality import frame_allowed, hole_fractions

                mask_path = checked.get("void_mask")
                if not mask_path:
                    reasons.append("experimental_void_mask_missing")
                else:
                    with Image.open(_sealed(mask_path, files)) as mask_image:
                        mask = np.asarray(mask_image)
                    with Image.open(paths["render_report"].parent / item["file"]) as image:
                        if mask.shape != (image.height, image.width):
                            raise ValueError("Void-mask resolution differs from reviewed image")
                    if not frame_allowed(checked, hole_fractions(mask), profile=policy["name"]):
                        reasons.append("experimental_scene_review_not_passed")
            elif checked.get("verdict") != "pass" or checked.get("issues") != []:
                reasons.append("novel_view_review_not_passed")
            bindings = {entry["actor_id"]: entry for entry in checked.get("static_bindings", [])}
            if native:
                for token, asset_binding in item["object_bindings"].items():
                    geometry = asset_binding.get("source_obb_registration", {})
                    from driversi.rendering.gaussian.mtgs_bindings import observed_3d_box_corners

                    anchor_source = canonical[asset_binding["source_token"]]
                    source_corners, source_obb = observed_3d_box_corners(
                        anchor_source, token, scene.dimensions[actor_ids.index(token)]
                    )
                    if geometry.get("source_3d_observation") != source_obb:
                        raise ValueError("Native source OBB does not match full 3D observation")
                    if (
                        geometry.get("gaussian_count", 0) <= 0
                        or geometry.get("native_center_inside_physical_source_box") is not True
                    ):
                        object_rejections.append("native_source_object_registration_unresolved")
                    if item["terrain_evidence"][token].get("status") == "source_identity":
                        # No new height inference is made; inspect this fixed
                        # source asset against its actual physical box instead.
                        # Exact unchanged native geometry stays in its bound
                        # full-3D observation, not the planar simulator Z.
                        corners = source_corners
                        outside = box_frustum_support(
                            corners, item["camera_to_world"], item["output_ray_bounds"]
                        )["outside_frustum"]
                        binding = bindings.get(token, {})
                        if not outside and (
                            binding.get("appearance_confirmed") is not True
                            or binding.get("physical_box_alignment_confirmed") is not True
                        ):
                            object_rejections.append(
                                "source_identity_object_visual_registration_unresolved"
                            )
            for token in missing:
                corners = frame.get("unresolved_physical_boxes", {}).get(token)
                if (
                    corners is not None
                    and box_frustum_support(
                        corners, item["camera_to_world"], item["output_ray_bounds"]
                    )["outside_frustum"]
                ):
                    continue
                evidence, binding = static.get(token), bindings.get(token)
                if (
                    policy["limited_appearance_experiment"]
                    and evidence
                    and binding
                    and corners is not None
                    and token in background_bindings
                ):
                    from .scene_quality import (
                        projected_box_fraction,
                        small_static_appearance_allowed,
                    )

                    if small_static_appearance_allowed(
                        scene.metadata.get("source_actor_kinds", {}).get(token),
                        evidence,
                        binding,
                        projected_box_fraction(corners, item),
                    ):
                        continue
                # Static nodes are optional, but same physical pose, spatial GS
                # support and independently inspected appearance are mandatory.
                if not evidence or not binding or binding.get("appearance_confirmed") is not True:
                    object_rejections.append("physical_object_appearance_unresolved")
                    continue
                geometry = evidence.get("physical_background_geometry", {})
                if (
                    evidence.get("physical_immobile") is not True
                    or geometry.get("gaussian_count", 0) <= 0
                    or binding.get("physical_box_alignment_confirmed") is not True
                ):
                    object_rejections.append("static_physical_registration_unresolved")
    mean_psnr = float(np.mean(scores))
    photometric = photometric_rejection(mean_psnr, policy["name"])
    if photometric:
        reasons.append(photometric)
    reasons = sorted(set(reasons))
    return {
        **value,
        "admitted": not reasons,
        "rejection_reasons": reasons,
        "diagnostic_warnings": sorted(set(diagnostics)),
        "branch_report_sha256": digest(paths["branch_report"]),
        "visual_quality_policy": policy,
        "source_view_psnr_threshold": policy["source_psnr_min_db"],
        "mean_source_psnr_db": mean_psnr,
        "min_source_psnr_db": min(scores),
        "source_view_count": len(scores),
        "trajectory_window_data": window,
        "scope": "bounded_executed_branch_visual_and_trajectory_admission_not_model_benefit",
    }


# Explicit public name for consumers that handle several sensor backends.
validate_visual_admission = validate_admission
