"""Explicit numerical experimental admission; never an independent visual review."""

from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json

PROFILE = "experimental_numeric25_v1"


def strict_problem_profile(profile, threshold):
    return (profile in ('dynamic_history25_v1','dynamic_anchor25_v1','experimental_numeric27_strict_ade05_v1','experimental_numeric25_spatial05_v1','experimental_numeric25_adaptive_v1')
            and threshold==numeric_settings(profile)['psnr_db'])


def numeric_settings(profile=PROFILE):
    policies = {
        "dynamic_history25_v1": dict(psnr_db=25.,strict_psnr=True,minimum_ade_m=.20),
        "dynamic_anchor25_v1": dict(psnr_db=25.,strict_psnr=True,minimum_ade_m=.20),
        "experimental_numeric25_adaptive_v1": dict(psnr_db=25.,strict_psnr=True,minimum_ade_m=.20),
        PROFILE: dict(psnr_db=25.0, strict_psnr=True, minimum_ade_m=0.5),
        "experimental_numeric24_ade03_v1": dict(psnr_db=24.0, strict_psnr=False, minimum_ade_m=0.3),
        "experimental_numeric27_ade05_v1": dict(psnr_db=27.0, strict_psnr=False, minimum_ade_m=0.5),
        "experimental_numeric27_strict_ade05_v1": dict(psnr_db=27.0, strict_psnr=True, minimum_ade_m=0.5),
        "experimental_numeric25_spatial05_v1": dict(psnr_db=25.0, strict_psnr=True, minimum_ade_m=0.5, minimum_path_ade_m=0.5),
    }
    if profile not in policies:
        raise ValueError("Unknown numeric admission profile")
    return policies[profile]


def validate_diversity(generation_path, diversity_path, *, profile=PROFILE):
    from driversi.rsi.decisions.trajectory_diversity import compare, describe
    from driversi.rsi.plugins.synthesis import check_files

    if profile in ("dynamic_history25_v1", "dynamic_anchor25_v1", "experimental_numeric25_adaptive_v1"):
        return validate_adaptive_diversity(generation_path,diversity_path)
    settings=numeric_settings(profile)
    minimum = settings["minimum_ade_m"]
    spatial=settings.get('minimum_path_ade_m')
    g, evidence = read_mapping(generation_path), read_mapping(diversity_path)
    check_files(g["files"])
    description, decision = evidence["description"], evidence["decision"]
    measured = describe(dict(g, status="trajectory_ready"), description["candidate_id"])
    if spatial is not None and not measured.get('environment_speed_policy'):
        raise ValueError('New spatial protocol requires a frozen environment speed policy')
    if description != measured:
        raise ValueError("Diversity descriptor differs from sealed actual DriveRL execution")
    if decision.get("distinct") is not True or decision.get("minimum_ade_m") != minimum:
        raise ValueError("A retained ADE decision matching the declared profile is required")
    rule=('full_future_time_ADE_and_common_arc_path_ADE_v1' if spatial is not None
          else 'both_full_7s_and_future_5s_ADE_must_reach_minimum')
    if decision.get("rule") != rule:
        raise ValueError("Unknown ADE protocol")
    retained = evidence.get("retained")
    if retained is None:
        # Explicit migration compatibility for the previously sealed first family member.
        if not description["candidate_id"].endswith("-v000") or decision["comparisons"]:
            raise ValueError("Prior trajectory descriptors are required to replay ADE")
        retained = []
    for previous in retained:
        check_files(previous["files"])
        paths = [Path(p) for p in previous["files"] if Path(p).name == "teacher.npz"]
        if len(paths) != 1:
            raise ValueError("Prior diversity requires a unique executed trajectory")
        with np.load(paths[0], allow_pickle=False) as data:
            i = data["actor_ids"].tolist().index("ego")
            if not np.array_equal(
                data["states"][:, i, :2].astype(float), np.asarray(previous["xy"])
            ):
                raise ValueError("Prior ADE trajectory differs from executed state")
    if compare(description, retained, minimum_ade_m=minimum,minimum_path_ade_m=spatial) != decision:
        raise ValueError("ADE decision differs from actual retained trajectories")
    for row in decision["comparisons"]:
        values = [row["ade_7s_m"], row["ade_future_5s_m"]]
        if not np.isfinite(values).all() or min(values) < minimum or row["duplicate"]:
            raise ValueError("Duplicate trajectory cannot be admitted")
    return g, measured


def validate_adaptive_diversity(generation_path,diversity_path):
    from driversi.rsi.decisions.trajectory_diversity import describe
    from driversi.rsi.decisions.adaptive_diversity import compare
    from driversi.rsi.plugins.synthesis import check_files
    g=read_mapping(generation_path);evidence=read_mapping(diversity_path)
    check_files(g['files'])
    measured=describe(dict(g,status='trajectory_ready'),evidence['description']['candidate_id'])
    if measured!=evidence['description']:raise ValueError('Actual adaptive trajectory differs')
    retained=evidence['retained']
    for previous in retained:
        check_files(previous['files'])
        paths=[Path(p) for p in previous['files'] if Path(p).name=='teacher.npz']
        if len(paths)!=1:raise ValueError('Unique prior native trajectory required')
        with np.load(paths[0],allow_pickle=False) as data:
            ego=data['actor_ids'].tolist().index('ego')
            if not np.array_equal(data['states'][:,ego,:2].astype(float),np.asarray(previous['xy'])):
                raise ValueError('Prior executed trajectory differs')
    decision=compare(measured,retained,measured['adaptive_diversity_policy'])
    if decision!=evidence['decision'] or not decision['distinct']:
        raise ValueError('Adaptive diversity failed actual execution replay')
    return g,measured


def validate_ego_motion(frame, scene, states):
    from driversi.rendering.gaussian.worldengine_replay import physical_delta

    ego, tick = scene.actor_ids.index("ego"), frame["tick"]
    expected = physical_delta(states[tick, ego], scene.poses[tick, ego])
    actual = np.asarray(frame["motions"]["ego"])
    if (
        actual.shape != np.asarray(expected).shape
        or not np.isfinite(actual).all()
        or not np.allclose(actual, expected, rtol=0, atol=1e-8)
    ):
        raise ValueError("Renderer camera motion does not follow actual ego execution")


def validate_numeric_pixels(
    value, paths, branch, report, source_rows, canonical, scene, files, window
):
    from driversi.rendering.diagnostics.worldengine_admission import (
        _validate_source_frame,
        source_view_psnr,
        visual_policy,
    )

    profile = value["visual_quality_profile"]
    settings = numeric_settings(profile)
    expected_contract = {"dynamic_history25_v1": "history_rgb_future_states_v1",
                         "dynamic_anchor25_v1": "anchor_rgb_history_future_states_v1"}.get(profile)
    if expected_contract is not None and report.get("image_contract") != expected_contract:
        raise ValueError("Dynamic admission profile and image contract disagree")
    g, description = validate_diversity(paths["generation"], paths["diversity"], profile=profile)
    if (
        Path(g["branch_report"]).resolve() != paths["branch_report"]
        or Path(g["render_payload"]).resolve() != paths["render_report"]
    ):
        raise ValueError("Metric admission generation belongs to another physical branch")
    if branch["trajectory_evidence"].get("admission_profile") != "pdms_gt95_v1":
        raise ValueError("Numeric experiment requires strict native PDMS > 95")
    if branch.get("future_frames") != 10:
        raise ValueError("Complete 4 history plus 10 future window required")
    with np.load(Path(branch["execution"]) / "teacher.npz", allow_pickle=False) as data:
        states = data["states"]
    scores = []
    for frame in report["frames"]:
        validate_ego_motion(frame, scene, states)
        for camera, item in frame["cameras"].items():
            source = source_rows[(frame["tick"], camera)]
            if source["source_token"] != frame["source_token"]:
                raise ValueError("Source capture differs")
            rendered, real = _validate_source_frame(
                source,
                canonical[frame["source_token"]],
                camera,
                files,
                native_canonical_pose=item["canonical_source_camera_to_world"]
                if report.get("renderer_backend") == "mtgs_native"
                else None,
            )
            expected = (
                Path(scene.metadata["navsim_source"]["sensor_root"])
                / item["source_image_relative_path"]
            )
            if (
                real != expected.resolve()
                or source["exposure_factor"] != item["source_exposure_factor"]
            ):
                raise ValueError("Source photo or exposure changed")
            scores.append(source_view_psnr(rendered, real, source["exposure_factor"]))
    mean = float(np.mean(scores))
    threshold = settings["psnr_db"]
    passed = bool(
        np.isfinite(scores).all()
        and (mean > threshold if settings["strict_psnr"] else mean >= threshold)
    )
    binding = None
    reasons = [] if passed else ["source_reconstruction_psnr_not_above_25db" if profile in ("dynamic_history25_v1", "dynamic_anchor25_v1") else "source_reconstruction_psnr_failed"]
    if profile in ("dynamic_history25_v1", "dynamic_anchor25_v1"):
        from driversi.rendering.diagnostics.dynamic_binding import validate_history_binding

        with np.load(Path(branch["execution"]) / "teacher.npz", allow_pickle=False) as data:
            binding = validate_history_binding(report, scene, data["states"], data["presence"], canonical)
        reasons.extend(binding["rejection_reasons"])
    return dict(
        value,
        admitted=passed and not reasons,
        object_binding_audit=binding,
        image_contract=report.get("image_contract", "history_future_rgb_v1"),
        rejection_reasons=reasons if profile in ("dynamic_history25_v1", "dynamic_anchor25_v1") else []
        if passed
        else [
            "source_reconstruction_psnr_not_above_" + str(int(threshold)) + "db"
            if settings["strict_psnr"]
            else "source_reconstruction_psnr_below_" + str(int(threshold)) + "db"
        ],
        branch_report_sha256=digest(paths["branch_report"]),
        visual_quality_policy=visual_policy(profile),
        source_view_psnr_threshold=threshold,
        mean_source_psnr_db=mean,
        min_source_psnr_db=min(scores),
        source_view_count=len(scores),
        trajectory_window_data=window,
        independent_visual_review_performed=False,
        diagnostic_warnings=(["novel_view_pixels_have_no_ground_truth", "source_pitch_roll_retained_terrain_not_independently_certified"] if profile in ("dynamic_history25_v1", "dynamic_anchor25_v1") else [
            "appearance_alignment_visibility_not_independently_verified",
            "terrain_not_an_admission_gate",
        ]),
        scope="same_branch_dynamic_binding_history_only_not_novel_view_ground_truth" if binding is not None else "operator_authorized_numeric_experiment_not_verified_novel_view_quality",
        history_distinct_from_source=True,
        sampling_seed=description["sampling_seed"],
    )


def admit(generation, diversity, output, *, profile=PROFILE):
    from driversi.rendering.diagnostics.worldengine_admission import validate_admission

    gp, dp = Path(generation).resolve(), Path(diversity).resolve()
    g, description = validate_diversity(gp, dp, profile=profile)
    bp, rp, sp = [
        Path(g[k]).resolve() for k in ("branch_report", "render_payload", "source_report")
    ]
    branch, render, source = map(read_mapping, (bp, rp, sp))
    window = bp.parent / "trajectory-window.json"
    files = {
        **g["files"],
        **description["files"],
        **branch["files"],
        **branch["upstream_source_files"],
        **render["inputs"],
        **render.get("producer_code", {}),
        **source["inputs"],
        **source["producer_code"],
        **source["files"],
    }
    for frame in render["frames"]:
        for item in frame["cameras"].values():
            p = rp.parent / item["file"]
            files[str(p)] = digest(p)
    for p in (gp, dp, bp, rp, sp, window):
        files[str(p)] = digest(p)
    value = dict(
        schema_version=1,
        kind="worldengine_numeric_admission",
        source_log=branch["source_log"],
        parent_token=next(f["source_token"] for f in render["frames"] if f["tick"] == 20),
        branch_identity=read_mapping(g["render_report"])["branch_identity"],
        branch_report=str(bp),
        render_report=str(rp),
        source_comparison=str(sp),
        trajectory_window=str(window),
        generation=str(gp),
        diversity=str(dp),
        visual_quality_profile=profile,
        files=files,
        authorization=(
            "operator_dynamic_history_replay_confirmation_2026_10_30"
            if profile in ("dynamic_history25_v1", "dynamic_anchor25_v1") else
            "operator_adaptive_diversity_confirmation_2026_10_28"
            if profile == "experimental_numeric25_adaptive_v1" else
            "operator_numeric25_spatial_confirmation_2026_10_28"
            if profile == 'experimental_numeric25_spatial05_v1' else
            "operator_strict27_confirmation_2026_10_28"
            if profile == 'experimental_numeric27_strict_ade05_v1' else
            "operator_numeric_admission_2026_10_26"
            if profile == PROFILE
            else "operator_numeric27_ade05_confirmation_2026_10_27"
            if profile == "experimental_numeric27_ade05_v1"
            else "operator_numeric24_ade03_confirmation_2026_10_27"
        ),
        independent_visual_review_performed=False,
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "admission.json"
    frozen_json(path, value)
    result = validate_admission(path)
    write_json(out / "result.json", result)
    return result
