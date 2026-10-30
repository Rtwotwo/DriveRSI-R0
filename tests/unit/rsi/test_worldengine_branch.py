import copy

import pytest

from driversi.rsi.plugins.worldengine import CAMERAS, FUTURE, HISTORY, validate_render
from driversi.rsi.workflow.providers import builtin_providers


def test_inspection_render_cannot_feed_training():
    p = builtin_providers().get("worldengine_selected_branch_render")
    assert p.inputs == (("branch", "selected_native_branch"),)
    assert p.output == "worldengine_inspection_window"
    assert p.output != "native_training_dataset"


def report_fixture(root):
    rows = []
    for tick in range(0, 61, 5):
        cameras = {}
        for cam in CAMERAS:
            name = f"{tick:04d}-{cam}.png"
            (root / name).write_bytes(b"test")
            cameras[cam] = dict(
                file=name,
                source_token=str(tick),
                terrain_evidence={"ego": {"status": "unsupported", "reason": "fixture"}},
                terrain_fully_supported=False,
            )
        rows.append(
            dict(tick=tick, source_token=str(tick), cameras=cameras, motions={"ego": [0, 0, 0]})
        )
    return dict(
        status="rendered_diagnostic",
        image_count=52,
        history_ticks=HISTORY,
        future_ticks=FUTURE,
        quality_admitted=False,
        trainable=False,
        terrain_fully_supported=False,
        inputs={"source": "hash"},
        frames=rows,
    )


def test_render_rejects_stale_branch_missing_frame_and_self_admission(tmp_path):
    report = report_fixture(tmp_path)
    assert len(validate_render(report, tmp_path, {"source": "hash"})) == 52
    for field, value in [
        ("inputs", {}),
        ("trainable", True),
        ("quality_admitted", True),
        ("future_ticks", FUTURE[:-1]),
        ("frames", report["frames"][:-1]),
    ]:
        invalid = copy.deepcopy(report)
        invalid[field] = value
        with pytest.raises(ValueError):
            validate_render(invalid, tmp_path, {"source": "hash"})
    (tmp_path / "0060-CAM_B0.png").unlink()
    with pytest.raises(ValueError, match="Missing"):
        validate_render(report, tmp_path, {"source": "hash"})


def test_render_rejects_cross_camera_source_and_foreign_path(tmp_path):
    report = report_fixture(tmp_path)
    for field, value in [("file", "../foreign.png"), ("source_token", "wrong")]:
        invalid = copy.deepcopy(report)
        invalid["frames"][0]["cameras"]["CAM_F0"][field] = value
        with pytest.raises(ValueError, match="identity"):
            validate_render(invalid, tmp_path, {"source": "hash"})


def test_render_rejects_forged_terrain_support(tmp_path):
    report = report_fixture(tmp_path)
    report["terrain_fully_supported"] = True
    with pytest.raises(ValueError, match="terrain summary"):
        validate_render(report, tmp_path, {"source": "hash"})
    report["terrain_fully_supported"] = False
    report["frames"][0]["cameras"]["CAM_F0"]["terrain_evidence"] = {}
    with pytest.raises(ValueError, match="terrain motion"):
        validate_render(report, tmp_path, {"source": "hash"})


def test_ten_future_frames_cannot_be_filled_by_eight_frame_receipt(tmp_path):
    report = report_fixture(tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_render(report, tmp_path, {"source": "hash"}, future_points=10)
    for tick in (65, 70):
        row = copy.deepcopy(report["frames"][-1])
        row.update(tick=tick, source_token=str(tick))
        for camera, value in row["cameras"].items():
            value.update(file=f"{tick:04d}-{camera}.png", source_token=str(tick))
            (tmp_path / value["file"]).write_bytes(b"test")
        report["frames"].append(row)
    report.update(image_count=60, future_ticks=list(range(25, 71, 5)))
    assert len(validate_render(report, tmp_path, {"source": "hash"}, future_points=10)) == 60
    with pytest.raises(ValueError, match="Incomplete"):
        validate_render(report, tmp_path, {"source": "hash"})
    report["frames"][-1]["tick"] = 65
    with pytest.raises(ValueError, match="Missing or repeated"):
        validate_render(report, tmp_path, {"source": "hash"}, future_points=10)


def test_source_identity_requires_exact_bound_pose_and_native_backend(tmp_path):
    from driversi.rendering.gaussian.mtgs_bindings import exact_float32_identity

    report = report_fixture(tmp_path)
    report["renderer_backend"] = "mtgs_native"
    row = report["frames"][0]
    row["motions"] = {"car": [0.0, 0.0, 0.0]}
    for camera in row["cameras"].values():
        binding = dict(
            source_token="anchor",
            reference_tick=0,
            reference_physical_pose=[3.0, 2.0, 0.0, 0.0],
            target_physical_pose=[3.0, 2.0, 0.0, 0.0],
        )
        camera["object_bindings"] = {"car": binding}
        camera["terrain_evidence"] = {
            "car": dict(
                status="source_identity",
                method="source_identity_transport_v1",
                source_token="anchor",
                reference_tick=0,
                identity=exact_float32_identity(
                    binding["reference_physical_pose"], binding["target_physical_pose"]
                ),
            )
        }
        camera["terrain_fully_supported"] = True
    validate_render(report, tmp_path, {"source": "hash"})
    bad = copy.deepcopy(report)
    bad["renderer_backend"] = "worldengine_portable"
    with pytest.raises(ValueError, match="identity"):
        validate_render(bad, tmp_path, {"source": "hash"})
    bad = copy.deepcopy(report)
    bad["frames"][0]["cameras"]["CAM_F0"]["object_bindings"]["car"]["target_physical_pose"][0] += (
        1e-5
    )
    with pytest.raises(ValueError, match="moves geometry"):
        validate_render(bad, tmp_path, {"source": "hash"})


def test_measured_terrain_rejects_nan_independent_agreement(tmp_path):
    report = report_fixture(tmp_path)
    value = dict(
        status="supported",
        method="measured_road_source_clearance_v1",
        source_clearance_m=1.0,
        vertical_correction_m=0.0,
        vertical_displacement_m=0.0,
    )
    for side in ("source", "target"):
        value[side] = dict(
            status="supported",
            independent_surface_validation=dict(
                method="dominant_measured_road_layer_disjoint_traversal_validation_v1",
                independent_center_difference_m=float("nan"),
                independent_max_plane_difference_m=0.0,
            ),
        )
    camera = report["frames"][0]["cameras"]["CAM_F0"]
    camera["terrain_evidence"] = {"ego": value}
    camera["terrain_fully_supported"] = True
    with pytest.raises(ValueError, match="independent traversal"):
        validate_render(report, tmp_path, {"source": "hash"})


def test_traversal_consensus_dispatch_keeps_strict_independent_support(tmp_path):
    import numpy as np

    from driversi.rendering.gaussian.mtgs_road_surface import (
        MeasuredRoadSurface,
        TraversalConsensusRoadSurface,
    )

    x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    surfaces = [MeasuredRoadSurface(
        np.column_stack((x.ravel(), y.ravel(), (1 + .02*x + shift).ravel())),
        {"independent_test_traversal": shift},
    ) for shift in (0., .005, .01)]
    road = TraversalConsensusRoadSurface(surfaces[0], surfaces[1:])
    _, terrain = road.lift([0., 0., 1.5], [.1, .1, 0.])
    report = report_fixture(tmp_path)
    camera = report["frames"][0]["cameras"]["CAM_F0"]
    camera["terrain_evidence"] = {"ego": terrain}
    camera["terrain_fully_supported"] = True
    assert len(validate_render(report, tmp_path, {"source": "hash"})) == 52
    independent = terrain["source"]["independent_surface_validation"]
    independent["traversals"].pop()
    with pytest.raises(ValueError, match="two independent"):
        validate_render(report, tmp_path, {"source": "hash"})


def test_stored_source_preserves_original_identity_and_rejects_changed_receipt(
    monkeypatch, tmp_path
):
    from driversi.rsi.plugins import chainflow_synthetic, worldengine

    receipt = {"summary": {"artifact_reference": {"identity": "original"}}}
    monkeypatch.setattr(
        chainflow_synthetic, "_stored_stage", lambda *args: ({}, receipt, {"identity": "original"})
    )
    monkeypatch.setattr(
        worldengine, "execute_source", lambda cfg, inputs, output, registry: inputs["render"]
    )
    cfg = {
        "reference": {"source_task": "task", "source_stage": "render"},
        "identity": "original",
        "source_config": {},
    }
    assert worldengine.execute_stored_source(cfg, {}, tmp_path, None) is receipt
    cfg["identity"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        worldengine.execute_stored_source(cfg, {}, tmp_path, None)


def test_admission_default_is_strict_and_pilot_profile_explicit(tmp_path):
    from driversi.rsi.plugins.worldengine import prepare_admission

    review = tmp_path / "review.json"
    review.write_text("{}")
    default, _ = prepare_admission({"visual_review": str(review)}, tmp_path, None)
    assert default["visual_quality_profile"] == "source27_v1"
    pilot, _ = prepare_admission(
        {"visual_review": str(review), "visual_quality_profile": "experimental_source24_v1"},
        tmp_path,
        None,
    )
    assert pilot["visual_quality_profile"] == "experimental_source24_v1"
    with pytest.raises(ValueError):
        prepare_admission({"visual_review": str(review), "threshold": 0}, tmp_path, None)


def test_legacy_pooled_terrain_cannot_bypass_residual_proof(tmp_path):
    import numpy as np

    from driversi.rendering.gaussian.mtgs_road_surface import (
        IndependentlyValidatedRoadSurface,
        MeasuredRoadSurface,
    )

    x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    surfaces = [MeasuredRoadSurface(
        np.column_stack((x.ravel(), y.ravel(), (1 + .02*x + shift).ravel())),
        {"independent_test_traversal": shift},
    ) for shift in (0., .005)]
    road = IndependentlyValidatedRoadSurface(*surfaces)
    _, terrain = road.lift([0., 0., 1.5], [.1, .1, 0.])
    report = report_fixture(tmp_path)
    camera = report["frames"][0]["cameras"]["CAM_F0"]
    camera["terrain_evidence"] = {"ego": terrain}
    camera["terrain_fully_supported"] = True
    assert len(validate_render(report, tmp_path, {"source": "hash"})) == 52
    # Keeping good scalar agreement cannot rescue a stale producer proof.
    bad = copy.deepcopy(report)
    sample = bad["frames"][0]["cameras"]["CAM_F0"]["terrain_evidence"]["ego"]["source"]
    validation = sample["independent_surface_validation"]
    validation.pop("terrain_proof_version", None)
    sample.pop("terrain_proof_version", None)
    with pytest.raises(ValueError):
        validate_render(bad, tmp_path, {"source": "hash"})


def test_declared_render_size_is_checked_against_pixels_and_camera(tmp_path):
    from PIL import Image

    report = report_fixture(tmp_path)
    report["output_image_size"] = [960, 540]
    for frame in report["frames"]:
        for camera in frame["cameras"].values():
            camera["output_image_size"] = [960, 540]
            Image.new("RGB", (960, 540)).save(tmp_path / camera["file"])
    assert len(validate_render(report, tmp_path, {"source": "hash"})) == 52
    report["frames"][0]["cameras"]["CAM_F0"]["output_image_size"] = [1920, 1080]
    with pytest.raises(ValueError, match="output size differ"):
        validate_render(report, tmp_path, {"source": "hash"})
