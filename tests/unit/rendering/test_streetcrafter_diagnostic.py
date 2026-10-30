"""Prevent a small native pilot from masquerading as paired/complete supervision."""

import json
import subprocess
import sys
import types

import numpy as np
import pytest

from driversi.agents.backends.native import digest
from driversi.rendering.video.streetcrafter import diagnostic


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(diagnostic, "SIZE", (4, 3))
    monkeypatch.setattr(diagnostic, "__file__", str(tmp_path / "src/rendering/video/streetcrafter/diagnostic.py"))
    source = tmp_path / "source.png"
    guide = tmp_path / "guide.png"
    Image.new("RGB", (8, 6), (32, 64, 96)).save(source)
    Image.new("RGB", (4, 3), (32, 64, 96)).save(guide)
    bundle_path = tmp_path / "bundle.json"
    bundle = dict(branch_identity="sealed", camera_order=["CAM_F0", "CAM_L0"],
                  source_start_timestamp_us=1000, history_ticks=[5, 10, 15, 20],
                  future_ticks=list(range(25, 71, 5)))
    bundle_path.write_text(json.dumps(bundle))
    monkeypatch.setattr(diagnostic, "validate_bundle", lambda _: bundle)
    pose = np.eye(4)
    target = pose.copy()
    target[0, 3] = 0.0015
    geometry = dict(
        kind="calibrated_video_conditions", source_control=False, branch_identity="sealed",
        bundle=str(bundle_path), camera_order=bundle["camera_order"],
        initial_reference_images={"CAM_F0": str(source)},
        initial_reference_ego_to_world=pose.tolist(),
        frames=[dict(
            tick=t, timestamp_us=1000 + t * 100000, source_timestamp_us=1000 + t * 100000,
            delta_xy_yaw=[0, 0, 0], source_ego_to_world=pose.tolist(),
            cameras={"CAM_F0": dict(source_image=str(source), camera_to_ego=pose.tolist(),
                                    camera_to_world=target.tolist())},
        ) for t in range(71)],
        files={str(source): digest(source), str(bundle_path): digest(bundle_path)},
    )
    geometry_path = tmp_path / "geometry.json"
    spec = dict(
        kind="streetcrafter_native_conditions", schema_version=1, branch_identity="sealed",
        bundle=str(bundle_path), geometry=str(geometry_path), camera_order=bundle["camera_order"],
        initial_reference_images={"CAM_F0": str(source)},
        point_cloud_recipe={"resolution": [3, 4]},
        frames=[dict(tick=t, timestamp_us=1000 + t * 100000,
                     cameras={"CAM_F0": {"guide": str(guide)}}) for t in range(71)],
        files={str(source): digest(source), str(guide): digest(guide)},
    )
    prepared = tmp_path / "conditions.json"

    def seal():
        geometry_path.write_text(json.dumps(geometry))
        spec["files"][str(geometry_path)] = digest(geometry_path)
        prepared.write_text(json.dumps(spec))

    seal()
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"unit-test-only")
    return types.SimpleNamespace(
        spec=spec, geometry=geometry, prepared=prepared, seal=seal, checkpoint=checkpoint,
        output=tmp_path / "exp/pilot", guide=guide,
    )


def fake_native(monkeypatch, *, failure=None, malformed=None):
    calls = []

    class Tensor:
        def __init__(self, pixels):
            self.pixels = pixels

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def permute(self, *axes):
            self.pixels = self.pixels.transpose(axes)
            return self

        def numpy(self):
            return self.pixels

    class Model:
        def forward(self, batch, **kwargs):
            calls.append((batch, kwargs))
            if failure:
                raise RuntimeError(failure)
            pixels = (batch["guide_seq"] + 1) / 2
            if malformed == "missing_frame":
                pixels = pixels[:-1]
            elif malformed == "nan":
                pixels[0, 0, 0, 0] = np.nan
            elif malformed == "range":
                pixels[0, 0, 0, 0] = 1.1
            return Tensor(pixels)

    monkeypatch.setattr(diagnostic.runtime, "load_native", lambda _: Model())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        from_numpy=lambda x: x, cuda=types.SimpleNamespace(is_available=lambda: False)))
    return calls


def test_actual_dense_input_and_small_numeric_pose_difference(inputs):
    *_, error = diagnostic.validate_inputs(inputs.prepared, "CAM_F0")
    assert error["translation_m"] == pytest.approx(0.0015)


@pytest.mark.parametrize("fault", [
    "stale_pose", "capture_time", "execution_delta", "reference_identity", "branch",
    "timestamp", "padding", "unsealed_guide", "source_control",
])
def test_invalid_or_unpaired_conditions_rejected(inputs, fault):
    if fault == "stale_pose":
        inputs.geometry["frames"][0]["cameras"]["CAM_F0"]["camera_to_world"][0][3] = 0.01
    elif fault == "capture_time":
        inputs.geometry["frames"][0]["source_timestamp_us"] += 1
    elif fault == "execution_delta":
        inputs.geometry["frames"][0]["delta_xy_yaw"][0] = 0.1
    elif fault == "reference_identity":
        inputs.geometry["initial_reference_images"]["CAM_F0"] = "different-real-image.png"
    elif fault == "branch":
        inputs.geometry["branch_identity"] = "another-branch"
    elif fault == "timestamp":
        inputs.spec["frames"][1]["timestamp_us"] += 1
    elif fault == "padding":
        inputs.spec["frames"][-1] = inputs.spec["frames"][-2]
    elif fault == "unsealed_guide":
        inputs.spec["files"].pop(str(inputs.guide))
    elif fault == "source_control":
        inputs.geometry["source_control"] = True
    inputs.seal()
    with pytest.raises(ValueError):
        diagnostic.validate_inputs(inputs.prepared, "CAM_F0")


def test_pilot_uses_native_recipe_and_cannot_claim_full_or_paired_output(inputs, monkeypatch):
    calls = fake_native(monkeypatch)
    report = diagnostic.run(inputs.prepared, inputs.checkpoint, inputs.output)
    assert report["status"] == "generated_diagnostic_partial"
    assert report["kind"] != "conditioned_video_output"
    assert report["complete_conditioned_video"] is False
    assert report["trainable"] is report["visual_admitted"] is False
    assert report["causal_history_verified"] is report["gs_refinement_performed"] is False
    assert report["generated_ticks"] == list(range(25))
    assert report["omitted_ticks"] == list(range(25, 71))
    assert report["omitted_cameras"] == ["CAM_L0"]
    assert len(report["guide_inputs"]) == 25
    assert report["reference_comparison_count"] == 1
    assert report["frames"][0]["reference_comparison"]["exact_match"] is True
    assert report["frames"][0]["reference_comparison"]["novel_view_quality"] is False
    assert all(not f["reference_comparison"]["evaluated"] for f in report["frames"][1:])
    assert len(calls) == 1
    batch, kwargs = calls[0]
    assert kwargs == dict(scale=1.0, cond_indices=[0])
    assert batch["training_free_guidance"] is False
    assert batch["img_seq"].shape[0] == 25
    assert np.count_nonzero(batch["img_seq"][1:]) == 0
    assert report["native_recipe"]["steps"] == 50
    assert report["native_recipe"]["guidance_scale"] == 2.5
    assert report["native_recipe"]["seed"] == 23
    assert set(report["stage_seconds"]) == {
        "validate_conditions", "prepare_inputs", "verify_checkpoint", "load_native",
        "native_forward", "save_rgb_and_metrics",
    }
    assert len(list(inputs.output.glob("????-CAM_F0.png"))) == 25


@pytest.mark.parametrize("fault", ["failure", "missing_frame", "nan", "range"])
def test_native_failure_is_recorded_without_claiming_outputs(inputs, monkeypatch, fault):
    fake_native(monkeypatch, failure="native failure" if fault == "failure" else None,
                malformed=fault)
    with pytest.raises((ValueError, RuntimeError)):
        diagnostic.run(inputs.prepared, inputs.checkpoint, inputs.output)
    report = json.loads((inputs.output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["generated_ticks"] == []
    assert report["trainable"] is False
    assert report["stage_seconds"]["native_forward"] >= 0
    assert (inputs.output / "inputs/reference-native.png").exists()


def test_checkpoint_checksum_mismatch_prevents_native_load(inputs, monkeypatch):
    def forbidden(_):
        pytest.fail("A mismatched checkpoint reached native loading")

    monkeypatch.setattr(diagnostic.runtime, "load_native", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        diagnostic.run(inputs.prepared, inputs.checkpoint, inputs.output,
                       expected_checkpoint_sha256="0" * 64)


def test_clip_budget_cannot_silently_expand(inputs):
    with pytest.raises(ValueError, match="exactly one"):
        diagnostic.run(inputs.prepared, inputs.checkpoint, inputs.output, max_clips=2)
    assert not inputs.output.exists()


def test_supervisor_deadline_marks_its_own_worker_failed(inputs, monkeypatch):
    class Worker:
        pid = 12345
        terminated = False

        def wait(self, timeout=None):
            if not self.terminated:
                inputs.output.mkdir(parents=True)
                diagnostic._save(inputs.output, dict(status="native_forward", trainable=False))
                raise subprocess.TimeoutExpired("native pilot", timeout)
            return -15

        def terminate(self):
            self.terminated = True

    worker = Worker()
    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda _: worker)
    monkeypatch.setattr(sys, "argv", ["diagnostic.py", "--prepared", str(inputs.prepared),
                                    "--checkpoint", str(inputs.checkpoint),
                                    "--output", str(inputs.output), "--max-wall-seconds", "1"])
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 124
    report = json.loads((inputs.output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["terminated_worker_pid"] == worker.pid
    assert report["wall_time_budget_seconds"] == 1


@pytest.fixture
def gs_inputs(inputs):
    from PIL import Image

    mask = inputs.prepared.parent / "mask.png"
    Image.new("L", (4, 3), 255).save(mask)
    spec = dict(
        kind="streetcrafter_gs_matched_conditions", schema_version=1,
        diagnostic_only=True, trainable=False, branch_identity="sealed",
        camera_order=["CAM_F0"], source_conditions=str(inputs.prepared),
        geometry=inputs.spec["geometry"], bundle=inputs.spec["bundle"],
        initial_reference_images={"CAM_F0": str(inputs.guide)},
        frames=[dict(
            tick=tick, timestamp_us=1000 + tick * 100000,
            source_timestamp_us=1000 + tick * 100000,
            geometry_tick=tick, geometry_frame_source_timestamp_us=1000 + tick * 100000,
            cameras={"CAM_F0": dict(
                gs=str(inputs.guide), guide=str(inputs.guide),
                render_mask=str(mask), guide_mask=str(mask),
                calibration=dict(camera_to_world=np.eye(4)[:3].tolist(),
                                 distorted_intrinsic=np.eye(3).tolist(),
                                 distortion=[0] * 5, output_calibration_size=[8, 6]),
            )},
        ) for tick in range(25)],
        files={str(p): digest(p) for p in (
            inputs.prepared, inputs.spec["geometry"], inputs.spec["bundle"], inputs.guide, mask,
        )},
    )
    path = inputs.prepared.parent / "gs.json"

    def seal():
        path.write_text(json.dumps(spec))

    seal()
    return types.SimpleNamespace(spec=spec, path=path, seal=seal)


def test_actual_gs_sequence_uses_native_refinement_not_stale_raw_photo(
    inputs, gs_inputs, monkeypatch,
):
    calls = fake_native(monkeypatch)
    report = diagnostic.run(inputs.prepared, inputs.checkpoint, inputs.output,
                            gs_sequence=gs_inputs.path)
    batch, kwargs = calls[0]
    assert kwargs == dict(scale=0.3, cond_indices=[0])
    assert batch["training_free_guidance"] is True
    assert batch["img_mask_seq"].shape == (25, 1, 3, 4)
    assert batch["guide_mask_seq"].shape == (25, 1, 3, 4)
    assert np.count_nonzero(batch["img_seq"][1:]) != 0
    assert report["native_recipe"]["active_denoising_steps"] == 15
    assert report["gs_refinement_performed"] is True
    assert report["native_mask_blending_active"] is False
    assert report["raw_photo_reference_comparison"] is False
    assert report["reference_comparison_count"] == 0
    assert report["gs_change_comparison_count"] == 25
    assert len(report["gs_inputs"]) == 25
    assert all(not row["reference_comparison"]["evaluated"] for row in report["frames"])
    assert all(not row["gs_input_comparison"]["quality_score"] for row in report["frames"])


@pytest.mark.parametrize("fault", [
    "branch", "wrong_condition", "missing_tick", "timestamp", "unsealed_render",
    "wrong_initial", "nonrigid_camera", "wrong_aspect", "padded_capture", "source_time",
    "geometry_link",
])
def test_gs_refinement_rejects_unmatched_inputs(inputs, gs_inputs, monkeypatch, fault):
    spec = gs_inputs.spec
    frame = spec["frames"][0]["cameras"]["CAM_F0"]
    if fault == "branch":
        spec["branch_identity"] = "different"
    elif fault == "wrong_condition":
        spec["source_conditions"] = "other-condition.json"
    elif fault == "missing_tick":
        spec["frames"] = spec["frames"][1:]
    elif fault == "timestamp":
        spec["frames"][0]["timestamp_us"] += 1
    elif fault == "unsealed_render":
        spec["files"].pop(str(inputs.guide))
    elif fault == "wrong_initial":
        spec["initial_reference_images"]["CAM_F0"] = "stale-photo.png"
    elif fault == "nonrigid_camera":
        frame["calibration"]["camera_to_world"][0][0] = 2
    elif fault == "wrong_aspect":
        frame["calibration"]["output_calibration_size"] = [8, 5]
    elif fault == "padded_capture":
        spec["frames"][1]["source_timestamp_us"] = spec["frames"][0]["source_timestamp_us"]
    elif fault == "source_time":
        spec["frames"][0]["source_timestamp_us"] += 60001
    elif fault == "geometry_link":
        spec["frames"][0]["geometry_frame_source_timestamp_us"] += 1
    gs_inputs.seal()

    def forbidden(_):
        pytest.fail("Unmatched GS receipt reached native loading")

    monkeypatch.setattr(diagnostic.runtime, "load_native", forbidden)
    with pytest.raises(ValueError):
        diagnostic.run(inputs.prepared, inputs.checkpoint, inputs.output,
                       gs_sequence=gs_inputs.path)


@pytest.mark.parametrize("offset_us", [-55000, 55000])
def test_gs_capture_within_60ms_but_rounding_to_neighbor_is_rejected(inputs, gs_inputs, offset_us):
    # Both pass the wider source-time tolerance, yet denote a neighboring native
    # tick. Checking only <=60ms would silently permit the wrong physical frame.
    gs_inputs.spec["frames"][5]["source_timestamp_us"] += offset_us
    gs_inputs.seal()
    _, geometry, bundle, _ = diagnostic.validate_inputs(inputs.prepared, "CAM_F0")
    with pytest.raises(ValueError, match="rounds to another physical tick"):
        diagnostic.validate_gs_inputs(gs_inputs.path, inputs.prepared, "CAM_F0", geometry, bundle)


@pytest.mark.parametrize("offset_us", [-49000, 49000])
def test_gs_capture_rounding_to_own_tick_is_supported(inputs, gs_inputs, offset_us):
    gs_inputs.spec["frames"][5]["source_timestamp_us"] += offset_us
    gs_inputs.seal()
    _, geometry, bundle, _ = diagnostic.validate_inputs(inputs.prepared, "CAM_F0")
    result = diagnostic.validate_gs_inputs(gs_inputs.path, inputs.prepared, "CAM_F0", geometry, bundle)
    assert result["frames"][5]["tick"] == 5

