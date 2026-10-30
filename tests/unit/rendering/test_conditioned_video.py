"""Contracts that prevent generated pixels from becoming unverified supervision."""

import copy
import json

import numpy as np
import pytest

from driversi.agents.backends.native import digest
from driversi.rendering.video import validation
from driversi.rendering.video.bundle import CAMERAS, FUTURE, HISTORY, clip_ranges
from driversi.rendering.video.geometry import SIGNS, rigid
from driversi.rsi.plugins.conditioned_video import execute_stored, prepare
from driversi.rendering.video.worlddreamer.prepare import native_box_corners


def test_clips_use_only_actual_contiguous_ticks():
    clips = clip_ranges(range(71))
    assert len(clips) == 4
    assert all(len(row) == 25 and len(set(row)) == 25 for row in clips)
    assert set().union(*map(set, clips)) == set(range(71))
    assert clips[-1][-1] == 70


@pytest.mark.parametrize("ticks", [range(24), [*range(30), 32], [0] * 30])
def test_clips_refuse_padding_and_missing_physics(ticks):
    with pytest.raises(ValueError, match="actual physics"):
        clip_ranges(ticks)


@pytest.mark.parametrize("yaw", [0, 0.7, -2.1])
def test_native_boxes_keep_order_and_separate_layout_origin(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    centre, dims = np.array([4, 2, -0.5]), np.array([4.2, 1.8, 1.4])
    physical = (SIGNS * dims / 2) @ rotation.T + centre
    layout, tokens = native_box_corners([dict(corners_ego=physical, dimensions=dims)])
    # Independent native row-vector LiDAR box convention from pinned converter.
    native_yaw = -yaw - np.pi
    a, b = np.cos(native_yaw), np.sin(native_yaw)
    expected = (SIGNS * dims[[1, 0, 2]] / 2) @ np.array([[a, -b, 0], [b, a, 0], [0, 0, 1]]) + [
        -centre[1],
        centre[0],
        centre[2],
    ]
    np.testing.assert_allclose(layout[0], expected, atol=1e-6)
    np.testing.assert_allclose(tokens[0], expected + [0, 0, dims[2] / 2], atol=1e-6)


def test_native_empty_population_is_supported():
    a, b = native_box_corners([])
    assert a.shape == b.shape == (0, 8, 3)


def test_nonrigid_camera_refused():
    bad = np.eye(4)[:3].copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        rigid(bad)


@pytest.fixture
def video(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image = tmp_path / "frame.png"
    Image.new("RGB", (400, 224), "white").save(image)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (1920, 1080), "black").save(raw)
    frames = []
    geometry_frames = []
    for tick in HISTORY + FUTURE:
        frames.append(
            dict(
                tick=tick,
                timestamp_us=1000 + tick * 100000,
                cameras={
                    cam: dict(path=str(image), sha256=digest(image), size=[400, 224])
                    for cam in CAMERAS
                },
            )
        )
        geometry_frames.append(
            dict(tick=tick, cameras={cam: dict(source_image=str(raw)) for cam in CAMERAS})
        )
    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps(
            dict(
                branch_identity="same",
                source_control=False,
                bundle="fixture",
                frames=geometry_frames,
                files={str(raw): digest(raw)},
            )
        )
    )
    condition = tmp_path / "conditions.json"
    condition.write_text(
        json.dumps(
            dict(
                kind="worlddreamer_native_conditions",
                geometry=str(geometry),
                branch_identity="same",
                files={str(geometry): digest(geometry)},
            )
        )
    )
    bundle = dict(branch_identity="same", source_start_timestamp_us=1000)
    monkeypatch.setattr(validation, "validate_bundle", lambda _: bundle)
    report = dict(
        kind="conditioned_video_output",
        status="generated_diagnostic",
        backend="worlddreamer",
        native_recipe={"steps": 50, "resolution": [224, 400]},
        branch_identity="same",
        conditioning_report=str(condition),
        conditioning_sha256=digest(condition),
        source_control=False,
        camera_order=list(CAMERAS),
        history_ticks=list(HISTORY),
        future_ticks=list(FUTURE),
        frames=frames,
        trainable=False,
        visual_admitted=False,
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path, report, geometry, condition, image


def test_output_is_complete_but_not_training_admission(video):
    path, *_ = video
    value, _, _ = validation.validate_output(path)
    assert value["trainable"] is False


@pytest.mark.parametrize(
    "fault", ["timestamp", "missing_camera", "future_padding", "self_admission", "identity"]
)
def test_video_chain_refuses_semantic_mismatch(video, fault):
    path, report, *_ = video
    if fault == "timestamp":
        report["frames"][-1]["timestamp_us"] += 1
    if fault == "missing_camera":
        report["frames"][0]["cameras"].pop(CAMERAS[0])
    if fault == "future_padding":
        report["frames"][-1] = copy.deepcopy(report["frames"][-2])
    if fault == "self_admission":
        report["trainable"] = True
    if fault == "identity":
        report["branch_identity"] = "foreign"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validation.validate_output(path)


def test_changed_pixels_refused(video):
    path, _, _, _, image = video
    image.write_bytes(b"different")
    with pytest.raises(ValueError, match="pixels changed"):
        validation.validate_output(path)


def test_changed_geometry_refused(video):
    path, _, geometry, *_ = video
    geometry.write_text(geometry.read_text() + " ")
    with pytest.raises(ValueError):
        validation.validate_output(path)


def test_source_psnr_is_not_novel_view_quality(video):
    path, report, geometry, condition, _ = video
    original = validation.validate_output(path)
    source_geometry = json.loads(geometry.read_text())
    source_geometry["source_control"] = True
    report = copy.deepcopy(report)
    report["source_control"] = True
    # Isolate metric interpretation; actual file integrity is tested above.
    source = (report, source_geometry, {})
    old = validation.validate_output
    validation.validate_output = lambda p: original if str(p) == str(path) else source
    source_path = path.parent / "source.json"
    source_path.write_text("{}")
    try:
        result = validation.source_comparison(path, source_path)
    finally:
        validation.validate_output = old
    assert result["source_psnr_mean_db"] == 0
    assert result["novel_view_psnr_used"] is False
    assert result["admitted"] is False
    assert "source_reconstruction_below_unchanged_27db_gate" in result["rejection_reasons"]


def test_renderer_requires_bounded_wall_time(tmp_path):
    config = dict(
        backend="worlddreamer",
        python="python",
        conditions="c",
        weights="w",
        gpu=0,
        max_wall_seconds=1801,
    )
    with pytest.raises(ValueError, match="budget"):
        prepare(config, tmp_path, tmp_path / "registry")


def test_stored_branch_rechecked_before_work(monkeypatch, tmp_path):
    from driversi.rsi.plugins import chainflow_synthetic

    monkeypatch.setattr(
        chainflow_synthetic, "_stored_stage", lambda *args: ({}, {}, {"identity": "changed"})
    )
    with pytest.raises(ValueError, match="changed after registration"):
        execute_stored(
            dict(branch=dict(source_task="t", source_stage="s", identity="frozen")),
            {},
            tmp_path,
            tmp_path / "r",
        )


def test_png_header_checks_crc_without_optional_image_packages(tmp_path):
    import struct
    import zlib

    header = b"IHDR" + struct.pack(">IIBBBBB", 400, 224, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + header
        + struct.pack(">I", zlib.crc32(header))
    )
    path = tmp_path / "header.png"
    path.write_bytes(blob)
    assert validation.png_size(path) == [400, 224]
    path.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    with pytest.raises(ValueError, match="Corrupt"):
        validation.png_size(path)


def test_quality_worker_cannot_promote_its_own_images(monkeypatch, tmp_path):
    from driversi.rsi.plugins import chainflow_synthetic, conditioned_video

    monkeypatch.setattr(conditioned_video, "check_files", lambda _: None)
    monkeypatch.setattr(conditioned_video, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        chainflow_synthetic,
        "_stored_stage",
        lambda *args: ({}, {"artifacts": {"render_report": "render.json"}}, {"identity": "frozen"}),
    )

    def fake_process(command, output, *args):
        (output / "quality.json").write_text(json.dumps({"admitted": True}))

    monkeypatch.setattr(conditioned_video, "run_process", fake_process)
    config = dict(
        files={},
        references={
            side: dict(source_task=side, source_stage="render", identity="frozen")
            for side in ("novel", "source")
        },
        python="python",
        max_wall_seconds=10,
    )
    with pytest.raises(ValueError, match="cannot authorize"):
        conditioned_video.execute_quality(
            config, {}, tmp_path / "exp/quality", tmp_path / "registry"
        )
