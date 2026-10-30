"""Cache invariants for the isolated rendering comparison."""

import copy
import json
from types import SimpleNamespace

import pytest

from driversi.rendering.diagnostics.render_benchmark import (
    SourceSession,
    compare_source,
    replay_adapter,
)


@pytest.fixture
def session(tmp_path):
    paths = [tmp_path / n for n in ("asset.ckpt", "metadata.pkl", "producer.py")]
    for p in paths:
        p.write_bytes(p.name.encode())
    source = dict(
        token="source-a",
        timestamp=123,
        cams={
            c: dict(data_path=c + ".png", v_adjust=1.0)
            for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
        },
    )
    for camera in source["cams"].values():
        (tmp_path / camera["data_path"]).write_bytes(b"image")
    renderer = SimpleNamespace(binding=SimpleNamespace(resolve=lambda _: source))
    obj = SourceSession(renderer, paths[0], paths[1], [paths[2]])
    return (
        obj,
        source,
        tmp_path,
        dict(frames=[dict(tick=0, source_token="source-a")], output_image_size=[960, 540]),
    )


def test_source_cache_ignores_counterfactual_motions(session):
    obj, _, root, report = session
    a = obj.contract(report, root)
    b = copy.deepcopy(report)
    b["frames"][0]["motions"] = {"ego": [10.0, 1.0, 0.1]}
    assert obj.contract(b, root) == a


@pytest.mark.parametrize("change", ["time", "exposure", "pixels", "size"])
def test_source_cache_binds_real_render_conditions(session, change):
    obj, source, root, report = session
    before = obj.contract(report, root)
    if change == "time":
        source["timestamp"] += 1
    elif change == "exposure":
        source["cams"]["CAM_F0"]["v_adjust"] = 1.1
    elif change == "pixels":
        (root / "CAM_F0.png").write_bytes(b"different")
    else:
        report["output_image_size"] = [1920, 1080]
    assert obj.contract(report, root) != before


@pytest.mark.parametrize("name", ["asset.ckpt", "metadata.pkl", "producer.py"])
def test_source_session_rejects_changed_asset_or_code(session, name):
    obj, _, root, report = session
    (root / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset/code changed"):
        obj.contract(report, root)


def test_source_session_rejects_changed_report(session):
    from driversi.agents.backends.native import digest

    obj, _, root, _ = session
    path = root / "cached-report.json"
    path.write_text(json.dumps({"files": {}, "score": 27.5}))
    obj.entries["key"] = (path, digest(path))
    assert obj.cached_report("key")[1]["score"] == 27.5
    path.write_text(json.dumps({"files": {}, "score": 99.0}))
    with pytest.raises(ValueError, match="report changed"):
        obj.cached_report("key")


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_camera_to_world", [[1, 2, 3, 5]]),
        ("source_appearance_timestamp_us", 999),
        ("source_motions", {"ego": [1, 0, 0]}),
        ("exposure_factor", 2.0),
        ("real_source_image", "/changed-source.png"),
    ],
)
def test_source_parity_rejects_changed_evidence_with_same_pixels(tmp_path, field, value):
    from PIL import Image

    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image)
    frame = dict(
        tick=0,
        camera="CAM_F0",
        source_token="s",
        rendered_source_image=str(image),
        measured_psnr_db=28.0,
        source_camera_to_world=[[1, 2, 3, 4]],
        source_appearance_timestamp_us=123,
        source_motions={},
        exposure_factor=1.0,
        real_source_image="/source.png",
    )
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps({"frames": [frame]}))
    frame[field] = value
    new.write_text(json.dumps({"frames": [frame]}))
    result = compare_source(old, new)
    assert result["pixels_exact"] and result["max_psnr_delta"] == 0
    assert not result["evidence_exact"]


def test_adapter_never_reuses_other_branch_motion():
    calls = []

    class Renderer:
        binding = SimpleNamespace(resolve=lambda _: {"token": "source"})

        def render_many(self, source, cameras, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {camera: (kwargs["motions"]["ego"], {}) for camera in cameras}

    adapted = replay_adapter(Renderer())
    for offset in (1, 2, 1):
        for camera in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            assert adapted.render("source", camera, motions={"ego": offset})[0] == offset
    assert len(calls) == 3


def test_adapter_refreshes_consumed_camera():
    calls = []

    class Renderer:
        binding = SimpleNamespace(resolve=lambda _: {"token": "source"})

        def render_many(self, source, cameras, **kwargs):
            calls.append(1)
            return {c: (len(calls), {}) for c in cameras}

    adapted = replay_adapter(Renderer())
    assert adapted.render("source", "CAM_F0")[0] == 1
    assert adapted.render("source", "CAM_F0")[0] == 2
