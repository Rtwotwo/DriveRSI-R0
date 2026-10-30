"""Visual exporter checks actual pixels, clocks and hostile source labels."""

import base64
import json
import re
import struct
import zlib

import numpy as np
import pytest

from driversi.rendering.visual.visual_replay import export_visual_replay


def record():
    actor = '</script><img src=x onerror="alert(1)">'
    metadata = {
        "schema_version": 1, "kind": "visual_engineering_rollout", "modality": "primitive_rgb",
        "learning_status": "diagnostic_policy_no_model_update", "action_space": "jerk_discrete_12",
        "steps": 3, "dt_seconds": .1,
        "configuration": {"rig": {"cameras": [{"camera_id": "front", "width": 16, "height": 16}]}},
        "scene_metadata": {"source_timing": {"interval_seconds": {"median": .5}}},
        "frame_identity": [{"episode": "episode", "branch": actor, "generation": 0, "tick": t}
                           for t in (0, 0, 2)],
    }
    rgb = np.zeros((3, 2, 2, 1, 16, 16, 3), np.uint8)
    rgb[:, :, -1] = [10, 20, 30]
    rgb[2, :, 0] = [10, 20, 30]
    rgb[2, :, -1] = [70, 150, 225]
    ticks = np.asarray([[[-1, 0], [-1, 0]], [[-1, 0], [-1, 0]], [[0, 2], [0, 2]]])
    data = {"actor_ids": np.asarray([actor, "b"]), "physical_actor_ids": np.asarray(["replay", actor, "b"]),
            "states": np.zeros((4, 3, 13)), "motion": np.zeros((4, 3, 3)), "rgb": rgb,
            "history_ticks": ticks, "history_valid": (ticks >= 0)[..., None],
            "telemetry": np.zeros((3, 2, 4)), "navigation": np.zeros((3, 2, 2)),
            "observation_tick": np.asarray([0, 0, 2]),
            "newly_decided": np.asarray([True, False, True]),
            "policy_actions": np.asarray([[0, 0], [0, 0], [4, 4]]),
            "executed_actions": np.asarray([[0, 0], [5, 0], [4, 4]]),
            "memories": np.zeros((4, 2, 1))}
    for name in ("overrides", "trainable", "valid", "terminated", "truncated", "collision", "offroad", "goal"):
        data[name] = np.zeros((3, 2), bool)
    data["overrides"][1, 0] = True
    data["valid"][:] = True
    data["truncated"][-1] = True
    return metadata, data


def save(tmp_path, metadata, data):
    source = tmp_path / "visual.npz"
    np.savez_compressed(source, metadata=json.dumps(metadata), **data)
    return source


def payload(html):
    return json.loads(re.search(r"const D=(.*?);\nconst M", html, re.S).group(1))


def decode_png(uri):
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed = 8, b""
    while offset < len(raw):
        size = struct.unpack(">I", raw[offset:offset+4])[0]
        kind = raw[offset+4:offset+8]
        data = raw[offset+8:offset+8+size]
        crc = struct.unpack(">I", raw[offset+8+size:offset+12+size])[0]
        assert zlib.crc32(kind+data) & 0xffffffff == crc
        if kind == b"IHDR":
            width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            assert (bits, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
        if kind == b"IDAT":
            compressed += data
        offset += 12+size
    rows = np.frombuffer(zlib.decompress(compressed), np.uint8).reshape(height, 1+3*width)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, 3)


def test_actual_png_frames_held_clock_and_escaped_labels(tmp_path):
    metadata, arrays = record()
    output = tmp_path / "viewer.html"
    report = export_visual_replay(save(tmp_path, metadata, arrays), output)
    assert report["frames"] == 3 and report["unique_images"] == 2
    assert report["actors"] == 2 and report["cameras"] == 1
    html = output.read_text(encoding="utf-8")
    data = payload(html)
    assert data["observation_tick"] == [0, 0, 2]
    assert data["frame_images"][0] == data["frame_images"][1]
    assert len(data["frame_images"]) == 3  # no invented successor camera
    assert len(data["states"]) == 4  # final world state is still retained
    for step in range(3):
        for actor in range(2):
            image = decode_png(data["images"][data["frame_images"][step][actor][0]])
            np.testing.assert_array_equal(image, arrays["rgb"][step, actor, -1, 0])
    assert html.count("</script>") == 1
    assert str(arrays["actor_ids"][0]) not in html
    assert data["actor_ids"] == arrays["actor_ids"].tolist()
    assert "textContent=actor" in html
    assert "物理输入 tick" in html and "图像观测 tick" in html
    assert "本页展示决策前输入" in html and "诊断策略，未进行模型学习" in html


@pytest.mark.parametrize("field,change", [
    ("rgb", lambda x: x[:, :, 0]),
    ("history_valid", lambda x: x.astype(np.int8)),
    ("history_ticks", lambda x: np.full_like(x, 100)),
    ("observation_tick", lambda x: np.asarray([0, 1, 2])),
    ("policy_actions", lambda x: np.full_like(x, 12)),
    ("overrides", lambda x: np.zeros_like(x)),
    ("trainable", lambda x: np.ones_like(x)),
    ("physical_actor_ids", lambda x: np.asarray(["other", "a", "b"])),
])
def test_rejects_invalid_shapes_clocks_and_training_claims(tmp_path, field, change):
    metadata, arrays = record()
    arrays[field] = change(arrays[field])
    with pytest.raises(ValueError):
        export_visual_replay(save(tmp_path, metadata, arrays), tmp_path / "viewer.html")
    assert not (tmp_path / "viewer.html").exists()


def test_held_observation_cannot_silently_change_pixels(tmp_path):
    metadata, arrays = record()
    arrays["rgb"][1, 0, -1, 0, 0, 0] = [0, 0, 0]
    with pytest.raises(ValueError, match="Held observation"):
        export_visual_replay(save(tmp_path, metadata, arrays), tmp_path / "viewer.html")


def test_preserves_existing_output(tmp_path):
    output = tmp_path / "viewer.html"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_visual_replay(tmp_path / "missing.npz", output)
    assert output.read_text(encoding="utf-8") == "keep"
