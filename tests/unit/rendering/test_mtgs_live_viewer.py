"""Publishing requires aligned, local evidence and keeps source data inert."""

import json

import pytest

from driversi.rendering.gaussian.mtgs_live_viewer import live_html, write_session_viewer


@pytest.fixture
def session(tmp_path):
    (tmp_path / "frame 0.png").write_bytes(b"renderer owns PNG verification")
    manifest = {"schema_version": 1, "kind": "mtgs_world_session", "status": "completed",
                "trainable": False, "source_log": "example", "limits": ["No learned policy"],
                "frames": [{"tick": 0, "time_seconds": 0.0, "actors": [],
                            "cameras": [{"camera_id": "CAM_F0", "image": "frame 0.png"}]}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_live_token_is_inert_even_if_it_contains_html():
    token = '</script><img src="x" onerror="alert(1)">'
    html = live_html(token)
    assert token not in html
    encoded = html.split("const CONFIG=", 1)[1].split(",$=", 1)[0]
    assert json.loads(encoded)["token"] == token
    assert "X-DriveRSI-Token" in html
    assert "尚不能作为学习型视觉 Self-Play" in html
    assert "外部策略输入若未记录" in html


@pytest.mark.parametrize("token", ["", None, "x" * 513])
def test_live_page_requires_a_token(token):
    with pytest.raises(ValueError, match="session token"):
        live_html(token)


def test_recorded_session_preserves_reference_images_and_source_identity(session):
    path, manifest = session
    frame = manifest["frames"][0]
    frame["cameras"][0].update(source="frame 0.png", pinhole="frame 0.png")
    manifest["source_log"] = '</script><svg onload="alert(1)">'
    write_manifest(path, manifest)
    before = path.read_bytes()
    result = write_session_viewer(path)
    html = result.read_text(encoding="utf-8")
    assert manifest["source_log"] not in html
    encoded = html.split("const CONFIG=", 1)[1].split(",$=", 1)[0]
    embedded = json.loads(encoded)
    assert embedded["mode"] == "recorded"
    camera = embedded["manifest"]["frames"][0]["cameras"][0]
    assert camera["url"] == camera["source_url"] == camera["pinhole_url"] == "frame%200.png"
    assert embedded["manifest"]["source_log"] == manifest["source_log"]
    assert path.read_bytes() == before
    assert "采集照片是原日志参考，干预后不是真值" in html
    with pytest.raises(FileExistsError):
        write_session_viewer(path)


def test_recorded_lens_support_mask_remains_available_with_camera_identity(session):
    path, manifest = session
    camera = manifest["frames"][0]["cameras"][0]
    camera.update(valid="frame 0.png", owner_actor_id="ego", source_lens_valid_fraction=.98)
    write_manifest(path, manifest)
    html = write_session_viewer(path).read_text(encoding="utf-8")
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    camera = embedded["manifest"]["frames"][0]["cameras"][0]
    assert camera["valid_url"] == "frame%200.png"
    assert camera["owner_actor_id"] == "ego"
    assert camera["source_lens_valid_fraction"] == .98
    assert "不同相机不代表第二辆策略车的视角" in html
    assert "该比例不衡量新视角重建质量" in html


def test_recorded_support_mask_cannot_point_outside_session(session):
    path, manifest = session
    manifest["frames"][0]["cameras"][0]["valid"] = "../outside.png"
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="image"):
        write_session_viewer(path)


@pytest.mark.parametrize("asset", ["../outside.png", "/tmp/image.png", "C:/image.png",
                                  "images\\frame.png", "https://host/image.png", "missing.png"])
def test_recorded_session_rejects_missing_or_external_images(session, asset):
    path, manifest = session
    manifest["frames"][0]["cameras"][0]["image"] = asset
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="image"):
        write_session_viewer(path)
    assert not (path.parent / "viewer.html").exists()


@pytest.mark.parametrize("change", [
    {"trainable": True}, {"status": "running"}, {"status": "failed"},
    {"kind": "mtgs_renderer_probe"}, {"frames": []},
])
def test_recorded_session_cannot_publish_incomplete_or_different_evidence(session, change):
    path, manifest = session
    manifest.update(change)
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="completed non-training"):
        write_session_viewer(path)


@pytest.mark.parametrize("tick,time", [(0, .2), (2, 0.0), (-1, .1), (1, float("nan"))])
def test_recorded_frames_cannot_repeat_or_reverse_the_time_axis(session, tick, time):
    path, manifest = session
    second = {**manifest["frames"][0], "tick": tick, "time_seconds": time}
    manifest["frames"].append(second)
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="strictly increase"):
        write_session_viewer(path)


def test_duplicate_camera_identity_is_rejected(session):
    path, manifest = session
    manifest["frames"][0]["cameras"] *= 2
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="Camera IDs"):
        write_session_viewer(path)


def test_reset_generations_keep_independent_playback_timelines(session):
    path, manifest = session
    first = manifest["frames"][0]
    manifest["frames"] = [
        {**first, "generation": 0, "tick": 0, "time_seconds": 0},
        {**first, "generation": 0, "tick": 6, "time_seconds": .6},
        {**first, "generation": 1, "tick": 0, "time_seconds": 0},
        {**first, "generation": 1, "tick": 2, "time_seconds": .2},
    ]
    write_manifest(path, manifest)
    html = write_session_viewer(path).read_text(encoding="utf-8")
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    assert embedded["manifest"]["viewer_generations"] == [
        {"id": 0, "frame_indices": [0, 1]}, {"id": 1, "frame_indices": [2, 3]},
    ]
    assert 'id="generation"' in html


def test_reset_cannot_hide_a_nonmonotonic_frame_in_an_earlier_generation(session):
    path, manifest = session
    first = manifest["frames"][0]
    manifest["frames"] = [
        {**first, "generation": 0, "tick": 0, "time_seconds": 0},
        {**first, "generation": 1, "tick": 0, "time_seconds": 0},
        {**first, "generation": 0, "tick": 0, "time_seconds": 0},
    ]
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="strictly increase"):
        write_session_viewer(path)


def test_virtual_owner_has_no_false_source_image(session):
    path, manifest = session
    camera = manifest["frames"][0]["cameras"][0]
    camera.update(camera_id="lead__virtual_front", owner_actor_id="lead",
                  virtual=True, source=None, pinhole="frame 0.png", valid="frame 0.png",
                  support={"trainable": False, "quality": "unverified"})
    write_manifest(path, manifest)
    html = write_session_viewer(path).read_text(encoding="utf-8")
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    published = embedded["manifest"]["frames"][0]["cameras"][0]
    assert published["source"] is None
    assert published.get("source_url") is None
    assert published["owner_actor_id"] == "lead"
    assert published["virtual"] is True
    assert published["support"]["trainable"] is False
    assert published["url"] == published["pinhole_url"] == "frame%200.png"


def test_height_binding_provenance_is_preserved_and_cannot_inject_html(session):
    path, manifest = session
    camera = manifest["frames"][0]["cameras"][0]
    reason = '</script><img src="x" onerror="alert(1)">'
    binding = {"mode": "reference-track", "actor_id": "lead", "supported": False,
               "reason": reason, "correction_m": 0.0,
               "source_projection": {"distance_m": 1.0},
               "target_projection": {"distance_m": 12.0}}
    camera.update(height_binding=binding, scene_height_supported=False)
    write_manifest(path, manifest)
    before = path.read_bytes()
    html = write_session_viewer(path).read_text(encoding="utf-8")
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    published = embedded["manifest"]["frames"][0]["cameras"][0]
    assert published["height_binding"] == binding
    assert published["scene_height_supported"] is False
    assert reason not in html
    assert path.read_bytes() == before
    assert "高程查询不支持：保留源高度" in html
    assert "这是有限轨迹代理，不是地形网格" in html
    assert "不能证明真实横坡、物理配准或新视角质量" in html
    assert "至少一个世界车辆的高程查询失败" in html
    assert "即使当前相机主体查询成功" in html


def test_legacy_height_information_is_not_synthesized_as_supported(session):
    path, manifest = session
    html = write_session_viewer(path).read_text(encoding="utf-8")
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    published = embedded["manifest"]["frames"][0]["cameras"][0]
    assert "height_binding" not in published
    assert "scene_height_supported" not in published
    assert "高程支持信息未提供；不能据此判断车辆是否贴地" in html
    assert "其他世界车辆的高程支持状态未提供" in html


def test_height_binding_keeps_owner_query_separate_from_world_support(session):
    path, manifest = session
    camera = manifest["frames"][0]["cameras"][0]
    camera.update(height_binding={"mode": "reference-track", "actor_id": "ego",
                                  "supported": True, "reason": "supported",
                                  "correction_m": -0.25},
                  scene_height_supported=False)
    write_manifest(path, manifest)
    html = write_session_viewer(path).read_text(encoding="utf-8")
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])
    published = embedded["manifest"]["frames"][0]["cameras"][0]
    assert published["height_binding"]["supported"] is True
    assert published["height_binding"]["correction_m"] == -.25
    assert published["scene_height_supported"] is False
    assert "这不是图像质量验收通过" in html
    assert "源时间高度不保证反事实车辆贴地" in html


def test_failed_maneuver_render_preserves_source_qualification(session):
    path, manifest = session
    manifest["challenge_replay"] = {"candidate_id": "challenge", "diagnostic_failure": True,
                                    "maneuver_validation_pass": False,
                                    "rendered_rgb_used_for_replay_actions": False}
    write_manifest(path, manifest)
    html = write_session_viewer(path).read_text()
    embedded = json.loads(html.split("const CONFIG=", 1)[1].split(",$=", 1)[0])["manifest"]
    assert embedded["challenge_replay"] == manifest["challenge_replay"]
    assert 'id="trajectory-source"' in html
    assert "机动未通过验收：仅诊断渲染" in html
