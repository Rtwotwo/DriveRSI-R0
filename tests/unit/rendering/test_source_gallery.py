"""Source gallery provenance, time-axis and local-asset boundaries."""

import json
import re

import pytest

from driversi.rendering.visual.source_gallery import export_source_gallery


@pytest.fixture
def gallery_bundle(tmp_path):
    (tmp_path / "camera.jpg").write_bytes(b"fixture-image")
    frames = []
    for index, microseconds in enumerate((0, 480_000, 1_130_000)):
        frames.append({
            "source_index": 15 + index, "timestamp_us": 1_610_000_000_000_000 + microseconds,
            "time_seconds": microseconds / 1e6, "ego_pose": [0, 0, 0],
            "objects": [{"id": "car-1", "kind": "vehicle", "pose": [10, 0, 0, 0],
                         "dimensions": [4, 2, 1.7]}],
            "cameras": [{"id": "CAM_F0", "image": "camera.jpg", "width": 1920,
                         "height": 1080, "available": True, "note": "source image",
                         "projection": [{"id": "car-1", "kind": "vehicle",
                                         "segments": [[100, 100, 120, 100]]}]},
                        {"id": "CAM_B0", "image": None, "width": None, "height": None,
                         "available": False, "note": "not provided", "projection": []}],
        })
    return {"schema_version": 1, "kind": "source_camera_gallery_not_simulation",
            "timing_scope": "frame/LiDAR timestamp; camera exposure timestamps unavailable",
            "scenes": [{"id": "source-1", "label": "原始场景", "log_name": "source-log",
                        "map_name": "test-map", "source_rate_hz": 2, "frames": frames,
                        "map": {"bounds": [-20, -20, 20, 20],
                                "layers": {
                                    "dubins_nodes": {"availability": "available",
                                                     "feature_count": 1},
                                    "stop_sign": {"availability": "unavailable",
                                                  "feature_count": 0},
                                    "crosswalk": {"availability": "loaded_empty",
                                                  "feature_count": 0}},
                                "features": [{"layer": "dubins_nodes",
                                              "geometry": {"type": "Point",
                                                           "coordinates": [10, 0]}}]}}]}


def _embedded(path):
    html = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="source-data" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_preserves_source_timing_camera_slots_and_original_assets(tmp_path, gallery_bundle):
    output = tmp_path / "index.html"
    original = (tmp_path / "camera.jpg").read_bytes()
    result = export_source_gallery(gallery_bundle, output)
    data = _embedded(output)
    frames = data["scenes"][0]["frames"]
    assert [f["time_seconds"] for f in frames] == [0, 0.48, 1.13]
    assert [f["timestamp_us"] for f in frames] == [
        1_610_000_000_000_000, 1_610_000_000_480_000, 1_610_000_001_130_000]
    assert frames[1]["cameras"][1]["available"] is False
    assert frames[1]["cameras"][1]["image"] is None
    source_map = data["scenes"][0]["map"]
    assert source_map["layers"]["stop_sign"]["availability"] == "unavailable"
    assert source_map["layer_labels"] == {
        "dubins_nodes": "已提供 · 1 个要素", "stop_sign": "未提供", "crosswalk": "已加载但本区域为空"}
    assert data["timing_scope"] == gallery_bundle["timing_scope"]
    assert "未提供逐相机曝光时刻，不认证硬件同步" in output.read_text(encoding="utf-8")
    assert result["frames"] == 3
    assert (tmp_path / "camera.jpg").read_bytes() == original
    with pytest.raises(FileExistsError):
        export_source_gallery(gallery_bundle, output)


def test_source_labels_cannot_close_script_or_inject_html(tmp_path, gallery_bundle):
    attack = '</script><img src=x onerror="alert(1)"><script>'
    gallery_bundle["scenes"][0]["label"] = attack
    gallery_bundle["scenes"][0]["frames"][0]["cameras"][0]["note"] = attack
    output = tmp_path / "index.html"
    export_source_gallery(gallery_bundle, output)
    html = output.read_text(encoding="utf-8")
    assert attack not in html
    assert _embedded(output)["scenes"][0]["label"] == attack
    assert 'type="checkbox"> 显示源 3D 框' in html  # Source pixels stay unannotated initially.
    assert "不能直接计算 PSNR / SSIM" in html


def test_preserves_nested_geometry_collection_in_provided_layer(tmp_path, gallery_bundle):
    geometry = {"type": "GeometryCollection", "geometries": [
        {"type": "Point", "coordinates": [10, 0]},
        {"type": "GeometryCollection", "geometries": [
            {"type": "LineString", "coordinates": [[0, 0], [5, 5]]},
            {"type": "GeometryCollection", "geometries": []}]}]}
    gallery_bundle["scenes"][0]["map"]["features"][0]["geometry"] = geometry
    output = tmp_path / "index.html"
    export_source_gallery(gallery_bundle, output)
    source_map = _embedded(output)["scenes"][0]["map"]
    assert source_map["features"][0]["geometry"] == geometry
    assert source_map["layer_labels"]["dubins_nodes"] == "已提供 · 1 个要素"


@pytest.mark.parametrize("geometry", [
    {"type": "GeometryCollection"},
    {"type": "GeometryCollection", "geometries": [{"type": "Unknown"}]},
    {"type": "GeometryCollection", "geometries": [
        {"type": "Point", "coordinates": [1, "invalid"]}]},
])
def test_rejects_invalid_nested_geometry_without_silently_dropping_it(
        tmp_path, gallery_bundle, geometry):
    gallery_bundle["scenes"][0]["map"]["features"][0]["geometry"] = geometry
    with pytest.raises(ValueError):
        export_source_gallery(gallery_bundle, tmp_path / "index.html")


@pytest.mark.parametrize("image", [
    "../outside.jpg", "/absolute.jpg", "C:/image.jpg", "https://example.com/a.jpg",
    "//example.com/a.jpg", "javascript:alert(1)", "a\\b.jpg", "%2e%2e/a.jpg",
    "camera.jpg?x=1", "camera.jpg#x", "not-present.jpg",
])
def test_rejects_external_missing_or_ambiguous_image_paths(tmp_path, gallery_bundle, image):
    gallery_bundle["scenes"][0]["frames"][0]["cameras"][0]["image"] = image
    with pytest.raises(ValueError):
        export_source_gallery(gallery_bundle, tmp_path / "index.html")
    assert not (tmp_path / "index.html").exists()


def test_rejects_symlink_asset_escape(tmp_path, gallery_bundle):
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside-gallery")
    try:
        (gallery / "camera.jpg").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="escapes"):
        export_source_gallery(gallery_bundle, gallery / "index.html")


@pytest.mark.parametrize("defect", [
    "clock_scale", "timestamp_repeat", "camera_reorder", "object_id", "nan", "geometry",
    "available_without_image", "unavailable_with_image", "schema",
])
def test_rejects_ambiguous_provenance_before_writing(tmp_path, gallery_bundle, defect):
    frames = gallery_bundle["scenes"][0]["frames"]
    if defect == "clock_scale":
        frames[1]["time_seconds"] = 0.1
    elif defect == "timestamp_repeat":
        frames[1]["timestamp_us"] = frames[0]["timestamp_us"]
    elif defect == "camera_reorder":
        frames[1]["cameras"].reverse()
    elif defect == "object_id":
        frames[0]["cameras"][0]["projection"][0]["id"] = "unknown"
    elif defect == "nan":
        frames[0]["ego_pose"][0] = float("nan")
    elif defect == "geometry":
        gallery_bundle["scenes"][0]["map"]["features"][0]["geometry"]["type"] = "Unknown"
    elif defect == "available_without_image":
        frames[0]["cameras"][0]["image"] = None
    elif defect == "unavailable_with_image":
        frames[0]["cameras"][1]["image"] = "camera.jpg"
    else:
        gallery_bundle["kind"] = "simulation"
    with pytest.raises(ValueError):
        export_source_gallery(gallery_bundle, tmp_path / "index.html")
    assert not (tmp_path / "index.html").exists()
