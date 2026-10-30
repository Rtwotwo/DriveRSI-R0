import pytest

from driversi.rendering.gaussian.mtgs_viewer import export_mtgs_viewer


def test_probe_viewer_requires_actual_comparison_images(tmp_path):
    frame = {name: f"{name}.png" for name in ("baseline", "intervention", "restored")}
    manifest = {"schema_version": 1, "kind": "mtgs_renderer_probe",
                "status": "completed", "trainable": False, "frames": [frame]}
    with pytest.raises(ValueError, match="missing"):
        export_mtgs_viewer(manifest, tmp_path / "viewer.html")
    for name in frame.values():
        (tmp_path / name).write_bytes(b"image bytes are validated by the renderer")
    page = export_mtgs_viewer(manifest, tmp_path / "viewer.html")
    assert "trainable=false" in page.read_text(encoding="utf-8")
    assert "原位姿重建" in page.read_text(encoding="utf-8")


def test_probe_viewer_cannot_label_probe_as_trainable(tmp_path):
    with pytest.raises(ValueError, match="non-training"):
        export_mtgs_viewer({"schema_version": 1, "kind": "mtgs_renderer_probe",
                            "trainable": True, "frames": [{}]}, tmp_path / "viewer.html")


def test_incomplete_probe_cannot_be_published_as_completed_gallery(tmp_path):
    with pytest.raises(ValueError, match="non-training"):
        export_mtgs_viewer({"schema_version": 1, "kind": "mtgs_renderer_probe",
                            "status": "failed", "trainable": False,
                            "frames": [{}]}, tmp_path / "viewer.html")


def test_optics_viewer_keeps_reference_domain_and_order_explicit(tmp_path):
    import json

    from driversi.rendering.gaussian.mtgs_viewer import export_optics_viewer

    directory = tmp_path / "frames"
    directory.mkdir()
    for key in ("raw", "lens", "native", "reference", "reference-checkpoint", "valid"):
        (directory / (key + ".png")).write_bytes(b"image fixture")
    data = {"trainable": False, "reference_domain": "official_checkpoint_exposure_aligned",
            "status": "completed", "expected_frames": 1,
            "frames": [{"timestamp_us": 10, "path": "frames"}]}
    (tmp_path / "sequence.json").write_text(json.dumps(data))
    (tmp_path / "report.json").write_text(json.dumps({"top_blue_pixels": {"raw": 5, "lens": 1}}))
    page = export_optics_viewer(tmp_path, tmp_path / "index.html").read_text()
    assert "reference-checkpoint.png" in page and "trainable=false" in page
    assert "未做光流补偿" in page
    assert "未完成三类节点的分层诊断" in page
    assert "顶部蓝白碎片来自天空节点" not in page
    data["status"] = "running"
    (tmp_path / "sequence.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        export_optics_viewer(tmp_path, tmp_path / "bad.html")
    data["status"] = "completed"
    data["expected_frames"] = 2
    data["frames"].append({"timestamp_us": 10, "path": "frames"})
    (tmp_path / "sequence.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="distinct source"):
        export_optics_viewer(tmp_path, tmp_path / "bad.html")
    data["frames"].pop()
    data["reference_domain"] = "raw_photo"
    (tmp_path / "sequence.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reference domain"):
        export_optics_viewer(tmp_path, tmp_path / "bad.html")


@pytest.mark.parametrize("counts, expected", [
    ({}, "未完成"),
    ({"background": 0, "skybox": 9}, "未完成"),
    ({"background": 0, "skybox": -1, "rigid": 0}, "未完成"),
    ({"background": 0, "skybox": True, "rigid": 0}, "未完成"),
    ({"background": 0, "skybox": 9, "rigid": 0}, "仅在天空节点"),
    ({"background": 2, "skybox": 9, "rigid": 0}, "未把"),
    ({"background": 0, "skybox": 0, "rigid": 0}, "未把"),
])
def test_optics_attribution_requires_complete_matching_evidence(counts, expected):
    from driversi.rendering.gaussian.mtgs_viewer import _optics_ablation_summary

    report = {"node_ablation": {k: {"top_blue_raw": v} for k, v in counts.items()}}
    assert expected in _optics_ablation_summary(report)
