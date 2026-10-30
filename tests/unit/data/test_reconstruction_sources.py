import pytest

from driversi.data.assets.reconstruction_sources import resolve_reconstruction_sensor


def test_separate_modalities_missing_and_content_conflicts(tmp_path):
    camera, lidar, replica = [tmp_path / x for x in ("images", "points", "replica")]
    for root in (camera, lidar, replica):
        (root / "log").mkdir(parents=True)
    (camera / "log/a.jpg").write_bytes(b"image")
    (lidar / "log/a.pcd").write_bytes(b"measured points")
    roots = {"camera": [camera], "lidar": [camera, lidar, replica]}
    resolved = resolve_reconstruction_sensor("log/a.pcd", modality="lidar", sensor_roots=roots)
    assert resolved["selected"]["path"] == str((lidar / "log/a.pcd").resolve())
    assert resolved["imputed"] is False
    missing = resolve_reconstruction_sensor("log/b.pcd", modality="lidar", sensor_roots=roots)
    assert missing["status"] == "missing" and missing["selected"] is None
    (replica / "log/a.pcd").write_bytes(b"different points")
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_reconstruction_sensor("log/a.pcd", modality="lidar", sensor_roots=roots)


@pytest.mark.parametrize("relative", ["../secret", "/outside/file", "C:\\outside", "log/../../secret"])
def test_reject_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        resolve_reconstruction_sensor(relative, modality="camera", sensor_roots={"camera": [tmp_path]})
