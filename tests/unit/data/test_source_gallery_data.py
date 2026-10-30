"""Source export preserves photographic bytes and does not require an active actor."""

import copy
import hashlib
import pickle
import struct
import zlib

import numpy as np
import pytest

from driversi.data.scenes.source_gallery import _scene


def _png():
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 6, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + bytes([50, 90, 130])*8)*6))
            + chunk(b"IEND", b""))


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "data"
    logs, sensors = root / "navsim_logs/mini", root / "sensor_blobs/mini"
    logs.mkdir(parents=True)
    sensors.mkdir(parents=True)
    (sensors / "source.png").write_bytes(_png())
    frame = {"timestamp": 1000000, "token": "token-0", "log_name": "log",
             "map_location": "map", "ego2global_translation": [100., 200., 0.],
             "ego2global_rotation": [1., 0., 0., 0.],
             "lidar2ego_translation": [10., 20., 30.], "lidar2ego_rotation": [1., 0., 0., 0.],
             "anns": {"track_tokens": ["ped"], "gt_names": ["pedestrian"],
                      "gt_boxes": np.asarray([[10., 0., .9, .5, .5, 1.8, 0.]])},
             "cams": {"CAM_F0": {"data_path": "source.png",
                                   "cam_intrinsic": [[4, 0, 4], [0, 4, 3], [0, 0, 1]],
                                   "distortion": [0., 0., 0., 0., 0.],
                                   "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                                   "sensor2lidar_translation": [0., 0., 1.5]}}}
    other = copy.deepcopy(frame)
    other.update(timestamp=1500200, token="token-1")
    with (logs / "log.pkl").open("wb") as stream:
        pickle.dump([frame, other], stream)
    monkeypatch.setattr("driversi.data.scenes.source_gallery.read_map_context",
                        lambda *args, **kwargs: ([], {"traffic_signs": {
                            "availability": "unavailable", "feature_count": 0}}, {}))
    return root, {"id": "sample", "log": "log.pkl", "start_frame": 0, "frames": 2}


def test_source_export_preserves_images_all_objects_and_native_timing(source, tmp_path):
    root, selection = source
    destination = tmp_path / "gallery"
    scene = _scene(selection, root, destination, trusted=True)
    assert len(scene["frames"]) == 2 and scene["frames"][-1]["time_seconds"] == .5002
    assert scene["frames"][0]["objects"][0]["kind"] == "pedestrian"
    assert scene["frames"][0]["objects"][0]["pose"][:2] == [10., 0.]
    assert len(scene["frames"][0]["cameras"]) == 8
    assert not scene["frames"][0]["cameras"][1]["available"]
    asset = scene["assets"][0]
    assert (destination / asset["image"]).read_bytes() == _png()
    assert asset["sha256"] == hashlib.sha256(_png()).hexdigest()
    assert scene["frames"][0]["cameras"][0]["projection"]  # LiDAR offset must not move camera.
    assert scene["map"]["layers"]["traffic_signs"]["availability"] == "unavailable"


def test_source_preview_requires_pickle_authorization_and_safe_id(source, tmp_path):
    root, selection = source
    with pytest.raises(ValueError, match="trusted"):
        _scene(selection, root, tmp_path / "out", trusted=False)
    selection["id"] = "../outside"
    with pytest.raises(ValueError, match="identifier"):
        _scene(selection, root, tmp_path / "out", trusted=True)
