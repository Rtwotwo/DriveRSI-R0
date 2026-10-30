import numpy as np
import pytest

from driversi.rendering.camera_review.exporter import (
    corners,
    previous_source,
    safe_media,
    transform,
    validate_camera,
)


def valid_camera():
    return dict(
        owner_actor_id="ego",
        camera_id="front",
        image="front.png",
        virtual=False,
        camera_model="opencv_distorted_source_lens",
        camera_pose_axes="nerfstudio_right_up_back",
        camera_coordinate_frame="mtgs_reconstruction_metres",
        camera_to_world=np.eye(4)[:3].tolist(),
        K=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        distortion=[0, 0, 0, 0, 0],
    )


def test_camera_identity_requires_world_presence():
    with pytest.raises(ValueError, match="not present"):
        validate_camera(valid_camera(), [dict(id="ego", present=False)], 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_pose_axes", "optical"),
        ("camera_coordinate_frame", "map"),
        ("camera_model", "unknown"),
        ("image", None),
    ],
)
def test_incompatible_camera_rejected(field, value):
    c = valid_camera()
    c[field] = value
    with pytest.raises(ValueError):
        validate_camera(c, [dict(id="ego", present=True)], 0)


def test_virtual_camera_cannot_claim_real_photo():
    c = valid_camera()
    c.update(virtual=True, source="borrowed.png")
    with pytest.raises(ValueError, match="real source"):
        validate_camera(c, [dict(id="ego", present=True)], 0)


def test_camera_rotation_validated():
    c = valid_camera()
    c["camera_to_world"][0][0] = -1
    with pytest.raises(ValueError, match="Reflected"):
        validate_camera(c, [dict(id="ego", present=True)], 0)


def test_causal_log_selection_and_gap():
    frames = [dict(timestamp_us=0), dict(timestamp_us=500000)]
    assert previous_source(frames, 499999) is frames[0]
    assert previous_source(frames, 500000) is frames[1]
    with pytest.raises(ValueError):
        previous_source(frames, 1300000)
    with pytest.raises(ValueError):
        previous_source(frames, -1)


def test_cuboid_keeps_measured_elevation_and_rotates():
    points = corners([3, 4, 5], [4, 2, 2], np.pi / 2)
    np.testing.assert_allclose(points.mean(axis=0), [3, 4, 5])
    np.testing.assert_allclose(np.ptp(points, axis=0), [2, 4, 2])
    matrix = np.eye(4)
    matrix[:3, 3] = [10, 20, 30]
    np.testing.assert_allclose(transform(points, matrix).mean(axis=0), [13, 24, 35])


def test_source_media_must_be_contained(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    (tmp_path / "external.png").touch()
    with pytest.raises(ValueError):
        safe_media(root, "../external.png")
    with pytest.raises(ValueError):
        safe_media(root, "missing.png")
    (root / "valid.png").touch()
    assert safe_media(root, "valid.png") == root / "valid.png"


def test_seed_reference_uses_timestamp_and_short_yaw_arc():
    from types import SimpleNamespace

    from driversi.rendering.camera_review.exporter import reference_at

    scene = SimpleNamespace(
        metadata={"source_start_timestamp_us": 0},
        times=np.array([0.0, 0.1]),
        poses=np.array([[[0, 0, 1, 3.1]], [[10, 0, 2, -3.1]]]),
        presence_at=lambda _: np.array([True]),
    )
    pose, present = reference_at(scene, 50000)
    np.testing.assert_allclose(pose[0, :3], [5, 0, 1.5])
    assert abs(pose[0, 3] - np.pi) < 1e-12
    with pytest.raises(ValueError):
        reference_at(scene, 200000)


def test_seed_reference_does_not_interpolate_an_absent_object():
    from types import SimpleNamespace

    from driversi.rendering.camera_review.exporter import reference_at

    scene = SimpleNamespace(
        metadata={"source_start_timestamp_us": 0},
        times=np.array([0.0, 0.1]),
        poses=np.array([[[1.0, 2, 3, 0]], [[100, 100, 100, 0]]]),
        presence_at=lambda i: np.array([i == 0]),
    )
    pose, _ = reference_at(scene, 50000)
    np.testing.assert_array_equal(pose, scene.poses[0])


def test_declared_image_size_must_match_png(tmp_path):
    import struct

    from driversi.rendering.camera_review.exporter import verify_png

    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 960, 540))
    verify_png(path, 960, 540)
    with pytest.raises(ValueError, match="dimensions"):
        verify_png(path, 160, 96)


@pytest.fixture
def public_bundle(tmp_path):
    import hashlib
    import json

    root = tmp_path / "review"
    root.mkdir()
    (root / "media").mkdir()
    content = b"test-media"
    checksum = hashlib.sha256(content).hexdigest()
    name = f"media/{checksum}.png"
    (root / name).write_bytes(content)
    for file in ["index.html", "projection.js", "viewer.js", "semantic.js", "downloads.js"]:
        (root / file).write_text("public")
    data = dict(
        kind="camera_bound_scene_inspection",
        trainable=False,
        camera_owners={"front": "ego"},
        frames=[
            dict(
                generation=0,
                tick=0,
                actors=[dict(id="ego", present=True)],
                cameras=[dict(camera_id="front", owner_actor_id="ego", virtual=False, image=name)],
            )
        ],
        media_sha256={name: checksum},
    )
    (root / "data.json").write_text(json.dumps(data))
    return root, data


def test_publication_checks_media_and_camera_ledger(public_bundle):
    from driversi.rendering.camera_review.publication import bundle_files

    root, data = public_bundle
    assert len(bundle_files(root)) == 7
    (root / next(iter(data["media_sha256"]))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        bundle_files(root)


@pytest.mark.parametrize(
    "defect", ["private_path", "missing_camera", "wrong_owner", "virtual_photo"]
)
def test_invalid_review_cannot_be_published(public_bundle, defect):
    import json

    from driversi.rendering.camera_review.publication import bundle_files

    root, data = public_bundle
    frame = data["frames"][0]
    if defect == "private_path":
        data["extra"] = "/data/private/source.pkl"
    elif defect == "missing_camera":
        frame["cameras"] = []
    elif defect == "wrong_owner":
        frame["cameras"][0]["owner_actor_id"] = "other-car"
    else:
        frame["cameras"][0].update(virtual=True, source=frame["cameras"][0]["image"])
    (root / "data.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        bundle_files(root)


def test_source_whitebox_preserves_original_pose_and_raw_lens():
    from driversi.rendering.camera_review.exporter import source_whitebox

    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    sensor = {
        "cameras": [
            {
                "camera_id": "CAM_F0",
                "asset": {
                    "file_exists": True,
                    "dimensions_status": "header_read_only",
                    "width": 1920,
                    "height": 1080,
                    "relative_path": "CAM_F0/token.jpg",
                },
                "T_scene_from_camera": pose.tolist(),
                "intrinsics": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                "distortion_coefficients": [0, 0, 0, 0, 0],
            }
        ],
        "T_scene_from_ego": np.eye(4).tolist(),
    }
    raw = {
        "timestamp": 100,
        "token": "frame",
        "anns": {
            "track_tokens": ["pedestrian"],
            "gt_names": ["pedestrian"],
            "gt_boxes": [[0, 0, 10, 1, 1, 2, 0]],
        },
    }
    c = source_whitebox(sensor, raw, "CAM_F0", 960, 540)
    assert c["camera_to_world"] == (pose @ np.diag([1, -1, -1, 1]))[:3].tolist()
    assert c["K"] == [[500, 0, 480], [0, 500, 270], [0, 0, 1]]
    assert c["source_frame_token"] == "frame"
    assert c["camera_token"] == "token"
    assert np.mean(c["objects"][0]["corners"], axis=0).tolist() == [0, 0, 10]


@pytest.fixture
def video_bundle(public_bundle):
    import hashlib
    import json
    import zipfile

    root, data = public_bundle
    data["session_id"] = "test-session"
    data["frames"][0]["cameras"][0]["virtual"] = True
    (root / "data.json").write_text(json.dumps(data))
    (root / "videos").mkdir()
    name = "videos/front--branch.mp4"
    content = b"encoded-video-fixture"
    (root / name).write_bytes(content)
    entry = dict(
        camera_id="front",
        owner_actor_id="ego",
        variant="branch",
        path=name,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        duration_seconds=1.0,
        encoded_frame_count=25,
    )
    manifest = dict(
        kind="camera_whitebox_videos",
        trainable=False,
        session_id="test-session",
        data_sha256=hashlib.sha256((root / "data.json").read_bytes()).hexdigest(),
        renderer_sha256={
            n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in ["projection.js", "semantic.js"]
        },
        videos=[entry],
        frame_count=25,
        output_fps=25,
    )
    (root / "videos/manifest.json").write_text(json.dumps(manifest))
    archive = root / "videos/all-whitebox-videos.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(root / name, "front--branch.mp4")
        z.write(root / "videos/manifest.json", "manifest.json")
    manifest["archive"] = dict(
        path="videos/all-whitebox-videos.zip",
        bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    (root / "videos.json").write_text(json.dumps(manifest))
    return root, manifest


def test_video_publication_includes_downloads(video_bundle):
    from driversi.rendering.camera_review.publication import bundle_files

    names = {n for n, _ in bundle_files(video_bundle[0])}
    assert {
        "videos.json",
        "videos/front--branch.mp4",
        "videos/all-whitebox-videos.zip",
        "videos/manifest.json",
    } <= names


@pytest.mark.parametrize(
    "defect", ["data", "renderer", "owner", "timing", "bytes", "missing_variant"]
)
def test_video_publication_rejects_stale_or_wrong_download(video_bundle, defect):
    import json

    from driversi.rendering.camera_review.publication import bundle_files

    root, manifest = video_bundle
    if defect == "data":
        manifest["data_sha256"] = "0" * 64
    elif defect == "renderer":
        (root / "semantic.js").write_text("changed renderer")
    elif defect == "owner":
        manifest["videos"][0]["owner_actor_id"] = "other"
    elif defect == "timing":
        manifest["videos"][0]["duration_seconds"] = 2
    elif defect == "bytes":
        (root / "videos/front--branch.mp4").write_bytes(b"changed")
    else:
        manifest["videos"] = []
    (root / "videos.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle_files(root)
