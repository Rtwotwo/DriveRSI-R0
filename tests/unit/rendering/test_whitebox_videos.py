import copy

import pytest

from driversi.rendering.camera_review.video_export import video_plan


def scene():
    camera = dict(
        camera_id="front", owner_actor_id="ego", virtual=False, source_whitebox={"token": "source"}
    )
    return dict(
        camera_owners={"front": "ego"},
        frames=[
            dict(generation=0, tick=i * 2, time_seconds=i * 0.2, cameras=[copy.deepcopy(camera)])
            for i in range(4)
        ],
    )


def test_video_keeps_original_timebase_and_does_not_extrapolate_terminal():
    plan = video_plan(scene(), 25)
    assert plan["frame_count"] == 15
    assert plan["encoded_duration_seconds"] == 0.6
    assert plan["frame_indices"] == [0] * 5 + [1] * 5 + [2] * 5
    assert {c["variant"] for c in plan["clips"]} == {"source", "branch", "comparison"}


def test_irregular_sampling_uses_previous_observation_only():
    data = scene()
    for f, t in zip(data["frames"], [0, 0.1, 0.35, 0.6]):
        f["time_seconds"] = t
    assert video_plan(data, 10)["frame_indices"] == [0, 1, 1, 1, 2, 2]


def test_virtual_camera_never_gets_fake_original_video():
    data = scene()
    for frame in data["frames"]:
        frame["cameras"][0].update(virtual=True, source_whitebox=None)
    assert [c["variant"] for c in video_plan(data)["clips"]] == ["branch"]


@pytest.mark.parametrize(
    "defect",
    [
        "reset",
        "duplicate_time",
        "nonfinite",
        "owner",
        "camera_missing",
        "unsafe_id",
        "source_missing",
    ],
)
def test_incompatible_video_sequence_rejected(defect):
    data = scene()
    if defect == "reset":
        data["frames"][-1]["generation"] = 1
    if defect == "duplicate_time":
        data["frames"][-1]["time_seconds"] = 0.2
    if defect == "nonfinite":
        data["frames"][-1]["time_seconds"] = float("nan")
    if defect == "owner":
        data["frames"][-1]["cameras"][0]["owner_actor_id"] = "other"
    if defect == "camera_missing":
        data["frames"][-1]["cameras"] = []
    if defect == "unsafe_id":
        data["camera_owners"] = {"../outside": "ego"}
    if defect == "source_missing":
        data["frames"][-1]["cameras"][0]["source_whitebox"] = None
    with pytest.raises(ValueError):
        video_plan(data)


def test_shared_renderer_omits_owner_and_marks_unreconstructed_geometry():
    pytest.importorskip("quickjs")
    from driversi.rendering.camera_review.video_export import WhiteboxPainter

    data = dict(map_features=[], map_layers={})
    painter = WhiteboxPainter(data)
    camera = dict(
        width=100,
        height=100,
        camera_to_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        K=[[50, 0, 50], [0, 50, 50], [0, 0, 1]],
        distortion=[0, 0, 0, 0, 0],
        objects=[
            dict(
                id="pedestrian",
                hidden_owner=False,
                reconstructed=False,
                corners=[
                    [-1, -1, -4],
                    [1, -1, -4],
                    [1, 1, -4],
                    [-1, 1, -4],
                    [-1, -1, -6],
                    [1, -1, -6],
                    [1, 1, -6],
                    [-1, 1, -6],
                ],
            )
        ],
    )
    image = painter.draw(camera)
    assert image.getpixel((50, 50)) == (246, 210, 228)
    camera["objects"][0]["hidden_owner"] = True
    assert painter.draw(camera).getpixel((50, 50)) == (219, 228, 238)


def test_h264_output_decodes_at_requested_timebase(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from PIL import Image

    from driversi.rendering.camera_review.video_export import encode

    info = encode(
        tmp_path / "clip.mp4",
        {0: Image.new("RGB", (64, 64), "red"), 1: Image.new("RGB", (64, 64), "blue")},
        [0] * 5 + [1] * 5,
        25,
    )
    assert info["encoded_frame_count"] == 10
    assert info["duration_seconds"] == 0.4
    assert info["codec"] == "h264"
    assert info["width"] == info["height"] == 64
