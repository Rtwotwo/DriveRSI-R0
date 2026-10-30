import copy

import pytest

from driversi.rendering.camera_review.gallery import partition


def data():
    return dict(
        session_id="s",
        scenario_id="scene",
        source_log="log",
        registration_mode="source-frame",
        camera_owners={"CAM_F0": "ego", "vehicle__virtual_front": "vehicle"},
        map_layers={},
        source_frames=[dict(timestamp_us=0)],
        frames=[
            dict(
                generation=0,
                tick=i * 2,
                time_seconds=i * 0.2,
                simulation_timestamp_us=i * 200000,
                source_index=0,
                actors=[dict(id="ego", present=True), dict(id="vehicle", present=True)],
                cameras=[
                    dict(camera_id="CAM_F0", owner_actor_id="ego", virtual=False, K=[[i]]),
                    dict(
                        camera_id="vehicle__virtual_front", owner_actor_id="vehicle", virtual=True
                    ),
                ],
            )
            for i in range(19)
        ],
    )


def test_partition_is_lossless_and_keeps_bounded_camera_chunks():
    original = data()
    before = copy.deepcopy(original)
    scene, chunks = partition(original)
    assert original == before
    assert len(chunks) == 6
    assert all(len(c["frames"]) <= 8 for c in chunks.values())
    for cid, c in scene["cameras"].items():
        restored = [f for ref in c["chunks"] for f in chunks[ref["path"]]["frames"]]
        assert restored == [
            next(c for c in f["cameras"] if c["camera_id"] == cid) for f in original["frames"]
        ]
    assert scene["frames"] == [
        {k: v for k, v in f.items() if k != "cameras"} for f in original["frames"]
    ]


@pytest.mark.parametrize("kind", ["missing_camera", "owner", "reset", "time", "future", "unsafe"])
def test_invalid_identity_and_timing_rejected(kind):
    d = data()
    if kind == "missing_camera":
        d["frames"][2]["cameras"].pop()
    if kind == "owner":
        d["frames"][2]["cameras"][0]["owner_actor_id"] = "other"
    if kind == "reset":
        d["frames"][3]["generation"] = 1
    if kind == "time":
        d["frames"][3]["time_seconds"] = 0
    if kind == "future":
        d["source_frames"][0]["timestamp_us"] = 1000
    if kind == "unsafe":
        d["camera_owners"]["../x"] = "ego"
    with pytest.raises(ValueError):
        partition(d)


def test_short_tail_and_exact_chunk_boundary():
    s, c = partition(data(), chunk_size=1)
    assert len(c) == 38
    assert c["chunks/CAM_F0-0018.json"]["ticks"] == [36]
    with pytest.raises(ValueError):
        partition(data(), chunk_size=0)
